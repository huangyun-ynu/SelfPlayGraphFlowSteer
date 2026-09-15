from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.application import (
    consolidate_selfplay_skills,
    create_adaptive_application,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.cli import _skill_context_config, build_parser
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory
from selfplay_graph_flowsteer.selfplay import (
    AlternatingSnapshots,
    SolverRollout,
    assemble_selfplay_result,
    validate_pats_group_context,
)
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer, SelfPlayRolloutRunner
from selfplay_graph_flowsteer.skill_evolution_v2 import (
    freeze_collection,
    load_bank,
    store_for_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    return replace(
        load_adaptive_config(ROOT / "configs/pats.example.toml"),
        skillbank_path=tmp_path / "skills.json",
        skill_cases_path=tmp_path / "cases.json",
        trace_path=tmp_path / "trace.jsonl",
        skillbank_embedding_model_path=None,
        persist_runtime_updates=False,
    )


def test_example_config_and_old_default_are_compatible(config):
    config.validate()
    assert config.pats.enabled
    assert not config.skillbank_context_enabled
    old = load_adaptive_config(ROOT / "configs/mock.toml")
    assert not old.pats.enabled
    assert old.skillbank_usage == "always"
    with pytest.raises(ValueError, match="director_skill_v2"):
        replace(config, skillbank_mode="legacy").validate()
    with pytest.raises(ValueError, match="usage"):
        replace(config, skillbank_usage="sometimes").validate()


@pytest.mark.parametrize(
    "mode,training,expected",
    [("auto", True, True), ("auto", False, False), ("on", False, True), ("off", True, False)],
)
def test_collection_and_evaluation_context_contract(config, mode, training, expected):
    resolved = _skill_context_config(config, SimpleNamespace(skill_context=mode), training=training)
    assert resolved.skillbank_context_enabled is expected
    assert config.skillbank_training is False


@pytest.mark.parametrize(
    "command,extra",
    [
        ("adaptive-solve", ["--task", "check evidence"]),
        ("benchmark", ["--data", "tasks.jsonl", "--output", "results.jsonl"]),
        ("selfplay-rollout", ["--seed", "demo", "--output", "run"]),
        ("selfplay-experiment", ["--seed-data", "tasks.jsonl", "--output", "run"]),
    ],
)
def test_cli_exposes_explicit_skill_ablation(command, extra):
    args = build_parser().parse_args([command, *extra, "--skill-context", "off"])
    assert args.skill_context == "off"


def test_default_inference_neither_loads_nor_updates_skill_state(config, tmp_path):
    application = create_adaptive_application(replace(config, verifier="none"), mock=True)
    try:
        assert application.solver.skillbank is None
        result = application.solve("Verify the evidence before the final answer", task_id="eval")
        assert result.solver_result.skill_context == {}
        assert consolidate_selfplay_skills(config, object(), mock=True, step=1) == ()
        assert not config.skill_cases_path.with_suffix(".v2.sqlite3").exists()
        assert freeze_collection(config, tmp_path / "eval") is config
    finally:
        application.close()


def test_raw_zero_and_one_groups_survive_maintenance_and_frozen_resume(config, tmp_path):
    training = replace(config, skillbank_training=True)
    cycle = tmp_path / "run" / "cycle_000"
    cycle.mkdir(parents=True)
    frozen = freeze_collection(training, cycle)
    before = frozen.skillbank_path.read_bytes()
    bank = load_bank(frozen)
    tasks = [
        TaskSpec(f"t{i}", "Inspect evidence", task_type="qa", metadata={"dataset": "nq"})
        for i in range(2)
    ]
    rows = []
    for i, task in enumerate(tasks):
        _, _, manifest = bank.select_context(
            task.prompt,
            task_type=task.task_type,
            tokenizer=ByteTokenizer(),
            task_metadata=task.metadata,
        )
        for j in range(2):
            rows.append(
                {
                    "rollout_id": f"t{i}-r{j}",
                    "task_id": task.task_id,
                    "reward": float(i),
                    "metadata": {
                        "reward_known": True,
                        "skill_context": manifest,
                        "solver_trace": {"events": []},
                    },
                }
            )
    (cycle / "solver_rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = SimpleNamespace(tasks=tasks, snapshots={"solver": "policy-0"})
    status = consolidate_selfplay_skills(
        frozen, result, mock=True, step=1, cycle_dir=cycle, tokenizer=ByteTokenizer()
    )
    assert status == (("pats", "review_recorded"),)
    receipt = json.loads((cycle / "pats_review.json").read_text())
    assert receipt["credible_groups"] == 2
    assert receipt["reviews"][0]["ema"] == pytest.approx(0.05)
    assert receipt["reviews"][0]["status"] == "mock_no_refiner"
    assert frozen.skillbank_path.read_bytes() == before
    assert freeze_collection(training, cycle).skillbank_path.read_bytes() == before
    # Exact replay does not increment EMA or double-count support.
    consolidate_selfplay_skills(
        frozen, result, mock=True, step=1, cycle_dir=cycle, tokenizer=ByteTokenizer()
    )
    assert json.loads((cycle / "pats_review.json").read_text()) == receipt
    next_config = freeze_collection(training, tmp_path / "run" / "cycle_001")
    next_snapshot = json.loads(next_config.skillbank_path.read_text())
    assert next_snapshot["pats"]["step"] == 1
    assert len(next_snapshot["pats"]["scopes"]) == 1
    assert len(store_for_config(training).cards()) == 8
    with pytest.raises(ValueError, match="another run"):
        freeze_collection(training, tmp_path / "different_run" / "cycle_000", step=2)
    with pytest.raises(ValueError, match="already committed step"):
        freeze_collection(training, tmp_path / "run" / "out_of_order", step=1)


def rollout(index, context="same guidance", **changes):
    manifest = {
        "pats_enabled": True,
        "snapshot_id": "frozen-0",
        "pats_scope": "qa",
        "context": context,
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        **changes,
    }
    graph = MultiAgentGraph()
    trajectory = TokenizedDirectorTrajectory(
        rollout_id=f"r{index}",
        task_id="task",
        token_ids=(1,),
        action_mask=(1,),
        reward=0,
        graph=graph.to_dict(),
        seed=index,
        metadata={"skill_context": manifest},
    )
    return SolverRollout(trajectory, graph)


def test_group_context_validation_uses_actual_text_and_accepts_empty_scaffold():
    validate_pats_group_context([rollout(0), rollout(1)])
    validate_pats_group_context([rollout(0, ""), rollout(1, "")])
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_pats_group_context([rollout(0, context_sha256="forged")])
    with pytest.raises(ValueError, match="missing frozen"):
        validate_pats_group_context([rollout(0), rollout(1, pats_enabled=False)])


def test_mixed_context_is_rejected_before_frontier_and_batch_assembly():
    rows = [rollout(0), rollout(1, "different guidance")]
    # The guard must run before either function touches proposals, scores, or trainers.
    with pytest.raises(ValueError, match="mixed skill contexts"):
        assemble_selfplay_result([], rows, AlternatingSnapshots("p0", "s0"))
    runner = SelfPlayRolloutRunner.__new__(SelfPlayRolloutRunner)
    with pytest.raises(ValueError, match="mixed skill contexts"):
        runner._collect_frontier_reverification([], rows)
