"""Offline integration through formal consolidation and immutable collection views."""

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.application import (
    AdaptiveApplicationConfig,
    consolidate_selfplay_skills,
)
from selfplay_graph_flowsteer.llm import LLMResponse
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer
from selfplay_graph_flowsteer.skill_evolution_v2 import (
    SCHEMA,
    _background_jobs,
    freeze_collection,
    ingest_rollouts,
    load_bank,
    store_for_config,
)

from .test_skill_evolution_v2 import proposal


def configuration(tmp_path):
    return replace(
        AdaptiveApplicationConfig(),
        skillbank_mode=SCHEMA,
        skillbank_activation_policy="checked",
        skillbank_embedding_model_path=None,
        skill_cases_path=tmp_path / "cases.json",
        skillbank_path=tmp_path / "bank.json",
        skillbank_prompt_token_budget=4000,
    )


def evidence(folder, n=20):
    folder.mkdir(parents=True, exist_ok=True)
    tasks = [TaskSpec(f"q{i}", f"public question {i}", task_type="qa") for i in range(n)]
    rows = [
        dict(
            task_id=t.task_id,
            rollout_id=f"{t.task_id}-{j}",
            reward=float(j),
            metadata=dict(
                reward_known=True,
                solver_answer="public answer",
                solver_trace={"events": [{"kind": "canvas_step", "output": "public evidence"}]},
            ),
        )
        for t in tasks
        for j in (0, 1)
    ]
    path = folder / "solver_rollouts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return SimpleNamespace(tasks=tasks), path.read_bytes()


def test_formal_background_generation_does_not_wait_or_change_frozen_collection(
    tmp_path, monkeypatch
):
    import selfplay_graph_flowsteer.application as app

    config = configuration(tmp_path)
    cycle = tmp_path / "cycle-0009"
    frozen_config = freeze_collection(config, cycle)
    result, raw = evidence(cycle)
    frozen = frozen_config.skillbank_path.read_bytes()
    entered, release = threading.Event(), threading.Event()

    class Backend:
        calls = 0

        def generate(self, messages, *, role):
            self.calls += 1
            assert role == "skill-distiller"
            entered.set()
            assert release.wait(5), "test must release background worker"
            return LLMResponse(text=json.dumps(proposal()), model="offline")

    backend = Backend()
    monkeypatch.setattr(app, "_create_runtime_backend", lambda *a, **kw: backend)
    # It must return with the model response still blocked.
    try:
        result_status = consolidate_selfplay_skills(
            config, result, mock=False, step=10, cycle_dir=cycle
        )
        assert result_status == (("skill_v2", "background_generation_submitted"),)
        assert entered.wait(2)
        assert not release.is_set()
        assert frozen_config.skillbank_path.read_bytes() == frozen
        assert consolidate_selfplay_skills(
            config, result, mock=False, step=10, cycle_dir=cycle
        ) == (("skill_v2", "generation_already_running"),)
    finally:
        release.set()
        thread = _background_jobs.get(
            str(config.skill_cases_path.with_suffix(".v2.sqlite3").resolve())
        )
        if thread:
            thread.join(10)
            assert not thread.is_alive()
    assert backend.calls == 20
    assert (cycle / "solver_rollouts.jsonl").read_bytes() == raw
    assert frozen_config.skillbank_path.read_bytes() == frozen
    after = freeze_collection(config, tmp_path / "cycle-0010")
    cards = json.loads(after.skillbank_path.read_text())["cards"]
    active = [c for c in cards if c["status"] == "active"]
    assert len(active) == 1
    bank = load_bank(after)
    chosen, context, manifest = bank.select_context(
        "incomplete evidence synthesis check", task_type="qa", tokenizer=ByteTokenizer()
    )
    assert active[0]["card"]["skill_id"] in [c.skill_id for c in chosen]
    assert manifest["context"] == context
    assert manifest["prompt_tokens"] <= config.skillbank_prompt_token_budget
    assert "task success is not causal validation" not in context
    with store_for_config(config).connect() as db:
        assert db.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM cases WHERE status='processed'").fetchone()[0] == 20


def test_below_generation_gate_and_mock_never_send_requests(tmp_path, monkeypatch):
    import selfplay_graph_flowsteer.application as app

    config = configuration(tmp_path)
    result, _ = evidence(tmp_path / "cycle", n=1)

    class Backend:
        def generate(self, *a, **kw):
            pytest.fail("below-threshold generation must not issue request")

    monkeypatch.setattr(app, "_create_runtime_backend", lambda *a, **kw: Backend())
    consolidate_selfplay_skills(config, result, mock=False, step=10, cycle_dir=tmp_path / "cycle")
    _background_jobs[str(config.skill_cases_path.with_suffix(".v2.sqlite3").resolve())].join(5)
    assert consolidate_selfplay_skills(
        config, result, mock=True, step=20, cycle_dir=tmp_path / "cycle"
    ) == (("skill_v2", "evidence_saved"),)


def test_malformed_unscored_reward_does_not_lose_other_usage(tmp_path):
    config = configuration(tmp_path)
    store = store_for_config(config)
    store.initialize_seeds()
    sid = store.cards()[0]["card"]["skill_id"]
    task = TaskSpec("h", "question", task_type="healthcare")
    rows = [
        dict(
            task_id="h",
            rollout_id=str(i),
            reward=reward,
            metadata={
                "reward_known": known,
                "skill_context": {"selected": [{"id": sid, "version": 1}]},
            },
        )
        for i, (reward, known) in enumerate(((None, False), ("invalid", False), (0.4, True)))
    ]
    ingest_rollouts(store, {"h": task}, rows, run="r", step=1)
    record = store.usage_summary()[0]
    assert record["valid_count"] == 1
    assert record["unscored_count"] == 2
    assert record["mean_reward"] == 0.4


def test_snapshot_bookkeeping_does_not_load_encoder(tmp_path, monkeypatch):
    import selfplay_graph_flowsteer.skill_evolution_v2 as v2

    config = replace(configuration(tmp_path), skillbank_embedding_model_path="not-a-real-model")

    def forbidden(*a, **kw):
        pytest.fail("snapshot bookkeeping must not load a model")

    monkeypatch.setattr(v2, "E5SkillEmbedder", forbidden)
    frozen = freeze_collection(config, tmp_path / "cycle")
    assert frozen.skillbank_path.exists()
    store = store_for_config(config)
    assert len(store.cards()) == 8


def test_semantic_duplicate_cannot_bypass_card_type_validation(tmp_path):
    from .test_skill_evolution_v2 import _ConstantSkillEmbedder, example_case

    store = store_for_config(configuration(tmp_path))
    store.embedder = _ConstantSkillEmbedder()
    store.enqueue(example_case(0))
    store.propose("case-0", proposal(), step=1, activate_after_checks=True)
    store.enqueue(example_case(1))
    with pytest.raises(ValueError):
        store.propose(
            "case-1", dict(proposal(), kind="invalid-type"), step=2, activate_after_checks=True
        )
    with store.connect() as db:
        assert db.execute("SELECT status FROM cases WHERE id='case-1'").fetchone()[0] == "pending"
