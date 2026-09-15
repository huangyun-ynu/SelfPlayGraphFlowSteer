from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.application import AdaptiveApplicationConfig
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.pats import (
    PatsConfig,
    PatsController,
    apply_metadata_updates,
    apply_operations,
    choose_review_mode,
    collect_evidence,
    render_cards,
    resolve_scope,
)
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer
from selfplay_graph_flowsteer.skill_evolution_v2 import (
    SCHEMA,
    DirectorSkillBankV2,
    SkillStore,
    freeze_collection,
    load_bank,
    store_for_config,
)
from selfplay_graph_flowsteer.skills import SkillCard


def card(plan="Check evidence before final synthesis"):
    return dict(
        name="Check evidence",
        description="Check public evidence before synthesis",
        trigger="Incomplete evidence",
        plan=plan,
        pitfall="Avoid unsupported claims",
        constraint="Use available public tools",
        kind="verification",
    )


def record(skill_id="existing", plan="Check evidence before final synthesis"):
    return dict(
        card=SkillCard(skill_id=skill_id, **card(plan), task_types=["qa"]).to_dict(),
        version=1,
        status="active",
        required_tools=[],
        excluded_task_types=[],
    )


def data(rewards=(0.0, 0.0), *, dataset="nq_open", difficulty="easy", offset=0):
    tasks, rows = {}, []
    for i, reward in enumerate(rewards, offset):
        task_id = f"task-{i}"
        tasks[task_id] = TaskSpec(
            task_id,
            "Find evidence for this question",
            task_type="qa",
            metadata={"dataset": dataset, "difficulty": difficulty},
        )
        for sibling in range(2):
            rows.append(
                dict(
                    rollout_id=f"{task_id}-{sibling}",
                    task_id=task_id,
                    reward=reward,
                    metadata={
                        "reward_known": True,
                        "skill_context": {"context": "fixed"},
                        "solver_trace": {"events": []},
                    },
                )
            )
    return tasks, rows


def examples(scope):
    return [dict(id=f"e{i}", task_id=f"t{i}", scope=scope) for i in range(2)]


def operation(op="ADD", *, skill_id=None, plan=None):
    value = dict(op=op, evidence_ids=["e0", "e1"])
    if skill_id is not None:
        value["skill_id"] = skill_id
    if op != "DELETE":
        value["card"] = card(plan or "Check evidence before synthesis")
    return value


@pytest.fixture
def config():
    return PatsConfig(enabled=True, max_tokens=10000)


def test_mode_bands_capacity_priority_and_validation():
    assert choose_review_mode(bank_pressure=0, group_sr_ema=0.1) == "EXPAND"
    assert choose_review_mode(bank_pressure=0, group_sr_ema=0.3) == "REVISE"
    assert choose_review_mode(bank_pressure=0, group_sr_ema=0.85) == "COMPRESS"
    assert choose_review_mode(bank_pressure=1, group_sr_ema=0.99) == "FORCED_PRUNE"
    with pytest.raises(ValueError):
        replace(PatsConfig(), min_groups=1).validate()
    with pytest.raises(ValueError):
        replace(PatsConfig(), max_edits=True).validate()


def test_all_failure_and_success_groups_are_evidence_but_untrusted_are_not():
    tasks, rows = data((0.0, 1.0, 0.5, 0.0))
    for row in rows[4:6]:
        row["metadata"]["infrastructure_failure"] = True
    for row in rows[6:]:
        row["metadata"]["reward_known"] = False
    tasks["test"] = TaskSpec("test", "hidden", task_type="qa", metadata={"is_final_test": True})
    rows.append(
        dict(rollout_id="test-0", task_id="test", reward=0, metadata={"reward_known": True})
    )
    evidence = collect_evidence(tasks, rows, run="run", step=0, policy_snapshot="solver0")
    assert len(evidence) == 2
    assert evidence[0]["all_failed"] and evidence[1]["all_passed"]
    assert evidence[0]["valid_rollouts"] == 2
    assert evidence[0]["policy_snapshot"] == "solver0"
    assert evidence[0]["context_hash"]


def test_mixed_context_and_conflicting_duplicates_fail_closed():
    tasks, rows = data()
    rows[0]["metadata"]["skill_context"] = {"context": "different"}
    assert len(collect_evidence(tasks, rows, run="r", step=0, policy_snapshot="p")) == 1
    duplicate = copy.deepcopy(rows[0])
    duplicate["reward"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        collect_evidence(tasks, [*rows, duplicate], run="r", step=0, policy_snapshot="p")


def test_heldout_swe_and_late_policy_mismatch_audits_are_excluded():
    tasks, rows = data(dataset="swe_bench")
    for task in tasks.values():
        task.metadata.update(source_split="verified", experiment_split="test")
    assert collect_evidence(tasks, rows, run="r", step=0, policy_snapshot="p") == []
    tasks, rows = data()
    updates = [
        {
            "rollout_id": row["rollout_id"],
            "metadata": {
                **row["metadata"],
                "training_exclusion_reasons": ["frozen_policy_probability_mismatch"],
            },
        }
        for row in rows[:2]
    ]
    audited = apply_metadata_updates(rows, updates)
    assert len(collect_evidence(tasks, audited, run="r", step=0, policy_snapshot="p")) == 1
    assert "training_exclusion_reasons" not in rows[0]["metadata"]
    rows[0]["metadata"]["training_exclusion_reasons"] = ["missing_policy_tokens"]
    assert len(collect_evidence(tasks, rows, run="r", step=0, policy_snapshot="p")) == 2


def test_ema_once_per_scope_cycle_zero_initialized_and_exact_replay(tmp_path, config):
    store = SkillStore(tmp_path / "bank.db")
    controller = PatsController(store, config, len)
    tasks, rows = data((0.0, 1.0))
    first = controller.maintain(tasks, rows, run="r", step=0, policy_snapshot="p0", mock=True)
    assert first["reviews"][0]["ema"] == pytest.approx(0.05)
    assert (
        controller.maintain(tasks, rows, run="r", step=0, policy_snapshot="p0", mock=True) == first
    )
    assert controller.snapshot()["scopes"][first["reviews"][0]["scope"]]["ema"] == pytest.approx(
        0.05
    )
    changed = copy.deepcopy(rows)
    changed[0]["reward"] = 1
    with pytest.raises(ValueError, match="replay"):
        controller.maintain(tasks, changed, run="r", step=0, policy_snapshot="p0", mock=True)
    with pytest.raises(ValueError, match="configuration"):
        PatsController(store, replace(config, ema_alpha=0.2), len).snapshot()


def test_scopes_separate_dataset_and_fixed_difficulty(tmp_path, config):
    tasks, rows = data()
    hard_tasks, hard_rows = data((1.0, 1.0), difficulty="hard", offset=2)
    other_tasks, other_rows = data((0.5, 0.5), dataset="hotpotqa", offset=4)
    controller = PatsController(SkillStore(tmp_path / "bank.db"), config, len)
    controller.maintain(
        {**tasks, **hard_tasks, **other_tasks},
        rows + hard_rows + other_rows,
        run="r",
        step=0,
        policy_snapshot="p",
        mock=True,
    )
    snapshot = controller.snapshot()
    assert len(snapshot["scopes"]) == 3
    assert sorted(s["ema"] for s in snapshot["scopes"].values()) == pytest.approx([0, 0.05, 0.1])


def test_atomic_edits_require_distinct_evidence_and_reject_foreign_scope(config):
    scope = resolve_scope("qa", {"dataset": "nq_open"})
    evidence = examples(scope)
    existing = [record()]
    original = copy.deepcopy(existing)
    bad = operation("UPDATE", skill_id="missing")
    with pytest.raises(ValueError, match="unknown skill"):
        apply_operations(
            existing,
            {"operations": [operation(), bad]},
            mode="EXPAND",
            evidence=evidence,
            config=config,
            token_counter=len,
            scope=scope,
            step=1,
        )
    assert existing == original
    evidence[1]["scope"] = resolve_scope("math", {"dataset": "aime"})
    with pytest.raises(ValueError, match="unknown evidence"):
        apply_operations(
            existing,
            {"operations": [operation()]},
            mode="EXPAND",
            evidence=evidence,
            config=config,
            token_counter=len,
            scope=scope,
            step=1,
        )
    evidence = examples(scope)
    evidence[1]["task_id"] = evidence[0]["task_id"]
    with pytest.raises(ValueError, match="distinct-task"):
        apply_operations(
            existing,
            {"operations": [operation()]},
            mode="EXPAND",
            evidence=evidence,
            config=config,
            token_counter=len,
            scope=scope,
            step=1,
        )


def test_compression_measures_rendered_tokens_and_prune_is_view_only(config):
    scope = resolve_scope("qa", {"dataset": "nq_open"})
    existing = [record(plan="Check evidence. " * 20)]
    shorter = apply_operations(
        existing,
        {"operations": [operation("UPDATE", skill_id="existing", plan="Verify.")]},
        mode="COMPRESS",
        evidence=examples(scope),
        config=config,
        token_counter=len,
        scope=scope,
        step=1,
    )
    assert len(render_cards(shorter)) < len(render_cards(existing))
    assert shorter[0]["version"] == 2 and existing[0]["version"] == 1
    with pytest.raises(ValueError, match="reduce actual"):
        apply_operations(
            shorter,
            {"operations": [operation("UPDATE", skill_id="existing", plan="Verify. " * 100)]},
            mode="COMPRESS",
            evidence=examples(scope),
            config=config,
            token_counter=len,
            scope=scope,
            step=2,
        )
    assert (
        apply_operations(
            existing,
            {"operations": [operation("DELETE", skill_id="existing")]},
            mode="FORCED_PRUNE",
            evidence=examples(scope),
            config=config,
            token_counter=len,
            scope=scope,
            step=1,
        )
        == []
    )
    assert existing[0]["status"] == "active"


def test_mock_backend_edits_real_parser_and_cycle_review_bound(tmp_path, config):
    calls = []

    def respond(messages, role):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        if "evidence" not in payload:
            return json.dumps(
                {
                    "cards": {
                        alias: {
                            "approved": True,
                            "reason": "Conditional public verification is legal",
                        }
                        for alias in payload["cards"]
                    }
                }
            )
        edit = operation()
        edit["evidence_ids"] = [e["id"] for e in payload["evidence"]]
        return json.dumps({"operations": [edit]})

    store = SkillStore(tmp_path / "bank.db")
    tasks, rows = data()
    more_tasks, more_rows = data(difficulty="hard", offset=2)
    controller = PatsController(store, replace(config, max_reviews_per_cycle=1), len)
    receipt = controller.maintain(
        {**tasks, **more_tasks},
        rows + more_rows,
        run="r",
        step=0,
        policy_snapshot="p",
        backend=MockBackend(handler=respond),
    )
    assert receipt["refiner_calls"] == receipt["semantic_checker_calls"] == 1
    assert len(calls) == 2
    assert sorted(r["status"] for r in receipt["reviews"]) == ["cycle_review_budget", "updated"]
    assert store.cards() == []  # No mutation of the global SkillBank.
    controller.maintain(
        {**tasks, **more_tasks},
        rows + more_rows,
        run="r",
        step=0,
        policy_snapshot="p",
        backend=MockBackend(handler=respond),
    )
    assert len(calls) == 2


def test_rejected_review_is_audited_without_partial_edit(tmp_path, config):
    tasks, rows = data()
    controller = PatsController(SkillStore(tmp_path / "bank.db"), config, len)
    receipt = controller.maintain(
        tasks,
        rows,
        run="r",
        step=0,
        policy_snapshot="p",
        backend=MockBackend(handler=lambda *_: '{"operations":[{"op":"ADD"}]}'),
    )
    assert receipt["reviews"][0]["status"] == "rejected"
    assert next(iter(controller.snapshot()["scopes"].values()))["cards"] == []


def test_review_budget_rotates_across_scopes_without_starvation(tmp_path, config):
    calls = []

    def respond(messages, role):
        calls.append(json.loads(messages[-1]["content"])["scope"])
        return '{"operations": []}'

    tasks, rows = {}, []
    for i, difficulty in enumerate(("easy", "medium", "hard")):
        more_tasks, more_rows = data(difficulty=difficulty, offset=2 * i)
        tasks.update(more_tasks)
        rows.extend(more_rows)
    controller = PatsController(SkillStore(tmp_path / "bank.db"), config, len)
    backend = MockBackend(handler=respond)
    controller.maintain(tasks, rows, run="r", step=0, policy_snapshot="p0", backend=backend)
    controller.maintain(tasks, rows, run="r", step=1, policy_snapshot="p1", backend=backend)
    assert len(calls) == 4 and len(set(calls)) == 3


def test_recent_evidence_gate_and_review_interval(tmp_path, config):
    controller = PatsController(
        SkillStore(tmp_path / "bank.db"), replace(config, max_policy_lag=0, review_interval=2), len
    )
    tasks, rows = data((0.0,))
    first = controller.maintain(tasks, rows, run="r", step=0, policy_snapshot="p0", mock=True)
    assert first["reviews"][0]["status"] == "insufficient_evidence"
    tasks, rows = data((0.0,), offset=1)
    second = controller.maintain(tasks, rows, run="r", step=1, policy_snapshot="p1", mock=True)
    assert second["reviews"][0]["status"] == "insufficient_evidence"


def test_scoped_empty_view_never_falls_back_to_seed_and_context_is_bound():
    scope = resolve_scope("qa", {"dataset": "nq_open"})
    bank = DirectorSkillBankV2(
        {
            "schema": SCHEMA,
            "snapshot_id": "frozen",
            "cards": [record()],
            "pats": {"snapshot_id": "pats0", "scopes": {scope: {"cards": []}}},
        }
    )
    selected, context, manifest = bank.select_context(
        "Check evidence",
        task_type="qa",
        tokenizer=ByteTokenizer(),
        task_metadata={"dataset": "nq_open"},
    )
    assert selected == [] and context == ""
    assert manifest["pats_enabled"] is True and manifest["pats_scope"] == scope
    assert manifest["snapshot_id"] == "frozen" and len(manifest["context_sha256"]) == 64
    assert bank.select_context(
        "Check evidence",
        task_type="qa",
        tokenizer=ByteTokenizer(),
        task_metadata={"dataset": "hotpotqa"},
    )[0]


def test_frozen_collection_view_preserved_after_scope_changes(tmp_path, config):
    app_config = replace(
        AdaptiveApplicationConfig(),
        pats=config,
        skillbank_enabled=True,
        skillbank_mode=SCHEMA,
        skillbank_training=True,
        skillbank_path=tmp_path / "bank.json",
        skill_cases_path=tmp_path / "cases.json",
        skillbank_embedding_model_path=None,
    )
    frozen = freeze_collection(app_config, tmp_path / "cycle0")
    original = frozen.skillbank_path.read_bytes()
    store = store_for_config(app_config)
    controller = PatsController(store, config, len)
    tasks, rows = data()
    controller.maintain(
        tasks, rows, run=str(tmp_path.resolve()), step=0, policy_snapshot="p", mock=True
    )
    assert (
        freeze_collection(app_config, tmp_path / "cycle0").skillbank_path.read_bytes() == original
    )
    assert load_bank(frozen)._pats["scopes"] == {}
    assert load_bank(app_config)._pats["scopes"]
    with pytest.raises(ValueError, match="contract"):
        freeze_collection(replace(app_config, skillbank_usage="off"), tmp_path / "cycle0")
    with pytest.raises(ValueError, match="contract"):
        freeze_collection(replace(app_config, pats=PatsConfig()), tmp_path / "cycle0")
    with pytest.raises(ValueError, match="configuration"):
        load_bank(replace(frozen, pats=replace(config, ema_alpha=0.2)))
