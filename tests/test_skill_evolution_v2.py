from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.application import (
    AdaptiveApplicationConfig,
    create_adaptive_application,
)
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer
from selfplay_graph_flowsteer.skill_evolution_v2 import (
    SCHEMA,
    DirectorSkillBankV2,
    SkillStore,
    distill_pending,
    freeze_collection,
    ingest_rollouts,
    load_bank,
)


def example_case(index=0, task=None):
    return {
        "case_id": f"case-{index}",
        "task_id": task or f"task-{index}",
        "step": 1,
        "task_type": "qa",
        "pattern": "missing_verification",
        "evidence_refs": [f"log#{index}"],
        "lower": {"failure_trace": {"events": []}},
        "higher": {"failure_trace": {"events": []}},
        "lower_reward": 0.2,
        "higher_reward": 0.6,
    }


def proposal():
    return {
        "name": "Check evidence",
        "description": "Inspect incomplete evidence before synthesis",
        "trigger": "Evidence is incomplete",
        "plan": "Assign a specific missing evidence check",
        "pitfall": "Do not repeat unsupported claims",
        "constraint": "Only use public evidence",
        "kind": "verification",
    }


class _ConstantSkillEmbedder:
    model_path = "constant-test-e5"

    def encode(self, texts, *, query):
        return [(1.0, 0.0) for _ in texts]


class _NearSkillEmbedder:
    model_path = "near-test-e5"

    def encode(self, texts, *, query):
        return [
            (0.92, (1.0 - 0.92**2) ** 0.5) if "near candidate" in text else (1.0, 0.0)
            for text in texts
        ]


@pytest.fixture
def store(tmp_path):
    value = SkillStore(tmp_path / "skills.db")
    value.initialize_seeds()
    return value


def test_seeds_zero_stats_immutable_snapshot_and_retirement(store, tmp_path):
    first = store.snapshot(tmp_path / "cycle0.json")
    assert len(first["cards"]) == 8
    assert all(
        c["provenance"] == "human_seed"
        and c["status"] == "seed"
        and c["card"]["stats"]["usage_count"] == 0
        for c in first["cards"]
    )
    retired = first["cards"][0]["card"]["skill_id"]
    store.deprecate(retired, 1, reason="explicit development evidence")
    store.initialize_seeds()
    assert store.snapshot(tmp_path / "cycle0.json") == first
    assert len(store.snapshot(tmp_path / "cycle1.json")["cards"]) == 7
    assert len(store.cards()) == 8


def test_candidate_and_case_commit_rollback_together(store, tmp_path):
    store.enqueue(example_case())
    with store.connect() as db:
        db.execute(
            "CREATE TRIGGER inject_failure BEFORE UPDATE ON cases BEGIN SELECT RAISE(ABORT,'crash'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.propose("case-0", proposal(), step=10)
    assert len(store.cards()) == 8
    with store.connect() as db:
        assert db.execute("SELECT status FROM cases").fetchone()[0] == "pending"
        db.execute("DROP TRIGGER inject_failure")
    identity = store.propose("case-0", proposal(), step=10)
    with pytest.raises(ValueError, match="consumed"):
        store.propose("case-0", proposal(), step=10)
    assert len(store.snapshot(tmp_path / "cycle.json")["cards"]) == 8
    store.deprecate(*identity, reason="candidate rejected")
    store.enqueue(example_case(1))
    assert store.propose("case-1", proposal(), step=20)[0] != identity[0]


def test_v2_e5_dedup_is_scoped_to_task_type(tmp_path):
    store = SkillStore(
        tmp_path / "skills.db",
        embedder=_ConstantSkillEmbedder(),
        dedup_threshold=0.93,
        dedup_review_threshold=0.90,
    )
    store.initialize_seeds()
    store.enqueue(example_case(0))
    first = store.propose("case-0", proposal(), step=10)

    changed = dict(proposal(), description="Inspect incomplete evidence before synthesis carefully")
    store.enqueue(example_case(1))
    assert store.propose("case-1", changed, step=11) == first
    assert len(store.cards()) == 9

    other = example_case(2)
    other["task_type"] = "math"
    store.enqueue(other)
    assert store.propose("case-2", changed, step=12)[0] != first[0]
    assert len(store.cards()) == 10
    with store.connect() as db:
        kinds = [row[0] for row in db.execute("SELECT kind FROM audit")]
    assert "skill_semantic_dedup" in kinds


def test_v2_e5_near_duplicate_is_retained_for_review(tmp_path):
    store = SkillStore(
        tmp_path / "skills.db",
        embedder=_NearSkillEmbedder(),
        dedup_threshold=0.93,
        dedup_review_threshold=0.90,
    )
    store.initialize_seeds()
    store.enqueue(example_case(0))
    first = store.propose("case-0", proposal(), step=10)
    near = dict(proposal(), description="near candidate: inspect incomplete evidence")
    store.enqueue(example_case(1))
    candidate = store.propose("case-1", near, step=11)
    card = next(
        item
        for item in store.cards()
        if item["card"]["skill_id"] == candidate[0] and item["version"] == candidate[1]
    )
    assert card["status"] == "candidate"
    assert card["dedup_review"]["status"] == "needs_review"
    assert card["dedup_review"]["matched_skill_id"] == first[0]


def test_background_only_consumes_selected_cases_and_recovers_crashed_claim(store):
    for i in range(25):
        store.enqueue(example_case(i))
    with store.connect() as db:
        db.execute("UPDATE cases SET status='processing' WHERE id='case-0'")
    backend = MockBackend([json.dumps(proposal())] * 20)
    assert len(distill_pending(store, backend, step=10)) == 20
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0] == 5
    assert distill_pending(store, backend, step=10, min_pending=1) == []


def test_failed_request_preserves_case_and_bounds_retries(store):
    class Broken:
        def generate(self, *a, **kw):
            raise RuntimeError("must not write raw credential-bearing exception text")

    store.enqueue(example_case())
    for step in (10, 20, 30):
        distill_pending(store, Broken(), step=step, min_pending=1)
    with store.connect() as db:
        row = db.execute("SELECT status,attempts,error,payload FROM cases").fetchone()
        assert tuple(row[:3]) == ("failed_archived", 3, "RuntimeError")
        assert json.loads(row[3])["case_id"] == "case-0"


def test_real_crash_resumes_same_generation_job_even_below_pending_gate(store):
    class Interrupted:
        def generate(self, *args, **kwargs):
            raise KeyboardInterrupt()

    store.enqueue(example_case())
    with pytest.raises(KeyboardInterrupt):
        distill_pending(store, Interrupted(), step=10, min_pending=1)
    assert (
        len(distill_pending(store, MockBackend([json.dumps(proposal())]), step=10, min_pending=20))
        == 1
    )
    with store.connect() as db:
        assert db.execute("SELECT status FROM generation_jobs").fetchone()[0] == "complete"


def test_same_task_quota_and_overflow_do_not_delete_records(store):
    for i in range(8):
        store.enqueue(example_case(i, task="one-task"), cap=5)
    distill_pending(store, MockBackend([json.dumps(proposal())] * 2), step=10, min_pending=1)
    with store.connect() as db:
        counts = dict(db.execute("SELECT status,COUNT(*) FROM cases GROUP BY status"))
    assert counts == {"overflow_archived": 3, "pending": 3, "processed": 2}


def test_edit_retains_original_and_excludes_unvalidated_revision(store, tmp_path):
    original = store.cards()[0]
    store.enqueue(example_case())
    identity = store.propose(
        "case-0", dict(proposal(), edit_skill_id=original["card"]["skill_id"]), step=10
    )
    assert identity == (original["card"]["skill_id"], 2)
    view = store.snapshot(tmp_path / "view.json")
    assert next(c for c in view["cards"] if c["card"]["skill_id"] == identity[0])["version"] == 1


def test_retrieval_filters_none_and_whole_card_budget(store, tmp_path):
    snapshot = store.snapshot(tmp_path / "view.json")
    # Synthetic lexical card lets us test exact inclusion and token accounting.
    record = snapshot["cards"][0]
    record["card"].update(proposal(), task_types=["qa"])
    snapshot["cards"] = [record]
    bank = DirectorSkillBankV2(snapshot, prompt_token_budget=10000)
    assert bank.retrieve("evidence", task_type="other") == []
    assert bank.retrieve("zzzzzzz", task_type="qa") == []
    chosen, context, manifest = bank.select_context(
        "evidence", task_type="qa", tokenizer=ByteTokenizer()
    )
    assert len(chosen) == 1 and "Constraint: Only use public evidence" in context
    assert manifest["prompt_tokens"] == len(context.encode())
    bank.prompt_token_budget = manifest["prompt_tokens"] - 1
    assert bank.select_context("evidence", task_type="qa", tokenizer=ByteTokenizer())[0] == []
    with pytest.raises(RuntimeError):
        bank.prune()


def test_incremental_embeddings_reuse_unchanged_cards(store, tmp_path):
    class Embedder:
        model_path = "unique-test-embedder"

        def __init__(self):
            self.encoded = []

        def encode(self, texts, *, query):
            if not query:
                self.encoded.extend(texts)
            return [(1.0, 0.0)] * len(texts)

    embedder = Embedder()
    snapshot = store.snapshot(tmp_path / "view.json")
    DirectorSkillBankV2(snapshot, embedder=embedder)
    assert len(embedder.encoded) == 8
    DirectorSkillBankV2(snapshot, embedder=embedder)
    assert len(embedder.encoded) == 8
    snapshot["cards"][0]["card"]["plan"] += " revised"
    DirectorSkillBankV2(snapshot, embedder=embedder)
    assert len(embedder.encoded) == 9


def test_continuous_reward_contrast_independent_of_ppo_and_private_data(store):
    def row(index, reward, **extra):
        return {
            "rollout_id": str(index),
            "task_id": "t",
            "reward": reward,
            "metadata": {
                "training_eligible": False,
                "training_exclusion_reasons": ["missing_tokens"],
                "reward_known": True,
                "solver_trace": {
                    "events": [
                        {
                            "kind": "canvas_step",
                            "payload": {"public": "missing output", "gold_answer": "SECRET"},
                        }
                    ],
                    "verification": {"reference_answer": "SECRET"},
                },
                "skill_context": {
                    "snapshot_id": "view",
                    "selected": [{"id": "seed", "version": 1}],
                },
                **extra,
            },
        }

    rows = [
        row(0, 0.2),
        row(1, 0.6),
        row(2, 0.0, training_exclusion_reasons=["worker_backend_failure"]),
    ]
    tasks = {"t": TaskSpec("t", "public question", task_type="healthbench")}
    for _ in range(2):
        ingest_rollouts(store, tasks, rows, run="run", step=1)
    with store.connect() as db:
        cases = list(db.execute("SELECT payload FROM cases"))
        assert len(cases) == 1 and "SECRET" not in cases[0][0]
        case = json.loads(cases[0][0])
        assert case["lower_reward"] == 0.2 and case["higher_reward"] == 0.6
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
        event = json.loads(
            db.execute("SELECT payload FROM events ORDER BY id LIMIT 1").fetchone()[0]
        )
        assert event["observed_following"] is None


def test_trial_protocol_promotion_and_snapshot_publication(store, tmp_path):
    store.enqueue(example_case())
    identity = store.propose("case-0", proposal(), step=10)
    frozen = store.snapshot(tmp_path / "old.json")
    protocol = {
        "split": "development",
        "task_ids": [str(i) for i in range(5)],
        "solver_snapshot": "frozen",
        "worker_config": "fixed",
        "budget": 900,
        "other_skills_snapshot": "view",
        "min_tasks": 5,
        "minimum_mean_gain": 0.0,
    }
    with pytest.raises(ValueError):
        store.begin_trial(*identity, protocol=dict(protocol, split="final_test"))
    trial = store.begin_trial(*identity, protocol=protocol)

    def evaluator(*, task_id, enabled, protocol):
        return dict(
            protocol,
            task_id=task_id,
            skill_enabled=enabled,
            skill_version=protocol["version"],
            reward_known=True,
            reward=0.7 if enabled else 0.5,
        )

    summary = store.run_trial(trial, evaluator)
    assert summary["promoted"] and summary["sign_test_p"] == 0.03125
    assert store.snapshot(tmp_path / "old.json") == frozen
    assert len(store.snapshot(tmp_path / "new.json")["cards"]) == 9
    # Resume must not call either arm again.
    store.run_trial(trial, lambda **_: pytest.fail("already completed trial rerun"))


def test_mismatched_trial_conditions_cannot_promote(store):
    store.enqueue(example_case())
    identity = store.propose("case-0", proposal(), step=10)
    protocol = {
        "split": "development",
        "task_ids": list("abcde"),
        "solver_snapshot": "frozen",
        "worker_config": "fixed",
        "budget": 900,
        "other_skills_snapshot": "v",
        "min_tasks": 5,
        "minimum_mean_gain": 0,
    }
    trial = store.begin_trial(*identity, protocol=protocol)
    summary = store.run_trial(trial, lambda **kw: {"reward_known": True, "reward": 1})
    assert summary["valid_tasks"] == 0 and not summary["promoted"]


def test_formal_factory_uses_v2_without_legacy_lifecycle(tmp_path):
    config = replace(
        AdaptiveApplicationConfig(),
        skillbank_mode=SCHEMA,
        skillbank_path=tmp_path / "bank.json",
        skill_cases_path=tmp_path / "cases.json",
        trace_path=tmp_path / "trace.jsonl",
        route_health_path=tmp_path / "routes.json",
    )
    frozen = freeze_collection(config, tmp_path / "cycle0")
    app = create_adaptive_application(frozen, mock=True, director_tokenizer=ByteTokenizer())
    assert isinstance(app.skillbank, DirectorSkillBankV2) and app.skill_lifecycle is None
    assert (
        load_bank(freeze_collection(config, tmp_path / "cycle0")).snapshot_id
        == app.skillbank.snapshot_id
    )
    # Actual synthetic Director path, without model services or environment endpoints.
    result = app.solve("Check evidence then answer", task_id="synthetic")
    assert result.solver_result.skill_context["snapshot_id"] == app.skillbank.snapshot_id
    assert result.solver_result.skill_context["prompt_tokens"] <= 1024
    old = tmp_path / "old-cycle"
    old.mkdir()
    (old / "solver_rollouts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="original legacy"):
        freeze_collection(config, old)


def test_trial_arms_resume_without_repeating_successful_arm(store):
    store.enqueue(example_case())
    identity = store.propose("case-0", proposal(), step=10)
    protocol = {
        "split": "development",
        "task_ids": list("abcde"),
        "solver_snapshot": "frozen",
        "worker_config": "fixed",
        "budget": 900,
        "other_skills_snapshot": "v",
        "min_tasks": 5,
        "minimum_mean_gain": 0,
    }
    trial = store.begin_trial(*identity, protocol=protocol)
    calls = []

    def interrupted(**kw):
        calls.append((kw["task_id"], kw["enabled"]))
        if kw["enabled"]:
            raise RuntimeError("interrupted")
        return {"reward": 0, "reward_known": False}

    with pytest.raises(RuntimeError):
        store.run_trial(trial, interrupted)
    assert calls == [("a", False), ("a", True)]
    calls.clear()
    with pytest.raises(RuntimeError):
        store.run_trial(trial, interrupted)
    assert calls == [("a", True)]


def test_trial_context_changes_only_target_skill(store, tmp_path):
    from selfplay_graph_flowsteer.skill_trials import TrialContextBank

    store.enqueue(example_case())
    identity = store.propose("case-0", proposal(), step=10)
    target = next(c for c in store.cards() if (c["card"]["skill_id"], c["version"]) == identity)
    snapshot = store.snapshot(tmp_path / "view.json")
    manifests = []
    for enabled in (False, True):
        bank = TrialContextBank(snapshot, target, enabled=enabled, prompt_token_budget=3000)
        _, _, manifest = bank.select_context(
            "evidence verification", task_type="qa", tokenizer=ByteTokenizer()
        )
        manifests.append(manifest)
    assert manifests[1]["selected"][:-1] == manifests[0]["selected"]
    assert manifests[1]["selected"][-1] == {"id": identity[0], "version": identity[1]}


def test_formal_trial_adapter_mock_and_resume(store, tmp_path):
    from selfplay_graph_flowsteer.skill_trials import run_application_trial

    store.enqueue(example_case())
    identity = store.propose("case-0", proposal(), step=10)
    config = replace(
        AdaptiveApplicationConfig(),
        skillbank_mode=SCHEMA,
        skillbank_prompt_token_budget=4000,
        route_health_path=tmp_path / "health.json",
    )
    tasks = [
        TaskSpec(
            str(i),
            "Check evidence and produce the answer",
            task_type="qa",
            metadata={"skill_evaluation_split": "development"},
        )
        for i in range(5)
    ]
    output = tmp_path / "trial"
    first = run_application_trial(
        config,
        tasks,
        store=store,
        skill_id=identity[0],
        version=identity[1],
        output=output,
        mock=True,
    )
    assert not first["promoted"]
    assert len(list(output.glob("task-*-o*.json"))) == 10
    assert all("result" in json.loads(p.read_text()) for p in output.glob("task-*-o*.json"))
    assert (
        run_application_trial(
            config,
            tasks,
            store=store,
            skill_id=identity[0],
            version=identity[1],
            output=output,
            mock=True,
        )
        == first
    )


def test_skill_maintenance_failure_does_not_cancel_training(tmp_path, monkeypatch):
    import selfplay_graph_flowsteer.skill_evolution_v2 as v2
    from selfplay_graph_flowsteer.application import consolidate_selfplay_skills

    def broken(*a, **kw):
        raise OSError("private error text not for logging")

    monkeypatch.setattr(v2, "consolidate", broken)
    config = replace(AdaptiveApplicationConfig(), skillbank_mode=SCHEMA)
    with pytest.warns(RuntimeWarning, match="OSError"):
        changes = consolidate_selfplay_skills(config, None, mock=True, step=1, cycle_dir=tmp_path)
    assert changes == (("skill_v2", "maintenance_failed:OSError"),)
    assert "private error" not in (tmp_path / "skillbank_v2_maintenance_error.json").read_text()


def test_checked_generation_publishes_only_future_snapshots(store, tmp_path):
    frozen = store.snapshot(tmp_path / "before.json")
    store.enqueue(example_case())
    changes = distill_pending(
        store,
        MockBackend([json.dumps(proposal())]),
        step=10,
        min_pending=1,
        activate_after_checks=True,
    )
    card = next(c for c in store.cards() if c["card"]["skill_id"] == changes[0][0])
    assert card["status"] == "active"
    assert card["provenance"] == "distilled_checked"
    assert "effectiveness_validated" not in card
    assert store.snapshot(tmp_path / "before.json") == frozen
    assert len(store.snapshot(tmp_path / "after.json")["cards"]) == 9
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0
        assert db.execute("SELECT kind FROM audit WHERE kind='skill_checked_activation'").fetchone()


def test_checked_edit_keeps_version_history(store, tmp_path):
    seed = store.cards()[0]["card"]["skill_id"]
    frozen = store.snapshot(tmp_path / "before.json")
    store.enqueue(example_case())
    store.propose(
        "case-0", dict(proposal(), edit_skill_id=seed), step=10, activate_after_checks=True
    )
    versions = {c["version"]: c["status"] for c in store.cards() if c["card"]["skill_id"] == seed}
    assert versions == {1: "deprecated", 2: "active"}
    assert store.snapshot(tmp_path / "before.json") == frozen


def test_checked_dedup_review_does_not_publish(tmp_path):
    store = SkillStore(tmp_path / "skills.db", embedder=_NearSkillEmbedder())
    store.enqueue(example_case(0))
    store.propose("case-0", proposal(), step=10, activate_after_checks=True)
    store.enqueue(example_case(1))
    identity = store.propose(
        "case-1",
        dict(proposal(), description="near candidate: evidence"),
        step=11,
        activate_after_checks=True,
    )
    card = next(c for c in store.cards() if c["card"]["skill_id"] == identity[0])
    assert card["status"] == "candidate"
    assert card["dedup_review"]["status"] == "needs_review"


def test_checked_encoder_failure_keeps_case_retryable(tmp_path):
    class BrokenEmbedder:
        def encode(self, *a, **kw):
            raise RuntimeError("encoder unavailable")

    store = SkillStore(tmp_path / "skills.db")
    store.enqueue(example_case(0))
    store.propose("case-0", proposal(), step=1, activate_after_checks=True)
    store.embedder = BrokenEmbedder()
    store.enqueue(example_case(1))
    backend = MockBackend([json.dumps(dict(proposal(), description="new evidence method"))])
    assert distill_pending(store, backend, step=10, min_pending=1, activate_after_checks=True) == []
    with store.connect() as db:
        assert db.execute("SELECT status FROM cases WHERE id='case-1'").fetchone()[0] == "pending"
    assert len(store.cards()) == 1


def test_usage_healthbench_mean_and_idempotence(store):
    sid = store.cards()[0]["card"]["skill_id"]
    tasks = {
        "h": TaskSpec(
            "h",
            "question",
            task_type="healthcare",
            metadata={"dataset": "healthbench_professional"},
        )
    }

    def row(i, reward, **extras):
        return dict(
            task_id="h",
            rollout_id=str(i),
            reward=reward,
            metadata={
                "reward_known": True,
                "solver_answer": "answer",
                "skill_context": {"selected": [{"id": sid, "version": 1}]},
                **extras,
            },
        )

    rows = [
        row(0, 0.2),
        row(1, 0.4),
        row(2, 0.3),
        row(3, 0, worker_backend_failure=True),
        row(4, 0, reward_known=False),
    ]
    for _ in range(2):
        ingest_rollouts(store, tasks, rows, run="usage", step=1)
    summary = store.usage_summary()[0]
    assert summary["usage_count"] == 5
    assert summary["valid_count"] == 3
    assert summary["mean_reward"] == pytest.approx(0.3)
    assert summary["helpful_count"] == summary["hurt_count"] == 0
    assert summary["net_score"] is None
    assert summary["infrastructure_failure_count"] == summary["unscored_count"] == 1


def test_usage_binary_filters_and_retirement_preserve_seed_snapshot(store, tmp_path):
    store.enqueue(example_case())
    sid, version = store.propose("case-0", proposal(), step=1, activate_after_checks=True)
    seed = next(c["card"]["skill_id"] for c in store.cards() if c["status"] == "seed")
    old = store.snapshot(tmp_path / "old.json")
    tasks = {"q": TaskSpec("q", "question", metadata={"dataset": "aime"})}
    rows = []
    for i, (reward, answer, extras) in enumerate(
        [
            (1, "answer", {}),
            (0, "wrong", {}),
            (0, "wrong", {}),
            (0, "", {}),
            (0, "wrong", {"extraction_failed": True}),
            (0, "wrong", {"worker_backend_failure": True}),
        ]
    ):
        rows.append(
            dict(
                task_id="q",
                rollout_id=str(i),
                reward=reward,
                metadata={
                    "reward_known": True,
                    "solver_answer": answer,
                    "skill_context": {
                        "selected": [{"id": sid, "version": version}, {"id": seed, "version": 1}]
                    },
                    **extras,
                },
            )
        )
    ingest_rollouts(store, tasks, rows, run="binary", step=1)
    r = next(r for r in store.usage_summary() if r["skill_id"] == sid)
    assert (r["helpful_count"], r["hurt_count"], r["net_score"]) == (1, 2, -1)
    assert store.retire_negative_usage(min_usage=4, step=10) == []
    assert store.retire_negative_usage(min_usage=3, step=10) == [(sid, version)]
    assert store.retire_negative_usage(min_usage=3, step=10) == []
    assert store.snapshot(tmp_path / "old.json") == old
    assert all(c["card"]["skill_id"] != sid for c in store.snapshot(tmp_path / "new.json")["cards"])
    assert next(c for c in store.cards() if c["card"]["skill_id"] == seed)["status"] == "seed"


def test_usage_healthbench_zero_never_retires_nonseed(store):
    store.enqueue(example_case())
    sid, version = store.propose("case-0", proposal(), step=1, activate_after_checks=True)
    for i in range(5):
        store.record_usage_outcome(
            dict(
                run="h",
                cycle=1,
                rollout=str(i),
                skill=sid,
                version=version,
                dataset="healthbench_professional",
                mode="continuous",
                reward=0,
                infrastructure_failure=False,
                feedback=None,
            )
        )
    assert store.retire_negative_usage(min_usage=3, step=10) == []
    assert store.usage_summary()[0]["mean_reward"] == 0
