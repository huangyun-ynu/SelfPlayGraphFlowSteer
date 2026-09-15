from __future__ import annotations

import copy
import json

import pytest

from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.pats import resolve_scope
from selfplay_graph_flowsteer.skill_evolution_v2 import SkillStore, ingest_rollouts

PATS_FIELDS = {"pats_scope", "pats_snapshot_id", "selection_revision"}
TABLES = ("events", "usage_outcomes")


def usage_row(rollout, skill, scope, reward, *, version=2):
    return {
        "task_id": "task",
        "rollout_id": rollout,
        "reward": reward,
        "metadata": {
            "reward_known": True,
            "solver_answer": "A substantive public deliverable",
            "skill_context": {
                "snapshot_id": "collection-view",
                "selected": [{"id": skill, "version": version}],
                "pats_enabled": True,
                "pats_scope": scope,
                "pats_snapshot_id": "pats-view",
                "selection_revision": "learned_first_v1",
            },
        },
    }


def ingest(store, rows):
    # Scope comes from the frozen manifest, not this task's present metadata.
    tasks = {
        "task": TaskSpec("task", "Public question", task_type="qa", metadata={"dataset": "nq"})
    }
    ingest_rollouts(store, tasks, rows, run="run", step=1)


def ledger(store, table):
    assert table in TABLES
    with store.connect() as db:
        return {row[0]: json.loads(row[1]) for row in db.execute(f"SELECT id,payload FROM {table}")}


def emulate_old_accounting(store, *, rollout=None):
    # Older code discarded these manifest fields before writing either ledger.
    for table in TABLES:
        old = ledger(store, table)
        with store.connect() as db:
            for key, event in old.items():
                if rollout is not None and event["rollout"] != rollout:
                    continue
                payload = {key: value for key, value in event.items() if key not in PATS_FIELDS}
                db.execute(
                    f"UPDATE {table} SET payload=? WHERE id=?",
                    (json.dumps(payload, ensure_ascii=False, sort_keys=True), key),
                )


def test_scoped_seed_versions_and_new_cards_are_recorded_without_global_rows(tmp_path):
    store = SkillStore(tmp_path / "skills.db")
    store.initialize_seeds()
    seeds = store.cards()
    skill = seeds[0]["card"]["skill_id"]
    easy = resolve_scope("qa", {"dataset": "nq", "difficulty_bucket": "easy"})
    hard = resolve_scope("qa", {"dataset": "nq", "difficulty_bucket": "hard"})
    rows = [
        usage_row("easy", skill, easy, 1.0),
        usage_row("hard", skill, hard, 0.0),
        usage_row("new", "pats_new", easy, 1.0, version=1),
    ]
    for _ in range(2):
        ingest(store, rows)
    assert store.cards() == seeds
    assert all(record["version"] == 1 for record in seeds)
    assert all(record["card"]["skill_id"] != "pats_new" for record in seeds)
    for table in TABLES:
        events = list(ledger(store, table).values())
        assert len(events) == 3
        assert {event["pats_scope"] for event in events} == {easy, hard}
        assert all(event["pats_snapshot_id"] == "pats-view" for event in events)
        assert all(event["selection_revision"] == "learned_first_v1" for event in events)
    summary = {(item["skill_id"], item["pats_scope"]): item for item in store.usage_summary()}
    assert len(summary) == 3
    assert summary[skill, easy]["usage_count"] == summary[skill, hard]["usage_count"] == 1
    assert summary[skill, easy]["mean_reward"] == 1.0
    assert summary[skill, hard]["mean_reward"] == 0.0
    assert summary[skill, easy]["helpful_count"] == summary[skill, hard]["hurt_count"] == 1
    assert summary["pats_new", easy]["version"] == 1


def test_old_accounting_enriches_only_when_original_manifest_is_replayed(tmp_path):
    store = SkillStore(tmp_path / "skills.db")
    rows = [usage_row("old", "seed", "old-scope", 0.0), usage_row("new", "seed", "new-scope", 1.0)]
    ingest(store, rows)
    emulate_old_accounting(store, rollout="old")
    original_keys = {table: set(ledger(store, table)) for table in TABLES}
    before = store.usage_summary()
    assert len(before) == 2
    assert next(item for item in before if "pats_scope" not in item)["mean_reward"] == 0.0
    assert next(item for item in before if "pats_scope" in item)["pats_scope"] == "new-scope"

    # Replaying the identical raw rows enriches in place, without double counting.
    ingest(store, rows)
    after = {table: ledger(store, table) for table in TABLES}
    for table in TABLES:
        assert set(after[table]) == original_keys[table]
        assert {event["pats_scope"] for event in after[table].values()} == {
            "old-scope",
            "new-scope",
        }
    assert {item["pats_scope"] for item in store.usage_summary()} == {"old-scope", "new-scope"}
    ingest(store, rows)
    assert {table: ledger(store, table) for table in TABLES} == after


@pytest.mark.parametrize(
    "table,method,reward_field",
    [
        ("events", "record_event", "task_reward"),
        ("usage_outcomes", "record_usage_outcome", "reward"),
    ],
)
def test_old_record_enrichment_does_not_hide_changed_reward(tmp_path, table, method, reward_field):
    store = SkillStore(tmp_path / "skills.db")
    row = usage_row("old", "pats_new", "scope", 0.0)
    ingest(store, [row])
    enriched = next(iter(ledger(store, table).values()))
    emulate_old_accounting(store)
    before = ledger(store, table)
    enriched[reward_field] = 1.0
    with pytest.raises(ValueError, match="conflicting repeated skill"):
        getattr(store, method)(enriched)
    assert ledger(store, table) == before


@pytest.mark.parametrize(
    "table,method", [("events", "record_event"), ("usage_outcomes", "record_usage_outcome")]
)
@pytest.mark.parametrize("change", ["extra_field", "removed_field", "numeric_type"])
def test_enrichment_preserves_every_original_field(tmp_path, table, method, change):
    store = SkillStore(tmp_path / "skills.db")
    ingest(store, [usage_row("old", "pats_new", "scope", 0.0)])
    enriched = next(iter(ledger(store, table).values()))
    emulate_old_accounting(store)
    before = ledger(store, table)
    if change == "extra_field":
        enriched["unapproved_metadata"] = "must not be added"
    elif change == "removed_field":
        del enriched["dataset"]
    else:
        # JSON booleans and numbers must not compare equal through Python's ==.
        enriched["infrastructure_failure"] = 0
    with pytest.raises(ValueError, match="conflicting repeated skill"):
        getattr(store, method)(enriched)
    assert ledger(store, table) == before


@pytest.mark.parametrize(
    "table,method", [("events", "record_event"), ("usage_outcomes", "record_usage_outcome")]
)
@pytest.mark.parametrize("field", sorted(PATS_FIELDS))
def test_recorded_pats_identity_cannot_change_on_replay(tmp_path, table, method, field):
    store = SkillStore(tmp_path / "skills.db")
    ingest(store, [usage_row("old", "pats_new", "scope", 0.0)])
    before = ledger(store, table)
    changed = copy.deepcopy(next(iter(before.values())))
    changed[field] = "different"
    with pytest.raises(ValueError, match="conflicting repeated skill"):
        getattr(store, method)(changed)
    assert ledger(store, table) == before


def test_legacy_accounting_and_metric_keys_remain_unchanged(tmp_path):
    store = SkillStore(tmp_path / "skills.db")
    row = usage_row("legacy", "seed", "must-not-be-recorded", 1.0, version=1)
    row["metadata"]["skill_context"]["pats_enabled"] = False
    ingest(store, [row])
    for table in TABLES:
        assert not PATS_FIELDS.intersection(next(iter(ledger(store, table).values())))
    assert store.usage_summary() == [
        {
            "skill_id": "seed",
            "version": 1,
            "dataset": "nq",
            "mode": "sesa_outcome",
            "usage_count": 1,
            "valid_count": 1,
            "reward_sum": 1.0,
            "helpful_count": 1,
            "hurt_count": 0,
            "infrastructure_failure_count": 0,
            "unscored_count": 0,
            "skipped_feedback_count": 0,
            "mean_reward": 1.0,
            "net_score": 1,
        }
    ]


def test_scope_usage_accumulates_across_frozen_snapshots(tmp_path):
    store = SkillStore(tmp_path / "skills.db")
    first = usage_row("first", "seed", "scope", 1.0)
    later = usage_row("later", "seed", "scope", 0.0)
    later["metadata"]["skill_context"]["pats_snapshot_id"] = "next-pats-view"
    ingest(store, [first, later])
    summary = store.usage_summary()
    assert len(summary) == 1
    assert summary[0]["pats_scope"] == "scope"
    assert summary[0]["usage_count"] == 2
    assert summary[0]["mean_reward"] == 0.5
    assert {event["pats_snapshot_id"] for event in ledger(store, "usage_outcomes").values()} == {
        "pats-view",
        "next-pats-view",
    }
