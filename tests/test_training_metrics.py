from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.training_metrics import (
    TrainingMetricsStore,
    _percentile,
    collect_cycle_metrics,
)


def test_percentile_uses_linear_interpolation() -> None:
    assert _percentile([0, 10], 0.50) == 5.0
    assert _percentile([0, 10], 0.95) == 9.5


def _skill_metrics(bank_path: Path, cycle_dir: Path | None = None):
    return collect_cycle_metrics(
        cycle=0,
        proposer_batch=TrainingBatch("proposer", ()),
        solver_batch=TrainingBatch("solver", ()),
        updates=(),
        skillbank_path=bank_path,
        cycle_dir=cycle_dir,
    )


def _versioned_card(skill_id: str, version: int = 1, *, status: str = "active"):
    return {
        "card": {"skill_id": skill_id, "stats": {"is_seed": status == "seed"}},
        "version": version,
        "status": status,
    }


def test_legacy_skillbank_size_and_csv_remain_compatible(tmp_path: Path) -> None:
    bank = tmp_path / "legacy.json"
    bank.write_text(json.dumps({"skills": [{"skill_id": "old", "stats": {"usage_count": 3}}]}))
    record = _skill_metrics(bank)
    assert record["skillbank"]["size"] == 1
    assert record["skillbank"]["total_usage"] == 3
    assert "pats" not in record["skillbank"]
    store = TrainingMetricsStore(tmp_path / "metrics")
    store.append(record)
    with store.csv_path.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["skillbank_size"] == "1"


def test_v2_snapshot_counts_global_records_without_pats(tmp_path: Path) -> None:
    bank = tmp_path / "snapshot.json"
    bank.write_text(
        json.dumps(
            {
                "schema": "director_skill_v2",
                "snapshot_id": "frozen-v2",
                "cards": [
                    _versioned_card("seed", status="seed"),
                    _versioned_card("learned", 2),
                    _versioned_card("retired", status="deprecated"),
                ],
            }
        )
    )
    metrics = _skill_metrics(bank)["skillbank"]
    assert metrics["size"] == metrics["global_card_count"] == 2
    assert metrics["global_seed_card_count"] == 1
    assert metrics["global_unique_skill_id_count"] == 2
    assert metrics["skill_ids"] == ["seed", "learned"]
    assert "pats" not in metrics


def test_cycle_snapshot_and_review_work_without_live_bank_file(tmp_path: Path) -> None:
    cycle = tmp_path / "cycle-0000"
    cycle.mkdir()
    bank = tmp_path / "custom-runtime-state" / "solver_skillbank.json"
    assert not bank.exists()
    scopes = {
        '["nq_open","qa","easy"]': {
            "cards": [_versioned_card("seed", status="seed"), _versioned_card("learned")],
            "ema": 0.1,
            "mode": "EXPAND",
            "policy_snapshot": "solver-before-update",
        },
        '["nq_open","qa","hard"]': {
            "cards": [_versioned_card("seed", 2), _versioned_card("learned")],
            "ema": 0.4,
            "mode": "REVISE",
        },
    }
    frozen = cycle / "director_skill_snapshot.v2.json"
    frozen.write_text(
        json.dumps(
            {
                "schema": "director_skill_v2",
                "snapshot_id": "frozen-cycle",
                "cards": [_versioned_card("seed", status="seed")],
                "pats": {"snapshot_id": "pats-before-review", "step": 0, "scopes": scopes},
            }
        )
    )
    (cycle / "pats_review.json").write_text(
        json.dumps(
            {
                "step": 1,
                "credible_groups": 4,
                "refiner_calls": 2,
                "reviews": [
                    {"status": "updated", "operations": [{"op": "ADD"}, {"op": "UPDATE"}]},
                    {"status": "rejected", "operations": [{"op": "DELETE"}]},
                    {"status": "cycle_review_budget"},
                ],
            }
        )
    )
    record = _skill_metrics(bank, cycle)
    metrics = record["skillbank"]
    assert metrics["available"] is True
    assert metrics["source_path"] == str(frozen)
    assert metrics["size"] == metrics["global_seed_card_count"] == 1
    pats = metrics["pats"]
    assert pats["scope_count"] == 2
    assert pats["scoped_card_instance_count"] == 4
    assert pats["scoped_unique_skill_id_count"] == 2
    assert pats["scopes"]['["nq_open","qa","hard"]']["skill_versions"][0] == {
        "id": "seed",
        "version": 2,
    }
    assert pats["snapshot_step"] == 0 and pats["review_step"] == 1
    assert pats["review_updated_scope_count"] == pats["review_rejected_scope_count"] == 1
    assert pats["refiner_calls"] == 2
    assert pats["applied_operation_counts"] == {"ADD": 1, "UPDATE": 1}
    assert pats["review_status_counts"] == {
        "cycle_review_budget": 1,
        "rejected": 1,
        "updated": 1,
    }
    store = TrainingMetricsStore(tmp_path / "metrics")
    store.append(record)
    with store.csv_path.open() as stream:
        assert next(csv.DictReader(stream))["skillbank_size"] == "1"
    saved = json.loads(store.latest_path.read_text())
    assert saved["skillbank"]["pats"]["scoped_card_instance_count"] == 4


def test_missing_pats_snapshot_and_review_do_not_claim_zero_counts(tmp_path: Path) -> None:
    cycle = tmp_path / "evaluation"
    cycle.mkdir()
    (cycle / "skill_context_contract.json").write_text(
        json.dumps({"pats_enabled": True, "context_enabled": False})
    )
    metrics = _skill_metrics(tmp_path / "missing.json", cycle)["skillbank"]
    assert metrics["available"] is False
    assert metrics["pats"] == {
        "context_enabled": False,
        "snapshot_available": False,
        "review_available": False,
    }


def test_known_empty_scoped_view_is_distinct_from_missing_review(tmp_path: Path) -> None:
    bank = tmp_path / "snapshot.json"
    bank.write_text(
        json.dumps(
            {
                "schema": "director_skill_v2",
                "snapshot_id": "frozen",
                "cards": [_versioned_card("seed", status="seed")],
                "pats": {"snapshot_id": "withdrawn", "scopes": {"scope": {"cards": []}}},
            }
        )
    )
    pats = _skill_metrics(bank)["skillbank"]["pats"]
    assert pats["snapshot_available"] is True
    assert pats["scope_count"] == 1
    assert pats["scoped_card_instance_count"] == pats["scoped_unique_skill_id_count"] == 0
    assert pats["review_available"] is False
    assert "review_updated_scope_count" not in pats


def test_policy_off_reports_legacy_low_diversity_without_collapse_alert() -> None:
    samples = [
        replace(
            _sample("s", index, 1.0),
            task_id="task",
            canonical_graph_key="single-agent",
            graph_features=(
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        )
        for index in range(5)
    ]
    solver = TrainingBatch("solver", tuple(samples))
    proposer = TrainingBatch("proposer", (_sample("p", 0, 1.0),))

    policy_off = collect_cycle_metrics(
        cycle=0,
        proposer_batch=proposer,
        solver_batch=solver,
        updates=(),
        context={"structural_exploration_policy": "off"},
    )["topology_policy"]
    stratified = collect_cycle_metrics(
        cycle=0,
        proposer_batch=proposer,
        solver_batch=solver,
        updates=(),
        context={"structural_exploration_policy": "stratified"},
    )["topology_policy"]

    assert policy_off["legacy_low_diversity_observed"] is True
    assert policy_off["collapse_alert"] is False
    assert policy_off["collapse_alert_reasons"] == []
    assert stratified["collapse_alert"] is True
    assert stratified["collapse_alert_reasons"] == ["legacy_low_diversity"]


def _sample(role: str, index: int, reward: float) -> TrainingSample:
    return TrainingSample(
        rollout_id=f"{role}-{index}",
        task_id=f"task-{index // 2}",
        token_ids=(1, 2, 3, 4),
        action_mask=(0, 1, 1, 0),
        reward=reward,
        advantage=reward - 0.5,
        density=0.8 + index * 0.1,
        canonical_graph_key=f"graph-{index % 2}",
        graph_features=tuple(float(index + value) for value in range(15)),
        metadata={
            "verification": {"passed": bool(reward)},
            "action_token_spans": [(0, 1), (1, 2), (2, 3)],
        },
    )


def test_semantic_graph_metrics_use_schema_and_canonical_topology() -> None:
    semantic = replace(
        _sample("s", 0, 1.0),
        canonical_graph_key="[[[0,false],[1,true]],[[0,1]],[]]",
        graph_features=tuple(0.01 for _ in range(9239)),
        metadata={
            "graph_feature_schema_id": "semantic-test",
            "graph_feature_schema": {"feature_dimension": 9239},
        },
    )
    solver = TrainingBatch("solver", (semantic,))
    record = collect_cycle_metrics(
        cycle=0,
        proposer_batch=TrainingBatch("proposer", (_sample("p", 0, 1.0),)),
        solver_batch=solver,
        updates=(),
    )

    assert record["graph"]["features"] == {}
    assert record["graph"]["dimensions"]["mean"] == 9239.0
    assert record["graph"]["schema_ids"] == {"semantic-test": 1}
    assert record["topology_policy"]["agent_count"]["mean"] == 2.0
    assert record["topology_policy"]["directed_edge_count"]["mean"] == 1.0
    assert record["topology_policy"]["relation_graph_rate"] == 1.0


def test_runtime_metrics_aggregate_routes_actions_tokens_and_errors() -> None:
    sample = _sample("s", 0, 1.0)
    sample.metadata.update(
        {
            "duration_s": 2.5,
            "token_in": 120,
            "token_out": 30,
            "interactive_turns": 4,
            "responsibility_violation_count": 2,
            "duplicate_responsibility_rejection_count": 1,
            "duplicate_responsibility_detection_count": 2,
            "duplicate_responsibility_decision_counts": {
                "record_only": 1,
                "rewrite_requested": 1,
            },
            "duplicate_responsibility_policy": "record_only",
            "cross_agent_exact_read_overlap": {
                "duplicate_read_only_exploration": True,
                "flagged_pair_count": 1,
                "pairs": [{"shared_exact_read_count": 4}],
            },
            "protocol_recovery_count": 3,
            "director_action_repairs": 2,
            "director_action_repair_successes": 1,
            "director_discarded_output_chars": 400,
            "solver_trace": {
                "events": [
                    {
                        "payload": {
                            "execution": {
                                "errors": {"broken": "diagnostic"},
                                "artifacts": {
                                    "artifact-1": {
                                        "answer": "WORKER_BACKEND_FAILURE",
                                        "unresolved_issues": [
                                            "transient_backend_error:APITimeoutError"
                                        ],
                                        "model_route": "gpt",
                                        "model": "gpt-test",
                                        "react_trace": [
                                            {
                                                "action": {"name": "search"},
                                                "observation": {"status": "ok"},
                                            },
                                            {
                                                "action": {"name": "search"},
                                                "observation": {
                                                    "status": "error",
                                                    "error": {"code": "timeout"},
                                                },
                                            },
                                        ],
                                    }
                                },
                            }
                        }
                    }
                ]
            },
        }
    )
    solver = TrainingBatch("solver", (sample,))
    record = collect_cycle_metrics(
        cycle=0,
        proposer_batch=TrainingBatch("proposer", (_sample("p", 0, 0.5),)),
        solver_batch=solver,
        updates=(),
    )
    runtime = record["runtime"]
    assert runtime["rollout_duration_s"]["mean"] == 2.5
    assert runtime["token_in"]["total"] == 120
    assert runtime["token_out"]["total"] == 30
    assert runtime["model_route_counts"] == {"gpt": 1}
    assert runtime["action_counts"] == {"search": 2}
    assert runtime["action_success_rate"] == 0.5
    assert runtime["action_error_codes"] == {"timeout": 1}
    assert runtime["execution_error_count"] == 1
    assert runtime["backend_failure_count"] == 1
    assert runtime["backend_failure_types"] == {"APITimeoutError": 1}
    assert runtime["protocol_failure_count"] == 0
    assert runtime["responsibility_protocol"] == {
        "violation_count": 2,
        "violation_rate_per_director_turn": 0.5,
        "protocol_recovery_steps": 3,
        "recovered_rollout_count": 1,
        "recovery_rate_per_rollout": 1.0,
    }
    assert runtime["responsibility_overlap_audit"] == {
        "duplicate_responsibility_detection_count": 2,
        "duplicate_responsibility_rejection_count": 1,
        "duplicate_responsibility_decision_counts": {
            "record_only": 1,
            "rewrite_requested": 1,
        },
        "duplicate_responsibility_policy_rollout_counts": {"record_only": 1},
        "read_overlap_audited_rollout_count": 1,
        "duplicate_read_only_rollout_count": 1,
        "duplicate_read_only_rollout_rate": 1.0,
        "duplicate_read_only_pair_count": 1,
        "shared_exact_read_count": 4,
    }
    assert runtime["director_action_protocol"] == {
        "repair_attempts": 2,
        "repair_successes": 1,
        "repair_success_rate": 0.5,
        "discarded_output_chars": 400,
        "discarded_output_chars_per_turn": 100.0,
    }
