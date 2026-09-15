import pytest

from selfplay_graph_flowsteer.benchmark_reporting import benchmark_summary, estimate
from selfplay_graph_flowsteer.qa_metrics import hotpot_evidence_metrics
from selfplay_graph_flowsteer.wandb_tracking import cycle_payload


def test_fixed_denominator_and_source_weighting():
    rows = [
        {"task_id": "a", "source_id": "a", "task_outcome_passed": True},
        {"task_id": "a-copy", "source_id": "a", "task_outcome_passed": True},
        {"task_id": "b", "source_id": "b", "task_outcome_passed": False},
        {"task_id": "c", "source_id": "c", "task_outcome_passed": None},
    ]
    report = benchmark_summary(rows, "aime")
    metric = report["metrics"]["accuracy"]
    assert metric["observed"]["mean"] == 0.5
    assert metric["missing_as_zero"]["mean"] == pytest.approx(1 / 3)
    assert metric["observed"]["observed_slots"] == 3
    assert metric["missing_as_zero"]["source_count"] == 3
    assert estimate(rows, "task_outcome_passed") == estimate(rows, "task_outcome_passed")


def test_healthbench_clips_after_averaging():
    rows = [
        {"task_id": "a", "healthbench_official_adjusted_score": -1.0},
        {"task_id": "b", "healthbench_official_adjusted_score": 0.5},
    ]
    report = benchmark_summary(rows, "healthbench_professional")
    assert report["metrics"]["length_adjusted_score"]["observed"]["mean"] == 0.0
    assert "success_rate" not in report["metrics"]


def test_hotpot_missing_gold_is_not_a_zero_score():
    assert hotpot_evidence_metrics("answer", None, {}) == {
        "gold_available": False,
        "submission_valid": False,
    }
    report = benchmark_summary([{"task_id": "a"}], "hotpotqa")
    assert report["metrics"]["joint_f1"]["observed"]["mean"] is None
    assert "missing_as_zero" not in report["metrics"]["joint_f1"]


def test_hotpot_joint_uses_precision_recall_products():
    result = hotpot_evidence_metrics(
        '{"answer":"x", "supporting_facts":[["A",0],["wrong",0]]}',
        [["A", 0]],
        {"answer_em": 0, "answer_precision": 1.0, "answer_recall": 0.5},
    )
    assert result["support_f1"] == pytest.approx(2 / 3)
    assert result["joint_f1"] == 0.5
    assert result["joint_em"] == 0


def test_benchmark_numeric_slices_reach_wandb_without_ids():
    report = benchmark_summary(
        [
            {"task_id": "private-question", "split": "valid_unseen", "task_outcome_passed": True},
        ],
        "alfworld",
    )
    payload = cycle_payload(
        {
            "cycle": 0,
            "experiment": {"evaluation_only": True},
            "outcomes": {"datasets": {"alfworld": {"benchmark": report}}},
        }
    )
    assert payload["eval/datasets/alfworld/benchmark/metrics/success_rate/observed/mean"] == 1
    assert not any("private-question" in key for key in payload)
