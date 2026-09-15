"""Read-only benchmark summaries with explicit populations and task-level uncertainty."""

import random
import statistics
from collections import defaultdict


def estimate(rows, field, *, missing_zero=False, clip=False):
    groups = defaultdict(list)
    known = 0
    for row in rows:
        value = row.get(field)
        if value is not None:
            known += 1
        if value is not None or missing_zero:
            groups[row.get("source_id") or row["task_id"]].append(float(value or 0))
    means = [statistics.fmean(values) for values in groups.values()]
    result = {"observed_slots": known, "planned_slots": len(rows), "source_count": len(means)}
    if not means:
        return {**result, "mean": None, "ci95_low": None, "ci95_high": None}
    transform = (lambda x: min(1.0, max(0.0, x))) if clip else (lambda x: x)
    mean = transform(statistics.fmean(means))
    low = high = None
    if len(means) > 1:
        rng = random.Random(20260915)
        draws = sorted(
            transform(statistics.fmean(rng.choices(means, k=len(means)))) for _ in range(1000)
        )
        low, high = draws[24], draws[974]
    return {**result, "mean": mean, "ci95_low": low, "ci95_high": high}


def benchmark_summary(rows, dataset):
    continuous = dataset == "healthbench_professional"
    fields = (
        {
            "length_adjusted_score": "healthbench_official_adjusted_score",
            "raw_rubric_score": "healthbench_raw_rubric_score",
        }
        if continuous
        else {"resolved_rate" if dataset == "swe_bench" else "success_rate": "task_outcome_passed"}
    )
    if dataset in {"hotpotqa", "nq_open"}:
        fields.update(answer_em="qa_answer_em", answer_f1="qa_answer_f1")
    if dataset == "webshop":
        fields["score"] = "task_score_raw"
    if dataset == "aime":
        fields = {"accuracy": "task_outcome_passed"}
    evidence = None
    if dataset == "hotpotqa":
        evidence = {
            "gold_available_slots": sum(
                bool(r.get("hotpot_evidence", {}).get("gold_available")) for r in rows
            ),
            "valid_submission_slots": sum(
                bool(r.get("hotpot_evidence", {}).get("submission_valid")) for r in rows
            ),
        }
        rows = [
            {**r, **{f"hotpot_{key}": value for key, value in r.get("hotpot_evidence", {}).items()}}
            for r in rows
        ]
        fields.update(
            {
                name: f"hotpot_{name}"
                for name in ("support_em", "support_f1", "joint_em", "joint_f1")
            }
        )
    metrics = {}
    for name, field in fields.items():
        metrics[name] = {
            "observed": estimate(rows, field, clip=continuous),
            "missing_as_zero": estimate(rows, field, missing_zero=True, clip=continuous),
        }
        if dataset == "hotpotqa" and name.startswith(("support_", "joint_")):
            metrics[name].pop("missing_as_zero")
    slices = {}
    for dimension in ("split", "task_type", "difficulty", "use_case", "red_teaming"):
        groups = defaultdict(list)
        for row in rows:
            value = row.get(dimension)
            if isinstance(value, (str, int, bool)):
                groups[str(value)].append(row)
        slices[dimension] = {
            label: {name: estimate(group, field, clip=continuous) for name, field in fields.items()}
            for label, group in groups.items()
        }
    return {
        "schema": "benchmark_report_v1",
        "swe_status": {
            "resolved": sum(r.get("task_outcome_passed") is True for r in rows),
            "official_completed": sum(
                r.get("swe_official_evaluation_complete") is True for r in rows
            ),
            "official_unresolved": sum(
                r.get("swe_official_evaluation_complete") is True
                and r.get("task_outcome_passed") is False
                for r in rows
            ),
            "unscored": sum(not r.get("reward_known") for r in rows),
            "valid_submission_ready": sum(
                r.get("swe_valid_submission_ready") is True for r in rows
            ),
        }
        if dataset == "swe_bench"
        else None,
        "aggregation": "equal_source_weight_after_averaging_repeated_rollouts",
        "ci_method": "source_bootstrap_percentile_1000_seed_20260915",
        "planned_slots": len(rows),
        "recorded_slots": sum(bool(row.get("recorded")) for row in rows),
        "unscored_slots": sum(not row.get("reward_known") for row in rows),
        "metrics": metrics,
        "evidence_coverage": evidence,
        "slices": slices,
    }
