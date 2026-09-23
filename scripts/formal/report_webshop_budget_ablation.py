"""Compare paired WebShop runs with separate execution and request token accounting."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


def read_run(root: Path) -> dict:
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    rows = {row["task_id"]: row for row in records}
    if len(records) != 128 or len(rows) != 128:
        raise ValueError(f"Expected 128 unique completed tasks: {root}")
    details, issues = {}, []
    for path in sorted((root / "trajectories").glob("*.json")):
        trace = json.loads(path.read_text())
        task_id = trace["task"]["task_id"]
        if task_id in details:
            raise ValueError(f"Duplicate trajectory: {task_id}")
        reports, artifacts, requests = set(), set(), set()
        worker_in = worker_out = director_in = director_out = 0
        blocked = active_credits = disabled_credits = request_quotes = 0
        action_count = 0
        feedback_totals = []
        failures = Counter()
        for event in trace.get("events", []):
            payload = event.get("payload", {})
            feedback_totals.extend(
                int(value)
                for value, _ in re.findall(r"tokens=(\d+)/(\d+)", payload.get("feedback", ""))
            )
            execution = payload.get("execution") or {}
            if not execution:
                continue
            # Final Canvas events can carry an earlier report again. Keep all
            # intermediate executions within each report, including peer passes
            # whose artifacts were replaced by the final revision.
            identity = json.dumps(
                [
                    execution.get("execution_events"),
                    sorted((key, value.get("artifact_id")) for key, value in execution.get("artifacts", {}).items()),
                    execution.get("token_in", 0),
                    execution.get("token_out", 0),
                ],
                sort_keys=True,
            )
            if identity not in reports:
                reports.add(identity)
                worker_in += execution.get("token_in", 0)
                worker_out += execution.get("token_out", 0)
            for artifact in execution.get("artifacts", {}).values():
                if artifact["artifact_id"] in artifacts:
                    continue
                artifacts.add(artifact["artifact_id"])
                progress = artifact.get("webshop_progress") or {}
                partition = progress.get("budget_partition") or {}
                active_credits += partition.get("execution_credit") is not None
                disabled_credits += partition.get("request_admission_enabled") is False
                action_count = max(action_count, progress.get("action_budget", {}).get("total_used", 0))
                blocked += sum(
                    diagnostic.get("no_request_dispatched") is True
                    and "credit_exhausted" in diagnostic.get("stage", "")
                    for diagnostic in artifact.get("protocol_diagnostics", [])
                )
                for request in artifact.get("backend_request_events", []):
                    event_id = request.get("event_id")
                    if not event_id or event_id in requests:
                        continue
                    requests.add(event_id)
                    request_quotes += bool(request.get("request_token_budget"))
                    if request.get("event") == "backend_request_failure":
                        failures[str(request.get("failure_type", request.get("stage")))] += 1
        director_requests = set()
        director_missing_usage = 0
        for turn in trace.get("director_run", {}).get("turns", []):
            for request in turn.get("action_diagnostics", {}).get("backend_request_events", []):
                event_id = request.get("event_id")
                if not event_id or event_id in director_requests:
                    continue
                director_requests.add(event_id)
                usage = request.get("completion_usage") or {}
                if "token_in" in usage and "token_out" in usage:
                    director_in += usage["token_in"]
                    director_out += usage["token_out"]
                elif request.get("event") == "backend_request_success":
                    director_missing_usage += 1
        worker_total = worker_in + worker_out
        if worker_total != max(feedback_totals, default=0):
            issues.append({"task_id": task_id, "kind": "ledger_feedback_mismatch", "ledger": worker_total, "feedback": max(feedback_totals, default=0)})
        row = rows[task_id]
        details[task_id] = {
            "passed": bool(row["passed"]), "reward": row["score"],
            "reported_worker_tokens": row["token_cost"],
            "worker_input_tokens": worker_in, "worker_output_tokens": worker_out,
            "worker_tokens": worker_total,
            "director_input_tokens": director_in, "director_output_tokens": director_out,
            "director_tokens": director_in + director_out,
            "combined_tokens": worker_total + director_in + director_out,
            "director_successes_without_usage": director_missing_usage,
            "duration_s": row["duration_s"], "action_count": action_count,
            "request_budget_blocks": blocked, "artifacts_with_active_credit": active_credits,
            "artifacts_with_disabled_credit": disabled_credits,
            "requests_with_budget_quotes": request_quotes,
            "worker_request_failures": dict(failures),
        }
    if details.keys() != rows.keys():
        raise ValueError("Trajectory and record task IDs differ")
    metrics = {
        "examples": len(rows), "strict_successes": sum(x["passed"] for x in details.values()),
        "strict_success_rate": mean(x["passed"] for x in details.values()),
        "mean_score_100": 100 * mean(x["reward"] for x in details.values()),
    }
    for key in ("reported_worker_tokens", "worker_input_tokens", "worker_output_tokens", "worker_tokens", "director_input_tokens", "director_output_tokens", "director_tokens", "combined_tokens", "duration_s", "action_count"):
        metrics["mean_" + key] = mean(x[key] for x in details.values())
        if "tokens" in key:
            metrics["total_" + key] = sum(x[key] for x in details.values())
    for key in ("request_budget_blocks", "artifacts_with_active_credit", "artifacts_with_disabled_credit", "requests_with_budget_quotes", "director_successes_without_usage"):
        metrics[key] = sum(x[key] for x in details.values())
    metrics["tasks_with_request_budget_blocks"] = sum(x["request_budget_blocks"] > 0 for x in details.values())
    return {"run": str(root), "metrics": metrics, "tasks": details, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = read_run(args.baseline), read_run(args.candidate)
    if before["tasks"].keys() != after["tasks"].keys():
        raise ValueError("Baseline and candidate task IDs differ")
    transitions = Counter()
    pairs = []
    for task, old in before["tasks"].items():
        new = after["tasks"][task]
        transitions[f"{int(old['passed'])}->{int(new['passed'])}"] += 1
        pair = {"task_id": task}
        for key in ("passed", "reward", "worker_tokens", "director_tokens", "combined_tokens", "request_budget_blocks", "action_count"):
            pair["baseline_" + key] = old[key]
            pair["candidate_" + key] = new[key]
            pair["delta_" + key] = new[key] - old[key]
        pairs.append(pair)
    discordant = transitions["0->1"] + transitions["1->0"]
    smaller = min(transitions["0->1"], transitions["1->0"])
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant)
    result = {
        "baseline": before, "candidate": after, "paired_transitions": dict(transitions),
        "mcnemar_exact_two_sided_p": p_value,
        "delta": {key: after["metrics"][key] - value for key, value in before["metrics"].items()},
        "previously_blocked_tasks": [p for p in pairs if p["baseline_request_budget_blocks"]],
        "token_accounting": "Worker execution reports are deduplicated and checked against cumulative Canvas feedback. Director request completion_usage is deduplicated by event_id, including separate thinking/action calls. Counts are recorded token usage, not monetary billing or estimates. Original records.jsonl token_cost is retained separately.",
        "interpretation_limit": "One historical run versus one new run on the same tasks; sampling and service conditions can vary. Source and configuration differences are recorded in the experiment manifest.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "budget_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "budget_paired_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({"baseline": before["metrics"], "candidate": after["metrics"], "delta": result["delta"], "paired_transitions": dict(transitions), "issues": before["issues"] + after["issues"]}, indent=2))
    if before["issues"] or after["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
