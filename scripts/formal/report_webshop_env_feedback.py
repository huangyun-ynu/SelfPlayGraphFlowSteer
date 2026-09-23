"""Compare complete WebShop runs and audit feedback delivery / Qwen thinking."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


def records(root: Path) -> dict:
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line]
    result = {row["task_id"]: row for row in rows}
    if len(rows) != 128 or len(result) != 128:
        raise ValueError(f"Expected 128 unique completed tasks: {root}, got {len(rows)}")
    return result


def audit(root: Path) -> dict:
    thinking = Counter()
    feedback = Counter()
    task_feedback = Counter()
    examples = []
    seen = set()
    for path in sorted((root / "trajectories").glob("*.json")):
        trajectory = json.loads(path.read_text())
        task = trajectory["task"]["task_id"]
        for turn in trajectory.get("director_run", {}).get("turns", []):
            key = (
                f"{turn.get('turn_kind')}:requested={turn.get('thinking_requested')}"
                f":effective={turn.get('thinking_effective')}"
            )
            thinking[key] += 1
        for event in trajectory.get("events", []):
            execution = event.get("payload", {}).get("execution") or {}
            for agent, artifact in execution.get("artifacts", {}).items():
                for entry in artifact.get("react_trace", []):
                    action = entry.get("action") or {}
                    call_id = action.get("call_id") or (
                        artifact.get("revision"), entry.get("round_index"), entry.get("phase")
                    )
                    key = (task, agent, call_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    observation = (entry.get("observation") or {}).get("output")
                    if not isinstance(observation, dict):
                        continue
                    note = observation.get("env_feedback", "")
                    if not note:
                        continue
                    kind = "repeat_query" if "same search query" in note else "return_to_search"
                    feedback[kind] += 1
                    task_feedback[task] += 1
                    if len(examples) < 6:
                        examples.append({"task_id": task, "action": action, "feedback": note})
    return {
        "director_turns": dict(thinking), "feedback_counts": dict(feedback),
        "tasks_with_feedback": len(task_feedback), "task_feedback": dict(task_feedback),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = records(args.baseline), records(args.candidate)
    if before.keys() != after.keys():
        raise ValueError("Baseline/candidate task IDs differ")
    transitions = Counter()
    pairs = []
    for task in before:
        old, new = before[task], after[task]
        transitions[f"{int(bool(old['passed']))}->{int(bool(new['passed']))}"] += 1
        pairs.append({
            "task_id": task, "baseline_passed": old["passed"], "candidate_passed": new["passed"],
            "baseline_reward": old["score"], "candidate_reward": new["score"],
            "reward_delta": new["score"] - old["score"],
        })
    def metrics(rows):
        values = list(rows.values())
        successes = sum(bool(row["passed"]) for row in values)
        return {
            "examples": len(values), "strict_successes": successes,
            "strict_success_rate": successes / len(values),
            "mean_reward": mean(row["score"] for row in values),
            "mean_token_cost": mean(row["token_cost"] for row in values),
            "mean_duration_s": mean(row["duration_s"] for row in values),
        }
    b, c = metrics(before), metrics(after)
    discordant = transitions["0->1"] + transitions["1->0"]
    smaller = min(transitions["0->1"], transitions["1->0"])
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant)
    summary = {
        "baseline_run": str(args.baseline), "candidate_run": str(args.candidate),
        "baseline": b, "candidate": c,
        "strict_success_delta_pp": 100 * (c["strict_success_rate"] - b["strict_success_rate"]),
        "mean_reward_delta": c["mean_reward"] - b["mean_reward"],
        "paired_transitions": dict(transitions), "mcnemar_exact_two_sided_p": p_value,
        "baseline_audit": audit(args.baseline), "candidate_audit": audit(args.candidate),
        "interpretation_limit": "Single seeded run against a historical reference, not a restored same-code control; model/source/service drift and sampling variation may contribute.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "paired_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_audit")}, indent=2))


if __name__ == "__main__":
    main()
