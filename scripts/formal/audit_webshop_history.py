"""Audit SkillFlow history activation and continuity from recorded Worker artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def audit(run: Path) -> dict:
    counters, details, issues = Counter(), [], []
    for path in sorted((run / "trajectories").glob("*.json")):
        trajectory = json.loads(path.read_text())
        task = trajectory["task"]["task_id"]
        seen, owner_counts, owner_chars, products = set(), {}, {}, set()
        applied, gaps, task_issues = False, [], []
        for event in trajectory.get("events", []):
            execution = event.get("payload", {}).get("execution") or {}
            for artifact in (execution.get("artifacts") or {}).values():
                aid, agent = artifact["artifact_id"], artifact["agent_id"]
                if aid in seen:
                    continue
                seen.add(aid)
                progress = artifact.get("webshop_progress") or {}
                memory = progress.get("worker_memory") or {}
                owner = progress.get("environment_access") == "mutable_owner"
                counters["artifacts"] += 1
                if memory.get("policy") != "skillflow_history_v1" or memory.get("applied") is not owner:
                    task_issues.append({"artifact": aid, "kind": "history_policy_mismatch"})
                if not owner:
                    counters["stateless_artifacts"] += 1
                    if memory.get("history_entries", 0) != 0:
                        task_issues.append({"artifact": aid, "kind": "stateless_history_leak"})
                    continue
                applied = True
                counters["owner_artifacts"] += 1
                counters["revision_owner_artifacts"] += bool(artifact.get("revision"))
                turns = []
                for turn in artifact.get("react_trace", []):
                    observation = turn.get("observation") or {}
                    error = observation.get("error") or {}
                    if isinstance(error, dict) and error.get("code") == "stateful_action_deferred":
                        counters["deferred_actions_excluded"] += 1
                        continue
                    turns.append(turn)
                    output = observation.get("output") or {}
                    if observation.get("status") == "ok" and isinstance(output, dict):
                        asin = str((output.get("product") or {}).get("asin", "")).strip().lower()
                        if asin:
                            products.add(asin)
                count, chars = memory.get("history_entries", 0), memory.get("history_observation_chars", 0)
                prior = owner_counts.get(agent, 0)
                if count < prior + len(turns):
                    task_issues.append({"artifact": aid, "kind": "history_entries_lost", "previous": prior, "new_turns": len(turns), "recorded": count})
                elif count > prior + len(turns):
                    # Execution reports can retain only the last artifact of a
                    # multi-pass execution. Such a gap is not evidence of truncation.
                    gaps.append({"artifact": aid, "unrepresented_prior_steps": count - prior - len(turns)})
                if chars < owner_chars.get(agent, 0):
                    task_issues.append({"artifact": aid, "kind": "history_chars_decreased"})
                owner_counts[agent], owner_chars[agent] = count, chars
                if progress.get("state_guidance_deliveries"):
                    task_issues.append({"artifact": aid, "kind": "old_state_guidance_recorded_as_delivered"})
                partition = progress.get("budget_partition") or {}
                if partition.get("request_admission_enabled") is not False:
                    task_issues.append({"artifact": aid, "kind": "request_admission_not_disabled"})
                action_budget = progress.get("action_budget") or {}
                if action_budget.get("total_limit") != 16 or action_budget.get("total_used", 0) > 16:
                    task_issues.append({"artifact": aid, "kind": "action_budget_changed"})
        details.append({
            "task_id": task, "history_applied": applied,
            "max_history_entries": max(owner_counts.values(), default=0),
            "max_history_observation_chars": max(owner_chars.values(), default=0),
            "observed_distinct_products": len(products), "artifact_gaps": gaps,
            "issues": task_issues,
        })
        issues.extend({"task_id": task, **issue} for issue in task_issues)
    active = [row for row in details if row["history_applied"]]
    chars = [row["max_history_observation_chars"] for row in active]
    counts = [row["max_history_entries"] for row in active]
    return {
        "run": str(run), "tasks": len(details), "tasks_with_history": len(active),
        "tasks_without_history": [row["task_id"] for row in details if not row["history_applied"]],
        "counts": dict(counters),
        "history_steps": {"maximum": max(counts, default=0), "median": median(counts) if counts else None, "mean": mean(counts) if counts else None},
        "history_observation_chars": {"maximum": max(chars, default=0), "median": median(chars) if chars else None, "mean": mean(chars) if chars else None, "tasks_over_6000": sum(x > 6000 for x in chars), "tasks_over_8000": sum(x > 8000 for x in chars)},
        "tasks_with_artifact_gaps": sum(bool(row["artifact_gaps"]) for row in details),
        "tasks_with_over_six_observed_products": sum(row["observed_distinct_products"] > 6 for row in details),
        "issues": issues, "task_details": details,
        "method": "Deduplicate artifacts per task. Compare cumulative owner history counters against visible selected attempts across revisions; exclude discarded batch Actions. Audit-only product_inspections remain bounded and are not used to measure model-facing history. Character counts are raw history observation text, not tokens or complete request size. Source snapshot and offline request tests establish prompt wiring; production artifacts record counters, not full backend prompts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"task_details", "method"}}, indent=2))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
