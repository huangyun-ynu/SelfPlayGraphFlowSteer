"""Audit legacy identity in recorded observations without reading hidden task targets."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path


def audit(root: Path) -> dict:
    tasks = []
    for path in sorted((root / "trajectories").glob("*.json")):
        trace = json.loads(path.read_text())
        task = trace["task"]["task_id"]
        seen_artifacts, memories, opened = set(), {}, {}
        counts = Counter()
        mismatches = []
        for event in trace.get("events", []):
            execution = event.get("payload", {}).get("execution") or {}
            for artifact in execution.get("artifacts", {}).values():
                if artifact["artifact_id"] in seen_artifacts:
                    continue
                seen_artifacts.add(artifact["artifact_id"])
                agent = artifact["agent_id"]
                memory = memories.setdefault(agent, OrderedDict())
                visits = opened.setdefault(agent, Counter())
                progress = artifact.get("webshop_progress") or {}
                ledger = progress.get("candidate_ledger", [])
                counts["ledger_records_with_search_appearances"] += sum(
                    int(record.get("appearance_count", 0)) > 0
                    for record in ledger if isinstance(record, dict)
                )
                for recent in progress.get("recent_actions", []):
                    action = recent.get("action") or {}
                    if action.get("kind") == "open_product":
                        counts["remembered_product_actions"] += 1
                        counts["remembered_product_actions_missing_asin"] += not bool(action.get("asin"))
                for turn in artifact.get("react_trace", []):
                    observation = turn.get("observation") or {}
                    payload = observation.get("output")
                    if observation.get("status") != "ok" or not isinstance(payload, dict):
                        continue
                    page = payload.get("page_type")
                    if page == "search_results":
                        counts["search_observations"] += 1
                        for action in payload.get("valid_subactions", []):
                            if action.get("kind") != "open_product":
                                continue
                            counts["search_product_actions"] += 1
                            counts["search_product_actions_without_asin_field"] += "asin" not in action
                            match = re.fullmatch(r"open_product:\d+:([A-Za-z0-9]{10})", str(action.get("target_id", "")))
                            if not match:
                                counts["unparseable_product_targets"] += 1
                                continue
                            asin = match.group(1).lower()
                            counts["product_actions_marked_inspected"] += action.get("inspection_status") == "inspected"
                            if asin in memory:
                                counts["visible_candidates_in_retained_visit_history"] += 1
                                if action.get("inspection_status") != "inspected":
                                    counts["retained_visit_marked_uninspected"] += 1
                                    mismatches.append({"artifact": artifact["artifact_id"], "target_id": action["target_id"], "actual_status": action.get("inspection_status")})
                    if page in {"product", "product_section"}:
                        product = payload.get("product") or {}
                        asin = str(product.get("asin", "")).strip().lower()
                        if not asin:
                            continue
                        memory.pop(asin, None)
                        memory[asin] = True
                        # Existing runtime keeps six product inspections. Do not
                        # count intentionally evicted visits as an identity bug.
                        while len(memory) > 6:
                            memory.popitem(last=False)
                        action = turn.get("action") or {}
                        arguments = action.get("arguments") or {}
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if action.get("name") == "webshop_click" and str(arguments.get("target_id", "")).startswith("open_product:"):
                            counts["product_opens"] += 1
                            counts["repeat_product_opens"] += visits[asin] > 0
                            visits[asin] += 1
        tasks.append({"task_id": task, "counts": dict(counts), "mismatches": mismatches})
    total = Counter()
    for task in tasks:
        total.update(task["counts"])
    return {
        "run": str(root), "tasks": len(tasks), "counts": dict(total),
        "tasks_with_retained_visit_mismatch": sum(bool(task["mismatches"]) for task in tasks),
        "tasks_with_repeated_product_opens": sum(task["counts"].get("repeat_product_opens", 0) > 0 for task in tasks),
        "task_details": tasks,
        "method": "Replay the order of successful observed product pages per agent, retaining the last six ASINs. Check recorded search inspection labels against those visible visits. Artifact snapshots are deduplicated. Ledger/action-memory counts are per saved artifact. Reopening a product may be legitimate and is not automatically an error.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = audit(args.baseline), audit(args.candidate)
    result = {"baseline": before, "candidate": after}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({side: {k: v for k, v in data.items() if k not in {"task_details", "method"}} for side, data in result.items()}, indent=2))


if __name__ == "__main__":
    main()
