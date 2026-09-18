"""Summarize paired WebShop prompt experiments from durable local records."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics


ARMS = ("original", "removed", "best_so_far")


def failure_kinds(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if value.get("backend_failure") and value.get("kind"):
            found.add(str(value["kind"]))
        for child in value.values():
            found.update(failure_kinds(child))
    elif isinstance(value, list):
        for child in value:
            found.update(failure_kinds(child))
    return found


def read_arm(root: Path) -> dict:
    rows = {}
    for path in sorted((root / "samples").glob("*.json")):
        sample = json.loads(path.read_text())
        record = sample["record"]
        trajectory = json.loads((root / sample["trajectory_path"]).read_text())
        verification = trajectory.get("verification")
        detail = (verification or {}).get("detail") or {}
        if isinstance(detail, str):
            detail = json.loads(detail)
        metadata = trajectory["task"]["metadata"]
        progress = metadata.get("webshop_output_progress") or {}
        failures = failure_kinds(trajectory.get("events", []))
        rows[sample["example_id"]] = {
            "score": record["score"],
            "passed": record["passed"],
            "verified": verification is not None,
            "purchased": bool(detail.get("purchased")),
            "termination": detail.get("termination_reason", "no_verification"),
            "backend_failure": (
                ",".join(sorted(failures)) if failures
                else "unclassified_no_verification" if verification is None else None
            ),
            "actions_used": (progress.get("action_budget") or {}).get("total_used"),
            "queries": progress.get("queries_tried", []),
            "duration_s": record["duration_s"],
            "token_cost": record["token_cost"],
            "skills_used": trajectory.get("skills_used", []),
            "trajectory": str(root / sample["trajectory_path"]),
        }
    return rows


def summarize(rows: dict) -> dict:
    values = list(rows.values())
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "score_100": statistics.mean(r["score"] for r in values) * 100,
        "successes": sum(r["passed"] for r in values),
        "success_rate_pct": statistics.mean(r["passed"] for r in values) * 100,
        "purchases": sum(r["purchased"] for r in values),
        "verified": sum(r["verified"] for r in values),
        "terminations": dict(Counter(r["termination"] for r in values)),
        "backend_failures": dict(Counter(r["backend_failure"] for r in values if r["backend_failure"])),
        "mean_duration_s": statistics.mean(r["duration_s"] for r in values),
        "mean_token_cost": statistics.mean(r["token_cost"] for r in values),
    }


def paired(a: dict, b: dict, keys: list[str]) -> dict:
    if not keys:
        return {"count": 0}
    differences = [b[k]["score"] - a[k]["score"] for k in keys]
    rng = random.Random(20260916)
    boot = sorted(
        statistics.mean(rng.choices(differences, k=len(keys))) * 100
        for _ in range(10000)
    )
    return {
        "count": len(keys),
        "score_delta": statistics.mean(differences) * 100,
        "score_delta_bootstrap_95pct": [boot[250], boot[9749]],
        "improved": sum(d > 1e-9 for d in differences),
        "worse": sum(d < -1e-9 for d in differences),
        "same": sum(abs(d) <= 1e-9 for d in differences),
        "successes_gained": sum(b[k]["passed"] and not a[k]["passed"] for k in keys),
        "successes_lost": sum(a[k]["passed"] and not b[k]["passed"] for k in keys),
        "purchases_gained": sum(b[k]["purchased"] and not a[k]["purchased"] for k in keys),
        "purchases_lost": sum(a[k]["purchased"] and not b[k]["purchased"] for k in keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    selection = json.loads((args.root / "selection.json").read_text())
    rows = {arm: read_arm(args.root / arm) for arm in ARMS}
    expected = set(selection["tasks"])
    for arm in ARMS:
        if set(rows[arm]) != expected:
            raise ValueError(f"Incomplete arm {arm}: {len(rows[arm])}/{len(expected)}")
    common = sorted(k for k in expected if all(rows[a][k]["verified"] for a in ARMS))
    executed = [
        k for k in common
        if all(rows[a][k]["termination"] != "missing_output_agent" for a in ARMS)
    ]
    clean = sorted(k for k in common if all(not rows[a][k]["backend_failure"] for a in ARMS))
    old = read_arm(Path(selection["source"]))
    report = {
        "selection": selection,
        "historical_selected": summarize({k: old[k] for k in expected}),
        "summaries": {arm: summarize(rows[arm]) for arm in ARMS},
        "common_verified_ids": common,
        "common_executed_ids": executed,
        "common_executed_summaries": {
            arm: summarize({k: rows[arm][k] for k in executed}) for arm in ARMS
        },
        "common_no_backend_failure_ids": clean,
        "common_no_backend_failure_summaries": {
            arm: summarize({k: rows[arm][k] for k in clean}) for arm in ARMS
        },
        "paired_vs_original": {
            arm: paired(rows["original"], rows[arm], sorted(expected)) for arm in ARMS[1:]
        },
        "paired_clean_vs_original": {
            arm: paired(rows["original"], rows[arm], clean) for arm in ARMS[1:]
        },
        "paired_executed_vs_original": {
            arm: paired(rows["original"], rows[arm], executed) for arm in ARMS[1:]
        },
        "tasks": {k: {arm: rows[arm][k] for arm in ARMS} for k in sorted(expected)},
        "limitations": [
            "Prior-failure subset, not the complete test set.",
            "One seed, sequential arms, backend availability may confound results.",
            "Reruns start fresh; they do not resume historical final states.",
            "Prompt policy changes at three injection sites; tool protocol is preserved.",
            "Hidden scores are used for reporting only, never candidate selection.",
        ],
    }
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# WebShop Closure Prompt Ablation", "", "| Arm | N | Score /100 | Successes | Purchases | Verified |", "|---|---:|---:|---:|---:|---:|"]
    for arm, s in report["summaries"].items():
        lines.append(f"| {arm} | {s['count']} | {s['score_100']:.2f} | {s['successes']} | {s['purchases']} | {s['verified']} |")
    lines.extend(["", "## Common Executed Tasks", "", "Transient recovered requests may remain in this subset.", "", "| Arm | N | Score /100 | Successes | Purchases |", "|---|---:|---:|---:|---:|"])
    for arm, s in report["common_executed_summaries"].items():
        if s["count"]:
            lines.append(f"| {arm} | {s['count']} | {s['score_100']:.2f} | {s['successes']} | {s['purchases']} |")
    lines.extend(["", f"Common tasks without any backend failure event: {len(clean)}.", "", "## Limitations", ""])
    lines.extend("- " + item for item in report["limitations"])
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("tasks", "selection")}, indent=2))


if __name__ == "__main__":
    main()
