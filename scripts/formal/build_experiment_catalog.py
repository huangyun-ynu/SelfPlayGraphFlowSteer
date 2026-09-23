#!/usr/bin/env python3
"""Export aggregate experiment metrics, never prompts, answers, or credentials."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


METRICS = ("examples", "mean_score", "pass_rate", "mean_token_cost", "mean_duration_s")
SETTINGS = (
    "evaluation_only", "parameter_updates", "director_model", "director_thinking",
    "deepseek_thinking", "minimax_thinking", "workers", "datasets", "examples_planned",
    "worker_logical_routes", "dataset_token_limits", "swe_enabled",
)
ROOTS = ("formal-eval", "experiments", "sota-20260916", "sota-20260917")


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root: Path) -> dict:
    directories = set()
    for group in ROOTS:
        # Evaluation runs live at most two levels below these roots. Do not
        # recursively inspect private state, model files, or source snapshots.
        for pattern in ("*/run_manifest.json", "*/*/run_manifest.json", "*/summary.json"):
            for path in (root / "state" / group).glob(pattern):
                if path.parent.name != "webshop-design-audit-20260923":
                    directories.add(path.parent)
    runs = []
    for directory in sorted(directories):
        manifest = read(directory / "run_manifest.json")
        aggregate = read(directory / "aggregate_summary.json")
        summary = read(directory / "summary.json")
        if not manifest and summary.get("schema") != "static_skill_benchmark_summary_v1":
            continue
        counts = {k: v for k, v in aggregate.get("run", {}).items()
                  if k in ("planned", "completed", "failed", "workers", "parameter_updates")}
        metrics = {dataset: {k: values[k] for k in METRICS if k in values}
                   for dataset, values in aggregate.items()
                   if dataset not in ("run", "overall", "cleanup") and isinstance(values, dict)}
        if summary.get("schema") == "static_skill_benchmark_summary_v1":
            for dataset, values in summary.get("datasets", {}).items():
                if values.get("recorded", 0):
                    metrics[dataset] = {k: values.get(k) for k in
                                        ("recorded", "completed", "failed", "mean_score", "pass_rate")}
        if aggregate:
            status = "complete" if counts.get("completed") == counts.get("planned") and not counts.get("failed") else "incomplete"
        else:
            status = "summary_only" if metrics else "manifest_only"
        if aggregate.get("cleanup"):
            status = "complete_with_exclusions"
        verifier_counts: Counter = Counter()
        submission_counts: Counter = Counter()
        task_keys = []
        for path in sorted((directory / "trajectories").glob("*.json")):
            trace = read(path)
            verification = trace.get("verification") or {}
            verifier_counts[verification.get("verifier") or "unscored"] += 1
            submission_counts[(trace.get("answer_submission") or {}).get("method") or "none"] += 1
            task = trace.get("task") or {}
            task_keys.append(str(task.get("task_id", task.get("id", path.stem))))
        run = {
            "run": str(directory.relative_to(root)), "status": status,
            "counts": counts, "metrics": metrics,
            "settings": {k: manifest[k] for k in SETTINGS if k in manifest},
            "director_skill_enabled": bool(manifest.get("director_skill_root")) if manifest else None,
            "verifier_counts": dict(sorted(verifier_counts.items())),
            "submission_method_counts": dict(sorted(submission_counts.items())),
            "observed_task_count": len(task_keys),
            "observed_task_set_sha256": hashlib.sha256("\n".join(sorted(task_keys)).encode()).hexdigest() if task_keys else None,
            "evidence_sha256": {p.name: digest(p) for p in
                                [directory / name for name in ("aggregate_summary.json", "summary.json", "run_manifest.json")]
                                if p.exists()},
        }
        if aggregate.get("cleanup"):
            run["exclusions"] = aggregate["cleanup"]
        runs.append(run)
    return {
        "schema": "spgfs-public-experiment-catalog-v1", "as_of": "2026-09-23",
        "scope": "Local evaluation runs in the four documented state roots; synthetic unit tests excluded.",
        "units": {"mean_score": "0..1 (dataset/verifier dependent)", "pass_rate": "0..1 (verifier dependent)",
                  "mean_token_cost": "reported Worker tokens per completed example; excludes Director"},
        "warning": "Completion is runner completion, not task success. Manifest-only runs have no verified aggregate result. Historical source is not implied by a run name.",
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(result['runs'])} runs to {args.output}")


if __name__ == "__main__":
    main()
