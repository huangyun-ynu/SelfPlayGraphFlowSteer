"""Audit a completed WebShop guidance evaluation without printing model reasoning."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(root: Path, policy: str) -> dict:
    manifest = json.loads((root / "run_manifest.json").read_text())
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line]
    tasks = {row["task_id"] for row in records}
    if len(records) != 128 or len(tasks) != 128:
        raise ValueError("Expected 128 unique completed tasks")
    artifacts_seen, requests_seen, trajectory_tasks = set(), set(), set()
    guidance, director, routes, requests = Counter(), Counter(), Counter(), Counter()
    applied_tasks, worker_tasks, mutable_owner_tasks = set(), set(), set()
    issues = []
    if manifest.get("director_thinking") is not True:
        issues.append("Director thinking is not enabled in run manifest")
    if manifest.get("deepseek_thinking") is not True:
        issues.append("DeepSeek thinking is not enabled in run manifest")
    if manifest.get("worker_logical_routes") != ["deepseek"]:
        issues.append("Unexpected Worker routes in run manifest")
    for path in sorted((root / "trajectories").glob("*.json")):
        trajectory = json.loads(path.read_text())
        task = trajectory["task"]["task_id"]
        if task in trajectory_tasks:
            issues.append(f"Duplicate trajectory for {task}")
        trajectory_tasks.add(task)
        for turn in trajectory.get("director_run", {}).get("turns", []):
            kind = turn.get("turn_kind")
            requested, effective = turn.get("thinking_requested"), turn.get("thinking_effective")
            director[f"{kind}:requested={requested}:effective={effective}"] += 1
            if kind == "graph_action" and (requested is not True or effective is not True):
                issues.append(f"Director thinking mismatch: {task}")
        for event in trajectory.get("events", []):
            execution = event.get("payload", {}).get("execution") or {}
            for artifact in execution.get("artifacts", {}).values():
                key = (task, artifact["artifact_id"])
                if key in artifacts_seen:
                    continue
                artifacts_seen.add(key)
                worker_tasks.add(task)
                routes[str(artifact.get("model_route"))] += 1
                progress = artifact.get("webshop_progress") or {}
                if progress:
                    value = progress.get("worker_guidance") or {}
                    expected_applied = progress.get("environment_access") == "mutable_owner"
                    if expected_applied:
                        mutable_owner_tasks.add(task)
                    if value.get("policy") != policy or value.get("applied") is not expected_applied:
                        issues.append(f"Guidance mismatch: {task}/{artifact['artifact_id']}")
                    guidance["applied" if value.get("applied") else "not_applied"] += 1
                    if value.get("applied"):
                        applied_tasks.add(task)
                for request in artifact.get("backend_request_events", []):
                    request_id = request.get("event_id")
                    if not request_id or request_id in requests_seen:
                        continue
                    requests_seen.add(request_id)
                    requests[str(request.get("event"))] += 1
                    usage = request.get("provider_usage") or {}
                    reasoning_tokens = usage.get("reasoning_tokens")
                    if isinstance(reasoning_tokens, (int, float)):
                        requests["provider_reports_reasoning_tokens"] += 1
                        if reasoning_tokens > 0:
                            requests["positive_reasoning_tokens"] += 1
                    else:
                        requests["reasoning_tokens_unreported"] += 1
    if trajectory_tasks != tasks:
        issues.append("Trajectory task IDs do not match completed records")
    if applied_tasks != mutable_owner_tasks:
        issues.append("Guidance application does not match tasks with mutable-owner execution")
    if not mutable_owner_tasks:
        issues.append("No mutable-owner executions were found")
    if set(routes) != {"deepseek"}:
        issues.append(f"Unexpected Worker artifact routes: {dict(routes)}")
    return {
        "run": str(root), "policy": policy, "tasks": len(tasks),
        "tasks_with_guidance": len(applied_tasks), "unique_artifacts": len(artifacts_seen),
        "tasks_with_worker_execution": len(worker_tasks),
        "tasks_with_mutable_owner_execution": len(mutable_owner_tasks),
        "tasks_without_worker_execution": sorted(tasks - worker_tasks),
        "guidance_artifacts": dict(guidance), "worker_routes": dict(routes),
        "director_turns": dict(director), "worker_request_events": dict(requests),
        "run_manifest_thinking": {
            "director": manifest.get("director_thinking"),
            "deepseek": manifest.get("deepseek_thinking"),
        },
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", default="laser_checklist_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
