"""Read-only, cross-version WebShop design audit; never print reasoning transcripts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def extract(path: Path, label: str) -> dict:
    t = json.loads(path.read_text())
    verification = t.get("verification") or {}
    try:
        detail = json.loads(verification.get("detail", "{}"))
    except (TypeError, ValueError):
        detail = {}
    row = {
        "run": label, "task": t["task"]["task_id"], "path": str(path),
        "passed": bool(verification.get("passed")), "score": verification.get("score", 0),
        "termination": detail.get("termination_reason"), "purchased": detail.get("purchased"),
        "official_steps": detail.get("steps"), "flags": [], "actions": [], "artifacts": [],
        "director_rejections": [], "director_events": [], "wrong_navigation": [],
        "forgotten_inspections": [], "large_pages": [],
    }
    seen, flags, previous, visited = set(), set(), {}, set()
    for event in t.get("events", []):
        payload = event.get("payload", {})
        rejection = payload.get("rejection_code")
        if rejection:
            row["director_rejections"].append(rejection)
        raw_action = payload.get("raw_action", "")
        row["director_events"].append({
            "sequence": event.get("sequence"), "kind": event.get("kind"),
            "action": raw_action[:1200] if isinstance(raw_action, str) else raw_action,
            "accepted": payload.get("accepted"), "rejection": rejection,
            "feedback": str(payload.get("feedback", ""))[:700],
        })
        for agent, artifact in (payload.get("execution") or {}).get("artifacts", {}).items():
            key = artifact["artifact_id"]
            if key in seen:
                continue
            seen.add(key)
            progress = artifact.get("webshop_progress") or {}
            diagnostics = artifact.get("protocol_diagnostics", [])
            concise = {
                "id": key, "agent": agent, "revision": artifact.get("revision"),
                "state": progress.get("state"), "access": progress.get("environment_access"),
                "commit_ready": progress.get("commit_ready"),
                "policy_failure": progress.get("policy_failure"),
                "accounting": progress.get("execution_accounting"),
                "action_budget": progress.get("action_budget"),
                "prompt_projection": progress.get("prompt_projection"),
                "diagnostics": [{k: d[k] for k in ("stage", "rejection_reason", "no_request_dispatched", "request_token_budget", "finish_reason", "requested_max_tokens") if k in d} for d in diagnostics],
            }
            row["artifacts"].append(concise)
            if progress.get("commit_ready"):
                flags.add("purchase_staged")
            if progress.get("policy_failure"):
                flags.add("policy_fuse")
            if any(d.get("no_request_dispatched") for d in diagnostics):
                flags.add("request_admission_block")
            if artifact.get("answer") == "WORKER_BACKEND_FAILURE":
                flags.add("worker_backend_failure")
            if artifact.get("answer") == "WORKER_PROTOCOL_FAILURE":
                flags.add("worker_protocol_failure")
            if any(e.get("event") == "backend_request_failure" for e in artifact.get("backend_request_events", [])):
                flags.add("backend_request_failure_event")
            for turn in artifact.get("react_trace", []):
                action, observation = turn.get("action") or {}, turn.get("observation") or {}
                args = action.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                        flags.add("malformed_action_arguments")
                if not isinstance(args, dict):
                    args = {}
                output = observation.get("output")
                output = output if isinstance(output, dict) else {}
                product = output.get("product") or {}
                asin = str(product.get("asin", "")).lower()
                targets = output.get("valid_subactions", [])
                before = previous.get(agent, {})
                item = {
                    "artifact": key, "agent": agent, "round": turn.get("round_index"),
                    "name": action.get("name"), "query": args.get("query"), "target": args.get("target_id"),
                    "status": observation.get("status"), "error": observation.get("error"),
                    "page": output.get("page_type"), "asin": asin,
                    "title": str(product.get("title", ""))[:160], "selected": output.get("selected_options"),
                    "budget": turn.get("remaining_budget"), "env_remaining": output.get("remaining_steps"),
                    "staged": output.get("commit_pending"), "page_chars": len(str(output.get("page_text", ""))),
                    "action_count": len(targets), "observation_truncated": output.get("observation_truncated"),
                    "purchase_evidence_status": output.get("purchase_evidence_status"),
                }
                row["actions"].append(item)
                if item["status"] == "error":
                    flags.add("action_error")
                if item["staged"]:
                    flags.add("purchase_staged")
                if item["page_chars"] > 8000 or item["observation_truncated"]:
                    row["large_pages"].append(len(row["actions"]) - 1)
                if output.get("page_type") == "product":
                    bad = [a for a in targets if str(a.get("target_id", "")).startswith("previous_page:") and a.get("navigation_effect") == "return_to_current_product_page"]
                    if bad:
                        flags.add("wrong_product_previous_semantics")
                        row["wrong_navigation"].append({"action_index": len(row["actions"]) - 1, "asin": asin, "targets": [a["target_id"] for a in bad]})
                if (
                    str(args.get("target_id", "")).startswith("previous_page:")
                    and before.get("page_type") == "product"
                    and output.get("page_type") == "search_results"
                    and any(a.get("navigation_effect") == "return_to_current_product_page" for a in before.get("valid_subactions", []) if str(a.get("target_id", "")).startswith("previous_page:"))
                ):
                    flags.add("clicked_misdescribed_previous")
                if output.get("page_type") == "search_results":
                    for candidate in targets:
                        target = str(candidate.get("target_id", ""))
                        if not target.startswith("open_product:"):
                            continue
                        candidate_asin = target.split(":", 2)[-1].lower()
                        if candidate_asin in visited and candidate.get("inspection_status") == "not_inspected":
                            flags.add("visited_marked_uninspected")
                            row["forgotten_inspections"].append({"action_index": len(row["actions"])-1, "asin": candidate_asin, "asin_field": candidate.get("asin")})
                if output.get("page_type") in {"product", "product_section"} and asin:
                    visited.add(asin)
                if output.get("page_type"):
                    previous[agent] = output
    if not row["artifacts"]:
        flags.add("no_worker")
    if row["large_pages"]:
        flags.add("page_truncation_exposure")
    if "purchase_staged" in flags and not row["purchased"]:
        flags.add("staged_but_not_purchased")
    row["flags"] = sorted(flags)
    return row


def main() -> None:
    versions = json.loads(Path("state/experiments/webshop-version-comparison-20260923.json").read_text())["versions"]
    runs = [(v["comparison_label"], Path(v["run"])) for v in versions]
    runs.append(("V7", Path("state/formal-eval/webshop-laser-checklist-c24-20260923-182224")))
    for label, subdir in [("closure_original", "original"), ("closure_removed", "removed"), ("closure_best", "best_so_far")]:
        runs.append((label, Path("state/sota-20260916/webshop-closure-ablation-deepseek-v1") / subdir))
    runs += [("options_fixed", Path("state/sota-20260916/webshop-option-fixes-deepseek-no-skill-v1")), ("options_retry", Path("state/sota-20260916/webshop-option-fixes-deepseek-no-skill-retry-v1"))]
    out = Path("state/experiments/webshop-design-audit-20260923")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (out / "trajectory_index.jsonl").open("w") as handle:
        for label, root in runs:
            rows = [extract(path, label) for path in sorted((root / "trajectories").glob("*.json"))]
            records = {r["task_id"]: r for line in (root / "records.jsonl").read_text().splitlines() if line for r in [json.loads(line)]}
            for row in rows:
                record = records[row["task"]]
                row.update(passed=record["passed"], score=record["score"], token_cost=record["token_cost"], duration_s=record["duration_s"])
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary[label] = {
                "root": str(root), "tasks": len(rows), "passed": sum(r["passed"] for r in rows),
                "terminations": dict(Counter(str(r["termination"]) for r in rows)),
                "flags_all": dict(Counter(f for r in rows for f in r["flags"])),
                "flags_failed": dict(Counter(f for r in rows if not r["passed"] for f in r["flags"])),
                "director_rejections": dict(Counter(c for r in rows for c in r["director_rejections"])),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
