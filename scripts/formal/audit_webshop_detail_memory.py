"""Measure recorded WebShop detail evidence and verify its source text retention."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def evidence_functions(runtime: Path):
    source = runtime.read_text()
    node = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_webshop_section_evidence"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(runtime), "exec"), namespace)
    retained = namespace[node.name]
    limit = None
    if isinstance(node.body[-1].value, ast.Subscript):
        limit = node.body[-1].value.slice.upper.value
        node.body[-1].value = node.body[-1].value.value
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(runtime), "exec"), namespace)
    return retained, namespace[node.name], limit


def stats(values):
    return {
        "entries": len(values), "lt_1400": sum(x < 1400 for x in values),
        "eq_1400": sum(x == 1400 for x in values), "gt_1400": sum(x > 1400 for x in values),
        "ge_1400": sum(x >= 1400 for x in values),
        "maximum": max(values, default=None), "median": median(values) if values else None,
    }


def audit(run: Path, runtime: Path):
    retained, full, limit = evidence_functions(runtime)
    saved, sources, tasks = {}, {}, set()
    snapshot_lengths, successful_views = [], 0
    for path in sorted((run / "trajectories").glob("*.json")):
        trace = json.loads(path.read_text())
        task = trace["task"]["task_id"]
        tasks.add(task)
        seen = set()
        for event in trace.get("events", []):
            execution = event.get("payload", {}).get("execution") or {}
            for artifact in (execution.get("artifacts") or {}).values():
                aid = artifact["artifact_id"]
                if aid in seen:
                    continue
                seen.add(aid)
                agent = artifact["agent_id"]
                for record in (artifact.get("webshop_progress") or {}).get("product_inspections", []):
                    asin = str(record.get("asin", "")).strip().casefold()
                    for section, value in (record.get("section_evidence") or {}).items():
                        text = str(value)
                        saved[(task, agent, asin, section, text)] = aid
                        snapshot_lengths.append(len(text))
                for turn in artifact.get("react_trace", []):
                    args = (turn.get("action") or {}).get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    target = str(args.get("target_id", ""))
                    obs = turn.get("observation") or {}
                    payload = obs.get("output") or {}
                    if obs.get("status") != "ok" or not isinstance(payload, dict):
                        continue
                    if payload.get("page_type") not in {"product", "product_section"} or not target.startswith("view_"):
                        continue
                    asin = str((payload.get("product") or {}).get("asin", "")).strip().casefold()
                    section = target.split(":", 1)[0].removeprefix("view_")[:40]
                    page = payload.get("page_text", "")
                    text = full(page)
                    if asin and section and text:
                        sources[(task, agent, asin, section, text)] = retained(page)
                        successful_views += 1
    expected = defaultdict(list)
    for key, text in sources.items():
        expected[key[:4] + (text,)].append(len(key[4]))
    rows, issues = [], []
    for key, aid in saved.items():
        task, agent, asin, section, text = key
        matches = sorted(set(expected.get(key, [])))
        row = {
            "task_id": task, "agent_id": agent, "asin": asin, "section": section,
            "artifact_id": aid, "stored_chars": len(text), "source_chars": matches,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        rows.append(row)
        if not matches:
            issues.append(row)
    with_memory = {x[0] for x in saved}
    long_memory = {x[0] for x in saved if len(x[4]) >= 1400}
    shortened = [r for r in rows if r["source_chars"] and min(r["source_chars"]) > r["stored_chars"]]
    result = {
        "run": str(run), "runtime_snapshot": str(runtime), "section_char_limit": limit,
        "tasks": len(tasks), "successful_detail_views": successful_views,
        "stored": stats([len(k[4]) for k in saved]),
        "before_clipping": stats([len(k[4]) for k in sources]),
        "stored_snapshot_occurrences": stats(snapshot_lengths),
        "task_counts": {
            "with_detail_memory": len(with_memory), "any_ge_1400": len(long_memory),
            "all_lt_1400": len(with_memory - long_memory), "without_detail_memory": len(tasks - with_memory),
        },
        "shortened_saved_entries": len(shortened),
        "tasks_with_shortened_saved_entries": len({r["task_id"] for r in shortened}),
        "max_characters_discarded": max((min(r["source_chars"]) - r["stored_chars"] for r in shortened), default=0),
        "source_match_issues": issues, "stored_entries": rows,
        "method": "Characters, not tokens. Deduplicate (task, agent, ASIN, section, exact text). Ignore mirrored fields. Compare saved evidence with successful tool observations using each run's own runtime snapshot. Absent evidence is not zero length. Original whitespace/navigation cleanup is preserved.",
    }
    if limit is None and shortened:
        raise AssertionError("Uncapped memory was unexpectedly shortened")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for side, directory in [("baseline", args.baseline_audit), ("candidate", args.candidate_audit)]:
        manifest = json.loads((directory / "experiment_manifest.json").read_text())
        results[side] = audit(Path(manifest["output"]), directory / "snapshot/src/selfplay_graph_flowsteer/runtime.py")
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({s: {k: v for k, v in r.items() if k not in {"stored_entries", "method"}} for s, r in results.items()}, indent=2))
    if any(r["source_match_issues"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
