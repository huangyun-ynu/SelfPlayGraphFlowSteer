#!/usr/bin/env python3
"""Read-only, pre-service validation of an explicitly declared WebShop run.

No environment client or model backend is constructed. Records contain file
hashes and public configuration fields, never environment variable values.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(*, source: Path, commit: str, config: Path, dataset: Path, output: Path,
          profile: str, goals: Path, declared_patch_sha256: str | None = None) -> dict:
    source = source.resolve()
    if output.exists():
        raise ValueError("run output already exists: choose a new run/attempt")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])
    if git("rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("source commit differs from declared commit")
    diff = git("diff", "HEAD", "--binary")
    if diff and hashlib.sha256(diff).hexdigest() != declared_patch_sha256:
        raise ValueError("unrecorded development patch in source tree")
    if git("ls-files", "--others", "--exclude-standard", "src").strip():
        raise ValueError("untracked source files must be committed before evaluation")
    # Deliberately check the caller's actual import resolution. Do not silently
    # repair a wrong PYTHONPATH, which would conceal a misconfigured launch.
    paths = {}
    for name in ("application", "runtime", "canvas", "director", "pats", "webshop", "execution_contract"):
        module = importlib.import_module("selfplay_graph_flowsteer." + name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source / "src"):
            raise ValueError(f"mixed source tree: {name} resolved to {path}")
        paths[name] = str(path)
    from selfplay_graph_flowsteer.application import load_adaptive_config
    cfg = load_adaptive_config(config)
    if cfg.director_prompt_variant != "v2.2" or cfg.pats.director_prompt_variant != "v2.2":
        raise ValueError("Director and PATS must both resolve to v2.2")
    if cfg.canvas.submission_protocol != "legacy":
        raise ValueError("this comparison uses the v2.2 single-owner submission protocol")
    if cfg.webshop.compatibility_profile != profile:
        raise ValueError("WebShop profile differs from declared experiment")
    if cfg.runtime_pool()["deepseek"].max_concurrency != 40:
        raise ValueError("DeepSeek API concurrency must be 40")
    if not cfg.runtime_pool()["deepseek"].enable_thinking:
        raise ValueError("DeepSeek thinking configuration changed")
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    ids = [r.get("id", r.get("task_id")) for r in records]
    if not records or len(ids) != len(set(ids)) or None in ids:
        raise ValueError("dataset must contain unique explicit task IDs")
    from webshop_dataset_index import public_goal_order, validate_records
    mismatches = validate_records(records, *public_goal_order(goals))
    if mismatches:
        raise ValueError(f"{len(mismatches)}/{len(records)} task texts disagree with environment goal indices; first: {mismatches[0]['id']}")
    files = {str(path.relative_to(source)): digest(path)
             for path in sorted((source / "src/selfplay_graph_flowsteer").rglob("*.py"))}
    formal = source / "configs/formal_training.toml"
    def flat(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = prefix + "." + key if prefix else key
            if isinstance(item, dict):
                result.update(flat(item, name))
            else:
                result[name] = item
        return result
    base_values = flat(tomllib.loads(formal.read_text()))
    run_values = flat(tomllib.loads(config.read_text()))
    # Record override names and value hashes, so credentials supplied literally
    # in a local TOML are never copied into the public run record.
    overrides = {key: {"formal_value_sha256": hashlib.sha256(json.dumps(base_values.get(key), sort_keys=True).encode()).hexdigest(),
                       "run_value_sha256": hashlib.sha256(json.dumps(run_values.get(key), sort_keys=True).encode()).hexdigest()}
                 for key in sorted(base_values.keys() | run_values.keys()) if base_values.get(key) != run_values.get(key)}
    return {"source_root": str(source), "code_commit": commit,
            "formal_config_source": str(formal), "formal_config_sha256": digest(formal),
            "config_overrides": overrides,
            "validation_script_sha256": digest(Path(__file__)),
            "dataset_validation_script_sha256": digest(Path(__file__).with_name("webshop_dataset_index.py")),
            "development_patch_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
            "loaded_module_paths": paths, "source_files_sha256": files,
            "config": str(config.resolve()), "config_sha256": digest(config),
            "dataset": str(dataset.resolve()), "dataset_sha256": digest(dataset),
            "goals_sha256": digest(goals), "public_goal_mapping_mismatches": 0,
            "task_count": len(records), "director_prompt_variant": cfg.director_prompt_variant,
            "pats_prompt_variant": cfg.pats.director_prompt_variant,
            "webshop_profile": profile, "deepseek_max_concurrency": 40}


def main():
    parser = argparse.ArgumentParser()
    for option in ("source", "commit", "config", "dataset", "output", "profile", "record", "goals"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--declared-patch-sha256")
    args = parser.parse_args()
    result = check(source=Path(args.source), commit=args.commit, config=Path(args.config),
                   dataset=Path(args.dataset), output=Path(args.output), profile=args.profile,
                   goals=Path(args.goals),
                   declared_patch_sha256=args.declared_patch_sha256)
    path = Path(args.record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Preflight verified: v2.2, {result['task_count']} tasks, {args.profile}, {args.commit[:7]}")


if __name__ == "__main__":
    main()
