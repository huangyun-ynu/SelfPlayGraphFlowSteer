#!/usr/bin/env python3
"""Provision public test environments from public task identities only.

Run with PYTHONPATH=src. No verifier, hidden test patch, model endpoint or cloud
resource is accessed. uv must already be installed. Each environment becomes
ready only after a real, non-empty public smoke test succeeds.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from selfplay_graph_flowsteer.swe_public_recipes import RECIPES, environment_key
from selfplay_graph_flowsteer.swe_public_tests import (
    environment_lock,
    environment_status,
    resolve_target,
    run_public_test,
)
from selfplay_graph_flowsteer.swebench import SWEWorkspaceLifecycle, _git_safe_environment


def prepare(row, args):
    repo, version, commit = (str(row[k]) for k in ("repo", "version", "base_commit"))
    recipe = RECIPES[repo, version]
    directory = args.env_root / environment_key(repo, version)
    with environment_lock(directory, 3600):
        status = environment_status(args.env_root, repo, version)
        if status["ready"] and not args.recheck:
            return {**status, "cached": True}
        (directory / "ready.json").unlink(missing_ok=True)
        with (directory / "provision.log").open("w") as log:
            def run(command):
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=900, env={**os.environ, "UV_DEFAULT_INDEX": args.index_url,
                                                "PIP_CONFIG_FILE": os.devnull,
                                                "PIP_EXTRA_INDEX_URL": "", "PIP_INDEX_URL": args.index_url})
            python = directory / "venv/bin/python"
            if python.exists():
                installed_version = subprocess.check_output(
                    [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True).strip()
                if installed_version != recipe.python:
                    shutil.rmtree(directory / "venv")
            if not python.exists():
                if recipe.python == "3.6":
                    if not args.micromamba:
                        raise ValueError("Python 3.6 recipes require --micromamba /path/to/micromamba")
                    run([args.micromamba, "create", "-y", "--override-channels", "-c", "conda-forge",
                         "--root-prefix", str(args.env_root / "mamba"), "-p", str(directory / "venv"),
                         "python=3.6", "pip"])
                else:
                    run([args.uv, "venv", "--python", recipe.python, str(directory / "venv")])
            installer = ([str(python), "-m", "pip", "install"] if recipe.python == "3.6"
                         else [args.uv, "pip", "install", "--python", str(python)])
            bootstrap = [p for p in recipe.packages if p.split("==")[0] in {"pip", "setuptools", "wheel"}]
            dependencies = [p for p in recipe.packages if p not in bootstrap]
            run([*installer, "--index-url", args.index_url, *bootstrap])
            run([*installer, "--index-url", args.index_url, "--no-build-isolation", *dependencies])
        cache = (args.repo_cache / repo).resolve()
        with tempfile.TemporaryDirectory(prefix="swe-public-base-") as temporary:
            checkout = Path(temporary) / "repo"
            with _git_safe_environment(cache) as overrides:
                env = {**os.environ, **overrides}
                # Use only the local cache. A newly fetched commit can be
                # unreferenced and therefore absent from a normal clone.
                subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(cache), str(checkout)],
                               check=True, env=env, capture_output=True, timeout=120)
                subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "--no-tags",
                                str(cache), commit],
                               check=True, env=env, capture_output=True, timeout=120)
                subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", commit],
                               check=True, env=env, capture_output=True, timeout=60)
            target = resolve_target(checkout, repo, version, "public_smoke", None)
            result = run_public_test(root=args.env_root, repo=repo, version=version, workspace=checkout,
                                     target=target, run_process=SWEWorkspaceLifecycle._run_process,
                                     timeout=args.test_timeout, setup_timeout=args.setup_timeout,
                                     require_ready=False, lock_held=True)
            (directory / "smoke.json").write_text(json.dumps(result, indent=2))
            if not result["test_executed"] or result["returncode"] or result["timed_out"]:
                return {"repo": repo, "version": version, "ready": False,
                        "phase": result["phase"], "returncode": result["returncode"],
                        "details": str(directory / "smoke.json")}
            freeze = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], text=True)
            (directory / "installed.txt").write_text(freeze)
            manifest = {"repo": repo, "version": version, "recipe_sha256": recipe.fingerprint,
                        "smoke_passed": True, "base_commit": commit, "target": target,
                        "tests_started": result["tests_started"], "runner": recipe.runner}
            temporary_manifest = directory / "ready.tmp"
            temporary_manifest.write_text(json.dumps(manifest, indent=2))
            temporary_manifest.replace(directory / "ready.json")
            return {**manifest, "ready": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="JSON array of public repo/version/base_commit")
    parser.add_argument("--repo-cache", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--micromamba", help="Required for legacy Python 3.6 repositories")
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument("--repo", action="append")
    parser.add_argument("--version")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--one-version-per-repo", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--test-timeout", type=float, default=120)
    parser.add_argument("--setup-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.jobs < 1 or min(args.test_timeout, args.setup_timeout) <= 0:
        parser.error("jobs and timeouts must be positive")
    args.env_root = args.env_root.resolve()
    unique = {}
    for row in json.loads(args.tasks.read_text()):
        if args.repo and row["repo"] not in args.repo:
            continue
        if args.version and row["version"] != args.version:
            continue
        key = row["repo"] if args.one_version_per_repo else (row["repo"], row["version"])
        unique.setdefault(key, {k: row[k] for k in ("repo", "version", "base_commit")})
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(prepare, row, args): row for row in unique.values()}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {**futures[future], "ready": False, "error": str(exc)}
            results.append(result)
            print(json.dumps(result), flush=True)
    args.env_root.mkdir(parents=True, exist_ok=True)
    (args.env_root / "preparation-report.json").write_text(json.dumps(results, indent=2))
    return 0 if results and all(row["ready"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
