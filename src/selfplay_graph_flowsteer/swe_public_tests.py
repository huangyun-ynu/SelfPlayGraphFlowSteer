"""Run public tests against a snapshot, without touching the exported patch.

Environments are explicitly provisioned before an experiment. A per-environment
lock serializes editable installs; every call replaces the source tree with the
current worker snapshot. No private verifier data or network install is used by
the action. Both setup failures and zero collected tests are explicit evidence.
"""
import fcntl
import json
import os
import re
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from .swe_public_recipes import RECIPES, environment_key

PUBLIC_TEST_PROFILES = ("public_tests", "public_smoke")


class PublicTestError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@contextmanager
def environment_lock(directory: Path, timeout: float):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PublicTestError("test_environment_busy", "public test environment is busy") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment_status(root: Path, repo: str, version: str) -> dict[str, Any]:
    recipe = RECIPES.get((repo, version))
    result: dict[str, Any] = {"repo": repo, "version": version, "ready": False}
    if recipe is None:
        return {**result, "error": "test_repository_version_unsupported"}
    directory = root / environment_key(repo, version)
    try:
        manifest = json.loads((directory / "ready.json").read_text())
    except (OSError, ValueError):
        manifest = {}
    ready = (
        manifest.get("repo") == repo and manifest.get("version") == version
        and manifest.get("recipe_sha256") == recipe.fingerprint
        and manifest.get("smoke_passed") is True
        and (directory / "venv/bin/python").is_file()
    )
    return {**result, "ready": ready, "runner": recipe.runner, "smoke_target": recipe.smoke,
            "error": None if ready else "test_environment_not_prepared"}


def resolve_target(workspace: Path, repo: str, version: str, profile: str, target: object) -> str:
    recipe = RECIPES.get((repo, version))
    if recipe is None:
        raise PublicTestError("test_repository_version_unsupported", f"No recipe for {repo}@{version}")
    if profile == "public_smoke":
        if target not in (None, ""):
            raise PublicTestError("invalid_test_target", "public_smoke has a fixed target; omit target")
        raw = recipe.smoke
    else:
        raw = str(target or "").strip()
    if not raw or raw.startswith("-") or any(c in raw for c in "\x00\n\r"):
        raise PublicTestError("invalid_test_target", "Specify one public test path or test identifier")
    if recipe.runner == "django" and "/" not in raw and not raw.endswith(".py"):
        parts = raw.split(".")
        if not all(p.isidentifier() for p in parts) or not (workspace / "tests" / parts[0]).exists():
            raise PublicTestError("invalid_test_target", "Django target must name an existing public tests module")
        return raw
    path, separator, node = raw.partition("::")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
        raise PublicTestError("invalid_test_target", "Test path must stay inside the workspace")
    resolved = (workspace / relative).resolve()
    if not resolved.is_relative_to(workspace.resolve()) or not resolved.exists():
        raise PublicTestError("invalid_test_target", "Public test path does not exist inside the workspace")
    if recipe.runner == "django":
        if not relative.parts or relative.parts[0] != "tests" or separator:
            raise PublicTestError("invalid_test_target", "Use tests/module.py or a Django dotted test label")
        return ".".join(relative.with_suffix("").parts[1:])
    if recipe.runner == "sympy" and separator:
        raise PublicTestError("invalid_test_target", "SymPy bin/test accepts a test file or directory")
    if separator and not node:
        raise PublicTestError("invalid_test_target", "Empty pytest node identifier")
    # Prefixing ./ also prevents a path component from becoming a runner option.
    return "./" + relative.as_posix() + ("::" + node if separator else "")


def snapshot_source(workspace: Path, directory: Path) -> Path:
    source = directory / "source"
    for path in workspace.rglob("*"):
        if ".git" in path.relative_to(workspace).parts:
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(workspace.resolve()):
            raise PublicTestError("test_snapshot_symlink_escape", "Source symlink leaves the workspace")
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(workspace, source, symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    return source


def run_public_test(*, root: Path, repo: str, version: str, workspace: Path, target: str,
                    run_process, timeout: float, setup_timeout: float,
                    require_ready: bool = True, lock_held: bool = False) -> dict[str, Any]:
    recipe = RECIPES[repo, version]
    directory = root / environment_key(repo, version)
    with nullcontext() if lock_held else environment_lock(directory, setup_timeout):
        if require_ready:
            status = environment_status(root, repo, version)
            if not status["ready"]:
                raise PublicTestError(str(status["error"]), "Prepare this repository/version before running public tests")
        source = snapshot_source(workspace, directory)
        # Nested pytest self-tests must not discover the training project's
        # pyproject.toml above the environment directory.
        (directory / "pytest.ini").write_text("[pytest]\n")
        for name in ("home", "tmp", "cache"):
            (directory / name).mkdir(exist_ok=True)
        python = str(directory / "venv/bin/python")
        environment = {
            "PATH": str(directory / "venv/bin") + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(directory / "home"), "TMPDIR": str(directory / "tmp"),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(source / p) for p in ("", "src", "lib", "tests")),
            "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "SETUPTOOLS_SCM_PRETEND_VERSION": version.lstrip("v"),
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(directory / "cache"),
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "LC_ALL": "C.UTF-8",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        }
        if repo == "matplotlib/matplotlib":
            # Use provisioned native libraries; building a patch must not fetch
            # freetype/qhull from third-party download servers.
            setup_config = directory / "matplotlib-setup.cfg"
            setup_config.write_text("[libs]\nsystem_freetype = true\nsystem_qhull = true\n")
            environment["MPLSETUPCFG"] = str(setup_config)
        setup = run_process(
            [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."],
            cwd=source, timeout_s=setup_timeout, environment_overrides=environment,
        )
        if setup[0] or setup[3]:
            return {"test_executed": False, "phase": "setup", "returncode": setup[0],
                    "stdout": setup[1], "stderr": setup[2], "timed_out": setup[3],
                    "error": {"code": "test_environment_setup_failed"}}
        evidence_path = directory / "test-evidence.json"
        evidence_path.unlink(missing_ok=True)
        if recipe.runner == "pytest":
            command = [python, str(Path(__file__).with_name("_swe_public_probe.py")),
                       str(evidence_path), target]
        elif recipe.runner == "django":
            command = [python, "tests/runtests.py", "--verbosity=1", "--settings=test_sqlite",
                       "--parallel=1", target]
        else:
            command = [python, "bin/test", "-C", "--verbose", target.removeprefix("./")]
        returncode, stdout, stderr, timed_out = run_process(
            command, cwd=source, timeout_s=timeout, environment_overrides=environment,
        )
        started = collected = ran = setup_errors = 0
        if recipe.runner == "pytest":
            if evidence_path.is_file():
                evidence = json.loads(evidence_path.read_text())
                started, collected = evidence["started"], evidence["collected"]
                ran, setup_errors = evidence["ran"], evidence["setup_errors"]
        elif recipe.runner == "django":
            match = re.search(r"Ran (\d+) tests? in", stdout + stderr)
            started = int(match[1]) if match else 0
            ran = started
        else:
            started = sum(int(n) for n in re.findall(r"(\d+) (?:passed|failed|skipped|xfailed)", stdout))
            ran = started
        result = {"test_executed": ran > 0, "phase": "test", "runner": recipe.runner,
                  "tests_started": started, "tests_collected": collected,
                  "tests_run": ran, "setup_errors": setup_errors,
                  "returncode": returncode, "stdout": stdout, "stderr": stderr, "timed_out": timed_out}
        if not ran:
            result["error"] = {"code": "public_tests_not_executed"}
        return result
