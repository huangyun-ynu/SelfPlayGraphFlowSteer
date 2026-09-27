import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.application import load_adaptive_config
from selfplay_graph_flowsteer.graph import AgentNode
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import _finalize_swe_progress
from selfplay_graph_flowsteer.swe_public_recipes import RECIPES, PublicTestRecipe
from selfplay_graph_flowsteer.swe_public_tests import (
    PublicTestError,
    environment_status,
    resolve_target,
    run_public_test,
)
from selfplay_graph_flowsteer.swebench import (
    CodeArtifactStore,
    SWETestTool,
    SWEWorkspaceLifecycle,
)

from .test_swe_candidate_integrity import _tested_patch


@pytest.mark.parametrize("filename", ["formal_training.toml", ".swe_gpu4_c10.toml"])
def test_active_configs_enable_repository_public_tests(filename, monkeypatch):
    monkeypatch.setattr("selfplay_graph_flowsteer.application._api_key", lambda _: "fixture")
    monkeypatch.setattr("selfplay_graph_flowsteer.application._load_project_env", lambda _: None)
    root = Path(__file__).resolve().parents[1]
    config = load_adaptive_config(root / "configs" / filename, validate=False)
    assert config.swe.public_test_environment_root == root / "state/swe/public-test-envs"
    assert config.swe.public_test_setup_timeout_s == 600
    assert config.swe.max_total_calls == 32


@pytest.mark.parametrize("repo,version,target,expected", [
    ("django/django", "3.1", "utils_tests.test_datastructures", "utils_tests.test_datastructures"),
    ("django/django", "3.1", "tests/utils_tests/test_datastructures.py", "utils_tests.test_datastructures"),
    ("pytest-dev/pytest", "5.1", "testing/test_assertion.py::Test::test_x", "./testing/test_assertion.py::Test::test_x"),
    ("sympy/sympy", "1.1", "sympy/core/tests/test_numbers.py", "./sympy/core/tests/test_numbers.py"),
])
def test_repository_specific_targets(tmp_path, repo, version, target, expected):
    path = tmp_path / ("tests/utils_tests/test_datastructures.py" if repo.startswith("django/")
                       else target.split("::")[0])
    path.parent.mkdir(parents=True)
    path.touch()
    assert resolve_target(tmp_path, repo, version, "public_tests", target) == expected


@pytest.mark.parametrize("target", ["--help", "../outside.py", "/tmp/test.py", ".git/config", "", "tests/test_x.py::"])
def test_invalid_target_never_becomes_runner_argument(tmp_path, target):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").touch()
    with pytest.raises(PublicTestError):
        resolve_target(tmp_path, "pallets/flask", "2.3", "public_tests", target)


def test_symlink_target_outside_workspace_is_rejected(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "outside.py").touch()
    (work / "test.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(PublicTestError):
        resolve_target(work, "pallets/flask", "2.3", "public_tests", "test.py")


def test_unknown_version_and_missing_environment_are_explicit(tmp_path):
    assert environment_status(tmp_path, "django/django", "3.1")["error"] == "test_environment_not_prepared"
    assert environment_status(tmp_path, "django/django", "999")["error"] == "test_repository_version_unsupported"
    called = []
    with pytest.raises(PublicTestError, match="Prepare"):
        run_public_test(root=tmp_path, repo="django/django", version="3.1", workspace=tmp_path,
                        target="utils_tests", run_process=lambda *a, **kw: called.append(a),
                        timeout=1, setup_timeout=1)
    assert not called


@pytest.mark.parametrize("output", [
    {"status": "error", "error": {"code": "test_environment_not_prepared"}},
    {"status": "error", "error": {"code": "stale_workspace_version"}},
    {"status": "error", "error": {"code": "test_profile_not_allowed"}},
    {"status": "ok", "returncode": 0, "test_executed": False},
    {"status": "ok"},
    {"status": "ok", "returncode": "0"},
    {"status": "ok", "returncode": 0, "workspace_version": 0},
])
def test_rejected_or_unexecuted_tests_do_not_satisfy_submission(output):
    artifact = _tested_patch()
    artifact.react_trace = artifact.react_trace[:2]
    artifact.react_trace[0]["observation"]["output"]["workspace_version"] = 1
    artifact.react_trace[1]["observation"]["output"] = copy.deepcopy(output)
    artifact.swe_progress = {}
    _finalize_swe_progress(artifact, node=AgentNode("b", "fix", metadata={"result_scope": "task_result"}))
    assert not artifact.swe_progress["commit_ready"]
    assert artifact.swe_progress["state"] == "test_required"


@pytest.mark.parametrize("returncode", [0, 1])
def test_actual_test_failures_remain_usable_evidence(returncode):
    artifact = _tested_patch()
    artifact.react_trace = artifact.react_trace[:2]
    artifact.react_trace[0]["observation"]["output"]["workspace_version"] = 1
    artifact.react_trace[1]["observation"]["output"].update(
        workspace_version=1, returncode=returncode, test_executed=True)
    artifact.swe_progress = {}
    _finalize_swe_progress(artifact, node=AgentNode("b", "fix", metadata={"result_scope": "task_result"}))
    assert artifact.swe_progress["commit_ready"]
    assert artifact.swe_progress["test_failure_count"] == returncode


def test_real_pytest_observes_current_patch_and_leaves_workspace_clean(tmp_path, monkeypatch):
    """Use real pip/pytest in an isolated venv, with a failing then repaired source."""
    repo = tmp_path / "cache/example/repo"
    repo.mkdir(parents=True)
    (repo / "setup.py").write_text("from setuptools import setup\nsetup(name='swe-public-fixture', version='1.0', py_modules=['answer'])\n")
    (repo / "answer.py").write_text("value = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_answer.py").write_text("from answer import value\ndef test_answer():\n    assert value == 2\n")
    (repo / "tests/test_setup_failure.py").write_text(
        "import pytest\n@pytest.fixture(autouse=True)\ndef broken():\n"
        "    raise RuntimeError('missing dependency fixture')\n"
        "def test_never_entered():\n    assert False\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=a@b.invalid",
                    "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    recipe = PublicTestRecipe("3.11", (), "pytest", "tests/test_answer.py")
    monkeypatch.setitem(RECIPES, ("example/repo", "1.0"), recipe)
    env_root = tmp_path / "environments"
    directory = env_root / "example__repo--1.0"
    subprocess.run([sys.executable, "-m", "venv", str(directory / "venv")], check=True)
    # Share only the test dependencies with this fixture venv, not editable source.
    site = next((directory / "venv/lib").glob("python*/site-packages"))
    (site / "fixture_dependencies.pth").write_text(str(Path(pytest.__file__).parent.parent) + "\n")
    (directory / "ready.json").write_text(json.dumps({"repo": "example/repo", "version": "1.0",
                                                       "recipe_sha256": recipe.fingerprint, "smoke_passed": True}))
    life = SWEWorkspaceLifecycle(repo_cache_root=tmp_path / "cache", workspace_root=tmp_path / "work",
        artifact_store=CodeArtifactStore(tmp_path / "artifacts"), public_test_environment_root=env_root)
    life.bind_task(TaskSpec("fixture", "Fix value", metadata={"dataset": "swe_bench", "instance_id": "example__repo-1",
        "repo": "example/repo", "version": "1.0", "base_commit": commit}))
    try:
        initial = life.begin_execution(agent_id="a", seed=0, revision=False)
        assert initial["public_test_environment"]["ready"]
        assert SWETestTool(life).parameters["properties"]["profile"]["enum"] == ["public_smoke", "public_tests"]
        before = life.test("public_tests", "tests/test_answer.py::test_answer", workspace_version=0)
        assert before["test_executed"] and before["returncode"] == 1, before
        assert life.status()["changed_files"] == []
        change = life.edit({"path": "answer.py", "operation": "replace", "workspace_version": 0,
            "expected_sha256": hashlib.sha256(b"value = 1\n").hexdigest(),
            "old_content": "value = 1", "new_content": "value = 2"})
        after = life.test("public_smoke", None, workspace_version=change["workspace_version"])
        assert after["test_executed"] and after["returncode"] == 0, after
        assert after["tests_started"] == 1
        assert life.status()["changed_files"] == ["answer.py"]
        empty = life.test("public_tests", "tests/test_answer.py::test_missing", workspace_version=1)
        assert empty["status"] == "error" and not empty["test_executed"]
        setup_failure = life.test("public_tests", "tests/test_setup_failure.py", workspace_version=1)
        assert setup_failure["status"] == "error" and not setup_failure["test_executed"]
        assert setup_failure["tests_run"] == 0 and setup_failure["setup_errors"] == 1
    finally:
        life.end_execution()
    assert life.result_for("a")["code_artifact_ref"]["changed_files"] == ["answer.py"]
