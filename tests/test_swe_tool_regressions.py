"""Reproduce the large-file, root-path and false syntax-evidence SWE failures."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from selfplay_graph_flowsteer.contracts import CodeArtifactRef
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.graph import AgentNode
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, _finalize_swe_progress
from selfplay_graph_flowsteer.swe_public_recipes import RECIPES
from selfplay_graph_flowsteer.swe_public_tests import environment_status
from selfplay_graph_flowsteer.swebench import (
    CodeArtifactStore, SWEActionError, SWEWorkspaceLifecycle, swe_tools,
)

from .test_swe_candidate_integrity import _tested_patch


@pytest.fixture
def life(tmp_path):
    repo = tmp_path / "cache/example/repo"
    repo.mkdir(parents=True)
    (repo / "sample.py").write_text("answer = 1\n")
    (repo / "large.py").write_text(("# " + "padding" * 10 + "\n") * 5000 + "ANSWER = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    lifecycle = SWEWorkspaceLifecycle(
        repo_cache_root=tmp_path / "cache", workspace_root=tmp_path / "work",
        artifact_store=CodeArtifactStore(tmp_path / "artifacts"),
        test_profiles={"python_syntax": (sys.executable, "-m", "py_compile")},
    )
    lifecycle.bind_task(TaskSpec("fixture", "Fix answer", metadata={
        "dataset": "swe_bench", "repo": "example/repo", "instance_id": "example__repo-1",
        "base_commit": commit,
    }))
    lifecycle.begin_execution(agent_id="a", seed=0, revision=False)
    try:
        yield lifecycle
    finally:
        lifecycle.end_execution()


def edit_sample(life, new="answer = 2"):
    observed = life.read_file("sample.py", workspace_version=0, start_line=1, end_line=1)
    return life.edit({"path": "sample.py", "operation": "replace", "workspace_version": 0,
                      "expected_sha256": observed["file_sha256"],
                      "old_content": "answer = 1", "new_content": new})


def progress_for(output):
    artifact = _tested_patch()
    artifact.react_trace = artifact.react_trace[:2]
    artifact.react_trace[0]["observation"]["output"]["workspace_version"] = 1
    artifact.react_trace[1]["observation"]["output"] = output
    artifact.swe_progress = {}
    node = AgentNode("b", "fix", metadata={"result_scope": "task_result"})
    _finalize_swe_progress(artifact, node=node)
    return artifact.swe_progress


def test_large_file_read_search_edit_and_patch_roundtrip(life):
    source = life._active.workspace / "large.py"
    assert source.stat().st_size > 341_113
    read = life.read_file("large.py", workspace_version=0, start_line=5001, end_line=5001)
    assert read["content"] == "ANSWER = 1"
    assert read["file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    found = life.search("ANSWER = 1", "", workspace_version=0, max_results=100)
    assert found["matches"] == [{"path": "large.py", "line": 5001, "text": "ANSWER = 1"}]
    life.edit({"path": "large.py", "operation": "replace", "workspace_version": 0,
               "expected_sha256": read["file_sha256"], "old_content": "ANSWER = 1",
               "new_content": "ANSWER = 2"})
    assert source.read_text().endswith("ANSWER = 2\n")
    life.end_execution()
    ref = CodeArtifactRef(**life.result_for("a")["code_artifact_ref"])
    assert b"+ANSWER = 2" in life.artifact_store.read(ref)


def test_large_file_responses_remain_bounded_and_report_skips(life):
    life.max_output_chars = 200
    read = life.read_file("large.py", workspace_version=0, start_line=1, end_line=401)
    assert read["truncated"] and len(read["content"]) == 200
    found = life.search("padding", ".", workspace_version=0, max_results=100)
    assert found["truncated"] and len(found["matches"]) < 100
    life.max_file_bytes = 200_000
    with pytest.raises(SWEActionError) as exc:
        life.read_file("large.py", workspace_version=0, start_line=1, end_line=1)
    assert exc.value.code == "file_too_large"
    assert exc.value.details["size_bytes"] > exc.value.details["max_file_bytes"]
    found = life.search("ANSWER", ".", workspace_version=0, max_results=100)
    assert found["skipped_count"] == 1
    assert found["skipped_files"][0]["code"] == "file_too_large"


def test_file_limit_is_utf8_bytes_and_rejected_edit_is_atomic(life):
    source = life._active.workspace / "sample.py"
    original = source.read_bytes()
    life.max_file_bytes = 40
    with pytest.raises(SWEActionError) as exc:
        edit_sample(life, 'answer = "' + '汉' * 20 + '"')
    assert exc.value.code == "file_too_large"
    assert source.read_bytes() == original and life.status()["workspace_version"] == 0


@pytest.mark.parametrize("alias", ["", " ", ".", "./", ". "])
def test_root_aliases_work_for_list_and_search(life, alias):
    listed = life.list_files(alias, workspace_version=0, max_entries=200)
    assert {x["path"] for x in listed["entries"]} == {"sample.py", "large.py"}
    found = life.search("answer =", alias, workspace_version=0, max_results=100)
    assert found["matches"][0]["path"] == "sample.py"


def test_root_aliases_cannot_evade_runtime_repeat_guard(life):
    backend = MockBackend([
        json.dumps({"action_call": {"name": "swe_list", "arguments": {
            "path": path, "workspace_version": 0}}}) for path in ("", ".", "./", ". ", "")
    ] + [json.dumps({"answer": "Inspected repository", "summary": "Directory inspected"})] * 6)
    life.end_execution()
    tools = swe_tools(life)
    node = AgentNode("a", "Inspect repository", allowed_tools=tuple(tools),
                     initial_tool_budget=10, revision_tool_budget=0, total_tool_budget=10,
                     operation_policy_configured=True,
                     metadata={"action_adapter": "swe_bench", "result_scope": "subtask"})
    registry = default_dataset_action_registry(tools, swe_budgets=(10, 0, 10))
    result = ModelAgentExecutor(backend, tools=tools, action_registry=registry).execute(
        task="Inspect repository", node=node, upstream=[], peers=[], revision=False, seed=0)
    observations = [step["observation"] for step in result.react_trace]
    assert sum(obs.get("output", {}).get("status") == "ok" for obs in observations) == 1
    assert sum(obs.get("error", {}).get("code") == "repeated_no_progress_action"
               for obs in observations) == 4


@pytest.mark.parametrize("path,code", [("../escape", "path_outside_workspace"),
    ("/tmp", "path_outside_workspace"), (".git", "path_outside_workspace"),
    (None, "invalid_path")])
def test_root_alias_change_preserves_path_guards(life, path, code):
    with pytest.raises(SWEActionError) as exc:
        life.list_files(path, workspace_version=0, max_entries=200)
    assert exc.value.code == code


def test_empty_path_is_not_a_file_and_search_never_follows_escaping_symlink(life, tmp_path):
    with pytest.raises(SWEActionError) as exc:
        life.read_file("", workspace_version=0, start_line=1, end_line=1)
    assert exc.value.code == "invalid_path"
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET_TEST_SENTINEL = 1\n")
    (life._active.workspace / "escape.py").symlink_to(outside)
    with pytest.raises(SWEActionError) as exc:
        life.read_file("escape.py", workspace_version=0, start_line=1, end_line=1)
    assert exc.value.code == "symlink_escape"
    found = life.search("SECRET_TEST_SENTINEL", ".", workspace_version=0, max_results=100)
    assert not found["matches"] and found["skipped_files"][0]["code"] == "symlink_escape"
    (life._active.workspace / "escape.py").unlink()


@pytest.mark.parametrize("broken", [False, True])
def test_omitted_syntax_target_checks_changed_file_without_bytecode(life, broken):
    edit_sample(life, "answer = (" if broken else "answer = 2")
    output = life.test("python_syntax", None, workspace_version=1)
    assert output["test_executed"] and output["files_checked"] == ["./sample.py"]
    assert output["test_passed"] is (not broken)
    assert output["returncode"] == int(broken)
    assert not list(life._active.workspace.rglob("*.pyc"))
    progress = progress_for(output)
    assert progress["test_after_latest_edit"] and progress["commit_ready"]
    assert progress["test_failure_count"] == int(broken)
    stale = copy.deepcopy(output)
    stale["workspace_version"] = 0
    assert not progress_for(stale)["commit_ready"]


def test_no_changed_python_files_never_launches_empty_command(life, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("empty syntax check must not launch a process")
    with monkeypatch.context() as patch:
        patch.setattr(life, "_changed_files", lambda _: ())
        patch.setattr(life, "_run_process", unexpected)
        with pytest.raises(SWEActionError) as exc:
            life.test("python_syntax", None, workspace_version=0)
    assert exc.value.code == "test_target_required"


def test_auto_syntax_checks_added_files_and_omits_deleted_files(life):
    source = life._active.workspace / "sample.py"
    life.edit({"path": "sample.py", "operation": "delete", "workspace_version": 0,
               "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    life.edit({"path": "new.py", "operation": "create", "workspace_version": 1,
               "new_content": "value = 1\n"})
    output = life.test("python_syntax", "", workspace_version=2)
    assert output["files_checked"] == ["./new.py"] and output["test_passed"]


@pytest.mark.parametrize("failure", ["startup", "timeout", "old_usage_error"])
def test_failed_checker_start_or_missing_evidence_never_satisfies_submission(
    life, monkeypatch, failure,
):
    edit_sample(life)
    if failure == "startup":
        life.test_profiles["python_syntax"] = ("/no/such/python", "-m", "py_compile")
        with pytest.raises(SWEActionError) as exc:
            life.test("python_syntax", None, workspace_version=1)
        assert exc.value.code == "test_environment_setup_failed"
        output = exc.value.to_dict()
    else:
        monkeypatch.setattr(life, "_run_process", lambda *args, **kw:
                            (2, "", "filenames required", failure == "timeout"))
        output = life.test("python_syntax", None, workspace_version=1)
        assert output["test_executed"] is False and output["test_passed"] is None
    assert not progress_for(output)["commit_ready"]


def test_old_false_pycompile_marker_is_not_accepted_as_execution():
    output = {"status": "ok", "profile": "python_syntax", "returncode": 2,
              "test_executed": True, "workspace_version": 1, "stderr": "filenames required"}
    assert not progress_for(output)["test_after_latest_edit"]


@pytest.mark.parametrize("ran", [0, 1])
def test_public_test_failure_requires_actual_execution_but_not_a_pass(ran):
    output = {"status": "ok", "test_kind": "public_tests", "returncode": -9,
              "test_executed": True, "tests_run": ran, "workspace_version": 1,
              "timed_out": True}
    # Preserve the existing submission policy for real test failures, including
    # timeout after a test body ran; no collected/executed test is not evidence.
    assert progress_for(output)["test_after_latest_edit"] is bool(ran)


@pytest.mark.parametrize("version", ["0.20", "0.22"])
def test_six_recipe_invalidates_previously_ready_environment(tmp_path, version):
    recipe = RECIPES["scikit-learn/scikit-learn", version]
    assert "six==1.16.0" in recipe.packages
    from dataclasses import replace
    old = replace(recipe, packages=tuple(p for p in recipe.packages if not p.startswith("six==")))
    directory = tmp_path / ("scikit-learn__scikit-learn--" + version)
    (directory / "venv/bin").mkdir(parents=True)
    (directory / "venv/bin/python").touch()
    manifest = {"repo": "scikit-learn/scikit-learn", "version": version,
                "recipe_sha256": old.fingerprint, "smoke_passed": True}
    (directory / "ready.json").write_text(json.dumps(manifest))
    assert not environment_status(tmp_path, manifest["repo"], version)["ready"]
