"""Exercise patch recovery and pre-finalization tests with real isolated git workspaces."""

import hashlib
import json

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentArtifact, AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import (
    ModelAgentExecutor,
    MultiAgentRuntime,
    _swe_test_recovery_messages,
)
from selfplay_graph_flowsteer.swe_failure_attribution import (
    project_swe_step,
    swe_failure_attribution,
)
from selfplay_graph_flowsteer.swebench import SWEActionError, swe_tools

from .test_swe_tool_regressions import edit_sample
from .test_swe_tool_regressions import life as _life
from .test_unified_submission import add, step

FINAL = json.dumps({"answer": "Applied the fix", "summary": "Repository evidence inspected"})
life = _life


def test_recovery_does_not_mislabel_source_files_as_public_test_targets():
    paths = ["django/core/servers/basehttp.py", "django/test/utils.py",
             "sympy/combinatorics/permutations.py", "tests/servers/tests.py",
             "sympy/combinatorics/tests/test_permutations.py"]
    trace = [{"action": {"name": "swe_read"}, "observation": {"output": {"path": p}}}
             for p in paths]
    messages = _swe_test_recovery_messages({}, trace, workspace_version=1, specs=[],
                                          remaining={"phase": 1, "total": 1})
    assert json.loads(messages[1]["content"])["observed_test_paths"] == sorted(paths[3:])


def action(name, **arguments):
    return json.dumps({"action_calls": [{"name": name, "arguments": arguments}]})


def edit():
    return action("swe_edit", path="sample.py", operation="replace", workspace_version=0,
        expected_sha256=hashlib.sha256(b"answer = 1\n").hexdigest(),
        old_content="answer = 1", new_content="answer = 2")


def test_action(**overrides):
    return action("swe_test", **{"profile": "python_syntax", "workspace_version": 1, **overrides})


test_action.__test__ = False


def worker(life, responses, *, budget=8, backend=None):
    life.end_execution()
    tools = swe_tools(life)
    registry = default_dataset_action_registry(tools, swe_budgets=(budget, 0, budget),
                                                action_budget_policy="shared_total_v1")
    backend = backend or MockBackend(responses)
    executor = ModelAgentExecutor(backend, tools=tools, action_registry=registry)
    executor.set_budget_scope("fixture")
    node = AgentNode("a", "Fix sample.py and test it", allowed_tools=tuple(tools),
        operation_policy_configured=True, initial_tool_budget=budget,
        revision_tool_budget=0, total_tool_budget=budget,
        metadata={"action_adapter": "swe_bench", "submission_protocol": "unified_task_result_v1",
                  "result_scope": "task_result",
                  "dataset_capability_policy": {"action_budget_policy": "shared_total_v1"}})
    artifact = executor.execute(task="Fix answer", node=node, upstream=[], peers=[], revision=False, seed=0)
    return artifact, executor, node, backend


@pytest.mark.parametrize("budget", [2, 8])
def test_early_final_gets_a_real_test_before_text_closure(life, budget):
    artifact, executor, node, backend = worker(life, [edit(), FINAL, FINAL, test_action(), FINAL], budget=budget)
    assert artifact.swe_progress["commit_ready"] and artifact.swe_progress["test_after_latest_edit"]
    assert [t["action"]["name"] for t in artifact.react_trace] == ["swe_edit", "swe_test"]
    assert artifact.react_trace[-1]["observation"]["output"]["files_checked"] == ["./sample.py"]
    assert executor.budget_ledger.remaining(node, revision=False, scope="fixture")["total"] == budget - 2
    recovery = [d for d in artifact.protocol_diagnostics if d["stage"] == "swe_post_edit_test_recovery"]
    assert len(recovery) == 1 and recovery[0]["workspace_version"] == 1
    assert [s["name"] for s in backend.calls[3]["actions"]] == ["swe_test"]
    assert backend.calls[4]["actions"] == []


@pytest.mark.parametrize("bad_test", [test_action(workspace_version=0), test_action(profile="missing"),
    action("swe_read", path="sample.py", workspace_version=1), FINAL])
def test_recovery_has_one_bounded_correction_and_rejects_other_tools(life, bad_test):
    artifact, _, _, backend = worker(life, [edit(), FINAL, FINAL, bad_test, test_action(), FINAL])
    assert artifact.swe_progress["test_after_latest_edit"]
    assert len([d for d in artifact.protocol_diagnostics if d["stage"] == "swe_post_edit_test_recovery"]) == 2
    assert len(backend.calls) == 6
    assert life.created_workspace_count == life.cleaned_workspace_count


def test_repeated_prose_cannot_bypass_test_or_loop_forever(life):
    artifact, _, _, backend = worker(life, [edit(), *([FINAL] * 12)])
    assert not artifact.swe_progress["commit_ready"]
    assert not artifact.swe_progress["test_after_latest_edit"]
    assert len([d for d in artifact.protocol_diagnostics if d["stage"] == "swe_post_edit_test_recovery"]) == 2
    assert len(backend.calls) == 7  # edit, two early finals, two test repairs, two report repairs.
    assert life.recoverable_code_artifacts()[0]["code_artifact_ref"] == artifact.code_artifact_ref.to_dict()


def test_zero_remaining_actions_never_gets_free_tests(life):
    artifact, executor, node, _ = worker(life, [edit(), FINAL, FINAL], budget=1)
    assert not artifact.swe_progress["test_after_latest_edit"]
    assert executor.budget_ledger.remaining(node, revision=False, scope="fixture")["total"] == 0
    assert not any(d["stage"] == "swe_post_edit_test_recovery" for d in artifact.protocol_diagnostics)


def test_already_tested_result_uses_no_recovery_requests(life):
    artifact, _, _, backend = worker(life, [edit(), test_action(), FINAL])
    assert artifact.swe_progress["commit_ready"]
    assert len(backend.calls) == 3
    assert not any(d["stage"] == "swe_post_edit_test_recovery" for d in artifact.protocol_diagnostics)


def test_python_syntax_failure_is_executed_but_not_passed(life):
    bad_edit = json.loads(edit())
    bad_edit["action_calls"][0]["arguments"]["new_content"] = "answer = ("
    artifact, _, _, _ = worker(life, [json.dumps(bad_edit), FINAL, FINAL, test_action(), FINAL])
    test = artifact.react_trace[-1]["observation"]["output"]
    assert test["test_executed"] and test["test_passed"] is False
    assert artifact.swe_progress["test_after_latest_edit"]
    assert artifact.swe_progress["test_failure_count"] == 1


def test_backend_exception_after_edit_still_exports_a_recovery_candidate(life):
    calls = 0

    def disconnected(messages, role):
        nonlocal calls
        calls += 1
        if calls == 1:
            return edit()
        raise ConnectionError("fixture disconnected after mutation")

    with pytest.raises(ConnectionError):
        worker(life, [], backend=MockBackend(handler=disconnected))
    assert life.created_workspace_count == life.cleaned_workspace_count
    candidates = life.recoverable_code_artifacts()
    assert len(candidates) == 1
    assert candidates[0]["code_artifact_ref"]["changed_files"] == ["sample.py"]


def test_historical_test_does_not_substitute_for_test_after_restore(life):
    edit_sample(life)
    assert life.test("python_syntax", "", workspace_version=1)["test_passed"]
    life.end_execution()
    ref = life.result_for("a")["code_artifact_ref"]
    artifact, _, _, _ = worker(life, [
        action("swe_apply_artifact", artifact_sha256=ref["artifact_sha256"], workspace_version=0),
        *([FINAL] * 10),
    ])
    assert artifact.code_artifact_ref.to_dict() == ref
    assert not artifact.swe_progress["test_after_latest_edit"]
    assert not artifact.swe_progress["commit_ready"]


def test_failed_test_start_does_not_create_test_evidence(life, monkeypatch):
    original = life._run_process

    def missing_python(args, **kwargs):
        if "_swe_syntax_probe.py" in " ".join(map(str, args)):
            raise OSError("fixture unavailable interpreter")
        return original(args, **kwargs)

    monkeypatch.setattr(life, "_run_process", missing_python)
    artifact, _, _, _ = worker(life, [edit(), FINAL, FINAL, test_action(), test_action(), FINAL, FINAL])
    assert not artifact.swe_progress["commit_ready"]
    assert not artifact.swe_progress["test_after_latest_edit"]
    assert artifact.code_artifact_ref and life.recoverable_code_artifacts()


def test_candidate_survives_empty_attempt_and_is_available_to_recreated_node(life):
    edit_sample(life)
    life.end_execution()
    ref = life.result_for("a")["code_artifact_ref"]
    life.set_visible_artifacts([])
    life.begin_execution(agent_id="a", seed=0, revision=True)
    life.end_execution()  # e.g. a request refused before any tool call.
    assert life.result_for("a")["code_artifact_ref"] is None
    state = life.begin_execution(agent_id="replacement", seed=0, revision=False)
    assert state["recoverable_code_artifacts"][0]["code_artifact_ref"] == ref
    assert state["workspace_version"] == 0 and not state["changed_files"]
    restored = life.apply_artifact(ref["artifact_sha256"], workspace_version=0)
    assert restored["workspace_version"] == 1
    assert life.test("python_syntax", "", workspace_version=1)["test_executed"]
    life.end_execution()
    assert life.result_for("replacement")["code_artifact_ref"] == ref


def test_archive_is_bound_to_one_task_and_patch_content(life):
    edit_sample(life)
    life.end_execution()
    ref = life.result_for("a")["code_artifact_ref"]
    copy = life.recoverable_code_artifacts()
    copy[0]["code_artifact_ref"]["artifact_sha256"] = "0" * 64
    assert life.recoverable_code_artifacts()[0]["code_artifact_ref"] == ref
    life.begin_execution(agent_id="b", seed=0, revision=False)
    (life.artifact_store.root / (ref["artifact_sha256"] + ".patch")).write_text("tampered")
    with pytest.raises(RuntimeError, match="integrity"):
        life.apply_artifact(ref["artifact_sha256"], workspace_version=0)
    assert life.status()["workspace_version"] == 0
    life.end_execution()
    life.bind_task(TaskSpec("new", "Another issue", metadata={"dataset": "swe_bench",
        "repo": ref["repo"], "instance_id": "example__repo-2", "base_commit": ref["base_commit"]}))
    assert not life.recoverable_code_artifacts()
    life.begin_execution(agent_id="a", seed=0, revision=False)
    with pytest.raises(SWEActionError, match="visible"):
        life.apply_artifact(ref["artifact_sha256"], workspace_version=0)


def test_changed_dependencies_keep_candidate_recoverable_but_require_new_execution_and_test(life, tmp_path):
    life.end_execution()
    tools = swe_tools(life)
    registry = default_dataset_action_registry(tools, swe_budgets=(24, 8, 32), action_budget_policy="shared_total_v1")

    class Executor(ModelAgentExecutor):
        fail = False

        def execute(self, **kwargs):
            if self.fail:
                life.begin_execution(agent_id=kwargs["node"].agent_id, seed=0, revision=True)
                life.end_execution()
                return AgentArtifact("pending", kwargs["node"].agent_id, "WORKER_PROTOCOL_FAILURE")
            return super().execute(**kwargs)

    backend = MockBackend([action("swe_read", path="sample.py", start_line=1, end_line=1, workspace_version=0),
                           FINAL, edit(), test_action(), FINAL])
    executor = Executor(backend, tools=tools, action_registry=registry)
    canvas = GraphCanvas(task="Fix answer", dataset="swe_bench", runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get("swe_bench"), config=CanvasConfig(
            submission_protocol="unified_task_result_v1", director_budget_policy="edits_v1",
            max_director_edits=24, max_total_tokens=350000, submission_journal_dir=str(tmp_path / "journal")))
    add(canvas, "a", "subtask")
    add(canvas, "b", "task_result")
    original = canvas.runtime.artifacts["b"]
    assert original.swe_progress["commit_ready"]
    executor.fail = True
    assert step(canvas, dict(action="set_relation", source="a", target="b", relation="bidirectional")).accepted
    assert canvas.runtime.artifacts["b"].code_artifact_ref is None
    assert not canvas.submission_assessment("b")["submit_ready"]
    assert canvas.control_snapshot()["recoverable_code_artifacts"][0]["code_artifact_ref"] == original.code_artifact_ref.to_dict()
    attribution = swe_failure_attribution(map(project_swe_step, canvas.history), worker_token_limit=350000)
    assert "swe_candidate_recovery_failed" not in attribution["reason_codes"]
    executor.fail = False
    executor.backend = MockBackend([
        action("swe_apply_artifact", artifact_sha256=original.code_artifact_ref.artifact_sha256, workspace_version=0),
        FINAL, FINAL, test_action(), FINAL,
    ])
    recovered = step(canvas, dict(action="run_agent", target="b"))
    assert recovered.accepted
    current = canvas.runtime.artifacts["b"]
    assert current.artifact_id != original.artifact_id
    assert current.code_artifact_ref == original.code_artifact_ref
    assert current.swe_progress["test_after_latest_edit"]
    assert canvas.runtime.artifact_matches_current_input_signature("b", task=canvas.worker_task, graph=canvas.graph)
    assert canvas.submission_assessment("b")["submit_ready"]
