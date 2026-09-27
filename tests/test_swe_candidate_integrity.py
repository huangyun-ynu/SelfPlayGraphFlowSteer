"""A tested task_result patch is independent of legacy output selection."""

import copy

import pytest

from selfplay_graph_flowsteer.contracts import AgentArtifact, CodeArtifactRef
from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
from selfplay_graph_flowsteer.runtime import _enforce_artifact_integrity

from .test_swe_execution_admission import BudgetedExecutor, make_canvas
from .test_unified_submission import add, step


def _tested_patch():
    return AgentArtifact(
        "patch", "b", "Implemented the fix", confidence=0.8,
        code_artifact_ref=CodeArtifactRef("a" * 64, "fixture-1", "fixture/repo", "b" * 40, 40, ("sample.py",)),
        react_trace=[
            {"action": {"name": "swe_edit"}, "observation": {
                "status": "ok", "output": {"status": "ok", "changed_files": ["sample.py"]}}},
            {"action": {"name": "swe_test"}, "observation": {
                "status": "ok", "output": {"status": "ok", "returncode": 0}}},
            {"action": {"name": "swe_read"}, "observation": {
                "status": "error", "error": {"code": "repeated_no_progress_action"}}},
        ],
        swe_progress={
            "trusted": True, "state": "tested", "result_scope": "task_result",
            "selected_as_output": False, "commit_required": True,
            "commit_ready": True, "workspace_changed": True, "test_after_latest_edit": True,
        },
    )


@pytest.mark.parametrize("scope,selected,waived", [
    ("task_result", False, True), (None, True, True),
    ("subtask", False, False), ("subtask", True, False), (None, False, False),
])
def test_repeat_read_waiver_uses_current_result_scope(scope, selected, waived):
    artifact = _tested_patch()
    artifact.swe_progress.update(result_scope=scope, selected_as_output=selected)
    original_trace = copy.deepcopy(artifact.react_trace)
    _enforce_artifact_integrity(artifact)
    assert ("terminal_tool_failure" not in artifact.integrity_risks) is waived
    assert artifact.runtime_tool_evidence["failed_count"] == 1
    assert artifact.runtime_tool_evidence["failure_codes"] == ["repeated_no_progress_action"]
    assert artifact.react_trace == original_trace


@pytest.mark.parametrize("change", [
    "untrusted", "not_ready", "unchanged", "untested", "write_rejection", "transport_error", "missing_action",
])
def test_candidate_fix_does_not_waive_real_errors_or_incomplete_evidence(change):
    artifact = _tested_patch()
    field = {"untrusted": "trusted", "not_ready": "commit_ready",
             "unchanged": "workspace_changed", "untested": "test_after_latest_edit"}.get(change)
    if field:
        artifact.swe_progress[field] = False
    elif change == "write_rejection":
        artifact.react_trace[-1]["action"]["name"] = "swe_edit"
    elif change == "missing_action":
        artifact.react_trace[-1]["action"] = None
    else:
        artifact.react_trace[-1]["observation"]["error"]["code"] = "transport_failure"
    _enforce_artifact_integrity(artifact)
    assert "terminal_tool_failure" in artifact.integrity_risks


def test_graph_only_blocker_does_not_trigger_candidate_rerun(tmp_path):
    class Executor(BudgetedExecutor):
        def execute(self, **kwargs):
            original = super().execute(**kwargs)
            return _tested_patch() if kwargs["node"].agent_id == "b" else original

    executor = Executor(charge=False)
    canvas = make_canvas(tmp_path, executor)
    add(canvas, "a", "subtask")
    add(canvas, "b", "task_result")
    candidate = canvas.runtime.artifacts["b"]
    assert "terminal_tool_failure" not in candidate.integrity_risks
    assert canvas.submission_assessment("b")["blockers"] == ["graph:agents cannot influence output: a"]
    assert not canvas._unified_can_run("b")
    assert not step(canvas, {"action": "run_agent", "target": "b"}).accepted
    assert step(canvas, {"action": "delete_agent", "target": "a"}).accepted
    assert canvas.submission_assessment("b")["submit_ready"]
    assert step(canvas, {"action": "finish", "target": "b"}, True).accepted
    assert canvas.runtime.artifacts["b"] is candidate
    assert len(executor.calls) == 2
    assert canvas.submission_receipt.artifact_id == candidate.artifact_id


@pytest.mark.parametrize("additional_error", [None, "transport_failure", "action_execution_failed"])
def test_shared_budget_error_mapping_needs_independent_terminal_evidence(additional_error):
    failure_codes = ["total_action_budget_exhausted"]
    if additional_error:
        failure_codes.append(additional_error)
    args = dict(
        terminal=True, rejection_codes=["director_edit_budget_dead_end"],
        artifacts={}, output_agent=None, rounds=35, max_rounds=0,
        director_edits=24, director_edit_limit=24, worker_tokens=1000, worker_token_limit=350000,
        historical_artifacts={"deleted": {"runtime_tool_evidence": {
            "failed_count": len(failure_codes), "failure_codes": failure_codes}}},
    )
    outcome = terminal_policy_failure("swe_bench", **args)
    if additional_error:
        assert outcome is None
    else:
        assert outcome["code"] == "director_edit_budget_exhausted"
    assert terminal_policy_failure("swe_bench", **{**args, "rejection_codes": []}) is None
    assert terminal_policy_failure("swe_bench", **{**args, "terminal": False}) is None
    assert terminal_policy_failure("swe_bench", **args, infrastructure_failure=True) is None
