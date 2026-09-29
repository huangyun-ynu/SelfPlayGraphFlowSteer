import json

import pytest

from selfplay_graph_flowsteer.aime_verification import POLICY, RESULT_FIELDS
from selfplay_graph_flowsteer.agent_tools import PythonExecutionTool
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import DatasetActionAdapter, DatasetActionRegistry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from .test_unified_submission import add, make_canvas, prompt, step


def result(answer, issues=()):
    return json.dumps(dict(zip(RESULT_FIELDS, [[], "Computed the requested sum.", 0.6,
                                               list(issues), [], answer])))


def call(code):
    return json.dumps({"action_call": {"name": "python_exec", "arguments": {"code": code, "purpose": "verify"}}})


@pytest.mark.parametrize("policy,expected", [("off", False), (POLICY, True)])
def test_clean_result_review_is_opt_in_and_finish_stays_zero_worker(tmp_path, policy, expected):
    c = make_canvas(tmp_path, aime_verification_policy=policy)
    add(c)
    assert ("run_agent" in c.control_snapshot()["allowed_actions"]) is expected
    if expected:
        before = len(c.runtime.executor.calls)
        reviewed = step(c, dict(action="run_agent", target="solver"))
        assert reviewed.accepted
        assert len(c.runtime.executor.calls) == before + 1
        assert c.runtime.executor.calls[-1]["prior_answer"] == "35"
        assert "aime_verification" in reviewed.execution.invalidation_reasons["solver"]
        assert reviewed.execution.cache_hits == 0
        assert not step(c, dict(action="run_agent", target="solver")).accepted
    before = len(c.runtime.executor.calls)
    assert step(c, dict(action="finish", target="solver"), True).accepted
    assert len(c.runtime.executor.calls) == before


def test_review_allowance_survives_prompt_edits_and_node_recreation(tmp_path):
    c = make_canvas(tmp_path, aime_verification_policy=POLICY)
    add(c)
    assert step(c, dict(action="run_agent", target="solver")).accepted
    edit = prompt()
    edit["scope"] = "Check the mathematical constraints independently"
    assert step(c, edit).accepted
    assert "run_agent" not in c.control_snapshot()["allowed_actions"]
    assert step(c, dict(action="delete_agent", target="solver")).accepted
    add(c)
    assert "run_agent" not in c.control_snapshot()["allowed_actions"]
    assert c.control_snapshot()["aime_verification"]["used"] == 1


def test_review_backend_failure_preserves_incumbent_but_consumes_allowance(tmp_path):
    c = make_canvas(tmp_path, aime_verification_policy=POLICY)
    add(c)
    before = c.runtime.artifacts["solver"].artifact_id
    original = c.runtime.executor.execute
    def fail(**kwargs):
        artifact = original(**kwargs)
        artifact.answer = "WORKER_PROTOCOL_FAILURE"
        artifact.integrity_risks = ["terminal_protocol_failure"]
        return artifact
    c.runtime.executor.execute = fail
    assert step(c, dict(action="run_agent", target="solver")).accepted
    assert c.runtime.artifacts["solver"].artifact_id == before
    assert c.submission_assessment("solver")["submit_ready"]
    assert not step(c, dict(action="run_agent", target="solver")).accepted


@pytest.mark.parametrize("recover", [False, True])
def test_real_computation_has_fresh_bounded_review_budget_and_answer_last(tmp_path, recover):
    responses = [call("print(29+36)"), result("13", ["Candidate contradicts the computed sum."]),
                 call("print(29+36)"), call("print(65-29)")]
    if recover:
        responses.append('{"answer":null}')
    responses.append(result("65"))
    backend = MockBackend(responses)
    tool = PythonExecutionTool()
    adapter = DatasetActionAdapter("aime", ("aime",), (tool.name,), 1, 0, 1)
    registry = DatasetActionRegistry((adapter,), available_actions=(tool.name,))
    executor = ModelAgentExecutor(backend, tools={tool.name: tool}, action_registry=registry)
    c = GraphCanvas(task="Compute 29+36.", dataset="aime", action_adapter=adapter,
        runtime=MultiAgentRuntime(executor), config=CanvasConfig(
            submission_protocol="unified_task_result_v1", aime_verification_policy=POLICY,
            aime_verification_tool_budget=2, submission_journal_dir=str(tmp_path),
            max_rounds=30, remaining_token_admission_enabled=False))
    add(c)
    ordinary_scope = c.graph.nodes["solver"].metadata["_runtime_tool_scope"]
    assert executor.budget_ledger.usage[ordinary_scope].total_used == 1
    assessment = c.control_snapshot()["result_assessments"]["solver"]
    assert assessment["submit_ready"]
    assert "Candidate contradicts" in str(assessment["quality_warnings"])
    reviewed = step(c, dict(action="run_agent", target="solver"))
    assert reviewed.accepted, reviewed.feedback
    assert c.runtime.artifacts["solver"].answer == "65"
    assert c.submission_assessment("solver")["submit_ready"]
    assert executor.budget_ledger.usage[ordinary_scope].total_used == 1
    assert executor.budget_ledger.usage["aime-verification:whole-task"].total_used == 2
    assert c.graph.nodes["solver"].total_tool_budget == 1
    before = len(backend.calls)
    assert not step(c, dict(action="run_agent", target="solver")).accepted
    assert len(backend.calls) == before
    for backend_call in backend.calls:
        messages = json.dumps(backend_call["messages"])
        assert "Write answer LAST" in messages
        assert "Compute 29+36." in messages
    review_messages = json.dumps(backend.calls[-1]["messages"])
    assert "untrusted" in review_messages
    assert '"candidate_artifact_id"' in review_messages.replace('\\"', '"')
    context = json.loads(backend.calls[-1]["messages"][-1]["content"])
    assert tuple(context["artifact_schema"]) == RESULT_FIELDS
    assert step(c, dict(action="finish", target="solver"), True).accepted
    assert len(backend.calls) == before


@pytest.mark.parametrize("dataset", ["nq_open", "hotpotqa", "webshop", "alfworld", "swe_bench"])
def test_aime_switch_does_not_enable_other_dataset_reviews(tmp_path, dataset):
    c = make_canvas(tmp_path, dataset, aime_verification_policy=POLICY)
    add(c)
    assert "aime_verification" not in c.control_snapshot()
    assert "aime_verification_policy" not in c.graph.nodes["solver"].metadata


def test_review_respects_token_exhaustion_and_subtask_scope(tmp_path):
    c = make_canvas(tmp_path, aime_verification_policy=POLICY)
    add(c, scope="subtask")
    assert "run_agent" not in c.control_snapshot()["allowed_actions"]
    assert step(c, prompt()).accepted
    c.total_tokens = c.config.max_total_tokens
    assert "run_agent" not in c.control_snapshot()["allowed_actions"]
