"""Regressions grounded in the six-dataset live smoke traces."""

import json

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.delegation import (
    delegation_safety_issue,
    delegation_task_alignment_issue,
)
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import (
    ModelAgentExecutor,
    MultiAgentRuntime,
    _recover_text_action_envelope,
    _text_action_calls,
)

from .helpers import RecordingExecutor
from .test_runtime import FakePythonTool


@pytest.mark.parametrize("binary", [False, True])
def test_repair_feedback_matches_admission_including_pending_relation(binary):
    canvas = GraphCanvas(
        task="task", runtime=MultiAgentRuntime(RecordingExecutor()),
        binary_relation_policy=binary,
    )
    for agent in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent})).accepted
        result = canvas.step(json.dumps({
            "action": "set_prompt", "target": agent, "role": "Analyst",
            "objective": "Find relevant facts.", "scope": "Visible evidence.",
            "expected_output": "Return findings.",
        }))
        assert result.accepted
        if agent == "a":
            assert canvas.step('{"action":"set_output","target":"a"}').accepted
    assert result.structural_repair["reason"] == "output_reachability"
    assert result.structural_repair["allowed_actions"] == result.control_snapshot["allowed_actions"]
    allowed = result.structural_repair["allowed_actions"]
    if binary:
        assert "set_relation" not in allowed and "remove_relation" not in allowed
        pending = canvas.step('{"action":"consider_relation","source":"b","target":"a"}')
        assert pending.accepted
        assert pending.structural_repair["allowed_actions"] == ["relation_choice"]
        assert pending.control_snapshot["allowed_actions"] == ["relation_choice"]
        resolved = canvas.resolve_relation_choice("on")
    else:
        assert "consider_relation" not in allowed
        resolved = canvas.step(
            '{"action":"set_relation","source":"b","target":"a","relation":"bidirectional"}'
        )
    assert resolved.accepted and not resolved.structural_repair["required"]


CALL = {"action_calls": [{"name": "python_exec", "arguments": {"code": "print(1)", "purpose": "verify"}}]}
FINAL = {"answer": "unexecuted claim", "summary": "premature", "confidence": 1}


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("suffix", [FINAL, {"answer": None}, {"answer": "", "evidence": {}}, None])
def test_concatenated_action_executes_once_and_discards_unexecuted_answer(copies, suffix):
    response = json.dumps(CALL) * copies + (json.dumps(suffix) if suffix is not None else "")
    backend = MockBackend([response, '{"answer":"1","summary":"from real observation","confidence":1}'])
    tool = FakePythonTool()
    node = AgentNode(
        "solver", "solve", allowed_tools=("python_exec",), operation_policy_configured=True,
        initial_tool_budget=2, total_tool_budget=2,
    )
    artifact = ModelAgentExecutor(backend, tools={"python_exec": tool}).execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0,
    )
    assert tool.calls == [CALL["action_calls"][0]["arguments"]]
    assert artifact.answer == "1"
    assert len(artifact.react_trace) == 1
    replay = backend.calls[1]["messages"]
    assert all("unexecuted claim" not in str(item) for item in replay)
    assert any('"stdout": "1"' in str(item.get("content")) for item in replay)
    if copies > 1 or suffix is not None:
        diagnostic = artifact.protocol_diagnostics[0]
        assert diagnostic["stage"] == "text_action_envelope_recovered"
        assert diagnostic["raw_response"] == response


@pytest.mark.parametrize("response", [
    json.dumps(FINAL) + json.dumps(CALL),
    "Here is an example: " + json.dumps(CALL) + json.dumps(FINAL),
    json.dumps(CALL) + "???" + json.dumps(FINAL),
    json.dumps(CALL) + json.dumps({"action_observation": {"status": "ok"}}) + json.dumps(FINAL),
    json.dumps(CALL) + json.dumps({"action_calls": [{"name": "python_exec", "arguments": {"code": "print(2)"}}]}) + json.dumps(FINAL),
    json.dumps(CALL) + json.dumps(FINAL) + "junk",
    json.dumps(CALL) + '{"answer":null,"answer":"conflicting"}',
])
def test_ambiguous_or_fabricated_action_stream_is_not_executed(response):
    assert _recover_text_action_envelope(response) is None
    assert _text_action_calls(response) == []


def test_webshop_mount_grammar_preserves_public_constraint():
    public_task = "i want a 24w led light with high power lamp. it should mount on a wall"
    fields = {"objective": "find and evaluate 24W LED wall-mounted lights with high power"}
    assert delegation_task_alignment_issue(fields, public_task=public_task, dataset="webshop") is None
    # Other placements and conditions remain substantive task drift.
    for task, objective in (
        (public_task, "find high power ceiling mounted lamps"),
        ("I use a white laptop", "find a white used laptop"),
        ("I want a wall light", "find wall mounted lights"),
    ):
        assert delegation_task_alignment_issue(
            {"objective": objective, "scope": objective + " only"},
            public_task=task, dataset="webshop",
        ) is not None


def test_director_schema_recovery_does_not_ask_to_change_responsibility():
    valid = json.dumps({
        "action": "set_prompt", "target": "a", "role": "Analyst",
        "objective": "Determine the requested value.", "scope": "The original problem.",
        "expected_output": "The requested value.",
    })
    backend = MockBackend([
        '{"action":"add_agent","agent_id":"a"}', valid[:-2], valid,
        '{"action":"set_output","target":"a"}', '{"action":"finish"}',
    ])
    canvas = GraphCanvas(task="problem", runtime=MultiAgentRuntime(RecordingExecutor()))
    result = GraphDirector(backend=backend, canvas=canvas).run()
    assert result.finished
    rejected = canvas.history[1]
    assert not rejected.accepted
    assert rejected.rejection_code == "director_json_syntax_error"
    assert rejected.graph == canvas.history[0].graph
    assert rejected.executed_agents == []
    assert canvas.history[2].accepted
    assert len(canvas.graph.nodes) == 1
    assert canvas.graph.nodes["a"].metadata["director_delegation"] == {
        key: json.loads(valid)[key] for key in ("role", "objective", "scope", "expected_output")
    }
    retry = backend.calls[2]["messages"][-1]["content"]
    assert "director_json_syntax_error" in retry
    assert "Action encoding failure (separate from responsibility or graph legality)" in retry
    assert "The responsibility overlaps" not in retry


@pytest.mark.parametrize("second_layer, expected_relation", [(0, "bidirectional"), (1, "directed")])
def test_authoritative_pair_proposal_accepts_both_orders_without_choosing_edge(second_layer, expected_relation):
    canvas = GraphCanvas(
        task="problem", runtime=MultiAgentRuntime(RecordingExecutor()), binary_relation_policy=True,
    )
    for agent in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent})).accepted
        assert canvas.step(json.dumps({
            "action": "set_prompt", "target": agent, "role": "Analyst",
            "objective": "Determine relevant facts.", "scope": "Visible inputs.",
            "expected_output": "Findings.",
        })).accepted
    if second_layer:
        assert canvas.step('{"action":"set_layer","target":"b","layer":1}').accepted
    raw = '{"action":"consider_relation","source":"b","target":"a"}'
    proposed = canvas.step(raw, authoritative_director=True)
    assert proposed.accepted and proposed.action.raw_text == raw
    assert not canvas.graph.directed_edges and not canvas.graph.bidirectional_edges
    pending = canvas.pending_relation_decision
    assert (pending.source, pending.target, pending.relation_type.value) == ("a", "b", expected_relation)
    assert canvas.resolve_relation_choice("on").accepted
    assert (("a", "b") in canvas.graph.directed_edges) == (expected_relation == "directed")
    assert (("a", "b") in canvas.graph.bidirectional_edges) == (expected_relation == "bidirectional")
    # Canonicalization does not reopen an already considered pair at this version.
    canvas.step('{"action":"consider_relation","source":"a","target":"b"}', authoritative_director=True)
    if canvas.pending_relation_decision:
        canvas.resolve_relation_choice("on")
    duplicate = canvas.step(raw, authoritative_director=True)
    assert not duplicate.accepted


@pytest.mark.parametrize("divisor", [97, 1000])
@pytest.mark.parametrize("representation", ["the remainder", "that count", "this count", "count", "N"])
def test_public_requested_modulo_deliverable_is_not_a_solution_method(divisor, representation):
    task = f"Let N be the requested count. Find the remainder when N is divided by ${divisor}.$"
    text = f"Determine the requested count, then compute {representation} modulo {divisor}"
    assert delegation_safety_issue(text, field="objective", public_task=task) is None
    assert delegation_safety_issue(text, field="objective") is not None
    assert delegation_safety_issue(text, field="scope", public_task=task) is not None
    wrong_modulus = task.replace(str(divisor), str(divisor + 1))
    assert delegation_safety_issue(text, field="objective", public_task=wrong_modulus) is not None
    for method in (
        "Use generating functions to derive the count, ",
        "First compute a dynamic programming table, ",
    ):
        assert delegation_safety_issue(method + text, field="objective", public_task=task) is not None
    assert delegation_safety_issue(
        text + ", then derive the result using generating functions", field="objective", public_task=task,
    ) is not None
    assert delegation_safety_issue(
        f"Determine the count, then compute X modulo {divisor}", field="objective", public_task=task,
    ) is not None
    assert delegation_safety_issue(
        f"Determine the count, then compute N+1 modulo {divisor}", field="objective", public_task=task,
    ) is not None


def test_public_conditional_unanswerable_output_rule_is_allowed_without_answer_leak():
    task = (
        "Based on the following passage, answer the question. If the answer cannot be found "
        "in the passage, respond with 'unanswerable'."
    )
    conditional = (
        "Determine whether the passage answers the question; otherwise, state that "
        "the answer is 'unanswerable'"
    )
    assert delegation_safety_issue(conditional, field="expected_output", public_task=task) is None
    assert delegation_safety_issue(conditional, field="expected_output") is not None
    assert delegation_safety_issue("The answer is unanswerable", field="expected_output", public_task=task) is not None
    assert delegation_safety_issue(
        conditional + "; final answer is Paris", field="expected_output", public_task=task,
    ) is not None


def test_canvas_admits_public_modulo_goal_and_still_requires_director_finish():
    canvas = GraphCanvas(
        task="Let N be the count. Find the remainder when N is divided by $1000.$",
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    result = canvas.step(json.dumps({
        "action":"set_prompt", "target":"a", "role":"Analyst",
        "objective":"Determine the count, then compute that count modulo 1000",
        "scope":"The original counting problem.", "expected_output":"The requested integer.",
    }), authoritative_director=True)
    assert result.accepted
    assert canvas.graph.output_agent is None
    assert canvas.active
