from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.actions import ActionParser, ActionType
from selfplay_graph_flowsteer.canvas import CanvasState, GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.dataset_actions import DatasetActionAdapter
from selfplay_graph_flowsteer.director import (
    DIRECTOR_BASE_PROMPT,
    PROBLEM_TYPE_HINTS,
    GraphDirector,
    director_prompt_components,
    infer_director_problem_type,
)
from selfplay_graph_flowsteer.llm import BinaryChoiceResponse, MockBackend
from selfplay_graph_flowsteer.mace import MACEModelRouter
from selfplay_graph_flowsteer.observability import TaskSpec, trace_from_canvas
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime

from .helpers import NumericRecordingExecutor, RecordingExecutor


def test_empty_canvas_allows_one_recovery_then_stops_without_fifth_call():
    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    backend = MockBackend(["not JSON"] * 4)
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert len(backend.calls) == len(run.turns) == 4
    assert not canvas.graph.nodes and not canvas.topology_edits_frozen
    assert canvas.history[2].control_snapshot["allowed_actions"] == ["add_agent"]
    assert "Bounded recovery" in backend.calls[3]["messages"][-1]["content"]
    assert [t.action_diagnostics["no_progress_streak_before_call"] for t in run.turns] == [
        0,
        1,
        2,
        3,
    ]
    assert run.turns[-1].action_diagnostics["bounded_recovery_call"]
    assert canvas.history[-1].rejection_code == "director_no_progress_exhausted"
    assert canvas.history[-1].protocol_recovery
    assert all(turn.raw_action_text == "not JSON" for turn in run.turns)


def test_empty_canvas_recovery_progress_resets_streak_and_can_finish():
    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    backend = MockBackend(
        ["bad"] * 3
        + [
            '{"action":"add_agent","agent_id":"solver"}',
            '{"action":"set_prompt","target":"solver","role":"Analyst",'
            '"objective":"Solve the task.","scope":"Reason independently.",'
            '"expected_output":"Return a finding."}',
            '{"action":"set_output","target":"solver"}',
        ]
    )
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert run.finished
    assert len(backend.calls) == 6
    assert not any(s.rejection_code == "director_no_progress_exhausted" for s in canvas.history)


def test_no_legal_continuation_stops_before_request():
    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    canvas.topology_edits_frozen = True
    backend = MockBackend([])
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert not run.turns and not backend.calls
    assert canvas.history[-1].rejection_code == "director_no_legal_continuation"


def test_pending_configuration_stall_is_bounded_despite_different_bad_actions():
    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    backend = MockBackend(
        ['{"action":"add_agent"}']
        + [
            '{"action":"finish"}',
            '{"action":"set_output","target":"missing"}',
            '{"action":"set_layer","target":"missing","layer":2}',
            '{"action":"finish"}',
        ]
    )
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert len(run.turns) == len(backend.calls) == 5
    assert canvas.pending_agent_id is not None
    assert canvas.history[-1].rejection_code == "director_no_progress_exhausted"


def test_accepted_noops_and_version_bumps_do_not_reset_progress():
    class NoopCanvas(GraphCanvas):
        def step(self, raw_action, **kwargs):
            self.graph.version += 1
            self.round_index += 1
            action = self.parser.parse(raw_action) if isinstance(raw_action, str) else raw_action
            return self._record(action, accepted=True, feedback="noop")

    canvas = NoopCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    backend = MockBackend(['{"action":"add_agent"}'] * 4)
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert len(run.turns) == 4 and all(t.accepted for t in run.turns)
    assert canvas.history[-1].rejection_code == "director_no_progress_exhausted"


def test_truncated_reasoning_cannot_fall_back_to_a_quoted_canvas_action():
    from selfplay_graph_flowsteer.llm import DirectorContextExhausted, LLMResponse

    raw = 'I might choose {"action":"add_agent"}'

    class Backend(MockBackend):
        def generate(self, *args, **kwargs):
            if getattr(self, "called", False):
                raise DirectorContextExhausted("test terminal")
            self.called = True
            return LLMResponse(text=raw, model="test", raw_reasoning_text=raw, raw_action_text="")

    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(RecordingExecutor()))
    run = GraphDirector(backend=Backend([]), canvas=canvas).run()
    assert len(run.turns) == 1
    assert not run.turns[0].accepted
    assert run.turns[0].raw_reasoning_text == raw
    assert run.turns[0].raw_action_text == ""
    assert not canvas.graph.nodes


def test_context_exhaustion_preserves_graph_and_real_turns_without_synthetic_action():
    from selfplay_graph_flowsteer.llm import DirectorContextExhausted

    class ExhaustedBackend(MockBackend):
        def generate(self, *args, **kwargs):
            if getattr(self, "called", False):
                raise DirectorContextExhausted("context exhausted")
            self.called = True
            return super().generate(*args, **kwargs)

    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(executor=RecordingExecutor()))
    run = GraphDirector(backend=ExhaustedBackend(['{"action":"add_agent"}']), canvas=canvas).run()
    assert not run.finished
    assert len(run.turns) == 1
    assert len(canvas.graph.nodes) == 1
    assert canvas.graph.output_agent is None
    assert canvas.history[-1].rejection_code == "director_context_budget_exhausted"


def test_director_failure_retains_completed_turn_and_graph_without_executing_partial_action():
    class FailingBackend(MockBackend):
        def generate(self, *args, **kwargs):
            if getattr(self, "called", False):
                raise TimeoutError("simulated Director timeout")
            self.called = True
            return super().generate(*args, **kwargs)

    canvas = GraphCanvas(task="test", runtime=MultiAgentRuntime(executor=RecordingExecutor()))
    backend = FailingBackend(['{"action":"add_agent"}'])
    with pytest.raises(TimeoutError) as caught:
        GraphDirector(backend=backend, canvas=canvas).run()
    state = caught.value.partial_state
    assert len(state["director_turns"]) == 1
    assert state["director_turns"][0]["model_action"] == '{"action":"add_agent"}'
    assert len(state["history"]) == 1
    assert len(canvas.graph.nodes) == 1


def test_rejected_finish_only_recovery_terminates_without_busy_loop():
    class RejectingFinishCanvas(GraphCanvas):
        finish_attempts = 0

        def control_snapshot(self):
            snapshot = super().control_snapshot()
            snapshot["allowed_actions"] = ["finish"]
            return snapshot

        def _finish(self, action):
            self.finish_attempts += 1
            assert self.finish_attempts == 1, "automatic FINISH must not busy-loop"
            return self._record(
                action,
                accepted=False,
                feedback="SWE output incomplete",
                rejection_code="swe_output_commit_incomplete",
            )

    canvas = RejectingFinishCanvas(
        task="test", runtime=MultiAgentRuntime(executor=RecordingExecutor())
    )
    result = GraphDirector(backend=MockBackend([]), canvas=canvas).run()
    assert not result.finished
    assert not result.turns  # No invented model output or automatic graph repair.
    assert canvas.state is CanvasState.FAILED
    assert canvas.finish_attempts == 1
    assert len(canvas.history) == 2
    assert canvas.history[-1].rejection_code == "swe_output_commit_incomplete"


def test_frozen_topology_keeps_other_legal_recovery_actions(monkeypatch):
    class RecoverableCanvas(GraphCanvas):
        def control_snapshot(self):
            snapshot = super().control_snapshot()
            snapshot["allowed_actions"] = ["set_output", "finish"]
            return snapshot

        def _finish(self, action):
            return self._record(
                action,
                accepted=False,
                feedback="Output incomplete",
                rejection_code="swe_output_commit_incomplete",
            )

    canvas = RecoverableCanvas(task="test", runtime=MultiAgentRuntime(executor=RecordingExecutor()))
    canvas.step('{"action":"add_agent"}')
    canvas.graph.output_agent = "agent_1"
    canvas.state = CanvasState.BUILDING
    canvas.topology_edits_frozen = True
    monkeypatch.setattr(canvas.graph, "validate", lambda **kwargs: [])
    step = canvas.recover_finish_only()
    assert step is not None and not step.accepted
    assert canvas.state is not CanvasState.FAILED


class BackendFailureExecutor(RecordingExecutor):
    def execute(self, **kwargs):
        node = kwargs["node"]
        return AgentArtifact(
            "pending",
            node.agent_id,
            "WORKER_BACKEND_FAILURE",
            confidence=0.0,
            unresolved_issues=["transient_backend_error:APITimeoutError"],
        )


def test_director_preserves_e2_budget_diagnostics() -> None:
    budget = {"prompt_tokens": 1000, "max_tokens": 31704, "thinking_token_budget": 30680}

    class DiagnosticBackend(MockBackend):
        def generate(self, *args, **kwargs):
            response = super().generate(*args, **kwargs)
            response.metadata.update(finish_reason="length", director_dynamic_budget=budget)
            return response

    canvas = GraphCanvas(
        task="question",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_rounds=1),
    )
    result = GraphDirector(
        backend=DiagnosticBackend(['{"action":"add_agent"}']), canvas=canvas
    ).run()
    assert result.turns[0].action_diagnostics["finish_reason"] == "length"
    assert result.turns[0].action_diagnostics["director_dynamic_budget"] == budget


class _BinaryTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"off": [101], "on": [102]}.get(text, list(text.encode()))


class _RejectingRelationCanvas(GraphCanvas):
    """Model a factual Canvas admission rejection after a valid binary sample."""

    def resolve_relation_choice(self, choice: str, **kwargs):
        step = super().resolve_relation_choice(choice, **kwargs)
        step.accepted = False
        step.rejection_code = "time_budget_consolidation_required"
        return step


def test_director_uses_separate_auditable_binary_relation_turn() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent"}',
            '{"action":"set_prompt","target":"agent_1","role":"Source",'
            '"objective":"Find evidence.","scope":"Evidence only.",'
            '"expected_output":"Return findings."}',
            '{"action":"add_agent"}',
            '{"action":"set_prompt","target":"agent_2","role":"Synthesizer",'
            '"objective":"Synthesize evidence.","scope":"Final synthesis.",'
            '"expected_output":"Return the answer."}',
            '{"action":"consider_relation","source":"agent_1","target":"agent_2"}',
            '{"action":"set_output","target":"agent_2"}',
            '{"action":"finish"}',
        ],
        binary_responses=[
            BinaryChoiceResponse(
                choice="on",
                model="mock:graph-director",
                probabilities={"off": 0.4, "on": 0.6},
                log_probabilities={"off": -0.9162907319, "on": -0.5108256238},
                token_ids={"off": 101, "on": 102},
            )
        ],
    )
    canvas = GraphCanvas(
        task="question",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        binary_relation_policy=True,
    )

    result = GraphDirector(
        backend=backend,
        canvas=canvas,
        tokenizer=_BinaryTokenizer(),
    ).run()

    assert result.finished
    assert result.graph["relations"] == [
        {"source": "agent_1", "target": "agent_2", "relation": "bidirectional"}
    ]
    assert result.turns[4].turn_kind == "graph_action"
    assert result.turns[5].turn_kind == "relation_choice"
    assert result.turns[5].model_action == "on"
    assert result.turns[5].relation_decision["policy"]["probabilities"] == {
        "off": 0.4,
        "on": 0.6,
    }
    binary_call = next(call for call in backend.calls if "binary_choices" in call)
    assert binary_call["temperature"] == 1.0
    assert binary_call["top_p"] == 1.0
    assert binary_call["max_tokens"] == 1


def test_rejected_binary_relation_sample_remains_a_trainable_policy_turn() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent"}',
            '{"action":"set_prompt","target":"agent_1","role":"Source",'
            '"objective":"Find evidence.","scope":"Evidence only.",'
            '"expected_output":"Return findings."}',
            '{"action":"add_agent"}',
            '{"action":"set_prompt","target":"agent_2","role":"Reviewer",'
            '"objective":"Check evidence.","scope":"Independent check.",'
            '"expected_output":"Return findings."}',
            '{"action":"consider_relation","source":"agent_1","target":"agent_2"}',
            '{"action":"set_output","target":"agent_1"}',
            '{"action":"delete_agent","target":"agent_2"}',
            '{"action":"finish"}',
        ],
        binary_responses=[
            BinaryChoiceResponse(
                choice="off",
                model="mock:graph-director",
                probabilities={"off": 0.6, "on": 0.4},
                log_probabilities={"off": -0.5108256238, "on": -0.9162907319},
                token_ids={"off": 101, "on": 102},
            )
        ],
    )
    canvas = _RejectingRelationCanvas(
        task="question",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        binary_relation_policy=True,
    )

    result = GraphDirector(
        backend=backend,
        canvas=canvas,
        tokenizer=_BinaryTokenizer(),
    ).run()

    relation_turn = next(turn for turn in result.turns if turn.turn_kind == "relation_choice")
    assert not relation_turn.accepted
    assert relation_turn.rejection_code == "time_budget_consolidation_required"
    assert relation_turn.trainable


def test_director_uses_coarse_reusable_problem_types() -> None:
    cases = {
        ("aime", "qa", ""): "math",
        ("nq_open", "general", ""): "retrieval_qa",
        ("hotpotqa", "general", ""): "retrieval_qa",
        ("healthbench_professional", "healthcare", ""): "response",
        ("webshop", "general", ""): "environment",
        ("alfworld", "general", ""): "environment",
        ("swe_bench", "general", ""): "code",
        ("unknown", "math", ""): "math",
        ("unknown", "unknown", "retrieval_qa"): "retrieval_qa",
        ("unknown", "qa", ""): "general",
    }

    for (dataset, task_type, adapter_id), expected in cases.items():
        assert (
            infer_director_problem_type(
                dataset=dataset,
                task_type=task_type,
                action_adapter_id=adapter_id,
            )
            == expected
        )


def test_director_appends_only_the_selected_problem_type_hint() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent"}',
            '{"action":"set_prompt","target":"agent_1","role":"Clinical response owner",'
            '"objective":"Address the public request.","scope":"Use the public conversation.",'
            '"expected_output":"Return a coherent response."}',
            '{"action":"set_output","target":"agent_1"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(
        task="public conversation",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        task_type="healthcare",
        dataset="healthbench_professional",
    )

    result = GraphDirector(backend=backend, canvas=canvas).run()
    system_prompt = backend.calls[0]["messages"][0]["content"]

    assert result.finished
    assert "## Problem-Type Guidance: Context-Grounded Professional Response" in system_prompt
    assert "## Problem-Type Guidance: Retrieval Question Answering" not in system_prompt
    assert "healthbench_professional" not in system_prompt


def test_director_prompt_keeps_agent_count_neutral() -> None:
    prompt = DIRECTOR_BASE_PROMPT

    assert "neither to minimize nor to maximize the number of Agents" in prompt
    assert "Do not treat either one-Agent or multi-Agent organization as the default" in prompt
    assert "Before selecting an output" in prompt
    assert "a single-Agent graph is valid" in prompt
    assert "not a fixed role set or topology template" in prompt
    assert "smallest workflow" not in prompt
    assert "compact Agent" not in prompt


def test_director_v2_control_prompt_preserves_archived_shrink_prior() -> None:
    base, hints = director_prompt_components("v2")

    assert "build and revise a compact Agent" in base
    assert "Build the smallest workflow" in base
    assert "Prefer one Agent to own" in hints["environment"]
    assert "continuous\ninteraction" in hints["environment"]
    assert "Prefer one implementation owner" in hints["code"]
    assert "neither to minimize nor to maximize" not in base


def test_director_rejects_unknown_prompt_variant() -> None:
    with pytest.raises(ValueError, match="unsupported Director prompt variant"):
        director_prompt_components("v3")


def test_problem_type_owner_rules_do_not_limit_graph_size() -> None:
    hints = {name: " ".join(text.split()) for name, text in PROBLEM_TYPE_HINTS.items()}

    assert "output-ownership requirement is not a limit on graph size" in hints["response"]
    assert "environment-ownership constraint, not a limit on graph size" in hints["environment"]
    assert "commit-ownership constraint, not a limit on graph size" in hints["code"]
    assert "planning, constraint" in hints["environment"]
    assert "analyze tests, or review" in hints["code"]


def test_action_parser_repairs_unescaped_latex_inside_json_prompt() -> None:
    action = ActionParser().parse(
        r'{"action":"set_prompt","target":"solver","prompt":"Use \angle APB and \frac{x}{2}."}'
    )
    assert action.valid
    assert action.prompt == r"Use \angle APB and \frac{x}{2}."


def test_action_parser_validates_prompt_revision_evidence_fields() -> None:
    parsed = ActionParser().parse(
        '{"action":"set_prompt","target":"solver","role":"Verifier",'
        '"objective":"Recheck the task.","scope":"Resolve uncertainty.",'
        '"expected_output":"Return a checked finding.",'
        '"revision_basis":"unresolved_issue",'
        '"evidence_agent_ids":["solver","solver"]}'
    )
    invalid_basis = ActionParser().parse(
        '{"action":"set_prompt","target":"solver","role":"Verifier",'
        '"objective":"Recheck the task.","scope":"Resolve uncertainty.",'
        '"expected_output":"Return a checked finding.",'
        '"revision_basis":"model_said_so",'
        '"evidence_agent_ids":["solver"]}'
    )
    invalid_ids = ActionParser().parse(
        '{"action":"set_prompt","target":"solver","role":"Verifier",'
        '"objective":"Recheck the task.","scope":"Resolve uncertainty.",'
        '"expected_output":"Return a checked finding.",'
        '"revision_basis":"unresolved_issue",'
        '"evidence_agent_ids":"solver"}'
    )

    assert parsed.valid
    assert parsed.revision_basis is not None
    assert parsed.revision_basis.value == "unresolved_issue"
    assert parsed.evidence_agent_ids == ("solver",)
    assert not invalid_basis.valid
    assert "unknown prompt revision_basis" in str(invalid_basis.parse_error)
    assert not invalid_ids.valid
    assert "must be an array" in str(invalid_ids.parse_error)


def test_canvas_enforces_prompt_barrier_and_rolls_back_invalid_edit() -> None:
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    added = canvas.step('{"action":"add_agent","agent_id":"a"}')
    assert added.accepted
    assert canvas.state is CanvasState.AWAITING_PROMPT
    version = canvas.graph.version

    rejected = canvas.step('{"action":"add_agent","agent_id":"b"}')
    assert not rejected.accepted
    assert canvas.graph.version == version
    assert set(canvas.graph.nodes) == {"a"}

    prompted = canvas.step(
        '{"action":"set_prompt","target":"a","role":"Independent solver",'
        '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
        '"expected_output":"Return a direct final answer."}'
    )
    assert prompted.accepted
    assert prompted.execution is not None
    assert "Process signals: a(" in prompted.feedback
    assert canvas.state is CanvasState.BUILDING


def test_canvas_allows_single_agent_without_fixed_role_or_topology() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    fixed_role = canvas.step(
        '{"action":"set_prompt","target":"solver","structural_operator":"solver",'
        '"role":"Independent solver","objective":"Solve the assigned task.",'
        '"scope":"Reason independently.","expected_output":"Return a candidate."}'
    )
    assert not fixed_role.accepted
    assert "must not assign a fixed structural_operator" in fixed_role.feedback
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Independent analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return a direct answer."}'
    ).accepted
    checkpoint = canvas.step('{"action":"set_output","target":"solver"}')

    assert checkpoint.accepted and checkpoint.execution is None
    assert not checkpoint.final_execution
    assert len(executor.calls) == 1
    finished = canvas.step('{"action":"finish"}')
    assert finished.accepted
    assert finished.execution is not None
    assert finished.final_execution
    assert len(executor.calls) == 1
    assert len(canvas.graph.nodes) == 1
    assert not canvas.graph.directed_edges
    assert not canvas.graph.bidirectional_edges
    assert not canvas.evaluate_flowsteer_structure().enabled


def test_structural_exploration_requires_connected_dynamic_multi_agent_graph() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        structural_exploration_required=True,
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Develop one task-relevant solution.",'
        '"scope":"Reason independently.","expected_output":"Return a candidate."}'
    ).accepted

    blocked = canvas.step('{"action":"set_output","target":"a"}')

    assert not blocked.accepted
    assert blocked.rejection_code == "structural_exploration_required"
    snapshot = canvas.control_snapshot()
    assert "set_output" not in snapshot["allowed_actions"]
    assert "finish" not in snapshot["allowed_actions"]
    assert snapshot["structural_exploration"] == {
        "required": True,
        "satisfied": False,
        "waived_for_budget": False,
        "minimum_connected_agents": 2,
    }

    assert canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"b","role":"Verifier",'
        '"objective":"Check the candidate independently.",'
        '"scope":"Find errors and resolve them.","expected_output":"Return a check."}'
    ).accepted
    assert canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"bidirectional"}'
    ).accepted

    satisfied = canvas.control_snapshot()
    assert satisfied["structural_exploration"]["satisfied"] is True
    assert "set_output" in satisfied["allowed_actions"]
    assert canvas.step('{"action":"set_output","target":"b"}').accepted
    assert canvas.step('{"action":"finish"}').accepted


def test_canvas_allows_twenty_graph_turns_and_finish_reuses_incremental_result() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(max_rounds=20),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return an answer."}'
    ).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"b","role":"Checker",'
        '"objective":"Check the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return a checked answer."}'
    ).accepted
    assert canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"bidirectional"}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"a"}').accepted
    calls_before_output_switches = len(executor.calls)
    for index in range(13):
        target = "b" if index % 2 == 0 else "a"
        step = canvas.step(f'{{"action":"set_output","target":"{target}"}}')
        assert step.accepted
        assert step.execution is None
    assert canvas.round_index == 19
    assert len(executor.calls) == calls_before_output_switches

    finished = canvas.step('{"action":"finish"}')

    assert finished.accepted
    assert finished.final_execution
    assert finished.execution is not None
    assert canvas.round_index == 20
    assert not canvas.active
    assert len(executor.calls) == calls_before_output_switches


def test_director_completes_mock_graph() -> None:
    director_backend = MockBackend(
        [
            '{"action":"add_agent","agent_id":"solver"}',
            '{"action":"set_prompt","target":"solver","role":"Independent solver",'
            '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
            '"expected_output":"Return a direct final answer."}',
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    result = GraphDirector(backend=director_backend, canvas=canvas).run()
    assert result.finished
    assert result.output.startswith("solver:Role: Independent solver")
    assert len(result.turns) == 3
    assert all(turn.accepted for turn in result.turns)
    assert director_backend.calls[0]["max_tokens"] == 1000
    assert director_backend.calls[0]["enable_thinking"] is None
    assert director_backend.calls[1]["max_tokens"] == 1000
    assert director_backend.calls[1]["enable_thinking"] is None
    assert [len(call["messages"]) for call in director_backend.calls] == [2, 4, 6]
    assert "Process signals: solver(" in director_backend.calls[2]["messages"][-1]["content"]
    assert all(
        sum(str(message["content"]).count("Task:\ntask") for message in call["messages"]) == 1
        for call in director_backend.calls
    )
    assert all(
        sum(
            str(message["content"]).count("Authoritative Canvas control snapshot:")
            for message in call["messages"]
        )
        == 1
        for call in director_backend.calls
    )
    assert director_backend.calls[2]["messages"][2]["content"] == (
        '{"action":"add_agent","agent_id":"solver"}'
    )
    assert director_backend.calls[2]["messages"][4]["content"].startswith(
        '{"action":"set_prompt","target":"solver"'
    )


def test_director_rejects_multiple_json_actions_and_preserves_raw_policy_output() -> None:
    first = (
        "I will solve the task myself in a long analysis that must not be trained.\n"
        "<think>private internal reasoning</think>\n"
        '{"action":"add_agent","agent_id":"solver"}\n'
        '{"action":"finish"}\ntrailing prose'
    )
    backend = MockBackend(
        [
            first,
            '{"action":"add_agent","agent_id":"solver"}',
            '{"action":"set_prompt","target":"solver","role":"Independent solver",'
            '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
            '"expected_output":"Return a direct final answer."}',
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert result.turns[0].model_action == '{"action":"invalid"}'
    assert result.turns[0].raw_action_text == first
    assert "solve the task myself" in str(result.to_dict())
    assert not result.turns[0].accepted
    assert result.turns[0].trainable
    diagnostics = result.turns[0].action_diagnostics
    assert diagnostics["json_objects_found"] == 2
    assert diagnostics["repair_attempted"] is False


def test_director_keeps_missing_json_as_a_rejected_policy_turn_without_hidden_retry() -> None:
    backend = MockBackend(
        [
            "A long analysis with no graph action.",
            '{"action":"add_agent","agent_id":"solver"}',
            '{"action":"set_prompt","target":"solver","role":"Independent solver",'
            '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
            '"expected_output":"Return a direct final answer."}',
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert len(result.turns) == 4
    assert len(backend.calls) == 4
    assert result.turns[0].model_action == '{"action":"invalid"}'
    assert result.turns[0].raw_action_text == "A long analysis with no graph action."
    assert result.turns[0].action_diagnostics["repair_attempted"] is False
    assert result.turns[0].trainable
    assert "A long analysis" in str(result.to_dict())


def test_worker_backend_failure_is_recorded_and_terminates_without_repair() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(BackendFailureExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted

    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Independent solver",'
        '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
        '"expected_output":"Return a direct final answer."}'
    )

    assert prompted.accepted
    assert "backend_failure=True" in prompted.feedback
    assert prompted.execution is not None
    assert prompted.rejection_code == "worker_backend_unavailable"
    assert canvas.state is CanvasState.FAILED
    assert not canvas.active
    output = canvas.step('{"action":"set_output","target":"solver"}')
    assert not output.accepted
    assert output.execution is None
    finished = canvas.step('{"action":"finish"}')
    assert not finished.accepted
    assert not finished.final_execution
    assert finished.execution is None
    assert not canvas.active


def test_retired_mace_canvas_injection_fails_closed() -> None:
    with pytest.raises(ValueError, match="MACE is retired"):
        GraphCanvas(
            task="task",
            runtime=MultiAgentRuntime(RecordingExecutor()),
            runtime_routes=("minimax", "grok"),
            model_router=MACEModelRouter(seed=0),
        )


def test_director_explicitly_selects_runtime() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent","agent_id":"solver"}',
            '{"action":"set_prompt","target":"solver","role":"Solver","objective":"Solve",'
            '"scope":"Question","expected_output":"Answer"}',
            '{"action":"set_model","target":"solver","runtime_route":"grok"}',
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        runtime_routes=("minimax", "grok"),
    )
    result = GraphDirector(backend=backend, canvas=canvas).run()
    assert result.finished
    assert result.graph["nodes"][0]["metadata"]["runtime_route"] == "grok"
    assert "SET_MODEL" in backend.calls[0]["messages"][0]["content"]
    assert any("grok" in str(call["messages"]) for call in backend.calls)


def test_aime_canvas_applies_dataset_actions_before_execution() -> None:
    executor = NumericRecordingExecutor()
    canvas = GraphCanvas(
        task="AIME problem",
        runtime=MultiAgentRuntime(executor),
        action_adapter=DatasetActionAdapter(
            adapter_id="aime",
            datasets=("aime",),
            action_names=("symbolic_compute", "finite_search", "python_exec"),
            initial_action_budget=3,
            revision_action_budget=1,
            total_action_budget=4,
        ),
    )

    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Independent solver",'
        '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
        '"expected_output":"Return a direct final answer."}'
    )
    assert prompted.accepted
    assert canvas.state is CanvasState.BUILDING
    assert len(executor.calls) == 1
    node = canvas.graph.nodes["solver"]
    assert node.allowed_tools == ("symbolic_compute", "finite_search", "python_exec")
    assert (node.initial_tool_budget, node.revision_tool_budget, node.total_tool_budget) == (
        3,
        1,
        4,
    )
    assert node.metadata["action_adapter"] == "aime"
    checkpoint = canvas.step('{"action":"set_output","target":"solver"}')
    assert checkpoint.accepted
    assert checkpoint.execution is None
    assert len(executor.calls) == 1
    finished = canvas.step('{"action":"finish"}')
    assert finished.accepted
    assert finished.final_execution
    assert len(executor.calls) == 1

    removed_director_action = canvas.step(
        '{"action":"set_operation_policy","target":"solver","allowed_tools":[]}'
    )
    assert not removed_director_action.accepted


def test_healthbench_canvas_configures_workers_with_no_actions_or_budget() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="Provide a clinically useful response.",
        runtime=MultiAgentRuntime(executor),
        action_adapter=DatasetActionAdapter(
            adapter_id="healthbench_professional",
            datasets=("healthbench_professional",),
            action_names=(),
            initial_action_budget=0,
            revision_action_budget=0,
            total_action_budget=0,
        ),
    )

    assert canvas.step('{"action":"add_agent","agent_id":"clinician"}').accepted
    prompted = canvas.step(
        '{"action":"set_prompt","target":"clinician",'
        '"role":"Clinical response drafter",'
        '"objective":"Answer the assigned healthcare request accurately",'
        '"scope":"Use the supplied conversation and assess safety considerations",'
        '"expected_output":"A concise clinically useful response"}'
    )

    assert prompted.accepted
    assert len(executor.calls) == 1
    node = canvas.graph.nodes["clinician"]
    assert node.allowed_tools == ()
    assert (node.initial_tool_budget, node.revision_tool_budget, node.total_tool_budget) == (
        0,
        0,
        0,
    )
    assert node.operation_policy_configured
    assert node.metadata["action_adapter"] == "healthbench_professional"
    assert node.metadata["dataset_capability_policy"] == {
        "environment_state": "stateless",
        "session_scope": "none",
        "action_execution": "batch",
        "commit_policy": "none",
        "commit_activation": "none",
        "supports_parallel_reads": True,
    }


def test_canvas_allows_task_focus_but_rejects_answer_solution_and_action_control() -> None:
    task = "Find the unique integer produced by the stated modular arithmetic construction."
    canvas = GraphCanvas(
        task=task,
        runtime=MultiAgentRuntime(RecordingExecutor()),
        action_adapter=DatasetActionAdapter(
            adapter_id="aime",
            datasets=("aime",),
            action_names=("python_exec", "finite_search"),
            initial_action_budget=3,
            revision_action_budget=1,
            total_action_budget=4,
        ),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    copied = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Solver",'
        f'"objective":"{task}","scope":"Reason carefully.",'
        '"expected_output":"Return a direct answer."}'
    )
    answer = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Solver",'
        '"objective":"Solve independently; final answer is 42.",'
        '"scope":"Reason carefully.","expected_output":"Return a direct answer."}'
    )
    action = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Solver",'
        '"objective":"Solve independently.","scope":"Use python_exec to calculate.",'
        '"expected_output":"Return a direct answer."}'
    )
    routing = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"MACE model router",'
        '"objective":"Select the best candidate model.","scope":"Rank runtime models.",'
        '"expected_output":"Return a model name."}'
    )
    assert [
        copied.rejection_code,
        answer.rejection_code,
        action.rejection_code,
        routing.rejection_code,
    ] == [
        None,
        "responsibility_violation",
        "responsibility_violation",
        "responsibility_violation",
    ]
    assert copied.accepted
    assert all(not step.accepted for step in (answer, action, routing))


def test_canvas_allows_natural_search_goal_but_rejects_search_action_control() -> None:
    canvas = GraphCanvas(
        task="When did Mount Fuji last erupt?",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        action_adapter=DatasetActionAdapter(
            adapter_id="retrieval_qa",
            datasets=("nq_open",),
            action_names=("search",),
            initial_action_budget=3,
            revision_action_budget=1,
            total_action_budget=4,
        ),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"researcher"}').accepted
    natural = canvas.step(
        '{"action":"set_prompt","target":"researcher",'
        '"role":"Historical researcher",'
        '"objective":"Determine the date of the last eruption of Mount Fuji",'
        '"scope":"Search geological and historical records for the confirmed event",'
        '"expected_output":"The eruption date with supporting evidence"}'
    )
    explicit = canvas.step(
        '{"action":"set_prompt","target":"researcher",'
        '"role":"Historical researcher",'
        '"objective":"Determine the eruption date",'
        '"scope":"Call the search tool",'
        '"expected_output":"The eruption date"}'
    )
    explicit_use = canvas.step(
        '{"action":"set_prompt","target":"researcher",'
        '"role":"Historical researcher",'
        '"objective":"Determine the eruption date",'
        '"scope":"Use search to find the event",'
        '"expected_output":"The eruption date"}'
    )

    assert natural.accepted
    assert not explicit.accepted
    assert not explicit_use.accepted
    assert explicit.rejection_code == "responsibility_violation"
    assert explicit_use.rejection_code == "responsibility_violation"


def test_canvas_rejects_concrete_math_solution_plan() -> None:
    canvas = GraphCanvas(
        task="Find the requested triangle length.",
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    planned = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Mathematician",'
        '"objective":"Calculate the requested triangle length",'
        '"scope":"Use the incenter and similar triangles to derive the ratio, then compute it",'
        '"expected_output":"The requested integer"}'
    )

    assert not planned.accepted
    assert planned.rejection_code == "responsibility_violation"
    assert "concrete solution procedure" in planned.feedback


def test_canvas_deterministically_compacts_only_overlong_delegation_fields() -> None:
    canvas = GraphCanvas(
        task="Find the requested AIME value.",
        dataset="aime",
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    long_scope = "Analyze the stated geometric configuration and constraints carefully. " * 8
    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Geometry analyst",'
        '"objective":"Determine the requested value","scope":'
        + json.dumps(long_scope)
        + ',"expected_output":"Return the requested value"}'
    )

    assert prompted.accepted
    assert {repair["field"] for repair in prompted.delegation_field_repairs} == {"scope"}
    node = canvas.graph.nodes["solver"]
    delegation = node.metadata["director_delegation"]
    assert delegation["role"] == "Geometry analyst"
    assert delegation["objective"] == "Determine the requested value"
    assert len(delegation["scope"]) <= 240
    assert delegation["expected_output"] == "Return the requested value"
    contract = node.metadata["system_managed_contract"]
    assert contract["version"] == "dataset-output-contract-v1"
    assert contract["dataset"] == "aime"
    assert "System-managed output contract" in node.prompt
    assert "python_exec" not in node.prompt


def test_canvas_appends_webshop_purchase_contract() -> None:
    canvas = GraphCanvas(
        task="Buy the requested product.",
        dataset="webshop",
        action_adapter=DatasetActionAdapter(
            adapter_id="webshop",
            datasets=("webshop",),
            action_names=("webshop_search", "webshop_click"),
            initial_action_budget=6,
            revision_action_budget=2,
            total_action_budget=8,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"shopper"}').accepted
    prompted = canvas.step(
        '{"action":"set_prompt","target":"shopper","role":"Product shopper",'
        '"objective":"Fulfill the requested shopping task",'
        '"scope":"Evaluate matching products and constraints",'
        '"expected_output":"Return the observed outcome"}'
    )

    assert prompted.accepted
    node = canvas.graph.nodes["shopper"]
    contract = node.metadata["system_managed_contract"]
    assert contract["dataset"] == "webshop"
    assert contract["version"] == "webshop-terminal-contract-v3-neutral"
    assert contract["rule_ids"] == [
        "public_catalog_search",
        "interpret_search_previews",
        "public_candidate_evidence",
        "product_option_state",
        "agent_purchase_authority",
        "complete_environment_purchase",
        "follow_live_subactions",
        "report_environment_outcome",
    ]
    assert "No purchase is automatically chosen" in node.prompt
    assert "no candidate or evidence-coverage order is selected" in node.prompt
    assert "The official environment determines the score" in node.prompt
    assert "joined/spaced spelling alternate" not in node.prompt
    assert "task-independent sibling evidence-coverage comparison" not in node.prompt
    assert "latest environment observation" in node.prompt
    assert "text is retained in webshop_progress" in node.prompt
    assert "latest live selected_options is authoritative" in node.prompt


def test_canvas_appends_alfworld_execution_contract() -> None:
    canvas = GraphCanvas(
        task="Put a cleaned apple in the fridge.",
        dataset="alfworld",
        action_adapter=DatasetActionAdapter(
            adapter_id="alfworld",
            datasets=("alfworld",),
            action_names=("alfworld_step",),
            initial_action_budget=50,
            revision_action_budget=50,
            total_action_budget=100,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"actor"}').accepted
    prompted = canvas.step(
        '{"action":"set_prompt","target":"actor","role":"Environment actor",'
        '"objective":"Complete the requested household task",'
        '"scope":"Act on the current environment observation",'
        '"expected_output":"Return the official environment outcome"}'
    )

    assert prompted.accepted
    node = canvas.graph.nodes["actor"]
    contract = node.metadata["system_managed_contract"]
    assert contract["dataset"] == "alfworld"
    assert contract["rule_ids"] == [
        "execute_environment_actions",
        "continue_until_environment_done",
        "follow_live_action_ids",
    ]
    assert "does not change the environment" in node.prompt
    assert "re-observing after every call" in node.prompt


def test_canvas_appends_swe_repository_patch_contract() -> None:
    canvas = GraphCanvas(
        task="Fix the reported repository issue.",
        dataset="swe_bench",
        action_adapter=DatasetActionAdapter(
            adapter_id="swe_bench",
            datasets=("swe_bench",),
            action_names=("swe_list", "swe_search", "swe_read", "swe_edit"),
            initial_action_budget=8,
            revision_action_budget=4,
            total_action_budget=12,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"worker"}').accepted
    prompted = canvas.step(
        '{"action":"set_prompt","target":"worker","role":"Bug analyst",'
        '"objective":"Determine the cause of the reported issue",'
        '"scope":"Inspect the affected repository behavior",'
        '"expected_output":"Return an analysis and proposed change"}'
    )

    assert prompted.accepted
    node = canvas.graph.nodes["worker"]
    contract = node.metadata["system_managed_contract"]
    assert contract["dataset"] == "swe_bench"
    assert contract["rule_ids"] == [
        "produce_repository_patch",
        "ground_changes_in_repository",
        "validate_workspace_change",
    ]
    assert "runtime-exported repository patch" in node.prompt
    assert "analysis or a proposed change in text alone" in node.prompt
    assert "available configured test profile" in node.prompt


def test_canvas_records_high_confidence_duplicate_swe_implementation_by_default() -> None:
    canvas = GraphCanvas(
        task="Implement missing distribution CDF methods.",
        dataset="swe_bench",
        action_adapter=DatasetActionAdapter(
            adapter_id="swe_bench",
            datasets=("swe_bench",),
            action_names=("swe_list", "swe_search", "swe_read", "swe_edit"),
            initial_action_budget=8,
            revision_action_budget=4,
            total_action_budget=12,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"developer"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"developer",'
        '"role":"Repository implementation owner",'
        '"objective":"Implement fixes for the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a repository code patch with corrected CDF methods"}'
    ).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"specialist"}').accepted
    recorded = canvas.step(
        '{"action":"set_prompt","target":"specialist",'
        '"role":"SymPy CDF implementation specialist",'
        '"objective":"Implement the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a code patch implementing corrected CDF methods"}'
    )

    assert recorded.accepted
    assert recorded.rejection_code is None
    assert recorded.responsibility_overlap_check["policy"] == "record_only"
    assert recorded.responsibility_overlap_check["decision"] == "record_only"
    comparison = recorded.responsibility_overlap_check["comparisons"][0]
    assert comparison["agent_id"] == "developer"
    assert comparison["high_confidence_duplicate"]
    assert canvas.graph.nodes["specialist"].configured


def test_canvas_reject_policy_preserves_hard_duplicate_responsibility_gate() -> None:
    canvas = GraphCanvas(
        task="Implement missing distribution CDF methods.",
        dataset="swe_bench",
        duplicate_responsibility_policy="reject",
        action_adapter=DatasetActionAdapter(
            adapter_id="swe_bench",
            datasets=("swe_bench",),
            action_names=("swe_list", "swe_search", "swe_read", "swe_edit"),
            initial_action_budget=8,
            revision_action_budget=4,
            total_action_budget=12,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"developer"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"developer",'
        '"role":"Repository implementation owner",'
        '"objective":"Implement fixes for the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a repository code patch with corrected CDF methods"}'
    ).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"specialist"}').accepted
    rejected = canvas.step(
        '{"action":"set_prompt","target":"specialist",'
        '"role":"SymPy CDF implementation specialist",'
        '"objective":"Implement the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a code patch implementing corrected CDF methods"}'
    )

    assert not rejected.accepted
    assert rejected.rejection_code == "duplicate_responsibility"
    assert rejected.responsibility_issue["details"]["conflicting_agent_id"] == "developer"
    assert rejected.responsibility_overlap_check["policy"] == "reject"
    assert rejected.responsibility_overlap_check["decision"] == "rejected"
    assert not canvas.graph.nodes["specialist"].configured


def test_canvas_warn_once_allows_duplicate_when_director_repeats_it() -> None:
    canvas = GraphCanvas(
        task="Implement missing distribution CDF methods.",
        dataset="swe_bench",
        duplicate_responsibility_policy="warn_once",
        action_adapter=DatasetActionAdapter(
            adapter_id="swe_bench",
            datasets=("swe_bench",),
            action_names=("swe_list", "swe_search", "swe_read", "swe_edit"),
            initial_action_budget=8,
            revision_action_budget=4,
            total_action_budget=12,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"developer"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"developer",'
        '"role":"Repository implementation owner",'
        '"objective":"Implement fixes for the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a repository code patch with corrected CDF methods"}'
    ).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"specialist"}').accepted
    duplicate_prompt = (
        '{"action":"set_prompt","target":"specialist",'
        '"role":"SymPy CDF implementation specialist",'
        '"objective":"Implement the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a code patch implementing corrected CDF methods"}'
    )

    warned = canvas.step(duplicate_prompt)
    accepted = canvas.step(duplicate_prompt)

    assert not warned.accepted
    assert warned.rejection_code == "duplicate_responsibility"
    assert warned.responsibility_overlap_check["decision"] == "rewrite_requested"
    assert accepted.accepted
    assert accepted.responsibility_overlap_check["decision"] == "accepted_after_warning"
    assert canvas.graph.nodes["specialist"].configured


def test_canvas_allows_distinct_swe_diagnosis_on_the_same_scope() -> None:
    canvas = GraphCanvas(
        task="Implement missing distribution CDF methods.",
        dataset="swe_bench",
        action_adapter=DatasetActionAdapter(
            adapter_id="swe_bench",
            datasets=("swe_bench",),
            action_names=("swe_list", "swe_search", "swe_read", "swe_edit"),
            initial_action_budget=8,
            revision_action_budget=4,
            total_action_budget=12,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"developer"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"developer",'
        '"role":"Repository implementation owner",'
        '"objective":"Implement fixes for the affected distribution CDF methods",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a repository code patch with corrected CDF methods"}'
    ).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"diagnostician"}').accepted
    distinct = canvas.step(
        '{"action":"set_prompt","target":"diagnostician",'
        '"role":"Root cause diagnostician",'
        '"objective":"Diagnose why the affected distribution behavior fails",'
        '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
        '"expected_output":"Return a root cause report with localized findings"}'
    )

    assert distinct.accepted
    assert distinct.responsibility_overlap_check["decision"] == "overlap_warning"


def test_director_duplicate_responsibility_retry_requires_distinct_contribution() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent","agent_id":"developer"}',
            '{"action":"set_prompt","target":"developer",'
            '"role":"Repository implementation owner",'
            '"objective":"Implement fixes for the affected distribution CDF methods",'
            '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
            '"expected_output":"Return a repository code patch with corrected CDF methods"}',
            '{"action":"add_agent","agent_id":"specialist"}',
            '{"action":"set_prompt","target":"specialist",'
            '"role":"SymPy CDF implementation specialist",'
            '"objective":"Implement the affected distribution CDF methods",'
            '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
            '"expected_output":"Return a code patch implementing corrected CDF methods"}',
            '{"action":"set_prompt","target":"specialist",'
            '"role":"Regression reviewer",'
            '"objective":"Review the affected behavior for regression risks",'
            '"scope":"SymPy probability distributions Arcsin Benini Beta BetaPrime",'
            '"expected_output":"Return an independent review with test findings"}',
        ]
    )
    canvas = GraphCanvas(
        task="Implement missing distribution CDF methods.",
        dataset="swe_bench",
        action_adapter=DatasetActionAdapter(
            adapter_id="swe_bench",
            datasets=("swe_bench",),
            action_names=("swe_list", "swe_search", "swe_read", "swe_edit"),
            initial_action_budget=8,
            revision_action_budget=4,
            total_action_budget=12,
        ),
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_rounds=5),
        duplicate_responsibility_policy="reject",
    )

    result = GraphDirector(backend=backend, canvas=canvas).run()

    duplicate = next(
        turn for turn in result.turns if turn.rejection_code == "duplicate_responsibility"
    )
    assert duplicate.trainable
    retry_prompt = str(backend.calls[4]["messages"])
    assert "genuinely distinct" in retry_prompt
    assert "Do not merely rename the role" in retry_prompt
    assert canvas.graph.nodes["specialist"].configured


def test_canvas_checks_semantic_leak_before_length_compaction() -> None:
    canvas = GraphCanvas(
        task="Find the requested value.",
        runtime=MultiAgentRuntime(RecordingExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    leaking_scope = (
        "Summarize all stated constraints without changing the task. " * 5
        + "Use the incenter and similar triangles to derive the ratio."
    )
    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Geometry analyst",'
        '"objective":"Determine the requested value","scope":'
        + json.dumps(leaking_scope)
        + ',"expected_output":"Return the requested value"}'
    )

    assert not prompted.accepted
    assert prompted.responsibility_issue == {
        "code": "concrete_solution_procedure",
        "field": "scope",
        "message": "SET_PROMPT prescribes a concrete solution procedure",
    }
    assert not prompted.delegation_field_repairs


def test_retrieval_contract_uses_explicit_dataset_not_shared_adapter_id() -> None:
    adapter = DatasetActionAdapter(
        adapter_id="retrieval_qa",
        datasets=("nq_open", "hotpotqa"),
        action_names=("search",),
        initial_action_budget=3,
        revision_action_budget=1,
        total_action_budget=4,
    )
    prompts = {}
    for dataset in ("nq_open", "hotpotqa"):
        canvas = GraphCanvas(
            task="Question",
            dataset=dataset,
            runtime=MultiAgentRuntime(RecordingExecutor()),
            action_adapter=adapter,
        )
        assert canvas.step('{"action":"add_agent","agent_id":"researcher"}').accepted
        assert canvas.step(
            '{"action":"set_prompt","target":"researcher",'
            '"role":"Researcher","objective":"Determine the requested fact",'
            '"scope":"Inspect relevant evidence",'
            '"expected_output":"Return the answer with evidence"}'
        ).accepted
        prompts[dataset] = canvas.graph.nodes["researcher"].prompt

    assert "concise answer span" in prompts["nq_open"]
    assert "connecting the relevant facts or entities" in prompts["hotpotqa"]


def test_director_rejections_remain_policy_turns_without_controller_graph_repair() -> None:
    original_task = "What is the requested AIME integer?"
    leaked = (
        '{"action":"set_prompt","target":"solver","role":"Solver",'
        '"objective":"What is the requested AIME integer? Final answer is 123.",'
        '"scope":"Use python_exec.","expected_output":"123"}'
    )
    backend = MockBackend(
        [
            '{"action":"add_agent","agent_id":"solver"}',
            leaked,
            leaked,
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(
        task="Question:\n" + original_task + "\n\nTrusted context documents:\nsecret evidence",
        director_task=original_task,
        runtime=MultiAgentRuntime(RecordingExecutor()),
        action_adapter=DatasetActionAdapter(
            adapter_id="aime",
            datasets=("aime",),
            action_names=("python_exec",),
            initial_action_budget=3,
            revision_action_budget=1,
            total_action_budget=4,
        ),
    )

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert not result.finished
    assert not result.output
    assert len(result.turns) == 5  # ADD_AGENT, three rejections, one final recovery.
    assert all(turn.trainable for turn in result.turns)
    assert all(turn.rejection_code == "responsibility_violation" for turn in result.turns[1:3])
    retry_messages = backend.calls[2]["messages"]
    assert original_task in str(retry_messages)
    # The rejected model output remains in the authentic policy history; the
    # Canvas retry instruction itself does not inject or repeat the answer.
    assert "123" in str(retry_messages)
    assert "123" not in retry_messages[-1]["content"]
    recovery_steps = [step for step in canvas.history if step.protocol_recovery]
    assert len(recovery_steps) == 1
    assert not recovery_steps[0].accepted
    assert recovery_steps[0].rejection_code == "director_no_progress_exhausted"


def test_director_does_not_recover_a_nonpending_prompt_as_none() -> None:
    invalid_rewrite = (
        '{"action":"set_prompt","target":"solver","role":"Solver",'
        '"objective":"Final answer is 42.","scope":"Reason carefully.",'
        '"expected_output":"Return 42."}'
    )
    backend = MockBackend(
        [
            '{"action":"add_agent","agent_id":"solver"}',
            '{"action":"set_prompt","target":"solver","role":"Independent solver",'
            '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
            '"expected_output":"Return a direct final answer."}',
            invalid_rewrite,
            invalid_rewrite,
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert len(result.turns) == 5
    assert "currently required SET_PROMPT for None" not in str(backend.calls)


def test_token_budget_failure_does_not_leave_executed_agent_dirty() -> None:
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_total_tokens=4),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted

    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Independent solver",'
        '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
        '"expected_output":"Return a direct final answer."}'
    )

    assert prompted.accepted
    assert "token budget exceeded (5/4)" in prompted.feedback
    assert prompted.execution is not None
    assert prompted.rejection_code == "execution_budget_exceeded"
    assert not prompted.final_execution
    assert canvas.state is CanvasState.FAILED
    assert not canvas.active
    assert canvas.dirty_agents == set()


def test_worker_backend_failure_stops_canvas_on_first_incremental_execution() -> None:
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(BackendFailureExecutor()),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted

    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Independent solver",'
        '"objective":"Solve the assigned task.","scope":"Reason and verify.",'
        '"expected_output":"Return a direct final answer."}'
    )

    assert prompted.accepted
    assert prompted.execution is not None
    assert prompted.rejection_code == "worker_backend_unavailable"
    assert "routes=unassigned" in prompted.feedback
    assert canvas.state is CanvasState.FAILED
    assert not canvas.active
    assert len(canvas.history) == 2


def test_canvas_reexecutes_only_dirty_subgraph_and_finish_uses_cache() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Evidence branch",'
        '"objective":"Determine one relevant fact.","scope":"Work independently.",'
        '"expected_output":"Return a concise finding."}'
    ).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"b","role":"Synthesizer",'
        '"objective":"Produce the assigned result.","scope":"Use visible inputs.",'
        '"expected_output":"Return a concise answer."}'
    ).accepted
    assert canvas.step('{"action":"set_layer","target":"b","layer":1}').accepted

    relation = canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"directed"}'
    )

    assert relation.accepted
    assert relation.execution is not None
    assert relation.execution.executed_agents == ["b"]
    assert relation.execution.reused_agents == ["a"]
    assert executor.calls[-1]["agent_id"] == "b"
    assert executor.calls[-1]["upstream"] == ["a"]
    assert canvas.step('{"action":"set_output","target":"b"}').execution is None
    calls_before_finish = len(executor.calls)
    finished = canvas.step('{"action":"finish"}')
    assert finished.accepted and finished.final_execution
    assert len(executor.calls) == calls_before_finish
    assert set(finished.execution.reused_agents) == {"a", "b"}


def test_structural_repair_blocks_graph_growth_and_preserves_dirty_execution() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Synthesizer",'
        '"objective":"Produce the assigned result.","scope":"Use visible inputs.",'
        '"expected_output":"Return a concise answer."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"a"}').accepted
    assert canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    configured = canvas.step(
        '{"action":"set_prompt","target":"b","role":"Evidence branch",'
        '"objective":"Determine one relevant fact.","scope":"Work independently.",'
        '"expected_output":"Return a concise finding."}'
    )
    assert configured.accepted
    assert configured.structural_repair["reason"] == "output_reachability"
    calls_before_block = len(executor.calls)

    blocked = canvas.step('{"action":"add_agent","agent_id":"c"}')

    assert not blocked.accepted
    assert blocked.rejection_code == "structural_repair_required"
    assert set(canvas.graph.nodes) == {"a", "b"}
    assert len(executor.calls) == calls_before_block
    layered = canvas.step('{"action":"set_layer","target":"a","layer":1}')
    assert layered.accepted
    assert layered.execution is not None
    assert layered.execution.executed_agents == ["a"]
    assert layered.structural_repair["required"] is True
    relation = canvas.step(
        '{"action":"set_relation","source":"b","target":"a","relation":"directed"}'
    )
    assert relation.accepted
    assert relation.execution is not None
    assert relation.execution.executed_agents == ["a"]
    assert relation.execution.reused_agents == ["b"]
    assert relation.structural_repair["transition"] == "resolved"
    assert relation.structural_repair["required"] is False
    trace = trace_from_canvas(
        run_id="repair-trace",
        task=TaskSpec("task-1", "task"),
        canvas=canvas,
    )
    assert trace.events[-1].payload["topology_audit"]["all_agents_reach_output"]
    assert trace.events[-1].payload["structural_repair"]["transition"] == "resolved"
    assert canvas.step('{"action":"add_agent","agent_id":"c"}').accepted


def test_relation_layer_failure_requires_repair_of_existing_pair() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    for agent_id in ("a", "b"):
        assert canvas.step(f'{{"action":"add_agent","agent_id":"{agent_id}"}}').accepted
        assert canvas.step(
            f'{{"action":"set_prompt","target":"{agent_id}","role":"Analyst",'
            '"objective":"Solve the assigned task.","scope":"Reason independently.",'
            '"expected_output":"Return a concise finding."}'
        ).accepted

    rejected = canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"directed"}'
    )
    assert not rejected.accepted
    assert rejected.rejection_code == "relation_layer_mismatch"
    assert rejected.structural_repair["reason"] == "relation_layer_mismatch"
    assert rejected.structural_repair["relation"] == "directed"
    assert "set b to layer 1" in rejected.structural_repair["guidance"]
    assert "set b to layer 1" in rejected.feedback
    blocked = canvas.step('{"action":"add_agent","agent_id":"c"}')
    assert not blocked.accepted
    assert blocked.rejection_code == "structural_repair_required"
    assert canvas.step('{"action":"set_layer","target":"b","layer":1}').accepted
    repaired = canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"directed"}'
    )
    assert repaired.accepted
    assert repaired.structural_repair["transition"] == "resolved"
    assert repaired.structural_repair["relation_layer_rejections_total"] == 1


def test_finish_without_output_locks_growth_until_output_is_selected() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return a concise answer."}'
    ).accepted
    rejected = canvas.step('{"action":"finish"}')
    assert not rejected.accepted
    assert rejected.rejection_code == "output_selection_required"
    assert not canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    selected = canvas.step('{"action":"set_output","target":"a"}')
    assert selected.accepted
    assert selected.structural_repair["transition"] == "resolved"
    assert canvas.step('{"action":"finish"}').accepted


def test_empty_graph_finish_rejection_keeps_add_agent_recoverable() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))

    rejected = canvas.step('{"action":"finish"}')

    assert not rejected.accepted
    assert canvas.structural_repair_reason is None
    assert canvas.control_snapshot()["allowed_actions"] == ["add_agent"]


def test_low_worker_budget_blocks_graph_growth_and_finishes_cached_agent() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(
            max_total_tokens=20,
            graph_growth_token_reserve=8,
        ),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return a concise answer."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"a"}').accepted
    canvas.total_tokens = 13
    calls_before = len(executor.calls)

    blocked = canvas.step('{"action":"add_agent","agent_id":"b"}')

    assert not blocked.accepted
    assert blocked.rejection_code == "token_budget_consolidation_required"
    assert blocked.structural_repair["reason"] == "token_budget_consolidation"
    assert blocked.structural_repair["allowed_actions"] == ["finish"]
    assert canvas.control_snapshot()["allowed_actions"] == ["finish"]
    assert set(canvas.graph.nodes) == {"a"}
    assert len(executor.calls) == calls_before
    finished = canvas.step('{"action":"finish"}')
    assert finished.accepted
    assert finished.structural_repair["transition"] == "resolved"
    assert len(executor.calls) == calls_before


def test_consolidation_latches_across_output_selection_until_finish() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(max_total_tokens=20, graph_growth_token_reserve=8),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return a concise answer."}'
    ).accepted
    canvas.total_tokens = 13

    blocked = canvas.step('{"action":"add_agent","agent_id":"b"}')

    assert not blocked.accepted
    assert blocked.structural_repair["reason"] == "token_budget_consolidation"
    assert canvas.control_snapshot()["allowed_actions"] == ["set_output"]

    selected = canvas.step('{"action":"set_output","target":"a"}')

    assert selected.accepted
    assert selected.structural_repair["reason"] == "token_budget_consolidation"
    assert selected.structural_repair["transition"] != "resolved"
    assert canvas.control_snapshot()["allowed_actions"] == ["finish"]
    rejected_growth = canvas.step('{"action":"add_agent","agent_id":"b"}')
    assert not rejected_growth.accepted
    assert rejected_growth.rejection_code == "token_budget_consolidation_required"
    assert canvas.step('{"action":"finish"}').accepted
    assert canvas.state is CanvasState.FINISHED


def test_repeated_same_output_is_noop_and_enters_bounded_recovery() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    version = canvas.graph.version

    repeated = [canvas.step('{"action":"set_output","target":"solver"}') for _ in range(3)]

    assert not any(step.accepted for step in repeated)
    assert [step.rejection_code for step in repeated] == ["output_already_selected"] * 3
    assert repeated[0].rejection_details["legal_recovery_actions"] == [{"action": "finish"}]
    assert canvas.graph.version == version
    assert canvas.topology_edits_frozen
    recovered = canvas.recover_frozen_topology()
    assert recovered[-1].accepted
    assert canvas.state is CanvasState.FINISHED


def test_repeated_effective_prompt_is_rejected_without_dirty_execution() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    prompt_action = (
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    )
    assert canvas.step(prompt_action).accepted
    version = canvas.graph.version
    calls = len(executor.calls)

    repeated = canvas.step(prompt_action)

    assert not repeated.accepted
    assert repeated.rejection_code == "no_effective_change"
    assert repeated.rejection_details["change_reason"] == "no_effective_change"
    assert repeated.rejection_details["mutation_changed"] is False
    assert repeated.execution is None
    assert repeated.dirty_agents == []
    assert canvas.graph.version == version
    assert len(executor.calls) == calls


def test_repeated_layer_is_rejected_without_dirty_execution() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    version = canvas.graph.version
    calls = len(executor.calls)

    repeated = canvas.step('{"action":"set_layer","target":"solver","layer":0}')

    assert not repeated.accepted
    assert repeated.rejection_code == "no_effective_change"
    assert repeated.execution is None
    assert repeated.dirty_agents == []
    assert canvas.graph.version == version
    assert len(executor.calls) == calls


def test_canvas_step_exposes_dirty_lifecycle_and_worker_call_counts() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))

    added = canvas.step('{"action":"add_agent","agent_id":"solver"}')

    assert added.invalidated_agents == ["solver"]
    assert added.scheduled_agents == []
    assert added.executed_agents == []
    assert added.reused_agents == []
    assert added.remaining_dirty == ["solver"]
    assert added.invalidation_reasons == {"solver": ["new_agent_initial"]}

    prompted = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    )

    assert prompted.invalidated_agents == ["solver"]
    assert prompted.scheduled_agents == ["solver"]
    assert prompted.executed_agents == ["solver"]
    assert prompted.reused_agents == []
    assert prompted.remaining_dirty == []
    assert prompted.invalidation_reasons == {"solver": ["prompt_changed"]}
    assert prompted.execution is not None
    assert prompted.execution.invalidation_reasons == {
        "solver": ["new_agent_initial", "prompt_changed"]
    }
    assert prompted.execution.initial_model_calls == 1
    assert prompted.execution.revision_model_calls == 0
    assert prompted.execution.worker_model_calls_total == 1
    assert prompted.execution.cache_hits == 0
    assert prompted.execution.component_execution_count == 1
    event = prompted.execution.execution_events[0]
    assert event["agent_id"] == "solver"
    assert event["phase"] == "initial"
    assert event["cache_hit"] is False
    assert event["reason_codes"] == ["new_agent_initial", "prompt_changed"]
    assert len(event["input_hash"]) == 64
    assert event["peer_packet_count"] == 0
    assert event["revision_wave"] == 0
    payload = prompted.to_dict()
    assert payload["invalidated_agents"] == ["solver"]
    assert payload["scheduled_agents"] == ["solver"]
    assert payload["executed_agents"] == ["solver"]
    assert payload["remaining_dirty"] == []
    assert payload["execution"]["worker_model_calls_total"] == 1


def test_prompt_revision_without_new_canvas_evidence_is_rejected_before_execution() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    version = canvas.graph.version
    calls = len(executor.calls)

    rejected = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Verifier",'
        '"objective":"Recheck the task.","scope":"Verify independently.",'
        '"expected_output":"Return a checked finding."}'
    )

    assert not rejected.accepted
    assert rejected.rejection_code == "prompt_revision_evidence_required"
    assert rejected.rejection_details["eligible_revision_evidence"] == {}
    assert rejected.execution is None
    assert rejected.invalidated_agents == []
    assert canvas.graph.version == version
    assert len(executor.calls) == calls
    snapshot = canvas.control_snapshot()
    assert snapshot["legal_action_parameters"]["set_prompt"]["targets"] == []
    assert "set_prompt" not in snapshot["allowed_actions"]


def test_prompt_revision_consumes_unresolved_issue_evidence_once() -> None:
    class UnresolvedExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.unresolved_issues = ["independent verification remains"]
            return artifact

    executor = UnresolvedExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    evidence = canvas.control_snapshot()["legal_action_parameters"]["set_prompt"]
    assert evidence["targets"] == ["solver"]
    assert evidence["revision_evidence_by_target"]["solver"]["unresolved_issue"][
        "evidence_agent_ids"
    ] == ["solver"]

    revised = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Verifier",'
        '"objective":"Recheck the task.","scope":"Resolve remaining uncertainty.",'
        '"expected_output":"Return a checked finding.",'
        '"revision_basis":"unresolved_issue",'
        '"evidence_agent_ids":["solver"]}'
    )

    assert revised.accepted
    assert revised.prompt_revision == {
        "target": "solver",
        "revision_index": 1,
        "basis": "unresolved_issue",
        "evidence_agent_ids": ["solver"],
        "evidence_signature_count": 1,
        "target_model_calls": 1,
        "target_cache_hits": 0,
        "worker_model_calls_total": 1,
    }
    trace = trace_from_canvas(
        run_id="prompt-revision-trace",
        task=TaskSpec("task-1", "task"),
        canvas=canvas,
    )
    revision_payload = trace.events[-1].payload
    assert revision_payload["prompt_revision"] == revised.prompt_revision
    assert revision_payload["invalidated_agents"] == ["solver"]
    assert revision_payload["scheduled_agents"] == ["solver"]
    assert revision_payload["executed_agents"] == ["solver"]
    assert revision_payload["remaining_dirty"] == []
    assert canvas.prompt_revision_counts == {"solver": 1}
    calls = len(executor.calls)
    version = canvas.graph.version

    repeated = canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Reviewer",'
        '"objective":"Review the task again.","scope":"Resolve remaining uncertainty.",'
        '"expected_output":"Return another checked finding.",'
        '"revision_basis":"unresolved_issue",'
        '"evidence_agent_ids":["solver"]}'
    )

    assert not repeated.accepted
    assert repeated.rejection_code == "prompt_revision_evidence_required"
    assert canvas.graph.version == version
    assert len(executor.calls) == calls


def test_prompt_revision_accepts_new_upstream_artifact_evidence() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    for agent_id, role in (("source", "Analyst"), ("target", "Synthesizer")):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": role,
                    "objective": "Solve one relevant part of the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a concise finding.",
                }
            )
        ).accepted
    assert canvas.step('{"action":"set_layer","target":"target","layer":1}').accepted
    assert canvas.step(
        '{"action":"set_relation","source":"source","target":"target","relation":"directed"}'
    ).accepted
    evidence = canvas.control_snapshot()["legal_action_parameters"]["set_prompt"]
    assert evidence["revision_evidence_by_target"]["target"]["upstream_artifact_changed"][
        "evidence_agent_ids"
    ] == ["source"]

    revised = canvas.step(
        '{"action":"set_prompt","target":"target","role":"Synthesizer",'
        '"objective":"Reassess the task from the available findings.",'
        '"scope":"Integrate the upstream contribution.",'
        '"expected_output":"Return a consolidated finding.",'
        '"revision_basis":"upstream_artifact_changed",'
        '"evidence_agent_ids":["source"]}'
    )

    assert revised.accepted
    assert revised.prompt_revision["basis"] == "upstream_artifact_changed"
    assert revised.prompt_revision["evidence_agent_ids"] == ["source"]
    assert revised.prompt_revision["target_model_calls"] == 1


def test_repeated_output_recovery_deletes_unreachable_agent_before_finish() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    for agent_id in ("output", "extra"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    assert canvas.step('{"action":"set_output","target":"output"}').accepted
    canvas._enter_structural_repair("token_budget_consolidation")

    repeated = [canvas.step('{"action":"set_output","target":"output"}') for _ in range(3)]

    assert not any(step.accepted for step in repeated)
    assert canvas.topology_edits_frozen
    assert canvas.control_snapshot()["allowed_actions"] == ["delete_agent"]
    recovered = canvas.recover_frozen_topology()
    assert [step.action.action_type for step in recovered] == [
        ActionType.DELETE_AGENT,
        ActionType.FINISH,
    ]
    assert all(step.accepted and step.protocol_recovery for step in recovered)
    assert set(canvas.graph.nodes) == {"output"}
    assert canvas.state is CanvasState.FINISHED


def test_consolidation_allows_one_director_output_choice_then_controller_closes() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    for agent_id in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    canvas._enter_structural_repair("time_budget_consolidation")
    backend = MockBackend(
        [
            '{"action":"set_output","target":"b"}',
            '{"action":"delete_agent","target":"a"}',
        ]
    )

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert len(backend.calls) == 2
    assert len(result.turns) == 2
    assert result.turns[0].model_action == '{"action":"set_output","target":"b"}'
    assert [step.action.action_type for step in canvas.history if step.protocol_recovery][-1:] == [
        ActionType.FINISH
    ]
    assert canvas.graph.output_agent == "b"


def test_consolidation_excludes_protocol_failure_and_switches_sole_candidate() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    for agent_id in ("usable", "failed"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    assert canvas.step('{"action":"set_output","target":"failed"}').accepted
    failed_artifact = canvas.runtime.artifacts["failed"]
    canvas.runtime.artifacts["failed"] = AgentArtifact(
        artifact_id=failed_artifact.artifact_id,
        agent_id="failed",
        answer="WORKER_PROTOCOL_FAILURE",
        confidence=1.0,
    )
    canvas._enter_structural_repair("time_budget_consolidation")

    snapshot = canvas.control_snapshot()
    assert snapshot["legal_action_parameters"]["set_output"]["targets"] == ["usable"]
    backend = MockBackend(
        [
            '{"action":"set_output","target":"usable"}',
            '{"action":"delete_agent","target":"failed"}',
        ]
    )
    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert len(backend.calls) == 2
    assert canvas.graph.output_agent == "usable"
    assert result.output != "WORKER_PROTOCOL_FAILURE"


def test_consolidation_without_usable_artifact_fails_closed_without_model_call() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"failed"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"failed","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    failed_artifact = canvas.runtime.artifacts["failed"]
    canvas.runtime.artifacts["failed"] = AgentArtifact(
        artifact_id=failed_artifact.artifact_id,
        agent_id="failed",
        answer="WORKER_PROTOCOL_FAILURE",
    )
    canvas._enter_structural_repair("token_budget_consolidation")
    backend = MockBackend([])

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert not result.finished
    assert backend.calls == []
    assert canvas.state is CanvasState.FAILED
    assert canvas.history[-1].rejection_code == "no_usable_output_artifact"
    assert canvas.history[-1].protocol_recovery


def test_consolidation_semantic_no_progress_history_is_bounded_and_freezes() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    for agent_id in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    canvas._enter_structural_repair("time_budget_consolidation")

    rejected = [
        canvas.step(
            '{"action":"set_output","target":"missing"}',
            authoritative_director=True,
        )
        for _ in range(3)
    ]

    assert not any(step.accepted for step in rejected)
    assert canvas.semantic_no_progress_streak == 2
    assert canvas.topology_edits_frozen
    repair = canvas.control_snapshot()["repair_progress"]
    assert len(repair["recent_actions"]) == 3
    assert repair["recent_actions"][-1]["semantic_progress"] is False


def test_director_forces_finish_only_without_another_model_call() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    canvas._enter_structural_repair("token_budget_consolidation")
    assert canvas.control_snapshot()["allowed_actions"] == ["finish"]
    backend = MockBackend([])

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert backend.calls == []
    assert canvas.history[-1].action.action_type is ActionType.FINISH
    assert canvas.history[-1].accepted
    assert canvas.history[-1].protocol_recovery


def test_repair_gate_fuses_different_actions_against_same_canvas_state() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    canvas._enter_structural_repair("token_budget_consolidation")
    version = canvas.graph.version

    rejected = [
        canvas.step('{"action":"set_output","target":"solver"}'),
        canvas.step('{"action":"add_agent","agent_id":"extra"}'),
        canvas.step('{"action":"set_output","target":"solver"}'),
    ]

    assert not any(step.accepted for step in rejected)
    assert [step.rejection_code for step in rejected] == [
        "structural_repair_required",
        "structural_repair_required",
        "structural_repair_required",
    ]
    assert [step.invalid_repeat_count for step in rejected] == [1, 2, 3]
    assert canvas.graph.version == version
    assert canvas.topology_edits_frozen
    recovered = canvas.recover_frozen_topology()
    assert recovered[-1].accepted
    assert canvas.state is CanvasState.FINISHED


def test_director_round_limit_deterministically_finishes_existing_graph() -> None:
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_rounds=3),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    assert not canvas.active
    backend = MockBackend([])

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert backend.calls == []
    assert canvas.state is CanvasState.FINISHED
    assert canvas.history[-1].protocol_recovery


def test_round_limit_fails_without_discarding_director_graph_content() -> None:
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_rounds=4),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    assert canvas.step('{"action":"add_agent","agent_id":"unfinished"}').accepted
    assert canvas.pending_agent_id == "unfinished"
    assert not canvas.active

    result = GraphDirector(backend=MockBackend([]), canvas=canvas).run()

    assert not result.finished
    assert canvas.pending_agent_id == "unfinished"
    assert set(canvas.graph.nodes) == {"solver", "unfinished"}
    assert canvas.history[-1].protocol_recovery
    assert canvas.history[-1].rejection_code == "max_rounds_exhausted"


def test_selected_output_recovery_reruns_only_target_in_bidirectional_component() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(executor))
    for agent_id in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    assert canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"bidirectional"}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"a"}').accepted
    executor.calls.clear()

    recovered = canvas.recover_selected_output_agent(reason_code="aime_terminal_tool_failure")

    assert recovered.accepted and recovered.protocol_recovery
    assert recovered.execution is not None
    assert recovered.execution.executed_agents == ["a"]
    assert recovered.execution.reused_agents == ["b"]
    assert [call["agent_id"] for call in executor.calls] == ["a"]
    assert executor.calls[0]["revision"] is True
    assert executor.calls[0]["prior"] == "a"
    assert executor.calls[0]["peers"] == ["b"]


def test_selected_output_recovery_does_not_read_transitive_bidirectional_peer() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(max_rounds=12),
    )
    for agent_id in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    assert canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"bidirectional"}'
    ).accepted
    assert canvas.step(json.dumps({"action": "add_agent", "agent_id": "c"})).accepted
    assert canvas.step(
        json.dumps(
            {
                "action": "set_prompt",
                "target": "c",
                "role": "Analyst",
                "objective": "Solve the task.",
                "scope": "Reason independently.",
                "expected_output": "Return a finding.",
            }
        )
    ).accepted
    assert canvas.step(
        '{"action":"set_relation","source":"b","target":"c","relation":"bidirectional"}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"a"}').accepted
    executor.calls.clear()

    recovered = canvas.recover_selected_output_agent(reason_code="aime_terminal_tool_failure")

    assert recovered.accepted and recovered.protocol_recovery
    assert recovered.execution is not None
    assert recovered.execution.executed_agents == ["a"]
    assert set(recovered.execution.reused_agents) == {"b", "c"}
    assert [call["agent_id"] for call in executor.calls] == ["a"]
    assert executor.calls[0]["revision"] is True
    assert executor.calls[0]["prior"] == "a"
    assert executor.calls[0]["peers"] == ["b"]


def test_director_round_limit_returns_typed_failure_when_nothing_is_recoverable() -> None:
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_rounds=0),
    )
    backend = MockBackend([])

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert not result.finished
    assert backend.calls == []
    assert canvas.state is CanvasState.FAILED
    assert canvas.history[-1].rejection_code == "max_rounds_exhausted"
    assert canvas.history[-1].protocol_recovery
    assert canvas.history[-1].control_snapshot["allowed_actions"] == []


def test_relation_repair_reserves_layer_relation_and_finish_tokens() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(
            max_total_tokens=24,
            graph_growth_token_reserve=0,
            worker_token_min_samples=1,
            worker_token_cold_start=8,
            finalization_token_reserve=5,
        ),
    )
    prompt_a = (
        '"role":"Analyst","objective":"Solve the assigned task.",'
        '"scope":"Reason independently.","expected_output":"Return a finding."}'
    )
    prompt_b = (
        '"role":"Verifier","objective":"Check the assigned task.",'
        '"scope":"Reason independently.","expected_output":"Return a verification."}'
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step('{"action":"set_prompt","target":"a",' + prompt_a).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    assert canvas.step('{"action":"set_prompt","target":"b",' + prompt_b).accepted
    rejected_relation = canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"directed"}'
    )
    assert rejected_relation.rejection_code == "relation_layer_mismatch"
    calls_before = len(executor.calls)

    blocked_layer = canvas.step('{"action":"set_layer","target":"b","layer":1}')

    assert not blocked_layer.accepted
    assert blocked_layer.rejection_code == "token_budget_admission_required"
    assert blocked_layer.token_admission["estimation_mode"] == (
        "current_edit_plus_pending_relation"
    )
    assert blocked_layer.token_admission["estimated_worker_tokens"] == 10
    assert blocked_layer.token_admission["finalization_token_reserve"] == 5
    assert blocked_layer.token_admission["required_tokens"] == 15
    assert blocked_layer.token_admission["remaining_worker_tokens"] == 14
    assert canvas.graph.nodes["b"].layer == 0
    assert len(executor.calls) == calls_before


def test_token_admission_covers_delete_dirty_closure_before_commit() -> None:
    executor = RecordingExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(
            max_total_tokens=29,
            graph_growth_token_reserve=0,
            worker_token_min_samples=1,
            finalization_token_reserve=5,
        ),
    )
    prompt_a = (
        '"role":"Analyst","objective":"Solve the assigned task.",'
        '"scope":"Reason independently.","expected_output":"Return a finding."}'
    )
    prompt_b = (
        '"role":"Verifier","objective":"Check the assigned task.",'
        '"scope":"Verify independently.","expected_output":"Return a verification."}'
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step('{"action":"set_prompt","target":"a",' + prompt_a).accepted
    assert canvas.step('{"action":"add_agent","agent_id":"b"}').accepted
    assert canvas.step('{"action":"set_prompt","target":"b",' + prompt_b).accepted
    assert canvas.step('{"action":"set_layer","target":"b","layer":1}').accepted
    assert canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"directed"}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"b"}').accepted
    calls_before = len(executor.calls)

    blocked = canvas.step('{"action":"delete_agent","target":"a"}')

    assert not blocked.accepted
    assert blocked.rejection_code == "token_budget_admission_required"
    assert blocked.token_admission["estimated_worker_tokens"] == 5
    assert blocked.token_admission["required_tokens"] == 10
    assert blocked.token_admission["remaining_worker_tokens"] == 9
    assert set(canvas.graph.nodes) == {"a", "b"}
    assert len(executor.calls) == calls_before


def test_token_admission_uses_observed_agent_cost_after_route_cold_start() -> None:
    class ExpensiveExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.token_in = 15
            artifact.token_out = 2
            artifact.unresolved_issues = ["independent verification remains"]
            return artifact

    executor = ExpensiveExecutor()
    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(
            max_total_tokens=35,
            graph_growth_token_reserve=0,
            worker_token_min_samples=99,
            worker_token_cold_start=4,
            finalization_token_reserve=5,
        ),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return a concise answer."}'
    ).accepted
    calls_before = len(executor.calls)

    blocked = canvas.step(
        '{"action":"set_prompt","target":"a","role":"Verifier",'
        '"objective":"Recheck the assigned task.","scope":"Verify independently.",'
        '"expected_output":"Return a checked answer.",'
        '"revision_basis":"unresolved_issue","evidence_agent_ids":["a"]}'
    )

    assert not blocked.accepted
    assert blocked.rejection_code == "token_budget_admission_required"
    assert blocked.token_admission["estimated_worker_tokens"] == 17
    assert blocked.token_admission["required_tokens"] == 22
    assert blocked.token_admission["remaining_worker_tokens"] == 18
    assert blocked.token_admission["routes"][0]["source"] == "observed_agent_floor"
    assert blocked.token_admission["routes"][0]["route_estimate_tokens"] == 4
    assert len(executor.calls) == calls_before


def test_canvas_bounds_worker_feedback_without_losing_budget_tail() -> None:
    class LongSummaryExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.summary = "X" * 5000
            artifact.answer = "Y" * 5000
            return artifact

    canvas = GraphCanvas(
        task="task",
        runtime=MultiAgentRuntime(LongSummaryExecutor()),
        config=CanvasConfig(feedback_max_chars=500, artifact_summary_max_chars=100),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"a"}').accepted
    prompted = canvas.step(
        '{"action":"set_prompt","target":"a","role":"Analyst",'
        '"objective":"Solve the assigned task.","scope":"Reason independently.",'
        '"expected_output":"Return an answer."}'
    )

    assert prompted.accepted
    assert len(prompted.feedback) <= 500
    assert "[truncated]" in prompted.feedback
    assert "cumulative Worker tokens=" in prompted.feedback


def test_canvas_rejects_stale_version_and_preserves_atomic_snapshot() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    added = canvas.step('{"action":"add_agent","expected_version":0}')
    assert added.accepted
    assert canvas.graph.version == 1
    before = canvas.graph.to_dict()

    stale = canvas.step(
        '{"action":"set_prompt","target":"agent_1","expected_version":0,'
        '"role":"Analyst","objective":"Solve the task.",'
        '"scope":"Reason independently.","expected_output":"Return a finding."}'
    )

    assert not stale.accepted
    assert stale.rejection_code == "stale_canvas_version"
    assert canvas.graph.to_dict() == before
    assert stale.rejection_details["current_version"] == 1
    assert stale.rejection_details["legal_agent_ids"] == ["agent_1"]


def test_director_binds_current_version_instead_of_trusting_model_metadata() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent","agent_id":"solver","expected_version":999}',
            '{"action":"set_prompt","target":"solver","role":"Analyst",'
            '"objective":"Solve the task.","scope":"Reason independently.",'
            '"expected_output":"Return a finding.","expected_version":999}',
            '{"action":"set_output","target":"solver","expected_version":999}',
        ]
    )
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert result.finished
    assert len(backend.calls) == 3
    assert not any(turn.rejection_code == "stale_canvas_version" for turn in result.turns)
    model_steps = [step for step in canvas.history if not step.protocol_recovery]
    assert [step.action.expected_version for step in model_steps] == [0, 1, 2]
    assert [turn.action_diagnostics["model_expected_version"] for turn in result.turns] == [
        999,
        999,
        999,
    ]


def test_control_snapshot_exposes_parameter_level_legal_surface() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted

    snapshot = canvas.control_snapshot()

    assert snapshot["legal_action_parameters"]["set_output"]["targets"] == ["solver"]
    assert snapshot["legal_action_parameters"]["set_prompt"]["targets"] == []
    assert "set_prompt" not in snapshot["allowed_actions"]
    assert snapshot["legal_action_parameters"]["set_relation"]["relations"] == []
    assert "set_relation" not in snapshot["allowed_actions"]
    assert snapshot["legal_action_parameters"]["finish"]["ready"] is False
    assert "finish" not in snapshot["allowed_actions"]

    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    selected = canvas.control_snapshot()
    assert selected["legal_action_parameters"]["set_output"]["targets"] == []
    assert "set_output" not in selected["allowed_actions"]
    assert selected["legal_action_parameters"]["finish"]["ready"] is True


def test_director_preflight_rejects_schema_valid_illegal_relation_tuple() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    for agent_id in ("a", "b"):
        assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent_id})).accepted
        assert canvas.step(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": agent_id,
                    "role": "Analyst",
                    "objective": "Solve the task.",
                    "scope": "Reason independently.",
                    "expected_output": "Return a finding.",
                }
            )
        ).accepted
    snapshot = canvas.control_snapshot()
    assert {
        "source": "a",
        "target": "b",
        "relation": "bidirectional",
    } in snapshot["legal_action_parameters"]["set_relation"]["relations"]
    version = canvas.graph.version

    rejected = canvas.step(
        '{"action":"set_relation","source":"a","target":"b","relation":"directed"}',
        authoritative_director=True,
    )

    assert not rejected.accepted
    assert rejected.rejection_code == "director_parameter_not_allowed"
    assert canvas.graph.version == version
    assert not canvas.graph.directed_edges
    assert not canvas.graph.bidirectional_edges


def test_unknown_agent_is_not_mislabeled_as_responsibility_violation() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent"}').accepted

    rejected = canvas.step(
        '{"action":"set_prompt","target":"agent_0",'
        '"role":"Analyst","objective":"Solve the task.",'
        '"scope":"Reason independently.","expected_output":"Return a finding."}'
    )

    assert not rejected.accepted
    assert rejected.rejection_code == "unknown_agent"
    assert rejected.responsibility_issue == {}
    assert rejected.rejection_details["legal_agent_ids"] == ["agent_1"]


def test_repeated_invalid_mutation_freezes_edits_but_keeps_existing_output() -> None:
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))
    assert canvas.step('{"action":"add_agent","agent_id":"solver"}').accepted
    assert canvas.step(
        '{"action":"set_prompt","target":"solver","role":"Analyst",'
        '"objective":"Solve the task.","scope":"Reason independently.",'
        '"expected_output":"Return a finding."}'
    ).accepted
    invalid = '{"action":"set_output","target":"agent_404"}'

    rejected = [canvas.step(invalid) for _ in range(3)]

    assert [step.rejection_code for step in rejected] == ["unknown_agent"] * 3
    assert [step.invalid_repeat_count for step in rejected] == [1, 2, 3]
    assert canvas.topology_edits_frozen
    recovered = canvas.recover_frozen_topology()
    assert recovered
    assert recovered[-1].accepted
    assert canvas.state is CanvasState.FINISHED
    assert set(canvas.graph.nodes) == {"solver"}


def test_director_does_not_retarget_repeated_unknown_pending_agent() -> None:
    backend = MockBackend(
        [
            '{"action":"add_agent","expected_version":0}',
            '{"action":"set_prompt","target":"agent_0","role":"Analyst",'
            '"objective":"Solve the task.","scope":"Reason independently.",'
            '"expected_output":"Return a finding.","expected_version":1}',
            '{"action":"set_prompt","target":"agent_0","role":"Analyst",'
            '"objective":"Solve the task.","scope":"Reason independently.",'
            '"expected_output":"Return a finding.","expected_version":1}',
            '{"action":"set_output","target":"agent_1","expected_version":2}',
            '{"action":"finish","expected_version":3}',
        ]
    )
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(RecordingExecutor()))

    result = GraphDirector(backend=backend, canvas=canvas).run()

    assert not result.finished
    assert set(canvas.graph.nodes) == {"agent_1"}
    assert not canvas.graph.nodes["agent_1"].configured
    assert all(not (step.protocol_recovery and step.accepted) for step in canvas.history)
    assert "legal_agent_ids" in backend.calls[1]["messages"][-1]["content"]


def test_canvas_rejects_webshop_delegation_content_absent_from_public_task() -> None:
    canvas = GraphCanvas(
        task=(
            "i'm looking for a case cover hard shell cases of rock ash color, "
            "and price lower than 50.00 dollars"
        ),
        dataset="webshop",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        action_adapter=DatasetActionAdapter(
            adapter_id="webshop",
            datasets=("webshop",),
            action_names=("webshop_search", "webshop_click"),
            initial_action_budget=16,
            revision_action_budget=4,
            total_action_budget=20,
        ),
    )
    assert canvas.step('{"action":"add_agent","agent_id":"shopper"}').accepted
    drift = canvas.step(
        '{"action":"set_prompt","target":"shopper",'
        '"role":"WebShop search specialist",'
        '"objective":"Find hard shell phone cases in rock ash color below 50 dollars",'
        '"scope":"phone cases only, exclude laptop cases",'
        '"expected_output":"matching products"}'
    )
    aligned = canvas.step(
        '{"action":"set_prompt","target":"shopper",'
        '"role":"WebShop search specialist",'
        '"objective":"Find case cover hard shell cases of rock ash color below 50 dollars",'
        '"scope":"catalog search for the requested product constraints",'
        '"expected_output":"matching products"}'
    )

    assert not drift.accepted
    assert drift.rejection_code == "responsibility_violation"
    assert drift.responsibility_issue["code"] == "task_constraint_drift"
    assert drift.responsibility_issue["details"]["novel_terms"] == ["phone"]
    assert aligned.accepted


@pytest.mark.parametrize(
    "dataset,remaining,reserve",
    [
        ("healthbench_professional", 200, 180),
        ("swe_bench", 140, 120),
        ("aime", 40, 20),
    ],
)
def test_time_admission_reserves_dataset_verification(dataset, remaining, reserve):
    from types import SimpleNamespace

    canvas = GraphCanvas(
        task="task", dataset=dataset, runtime=MultiAgentRuntime(RecordingExecutor())
    )
    canvas.rollout_deadline = SimpleNamespace(hard_remaining_s=lambda stage: remaining)
    event = canvas._insufficient_time_event(
        canvas.parser.parse('{"action":"add_agent"}'), {"estimated_worker_s": 30.0}
    )
    assert event["finalization_reserve_s"] == reserve
    assert event["required_s"] == reserve + 30
    canvas.rollout_deadline = SimpleNamespace(hard_remaining_s=lambda stage: reserve + 31)
    assert (
        canvas._insufficient_time_event(
            canvas.parser.parse('{"action":"add_agent"}'), {"estimated_worker_s": 30.0}
        )
        is None
    )
