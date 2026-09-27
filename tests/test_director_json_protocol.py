import json
import re

import pytest

from selfplay_graph_flowsteer.actions import ActionParser
from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.application import AdaptiveApplicationResult
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.director import GraphDirector, director_prompt_components
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.observability import NumericVerifier, TaskSpec
from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime

from .helpers import NumericRecordingExecutor
from .test_finish_submission_contract import rollout

GOOD_PROMPT = json.dumps({
    "action": "set_prompt", "target": "solver", "role": "Analyst",
    "objective": "Solve the assigned task", "scope": r"Review the task involving \omega",
    "expected_output": "Return the requested result",
})
BAD_PROMPT = GOOD_PROMPT.replace(r"\\omega", r"\omega")
SUFFIX = ['{"action":"set_output","target":"solver"}', '{"action":"finish"}']


@pytest.mark.parametrize("variant", ["v2", "v2.1", "v2.2"])
def test_rendered_prompt_has_valid_json_backslash_example(variant):
    prompt, _ = director_prompt_components(variant)
    assert r"backslash as `\\`" in prompt
    sample = re.search(r'JSON text (\{"scope":.*?\})', prompt).group(1)
    assert json.loads(sample)["scope"] == r"Review the role of \omega in the task"
    assert r"\\omega" in sample


@pytest.mark.parametrize("raw,code,count", [
    (BAD_PROMPT, "director_json_syntax_error", 1),
    ('{"action":"set_prompt",', "director_json_syntax_error", 0),
    ('No action here.', "director_json_object_missing", 0),
    ('{"action":"add_agent"}\n{"action":"finish"}', "director_multiple_action_objects", 2),
    ('{"action":"set_relation","relations":[{"source":"a","target":"b","relation":"directed"}]}',
     "director_action_schema_error", 1),
    ('{"action":"remove_relation","relations":[]}', "director_action_schema_error", 1),
])
def test_invalid_actions_have_typed_factual_diagnostics_without_repair(raw, code, count):
    parsed = ActionParser().parse_policy_output(raw)
    assert not parsed.action.valid and parsed.action_text is None
    assert parsed.action.parse_error_code == code
    assert parsed.candidate_count == count
    assert parsed.action.raw_text == raw
    if code == "director_action_schema_error":
        assert "relations" in parsed.action.parse_error_details["received_fields"]
        assert parsed.action.parse_error_details["structured_action_fields"] == [
            "action", "source", "target", "relation",
        ]


def test_syntax_position_addresses_original_action_text_and_hint_is_valid_json():
    raw = "\n\n" + BAD_PROMPT
    parsed = ActionParser().parse_policy_output(raw)
    with pytest.raises(json.JSONDecodeError) as caught:
        json.loads(raw)
    expected = caught.value
    details = parsed.action.parse_error_details
    assert (details["line"], details["column"], details["raw_action_offset"]) == (
        expected.lineno, expected.colno, expected.pos,
    )
    assert len(details["fragment"]) <= 72
    assert r"\omega" in details["fragment"]
    sample = re.search(r'("Use .*?")', details["hint"]).group(1)
    assert json.loads(sample) == r"Use \omega"


def test_normal_retry_receives_specific_error_and_does_not_mutate_failed_action():
    responses = iter(['{"action":"add_agent","agent_id":"solver"}',
                      BAD_PROMPT, GOOD_PROMPT, *SUFFIX])
    prompts = []

    def handler(messages, role):
        prompts.append(messages)
        if len(prompts) == 3:
            content = messages[-1]["content"]
            assert "director_json_syntax_error" in content
            assert "raw_action_offset" in content and "hint" in content
            assert "responsibility_violation" not in content
            assert "Return the next single JSON action" in content
        return next(responses)

    executor = NumericRecordingExecutor()
    canvas = GraphCanvas(task="Solve the task involving omega", dataset="aime",
                         runtime=MultiAgentRuntime(executor))
    backend = MockBackend(handler=handler)
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert run.finished and run.output == "35"
    assert len(backend.calls) == len(run.turns) == 5
    failed = canvas.history[1]
    assert failed.rejection_code == "director_json_syntax_error"
    assert not failed.responsibility_issue
    assert not failed.executed_agents and not failed.graph["nodes"][0]["prompt"]
    assert run.turns[1].raw_action_text == BAD_PROMPT
    assert not run.turns[1].action_diagnostics["repair_attempted"]
    assert BAD_PROMPT in [m["content"] for m in prompts[2] if m["role"] == "assistant"]
    assert len(executor.calls) == 2  # Prompt execution and output-role execution only.


def test_failed_policy_preserves_provider_token_ids_logprobs_and_reasoning():
    response = LLMResponse(
        text=BAD_PROMPT, model="recorded-fixture", raw_action_text=BAD_PROMPT,
        raw_reasoning_text="retained reasoning", prompt_token_ids=(17, 18),
        completion_token_ids=(19, 20), behavior_log_probs=(-0.7, -0.9),
        token_provenance="fixture_exact_provider_trace", training_eligible=True,
        metadata={"finish_reason": "stop"},
    )
    backend = MockBackend([response])
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(NumericRecordingExecutor()),
                         config=CanvasConfig(max_rounds=1))
    run = GraphDirector(backend=backend, canvas=canvas).run()
    turn = run.turns[0]
    assert turn.raw_action_text == BAD_PROMPT and turn.raw_reasoning_text == "retained reasoning"
    assert turn.prompt_token_ids == response.prompt_token_ids
    assert turn.completion_token_ids == response.completion_token_ids
    assert turn.behavior_log_probs == response.behavior_log_probs
    assert turn.trainable and not turn.accepted
    assert not canvas.graph.nodes and len(backend.calls) == 1


def test_syntax_retry_does_not_reuse_a_previous_responsibility_rejection():
    leaking = json.loads(GOOD_PROMPT)
    leaking["objective"] = "Solve the task. Final answer is 123."
    backend = MockBackend(['{"action":"add_agent","agent_id":"solver"}',
                           json.dumps(leaking), BAD_PROMPT, GOOD_PROMPT, *SUFFIX])
    canvas = GraphCanvas(task="Solve the task involving omega", dataset="aime",
                         runtime=MultiAgentRuntime(NumericRecordingExecutor()))
    run = GraphDirector(backend=backend, canvas=canvas).run()
    assert run.turns[1].rejection_code == "responsibility_violation"
    assert run.turns[2].rejection_code == "director_json_syntax_error"
    current_feedback = backend.calls[3]["messages"][-1]["content"]
    assert "director_json_syntax_error" in current_feedback
    assert "responsibility_violation" not in current_feedback
    assert "Your previous SET_PROMPT was rejected" not in current_feedback
    assert run.finished


def test_snapshot_field_requirements_do_not_advertise_masked_actions():
    canvas = GraphCanvas(task="task", runtime=MultiAgentRuntime(NumericRecordingExecutor()))
    snapshot = canvas.control_snapshot()
    assert snapshot["action_field_requirements"] == {"add_agent": ["action"]}
    canvas.step('{"action":"add_agent","agent_id":"solver"}')
    snapshot = canvas.control_snapshot()
    assert set(snapshot["action_field_requirements"]) == set(snapshot["allowed_actions"])
    assert snapshot["action_field_requirements"]["set_prompt"] == [
        "action", "target", "role", "objective", "scope", "expected_output",
    ]


def test_exhausted_complete_json_failures_become_known_policy_zero_without_submission():
    backend = MockBackend(['{"action":"add_agent","agent_id":"solver"}'] + [BAD_PROMPT] * 4)
    solver = AdaptiveWorkflowSolver(
        director_backend=backend, runtime=MultiAgentRuntime(NumericRecordingExecutor()),
        verifier=NumericVerifier(), canvas_config=CanvasConfig(max_rounds=24),
    )
    task = TaskSpec("json-stall", "Solve the task involving omega", reference=35,
                    metadata={"dataset": "aime"})
    result = solver.solve(task, run_id="json-stall")
    assert len(backend.calls) == 5 and not result.director_run.finished
    assert result.director_run.submission_receipt is None and result.verification is None
    assert solver.active_canvas.total_tokens == 0
    assert result.outcome_decision.status == "policy_failure"
    assert result.outcome_decision.reason == "director_action_protocol_exhausted"
    trained = rollout(AdaptiveApplicationResult("json-stall", task, result, (), None, ""))
    assert trained.trajectory.reward == 0
    assert trained.trajectory.metadata["reward_known"]
    assert trained.trajectory.metadata["training_eligible"]
    assert not trained.trajectory.metadata["answer_reward_released"]


@pytest.mark.parametrize("raw,finish_reason", [
    ('{"action":', "length"), (BAD_PROMPT, "length"), ("", "stop"),
])
def test_missing_or_truncated_responses_do_not_get_complete_protocol_failure_attribution(raw, finish_reason):
    def handler(messages, role):
        return LLMResponse(text=raw, raw_action_text=raw, model="fixture",
                           metadata={"mock": True, "finish_reason": finish_reason})

    canvas = GraphCanvas(task="task", dataset="aime", runtime=MultiAgentRuntime(NumericRecordingExecutor()))
    backend = MockBackend(handler=handler)
    GraphDirector(backend=backend, canvas=canvas).run()
    assert len(backend.calls) == 4
    assert canvas.history[-1].rejection_code == "director_no_progress_exhausted"


def test_infrastructure_evidence_still_blocks_protocol_failure_zero():
    result = terminal_policy_failure(
        "aime", terminal=True, rejection_codes=["director_action_protocol_exhausted"],
        artifacts={}, output_agent=None, rounds=5, max_rounds=24, worker_tokens=0,
        worker_token_limit=240000, infrastructure_failure=True,
    )
    assert result is None
