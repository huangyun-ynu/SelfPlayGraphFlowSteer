import json
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.aime_submission import parse_aime_answer
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.application import AdaptiveSolverApplication
from selfplay_graph_flowsteer.canvas import CanvasState, GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.mace import MACEModelRouter
from selfplay_graph_flowsteer.observability import NumericVerifier, TaskSpec
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime, PeerInteraction

from .helpers import RecordingExecutor


@pytest.mark.parametrize(
    "raw,answer",
    [
        ("0", "0"),
        ("999", "999"),
        ("35", "35"),
        ("035", "035"),
        (r"\boxed{035}", "035"),
        ("Final answer: 035", "035"),
        ('{"answer":"035"}', "035"),
        ('{"answer":0}', "0"),
        ('{"answer":"035","final_answer":"35"}', "035"),
        ('```json\n{"answer":"035"}\n```', "035"),
        (r"Work mentions 12. Final answer: \boxed{035}.", "035"),
        ("Final answer: 35. 检查中用到了3", "35"),
        ("推导用到了3。答案：35", "35"),
    ],
)
def test_accept_only_explicit_aime_answer(raw, answer):
    result = parse_aime_answer(raw)
    assert result.valid
    assert result.answer == answer


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[12, 7, 3]",
        "计算得到边长为3，还需要继续求解",
        "答案可能是35或36",
        "Final answer: 35 or 36",
        "Final answer: 35\nFinal answer: 36",
        "Final answer: 35. Final answer: 36",
        r"\boxed{35} or \boxed{36}",
        r"\boxed{35} or 36",
        "Final answer: 35. Maybe 36",
        "Final answer: 35. Answer: 36",
        "35.6",
        "1000",
        "-1",
        "1e2",
        "35/1",
        "True",
        "NaN",
        '{"answer":[12,7,3]}',
        '{"answer":{"value":35}}',
        '{"answer":true}',
        '{"answer":35.0}',
        '{"answer":"35","answer":"36"}',
        '{"answer":"35","final_answer":"36"}',
        '{"summary":"answer 35"}',
        "Final answer: 35/1",
        "Final answer: 35.6",
        r"\boxed{-1}",
        "WORKER_PROTOCOL_FAILURE: finalization_2",
    ],
)
def test_reject_without_last_number_fallback(raw):
    result = parse_aime_answer(raw)
    assert not result.valid
    assert result.answer == ""
    # Even a reference equal to the tempting last digit cannot turn it into success.
    task = TaskSpec("aime", "question", reference=3, metadata={"dataset": "aime"})
    assert not NumericVerifier().verify(task, raw).passed


@pytest.mark.parametrize("enabled", [True, False])
def test_finalizer_checks_even_when_disabled_and_never_uses_summary_or_reference(enabled):
    finalizer = AnswerFinalizer(AnswerSubmissionConfig(enabled=enabled))
    outputs = []
    for reference in (3, 35, 999):
        task = TaskSpec("aime", "question", reference=reference, metadata={"dataset": "aime"})
        outputs.append(finalizer.finalize(task, "[12,7,3]", raw_summary="Final answer: 35"))
    assert outputs[0] == outputs[1] == outputs[2]
    assert not outputs[0].valid
    assert outputs[0].raw_answer == "[12,7,3]"
    assert outputs[0].submitted_answer == ""


class AnswerExecutor(RecordingExecutor):
    def execute(self, **kwargs):
        artifact = super().execute(**kwargs)
        artifact.answer = "35" if "Finalize" in kwargs["node"].prompt else "[12,7,3]"
        return artifact


def configure(canvas, agent="solver", role="Analyst"):
    assert canvas.step(json.dumps({"action": "add_agent", "agent_id": agent})).accepted
    assert canvas.step(
        json.dumps(
            {
                "action": "set_prompt",
                "target": agent,
                "role": role,
                "objective": "Solve the assigned task.",
                "scope": "Reason independently.",
                "expected_output": "Return the requested result.",
            }
        )
    ).accepted


def test_finish_rejection_preserves_artifact_and_director_can_repair():
    executor = AnswerExecutor()
    canvas = GraphCanvas(task="question", dataset="aime", runtime=MultiAgentRuntime(executor))
    configure(canvas)
    assert canvas.runtime.artifacts["solver"].answer == "[12,7,3]"
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    before = len(executor.calls)
    rejected = canvas.step('{"action":"finish"}')
    assert not rejected.accepted
    assert rejected.rejection_code == "output_answer_invalid"
    assert canvas.active
    assert canvas.state != CanvasState.FINISHED
    assert len(executor.calls) == before  # No automatic format-retry model call.
    assert canvas.runtime.artifacts["solver"].answer == "[12,7,3]"
    assert canvas.recover_finish_only() is None
    assert any(
        action["action"] == "set_prompt"
        for action in rejected.rejection_details["legal_recovery_actions"]
    )
    assert not any(
        action["action"] == "finish"
        for action in rejected.rejection_details["legal_recovery_actions"]
    )
    fixed = canvas.step(
        json.dumps(
            {
                "action": "set_prompt",
                "target": "solver",
                "role": "Finalize",
                "objective": "Provide the final result.",
                "scope": "Resolve the final submission.",
                "expected_output": "One final integer.",
                "revision_basis": "protocol_failure",
                "evidence_agent_ids": ["solver"],
            }
        )
    )
    assert fixed.accepted, fixed.feedback
    assert canvas.step('{"action":"finish"}').accepted
    assert canvas.state == CanvasState.FINISHED
    assert len(executor.calls) == before + 1


def test_repeated_invalid_finish_uses_existing_round_budget():
    canvas = GraphCanvas(
        task="question",
        dataset="aime",
        runtime=MultiAgentRuntime(AnswerExecutor()),
        config=CanvasConfig(max_rounds=4),
    )
    configure(canvas)
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    assert not canvas.step('{"action":"finish"}').accepted
    assert not canvas.active
    assert canvas.state != CanvasState.FINISHED


def test_wrong_but_well_formed_answer_finishes_without_correctness_feedback():
    canvas = GraphCanvas(
        task="question", dataset="aime", runtime=MultiAgentRuntime(AnswerExecutor())
    )
    configure(canvas, role="Finalize")
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    assert canvas.step('{"action":"finish"}').accepted
    task = TaskSpec("aime", "question", reference=36, metadata={"dataset": "aime"})
    assert not NumericVerifier().verify(task, "35").passed


def test_non_aime_numeric_behavior_unchanged():
    task = TaskSpec("numeric", "question", reference=-1.25, task_type="math")
    assert NumericVerifier().verify(task, "Final answer: -1.25").passed
    assert AnswerFinalizer(AnswerSubmissionConfig(enabled=True)).finalize(task, "-1.25").valid


def test_retired_peer_api_never_scores_even_injected_legacy_interactions():
    runtime = MultiAgentRuntime(RecordingExecutor())
    invalid = PeerInteraction("a", "b", "[12,7,3]", "35", 1)
    valid = PeerInteraction("c", "d", "34", "35", 1)
    runtime.peer_interactions = [invalid, valid]

    def scorer(_answer):
        raise AssertionError("removed peer scorer must never run")

    assert runtime.finalize_peer_rewards(scorer) == []
    assert runtime.peer_interactions == []
    for interaction in (invalid, valid):
        assert interaction.audit_event["reward_status"] == "discarded"
        assert interaction.audit_event["discard_reason"] == "peer_selector_removed"
        assert "score_delta" not in interaction.audit_event


@pytest.mark.parametrize("role,valid", [("Analyst", False), ("Finalize", True)])
def test_local_aime_artifacts_never_update_retired_model_router(role, valid):
    del role, valid
    with pytest.raises(ValueError, match="retired"):
        AdaptiveWorkflowSolver(
            director_backend=MockBackend([]),
            runtime=MultiAgentRuntime(AnswerExecutor()),
            model_router=MACEModelRouter(seed=0),
        )


def test_real_task5_geometry_list_is_not_submitted_as_three():
    # Exact raw-answer structure from real5 step2, task-5, solver_rollouts line 24.
    raw = json.dumps(
        [
            {
                "U_1 side_lengths": "3k, 4k, 5k",
                "area": "6k^2",
                "configuration": "U_1 at the vertex between sides 3 and 4",
                "cut_segment": "parallel to side 5",
            },
            {
                "U_1 side_lengths": "3k, 4k, 5k",
                "area": "6k^2",
                "configuration": "U_1 at the vertex between sides 3 and 5",
                "cut_segment": "parallel to side 4",
            },
            {
                "U_1 side_lengths": "3k, 4k, 5k",
                "area": "6k^2",
                "configuration": "U_1 at the vertex between sides 4 and 5",
                "cut_segment": "parallel to side 3",
            },
        ]
    )
    parsed = parse_aime_answer(raw)
    assert parsed.reason == "non_scalar_answer"
    assert parsed.answer == ""


def test_partial_upstream_artifact_remains_usable_for_collaboration():
    canvas = GraphCanvas(
        task="question", dataset="aime", runtime=MultiAgentRuntime(AnswerExecutor())
    )
    configure(canvas, agent="evidence")
    configure(canvas, agent="output", role="Finalize")
    assert canvas.step('{"action":"set_layer","target":"output","layer":1}').accepted
    assert canvas.step(
        '{"action":"set_relation","source":"evidence","target":"output","relation":"directed"}'
    ).accepted
    assert canvas.step('{"action":"set_output","target":"output"}').accepted
    assert canvas.step('{"action":"finish"}').accepted
    assert canvas.runtime.artifacts["evidence"].answer == "[12,7,3]"


@pytest.mark.parametrize("role,valid", [("Analyst", False), ("Finalize", True)])
def test_counterfactual_evaluator_checks_submission_validity(role, valid):
    class CountingVerifier(NumericVerifier):
        def __init__(self):
            super().__init__()
            self.calls = []

        def verify(self, task, prediction):
            self.calls.append(prediction)
            return super().verify(task, prediction)

    runtime = MultiAgentRuntime(AnswerExecutor())
    canvas = GraphCanvas(task="question", dataset="aime", runtime=runtime)
    configure(canvas, role=role)
    assert canvas.step('{"action":"set_output","target":"solver"}').accepted
    app = object.__new__(AdaptiveSolverApplication)
    app.runtime = runtime
    app.config = SimpleNamespace(canvas=CanvasConfig())
    verifier = CountingVerifier()
    app.solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend([]), runtime=runtime, verifier=verifier
    )
    result = app.evaluate_graph(
        TaskSpec("aime", "question", reference=35, metadata={"dataset": "aime"}),
        canvas.graph,
        seed=0,
        return_verification=True,
    )
    assert bool(result["score"]) is valid
    assert result["prediction"] == ("35" if valid else "")
    assert verifier.calls == (["35"] if valid else [])


def test_aime_never_calls_optional_qa_formatter():
    backend = MockBackend([])
    finalizer = AnswerFinalizer(
        AnswerSubmissionConfig(enabled=True, qa_model_enabled=True, runtime_route="unused"),
        qa_backend=backend,
    )
    task = TaskSpec("aime", "question", reference=35, metadata={"dataset": "aime"})
    assert not finalizer.finalize(task, "[12,7,3]", raw_summary="Final answer: 35").valid
    assert backend.calls == []
