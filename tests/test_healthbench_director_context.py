from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime


@pytest.mark.parametrize("dataset", ["healthbench_professional", "other_dataset"])
@pytest.mark.parametrize("with_history", [False, True])
def test_solver_sends_public_healthbench_history_to_director_and_worker(
    dataset: str, with_history: bool
) -> None:
    history = [
        {"role": "user", "content": "Discuss the fictional SYNTHETIC_CONTEXT condition."},
        {"role": "assistant", "content": "The earlier PUBLIC_ASSISTANT_REPLY."},
    ] if with_history else []
    question = "What about the outcome?"
    conversation = {"messages": [*history, {"role": "user", "content": question}]}
    task = TaskSpec(
        task_id="synthetic-context-check",
        prompt=question,
        metadata={
            "dataset": dataset,
            "conversation": conversation,
            "physician_response": "PRIVATE_PHYSICIAN_RESPONSE",
            "canary_string": "PRIVATE_CANARY",
        },
        private_verifier_payload={"rubric_items": [{"criterion_text": "PRIVATE_RUBRIC", "points": 1}]},
    )
    director = MockBackend([
        '{"action":"add_agent"}',
        json.dumps({"action": "set_prompt", "target": "agent_1", "role": "Responder",
                    "objective": "Answer the latest question using the public conversation.",
                    "scope": "Public context and the requested outcome.",
                    "expected_output": "Complete response with uncertainty."}),
        '{"action":"set_model","target":"agent_1","runtime_route":"gpt"}',
        '{"action":"set_output","target":"agent_1"}',
        '{"action":"finish"}',
    ])
    response = json.dumps({
        "answer": "A complete synthetic response.", "summary": "Internal summary.",
        "confidence": 0.5, "evidence": [], "unresolved_issues": [], "tool_summary": [],
    })
    # Model selection executes the initial responsibility; assigning output
    # changes its contract and executes it again. Both calls need a fixture.
    worker = MockBackend([response, response])
    registry = default_dataset_action_registry(())
    solver = AdaptiveWorkflowSolver(
        director_backend=director,
        runtime=MultiAgentRuntime(ModelAgentExecutor(worker, action_registry=registry)),
        runtime_routes=("gpt",), action_registry=registry,
        answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
    )
    result = solver.solve(task, run_id="context-regression")
    assert result.director_run.finished
    assert result.answer_submission.submitted_answer == "A complete synthetic response."
    initial = next(m["content"] for m in director.calls[0]["messages"] if m["role"] == "user")
    assert question in initial
    assert "Submission contract:" not in initial
    if dataset == "healthbench_professional":
        assert "user: " + question in initial
        context = json.loads(worker.calls[0]["messages"][1]["content"])["public_task_context"]
        assert question in context
        assert "Submission contract:" in context
        assert "complete, self-contained result" in context
        assert "all supporting explanation in summary or evidence" not in json.dumps(worker.calls)
        if with_history:
            assert "user: Discuss the fictional SYNTHETIC_CONTEXT condition." in initial
            assert "assistant: The earlier PUBLIC_ASSISTANT_REPLY." in initial
            assert "SYNTHETIC_CONTEXT" in context
    else:
        assert initial.startswith("Task:\n" + question + "\n")
        assert "SYNTHETIC_CONTEXT" not in initial
    all_requests = json.dumps([director.calls, worker.calls])
    for private in ("PRIVATE_PHYSICIAN_RESPONSE", "PRIVATE_CANARY", "PRIVATE_RUBRIC"):
        assert private not in all_requests
