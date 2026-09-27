from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.dataset_adapters import (
    HealthBenchOfficialRubricVerifier,
    solver_task_text,
)
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("revision", [False, True])
def test_healthbench_complete_answer_reaches_judge_without_internal_fields(
    recover: bool, revision: bool
) -> None:
    # Synthetic content: no benchmark example or private rubric informs generation.
    answer = (
        "The supplied study reports an association.\n\n"
        "Its observational design does not establish causation; the effect remains uncertain."
    )
    internal_summary = "INTERNAL_SUMMARY_ONLY"
    private_criterion = "PRIVATE_RUBRIC_MARKER: mentions the study limitation."
    task = TaskSpec(
        task_id="healthbench-protocol-synthetic",
        prompt="Explain the finding and its limitation.",
        metadata={
            "dataset": "healthbench_professional",
            "verifier": "healthbench_rubric",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "This is an observational study."},
                    {"role": "assistant", "content": "An association was reported."},
                    {"role": "user", "content": "Explain the finding and its limitation."},
                ]
            },
        },
        private_verifier_payload={
            "rubric_items": [{"criterion_text": private_criterion, "points": 5}]
        },
    )
    response = json.dumps(
        {
            "answer": answer,
            "summary": internal_summary,
            "confidence": 0.7,
            "evidence": ["INTERNAL_EVIDENCE_ONLY"],
            "unresolved_issues": [],
            "tool_summary": [],
        }
    )
    responses = [response]
    if recover:
        responses.insert(0, LLMResponse(text="", model="reasoning-only"))
    worker = MockBackend(responses)
    executor = ModelAgentExecutor(
        worker, action_registry=default_dataset_action_registry(())
    )
    node = AgentNode(
        "reviewer",
        "Explain only the supplied study's finding and limitation.",
        operation_policy_configured=True,
        initial_tool_budget=0,
        total_tool_budget=0,
        metadata={"action_adapter": "healthbench_professional"},
    )
    artifact = executor.execute(
        task=solver_task_text(task, include_submission_contract=True),
        node=node,
        upstream=[],
        peers=[],
        revision=revision,
        seed=0,
    )
    submission = AnswerFinalizer(AnswerSubmissionConfig(enabled=True)).finalize(
        task, artifact.answer, raw_summary=artifact.summary
    )
    judge = MockBackend(
        [json.dumps({"criteria_met": True, "explanation": "Limitation is present."})]
    )
    verification = HealthBenchOfficialRubricVerifier(judge).verify(
        task, submission.submitted_answer
    )

    assert artifact.answer == submission.submitted_answer == answer
    assert submission.method == "passthrough"
    assert json.loads(verification.detail)["criteria_met"] == 1
    judge_prompt = json.dumps(judge.calls[0]["messages"], ensure_ascii=False)
    assert json.dumps(answer)[1:-1] in judge_prompt
    assert internal_summary not in judge_prompt
    assert "INTERNAL_EVIDENCE_ONLY" not in judge_prompt
    assert len(worker.calls) == (2 if recover else 1)
    for call in worker.calls:
        system = call["messages"][0]["content"]
        assert "complete, self-contained result" in system
        assert "An intermediate Worker should complete only its assigned sub-task" in system
        assert "all explanation in summary or evidence" not in system
        assert "shortest answer span" not in system
        context = json.loads(call["messages"][1]["content"])
        assert context["assigned_task"] == node.prompt
        assert "This is an observational study." in context["public_task_context"]
        assert "Submission contract:" in context["public_task_context"]
        assert "complete, self-contained result" in context["public_task_context"]
        assert "information essential to the reader must also appear in answer" in context["public_task_context"]
        assert "all supporting explanation in summary or evidence" not in json.dumps(call)
        assert private_criterion not in json.dumps(call)
