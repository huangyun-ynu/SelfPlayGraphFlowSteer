"""Exercise public context, recovery, and submission with the Hotpot contract."""

import json

import pytest

from selfplay_graph_flowsteer.answer_submission import (
    AnswerFinalizer, AnswerSubmissionConfig, submission_contract,
)
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.hotpot_answer_contract import HOTPOT_RESULT_FIELDS
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime, _worker_output_instruction


@pytest.mark.parametrize("recover", [False, True])
def test_evidence_first_artifact_survives_runtime_and_submission_without_gold(recover):
    result = dict(zip(HOTPOT_RESULT_FIELDS, [
        ["[Eastmere] Latitude 55 N", "[Westmere] Latitude 32 N"],
        "Eastmere is farther north because 55 N exceeds 32 N.",
        0.95, [], [], "Eastmere",
    ]))
    malformed = {**result, "confidence": "high"}
    backend = MockBackend(([json.dumps(malformed)] if recover else []) + [json.dumps(result)])
    question = "[Eastmere] Latitude 55 N. [Westmere] Latitude 32 N. Which is farther north?"
    task = TaskSpec("synthetic", question, reference="PRIVATE_REFERENCE_DO_NOT_EXPOSE",
                    metadata={"dataset": "hotpotqa"})
    canvas = GraphCanvas(task=question, dataset="hotpotqa", runtime_routes=("mock",),
                         runtime=MultiAgentRuntime(ModelAgentExecutor(backend)),
                         config=CanvasConfig(max_rounds=50))
    actions = [
        dict(action="add_agent", agent_id="a"),
        dict(action="set_prompt", target="a", role="Evidence analyst",
             objective="Identify Eastmere's latitude.", scope="Read the Eastmere passage.",
             expected_output="The latitude and evidence."),
        dict(action="set_output", target="a"),
        dict(action="set_model", target="a", runtime_route="mock"),
    ]
    for action in actions:
        if action["action"] == "set_model":
            canvas.graph.nodes["a"].operation_policy_configured = True
        response = canvas.step(json.dumps(action), authoritative_director=True)
        assert response.accepted, response.feedback
    artifact = canvas.runtime.artifacts["a"]
    assert artifact.answer == "Eastmere"
    assert len(backend.calls) == (2 if recover else 1)
    for call in backend.calls:
        encoded = json.dumps(call["messages"])
        assert task.reference not in encoded
        assert question in encoded
        assert "Write answer LAST" in encoded
        assert "shortest direct answer" not in encoded
        assert "original question" in encoded
    if recover:
        context = json.loads(backend.calls[-1]["messages"][-1]["content"])
        assert tuple(context["artifact_schema"]) == HOTPOT_RESULT_FIELDS
    # Submission does not read the summary to overwrite the answer or consult gold.
    final = AnswerFinalizer(AnswerSubmissionConfig(enabled=True)).finalize(
        task=task, raw_answer=artifact.answer,
    )
    assert final.submitted_answer == "Eastmere"


@pytest.mark.parametrize("dataset", ["nq_open", "aime", "healthbench_professional", "webshop"])
def test_other_dataset_worker_contract_unchanged(dataset):
    from selfplay_graph_flowsteer.qa_submission import is_short_qa_dataset
    instruction = _worker_output_instruction([], dataset=dataset, action_adapter=dataset,
                                             short_answer_qa=is_short_qa_dataset(dataset),
                                             is_output_agent=True)
    assert "Write answer LAST" not in instruction
    assert "Return one final JSON object with answer, summary" in instruction


def test_hotpot_submission_uses_public_question_without_mutating_reference():
    task = TaskSpec("span", "What type of music?", reference="secret",
                    metadata={"dataset": "hotpotqa"})
    assert "answer type and granularity" in submission_contract(task)
    assert "secret" not in submission_contract(task)
    assert task.reference == "secret"
