"""The NQ prompt requests concise answers without exposing hidden labels."""

import json

from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.output_contract import selected_output_instruction
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor

from .test_nq_corpus_evidence import SEARCH, SearchTool, context, run, supported


def test_nq_output_worker_receives_singular_answer_and_citation_contract():
    backend = MockBackend([SEARCH, supported()])
    executor = ModelAgentExecutor(
        backend, tools={"search": SearchTool()}, nq_evidence_context=context(),
    )
    run(executor)

    first_system_prompt = backend.calls[0]["messages"][0]["content"]
    assert "choose one best-supported candidate" in first_system_prompt
    assert "explain ties or alternatives in summary" in first_system_prompt
    assert "Match the answer type in the question" in first_system_prompt
    assert "evidence_refs" in first_system_prompt
    assert "verbatim supporting quote" in first_system_prompt
    # The synthetic gold appears only in the runtime search result, not in the
    # initial task prompt. Retrieval evidence may of course contain the answer.
    assert "Polonium" not in json.dumps(backend.calls[0]["messages"])
    assert "target_answers" not in json.dumps(backend.calls[0]["messages"])


def test_nq_answer_format_is_scoped_to_output_agent_and_dataset():
    args = {"action_adapter": "", "short_answer_qa": True, "is_output_agent": True}
    nq = selected_output_instruction(dataset="nq_open", **args)
    hotpot = selected_output_instruction(dataset="hotpotqa", **args)
    non_output = selected_output_instruction(
        dataset="nq_open", **{**args, "is_output_agent": False},
    )
    assert "only the letter" in nq
    assert "Include multiple items only when" in nq
    assert "only the letter" not in hotpot
    assert non_output == ""


def test_finalizer_does_not_choose_a_gold_match_from_a_verbose_answer():
    task = TaskSpec("nq-test", "What is the highest-scoring Scrabble letter?",
                    reference="Q", metadata={"dataset": "nq_open"})
    raw = "Q and Z (10 points each)"
    submission = AnswerFinalizer(AnswerSubmissionConfig(enabled=True)).finalize(task, raw)
    assert submission.submitted_answer == raw
