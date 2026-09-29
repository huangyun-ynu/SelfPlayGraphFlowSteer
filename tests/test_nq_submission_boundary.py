import json

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.application import AdaptiveApplicationResult
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.evaluation import from_adaptive_result
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import MultiAnswerExactMatchVerifier, TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.submission_contract import receipt_error

from .test_nq_corpus_evidence import SEARCH, SearchTool, insufficient, supported


def solve_corpus(responses, tool=None):
    actions = [
        {"action": "add_agent", "agent_id": "solver"},
        {"action": "set_prompt", "target": "solver", "role": "Analyst",
         "objective": "Answer the public question", "scope": "Use public corpus evidence",
         "expected_output": "Return the requested result"},
        {"action": "set_output", "target": "solver"},
        {"action": "finish"},
    ]
    registry = default_dataset_action_registry(("search",), nq_evidence_mode="corpus_tool")
    runtime = MultiAgentRuntime(ModelAgentExecutor(
        MockBackend(responses), tools={"search": tool or SearchTool()}, action_registry=registry,
    ))
    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend([json.dumps(action) for action in actions]),
        runtime=runtime, verifier=MultiAnswerExactMatchVerifier(), action_registry=registry,
        answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
        canvas_config=CanvasConfig(max_rounds=len(actions)),
        nq_evidence_mode="corpus_tool", nq_policy={"max_submission_repairs": 1},
    )
    question = "What did Curie name after Poland?"
    task = TaskSpec("nq-receipt", question, reference=["Polonium"], metadata={
        "dataset": "nq_open", "evidence_mode": "corpus_tool", "original_question": question,
    })
    result = solver.solve(task, run_id="nq-receipt-run")
    return AdaptiveApplicationResult("nq-receipt-run", task, result, (), None, ""), solver


def test_corpus_abstention_finish_receipt_is_accepted_and_scored_zero():
    app, solver = solve_corpus([SEARCH, insufficient(), insufficient()])
    result = app.solver_result
    receipt = result.director_run.submission_receipt
    assert result.director_run.finished and receipt is not None
    assert receipt.submitted_answer_snapshot == "insufficient_evidence"
    assert receipt_error(receipt, run=result.director_run, events=solver.active_canvas.history,
                         run_id=app.run_id, dataset="nq_open") is None
    assert app.task.metadata["nq_corpus_submission"]["status"] == "insufficient_evidence"
    assert app.task.metadata["submission_status"] == "submitted"
    assert result.outcome_decision.status == "scored"
    assert result.verification.score == 0 and not result.verification.passed
    assert from_adaptive_result(app).score == 0


def test_correct_answer_with_invalid_citations_is_terminal_policy_zero():
    bad = supported("forged-id")
    app, _ = solve_corpus([SEARCH, bad, bad, bad])
    result = app.solver_result
    assert result.director_run.submission_receipt is None
    assert result.outcome_decision.status == "policy_failure"
    assert result.outcome_decision.reason == "nq_invalid_evidence_submission"
    assert app.task.metadata["runtime_terminal_policy_failure"]["code"] == "nq_invalid_evidence_submission"
    assert from_adaptive_result(app).score == 0
    assert from_adaptive_result(app).passed is False


def test_retrieval_service_outage_remains_unscored():
    class BrokenSearch(SearchTool):
        def execute(self, arguments):
            raise ConnectionError("test corpus service unavailable")

    app, _ = solve_corpus([SEARCH, insufficient(), insufficient(), insufficient()], BrokenSearch())
    result = app.solver_result
    assert result.director_run.submission_receipt is None
    assert result.outcome_decision.status == "unsubmitted_unknown"
    assert result.verification is None
    assert app.task.metadata["runtime_terminal_policy_failure"] is None
    assert from_adaptive_result(app).score is None
