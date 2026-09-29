"""Training branches retain corpus provenance after the R2D2 promotion."""

import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.application import (
    AdaptiveSolverApplication,
    GraphEvaluationIncompleteError,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import MultiAnswerExactMatchVerifier, TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime

from .test_nq_corpus_evidence import SEARCH, SearchTool, node, supported, insufficient


def branch_app(responses, tool=None):
    config = load_adaptive_config("configs/formal_training.toml", validate=False)
    config = replace(config, retrieval=replace(
        config.retrieval, nq_policy=replace(config.retrieval.nq_policy, max_submission_repairs=0),
    ))
    registry = default_dataset_action_registry(("search",), nq_evidence_mode="corpus_tool")
    tool = tool or SearchTool()
    runtime = MultiAgentRuntime(ModelAgentExecutor(
        MockBackend(responses), tools={"search": tool}, action_registry=registry,
    ))
    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend([]), runtime=runtime,
        action_registry=registry, verifier=MultiAnswerExactMatchVerifier(),
        answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
        nq_evidence_mode="corpus_tool", nq_policy=config.retrieval.nq_policy,
    )
    app = AdaptiveSolverApplication(config=config, solver=solver, runtime=runtime,
                                    skillbank=None, skill_lifecycle=None)
    graph = MultiAgentGraph()
    graph.add_agent("a")
    graph.nodes["a"] = node()
    graph.set_output("a")
    question = "What did Curie name after Poland?"
    task = TaskSpec("nq-cf", question, reference=["Polonium"], metadata={
        "dataset": "nq_open", "evidence_mode": "corpus_tool", "original_question": question,
    })
    return app, task, graph, tool


def test_training_counterfactuals_search_with_independent_evidence_and_budgets():
    app, task, graph, tool = branch_app([SEARCH, supported(), SEARCH, supported()])
    try:
        primary_metadata = json.dumps(task.metadata, sort_keys=True)
        assert app.evaluate_graph(task, graph, seed=0) == 1
        first = app.last_graph_evaluation["nq_corpus_evidence_audit"]
        assert app.evaluate_graph(task, graph, seed=0) == 1
        second = app.last_graph_evaluation["nq_corpus_evidence_audit"]
        assert len(tool.calls) == 2
        assert first["search_calls_used"] == second["search_calls_used"] == 1
        assert first["trajectory_id"] != second["trajectory_id"]
        assert json.dumps(task.metadata, sort_keys=True) == primary_metadata
        assert app.last_graph_evaluation["nq_corpus_submission"]["valid"]
    finally:
        app.close()


@pytest.mark.parametrize("answer", [supported("invented-evidence"), insufficient()])
def test_counterfactual_invalid_evidence_and_abstention_cannot_earn_answer_reward(answer):
    app, task, graph, _ = branch_app([SEARCH, answer])
    try:
        assert app.evaluate_graph(task, graph, seed=0) == 0
    finally:
        app.close()


def test_counterfactual_retrieval_outage_is_not_a_scored_training_zero():
    class BrokenSearch(SearchTool):
        def execute(self, arguments):
            raise ConnectionError("offline corpus unavailable")

    app, task, graph, _ = branch_app([SEARCH, insufficient()], BrokenSearch())
    try:
        with pytest.raises(GraphEvaluationIncompleteError, match="retrieval did not complete"):
            app.evaluate_graph(task, graph, seed=0)
    finally:
        app.close()
