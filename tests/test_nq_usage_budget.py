"""NQ corpus requests use actual provider usage; all responses here are offline fixtures."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.application import (
    AdaptiveApplicationResult,
    GraphEvaluationIncompleteError,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.budget_policy import LEGACY_ALLOCATION_FIELDS, POLICY
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.contracts import ExecutionReport
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.evaluation import from_adaptive_result
from selfplay_graph_flowsteer.llm import MockBackend, WorkerUsageDispatchStopped
from selfplay_graph_flowsteer.observability import MultiAnswerExactMatchVerifier, TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger, worker_usage_scope

from .test_hotpot_usage_budget import GatewayWorker, policy, prompt, step
from .test_hotpot_usage_budget import gateway_slot as gateway_slot
from .test_nq_corpus_evidence import SEARCH, SearchTool, context, node, run, supported
from .test_nq_formal_promotion import branch_app
from .test_nq_improvements import selection

pytestmark = pytest.mark.usefixtures("gateway_slot")

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What element did Curie name after Poland?"


@pytest.fixture
def account(tmp_path):
    ledger = WorkerUsageLedger(tmp_path / "usage.sqlite3", question_attempt_id="nq-question",
                               threshold=240000, dataset="nq_open")
    yield ledger
    ledger.close()


def nq_task():
    return TaskSpec("nq/engineering", QUESTION, reference=["Polonium"], metadata={
        "dataset": "nq_open", "evidence_mode": "corpus_tool", "original_question": QUESTION,
    })


def formal_config(tmp_path):
    loaded = load_adaptive_config(ROOT / "configs/formal_training.toml", validate=False)
    return replace(loaded, canvas=replace(loaded.canvas, submission_journal_dir=str(tmp_path)))


@pytest.mark.parametrize("filename", ["formal_training.toml", "nq_corpus_eval.toml"])
def test_current_nq_entries_enable_shared_usage_without_changing_routes(filename):
    loaded = load_adaptive_config(ROOT / "configs" / filename, validate=False)
    assert loaded.canvas.worker_usage_policy("nq_open") == policy()
    assert loaded.canvas.token_budget_for_dataset("nq_open")[1] == 240000
    assert not hasattr(loaded.canvas, 'remaining_time_admission_enabled')
    assert not hasattr(loaded.canvas, 'remaining_token_admission_enabled')
    contract = loaded.model_manifest()["execution_semantics"]["worker_usage_accounting_contract"]["nq_open"]
    assert contract["budget_threshold"] == 240000
    assert contract["counted_roles"] == ["worker"]
    assert not any(contract[key] for key in ("predicted_admission", "per_execution_allocations", "closure_reserves"))
    assert loaded.retrieval.nq_policy.max_search_calls_per_task == 4
    assert loaded.retrieval.nq_policy.max_submission_repairs == 1


def test_search_selection_and_evidence_repair_charge_one_account(account):
    worker = GatewayWorker([
        (SEARCH, 100, 10), (supported("forged"), 200, 20),
        ("invalid selector JSON", 300, 30), (supported(), 400, 40),
    ])
    evidence, tool = context(answer_selection_enabled=True), SearchTool()
    executor = ModelAgentExecutor(worker, tools={"search": tool}, nq_evidence_context=evidence)
    current = node()
    current.metadata.update({key: 1 for key in LEGACY_ALLOCATION_FIELDS})
    current.metadata["_runtime_budget_kind"] = "short_qa_request_credit_v1"
    with worker_usage_scope(account, agent_id="a", execution_id="first"):
        result = run(executor, current)
    assert result.answer == "Polonium" and result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert account.status()["confirmed_used"] == 1100
    assert account.status()["attempt_count"] == 4 and len(tool.calls) == 1
    assert all(key not in current.metadata for key in LEGACY_ALLOCATION_FIELDS)
    assert current.metadata["_runtime_budget_kind"] == POLICY
    assert [request["max_tokens"] for request in worker.calls] == [16384, 16384, 2048, 2048]
    assert "nq_evidence_repair" in {item["stage"] for item in result.protocol_diagnostics}


def test_two_agents_model_switch_and_format_repair_share_account(account):
    evidence, tool = context(), SearchTool()
    first = GatewayWorker([(SEARCH, 100, 10), (supported(), 200, 20)])
    second = GatewayWorker([(SEARCH, 300, 30), ("{broken JSON", 400, 40), (supported(), 500, 50)],
                           route="another-model")
    for name, worker in (("first", first), ("replacement", second)):
        with worker_usage_scope(account, agent_id=name, execution_id=name):
            result = run(ModelAgentExecutor(worker, tools={"search": tool}, nq_evidence_context=evidence),
                         node(name))
        assert result.answer == "Polonium" and result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert len(tool.calls) == 2 and account.status()["confirmed_used"] == 1650
    assert account.status()["attempt_count"] == 5
    rows = account._db.execute("SELECT agent_id, model FROM attempts ORDER BY started_at").fetchall()
    assert [row[0] for row in rows] == ["first", "first", "replacement", "replacement", "replacement"]
    assert rows[-1][1] == "another-model"


@pytest.mark.parametrize("last_input,last_output,reason", [
    (None, 10, "worker_usage_unsettled_limit"),
    (239000, 1000, "worker_usage_threshold_reached"),
    (239000, 5000, "worker_usage_threshold_reached"),
])
def test_stopped_selector_preserves_grounded_answer(account, last_input, last_output, reason):
    worker = GatewayWorker([(SEARCH, None if last_input is None else 0, 0),
                            (supported(), last_input, last_output)])
    evidence = context(answer_selection_enabled=True)
    with worker_usage_scope(account, agent_id="a", execution_id="1"):
        result = run(ModelAgentExecutor(worker, tools={"search": SearchTool()}, nq_evidence_context=evidence))
    assert len(worker.calls) == account.status()["attempt_count"] == 2
    assert result.answer == "Polonium" and result.raw_response == supported()
    assert result.runtime_tool_evidence["nq_corpus"]["valid"]
    audit = result.runtime_tool_evidence["nq_answer_selection"]
    assert audit["reason"] == reason and audit["draft_preserved"] and audit["no_request_dispatched"]
    assert account.status()["stop_reason"] == reason


def test_selector_can_be_the_last_legal_request(account):
    worker = GatewayWorker([(SEARCH, 100, 0), (supported(), 238900, 0), (selection(), 4500, 500)])
    with worker_usage_scope(account, agent_id="a", execution_id="1"):
        result = run(ModelAgentExecutor(worker, tools={"search": SearchTool()},
                                       nq_evidence_context=context(answer_selection_enabled=True)))
    assert result.runtime_tool_evidence["nq_answer_selection"]["accepted"]
    assert result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert account.status()["confirmed_used"] == 244000
    assert account.status()["confirmed_overshoot"] == 4000
    assert len(worker.calls) == 3


def test_exhausted_repair_keeps_invalid_evidence_rejected(account):
    worker = GatewayWorker([(SEARCH, 100, 0), (supported("forged"), 239000, 5000)])
    with worker_usage_scope(account, agent_id="a", execution_id="1"):
        result = run(ModelAgentExecutor(worker, tools={"search": SearchTool()}, nq_evidence_context=context()))
    assert not result.runtime_tool_evidence["nq_corpus"]["valid"]
    assert result.raw_response == supported("forged") and len(worker.calls) == 2
    assert any(item.get("no_request_dispatched") and item.get("rejection_reason") == "worker_usage_threshold_reached"
               for item in result.protocol_diagnostics)


def test_search_failure_cannot_refund_model_usage(account):
    class BrokenSearch(SearchTool):
        def execute(self, arguments):
            raise ConnectionError("offline fixture")

    worker = GatewayWorker([(SEARCH, 239000, 5000)])
    evidence = context()
    with (worker_usage_scope(account, agent_id="a", execution_id="1"),
          pytest.raises(WorkerUsageDispatchStopped)):
        run(ModelAgentExecutor(worker, tools={"search": BrokenSearch()}, nq_evidence_context=evidence))
    assert account.status()["confirmed_used"] == 244000 and len(worker.calls) == 1
    assert evidence.audit()["calls"][0]["status"] == "error"


def test_revisions_cache_and_new_agent_bypass_old_allocations(tmp_path, account, monkeypatch):
    registry = default_dataset_action_registry(("search",), nq_evidence_mode="corpus_tool")
    worker = GatewayWorker([(SEARCH, 100, 10), (supported(), 200, 20),
                            (supported(), 238600, 70), (SEARCH, 100, 0), (supported(), 4000, 900)])
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker, tools={"search": SearchTool()}, action_registry=registry))
    runtime.worker_usage_ledger = account
    runtime.configure_nq_corpus(task_id="nq/engineering", policy=formal_config(tmp_path).retrieval.nq_policy,
                                run_id="nq-question", trajectory_id="nq-question")
    cfg = replace(formal_config(tmp_path).canvas.for_dataset('nq_open'))
    canvas = GraphCanvas(task=QUESTION, dataset="nq_open", runtime=runtime, config=cfg,
                         action_adapter=registry.resolve(nq_task()))
    canvas.run_id = "nq-question"
    def forbidden(*args, **kwargs):
        pytest.fail("NQ shared usage reached a predictive estimator")
    for method in ("estimate_execution_tokens", "estimate_execution_s", "estimate_new_agent_tokens", "estimate_new_agent_s"):
        monkeypatch.setattr(runtime, method, forbidden, raising=False)
    assert step(canvas, dict(action="add_agent", agent_id="first")).accepted
    assert step(canvas, prompt("first", "subtask")).accepted
    cached, hit, token_in, token_out = runtime._run_agent(
        task=canvas.worker_task, graph=canvas.graph, agent_id="first", upstream=[], peers=[], revision=False,
        report=ExecutionReport(), reason_codes=[])
    assert hit and cached.answer == "Polonium" and token_in == token_out == 0
    assert account.status()["attempt_count"] == 2
    assert step(canvas, prompt("first", "subtask", objective="Recheck the evidence")).accepted
    assert canvas.total_tokens == 239000
    assert step(canvas, dict(action="delete_agent", target="first")).accepted
    assert step(canvas, dict(action="add_agent", agent_id="last")).accepted
    # A tiny old allocation and large input must not shrink this request.
    canvas.graph.nodes["last"].metadata.update({key: 1 for key in LEGACY_ALLOCATION_FIELDS})
    canvas.worker_task = "Public question:\n" + QUESTION + "\n" + "Public context. " * 18000
    assert step(canvas, prompt("last")).accepted
    assert canvas.total_tokens == 244000
    assert worker.calls[-1]["max_tokens"] == 16384
    assert not step(canvas, dict(action="run_agent", target="last")).accepted
    submitted = step(canvas, dict(action="finish", target="last"), finish=True)
    assert submitted.accepted, submitted.feedback
    assert canvas.submission_receipt.submitted_answer_snapshot == "Polonium"
    assert canvas.submission_receipt.worker_tokens_used == 244000 and len(worker.calls) == 5
    assert all(key not in canvas.graph.nodes["last"].metadata for key in LEGACY_ALLOCATION_FIELDS)


@pytest.mark.parametrize("unknown", [False, True])
def test_formal_solver_scores_answer_and_records_usage_separately(tmp_path, unknown):
    loaded = formal_config(tmp_path)
    loaded = replace(loaded, retrieval=replace(loaded.retrieval, nq_policy=replace(
        loaded.retrieval.nq_policy, answer_selection_enabled=True)))
    registry = default_dataset_action_registry(("search",), nq_evidence_mode="corpus_tool")
    worker = GatewayWorker([(SEARCH, None if unknown else 0, 0),
                            (supported(), None if unknown else 239000, 5000)])
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker, tools={"search": SearchTool()}, action_registry=registry))
    director = MockBackend([json.dumps(dict(action="add_agent", agent_id="solver")),
                            json.dumps(prompt("solver")), json.dumps(dict(action="finish", target="solver"))])
    solver = AdaptiveWorkflowSolver(
        director_backend=director, runtime=runtime, verifier=MultiAnswerExactMatchVerifier(),
        answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)), action_registry=registry,
        director_prompt_variant="v3", canvas_config=loaded.canvas,
        nq_evidence_mode="corpus_tool", nq_policy=loaded.retrieval.nq_policy)
    task = nq_task()
    solved = solver.solve(task, run_id="nq-shared-usage-engineering")
    record = from_adaptive_result(AdaptiveApplicationResult("nq-shared-usage-engineering", task, solved, (), None, "fixture"))
    assert record.passed and solved.outcome_decision.status == "scored"
    assert record.worker_usage["usage_complete"] is not unknown
    assert record.worker_usage["dispatch_policy_valid"]
    assert record.worker_usage["confirmed_used"] == (5000 if unknown else 244000)
    assert record.worker_usage["unsettled_attempt_count"] == (2 if unknown else 0)
    assert record.worker_usage["dataset"] == "nq_open"
    assert len(worker.calls) == 2 and len(director.calls) == 3
    runtime.close_worker_usage_ledger()


@pytest.mark.parametrize("unknown", [False, True])
def test_training_branches_use_independent_accounts_and_skip_estimators(tmp_path, monkeypatch, unknown):
    app, task, graph, tool = branch_app([])
    app.config = replace(app.config, canvas=formal_config(tmp_path).canvas)
    worker = GatewayWorker([(SEARCH, None if unknown else 0, 0),
                            (supported(), None if unknown else 239000, 5000)] * 2)
    app.runtime.executor.backend = worker
    def forbidden(*args, **kwargs):
        pytest.fail("NQ training branch reached a predictive estimator")
    monkeypatch.setattr(app.runtime, 'estimate_execution_tokens', forbidden, raising=False)
    states = []
    try:
        for _ in range(2):
            assert app.evaluate_graph(task, graph, seed=0) == 1
            states.append(app.last_graph_evaluation["worker_usage"])
        assert states[0]["question_attempt_id"] != states[1]["question_attempt_id"]
        assert all(state["confirmed_used"] == (5000 if unknown else 244000) for state in states)
        assert all(state["usage_complete"] is not unknown for state in states)
        assert len(tool.calls) == 2 and len(worker.calls) == 4
    finally:
        app.close()


def test_training_branch_exhaustion_without_valid_answer_is_incomplete(tmp_path):
    app, task, graph, _ = branch_app([])
    app.config = replace(app.config, canvas=formal_config(tmp_path).canvas)
    worker = GatewayWorker([(SEARCH, 0, 0), (supported("forged"), 239000, 5000)])
    app.runtime.executor.backend = worker
    try:
        with pytest.raises(GraphEvaluationIncompleteError, match="no valid evidence-backed answer"):
            app.evaluate_graph(task, graph, seed=0)
        assert app.last_graph_evaluation["worker_usage"]["confirmed_used"] == 244000
        assert not app.last_graph_evaluation["nq_corpus_submission"]["valid"]
        assert len(worker.calls) == 2
    finally:
        app.close()
