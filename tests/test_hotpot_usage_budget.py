"""Engineering fixtures exercise physical dispatches; no formal usage is fabricated."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.application import AdaptiveApplicationResult, AdaptiveSolverApplication, load_adaptive_config
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.budget_policy import LEGACY_ALLOCATION_FIELDS, POLICY
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.deadline import RolloutDeadline
from selfplay_graph_flowsteer.evaluation import from_adaptive_result
from selfplay_graph_flowsteer.observability import FlowSteerQAVerifier
from selfplay_graph_flowsteer.contracts import ExecutionReport
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.submission_contract import _director_call_context
from selfplay_graph_flowsteer.unified_contract import PROTOCOL
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger, UsageDispatchStopped, worker_usage_scope
from pathlib import Path
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.hotpot_answer_contract import HOTPOT_RESULT_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def task(reference="Paris"):
    return TaskSpec("hotpotqa/synthetic", "Evidence:\nFrance's capital is Paris.\n\nQuestion: What is France's capital?",
                    reference=reference, task_type="factual_qa", metadata={"dataset": "hotpotqa", "verifier": "flowsteer_qa"})


def artifact(answer="Paris"):
    return json.dumps(dict(zip(HOTPOT_RESULT_FIELDS, [
        ["France's capital is Paris."], "The supplied evidence identifies the capital.",
        0.95, [], [], answer,
    ])))


def policy(threshold=240000):
    return dict(policy=POLICY, start_threshold=threshold, accounting_scope="question_attempt",
                max_inflight_requests=1, unknown_usage_policy="continue_bounded", max_unsettled_attempts=2)


def config(tmp_path, threshold=240000, **kwargs):
    return CanvasConfig(submission_protocol=PROTOCOL, submission_journal_dir=str(tmp_path),
                        max_rounds=40, max_total_tokens=threshold,
                        max_total_tokens_by_dataset={"hotpotqa": threshold},
                        worker_token_budget_by_dataset={"hotpotqa": policy(threshold)}, **kwargs)


def ledger(tmp_path, name="question", threshold=240000):
    return WorkerUsageLedger(tmp_path / (name + ".sqlite3"), question_attempt_id=name,
                             threshold=threshold, dataset="hotpotqa")


@pytest.fixture
def gateway_slot(monkeypatch):
    @contextmanager
    def slot(config, deadline, *, request_budget_cap_s, attempt):
        yield SimpleNamespace(route=config.route_name, timeout_s=request_budget_cap_s,
                              request_started_monotonic=time.monotonic(),
                              request_budget_s=request_budget_cap_s, queue_wait_s=0, priority="primary")
    monkeypatch.setattr(llm, "_request_slot", slot)


class GatewayWorker:
    """Returns synthetic provider responses through the actual Chat gateway boundary."""
    def __init__(self, responses, route="deepseek"):
        self.responses = iter(responses)
        self.calls = []
        self.config = SimpleNamespace(route_name=route, stream=False)

    def generate(self, messages, *, role, actions=(), max_tokens=None, **kwargs):
        assert not hasattr(llm, '_TOKEN_CREDIT')
        request = dict(model=self.config.route_name, messages=messages, max_tokens=max_tokens or 16384)
        def create(**sent):
            self.calls.append(sent)
            text, token_in, token_out, *finish = next(self.responses)
            return SimpleNamespace(id="response-" + str(len(self.calls)), text=text,
                                   finish_reason=finish[0] if finish else "stop",
                                   usage=SimpleNamespace(prompt_tokens=token_in, completion_tokens=token_out))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        response = llm._openai_completion_attempt(client, self.config, request, None,
                                                  attempt=1, request_budget_cap_s=30)
        return llm.LLMResponse(text=response.text, model=self.config.route_name,
                               token_in=response.usage.prompt_tokens or 0,
                               token_out=response.usage.completion_tokens or 0,
                               metadata={"finish_reason": response.finish_reason})


def step(canvas, action, finish=False):
    call_id = f"test:{len(canvas.history)}"
    canvas.observe_submission_candidates(call_id)
    return canvas.step(json.dumps(action), director_context=(
        _director_call_context(canvas.run_id, call_id) if finish else None))


def prompt(name, scope="task_result", objective="Answer the public question"):
    return dict(action="set_prompt", target=name, role="Evidence analyst", objective=objective,
                scope="The provided documents", expected_output="A supported answer", result_scope=scope)


def add(canvas, name, scope="task_result"):
    assert step(canvas, dict(action="add_agent", agent_id=name)).accepted
    result = step(canvas, prompt(name, scope))
    assert result.accepted, result.feedback


def canvas(tmp_path, worker, **options):
    registry = default_dataset_action_registry(())
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker, action_registry=registry))
    runtime.worker_usage_ledger = ledger(tmp_path)
    current = task()
    result = GraphCanvas(task=current.prompt, dataset="hotpotqa", runtime=runtime,
                         action_adapter=registry.resolve(current), config=config(tmp_path, **options))
    result.run_id = "question"
    return result


def test_formal_policy_and_threshold_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0,1,2")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "engineering-test")
    # QA budget regressions do not depend on unrelated environment assets.
    from selfplay_graph_flowsteer.application import ALFWorldConfig, SWEConfig
    monkeypatch.setattr(ALFWorldConfig, "validate", lambda self: None)
    monkeypatch.setattr(SWEConfig, "validate", lambda self: None)
    loaded = load_adaptive_config(ROOT / "configs/formal_training.toml")
    assert loaded.canvas.worker_usage_policy("hotpotqa") == policy()
    assert not hasattr(loaded.canvas, 'remaining_time_admission_enabled')
    assert not hasattr(loaded.canvas, 'remaining_token_admission_enabled')
    assert loaded.additional_runtimes["deepseek"].max_tokens == 16384
    contract = loaded.model_manifest()["execution_semantics"]["worker_usage_accounting_contract"]["hotpotqa"]
    assert contract["budget_threshold"] == 240000
    assert contract["counted_roles"] == ["worker"]
    assert not any(contract[key] for key in ("predicted_admission", "per_execution_allocations", "closure_reserves"))
    with pytest.raises(ValueError, match="must match"):
        CanvasConfig(submission_protocol=PROTOCOL, worker_token_budget_by_dataset={"hotpotqa": policy(239999)})
    with pytest.raises(ValueError, match="one local"):
        CanvasConfig(submission_protocol=PROTOCOL,
                     worker_token_budget_by_dataset={"hotpotqa": dict(policy(), max_inflight_requests=2)})


def test_full_long_context_not_estimated_and_legal_overshoot_submits(tmp_path, gateway_slot, monkeypatch):
    worker = GatewayWorker([(artifact(), 239000, 0), (artifact(), 4500, 500)])
    c = canvas(tmp_path, worker)
    # These obsolete estimators must remain unreachable even when old flags are true.
    def forbidden(*args, **kwargs):
        pytest.fail("Predictive admission was invoked")
    for method in ("estimate_execution_tokens", "estimate_execution_s", "estimate_new_agent_s", "estimate_new_agent_tokens"):
        monkeypatch.setattr(c.runtime, method, forbidden, raising=False)
    c.rollout_deadline = RolloutDeadline(total_timeout_s=30, no_progress_timeout_s=30, request_timeout_s=30)
    c.worker_task = c.task = "Context:\n" + "Public evidence. " * 18000 + "\n\nQuestion: Name the capital."
    add(c, "first", "subtask")
    assert c.total_tokens == 239000
    assert step(c, dict(action="delete_agent", target="first")).accepted
    # This used to hit the 8192 graph-growth reserve and split the last 1000 tokens.
    add(c, "second")
    assert c.total_tokens == 244000
    assert len(worker.calls) == 2
    assert worker.calls[-1]["max_tokens"] == 16384
    assert json.loads(worker.calls[-1]["messages"][-1]["content"])["public_task_context"] == c.worker_task
    state = c.runtime.worker_usage_ledger.status()
    assert state["confirmed_overshoot"] == 4000 and state["dispatch_policy_valid"]
    assert state["stop_reason"] == "worker_usage_threshold_reached"
    assert not step(c, dict(action="run_agent", target="second")).accepted
    accepted = step(c, dict(action="finish", target="second"), finish=True)
    assert accepted.accepted, accepted.feedback
    assert c.submission_receipt.submitted_answer_snapshot == "Paris"
    assert c.submission_receipt.worker_tokens_used == 244000
    assert len(worker.calls) == 2  # FINISH cannot generate a hidden report or repair.
    c.runtime.close_worker_usage_ledger()


def test_nodes_models_revisions_and_cache_share_one_account(tmp_path, gateway_slot):
    first = GatewayWorker([(artifact(), 100, 10), (artifact(), 200, 20), (artifact(), 300, 30)])
    second = GatewayWorker([(artifact(), 400, 40)], route="another-model")
    c = canvas(tmp_path, first)
    account = c.runtime.worker_usage_ledger
    add(c, "first", "subtask")
    initial = c.total_tokens
    cached, hit, token_in, token_out = c.runtime._run_agent(
        task=c.worker_task, graph=c.graph, agent_id="first", upstream=[], peers=[], revision=False,
        report=ExecutionReport(), reason_codes=[])
    assert hit and token_in == token_out == 0
    assert cached.answer == "Paris" and account.status()["confirmed_used"] == initial
    assert step(c, prompt("first", "subtask", objective="Recheck the provided evidence")).accepted
    assert step(c, dict(action="delete_agent", target="first")).accepted
    c.runtime.executor.backend = second
    add(c, "replacement", "task_result")
    assert c.runtime.worker_usage_ledger is account
    assert account.status()["confirmed_used"] == 110 + 220 + 440
    rows = account._db.execute("SELECT agent_id, model, execution_id FROM attempts ORDER BY started_at").fetchall()
    assert [r[0] for r in rows] == ["first", "first", "replacement"]
    assert rows[-1][1] == "another-model"
    assert len({r[2] for r in rows}) == 3
    c.runtime.close_worker_usage_ledger()


def test_stale_allocations_removed_and_format_repair_is_billed(tmp_path, gateway_slot):
    malformed = json.dumps({**json.loads(artifact()), "confidence": "high"})
    worker = GatewayWorker([(malformed, 10000, 100), (artifact(), 12000, 200)])
    c = canvas(tmp_path, worker)
    assert step(c, dict(action="add_agent", agent_id="solver")).accepted
    node = c.graph.nodes["solver"]
    node.metadata.update({field: 1 for field in LEGACY_ALLOCATION_FIELDS})
    node.metadata["_runtime_budget_kind"] = "short_qa_request_credit_v1"
    result = step(c, prompt("solver"))
    assert result.accepted, result.feedback
    assert len(worker.calls) == 2
    assert c.total_tokens == 22300
    node = c.graph.nodes["solver"]
    assert all(field not in node.metadata for field in LEGACY_ALLOCATION_FIELDS)
    assert node.metadata["_runtime_budget_kind"] == POLICY
    assert c.runtime.worker_usage_ledger.status()["attempt_count"] == 2
    c.runtime.close_worker_usage_ledger()


def test_invalid_response_at_threshold_cannot_dispatch_repair_or_invent_answer(tmp_path, gateway_slot):
    worker = GatewayWorker([("{broken JSON", 239000, 5000)])
    c = canvas(tmp_path, worker)
    add(c, "solver")
    assert len(worker.calls) == 1
    assert c.runtime.worker_usage_ledger.status()["confirmed_used"] == 244000
    assert not c.submission_assessment("solver")["submit_ready"]
    assert not step(c, dict(action="finish", target="solver"), finish=True).accepted
    assert c.submission_receipt is None
    c.runtime.close_worker_usage_ledger()


def test_truncated_response_keeps_qa_recovery_contract_without_token_reserves(tmp_path, gateway_slot):
    worker = GatewayWorker([(artifact(), 10000, 100, "length"), (artifact(), 12000, 200)])
    c = canvas(tmp_path, worker)
    add(c, "solver")
    assert len(worker.calls) == 2
    recovery = json.loads(worker.calls[-1]["messages"][-1]["content"])
    assert recovery["previous_response"] == artifact()
    assert recovery["previous_attempt_issue"] == "truncated_final_response"
    assert c.total_tokens == 22300
    assert c.submission_assessment("solver")["submit_ready"]
    c.runtime.close_worker_usage_ledger()


@pytest.mark.parametrize("surface", ["chat_completions", "responses"])
def test_unknown_usage_stops_after_two_and_preserves_partial_counters(tmp_path, gateway_slot, surface):
    account = ledger(tmp_path)
    count = []
    def create(**request):
        count.append(request)
        usage = (SimpleNamespace(prompt_tokens=None, completion_tokens=5)
                 if surface == "chat_completions" else SimpleNamespace(input_tokens=None, output_tokens=5))
        return SimpleNamespace(id="unknown-response", usage=usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                             responses=SimpleNamespace(create=create))
    method = llm._openai_completion_attempt if surface == "chat_completions" else llm._openai_response_attempt
    request = dict(model="test-model", **({"messages": []} if surface == "chat_completions" else {"input": []}))
    with worker_usage_scope(account, agent_id="no-tools", execution_id="1"):
        for index in (1, 2):
            method(client, SimpleNamespace(route_name="test-route", stream=False), request, None,
                   attempt=index, request_budget_cap_s=10)
        with pytest.raises(llm.WorkerUsageDispatchStopped, match="worker_usage_unsettled_limit"):
            method(client, SimpleNamespace(route_name="test-route", stream=False), request, None,
                   attempt=3, request_budget_cap_s=10)
    assert len(count) == 2 and account.status()["confirmed_used"] == 10
    assert not account.status()["usage_complete"]
    assert account.status()["unsettled_attempt_count"] == 2
    account.close()


def test_timeout_is_unknown_then_retry_has_a_separate_attempt(tmp_path, gateway_slot):
    account = ledger(tmp_path)
    count = []
    events = []
    def create(**request):
        count.append(request)
        if len(count) == 1:
            raise TimeoutError("Fixture request dispatched then timed out")
        return SimpleNamespace(id="known", usage=SimpleNamespace(prompt_tokens=60, completion_tokens=40))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with llm._capture_request_events(events, role="worker"), worker_usage_scope(account, agent_id="a", execution_id="1"):
        with pytest.raises(Exception, match="timed out"):
            llm._openai_completion_attempt(client, SimpleNamespace(route_name="test", stream=False),
                                           {"model": "test", "messages": []}, None,
                                           attempt=1, request_budget_cap_s=10)
        llm._openai_completion_attempt(client, SimpleNamespace(route_name="test", stream=False),
                                       {"model": "test", "messages": []}, None,
                                       attempt=2, request_budget_cap_s=10)
    assert account.status()["confirmed_used"] == 100
    assert not account.status()["usage_complete"]
    rows = account._db.execute("SELECT attempt_id, state, input_tokens FROM attempts").fetchall()
    assert {r[1] for r in rows} == {"unknown", "complete"}
    assert len({r[0] for r in rows}) == 2
    assert all(event.get("worker_usage_attempt_id") for event in events)
    account.close()


@pytest.mark.parametrize("surface", ["chat_completions", "responses"])
def test_local_serialization_error_does_not_create_a_sent_attempt(tmp_path, gateway_slot, surface):
    account = ledger(tmp_path)
    calls = []
    def create(**request):
        calls.append(request)
        pytest.fail("An invalid local payload reached the provider")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                             responses=SimpleNamespace(create=create))
    method = llm._openai_completion_attempt if surface == "chat_completions" else llm._openai_response_attempt
    with worker_usage_scope(account, agent_id="a", execution_id="1"):
        with pytest.raises(TypeError, match="not JSON serializable"):
            method(client, SimpleNamespace(route_name="test", stream=False),
                   {"model": "test", "unserializable_local_field": object()}, None,
                   attempt=1, request_budget_cap_s=10)
    assert not calls and account.status()["attempt_count"] == 0
    assert account.status()["usage_complete"]
    account.close()


def test_settlement_idempotence_and_persistent_identity(tmp_path):
    account = ledger(tmp_path)
    attempt = account.begin(route="r", agent_id="a", execution_id="1", request={})
    account.settle(attempt, input_tokens=30, output_tokens=20)
    account.settle(attempt, input_tokens=30, output_tokens=20)
    assert account.status()["confirmed_used"] == 50
    with pytest.raises(ValueError, match="conflicting"):
        account.settle(attempt, input_tokens=31, output_tokens=20)
    account.close()
    reopened = ledger(tmp_path)
    assert reopened.status()["confirmed_used"] == 50
    assert reopened.status()["dataset"] == "hotpotqa"
    reopened.close()


@pytest.mark.parametrize("same_question", [True, False])
def test_same_question_serializes_and_different_questions_can_overlap(tmp_path, gateway_slot, same_question):
    a = ledger(tmp_path, "a")
    b = a if same_question else ledger(tmp_path, "b")
    first_entered, second_started, second_entered, release_first = [threading.Event() for _ in range(4)]
    def dispatch(account, index):
        def create(**request):
            if index == 1:
                first_entered.set()
                assert release_first.wait(3)
            else:
                second_entered.set()
            return SimpleNamespace(id=str(index), usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with worker_usage_scope(account, agent_id=str(index), execution_id=str(index)):
            if index == 2:
                second_started.set()
            return llm._openai_completion_attempt(client, SimpleNamespace(route_name="r", stream=False),
                                                   {"model": "m", "messages": []}, None,
                                                   attempt=1, request_budget_cap_s=10)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(dispatch, a, 1)
            assert first_entered.wait(2)
            second = pool.submit(dispatch, b, 2)
            assert second_started.wait(2)
            if same_question:
                assert not second_entered.wait(0.1)
                assert a.status()["inflight_request_count"] == 1
            else:
                assert second_entered.wait(2)
            release_first.set()
            first.result(timeout=3)
            second.result(timeout=3)
        assert a.status()["confirmed_used"] == (30 if same_question else 15)
        assert a.status()["attempt_count"] == (2 if same_question else 1)
    finally:
        release_first.set()
        a.close()
        if b is not a:
            b.close()


def test_queue_timeout_does_not_record_a_request_that_was_not_sent(tmp_path):
    account = ledger(tmp_path)
    with account.dispatch_slot():
        with ThreadPoolExecutor(max_workers=1) as pool:
            def wait():
                with account.dispatch_slot(timeout_s=0.01):
                    pytest.fail("Second request must not acquire the slot")
            with pytest.raises(TimeoutError, match="fixed request timeout"):
                pool.submit(wait).result(timeout=2)
    assert account.status()["attempt_count"] == 0
    account.close()


@pytest.mark.parametrize("token_in,token_out,complete", [(None, 7, False), (239000, 5000, True)])
def test_scored_alias_and_usage_completeness_are_recorded_independently(tmp_path, gateway_slot, token_in, token_out, complete):
    director = llm.MockBackend([json.dumps(dict(action="add_agent", agent_id="solver")),
                                json.dumps(prompt("solver")),
                                json.dumps(dict(action="finish", target="solver"))])
    worker = GatewayWorker([(artifact("City of Paris"), token_in, token_out)])
    registry = default_dataset_action_registry(())
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker, action_registry=registry))
    solver = AdaptiveWorkflowSolver(director_backend=director, runtime=runtime, verifier=FlowSteerQAVerifier(),
                                    answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
                                    action_registry=registry, director_prompt_variant="v3", canvas_config=config(tmp_path))
    current = task(["Paris", "City of Paris"])
    solved = solver.solve(current, run_id="reported-usage-alias")
    assert solved.verification.passed and solved.outcome_decision.status == "scored"
    app_result = AdaptiveApplicationResult("reported-usage-alias", current, solved, (), None, "fixture")
    record = from_adaptive_result(app_result)
    usage = record.worker_usage
    assert usage["confirmed_input_tokens"] == (token_in or 0) and usage["confirmed_output_tokens"] == token_out
    assert usage["usage_complete"] is complete and usage["unsettled_attempt_count"] == (0 if complete else 1)
    assert usage["dataset"] == "hotpotqa" and usage["question_attempt_id"] == "reported-usage-alias"
    assert usage["budget_policy"] == POLICY and usage["budget_accounting_scope"] == "question_attempt"
    assert record.token_cost == (token_in or 0) + token_out and record.passed
    assert usage["confirmed_overshoot"] == (4000 if complete else 0)
    assert record.trajectory["worker_usage"] == usage
    assert len(worker.calls) == 1 and len(director.calls) == 3
    runtime.close_worker_usage_ledger()


def test_exact_threshold_stops_new_requests_and_crashed_pending_stays_unknown(tmp_path):
    account = ledger(tmp_path)
    attempt = account.begin(route="r", agent_id="a", execution_id="1", request={})
    account.settle(attempt, input_tokens=239000, output_tokens=1000)
    with pytest.raises(UsageDispatchStopped, match="worker_usage_threshold_reached"):
        account.begin(route="r", agent_id="b", execution_id="2", request={})
    assert account.status()["attempt_count"] == 1
    account.close()
    other = ledger(tmp_path, "interrupted")
    other.begin(route="r", agent_id="a", execution_id="1", request={})
    other.close()
    reopened = ledger(tmp_path, "interrupted")
    assert reopened.status()["unsettled_attempt_count"] == 1
    assert not reopened.status()["usage_complete"]
    assert reopened.status()["inflight_request_count"] == 0
    reopened.close()


def test_connected_workers_and_revisions_repeat_context_in_one_account(tmp_path, gateway_slot):
    worker = GatewayWorker([(artifact(), 100, 10), (artifact(), 100, 10),
                            (artifact("City of Paris"), 100, 10), (artifact(), 100, 10)])
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker))
    runtime.worker_usage_ledger = ledger(tmp_path)
    graph = MultiAgentGraph()
    for name in ("reader", "integrator"):
        graph.add_agent(name)
        graph.set_prompt(name, "Identify the capital from the supplied evidence")
        # No action adapter and no tools: the budget is independent of both.
        graph.nodes[name].metadata["system_managed_contract"] = {"dataset": "hotpotqa"}
    graph.set_layer("integrator", 1)
    graph.set_relation("reader", "integrator", "directed")
    graph.set_output("integrator")
    public = task().prompt * 100
    first = runtime.execute(task=public, graph=graph)
    assert first.executed_agents == ["reader", "integrator"]
    changed = graph.set_prompt("reader", "Recheck the supported city in the public evidence")
    second = runtime.execute(task=public, graph=graph, dirty_agents=changed.dirty_agents)
    assert second.executed_agents == ["reader", "integrator"]
    assert runtime.worker_usage_ledger.status()["confirmed_used"] == 440
    assert runtime.worker_usage_ledger.status()["attempt_count"] == 4
    for request in worker.calls:
        assert json.loads(request["messages"][-1]["content"])["public_task_context"] == public
    runtime.close_worker_usage_ledger()


def test_full_graph_entry_skips_estimator_and_accepts_legal_overshoot(tmp_path, gateway_slot, monkeypatch):
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0,1,2")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "engineering-test")
    # QA budget regressions do not depend on unrelated environment assets.
    from selfplay_graph_flowsteer.application import ALFWorldConfig, SWEConfig
    monkeypatch.setattr(ALFWorldConfig, "validate", lambda self: None)
    monkeypatch.setattr(SWEConfig, "validate", lambda self: None)
    loaded = replace(load_adaptive_config(ROOT / "configs/formal_training.toml"), canvas=config(tmp_path))
    worker = GatewayWorker([(artifact(), 239000, 5000)])
    registry = default_dataset_action_registry(())
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker, action_registry=registry))
    solver = AdaptiveWorkflowSolver(director_backend=llm.MockBackend([]), runtime=runtime, verifier=FlowSteerQAVerifier(),
                                    answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
                                    action_registry=registry, director_prompt_variant="v3", canvas_config=loaded.canvas)
    application = AdaptiveSolverApplication(config=loaded, solver=solver, runtime=runtime,
                                            skillbank=None, skill_lifecycle=None)
    graph = MultiAgentGraph()
    graph.add_agent("solver")
    graph.set_prompt("solver", "Answer the public question using the supplied context")
    graph.set_output("solver")
    graph.nodes["solver"].metadata.update({field: 1 for field in LEGACY_ALLOCATION_FIELDS})
    def forbidden(*args, **kwargs):
        pytest.fail("Full graph entry invoked a predictive estimator")
    monkeypatch.setattr(runtime, 'estimate_execution_tokens', forbidden, raising=False)
    result = application.evaluate_graph(task(), graph, seed=0, return_verification=True)
    assert result["verification"].passed
    usage = application.last_graph_evaluation["worker_usage"]
    assert usage["confirmed_used"] == 244000 and usage["confirmed_overshoot"] == 4000
    assert usage["dataset"] == "hotpotqa"
    assert len(worker.calls) == 1
    application.close()


def test_legacy_hotpot_finish_records_reported_usage_and_keeps_legal_overshoot(tmp_path, gateway_slot):
    worker = GatewayWorker([(artifact(), 239000, 5000)])
    runtime = MultiAgentRuntime(ModelAgentExecutor(worker))
    runtime.worker_usage_ledger = ledger(tmp_path)
    c = GraphCanvas(task=task().prompt, dataset="hotpotqa", runtime=runtime, runtime_routes=("deepseek",),
                    config=replace(config(tmp_path), submission_protocol="legacy"))
    c.run_id = "question"
    assert step(c, dict(action="add_agent", agent_id="solver")).accepted
    assignment = prompt("solver")
    assignment.pop("result_scope")
    assert step(c, assignment).accepted
    assert step(c, dict(action="set_output", target="solver")).accepted
    c.graph.nodes["solver"].operation_policy_configured = True
    assert step(c, dict(action="set_model", target="solver", runtime_route="deepseek")).accepted
    accepted = step(c, dict(action="finish"), finish=True)
    assert accepted.accepted, accepted.feedback
    receipt = c.submission_receipt
    assert receipt.worker_budget_policy == POLICY
    assert receipt.worker_dispatch_valid and receipt.worker_usage_complete
    assert receipt.worker_usage_ledger_digest
    assert receipt.worker_tokens_used == 244000 and receipt.submitted_answer_snapshot == "Paris"
    assert len(worker.calls) == 1
    c.runtime.close_worker_usage_ledger()


def test_policy_overrides_preserve_defaults_and_aliases_cannot_create_two_accounts():
    cfg = CanvasConfig(worker_token_budget_by_dataset={"hotpot": policy()})
    assert cfg.worker_usage_policy("hotpotqa") == policy()
    assert cfg.worker_usage_policy("nq_open") == policy()
    assert cfg.worker_usage_policy("aime") == policy()
    with pytest.raises(ValueError, match="duplicate"):
        CanvasConfig(worker_token_budget_by_dataset={"hotpot": policy(), "hotpotqa": policy()})
