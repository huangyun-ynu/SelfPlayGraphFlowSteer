"""The formal V3 switch keeps evidence, task results, and training boundaries."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import application as app_module
from selfplay_graph_flowsteer.actions import ActionParser
from selfplay_graph_flowsteer.config import DEFAULT_DATASET_MAX_TOTAL_TOKENS
from selfplay_graph_flowsteer.execution_contract import bind_rollout_contract, validate_training_contract
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.webshop import WebShopEnvironmentVerifier
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor
from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.submission_contract import SUBMISSION_CONTRACT_VERSION
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle

from .test_webshop_formal_promotion import load_formal
from .test_unified_environments import shopping
from .test_unified_submission import step
from .test_webshop_native import NativeClient, ASIN1, report


DATASETS = tuple(DEFAULT_DATASET_MAX_TOTAL_TOKENS)


@pytest.mark.parametrize("dataset", DATASETS)
def test_formal_training_rejects_legacy_receipt_for_every_dataset(dataset):
    config = app_module.load_adaptive_config("configs/formal_training.toml", validate=False)
    semantics = config.model_manifest()["execution_semantics"]
    assert semantics["director_prompt_variant_by_dataset"][dataset] == "v3"
    assert semantics["submission_contract_version_by_dataset"][dataset] == "unified_submission_v1"
    row = {"dataset": dataset, "model_roles": {"execution_semantics": semantics},
           "submission_contract_version": "unified_submission_v1", "training_eligible": False}
    sample = TrainingSample("task", "run", (1, 2), (0, 1), 1, 0, metadata=row)
    rollouts = [SimpleNamespace(trajectory=SimpleNamespace(metadata=row))]
    batches = bind_rollout_contract((TrainingBatch("proposer", ()), TrainingBatch("solver", (sample,))), rollouts)
    validate_training_contract(*batches, expected=semantics)
    row["submission_contract_version"] = SUBMISSION_CONTRACT_VERSION
    with pytest.raises(ValueError, match="does not match dataset"):
        validate_training_contract(*batches, expected=semantics)


def action(name, **args):
    return json.dumps({"action": name, **args})


def shopping_worker_calls():
    def call(name, arguments):
        return json.dumps({"action_call": {"name": name, "arguments": arguments}})
    return [call("webshop_search", {"query": "product"}),
            call("webshop_click", {"target_id": f"open_product:0:{ASIN1}", "state_version": 1}),
            call("webshop_click", {"target_id": f"purchase:{ASIN1}", "state_version": 2,
                                   "purchase_evidence": {"verified_requirements": ["Product page inspected"],
                                                         "unresolved_constraints": []}}), report()]


@pytest.mark.parametrize("selected", ["agent_1", "agent_2"])
def test_webshop_v3_legacy_scheduler_commits_only_selected_session_without_worker_on_finish(tmp_path, monkeypatch, selected):
    config = load_formal("formal_training.toml", monkeypatch)
    config = replace(config, skillbank_enabled=False, pats=replace(config.pats, enabled=False),
        persist_runtime_updates=False, verifier="none",
        canvas=replace(config.canvas, submission_journal_dir=str(tmp_path / "submissions")),
        trace_path=tmp_path / "traces.jsonl", route_health_path=tmp_path / "routes.json",
        retrieval=replace(config.retrieval, enabled=False, nq_evidence_mode=None),
        aime_actions=replace(config.aime_actions, enabled=False))
    # The legacy scheduler allows staging alternatives; the formal bounded
    # scheduler now submits an existing ready candidate immediately (tested below).
    config = replace(config, webshop=replace(config.webshop, scheduling_policy="off"))
    client = NativeClient()
    monkeypatch.setattr(app_module, "WebShopHTTPClient", lambda *a, **k: client)
    worker = MockBackend(shopping_worker_calls() * 2)
    actions = []
    for target in ("agent_1", "agent_2"):
        actions.extend([
            action("add_agent"),
            action("set_prompt", target=target, role="Shopper", objective="Purchase the requested item",
                   scope="Public shopping observations", expected_output="A prepared purchase",
                   result_scope="task_result"),
            action("set_model", target=target, runtime_route="gpt"),
        ])
    actions.extend([action("delete_agent", target="agent_2" if selected == "agent_1" else "agent_1"),
                    action("finish", target=selected)])
    director = MockBackend(actions)
    generate = director.generate
    before_finish = []

    def observe_finish(*args, **kwargs):
        response = generate(*args, **kwargs)
        if json.loads(response.text).get("action") == "finish":
            assert not client.commits  # Buy Now was staged, not purchased early.
            before_finish.append(len(worker.calls))
        return response

    monkeypatch.setattr(director, "generate", observe_finish)
    app = app_module.create_adaptive_application(config, mock=True,
        director_backend=director, worker_backend=worker, distiller_backend=MockBackend([]))
    try:
        life = app.runtime.executor.tools["webshop_search"].lifecycle
        assert isinstance(life, NativeWebShopLifecycle) and not life.require_native_actions
        adapter = app.solver.action_registry.get("webshop")
        assert adapter.action_budget_policy == "shared_total_v1" and adapter.total_action_budget == 16
        result = app.solve("Buy a product", task_id="v3-shop", metadata={"dataset": "webshop", "goal_id": "goal-1"})
        run = result.solver_result.director_run
        assert run.finished, [turn.feedback for turn in run.turns]
        assert before_finish == [len(worker.calls)] == [8]
        assert len(client.commits) == 1 and client.commits[0][0] == ("s1" if selected == "agent_1" else "s2")
        assert run.submission_receipt.payload["purchased"]
        assert run.submission_receipt.output_agent_id == selected
        assert app.solver.active_canvas.history[-1].executed_agents == []
        assert not any(json.loads(t.model_action).get("action") == "set_output" for t in run.turns)
        assert not ActionParser(unified=True).parse_policy_output(action("set_output", target=selected)).action.valid
        # Training interventions replay the graph with a fresh environment.
        worker.responses.extend(shopping_worker_calls())
        app.solver.verifier = WebShopEnvironmentVerifier()
        value = app.evaluate_graph(TaskSpec("v3-shop", "Buy a product",
            metadata={"dataset": "webshop", "goal_id": "goal-1"}), app.solver.active_canvas.graph, seed=9)
        assert value == 1.0
        assert len(client.commits) == 2 and client.commits[1][0] == "s3"
    finally:
        app.close()


def test_formal_webshop_finishes_ready_purchase_and_replays_training_intervention(tmp_path, monkeypatch):
    config = load_formal("formal_training.toml", monkeypatch)
    assert config.webshop.scheduling_policy == "bounded_research_v1"
    assert config.webshop.purchase_budget_policy == "completion_reserve_v2"
    assert config.webshop.worker_memory_policy == "factual_memory_v2"
    config = replace(config, skillbank_enabled=False, pats=replace(config.pats, enabled=False),
        persist_runtime_updates=False, verifier="none",
        canvas=replace(config.canvas, submission_journal_dir=str(tmp_path / "submissions")),
        trace_path=tmp_path / "traces.jsonl", route_health_path=tmp_path / "routes.json",
        retrieval=replace(config.retrieval, enabled=False, nq_evidence_mode=None),
        aime_actions=replace(config.aime_actions, enabled=False))
    client = NativeClient()
    monkeypatch.setattr(app_module, "WebShopHTTPClient", lambda *a, **k: client)
    worker = MockBackend(shopping_worker_calls())
    director = MockBackend([
        action("add_agent"),
        action("set_prompt", target="agent_1", role="Shopper", objective="Purchase the requested item",
               scope="Public shopping observations", expected_output="A prepared purchase", result_scope="task_result"),
        action("set_model", target="agent_1", runtime_route="gpt"),
        action("add_agent"),  # A ready purchase should forbid unnecessary new work.
        action("finish", target="agent_1"),
    ])
    generate = director.generate
    before_finish = []

    def observe_finish(*args, **kwargs):
        response = generate(*args, **kwargs)
        if json.loads(response.text).get("action") == "finish":
            assert not client.commits
            before_finish.append(len(worker.calls))
        return response

    monkeypatch.setattr(director, "generate", observe_finish)
    app = app_module.create_adaptive_application(config, mock=True,
        director_backend=director, worker_backend=worker, distiller_backend=MockBackend([]))
    try:
        result = app.solve("Buy a product", task_id="formal-v3-shop",
            metadata={"dataset": "webshop", "goal_id": "goal-1"})
        run = result.solver_result.director_run
        assert run.finished and run.submission_receipt.payload["purchased"]
        assert before_finish == [len(worker.calls)]
        assert len(client.commits) == 1 and client.commits[0][0] == "s1"
        assert app.solver.active_canvas.history[-1].executed_agents == []
        assert any(step.rejection_code == "webshop_scheduling_boundary"
                   for step in app.solver.active_canvas.history)
        worker.responses.clear()
        worker.responses.extend(shopping_worker_calls())
        app.solver.verifier = WebShopEnvironmentVerifier()
        score = app.evaluate_graph(TaskSpec("formal-v3-shop", "Buy a product",
            metadata={"dataset": "webshop", "goal_id": "goal-1"}), app.solver.active_canvas.graph, seed=9)
        assert score == 1.0 and len(client.commits) == 2 and client.commits[1][0] == "s2"
    finally:
        app.close()


def test_real_exhausted_trajectory_cannot_create_another_empty_session(tmp_path):
    from .test_webshop_real_trajectory_regression import replay
    c, life, client, backend = replay(tmp_path, "00369")
    assert c.runtime.shared_tool_budget_status(16)["remaining"] == 0
    assert "add_agent" not in c.control_snapshot()["allowed_actions"]
    calls = len(backend.calls)
    result = step(c, {"action": "add_agent", "agent_id": "empty_session"})
    assert not result.accepted and result.rejection_code == "webshop_tool_budget_exhausted"
    assert "empty_session" not in c.graph.nodes
    assert len(backend.calls) == calls
    assert step(c, {"action": "finish", "target": "agent_1"}, True).accepted


def test_real_sheet_search_delegation_remains_a_local_evidence_responsibility(tmp_path):
    # Exact public task and first accepted SET_PROMPT from the fresh failed
    # goal-00544 paired trial. Previously both prompts ordered this subtask to buy.
    from selfplay_graph_flowsteer.delegation import compile_delegation
    from .test_webshop_native import report
    public_task = "i am looking for a cotton sheet set for a light blue king size bed, and price lower than 120.00 dollars"
    fields = dict(role="Product Searcher",
        objective="Search for cotton sheet sets that are light blue, king size, and priced under $120.00",
        scope="subtask", expected_output="List of matching products with prices and product links")
    c, backend, life, client = shopping(tmp_path, [report("No candidates inspected yet; evidence remains unresolved")])
    c.task = c.worker_task = public_task
    life.require_native_actions = False
    native = c.runtime.executor
    c.runtime.executor = ModelAgentExecutor(backend, tools=native.tools, action_registry=native.action_registry,
        webshop_worker_guidance_policy="merged_checklist_v1")
    assert step(c, {"action": "add_agent", "agent_id": "searcher"}).accepted
    outcome = step(c, dict(action="set_prompt", target="searcher", result_scope="subtask", **fields))
    assert outcome.accepted, outcome.feedback
    assert "Expected output: " + fields["expected_output"] in c.graph.nodes["searcher"].prompt
    assert "Your objective is to complete a purchase" not in c.graph.nodes["searcher"].prompt
    system = backend.calls[0]["messages"][0]["content"]
    assert "WebShop decision order" not in system
    assert "local responsibility" in system
    assert not client.commits and not client.calls
    assert "local_result_only" in c.submission_assessment("searcher")["blockers"]
    for scope in (None, "task_result"):
        compilation, issue = compile_delegation(fields, dataset="webshop", result_scope=scope, public_task=public_task)
        assert issue is None
        assert compilation.director_fields["expected_output"].startswith("Stage purchase")
        assert "Your objective is to complete a purchase" in compilation.prompt
