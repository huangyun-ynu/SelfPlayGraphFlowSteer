from __future__ import annotations

from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.application import AdaptiveSolverApplication
from selfplay_graph_flowsteer.canvas import GraphCanvas, TokenBudgetExceeded
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.dataset_actions import DatasetActionAdapter, DatasetActionRegistry
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ExecutionReport, ModelAgentExecutor, MultiAgentRuntime

from .helpers import RecordingExecutor

ADAPTER = DatasetActionAdapter("webshop", ("webshop",), (), 0, 0, 0)


class QuotedBackend:
    """Exercise the real gateway admission check without any provider request."""

    def __init__(self):
        self.dispatched = 0

    def generate(self, messages, *, role, actions=()):
        credit = llm._TOKEN_CREDIT.get()
        if credit is not None:
            llm._request_credit_admission(
                {"messages": messages, "max_tokens": 400_000}, credit
            )
        self.dispatched += 1
        return llm.LLMResponse(
            text='{"answer":"reported","summary":"Observed result","confidence":0.8}',
            model="fixture",
            token_in=2,
            token_out=3,
        )


def runtime_with_backend():
    backend = QuotedBackend()
    runtime = MultiAgentRuntime(
        ModelAgentExecutor(backend, action_registry=DatasetActionRegistry([ADAPTER]))
    )
    return runtime, backend


def shopping_graph():
    graph = MultiAgentGraph()
    graph.add_agent("shopper")
    graph.set_prompt("shopper", "Report the observed shopping result.")
    graph.nodes["shopper"].metadata["action_adapter"] = "webshop"
    graph.set_output("shopper")
    return graph


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("closure", [False, True])
def test_canvas_request_gate_follows_switch_in_both_phases(enabled, closure):
    runtime, backend = runtime_with_backend()
    canvas = GraphCanvas(
        task="Buy the requested product.",
        dataset="webshop",
        runtime=runtime,
        config=CanvasConfig(max_total_tokens=350_000, remaining_token_admission_enabled=enabled),
    )
    canvas.graph = shopping_graph()
    # A restored graph must not reactivate the gate after the user disables it.
    canvas.graph.nodes["shopper"].metadata.update(
        _runtime_token_credit=1,
        _runtime_reserved_closure_tokens=100,
        _runtime_budget_phase="exploration",
    )
    canvas.dirty_agents = {"shopper"}
    canvas.dirty_reasons = {
        "shopper": {"webshop_output_closure_required"} if closure else {"prompt_changed"}
    }

    report = canvas._execute_dirty()

    assert backend.dispatched == (0 if enabled else 1)
    artifact = report.artifacts["shopper"]
    blocked = any(d.get("no_request_dispatched") for d in artifact.protocol_diagnostics)
    assert blocked is enabled
    if not enabled:
        assert artifact.answer == "reported"
        assert canvas.total_tokens == 5
        assert artifact.webshop_progress["budget_partition"]["execution_credit"] is None
        assert "_runtime_token_credit" not in canvas.graph.nodes["shopper"].metadata


def test_runtime_disabled_gate_ignores_stale_credit():
    runtime, backend = runtime_with_backend()
    graph = shopping_graph()
    graph.nodes["shopper"].metadata.update(
        _runtime_webshop_request_admission_enabled=False,
        _runtime_token_credit=1,
    )
    report = runtime.execute(task="Buy a product.", graph=graph, dirty_agents=None)
    assert backend.dispatched == 1
    assert report.output == "reported"


def test_disabled_request_gate_keeps_actual_token_limit():
    canvas = GraphCanvas(
        task="Buy a product.",
        dataset="webshop",
        runtime=MultiAgentRuntime(RecordingExecutor()),
        config=CanvasConfig(max_total_tokens=4, remaining_token_admission_enabled=False),
    )
    canvas.graph = shopping_graph()
    canvas.dirty_agents = {"shopper"}
    with pytest.raises(TokenBudgetExceeded):
        canvas._execute_dirty()
    assert canvas.total_tokens == 5
    assert not canvas.dirty_agents


@pytest.mark.parametrize("enabled", [False, True])
def test_full_graph_evaluation_respects_request_gate_switch(enabled):
    runtime, backend = runtime_with_backend()
    application = object.__new__(AdaptiveSolverApplication)
    application.config = SimpleNamespace(
        canvas=CanvasConfig(
            remaining_token_admission_enabled=enabled,
            max_total_tokens_by_dataset={"webshop": 350_000},
        )
    )
    application.runtime = runtime
    application.solver = SimpleNamespace(
        verifier=SimpleNamespace(verify=lambda *_: SimpleNamespace(score=1.0))
    )
    task = TaskSpec("fixture", "Buy a product.", metadata={"dataset": "webshop"})
    if enabled:
        with pytest.raises(RuntimeError, match="could not admit Worker execution"):
            application.evaluate_graph(task, shopping_graph(), seed=0)
        assert backend.dispatched == 0
    else:
        assert application.evaluate_graph(task, shopping_graph(), seed=0) == 1.0
        assert backend.dispatched == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_full_graph_closure_does_not_reenable_disabled_gate(monkeypatch, enabled):
    runtime, backend = runtime_with_backend()
    runtime.full_graph_replay = True
    graph = shopping_graph()
    graph.nodes["shopper"].metadata["_runtime_webshop_request_admission_enabled"] = enabled
    runtime.artifacts["shopper"] = AgentArtifact("prior", "shopper", "unfinished")
    report = ExecutionReport(artifacts=dict(runtime.artifacts))
    monkeypatch.setattr(
        runtime,
        "_environment_lifecycles",
        lambda: (SimpleNamespace(closure_budget_context=lambda _: {"eligible": True}),),
    )
    runtime.complete_full_graph_webshop_output(
        task="Buy a product.", graph=graph, report=report, remaining_token_credit=1
    )
    assert backend.dispatched == (0 if enabled else 1)
    if not enabled:
        assert report.output == "reported"
        assert "_runtime_token_credit" not in graph.nodes["shopper"].metadata
