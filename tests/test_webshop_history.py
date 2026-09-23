from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.application import (
    WebShopConfig,
    create_adaptive_application,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, RoutedModelAgentExecutor
from selfplay_graph_flowsteer.webshop import (
    WebShopClickTool,
    WebShopSearchTool,
    WebShopSessionLifecycle,
)

from .test_application import write_config

POLICY = "skillflow_history_v1"
FINAL = json.dumps(
    {"answer": "Inspected public evidence", "unresolved_issues": ["No purchase yet"]}
)
ASINS = [f"B{i:09d}" for i in range(1, 9)]
LONG_PAGE = "UNIQUE_FIRST_PRODUCT " + "detail " * 1700 + " TAIL_WATERPROOF_NO"


class HistoryClient:
    def __init__(self):
        self.version = 0
        self.asin = ""
        self.calls = []
        self.created = 0
        self.fail_search = False

    def page(self, kind):
        actions = [{"kind": "navigate", "target_id": "back_to_search:1", "label": "back to search"}]
        if kind in {"search", "search_results"}:
            actions += [
                {"kind": "open_product", "target_id": f"open_product:{i}:{asin}", "label": asin}
                for i, asin in enumerate(ASINS)
            ]
        state = {
            "page_type": kind,
            "page_text": f"PAGE {kind} {self.version}",
            "state_version": self.version,
            "valid_subactions": actions,
        }
        if kind == "product":
            state["product"] = {"asin": self.asin, "title": self.asin, "price": 10}
            state["page_text"] = LONG_PAGE if self.asin == ASINS[0] else f"PRODUCT {self.asin}"
        return state

    def create_session(self, goal_id, *, seed):
        self.created += 1
        return {"session_id": f"session-{self.created}", **self.page("search")}

    def search(self, session_id, query):
        if self.fail_search:
            raise RuntimeError("public transport failure")
        self.calls.append(("search", query))
        self.version += 1
        return self.page("search_results")

    def click(self, session_id, target_id):
        self.calls.append(("click", target_id))
        self.version += 1
        if target_id.startswith("open_product:"):
            self.asin = target_id.split(":")[-1]
            return self.page("product")
        return self.page("search_results")

    def close_session(self, session_id):
        pass


def action(name, **arguments):
    return json.dumps({"action_call": {"name": name, "arguments": arguments}})


def build(responses, *, policy=POLICY, routed=False, initial=16, total=20):
    client = HistoryClient()
    lifecycle = WebShopSessionLifecycle(
        client, search_observation_mode="legacy", max_observation_chars=0
    )
    lifecycle.bind_task(TaskSpec("first", "Buy a product", metadata={"goal_id": "goal-1"}))
    tools = {
        "webshop_search": WebShopSearchTool(lifecycle),
        "webshop_click": WebShopClickTool(lifecycle),
    }
    backend = MockBackend(list(responses))
    kwargs = {
        "tools": tools,
        "webshop_worker_memory_policy": policy,
        "action_registry": default_dataset_action_registry(available_actions=tuple(tools)),
    }
    executor = (
        RoutedModelAgentExecutor({"deepseek": backend}, ("deepseek",), **kwargs)
        if routed
        else ModelAgentExecutor(backend, **kwargs)
    )
    node = AgentNode(
        "agent_1",
        "Inspect products",
        allowed_tools=tuple(tools),
        operation_policy_configured=True,
        initial_tool_budget=initial,
        revision_tool_budget=total - initial,
        total_tool_budget=total,
        metadata={"action_adapter": "webshop", "runtime_route": "deepseek"},
    )
    return executor, backend, node, lifecycle, client


def execute(executor, node, *, revision=False):
    return executor.execute(
        task="Buy a product", node=node, upstream=[], peers=[], revision=revision, seed=0
    )


def environment(call):
    return json.loads(call["messages"][1]["content"])["action_environment"]


@pytest.mark.parametrize("routed", [False, True])
def test_full_history_survives_seven_products_and_long_pages(routed):
    responses = [action("webshop_search", query="product")]
    for i, asin in enumerate(ASINS[:7]):
        responses.append(
            action(
                "webshop_click", target_id=f"open_product:{i}:{asin}", state_version=len(responses)
            )
        )
        if i < 6:
            responses.append(
                action("webshop_click", target_id="back_to_search:1", state_version=len(responses))
            )
    responses.append(FINAL)
    executor, backend, node, lifecycle, client = build(responses, routed=routed)
    artifact = execute(executor, node)
    env = environment(backend.calls[-1])
    history = env["react_history"]
    assert len(history) == len(client.calls) == 14
    assert [r["step"] for r in history] == list(range(1, 15))
    assert history[0]["observation"] == "PAGE search 0"
    # On the third request the first product is current, with no 8000-char cut.
    assert environment(backend.calls[2])["state"]["page_text"] == LONG_PAGE
    assert history[2]["observation"] == LONG_PAGE
    assert history[2]["action"]["arguments"]["target_id"] == "back_to_search:1"
    assert all(r["status"] == "ok" for r in history)
    assert len(artifact.webshop_progress["product_inspections"]) == 6
    assert ASINS[0].lower() not in {
        r["asin"] for r in artifact.webshop_progress["product_inspections"]
    }
    assert "product_inspections" not in env["webshop_progress"]
    assert "public_constraint_matrix" not in env
    assert "action_decision_support" not in env["state"]
    assert set(env["webshop_progress"]) == {"policy_contract", "purchase_evidence_checkpoint"}
    assert lifecycle.runtime_transaction_journal_for(node.agent_id)["react_history"] == history
    assert artifact.webshop_progress["worker_memory"]["history_entries"] == 14
    assert artifact.webshop_progress["state_guidance_deliveries"] == []


def test_history_policy_preserves_tool_calls_but_removes_derived_memory_labels():
    responses = [
        action("webshop_search", query="product"),
        action("webshop_click", target_id=f"open_product:0:{ASINS[0]}", state_version=1),
        action("webshop_click", target_id="back_to_search:1", state_version=2),
        FINAL,
    ]
    off, off_backend, off_node, _, off_client = build(responses, policy="factual_memory_v1")
    on, on_backend, on_node, _, on_client = build(responses)
    old, new = execute(off, off_node), execute(on, on_node)
    assert off_client.calls == on_client.calls
    assert old.webshop_progress["action_budget"] == new.webshop_progress["action_budget"]
    old_state, new_state = (
        environment(off_backend.calls[-1])["state"],
        environment(on_backend.calls[-1])["state"],
    )
    assert [a["target_id"] for a in old_state["valid_subactions"]] == [
        a["target_id"] for a in new_state["valid_subactions"]
    ]
    assert any("inspection_status" in a for a in old_state["valid_subactions"])
    assert not any("inspection_status" in a for a in new_state["valid_subactions"])
    assert "react_history" not in environment(off_backend.calls[-1])
    assert "worker_memory" not in old.webshop_progress
    assert "webshop_progress.state_guidance" not in on_backend.calls[0]["messages"][0]["content"]
    assert "historical option clicks" in on_backend.calls[0]["messages"][0]["content"]


def test_revision_inherits_history_and_reviewer_and_new_task_are_isolated():
    executor, backend, owner, lifecycle, client = build(
        [action("webshop_search", query="first query"), FINAL], routed=True
    )
    execute(executor, owner)
    prior = copy.deepcopy(
        lifecycle.runtime_transaction_journal_for(owner.agent_id)["react_history"]
    )
    backend.responses.extend(
        [action("webshop_click", target_id=f"open_product:0:{ASINS[0]}", state_version=1), FINAL]
    )
    execute(executor, owner, revision=True)
    assert environment(backend.calls[2])["react_history"] == prior
    assert len(environment(backend.calls[3])["react_history"]) == 2
    assert client.created == 1
    reviewer = AgentNode(
        "reviewer",
        "Review evidence",
        allowed_tools=owner.allowed_tools,
        metadata=owner.metadata.copy(),
    )
    backend.responses.append(FINAL)
    reviewed = execute(executor, reviewer)
    assert "react_history" not in environment(backend.calls[-1])
    assert reviewed.webshop_progress["worker_memory"]["applied"] is False
    assert lifecycle.runtime_transaction_journal_for("reviewer") is None
    lifecycle.bind_task(TaskSpec("second", "Buy another product", metadata={"goal_id": "goal-2"}))
    executor.reset()
    backend.responses.append(FINAL)
    execute(executor, owner)
    assert environment(backend.calls[-1])["react_history"] == []
    assert client.created == 2


def test_zero_action_report_restores_history_without_new_session():
    executor, backend, owner, lifecycle, client = build(
        [
            action("webshop_search", query="retained query"),
            action("webshop_click", target_id=f"open_product:0:{ASINS[0]}", state_version=1),
            FINAL,
        ],
        initial=2,
        total=2,
    )
    execute(executor, owner)
    history = copy.deepcopy(
        lifecycle.runtime_transaction_journal_for(owner.agent_id)["react_history"]
    )
    backend.responses.append(FINAL)
    artifact = execute(executor, owner, revision=True)
    content = json.dumps(backend.calls[-1]["messages"])
    assert "retained query" in content and "react_history" in content
    context = json.loads(backend.calls[-1]["messages"][1]["content"])
    assert context["webshop_current_public_state"]["page_text"] == LONG_PAGE
    assert context["webshop_current_public_state"]["page_text_omitted_chars"] == 0
    assert artifact.webshop_progress["worker_memory"]["history_entries"] == 2
    assert lifecycle.runtime_transaction_journal_for(owner.agent_id)["react_history"] == history
    assert client.created == 1


def test_failed_attempt_is_preserved_as_error():
    executor, backend, node, lifecycle, client = build(
        [action("webshop_search", query="failed query"), FINAL]
    )
    client.fail_search = True
    artifact = execute(executor, node)
    history = lifecycle.runtime_transaction_journal_for(node.agent_id)["react_history"]
    assert len(history) == 1 and history[0]["status"] == "error"
    assert "public transport failure" in json.dumps(history[0]["error"])
    assert artifact.webshop_progress["worker_memory"]["history_entries"] == 1
    assert client.calls == []


def test_deferred_batch_action_is_not_added_as_an_executed_step():
    batch = json.dumps(
        {
            "action_calls": [
                {"name": "webshop_search", "arguments": {"query": "first query"}},
                {"name": "webshop_search", "arguments": {"query": "discarded query"}},
            ]
        }
    )
    executor, backend, node, lifecycle, client = build([batch, FINAL])
    artifact = execute(executor, node)
    history = environment(backend.calls[-1])["react_history"]
    assert client.calls == [("search", "first query")]
    assert len(history) == 1 and history[0]["action"]["arguments"]["query"] == "first query"
    assert artifact.react_trace[-1]["observation"]["error"]["code"] == "stateful_action_deferred"


def test_other_datasets_are_unchanged():
    backend = MockBackend([FINAL])
    executor = ModelAgentExecutor(backend, webshop_worker_memory_policy=POLICY)
    execute(executor, AgentNode("math", "Compute"))
    assert "react_history" not in json.dumps(backend.calls)


def test_configuration_wiring_and_candidate_has_only_memory_policy_delta(tmp_path):
    path = write_config(tmp_path)
    assert load_adaptive_config(path).webshop.worker_memory_policy == "factual_memory_v1"
    with path.open("a") as f:
        f.write(
            f'\n[webshop]\nenabled = true\nworker_memory_policy = "{POLICY}"\nsearch_observation_mode = "legacy"\nmax_observation_chars = 0\n'
        )
    app = create_adaptive_application(load_adaptive_config(path), mock=True)
    assert app.runtime.executor.webshop_worker_memory_policy == POLICY
    root = Path(__file__).resolve().parents[1]
    old = tomllib.loads((root / "configs/webshop_laser_checklist_eval.toml").read_text())
    new = tomllib.loads((root / "configs/webshop_skillflow_history_eval.toml").read_text())
    assert new["webshop"].pop("worker_memory_policy") == POLICY
    assert old == new


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="worker_memory_policy"):
        WebShopConfig(worker_memory_policy="skillfow").validate()


@pytest.mark.parametrize("mode,limit", [("structured_only", 0), ("legacy", 4000)])
def test_history_rejects_configuration_that_discards_page_text(mode, limit):
    with pytest.raises(ValueError, match="requires legacy or retain_page_text"):
        WebShopConfig(
            enabled=True,
            worker_memory_policy=POLICY,
            search_observation_mode=mode,
            max_observation_chars=limit,
        ).validate()
