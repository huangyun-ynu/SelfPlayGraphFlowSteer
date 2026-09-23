from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.application import WebShopConfig, load_adaptive_config
from selfplay_graph_flowsteer.canvas import CanvasState, GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import (
    ModelAgentExecutor,
    MultiAgentRuntime,
    RoutedModelAgentExecutor,
)
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
from selfplay_graph_flowsteer.webshop_native_protocol import NATIVE_POLICY, parse_native_action

ASIN1, ASIN2 = "B000000001", "B000000002"


def report(answer="Public evidence report", **kwargs):
    return json.dumps(
        {
            "answer": answer,
            "summary": answer,
            "confidence": 0.7,
            "evidence": [answer],
            "unresolved_issues": [],
            **kwargs,
        }
    )


class NativeClient:
    """Independent live pages; scoring happens only at an explicit commit."""

    def __init__(self):
        self.sessions = {}
        self.calls = []
        self.commits = []
        self.closed = []

    def health(self):
        return {"status": "ok", "goal_fingerprint": "fixed"}

    def page(self, sid, kind, asin=""):
        previous = self.sessions.get(sid, {})
        version = previous.get("state_version", -1) + 1
        actions = []
        if kind == "search_results":
            actions += [
                {
                    "kind": "open_product",
                    "label": "Public product title",
                    "target_id": f"open_product:{i}:{a}",
                    "raw_action": f"click[{a}]",
                }
                for i, a in enumerate((ASIN1, ASIN2))
            ]
        if kind == "product":
            actions += [
                {
                    "kind": "purchase",
                    "label": "Buy Now",
                    "target_id": f"purchase:{asin}",
                    "raw_action": "click[buy now]",
                }
            ]
        if kind != "search":
            actions += [
                {
                    "kind": "navigate",
                    "label": "Back to Search",
                    "target_id": f"back:{version}",
                    "raw_action": "click[back to search]",
                }
            ]
        raw = [a["raw_action"] for a in actions] if kind != "search" else ["search"]
        state = {
            "page_type": kind,
            "page_text": f"PRIVATE_{sid} {kind} {asin} " + "DETAIL " * 1500,
            "state_version": version,
            "valid_subactions": actions,
            "raw_available_actions": raw,
            "purchased": False,
            "done": False,
            "reward": 0.0,
        }
        if asin:
            state["product"] = {"asin": asin}
        self.sessions[sid] = state
        return copy.deepcopy(state)

    def create_session(self, goal_id, *, seed):
        sid = f"s{len(self.sessions) + 1}"
        return {"session_id": sid, **self.page(sid, "search")}

    def search(self, sid, query):
        self.calls.append((sid, "search", query))
        return self.page(sid, "search_results")

    def click(self, sid, target):
        self.calls.append((sid, "click", target))
        if target.startswith("purchase:"):
            raise AssertionError("purchases must use commit, never an ordinary click")
        if target.startswith("open_product:"):
            return self.page(sid, "product", target.split(":")[-1])
        return self.page(sid, "search")

    def commit(self, sid, target, *, commit_id):
        self.commits.append((sid, target, commit_id))
        return {**copy.deepcopy(self.sessions[sid]), "purchased": True, "done": True, "reward": 1.0}

    def close_session(self, sid):
        self.closed.append(sid)


def build(responses, *, routed=False):
    client = NativeClient()
    lifecycle = NativeWebShopLifecycle(
        client, max_observation_chars=0, search_observation_mode="legacy"
    )
    lifecycle.bind_task(TaskSpec("task", "Buy a product", metadata={"goal_id": "goal-1"}))
    tools = {
        "webshop_search": WebShopSearchTool(lifecycle),
        "webshop_click": WebShopClickTool(lifecycle),
    }
    registry = default_dataset_action_registry(tools, webshop_commit_on_finish=True)
    backend = MockBackend(list(responses))
    kwargs = dict(
        tools=tools, action_registry=registry, webshop_worker_execution_policy=NATIVE_POLICY
    )
    executor = (
        RoutedModelAgentExecutor({"deepseek": backend}, ("deepseek",), **kwargs)
        if routed
        else ModelAgentExecutor(backend, **kwargs)
    )
    return executor, backend, lifecycle, client, registry


def node(name="a"):
    return AgentNode(
        name,
        "Inspect the public product evidence",
        allowed_tools=("webshop_search", "webshop_click"),
        operation_policy_configured=True,
        initial_tool_budget=12,
        revision_tool_budget=4,
        total_tool_budget=16,
        metadata={"action_adapter": "webshop", "runtime_route": "deepseek"},
    )


def execute(executor, n=None, *, revision=False):
    return executor.execute(
        task="Buy a product", node=n or node(), upstream=[], peers=[], revision=revision, seed=0
    )


@pytest.mark.parametrize("routed", [False, True])
def test_raw_actions_full_private_history_and_staged_purchase(routed):
    e, b, life, client, _ = build(
        [
            "search[product]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report(),
        ],
        routed=routed,
    )
    artifact = execute(e)
    assert len(client.calls) == 2 and not client.commits
    assert artifact.answer == "purchase_staged"
    assert life.commit_ready_agents() == ("a",)
    assert artifact.webshop_progress["action_budget"]["total_used"] == 3
    action_prompt = b.calls[2]["messages"][1]["content"]
    assert "Observation 1" in action_prompt and "DETAIL " * 1500 in action_prompt
    assert "target_id" not in action_prompt and "state_version" not in action_prompt
    assert "purchase_evidence" not in action_prompt and "LASER" not in action_prompt
    assert "'click[buy now]'" in action_prompt
    assert not b.calls[0].get("actions")
    life.commit_pending("a")
    life.commit_pending("a")
    assert len(client.commits) == 1


def test_each_generic_agent_has_isolated_history_and_no_first_owner_privilege():
    e, b, life, client, _ = build(
        [
            "search[only A]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report("A"),
            report("B independent"),
        ]
    )
    execute(e, node("a"))
    execute(e, node("b"))
    b_prompt = json.dumps(b.calls[-1]["messages"])
    assert "PRIVATE_s1" not in b_prompt and "only A" not in b_prompt
    assert "PRIVATE_s2" in b_prompt
    assert life.allows_agent("a") and life.allows_agent("b")
    assert life.runtime_transaction_journal_for("b")["native_history"] == []
    assert len(client.sessions) == 2


def test_scheduled_revision_cancels_old_candidate_and_resumes_same_page():
    e, b, life, client, _ = build(
        [
            "search[first]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report("A"),
            "click[back to search]",
            "search[alternative]",
            f"click[{ASIN2}]",
            "click[buy now]",
            report("revised"),
        ]
    )
    execute(e)
    revised = execute(e, revision=True)
    assert len(client.sessions) == 1
    assert "PRIVATE_s1 product" in b.calls[4]["messages"][1]["content"]
    assert revised.webshop_progress["worker_memory"]["restored_entries"] == 3
    assert revised.webshop_progress["action_budget"]["revision_used"] == 4
    life.commit_pending("a")
    assert client.commits[0][1] == f"purchase:{ASIN2}"


def test_revision_without_new_purchase_does_not_commit_stale_proposal():
    e, _, life, client, _ = build(
        [
            "search[first]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report(),
            report("Evidence insufficient"),
        ]
    )
    execute(e)
    execute(e, revision=True)
    assert life.commit_ready_agents() == () and not client.commits


def test_invalid_actions_count_and_shared_budget_cannot_reset_with_new_agent():
    e, _, life, client, _ = build(
        ["click[invented]"] * 12 + [report(), report("B"), *(["nonsense"] * 4), report()]
    )
    first = execute(e)
    assert first.webshop_progress["action_budget"]["total_used"] == 12
    assert len(first.react_trace) == 12 and all(
        t["observation"]["status"] == "error" for t in first.react_trace
    )
    second = execute(e, node("b"))
    assert not second.react_trace
    last = execute(e, revision=True)
    assert last.webshop_progress["action_budget"]["total_used"] == 16
    assert not client.calls and not client.commits


def test_cache_reuses_current_candidate_without_repeating_environment_actions():
    e, b, life, client, _ = build(["search[q]", f"click[{ASIN1}]", "click[buy now]", report()])
    runtime = MultiAgentRuntime(e)
    graph = MultiAgentGraph()
    graph.nodes["a"] = node()
    first = runtime.execute(task="Buy a product", graph=graph, dirty_agents={"a"})
    second = runtime.execute(task="Buy a product", graph=graph, dirty_agents={"a"})
    assert first.executed_agents == ["a"] and second.reused_agents == ["a"]
    assert len(b.calls) == 4 and len(client.calls) == 2
    assert life.commit_ready_agents() == ("a",)


def canvas_build(responses):
    e, b, life, client, registry = build(responses)
    canvas = GraphCanvas(
        task="Buy a product",
        runtime=MultiAgentRuntime(e),
        config=CanvasConfig(
            max_total_tokens=1000000,
            remaining_token_admission_enabled=False,
            remaining_time_admission_enabled=False,
        ),
        action_adapter=registry.get("webshop"),
        dataset="webshop",
    )
    return canvas, b, life, client


def add(canvas, name):
    assert canvas.step(json.dumps({"action": "add_agent", "agent_id": name})).accepted
    result = canvas.step(
        json.dumps(
            {
                "action": "set_prompt",
                "target": name,
                "role": "Public evidence analysis",
                "objective": f"Assess product evidence {name}",
                "scope": "Use the assigned evidence",
                "expected_output": "Public finding",
            }
        )
    )
    assert result.accepted, result.feedback


def test_canvas_purchase_does_not_latch_graph_or_prune_nodes_and_finish_commits_once():
    c, b, life, client = canvas_build(
        [
            "search[q]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report("A finding"),
            report("B finding"),
            report("B after layer change"),
            "search[q]",
            f"click[{ASIN2}]",
            "click[buy now]",
            report("B with upstream"),
        ]
    )
    add(c, "a")
    assert not client.commits
    add(c, "b")
    selected = c.step('{"action":"set_output","target":"b"}')
    assert selected.accepted and c.state is not CanvasState.FINISHED
    assert set(c.graph.nodes) == {"a", "b"}
    rejected = c.step('{"action":"finish"}')
    assert not rejected.accepted and not client.commits
    assert c.step('{"action":"set_layer","target":"b","layer":1}').accepted
    linked = c.step('{"action":"set_relation","source":"a","target":"b","relation":"directed"}')
    assert linked.accepted and linked.execution.executed_agents == ["b"]
    assert "a" in linked.execution.reused_agents
    assert not client.commits
    calls_before = len(b.calls)
    finished = c.step('{"action":"finish"}')
    assert finished.accepted and c.state is CanvasState.FINISHED
    assert len(b.calls) == calls_before
    assert len(client.commits) == 1 and client.commits[0][1] == f"purchase:{ASIN2}"


def test_native_config_and_legacy_default_are_separate(monkeypatch):
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0,1,2,3,4,5,6,7")
    cfg = load_adaptive_config("configs/webshop_skillflow_native_eval.toml")
    assert cfg.webshop.worker_execution_policy == NATIVE_POLICY
    assert WebShopConfig().worker_execution_policy == "graph_tools_v1"
    with pytest.raises(ValueError, match="requires legacy"):
        replace(cfg.webshop, worker_guidance_policy="laser_checklist_v1").validate()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>click[wrong]</think><action>search[blue shirt]</action>", "search[blue shirt]"),
        ("Action: click[back to search]", "click[back to search]"),
        ("<action>skill_invoke[hidden]</action>", None),
        ("<think>click[wrong]</think>report only", None),
    ],
)
def test_native_parser_executes_only_action_content(text, expected):
    assert parse_native_action(text) == expected


def test_bidirectional_barrier_uses_only_first_pass_peer_packets():
    from selfplay_graph_flowsteer.contracts import RelationType

    e, b, life, _, _ = build(
        [
            report("FIRST_A"),
            report("FIRST_B"),
            report("REVISION_A"),
            report("REVISION_B"),
        ]
    )
    runtime = MultiAgentRuntime(e, bidirectional_revision_policy="always")
    graph = MultiAgentGraph()
    graph.nodes["a"], graph.nodes["b"] = node("a"), node("b")
    graph.set_relation("a", "b", RelationType.BIDIRECTIONAL)
    runtime.execute(task="Buy a product", graph=graph, dirty_agents={"a", "b"})
    initial_a, initial_b, revision_a, revision_b = [json.dumps(c["messages"]) for c in b.calls]
    assert "FIRST_A" not in initial_b and "FIRST_B" not in initial_a
    assert "FIRST_B" in revision_a and "FIRST_A" in revision_b
    assert "REVISION_A" not in revision_b
    assert "PRIVATE_s1" not in initial_b and "PRIVATE_s2" not in initial_a
    # Even connected peers receive a packet rather than the private page history.
    assert "PRIVATE_s2" not in revision_a and "PRIVATE_s1" not in revision_b
    assert len(life.owner_agents()) == 2


def test_dirty_closure_updates_peer_component_and_downstream_only():
    from selfplay_graph_flowsteer.contracts import RelationType

    e, _, _, _, _ = build([report(f"result {i}") for i in range(20)])
    runtime = MultiAgentRuntime(e, bidirectional_revision_policy="always")
    graph = MultiAgentGraph()
    for name in ("a", "b", "c", "d", "unrelated"):
        graph.nodes[name] = node(name)
    graph.set_layer("b", 1)
    graph.set_layer("c", 1)
    graph.set_layer("d", 2)
    graph.set_relation("b", "c", RelationType.BIDIRECTIONAL)
    graph.set_relation("c", "d", RelationType.DIRECTED)
    runtime.execute(task="Buy a product", graph=graph, dirty_agents=set(graph.nodes))
    mutation = graph.set_relation("a", "b", RelationType.DIRECTED)
    result = runtime.execute(task="Buy a product", graph=graph, dirty_agents=mutation.dirty_agents)
    assert set(result.scheduled_agents) == {"b", "c", "d"}
    assert {"a", "unrelated"} <= set(result.skipped_clean_agents)
    assert {"b", "c", "d"} <= set(result.executed_agents)


def test_new_task_resets_private_history_candidates_and_budget():
    e, b, life, client, _ = build(
        [
            "search[q]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report(),
            report("new task"),
        ]
    )
    execute(e)
    life.bind_task(TaskSpec("new", "Another product", metadata={"goal_id": "goal-2"}))
    e.reset()
    artifact = execute(e)
    assert artifact.webshop_progress["action_budget"]["total_used"] == 0
    assert artifact.webshop_progress["worker_memory"]["restored_entries"] == 0
    assert "PRIVATE_s1" not in json.dumps(b.calls[-1]["messages"])
    assert not life.commit_ready_agents() and not client.commits


def test_invalid_report_does_not_invent_a_purchase_or_drop_valid_candidate():
    e, _, life, client, _ = build(
        [
            "search[q]",
            f"click[{ASIN1}]",
            "click[buy now]",
            "malformed",
            "malformed",
        ]
    )
    artifact = execute(e)
    assert artifact.answer == "purchase_staged" and not client.commits
    assert "native_packet_format_failure" in artifact.unresolved_issues
    assert life.commit_ready_agents() == ("a",)


def test_compiler_keeps_director_responsibility_without_legacy_shopping_checklist():
    from selfplay_graph_flowsteer.delegation import compile_delegation

    compiled, issue = compile_delegation(
        {
            "role": "Comparison",
            "objective": "Compare public evidence",
            "scope": "Supplied packets",
            "expected_output": "A comparison of the supplied findings",
        },
        dataset="webshop",
        webshop_native=True,
    )
    assert issue is None
    assert compiled.director_fields["expected_output"] == "A comparison of the supplied findings"
    assert "purchase_evidence" not in compiled.prompt
    assert "target identifiers" not in compiled.prompt
    assert "FINISH" in compiled.prompt
