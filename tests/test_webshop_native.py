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
from selfplay_graph_flowsteer.webshop_native_protocol import (
    NATIVE_POLICY,
    native_observation,
    native_prompt,
    parse_native_action,
)

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


def build(responses, *, routed=False, conversation=False):
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
        tools=tools,
        action_registry=registry,
        webshop_worker_execution_policy=NATIVE_POLICY,
        webshop_native_conversation_history=conversation,
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


def test_analysis_report_does_not_claim_shopping_completion():
    e, _, life, client, _ = build([report("Requirements checked")])
    artifact = execute(e)
    assert artifact.answer == "Requirements checked"
    assert artifact.webshop_progress["state"] == "no_purchase_candidate"
    assert artifact.webshop_progress["stop_reason"] == "agent_report"
    assert artifact.webshop_progress["action_budget"]["total_used"] == 0
    assert not life.commit_ready_agents() and not client.calls


def test_output_materialization_resumes_selected_session_with_shared_revision_budget():
    c, backend, _, client = canvas_build(
        [
            "search[product]",
            report("Candidate list"),
            f"click[{ASIN1}]",
            "click[buy now]",
            report(),
        ]
    )
    c.config = replace(c.config, native_webshop_output_materialization=True)
    add(c, "a")
    assert c.step('{"action":"set_output","target":"a"}').accepted
    assert len(backend.calls) == 2 and not client.commits
    finished = c.step('{"action":"finish"}')
    assert finished.accepted and len(client.sessions) == len(client.commits) == 1
    assert finished.execution.executed_agents == ["a"]
    assert "selected_output_recovery_required" in finished.execution.invalidation_reasons["a"]
    budget = c.runtime.artifacts["a"].webshop_progress["action_budget"]
    assert (budget["initial_used"], budget["revision_used"], budget["total_used"]) == (1, 2, 3)
    assert '"execution_phase": "selected_output"' in backend.calls[2]["messages"][0]["content"]
    assert "PRIVATE_s1 search_results" in backend.calls[2]["messages"][1]["content"]
    count = len(backend.calls)
    assert not c.step('{"action":"finish"}').accepted
    assert len(backend.calls) == count and len(client.commits) == 1


@pytest.mark.parametrize("staged", [False, True])
def test_output_materialization_is_bounded_and_skips_existing_candidate(staged):
    responses = (
        ["search[q]", f"click[{ASIN1}]", "click[buy now]", report()]
        if staged
        else [report("Analysis"), *(["invalid"] * 4), report("No purchase")]
    )
    c, backend, _, client = canvas_build(responses)
    c.config = replace(c.config, native_webshop_output_materialization=True)
    add(c, "a")
    assert c.step('{"action":"set_output","target":"a"}').accepted
    assert c.step('{"action":"finish"}').accepted
    assert len(backend.calls) == (4 if staged else 6)
    assert len(client.commits) == int(staged)
    budget = c.runtime.artifacts["a"].webshop_progress["action_budget"]
    assert budget["revision_used"] == (0 if staged else 4)


def test_output_materialization_does_not_reallocate_exhausted_revision_allowance():
    c, backend, _, client = canvas_build([report("Analysis")])
    c.config = replace(c.config, native_webshop_output_materialization=True)
    add(c, "a")
    executor = c.runtime.executor
    for _ in range(4):
        assert executor.budget_ledger.consume(
            c.graph.nodes["a"], revision=True, scope=executor.budget_scope
        )[0]
    assert c.step('{"action":"set_output","target":"a"}').accepted
    assert c.step('{"action":"finish"}').accepted
    assert len(backend.calls) == 1 and not client.commits


def test_selected_ui_state_is_visible_without_catalog_defaults_or_reward():
    state = {
        "page_type": "product",
        "page_text": "Item size small large Buy Now",
        "raw_available_actions": ["click[small]", "click[large]", "click[buy now]"],
        "selected_options": {},
        "reward": 0.987654,
        "product": {"default_option": "HIDDEN_DEFAULT"},
        "goal_options": {"size": "HIDDEN_GOAL"},
    }
    before = native_observation(state)
    assert "Selected options: {}" in before
    state["selected_options"] = {"size": "large"}
    after = native_observation(state)
    assert before != after and '"size": "large"' in after
    history = [{"observation": before, "action": "click[large]"}]
    prompt = native_prompt(task="Buy an item", state=state, history=history)
    assert before in prompt and after in prompt
    assert all(s not in prompt for s in ["HIDDEN_DEFAULT", "HIDDEN_GOAL", "0.987654"])
    assert state["page_text"] == "Item size small large Buy Now"
    # An old product's selection must not appear on a search-results page.
    state["page_type"] = "search_results"
    assert native_observation(state) == state["page_text"]


def test_director_completion_status_reads_live_shared_budget_and_candidate():
    c, _, life, client = canvas_build(
        [
            "search[q]",
            f"click[{ASIN1}]",
            "click[buy now]",
            report(),
            report("Analysis only"),
            report("Candidate withdrawn"),
        ]
    )
    add(c, "a")
    add(c, "b")
    assert c.step('{"action":"set_output","target":"b"}').accepted
    status = c.control_snapshot()["environment_task_status"]
    assert status["candidate_agents"] == ["a"]
    assert "PRIVATE_" not in json.dumps(status)
    assert "reward" not in json.dumps(status)
    assert status["completion_state"] == "no_candidate"
    assert not status["purchase_committed"]
    assert status["remaining_actions_by_agent"]["a"] == status["remaining_actions_by_agent"]["b"]
    assert status["remaining_actions_by_agent"]["b"] == {
        "initial_actions_remaining": 9,
        "revision_actions_remaining": 4,
        "total_actions_remaining": 13,
    }
    assert c.step('{"action":"set_output","target":"a"}').accepted
    assert c.control_snapshot()["environment_task_status"]["selected_candidate_staged"]
    # A real re-execution cancels the candidate, even though the old Canvas
    # artifact still describes it. The snapshot must consult the live lifecycle.
    execute(c.runtime.executor, c.graph.nodes["a"], revision=True)
    assert not life.commit_ready_agents()
    assert c.runtime.artifacts["a"].webshop_progress["commit_ready"]
    assert c.control_snapshot()["environment_task_status"]["completion_state"] == "no_candidate"
    assert not client.commits


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
    status = c.control_snapshot()["environment_task_status"]
    assert status["completion_state"] == "purchased" and status["purchase_committed"]
    assert len(b.calls) == calls_before
    assert len(client.commits) == 1 and client.commits[0][1] == f"purchase:{ASIN2}"


def test_native_config_and_legacy_default_are_separate(monkeypatch, tmp_path):
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0,1,2,3,4,5,6,7")
    # This configuration unit test makes no requests and must not require
    # the original deployment's private credential files.
    monkeypatch.setattr("selfplay_graph_flowsteer.application._api_key", lambda _: "test-key")
    cfg = load_adaptive_config("configs/webshop_skillflow_native_eval.toml", validate=False)
    cfg.webshop.validate()  # Unrelated ALFWorld/SWE deployments are not test fixtures.
    assert cfg.webshop.worker_execution_policy == NATIVE_POLICY
    assert cfg.webshop.worker_memory_policy == "factual_memory_v1"
    assert WebShopConfig().worker_execution_policy == "graph_tools_v1"
    assert not cfg.webshop.native_conversation_history
    from pathlib import Path

    variant = tmp_path / "config.toml"
    variant.write_text(
        Path("configs/webshop_skillflow_native_eval.toml")
        .read_text()
        .replace("[webshop]", "[webshop]\nnative_conversation_history = true")
    )
    assert load_adaptive_config(variant, validate=False).webshop.native_conversation_history
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


@pytest.mark.parametrize("routed", [False, True])
def test_native_conversation_restores_private_action_turns_across_revision(routed):
    e, b, life, client, _ = build(
        [
            "search[product]",
            report("inspection"),
            f"click[{ASIN1}]",
            "click[buy now]",
            report("proposal"),
            report("other agent"),
        ],
        routed=routed,
        conversation=True,
    )
    execute(e)
    artifact = execute(e, revision=True)
    messages = b.calls[3]["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == "search[product]"
    assert messages[4]["content"] == f"click[{ASIN1}]"
    assert "search_results" in messages[3]["content"]
    assert "product" in messages[5]["content"]
    assert "Observation 1" not in messages[-1]["content"]
    assert artifact.webshop_progress["action_budget"]["total_used"] == 3
    assert len(client.sessions) == 1 and not client.commits
    reporting = json.loads(b.calls[4]["messages"][1]["content"])
    assert all("turn_prompt" not in row for row in reporting["private_history"])
    execute(e, node("b"))
    assert "PRIVATE_s1" not in json.dumps(b.calls[-1]["messages"])
    assert len(b.calls[-1]["messages"]) == 2


def test_native_conversation_keeps_invalid_attempt_and_error_when_resuming():
    e, b, life, client, _ = build(
        ["click[missing]", report(), "search[recovery]", report()], conversation=True
    )
    execute(e)
    artifact = execute(e, revision=True)
    messages = b.calls[2]["messages"]
    assert messages[2]["content"] == "click[missing]"
    assert "Previous action failed:" in messages[-1]["content"]
    assert artifact.webshop_progress["action_budget"]["total_used"] == 2
    assert len(client.calls) == 1


def test_native_conversation_mode_is_part_of_execution_cache_identity():
    e, _, _, _, _ = build([])
    runtime = MultiAgentRuntime(e)
    args = dict(task="Buy a product", node=node(), upstream=[], peers=[], revision=False)
    before = runtime._cache_payload(**args)
    e.webshop_native_conversation_history = True
    after = runtime._cache_payload(**args)
    assert runtime._cache_key(before) != runtime._cache_key(after)
    assert "environment_changed" in runtime._input_change_reasons(before, after, revision=False)
