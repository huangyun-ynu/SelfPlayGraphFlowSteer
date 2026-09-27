"""Director/Canvas adversarial scenarios for the opt-in native WebShop path."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.canvas import CanvasState
from selfplay_graph_flowsteer.contracts import RelationType
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import BinaryChoiceResponse, MockBackend
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime

from .test_director_timeline import render
from .test_director_timeline import tokenizer as tokenizer
from .test_webshop_native import ASIN1, ASIN2, add, build, canvas_build, execute, node, report


def step(c, action, **fields):
    result = c.step(json.dumps(dict(action=action, **fields)))
    assert result.accepted, (action, result.feedback)
    return result


def stage(asin=ASIN1):
    return ["search[product]", f"click[{asin}]", "click[buy now]", report()]


@pytest.mark.parametrize("mode", ["snapshot_dedup", "append_only"])
def test_real_director_driver_creates_two_agents_with_binary_relation_and_keeps_history(
    monkeypatch, mode, tokenizer
):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", mode)
    c, worker, life, client = canvas_build(
        [report("A evidence"), report("B initial"), report("B layer"), *stage()]
    )
    c.binary_relation_policy = True
    commands = [
        dict(action="add_agent", agent_id="a"),
        dict(
            action="set_prompt",
            target="a",
            role="Evidence analyst",
            objective="Assess the public requirements",
            scope="Public task",
            expected_output="Finding A",
        ),
        dict(action="add_agent", agent_id="b"),
        dict(
            action="set_prompt",
            target="b",
            role="Shopping analyst",
            objective="Find a suitable public candidate",
            scope="Task and supplied findings",
            expected_output="Candidate B",
        ),
        dict(action="set_layer", target="b", layer=1),
        dict(action="consider_relation", source="a", target="b"),
        dict(action="set_output", target="b"),
        dict(action="finish"),
    ]
    binary = BinaryChoiceResponse(
        choice="on",
        model="test",
        probabilities={"off": 0.1, "on": 0.9},
        log_probabilities={"off": -2.302585, "on": -0.10536},
        token_ids={
            word: tokenizer.encode(word, add_special_tokens=False)[0] for word in ("off", "on")
        },
    )

    class TokenBackend(MockBackend):
        def generate(self, messages, **kwargs):
            response = super().generate(messages, **kwargs)
            prompt = render(tokenizer, messages)
            completion = tokenizer.encode(
                "simulated reasoning</think>\n\n" + response.text + "<|im_end|>",
                add_special_tokens=False,
            )
            return replace(
                response,
                prompt_token_ids=tuple(prompt),
                completion_token_ids=tuple(completion),
                behavior_log_probs=tuple(-1.0 for _ in completion),
            )

        def choose_binary(self, messages, **kwargs):
            response = super().choose_binary(messages, **kwargs)
            return replace(
                response, prompt_token_ids=tuple(render(tokenizer, messages, thinking=False))
            )

    backend = TokenBackend([json.dumps(x) for x in commands], binary_responses=[binary])
    run = GraphDirector(backend=backend, canvas=c, tokenizer=tokenizer).run()
    assert run.finished, [(t.accepted, t.rejection_code) for t in run.turns]
    assert set(c.graph.nodes) == {"a", "b"} and c.graph.directed_edges == {("a", "b")}
    assert len(client.commits) == 1 and client.commits[0][0] == "s2"
    assert len(worker.calls) == 7
    assert any(call.get("binary_choices") == ["off", "on"] for call in backend.calls)
    assert "isolated" in backend.calls[0]["messages"][0]["content"]
    if mode == "append_only":
        assert "Finding A" in str(backend.calls[-1]["messages"])
        for index, turn in enumerate(run.turns):
            audit = turn.action_diagnostics["timeline_prefix_audit"]
            assert audit["timeline_merge_candidate"]
            if index:
                assert audit["previous_policy_is_exact_prefix"] is True
        assert "simulated reasoning" in str(backend.calls[-1]["messages"])


def test_prompt_edit_revises_same_session_and_does_not_retain_stale_candidate():
    c, b, life, client = canvas_build(
        [
            *stage()[:-1],
            report(unresolved_issues=["Price needs verification"]),
            report("No suitable candidate"),
        ]
    )
    add(c, "a")
    step(
        c,
        "set_prompt",
        target="a",
        role="Public evidence analysis",
        objective="Resolve price uncertainty",
        scope="Inspect available public evidence",
        expected_output="Updated finding",
        revision_basis="unresolved_issue",
        evidence_agent_ids=["a"],
    )
    assert len(client.sessions) == 1 and life.commit_ready_agents() == ()
    assert "PRIVATE_s1 product" in str(b.calls[-1]["messages"])
    step(c, "set_output", target="a")
    step(c, "finish")
    assert not client.commits


def test_selecting_another_output_never_commits_an_unselected_candidate():
    c, b, life, client = canvas_build(
        [*stage(), report("B independent"), report("B layer"), report("B receives finding")]
    )
    c.config = replace(c.config, output_selection_budget=4)
    add(c, "a")
    add(c, "b")
    step(c, "set_layer", target="b", layer=1)
    step(c, "set_relation", source="a", target="b", relation="directed")
    before = len(b.calls)
    step(c, "set_output", target="a")
    step(c, "set_output", target="b")
    assert len(b.calls) == before and life.commit_ready_agents() == ("a",)
    step(c, "finish")
    assert c.state is CanvasState.FINISHED and not client.commits


def test_delete_and_recreate_agent_closes_old_candidate_and_cannot_reuse_its_history_or_cache():
    c, b, life, client = canvas_build([*stage(), report("New identity")])
    add(c, "a")
    old_sig = life.cache_signature("a")
    step(c, "delete_agent", target="a")
    assert client.closed == ["s1"] and life.commit_ready_agents() == ()
    add(c, "a")
    assert life.cache_signature("a") != old_sig
    assert "PRIVATE_s1" not in str(b.calls[-1]["messages"])
    assert "PRIVATE_s2" in str(b.calls[-1]["messages"])
    assert c.runtime.artifacts["a"].webshop_progress["action_budget"]["total_used"] == 3
    assert not client.commits


@pytest.mark.parametrize(
    "command",
    [
        {"action": "set_layer", "target": "a", "layer": -1},
        {"action": "set_relation", "source": "a", "target": "a", "relation": "directed"},
        {"action": "set_output", "target": "missing"},
        {"action": "set_layer", "target": "a", "layer": 1, "expected_version": 0},
        {"action": "add_agent", "agent_id": "a"},
    ],
)
def test_invalid_canvas_edits_are_atomic_and_preserve_staged_candidate(command):
    c, b, life, client = canvas_build(stage())
    add(c, "a")
    graph = copy.deepcopy(c.graph.to_dict())
    sig = life.cache_signature("a")
    calls = len(b.calls)
    result = c.step(json.dumps(command))
    assert not result.accepted
    assert c.graph.to_dict() == graph and life.cache_signature("a") == sig
    assert len(b.calls) == calls and not client.commits


def test_noop_layer_and_duplicate_finish_do_not_repeat_purchase():
    c, b, life, client = canvas_build(stage())
    add(c, "a")
    before = len(b.calls)
    noop = c.step('{"action":"set_layer","target":"a","layer":0}')
    assert not noop.accepted
    assert len(b.calls) == before and life.commit_ready_agents() == ("a",)
    step(c, "set_output", target="a")
    step(c, "finish")
    result = c.step('{"action":"finish"}')
    assert not result.accepted and len(client.commits) == 1
    assert len(b.calls) == before


def test_commit_failure_cannot_switch_to_another_agents_candidate():
    c, b, life, client = canvas_build([*stage(), *stage(ASIN2)])
    add(c, "a")
    add(c, "b")
    step(c, "delete_agent", target="a")

    def fail(*args, **kwargs):
        raise TimeoutError("simulated unavailable commit")

    client.commit = fail
    step(c, "set_output", target="b")
    failed = c.step('{"action":"finish"}')
    assert not failed.accepted and failed.rejection_code == "webshop_environment_commit_failed"
    assert not client.commits and c.state is CanvasState.FAILED


def test_packet_format_failure_still_preserves_trusted_staged_purchase_through_canvas():
    c, b, life, client = canvas_build([*stage()[:-1], "malformed", "malformed"])
    add(c, "a")
    step(c, "set_output", target="a")
    finished = step(c, "finish")
    assert len(client.commits) == 1 and finished.final_execution
    assert c.runtime.artifacts["a"].confidence == 0
    assert c.runtime.artifacts["a"].environment_result["purchased"]


@pytest.mark.parametrize(
    "mutation", ["remove_peer", "remove_forward", "delete_peer", "change_prompt"]
)
def test_dirty_old_peer_component_and_descendants_reexecute_but_unrelated_branch_is_clean(mutation):
    e, b, life, client, _ = build([report(f"finding {i}") for i in range(60)])
    runtime = MultiAgentRuntime(e, bidirectional_revision_policy="always")
    g = MultiAgentGraph()
    for name in ["a", "b", "c", "d", "u"]:
        g.nodes[name] = node(name)
    g.set_layer("b", 1)
    g.set_layer("c", 1)
    g.set_layer("d", 2)
    g.set_relation("a", "b", RelationType.DIRECTED)
    g.set_relation("b", "c", RelationType.BIDIRECTIONAL)
    g.set_relation("c", "d", RelationType.DIRECTED)
    runtime.execute(task="Buy a product", graph=g, dirty_agents=set(g.nodes))
    a_id = runtime.artifacts["a"].artifact_id
    u_id = runtime.artifacts["u"].artifact_id
    if mutation == "remove_peer":
        change = g.remove_relation("b", "c", RelationType.BIDIRECTIONAL)
    elif mutation == "remove_forward":
        change = g.remove_relation("a", "b", RelationType.DIRECTED)
    elif mutation == "delete_peer":
        change = g.delete_agent("b")
        runtime.discard_environment_candidate("b")
    else:
        change = g.set_prompt("b", "Reassess the product evidence")
    result = runtime.execute(task="Buy a product", graph=g, dirty_agents=change.dirty_agents)
    expected = {"c", "d"} if mutation == "delete_peer" else {"b", "c", "d"}
    assert set(result.scheduled_agents) == expected
    assert {"a", "u"} <= set(result.skipped_clean_agents)
    assert runtime.artifacts["a"].artifact_id == a_id and runtime.artifacts["u"].artifact_id == u_id
    assert not client.commits


def test_identical_generic_agent_prompts_do_not_share_mutable_environment_cache():
    e, b, life, client, _ = build([*stage(), report("Independent B")])
    r = MultiAgentRuntime(e)
    g = MultiAgentGraph()
    g.nodes["a"] = node("a")
    g.nodes["b"] = node("b")
    outcome = r.execute(task="Buy a product", graph=g, dirty_agents={"a", "b"})
    assert outcome.executed_agents == ["a", "b"]
    assert "PRIVATE_s1" not in str(b.calls[-1]["messages"])
    assert life.commit_ready_agents() == ("a",) and len(client.sessions) == 2


def test_full_graph_replay_bypasses_cache_without_changing_generic_agent_capabilities():
    e, b, life, client, _ = build([report("one"), report("two")])
    r = MultiAgentRuntime(e)
    g = MultiAgentGraph()
    g.nodes["a"] = node("a")
    r.execute(task="Buy a product", graph=g, dirty_agents={"a"})
    r.full_graph_replay = True
    result = r.execute(task="Buy a product", graph=g, dirty_agents={"a"})
    assert result.executed_agents == ["a"] and not result.reused_agents and len(b.calls) == 2
    assert life.allows_agent("a") and life.allows_agent("b")


def test_backend_failure_after_action_keeps_attempt_and_releases_execution_guard():
    e, b, life, client, _ = build([])
    calls = 0

    def generate(messages, role):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "search[product]"
        raise TimeoutError("simulated backend outage")

    b.handler = generate
    with pytest.raises(TimeoutError):
        execute(e)
    assert life._native_active is None
    b.handler = None
    b.responses.append(report("Resumed"))
    art = execute(e, revision=True)
    assert art.webshop_progress["action_budget"]["total_used"] == 1
    assert art.webshop_progress["worker_memory"]["restored_entries"] == 1
    assert len(client.sessions) == 1


def test_native_flag_does_not_change_other_dataset_output_capabilities_or_canvas_hint():
    from selfplay_graph_flowsteer.actions import ActionType, CanvasAction
    from selfplay_graph_flowsteer.dataset_actions import DatasetActionAdapter

    c, _, _, _ = canvas_build([])
    c.dataset = "swe_bench"
    c.action_adapter = DatasetActionAdapter(
        adapter_id="swe_bench",
        datasets=("swe_bench",),
        action_names=(),
        initial_action_budget=0,
        revision_action_budget=0,
        total_action_budget=0,
        environment_state="stateful",
        session_scope="per_agent",
        action_execution="sequential",
        commit_policy="single_committer",
        commit_activation="export_latest_artifact",
    )
    c.graph.add_agent("a")
    c._apply(c.graph, CanvasAction(ActionType.SET_OUTPUT, target="a"))
    assert c.graph.nodes["a"].metadata.get("exclusive_capabilities") == ["code_commit"]
    assert c.control_snapshot()["environment_commit_resolution"] is None
    assert "environment_task_status" not in c.control_snapshot()


@pytest.mark.parametrize(
    "text",
    [
        "<think>I might use click[buy now]",
        "<think>example search[wrong]\n<action>click[buy now]</action>",
    ],
)
def test_unterminated_thinking_is_never_an_executable_action(text):
    from selfplay_graph_flowsteer.webshop_native_protocol import parse_native_action

    assert parse_native_action(text) is None


def test_cleanup_attempts_every_agent_even_if_one_session_close_fails():
    e, _, life, client, _ = build([report("A"), report("B")])
    execute(e, node("a"))
    execute(e, node("b"))
    original = client.close_session

    def close(sid):
        if sid == "s1":
            client.closed.append(sid)
            raise TimeoutError("simulated cleanup failure")
        original(sid)

    client.close_session = close
    with pytest.raises(TimeoutError):
        life.close_all()
    assert client.closed == ["s1", "s2"]


def test_finish_commit_does_not_rewrite_past_canvas_artifact_or_expose_reward_early():
    c, _, _, client = canvas_build(stage())
    add(c, "a")
    prior = c.history[-1].execution
    before = copy.deepcopy(prior.to_dict())
    step(c, "set_output", target="a")
    finished = step(c, "finish")
    assert len(client.commits) == 1
    assert prior.to_dict() == before
    assert prior.artifacts["a"].environment_result["purchased"] is False
    assert finished.execution.artifacts["a"].environment_result["purchased"] is True


def test_counterfactual_application_replays_complete_graph_with_fresh_sessions_and_commit_snapshot():
    from types import SimpleNamespace

    from selfplay_graph_flowsteer.application import AdaptiveSolverApplication
    from selfplay_graph_flowsteer.config import CanvasConfig
    from selfplay_graph_flowsteer.observability import TaskSpec
    from selfplay_graph_flowsteer.webshop import WebShopEnvironmentVerifier

    e, backend, life, client, registry = build([*stage(), *stage(ASIN2)])
    app = object.__new__(AdaptiveSolverApplication)
    app.runtime = MultiAgentRuntime(e)
    app.config = SimpleNamespace(canvas=CanvasConfig(remaining_token_admission_enabled=False))
    app.solver = SimpleNamespace(action_registry=registry, verifier=WebShopEnvironmentVerifier())
    g = MultiAgentGraph()
    g.nodes["a"] = node("a")
    g.set_output("a")
    task = TaskSpec("task", "Buy a product", metadata={"dataset": "webshop", "goal_id": "goal-1"})
    for _ in range(2):
        assert app.evaluate_graph(task, g, seed=0) == 1.0
        execution = app.last_graph_evaluation["execution"]
        assert execution["executed_agents"] == ["a"] and execution["cache_hits"] == 0
        assert execution["artifacts"]["a"]["environment_result"]["purchased"]
        assert not life.owner_agents() and not app.runtime.full_graph_replay
    assert len(backend.calls) == 8 and len(client.sessions) == 2
    assert [x[1] for x in client.commits] == [f"purchase:{ASIN1}", f"purchase:{ASIN2}"]
