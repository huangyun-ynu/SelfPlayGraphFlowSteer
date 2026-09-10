from __future__ import annotations

import pytest

from selfplay_graph_flowsteer.features import graph_kernel, structural_features
from selfplay_graph_flowsteer.graph import GraphValidationError, MultiAgentGraph


def configured_graph() -> MultiAgentGraph:
    graph = MultiAgentGraph()
    graph.add_agent("a")
    graph.set_prompt("a", "independent analysis")
    graph.add_agent("b")
    graph.set_prompt("b", "peer analysis")
    graph.add_agent("out")
    graph.set_prompt("out", "synthesize")
    graph.set_layer("out", 1)
    return graph


def test_relation_layer_constraints_and_final_reachability() -> None:
    graph = configured_graph()
    with pytest.raises(GraphValidationError, match=r"layer\(source\) < layer\(target\)"):
        graph.set_relation("a", "b", "directed")
    graph.set_relation("a", "b", "bidirectional")
    graph.set_relation("a", "out", "directed")
    graph.set_relation("b", "out", "directed")
    graph.set_output("out")
    graph.assert_valid(final=True)


def test_dirty_closure_includes_peer_component_and_downstream() -> None:
    graph = configured_graph()
    graph.set_relation("a", "b", "bidirectional")
    graph.set_relation("a", "out", "directed")
    graph.set_relation("b", "out", "directed")
    assert graph.dirty_closure({"a"}) == {"a", "b", "out"}


def test_same_effective_prompt_is_non_mutating_noop() -> None:
    graph = MultiAgentGraph()
    graph.add_agent("a")
    first = graph.set_prompt(
        "a",
        "independent analysis",
        runtime_route="worker-a",
        metadata_updates={"delegation": {"scope": "task"}},
    )
    version = graph.version

    repeated = graph.set_prompt(
        "a",
        "  independent analysis  ",
        runtime_route="worker-a",
        metadata_updates={"delegation": {"scope": "task"}},
    )

    assert first.changed
    assert not repeated.changed
    assert repeated.change_reason == "no_effective_change"
    assert repeated.dirty_agents == set()
    assert graph.version == version


def test_same_layer_is_non_mutating_noop() -> None:
    graph = configured_graph()
    version = graph.version

    repeated = graph.set_layer("out", 1)

    assert not repeated.changed
    assert repeated.change_reason == "no_effective_change"
    assert repeated.dirty_agents == set()
    assert graph.version == version


def test_same_action_environment_and_capability_are_non_mutating_noops() -> None:
    graph = MultiAgentGraph()
    graph.add_agent("a")
    graph.set_prompt("a", "independent analysis")
    graph.configure_action_environment(
        "a",
        adapter_id="test",
        action_names=("inspect",),
        initial_action_budget=1,
        revision_action_budget=1,
        total_action_budget=2,
        capability_policy={"read": True},
    )
    graph.assign_exclusive_capability("a", "commit")
    version = graph.version

    environment = graph.configure_action_environment(
        "a",
        adapter_id="test",
        action_names=("inspect",),
        initial_action_budget=1,
        revision_action_budget=1,
        total_action_budget=2,
        capability_policy={"read": True},
    )
    capability = graph.assign_exclusive_capability("a", "commit")

    assert not environment.changed
    assert not capability.changed
    assert environment.dirty_agents == capability.dirty_agents == set()
    assert graph.version == version


def test_structure_features_are_agent_id_invariant() -> None:
    left = configured_graph()
    left.set_relation("a", "b", "bidirectional")
    left.set_relation("a", "out", "directed")
    left.set_relation("b", "out", "directed")

    right = MultiAgentGraph()
    for key in ("x", "y", "z"):
        right.add_agent(key)
        right.set_prompt(key, key)
    right.set_layer("z", 1)
    right.set_relation("x", "y", "bidirectional")
    right.set_relation("x", "z", "directed")
    right.set_relation("y", "z", "directed")

    assert structural_features(left).values == structural_features(right).values
    assert graph_kernel(structural_features(left), structural_features(right)) == pytest.approx(1.0)


def test_flowsteer_structural_operators_round_trip() -> None:
    graph = MultiAgentGraph()
    graph.add_agent("solver")
    graph.set_prompt("solver", "solve", structural_operator="solver")
    graph.add_agent("checker")
    graph.set_prompt("checker", "check", structural_operator="checker")
    graph.add_agent("formatter")
    graph.set_prompt("formatter", "format", structural_operator="formatter")
    graph.set_layer("formatter", 1)
    graph.set_relation("solver", "formatter", "directed")
    graph.set_relation("checker", "formatter", "directed")
    graph.set_output("formatter")

    evaluation = graph.evaluate_flowsteer_structure(enabled=True)
    restored = MultiAgentGraph.from_dict(graph.to_dict())

    assert evaluation.complete
    assert restored.to_dict()["nodes"] == graph.to_dict()["nodes"]
