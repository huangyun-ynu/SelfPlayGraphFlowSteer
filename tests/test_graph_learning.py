from __future__ import annotations

import pytest

from selfplay_graph_flowsteer.features import (
    GraphFeatures,
    SemanticGraphFeatureExtractor,
    execution_policy_features_many,
    graph_kernel,
    graph_kernel_matrix,
)
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.graph_learning import (
    build_graph_training_batch,
    canonical_graph_key,
    correct_positive_density,
    correctness_gated_diversity_rewards,
)
from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory


def make_graph(left: str, right: str) -> MultiAgentGraph:
    graph = MultiAgentGraph()
    graph.add_agent(left)
    graph.set_prompt(left, "research")
    graph.add_agent(right)
    graph.set_prompt(right, "answer")
    graph.set_layer(right, 1)
    graph.set_relation(left, right, "directed")
    graph.set_output(right)
    return graph


class _FakeDelegationEmbedder:
    encoder_id = "offline-fake-e5"

    def encode(self, texts):
        return [
            (
                float("research" in text.casefold()),
                float("answer" in text.casefold()),
            )
            for text in texts
        ]


@pytest.mark.parametrize("semantic", [False, True])
def test_feature_pool_union_preserves_original_policy_graphs(semantic):
    graphs = []
    for routes, selected in [
        (("gpt", "gemini"), "gpt"),
        (("gpt",), "gpt"),
        (("gpt", "gemini"), "gemini"),
    ]:
        graph = MultiAgentGraph(runtime_routes=routes)
        graph.add_agent("a")
        graph.set_prompt("a", "answer")
        graph.set_model("a", selected)
        graph.set_output("a")
        graphs.append(graph)
    original = [graph.to_dict() for graph in graphs]
    vectors = (
        SemanticGraphFeatureExtractor(_FakeDelegationEmbedder()).extract_many(graphs)
        if semantic
        else execution_policy_features_many(graphs)
    )
    assert len({v.schema_id for v in vectors}) == 1
    assert graph_kernel(vectors[0], vectors[1]) == pytest.approx(1)
    assert graph_kernel(vectors[0], vectors[2]) < 1
    assert [graph.to_dict() for graph in graphs] == original


def test_empty_policy_graph_has_same_semantic_schema_without_fabricating_agents():
    empty = MultiAgentGraph(runtime_routes=("gpt",))
    graph = MultiAgentGraph(runtime_routes=("gpt",))
    graph.add_agent("a")
    graph.set_prompt("a", "answer")
    graph.set_model("a", "gpt")
    graph.set_output("a")
    extractor = SemanticGraphFeatureExtractor(_FakeDelegationEmbedder())
    vectors = extractor.extract_many([empty, graph])
    assert vectors[0].schema_id == vectors[1].schema_id
    assert not empty.nodes and not any(vectors[0].values)
    assert graph_kernel(vectors[0], vectors[1]) == 0
    standalone = SemanticGraphFeatureExtractor(_FakeDelegationEmbedder()).extract_many([empty])[0]
    assert standalone == vectors[0]


def test_semantic_graph_features_are_layer_ranked_and_relation_aware() -> None:
    extractor = SemanticGraphFeatureExtractor(_FakeDelegationEmbedder())
    first = make_graph("a", "b")
    renamed = make_graph("worker_9", "worker_2")
    shifted = MultiAgentGraph()
    shifted.add_agent("left")
    shifted.set_prompt("left", "research")
    shifted.add_agent("right")
    shifted.set_prompt("right", "answer")
    shifted.set_layer("left", 10)
    shifted.set_layer("right", 20)
    shifted.set_relation("left", "right", "directed")
    shifted.set_output("right")
    reversed_roles = MultiAgentGraph()
    reversed_roles.add_agent("x")
    reversed_roles.set_prompt("x", "answer")
    reversed_roles.add_agent("y")
    reversed_roles.set_prompt("y", "research")
    reversed_roles.set_layer("y", 1)
    reversed_roles.set_relation("x", "y", "directed")
    reversed_roles.set_output("y")

    vectors = extractor.extract_many([first, renamed, shifted, reversed_roles])

    assert vectors[0].schema is not None
    assert vectors[0].schema.output_encoding_mode == "node_block_is_output"
    assert graph_kernel(vectors[0], vectors[1]) == pytest.approx(1.0)
    assert graph_kernel(vectors[0], vectors[2]) == pytest.approx(1.0)
    assert graph_kernel(vectors[0], vectors[3]) < 1.0


def test_semantic_single_agent_has_zero_edge_blocks_and_schema_checks_capacity() -> None:
    extractor = SemanticGraphFeatureExtractor(_FakeDelegationEmbedder())
    single = MultiAgentGraph(max_agents=1)
    single.add_agent("only")
    single.set_prompt("only", "answer")
    single.set_output("only")
    vector = extractor.extract_many([single])[0]
    # structure=15, node=(embedding 2 + output flag)*1, then two 4-value edge blocks.
    assert any(vector.values[:18])
    assert vector.values[18:] == pytest.approx((0.0,) * 8)
    assert all(value == value for value in vector.values)

    larger = make_graph("a", "b")
    other = extractor.extract_many([larger])[0]
    with pytest.raises(ValueError, match="schemas do not match"):
        graph_kernel(vector, other)


def test_explicit_kernel_matrix_is_shared_by_frontier_and_density() -> None:
    features = [GraphFeatures((1.0,), ("x",))] * 3
    matrix = graph_kernel_matrix(features)
    correction = correct_positive_density([2.0, 2.0, -1.0], features, kernel_matrix=matrix)
    assert correction.densities == pytest.approx((2.0, 2.0, 1.0))


def test_canonical_key_ignores_agent_ids() -> None:
    assert canonical_graph_key(make_graph("a", "b")) == canonical_graph_key(
        make_graph("worker_9", "worker_2")
    )


def test_density_correction_downweights_duplicate_positive_graphs() -> None:
    same = GraphFeatures((1.0, 0.0), ("a", "b"))
    distinct = GraphFeatures((0.0, 1.0), ("a", "b"))
    result = correct_positive_density([2.0, 2.0, 2.0], [same, same, distinct])
    assert result.densities == pytest.approx((2.0, 2.0, 1.0))
    assert sum(result.advantages) == pytest.approx(6.0)
    assert result.advantages[2] > result.advantages[0]


def test_batch_builder_groups_rewards_and_stops_before_optimizer() -> None:
    graph = make_graph("a", "b")
    trajectories = [
        TokenizedDirectorTrajectory(
            rollout_id=f"r{index}",
            task_id="task",
            token_ids=(1, 2),
            action_mask=(1, 0),
            reward=reward,
            graph=graph.to_dict(),
        )
        for index, reward in enumerate((0.0, 1.0))
    ]
    batch = build_graph_training_batch(trajectories)
    assert batch.optimizer_steps == 0
    assert [sample.advantage for sample in batch.samples] == pytest.approx([-1.0, 1.0])
    assert {sample.metadata["director_reward_version"] for sample in batch.samples} == {"legacy_v1"}


def test_batch_builder_rejects_mixed_reward_versions() -> None:
    graph = make_graph("a", "b")
    trajectories = [
        TokenizedDirectorTrajectory(
            rollout_id=f"r{index}",
            task_id="task",
            token_ids=(1, 2),
            action_mask=(1, 0),
            reward=float(index),
            graph=graph.to_dict(),
            metadata={"director_reward_version": version},
        )
        for index, version in enumerate(("legacy_v1", "protocol_gate_v1"))
    ]

    with pytest.raises(ValueError, match="refusing to mix Director reward versions"):
        build_graph_training_batch(trajectories)


def test_diversity_bonus_is_small_correctness_gated_and_role_label_free() -> None:
    single = MultiAgentGraph()
    single.add_agent("a")
    single.set_prompt("a", "independent answer")
    single.set_output("a")
    multi = make_graph("source", "output")
    trajectories = [
        TokenizedDirectorTrajectory(
            rollout_id=f"r{index}",
            task_id="task" if index < 3 else "other-task",
            token_ids=(1,),
            action_mask=(1,),
            reward=1.0 if index < 3 else 0.0,
            graph=(single if index < 2 else multi).to_dict(),
            metadata={
                "finished": True,
                "verification": {"passed": index < 3},
                "base_director_reward": 1.0 if index < 3 else 0.0,
            },
        )
        for index in range(4)
    ]

    adjusted = correctness_gated_diversity_rewards(trajectories, max_bonus=0.1)

    assert adjusted[0].reward == pytest.approx(1.0)
    assert adjusted[1].reward == pytest.approx(1.0)
    assert adjusted[2].reward > 1.0
    assert adjusted[2].metadata["graph_novelty"] > 0.0
    assert 0.0 < adjusted[2].metadata["graph_diversity_bonus"] <= 0.1
    assert adjusted[2].metadata["graph_diversity_eligible"] is True
    assert adjusted[3].reward == 0.0
    assert adjusted[3].metadata["graph_diversity_bonus"] == 0.0
    assert adjusted[3].metadata["graph_diversity_eligible"] is False


def test_strict_outcome_only_suppresses_graph_diversity_bonus() -> None:
    first = make_graph("source", "output")
    second = MultiAgentGraph()
    second.add_agent("only")
    second.set_prompt("only", "solve independently")
    second.set_output("only")
    trajectories = [
        TokenizedDirectorTrajectory(
            rollout_id=f"swe-{index}",
            task_id="swe-task",
            token_ids=(1,),
            action_mask=(1,),
            reward=1.0,
            graph=graph.to_dict(),
            metadata={
                "finished": True,
                "protocol_qualified": True,
                "verification": {"passed": True},
                "base_director_reward": 1.0,
                "reward_semantics": "strict_outcome_only",
            },
        )
        for index, graph in enumerate((first, second))
    ]

    adjusted = correctness_gated_diversity_rewards(trajectories, max_bonus=0.1)

    assert [item.reward for item in adjusted] == [1.0, 1.0]
    assert all(item.metadata["graph_diversity_bonus"] == 0.0 for item in adjusted)
    assert all(item.metadata["graph_diversity_suppressed_by_outcome_only"] for item in adjusted)
