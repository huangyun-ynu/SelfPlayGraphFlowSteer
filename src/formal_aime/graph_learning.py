from __future__ import annotations

import itertools
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .features import (
    GraphFeatures,
    execution_policy_features_many,
    graph_kernel,
    graph_kernel_matrix,
    structural_features,
)
from .graph import MultiAgentGraph
from .learning import GraphRolloutScore, graph_level_advantages
from .protocol_reward import LEGACY_REWARD_VERSION
from .rollouts import TokenizedDirectorTrajectory, TrainingBatch, TrainingSample


def canonical_graph_key(graph: MultiAgentGraph, *, include_prompts: bool = False) -> str:
    """Exact permutation-invariant key for the small (<=8 node) Canvas graphs."""

    graph.assert_valid(final=False)
    groups: list[list[str]] = []
    for layer in graph.layers():
        layer_nodes = [key for key, node in graph.nodes.items() if node.layer == layer]
        if include_prompts:
            by_prompt: dict[tuple[object, ...], list[str]] = {}
            for key in layer_nodes:
                node = graph.nodes[key]
                signature = (
                    " ".join(node.prompt.split()),
                    str(node.metadata.get("runtime_route", "")),
                )
                by_prompt.setdefault(signature, []).append(key)
            groups.extend(by_prompt[key] for key in sorted(by_prompt))
        else:
            groups.append(layer_nodes)
    candidates: list[str] = []
    for permutations in itertools.product(*(itertools.permutations(group) for group in groups)):
        order = [node for group in permutations for node in group]
        index = {node: position for position, node in enumerate(order)}
        nodes = []
        for node_id in order:
            node = graph.nodes[node_id]
            item: list[object] = [node.layer, node_id == graph.output_agent]
            if include_prompts:
                item.append(" ".join(node.prompt.split()))
                item.append(str(node.metadata.get("runtime_route", "")))
            nodes.append(item)
        directed = sorted((index[source], index[target]) for source, target in graph.directed_edges)
        bidirectional = sorted(
            tuple(sorted((index[source], index[target])))
            for source, target in graph.bidirectional_edges
        )
        candidates.append(
            json.dumps([nodes, directed, bidirectional], ensure_ascii=False, separators=(",", ":"))
        )
    return min(candidates, default="[[],[],[]]")


@dataclass(frozen=True)
class DensityCorrection:
    advantages: tuple[float, ...]
    densities: tuple[float, ...]


def correctness_gated_diversity_rewards(
    trajectories: Iterable[TokenizedDirectorTrajectory],
    *,
    max_bonus: float = 0.1,
) -> list[TokenizedDirectorTrajectory]:
    """Add a small novelty bonus only to finished, verifier-passed sibling graphs.

    Similarity is role-label free: 80% comes from permutation-invariant graph
    structure and 20% from the aggregate free-text delegation vocabulary.  The
    bonus never rewards an incorrect graph or raw Agent/edge count by itself.
    """

    if not 0.0 <= float(max_bonus) <= 1.0:
        raise ValueError("max_bonus must be in [0, 1]")
    items = list(trajectories)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped[item.task_id].append(index)
    adjusted = list(items)
    for indices in grouped.values():
        graphs = [MultiAgentGraph.from_dict(items[index].graph) for index in indices]
        features = [structural_features(graph) for graph in graphs]
        prompt_tokens = [_graph_prompt_tokens(graph) for graph in graphs]
        for local_index, item_index in enumerate(indices):
            item = items[item_index]
            if item.metadata.get("diversity_reward_applied"):
                continue
            similarities = [
                0.8 * graph_kernel(features[local_index], features[other])
                + 0.2 * _jaccard(prompt_tokens[local_index], prompt_tokens[other])
                for other in range(len(indices))
                if other != local_index
            ]
            novelty = 1.0 - max(similarities, default=1.0)
            verification = item.metadata.get("verification")
            passed = bool(
                item.metadata.get("finished")
                and item.metadata.get("protocol_qualified", True)
                and isinstance(verification, dict)
                and verification.get("passed")
            )
            base_reward = float(item.metadata.get("base_director_reward", item.reward))
            outcome_only = item.metadata.get("reward_semantics") in {
                "strict_outcome_only",
                "outcome_only",
            }
            bonus = float(max_bonus) * novelty if passed and not outcome_only else 0.0
            metadata = {
                **item.metadata,
                "base_director_reward": base_reward,
                "graph_novelty": novelty,
                "graph_diversity_bonus": bonus,
                "graph_diversity_eligible": passed,
                "graph_diversity_suppressed_by_outcome_only": bool(passed and outcome_only),
                "diversity_reward_applied": True,
            }
            adjusted[item_index] = replace(
                item,
                reward=base_reward + bonus,
                metadata=metadata,
            )
    return adjusted


def _graph_prompt_tokens(graph: MultiAgentGraph) -> set[str]:
    return {
        token
        for node in graph.nodes.values()
        for token in re.findall(r"[a-z0-9_]+", node.prompt.casefold())
        if len(token) > 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def correct_positive_density(
    advantages: Iterable[float],
    features: Iterable[GraphFeatures],
    *,
    kernel_matrix: Iterable[Iterable[float]] | None = None,
) -> DensityCorrection:
    """Kernel-density correction for positive advantages with mass preservation."""

    values = [float(value) for value in advantages]
    vectors = list(features)
    if len(values) != len(vectors):
        raise ValueError("advantages and features must have equal length")
    matrix = (
        tuple(tuple(float(value) for value in row) for row in kernel_matrix)
        if kernel_matrix is not None
        else graph_kernel_matrix(vectors)
    )
    if len(matrix) != len(values) or any(len(row) != len(values) for row in matrix):
        raise ValueError("kernel matrix shape does not match advantages")
    positive = [index for index, value in enumerate(values) if value > 0.0]
    densities = [1.0] * len(values)
    raw = list(values)
    for index in positive:
        densities[index] += sum(matrix[index][other] for other in positive if other != index)
        raw[index] = values[index] / densities[index]
    original_mass = sum(values[index] for index in positive)
    corrected_mass = sum(raw[index] for index in positive)
    scale = original_mass / corrected_mass if corrected_mass else 1.0
    for index in positive:
        raw[index] *= scale
    return DensityCorrection(tuple(raw), tuple(densities))


def build_graph_training_batch(
    trajectories: Iterable[TokenizedDirectorTrajectory],
    *,
    features_by_rollout: dict[str, GraphFeatures] | None = None,
    kernels_by_task: dict[str, tuple[tuple[float, ...], ...]] | None = None,
) -> TrainingBatch:
    """Build a Director batch and stop at the explicit pre-optimizer boundary."""

    items = list(trajectories)
    reward_versions = {
        str(item.metadata.get("director_reward_version", LEGACY_REWARD_VERSION)) for item in items
    }
    if len(reward_versions) > 1:
        versions = ", ".join(sorted(reward_versions))
        raise ValueError(
            f"refusing to mix Director reward versions in one training batch: {versions}"
        )
    scores = [
        GraphRolloutScore(
            example_id=item.task_id,
            rollout_id=item.rollout_id,
            task_reward=item.reward,
            valid_graph=True,
            action_mask=item.action_mask,
        )
        for item in items
    ]
    base = graph_level_advantages(scores)
    graphs = [MultiAgentGraph.from_dict(item.graph) for item in items]
    fallback_features = execution_policy_features_many(graphs)
    features = [
        (features_by_rollout or {}).get(item.rollout_id, feature)
        for item, feature in zip(items, fallback_features, strict=True)
    ]
    corrected_values = [float(base[item.rollout_id]) for item in items]
    density_values = [1.0] * len(items)
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped_indices[item.task_id].append(index)
    for task_id, indices in grouped_indices.items():
        task_features = [features[index] for index in indices]
        matrix = (kernels_by_task or {}).get(task_id)
        correction = correct_positive_density(
            (corrected_values[index] for index in indices),
            task_features,
            kernel_matrix=matrix,
        )
        for local, index in enumerate(indices):
            corrected_values[index] = correction.advantages[local]
            density_values[index] = correction.densities[local]
    samples = tuple(
        TrainingSample(
            rollout_id=item.rollout_id,
            task_id=item.task_id,
            token_ids=item.token_ids,
            action_mask=item.action_mask,
            reward=item.reward,
            advantage=corrected_values[index],
            density=density_values[index],
            canonical_graph_key=canonical_graph_key(
                graphs[index], include_prompts=bool(graphs[index].runtime_routes)
            ),
            graph_features=features[index].values,
            metadata={
                **item.metadata,
                "group_reward_mean": statistics.fmean(
                    items[i].reward for i in grouped_indices[item.task_id]
                ),
                "group_reward_std": max(
                    0.01, statistics.pstdev(items[i].reward for i in grouped_indices[item.task_id])
                ),
                "base_group_advantage": base[item.rollout_id],
                "final_group_advantage": corrected_values[index],
                "seed": item.seed,
                "executor_version": item.executor_version,
                "director_reward_version": item.metadata.get(
                    "director_reward_version", LEGACY_REWARD_VERSION
                ),
                "topology_key": canonical_graph_key(graphs[index]),
                "execution_policy_key": canonical_graph_key(graphs[index], include_prompts=True),
                "graph_feature_schema_id": features[index].schema_id,
                "graph_feature_schema": (
                    features[index].schema.to_dict() if features[index].schema else None
                ),
            },
            policy_calls=item.policy_calls,
        )
        for index, item in enumerate(items)
    )
    return TrainingBatch(role="solver", samples=samples)
