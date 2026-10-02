from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import GraphValidationError, MultiAgentGraph


@dataclass(frozen=True)
class FixedDatasetExample:
    example_id: str
    task: str
    reference: Any = None
    metadata: dict[str, Any] | None = None


def load_fixed_jsonl(path: str | Path) -> list[FixedDatasetExample]:
    """Load a stable fixed-dataset snapshot without sampling or mutation."""

    examples: list[FixedDatasetExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            task = str(
                payload.get("task", payload.get("prompt", payload.get("problem", "")))
            ).strip()
            if not task:
                raise ValueError(f"missing task at JSONL line {line_index}")
            examples.append(
                FixedDatasetExample(
                    example_id=str(payload.get("id", line_index)),
                    task=task,
                    reference=(
                        payload["target_answers"][0]
                        if len(payload.get("target_answers", [])) == 1
                        else payload.get(
                            "target_answers", payload.get("reference", payload.get("answer"))
                        )
                    ),
                    metadata={
                        **dict(payload.get("metadata", {})),
                        **{
                            key: payload[key]
                            for key in (
                                "dataset",
                                "split",
                                "mode",
                                "task_type",
                                "verifier",
                                "context_documents",
                            )
                            if key in payload
                        },
                    },
                )
            )
    return examples


@dataclass(frozen=True)
class GraphRolloutScore:
    """Graph-level task reward record; no hand-written structure reward is included."""

    example_id: str
    rollout_id: str
    task_reward: float
    valid_graph: bool
    action_mask: tuple[int, ...] = ()


def compute_group_advantages(rewards: Iterable[float], *, min_std: float = 0.01) -> list[float]:
    """FlowSteer GRPO group normalization, kept independent of a trainer backend."""

    values = [float(reward) for reward in rewards]
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    mean_reward = sum(values) / len(values)
    variance = sum((reward - mean_reward) ** 2 for reward in values) / len(values)
    std_reward = max(math.sqrt(variance), min_std)
    return [(reward - mean_reward) / std_reward for reward in values]


def graph_level_advantages(scores: Iterable[GraphRolloutScore]) -> dict[str, float]:
    """Normalize task rewards only among sibling graphs for the same fixed example."""

    groups: dict[str, list[GraphRolloutScore]] = defaultdict(list)
    for score in scores:
        groups[score.example_id].append(score)
    advantages: dict[str, float] = {}
    for group in groups.values():
        normalized = compute_group_advantages(score.task_reward for score in group)
        advantages.update(
            (score.rollout_id, advantage)
            for score, advantage in zip(group, normalized, strict=True)
        )
    return advantages


@dataclass(frozen=True)
class RelationCounterfactual:
    source: str
    target: str
    relation: str
    q_absent: float
    q_present: float
    probability_present: float
    advantage_absent: float
    advantage_present: float

    @property
    def marginal_gain(self) -> float:
        return self.q_present - self.q_absent


def relation_counterfactual_probe(
    graph: MultiAgentGraph,
    *,
    source: str,
    target: str,
    probability_present: float,
    evaluate: Callable[[MultiAgentGraph, int], float],
    seed: int = 0,
) -> RelationCounterfactual:
    """Evaluate absent/present sibling graphs with the same seed.

    Same-layer pairs use a bidirectional relation. Cross-layer pairs use one
    lower-to-higher directed relation, exactly matching the binary choices in
    the project design.
    """

    probability_present = float(probability_present)
    if not 0.0 <= probability_present <= 1.0:
        raise ValueError("probability_present must be in [0, 1]")
    graph.require_node(source)
    graph.require_node(target)
    absent = _without_pair_relation(graph, source, target)
    present = absent.clone()
    source_layer = present.nodes[source].layer
    target_layer = present.nodes[target].layer
    if source_layer == target_layer:
        relation = "bidirectional"
        present.set_relation(source, target, relation)
    else:
        relation = "directed"
        edge_source, edge_target = (
            (source, target) if source_layer < target_layer else (target, source)
        )
        present.set_relation(edge_source, edge_target, relation)
    q_absent = float(evaluate(absent, seed))
    q_present = float(evaluate(present, seed))
    baseline = (1.0 - probability_present) * q_absent + probability_present * q_present
    return RelationCounterfactual(
        source=source,
        target=target,
        relation=relation,
        q_absent=q_absent,
        q_present=q_present,
        probability_present=probability_present,
        advantage_absent=q_absent - baseline,
        advantage_present=q_present - baseline,
    )


def _without_pair_relation(graph: MultiAgentGraph, source: str, target: str) -> MultiAgentGraph:
    sibling = graph.clone()
    pair = tuple(sorted((source, target)))
    if pair in sibling.bidirectional_edges:
        sibling.remove_relation(source, target, "bidirectional")
    elif (source, target) in sibling.directed_edges:
        sibling.remove_relation(source, target, "directed")
    elif (target, source) in sibling.directed_edges:
        sibling.remove_relation(target, source, "directed")
    try:
        sibling.assert_valid(final=False)
    except GraphValidationError as exc:
        raise GraphValidationError(f"invalid counterfactual prefix: {exc}") from exc
    return sibling
