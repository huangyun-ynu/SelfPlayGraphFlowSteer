from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .actions import ActionParser, ActionType, CanvasAction
from .graph import MultiAgentGraph
from .observability import ExecutionTrace


@dataclass(frozen=True)
class RelationDecision:
    # Index into DirectorRun.turns/action_token_spans, not TraceEvent.sequence.
    action_index: int
    source: str
    target: str
    probability_present: float
    graph_prefix: dict[str, Any]
    chosen_present: bool
    event_sequence: int = -1
    suffix_events: tuple[dict[str, Any], ...] = ()
    prefix_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_final_graph: dict[str, Any] = field(default_factory=dict)
    policy_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationCredit:
    rollout_id: str
    action_index: int
    source: str
    target: str
    relation: str
    q_absent: float
    q_present: float
    advantage_absent: float
    advantage_present: float
    seed: int
    action_token_span: tuple[int, int] | None = None
    chosen_present: bool = True
    probability_absent: float = 0.5
    probability_present: float = 0.5
    log_probability_absent: float | None = None
    log_probability_present: float | None = None
    choice_token_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def schedule_relation_decisions(
    trace: ExecutionTrace,
    *,
    limit: int = 1,
    seed: int = 0,
    action_token_spans: Sequence[tuple[int, int] | None] | None = None,
) -> list[RelationDecision]:
    """Select at most one auditable relation choice closest to policy uncertainty.

    Canvas traces also contain deterministic protocol-recovery mutations.  They
    have no sampled Director tokens and must never receive local policy credit.
    Only the separate, grammar-constrained off/on turn is eligible. Legacy
    set/remove actions and incomplete provider probability payloads fail closed.
    For repeated edits of one pair, only its latest effective decision remains.
    """

    del seed

    canvas_events = [event for event in trace.events if event.kind == "canvas_step"]
    candidates_by_pair: dict[tuple[str, str], RelationDecision] = {}
    inferred_director_turn = 0
    latest_artifacts: dict[str, dict[str, Any]] = {}
    for event_index, event in enumerate(canvas_events):
        is_recovery = bool(event.payload.get("protocol_recovery"))
        explicit_turn = event.payload.get("director_turn_index")
        if "director_turn_index" in event.payload:
            director_turn = None if explicit_turn is None else int(explicit_turn)
        elif is_recovery:
            director_turn = None
        else:
            # Backward-compatible reconstruction for old traces.  Counting only
            # model-originated Canvas steps is safe; using event.sequence is not.
            director_turn = inferred_director_turn
        if not is_recovery:
            inferred_director_turn = max(inferred_director_turn, int(director_turn or 0) + 1)
        execution = event.payload.get("execution")
        artifacts_before = copy.deepcopy(latest_artifacts)
        if is_recovery or director_turn is None or not event.payload.get("accepted"):
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        relation_decision = event.payload.get("relation_decision")
        if not isinstance(relation_decision, dict) or relation_decision.get("phase") != "choice":
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        source = str(relation_decision.get("source", "")).strip()
        target = str(relation_decision.get("target", "")).strip()
        if not source or not target or source == target:
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        if action_token_spans is not None and (
            director_turn < 0
            or director_turn >= len(action_token_spans)
            or action_token_spans[director_turn] is None
            or action_token_spans[director_turn][1] - action_token_spans[director_turn][0] != 1
        ):
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        policy = relation_decision.get("policy")
        if not isinstance(policy, dict):
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        probabilities = policy.get("probabilities")
        log_probabilities = policy.get("log_probabilities")
        token_ids = policy.get("token_ids")
        if not all(
            isinstance(item, dict) for item in (probabilities, log_probabilities, token_ids)
        ):
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        try:
            probability_absent = float(probabilities["off"])
            probability_present = float(probabilities["on"])
            log_probability_absent = float(log_probabilities["off"])
            log_probability_present = float(log_probabilities["on"])
            off_token_id = int(token_ids["off"])
            on_token_id = int(token_ids["on"])
            choice_token_id = int(token_ids[str(relation_decision.get("choice", ""))])
        except (KeyError, TypeError, ValueError):
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        if (
            not 0.0 <= probability_absent <= 1.0
            or not 0.0 <= probability_present <= 1.0
            or abs(probability_absent + probability_present - 1.0) > 1e-6
            or not math.isfinite(log_probability_absent)
            or not math.isfinite(log_probability_present)
            or policy.get("choice") != relation_decision.get("choice")
            or relation_decision.get("choice") not in {"off", "on"}
            or bool(relation_decision.get("chosen_present"))
            != (relation_decision.get("choice") == "on")
            or off_token_id == on_token_id
        ):
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        raw_prefix = event.payload.get("graph_before")
        has_recorded_prefix = isinstance(raw_prefix, dict) and "nodes" in raw_prefix
        evaluation_graph = raw_prefix if has_recorded_prefix else trace.final_graph
        if not isinstance(evaluation_graph, dict) or "nodes" not in evaluation_graph:
            latest_artifacts = _execution_artifacts(execution, latest_artifacts)
            continue
        suffix_events = (
            tuple(
                {
                    "sequence": later.sequence,
                    "raw_action": str(later.payload.get("raw_action", "")),
                    "graph_before": copy.deepcopy(later.payload.get("graph_before", {})),
                    "graph": copy.deepcopy(later.payload.get("graph", {})),
                    "protocol_recovery": bool(later.payload.get("protocol_recovery")),
                    "relation_decision": copy.deepcopy(later.payload.get("relation_decision", {})),
                }
                for later in canvas_events[event_index + 1 :]
                if later.payload.get("accepted")
            )
            if has_recorded_prefix
            else ()
        )
        candidates_by_pair[tuple(sorted((source, target)))] = RelationDecision(
            action_index=director_turn,
            source=source,
            target=target,
            probability_present=probability_present,
            graph_prefix=copy.deepcopy(evaluation_graph),
            chosen_present=bool(relation_decision.get("chosen_present")),
            event_sequence=event.sequence,
            suffix_events=suffix_events,
            prefix_artifacts=artifacts_before,
            expected_final_graph=copy.deepcopy(trace.final_graph),
            policy_audit={
                **copy.deepcopy(policy),
                "probability_absent": probability_absent,
                "probability_present": probability_present,
                "log_probability_absent": log_probability_absent,
                "log_probability_present": log_probability_present,
                "choice_token_id": choice_token_id,
            },
        )
        latest_artifacts = _execution_artifacts(execution, latest_artifacts)
    # A later DELETE_AGENT makes the relation choice structurally ineffective
    # in the terminal graph.  Such a token keeps graph-level advantage; it is
    # not a valid local off/on probe because one branch endpoint no longer
    # exists at the replay/evaluation boundary.
    final_node_ids = set(MultiAgentGraph.from_dict(trace.final_graph).nodes)
    candidates = [
        item
        for item in candidates_by_pair.values()
        if item.source in final_node_ids and item.target in final_node_ids
    ]
    count = min(1, max(0, limit), len(candidates))
    if count == 0:
        return []
    candidates.sort(
        key=lambda item: (
            abs(item.probability_present - 0.5),
            item.action_index,
            item.source,
            item.target,
        )
    )
    return candidates[:count]


def evaluate_relation_decision(
    decision: RelationDecision,
    *,
    rollout_id: str,
    seed: int,
    evaluate: Callable[[MultiAgentGraph, int], float],
    evaluate_pair: Callable[
        [MultiAgentGraph, MultiAgentGraph, int], tuple[float, float]
    ]
    | None = None,
    evaluate_from_prefix: Callable[
        [MultiAgentGraph, int, dict[str, dict[str, Any]], set[str]], float
    ]
    | None = None,
    action_token_span: tuple[int, int] | None = None,
) -> RelationCredit:
    absent, absent_dirty, relation = _replay_relation_branch(decision, present=False)
    present, present_dirty, _ = _replay_relation_branch(decision, present=True)
    if evaluate_from_prefix is not None:
        raise ValueError("prefix Artifact replay is retired; use full_graph_v1")
    left, right = _graph_without_version(absent), _graph_without_version(present)
    left.pop("relations", None)
    right.pop("relations", None)
    if left != right:
        raise ValueError("relation siblings changed non-relation graph configuration")
    if evaluate_pair is None:
        q_absent = float(evaluate(absent, seed))
        q_present = float(evaluate(present, seed))
    else:
        q_absent, q_present = (
            float(value) for value in evaluate_pair(absent, present, seed)
        )
    if not all(math.isfinite(q) for q in (q_absent, q_present)):
        raise ValueError("incomplete/non-finite counterfactual final score")
    baseline = (
        1.0 - decision.probability_present
    ) * q_absent + decision.probability_present * q_present
    return RelationCredit(
        rollout_id=rollout_id,
        action_index=decision.action_index,
        source=decision.source,
        target=decision.target,
        relation=relation,
        q_absent=q_absent,
        q_present=q_present,
        advantage_absent=q_absent - baseline,
        advantage_present=q_present - baseline,
        seed=seed,
        action_token_span=action_token_span,
        chosen_present=decision.chosen_present,
        probability_absent=1.0 - decision.probability_present,
        probability_present=decision.probability_present,
        log_probability_absent=decision.policy_audit.get("log_probability_absent"),
        log_probability_present=decision.policy_audit.get("log_probability_present"),
        choice_token_id=decision.policy_audit.get("choice_token_id"),
    )


def _execution_artifacts(
    execution: object,
    fallback: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(execution, dict):
        return fallback
    artifacts = execution.get("artifacts")
    if not isinstance(artifacts, dict):
        return fallback
    return {
        str(agent_id): copy.deepcopy(payload)
        for agent_id, payload in artifacts.items()
        if isinstance(payload, dict)
    }


def _replay_relation_branch(
    decision: RelationDecision,
    *,
    present: bool,
) -> tuple[MultiAgentGraph, set[str], str]:
    graph = MultiAgentGraph.from_dict(decision.graph_prefix)
    dirty, relation = _set_binary_relation(
        graph,
        source=decision.source,
        target=decision.target,
        present=present,
    )
    parser = ActionParser()
    for event in decision.suffix_events:
        relation_payload = event.get("relation_decision")
        if isinstance(relation_payload, dict):
            phase = relation_payload.get("phase")
            if phase == "proposal":
                continue
            if phase == "choice":
                other_source = str(relation_payload.get("source", ""))
                other_target = str(relation_payload.get("target", ""))
                if {other_source, other_target} == {decision.source, decision.target}:
                    raise ValueError("selected relation is not the latest effective pair decision")
                mutation_dirty, _ = _set_binary_relation(
                    graph,
                    source=other_source,
                    target=other_target,
                    present=bool(relation_payload.get("chosen_present")),
                )
                dirty.update(mutation_dirty)
                graph.assert_valid(final=False)
                continue
        action = parser.parse(str(event.get("raw_action", "")))
        if not action.valid:
            raise ValueError(f"invalid accepted replay action at sequence {event.get('sequence')}")
        if action.action_type is ActionType.FINISH:
            continue
        if _same_relation_pair(action, decision.source, decision.target):
            raise ValueError("selected relation is not the latest effective pair decision")
        after_payload = event.get("graph")
        if not isinstance(after_payload, dict):
            raise ValueError(f"missing replay graph at sequence {event.get('sequence')}")
        mutation_dirty = _apply_replay_action(
            graph, action, after_payload, recorded_before=event.get("graph_before")
        )
        dirty.update(mutation_dirty)
        graph.assert_valid(final=False)

    if decision.expected_final_graph:
        expected = MultiAgentGraph.from_dict(decision.expected_final_graph)
        _set_binary_relation(
            expected,
            source=decision.source,
            target=decision.target,
            present=present,
        )
        # The runtime refreshes these accounting fields during Worker execution,
        # outside sampled Canvas mutations. Restore the recorded terminal budget
        # on both siblings; all policy-selected node fields remain strict.
        for agent_id in graph.nodes.keys() & expected.nodes.keys():
            recorded = expected.nodes[agent_id].metadata
            replayed = graph.nodes[agent_id].metadata
            for key in (
                "_runtime_token_credit",
                "_runtime_budget_kind",
                "_runtime_finalization_output_reserve",
                "_runtime_reserved_closure_tokens",
                "_runtime_budget_phase",
                "_runtime_webshop_output_closure",
            ):
                if key in recorded:
                    replayed[key] = copy.deepcopy(recorded[key])
                else:
                    replayed.pop(key, None)
        if _graph_without_version(graph) != _graph_without_version(expected):
            raise ValueError("counterfactual suffix replay diverged from the recorded graph")
    return graph, dirty, relation


def _set_binary_relation(
    graph: MultiAgentGraph,
    *,
    source: str,
    target: str,
    present: bool,
) -> tuple[set[str], str]:
    graph.require_node(source)
    graph.require_node(target)
    dirty: set[str] = set()
    pair = tuple(sorted((source, target)))
    if pair in graph.bidirectional_edges:
        dirty.update(graph.remove_relation(source, target, "bidirectional").dirty_agents)
    elif (source, target) in graph.directed_edges:
        dirty.update(graph.remove_relation(source, target, "directed").dirty_agents)
    elif (target, source) in graph.directed_edges:
        dirty.update(graph.remove_relation(target, source, "directed").dirty_agents)

    source_layer = graph.nodes[source].layer
    target_layer = graph.nodes[target].layer
    if source_layer == target_layer:
        relation = "bidirectional"
        edge_source, edge_target = source, target
    else:
        relation = "directed"
        edge_source, edge_target = (
            (source, target) if source_layer < target_layer else (target, source)
        )
    if present:
        dirty.update(graph.set_relation(edge_source, edge_target, relation).dirty_agents)
    return dirty, relation


def _same_relation_pair(action: CanvasAction, source: str, target: str) -> bool:
    if action.action_type not in {ActionType.SET_RELATION, ActionType.REMOVE_RELATION}:
        return False
    return {str(action.source), str(action.target)} == {source, target}


def _apply_replay_action(
    graph: MultiAgentGraph,
    action: CanvasAction,
    recorded_after: dict[str, Any],
    *,
    recorded_before: dict[str, Any] | None = None,
) -> set[str]:
    after = MultiAgentGraph.from_dict(recorded_after)
    kind = action.action_type
    if kind is ActionType.ADD_AGENT:
        new_ids = set(after.nodes) - set(graph.nodes)
        if len(new_ids) != 1:
            raise ValueError("ADD_AGENT replay requires exactly one recorded new Agent")
        agent_id = next(iter(new_ids))
        mutation = graph.add_agent(agent_id)
        # The Canvas may attach dataset Action-policy metadata at ADD_AGENT time.
        # Copy that deterministic node configuration from the recorded transition.
        graph.nodes[agent_id] = copy.deepcopy(after.nodes[agent_id])
        return set(mutation.dirty_agents)
    if kind is ActionType.SET_PROMPT:
        target = str(action.target)
        node = after.require_node(target)
        mutation = graph.set_prompt(
            target,
            node.prompt,
            runtime_route=str(node.metadata.get("runtime_route", "")).strip() or None,
            structural_operator=node.structural_operator,
        )
        graph.nodes[target] = copy.deepcopy(node)
        return set(mutation.dirty_agents)
    if kind is ActionType.SET_MODEL:
        target = str(action.target)
        mutation = graph.set_model(target, str(action.runtime_route))
        if graph.nodes[target].metadata.get("runtime_route") != after.nodes[target].metadata.get(
            "runtime_route"
        ):
            raise ValueError("SET_MODEL replay route differs from recorded policy choice")
        graph.nodes[target] = copy.deepcopy(after.nodes[target])
        return set(mutation.dirty_agents)
    if kind is ActionType.SET_LAYER:
        return set(graph.set_layer(str(action.target), int(action.layer)).dirty_agents)
    if kind is ActionType.SET_RELATION:
        return set(
            graph.set_relation(str(action.source), str(action.target), action.relation).dirty_agents
        )
    if kind is ActionType.REMOVE_RELATION:
        return set(
            graph.remove_relation(
                str(action.source), str(action.target), action.relation
            ).dirty_agents
        )
    if kind is ActionType.DELETE_AGENT:
        return set(graph.delete_agent(str(action.target or action.agent_id)).dirty_agents)
    if kind is ActionType.SET_OUTPUT:
        selected = str(action.target)
        graph.set_output(selected)
        # Canvas SET_OUTPUT also transfers the adapter's single-committer
        # capability. Use the recorded target to select the known capability;
        # retain strict comparison for arbitrary metadata or policy drift.
        capabilities = after.require_node(selected).metadata.get("exclusive_capabilities", ())
        for capability in ("environment_commit", "code_commit"):
            if capability in capabilities:
                graph.assign_exclusive_capability(selected, capability)
        removed = set(graph.nodes) - set(after.nodes)
        if removed:
            if not recorded_before or "nodes" not in recorded_before:
                raise ValueError("SET_OUTPUT pruning replay requires the recorded before graph")
            before = MultiAgentGraph.from_dict(recorded_before)
            before.set_output(selected)
            expected_removed = {
                agent_id
                for agent_id, node in before.nodes.items()
                if agent_id != selected
                and (not node.configured or selected not in before.reachable_from(agent_id))
            }
            if set(before.nodes) != set(graph.nodes) or removed != expected_removed:
                raise ValueError("SET_OUTPUT pruning differs from recorded deterministic cleanup")
            # Replay the factual suffix cleanup on both siblings. Recomputing
            # reachability on the intervened graph would delete additional nodes
            # in the off branch and change more than the selected relation.
            for agent_id in sorted(removed):
                graph.delete_agent(agent_id)
        # Output selection changes no Worker input and therefore creates no
        # counterfactual execution dirtiness.
        return set()
    raise ValueError(f"unsupported replay action: {kind.value}")


def _graph_without_version(graph: MultiAgentGraph) -> dict[str, Any]:
    payload = graph.to_dict()
    payload.pop("version", None)
    return payload
