from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from typing import Any

from .delegation import compare_responsibilities

SWE_READ_ONLY_ACTIONS = frozenset({"swe_list", "swe_search", "swe_read", "swe_status"})
SWE_EDIT_ACTIONS = frozenset({"swe_edit", "swe_apply_artifact"})
MIN_SHARED_EXACT_READS = 3
MIN_LARGE_SHARED_EXACT_READS = 5
MIN_EXACT_READ_OVERLAP_RATIO = 0.5


def audit_cross_agent_read_overlap(
    events: list[object],
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Audit duplicated SWE exploration without changing execution or rewards."""

    nodes = {
        str(node.get("agent_id")): node
        for node in graph.get("nodes", ())
        if isinstance(node, dict) and node.get("agent_id")
    }
    relation_pairs = {
        _pair(str(relation.get("source")), str(relation.get("target")))
        for relation in graph.get("relations", ())
        if isinstance(relation, dict) and relation.get("source") and relation.get("target")
    }
    reads: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    successful_edits: dict[str, int] = defaultdict(int)
    artifact_ids: dict[str, set[str]] = defaultdict(set)
    source_artifact_ids: dict[str, set[str]] = defaultdict(set)
    seen_calls: set[tuple[str, str]] = set()

    for event in events:
        payload = _event_payload(event)
        execution = payload.get("execution")
        if not isinstance(execution, dict):
            continue
        artifacts = execution.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for artifact_key, artifact in artifacts.items():
            if not isinstance(artifact, dict):
                continue
            agent_id = str(artifact.get("agent_id") or artifact_key)
            artifact_id = str(artifact.get("artifact_id") or artifact_key)
            if artifact_id:
                artifact_ids[agent_id].add(artifact_id)
            source_artifact_ids[agent_id].update(
                str(value) for value in artifact.get("source_artifact_ids", ()) if str(value)
            )
            trace = artifact.get("react_trace")
            if not isinstance(trace, list):
                continue
            for index, turn in enumerate(trace):
                if not isinstance(turn, dict):
                    continue
                action = turn.get("action")
                observation = turn.get("observation")
                if not isinstance(action, dict) or not isinstance(observation, dict):
                    continue
                call_id = str(action.get("call_id") or f"{artifact_id}:{index}")
                call_identity = (agent_id, call_id)
                if call_identity in seen_calls:
                    continue
                seen_calls.add(call_identity)
                name = str(action.get("name", "")).strip()
                arguments = action.get("arguments", {})
                if not isinstance(arguments, dict) or not _observation_succeeded(observation):
                    continue
                if name in SWE_READ_ONLY_ACTIONS:
                    signature = _read_signature(name, arguments)
                    reads[agent_id][signature] = {
                        "name": name,
                        "arguments": _json_safe(arguments),
                    }
                elif name in SWE_EDIT_ACTIONS and _edit_advanced_workspace(arguments, observation):
                    successful_edits[agent_id] += 1

    pair_audits: list[dict[str, Any]] = []
    for left_id, right_id in combinations(sorted(nodes), 2):
        shared = set(reads[left_id]) & set(reads[right_id])
        if not shared:
            continue
        smaller_read_count = min(len(reads[left_id]), len(reads[right_id]))
        overlap_ratio = len(shared) / smaller_read_count if smaller_read_count else 0.0
        relation_present = _pair(left_id, right_id) in relation_pairs
        shared_versions = {
            int(version)
            for signature in shared
            for version in [reads[left_id][signature]["arguments"].get("workspace_version")]
            if isinstance(version, int) and not isinstance(version, bool)
        }
        same_workspace_version = bool(shared_versions)
        evidence_relayed = bool(
            artifact_ids[left_id] & source_artifact_ids[right_id]
            or artifact_ids[right_id] & source_artifact_ids[left_id]
        )
        left_fields = _director_fields(nodes[left_id])
        right_fields = _director_fields(nodes[right_id])
        comparison = compare_responsibilities(left_fields, right_fields)
        edit_count = successful_edits[left_id] + successful_edits[right_id]
        flagged = bool(
            comparison.high_confidence_duplicate
            and not relation_present
            and same_workspace_version
            and (
                len(shared) >= MIN_LARGE_SHARED_EXACT_READS
                or (
                    len(shared) >= MIN_SHARED_EXACT_READS
                    and overlap_ratio >= MIN_EXACT_READ_OVERLAP_RATIO
                )
            )
            and edit_count == 0
            and not evidence_relayed
        )
        pair_audits.append(
            {
                "agents": [left_id, right_id],
                "shared_exact_read_count": len(shared),
                "smaller_agent_read_count": smaller_read_count,
                "exact_read_overlap_ratio": round(overlap_ratio, 6),
                "same_workspace_version": same_workspace_version,
                "shared_workspace_versions": sorted(shared_versions),
                "relation_present": relation_present,
                "evidence_relayed": evidence_relayed,
                "successful_edit_count": edit_count,
                "responsibility_overlap": comparison.to_dict(),
                "duplicate_read_only_exploration": flagged,
                "shared_reads": [reads[left_id][signature] for signature in sorted(shared)[:12]],
            }
        )

    pair_audits.sort(
        key=lambda item: (
            not bool(item["duplicate_read_only_exploration"]),
            -int(item["shared_exact_read_count"]),
            item["agents"],
        )
    )
    flagged_count = sum(int(bool(item["duplicate_read_only_exploration"])) for item in pair_audits)
    return {
        "schema_version": 1,
        "policy": "record_only",
        "minimum_shared_exact_reads": MIN_SHARED_EXACT_READS,
        "minimum_large_shared_exact_reads": MIN_LARGE_SHARED_EXACT_READS,
        "minimum_exact_read_overlap_ratio": MIN_EXACT_READ_OVERLAP_RATIO,
        "pair_count": len(pair_audits),
        "flagged_pair_count": flagged_count,
        "duplicate_read_only_exploration": flagged_count > 0,
        "pairs": pair_audits,
    }


def _event_payload(event: object) -> dict[str, Any]:
    if isinstance(event, dict):
        payload = event.get("payload", event)
    else:
        payload = getattr(event, "payload", {})
    return payload if isinstance(payload, dict) else {}


def _director_fields(node: dict[str, Any]) -> dict[str, object]:
    metadata = node.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    fields = metadata.get("director_delegation", {})
    return fields if isinstance(fields, dict) else {}


def _observation_succeeded(observation: dict[str, Any]) -> bool:
    if str(observation.get("status", "")).casefold() != "ok":
        return False
    output = observation.get("output")
    return not (
        isinstance(output, dict)
        and str(output.get("status", "")).casefold() in {"error", "failed", "rejected"}
    )


def _edit_advanced_workspace(arguments: dict[str, Any], observation: dict[str, Any]) -> bool:
    before = arguments.get("workspace_version")
    output = observation.get("output")
    after = output.get("workspace_version") if isinstance(output, dict) else None
    return (
        isinstance(before, int)
        and not isinstance(before, bool)
        and isinstance(after, int)
        and not isinstance(after, bool)
        and after > before
    )


def _read_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {"name": name, "arguments": _json_safe(arguments)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]
