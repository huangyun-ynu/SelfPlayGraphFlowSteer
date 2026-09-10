"""Deterministic, sample-local admission for partial sibling groups."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

SCHEMA = "eligible_subset_v1"
INDEPENDENT_SCHEMA = "independent_frontier_v2"
SELECTION_SCHEMAS = {SCHEMA, INDEPENDENT_SCHEMA}


def trajectory_exclusions(trajectory: Any) -> list[str]:
    metadata = trajectory.metadata
    reasons = list(metadata.get("training_exclusion_reasons", ()))
    if "training_eligible" in metadata and not metadata["training_eligible"]:
        reasons.append("training_ineligible")
    if metadata.get("terminal_graph_status") == "unsafe_partial":
        reasons.append("unsafe_partial")
    if metadata.get("reward_known") is False or not math.isfinite(trajectory.reward):
        reasons.append("untrusted_reward")
    calls = trajectory.policy_calls
    if calls:
        if any(not call.behavior_log_probs for call in calls):
            reasons.append("missing_behavior_log_probs")
        if any(not all(math.isfinite(x) for x in call.behavior_log_probs) for call in calls):
            reasons.append("nonfinite_behavior_log_probs")
        if not any(len(c.token_ids) >= 2 and any(c.action_mask[1:]) for c in calls):
            reasons.append("missing_action_targets")
    elif len(trajectory.token_ids) < 2 or not any(trajectory.action_mask[1:]):
        reasons.append("missing_action_targets")
    return sorted(set(reasons))


def build_training_selection(
    proposals: list[Any], rollouts: dict[str, Any], planned: int, *, schema: str = SCHEMA
) -> dict:
    if schema not in SELECTION_SCHEMAS:
        raise ValueError("unknown training selection schema")
    if planned < 2:
        raise ValueError("partial-group training requires at least two planned rollouts")
    task_ids = [p.task.task_id for p in proposals]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate planned task IDs")
    planned_ids = {f"{task}-r{i}" for task in task_ids for i in range(planned)}
    if set(rollouts) - planned_ids:
        raise ValueError("unplanned rollout IDs in training selection")
    groups = []
    for proposal in proposals:
        task_id = proposal.task.task_id
        rows = []
        for index in range(planned):
            rid = f"{task_id}-r{index}"
            rollout = rollouts.get(rid)
            trajectory = rollout.trajectory if rollout else None
            if trajectory is not None and trajectory.task_id != task_id:
                raise ValueError("rollout task identity does not match planned slot")
            reasons = trajectory_exclusions(trajectory) if trajectory else ["missing_trajectory"]
            rows.append(
                {
                    "rollout_id": rid,
                    "eligible": not reasons,
                    "reward": trajectory.reward if trajectory else None,
                    "exclusion_reasons": reasons,
                    "source_sha256": hashlib.sha256(
                        json.dumps(asdict(trajectory), sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest()
                    if trajectory
                    else None,
                }
            )
        eligible = [r["rollout_id"] for r in rows if r["eligible"]]
        selected = eligible if len(eligible) >= 2 else []
        complete = len(eligible) == planned
        for row in rows:
            row["selected"] = row["rollout_id"] in selected
        groups.append(
            {
                "task_id": task_id,
                "planned_rollout_count": planned,
                "eligible_rollout_count": len(eligible),
                "selected_rollout_count": len(selected),
                "selected_rollout_ids": selected,
                "planned_group_complete": complete,
                "solver_group_eligible": bool(selected),
                "solver_skip_reason": None if selected else "insufficient_eligible_siblings",
                "proposer_exclusion": None if complete else "partial_solver_group",
                "rollouts": rows,
            }
        )
    result = {
        "schema_version": schema,
        "rollout_group_policy": "eligible_subset",
        "minimum_eligible_rollouts_per_task": 2,
        "groups": groups,
        "selected_rollout_ids": [rid for g in groups for rid in g["selected_rollout_ids"]],
    }
    if schema == INDEPENDENT_SCHEMA:
        from .proposer_learning import extend_selection

        extend_selection(result, proposals, rollouts)
    return result


def verify_frozen_selection(directory):
    """Check the durable transaction before loading it for an optimizer update."""
    path = directory / "training_selection.json"
    if not path.exists():
        return None
    selection = json.loads(path.read_text())
    if selection.get("schema_version") not in SELECTION_SCHEMAS or not selection.get(
        "postcollection_complete"
    ):
        raise ValueError("training selection has not committed a complete batch transaction")
    required = {
        "solver_batch.json",
        "proposer_batch.json",
        "frontier_scores.json",
        "snapshots.json",
        "training_relation_credits.jsonl",
    }
    hashes = selection.get("artifacts_sha256", {})
    if not required <= hashes.keys():
        raise ValueError("training selection lacks required artifact hashes")
    for name, digest in {**hashes, **selection.get("source_artifacts_sha256", {})}.items():
        # Manifest filenames are local artifact basenames, never arbitrary paths.
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("invalid selection artifact filename")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen training artifact changed: " + name)
    return selection
