"""Independent Frontier evidence and frozen, dataset-level Proposer baselines.

No model calls live here. Solver rewards (including HealthBench length adjustment)
are consumed unchanged. Old batches continue through the legacy selection schema.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .config import canonical_dataset_name

NORMALIZATION = "pair_mean_ref5_v1"
BASELINE_VERSION = "dataset_ema_shared_solver_reward_v1"
NO_BASELINE_VERSION = "frontier_no_baseline_v1"
REFERENCE_SCALE = 1.6
FRONTIER_STABILITY_VERSION = "trusted_reverify_tie_discount_v1"
DEFAULT_FRONTIER_TIE_WEIGHT = 0.1


def frontier_stability_policy(snapshot):
    """Old frozen batches keep the hard gate; new batches record the soft rule."""
    policy = snapshot.get("frontier_stability")
    if policy is None:
        return {"version": "sign_consistency_v1", "tie_weight": 0.0}
    if (
        not isinstance(policy, dict)
        or policy.get("version") != FRONTIER_STABILITY_VERSION
        or not isinstance(policy.get("tie_weight"), (int, float))
        or not math.isfinite(policy["tie_weight"])
        or not 0 <= policy["tie_weight"] <= 1
    ):
        raise ValueError("invalid frozen Frontier stability policy")
    return dict(policy)


def validate_baseline_snapshot(snapshot):
    if (
        snapshot.get("version") not in {BASELINE_VERSION, NO_BASELINE_VERSION}
        or snapshot.get("normalization") != NORMALIZATION
    ):
        raise ValueError("missing or incompatible frozen Proposer baseline")
    if snapshot["version"] == NO_BASELINE_VERSION and snapshot.get("values") != {}:
        raise ValueError("disabled baseline must have empty values")
    frontier_stability_policy(snapshot)


def _scoped_ema_path(state_path, policy):
    """Keep the current reward rule's history separate from old hard-gate EMA."""
    policy = frontier_stability_policy({"frontier_stability": policy})
    path = Path(state_path)
    weight = str(float(policy["tie_weight"])).replace(".", "p")
    return path.with_name(f"{path.stem}.{policy['version']}.alpha-{weight}{path.suffix}")


def _snapshot_ema_path(state_path, snapshot):
    filename = snapshot.get("ema_state_file")
    if filename is None:
        return Path(state_path)  # Pre-binding snapshots retain their original location.
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in {"", ".", ".."}
    ):
        raise ValueError("invalid frozen EMA state filename")
    return Path(state_path).with_name(filename)


def freeze_proposer_baseline(snapshot_path, *, state_path, cycle, decay, mode):
    """Freeze the requested baseline; persisted batches keep their semantics."""
    if mode not in {"none", "ema"}:
        raise ValueError("unknown Proposer baseline mode")
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        validate_baseline_snapshot(snapshot)
        if snapshot.get("cycle") != cycle:
            raise ValueError("frozen baseline cycle mismatch")
        if snapshot["version"] == NO_BASELINE_VERSION:
            return snapshot, None
        store = DatasetBaselineStore(
            _snapshot_ema_path(state_path, snapshot), decay=snapshot["decay"]
        )
        return store.freeze(snapshot_path, cycle=cycle), store
    if mode == "ema":
        policy = {"version": FRONTIER_STABILITY_VERSION, "tie_weight": DEFAULT_FRONTIER_TIE_WEIGHT}
        store = DatasetBaselineStore(_scoped_ema_path(state_path, policy), decay=decay)
        return store.freeze(snapshot_path, cycle=cycle, stability_policy=policy), store
    snapshot = dict(
        version=NO_BASELINE_VERSION,
        normalization=NORMALIZATION,
        cycle=cycle,
        values={},
        mode="none",
        frontier_stability={
            "version": FRONTIER_STABILITY_VERSION,
            "tie_weight": DEFAULT_FRONTIER_TIE_WEIGHT,
        },
    )
    _atomic_json(snapshot_path, snapshot)
    return snapshot, None


def frontier_multiplier(n: int, normalization: str) -> float:
    if normalization not in {"legacy", NORMALIZATION}:
        raise ValueError("unknown Frontier normalization")
    if n < 2:
        return 0.0
    if normalization == "legacy" or n == 5:
        return 4.0 / (n * n)
    return REFERENCE_SCALE / math.comb(n, 2)


def policy_record_exclusions(record) -> list[str]:
    reasons = []
    if record.metadata.get("pool_id") is not None and not record.metadata.get(
        "validated_pool_entry", False
    ):
        reasons.append("unvalidated_pool_entry")
    if record.metadata.get("training_eligible") is False:
        reasons.append("proposer_policy_ineligible")
    if not record.policy_calls:
        return reasons + ["missing_proposer_policy_calls"]
    has_targets = False
    for call in record.policy_calls:
        n = len(call.token_ids)
        if (
            n < 2
            or len(call.action_mask) != n
            or any(value not in (0, 1) for value in call.action_mask)
        ):
            reasons.append("invalid_proposer_tokens_or_mask")
        else:
            has_targets |= any(call.action_mask[1:])
        if len(call.behavior_log_probs) != n - 1 or not all(
            math.isfinite(float(x)) for x in call.behavior_log_probs
        ):
            reasons.append("invalid_proposer_behavior_probabilities")
    if not has_targets:
        reasons.append("missing_proposer_action_targets")
    return sorted(set(reasons))


def frontier_evidence_exclusions(rollout) -> list[str]:
    """Policy token/probability failures do not invalidate independently known outcomes."""
    if rollout is None:
        return ["missing_trajectory"]
    trajectory = rollout.trajectory
    metadata = trajectory.metadata
    reasons = []
    reward = float(trajectory.reward)
    if metadata.get("reward_known") is not True or not math.isfinite(reward):
        reasons.append("untrusted_reward")
    if (
        metadata.get("uncertain_attribution_zero")
        or metadata.get("reward_admission_reason") == "uncertain_attribution_zero"
    ):
        reasons.append("statistical_zero_without_trusted_outcome")
    if any(
        metadata.get(key)
        for key in (
            "worker_backend_failure",
            "infrastructure_failure",
            "swe_infrastructure_failure",
            "swe_synthetic_evaluation",
            "swe_non_train_split",
        )
    ):
        reasons.append("infrastructure_or_invalid_evaluation")
    if not 0.0 <= reward <= 1.0:
        reasons.append("invalid_task_reward")
    if metadata.get("task_reward", reward) != reward:
        reasons.append("task_reward_mismatch")
    if metadata.get("terminal_graph_status") == "unsafe_partial":
        reasons.append("unsafe_graph_record")
    # Graph.from_dict rebuilds the mutation counter. Compare actual contents,
    # not that counter, while retaining node prompts, routes and all relations.
    stored_graph = dict(trajectory.graph)
    actual_graph = rollout.graph.to_dict()
    stored_graph.pop("version", None)
    actual_graph.pop("version", None)
    if (
        stored_graph != actual_graph
        or rollout.graph.validate()
        or metadata.get("frontier_evidence_eligible") is False
    ):
        reasons.append("invalid_frontier_evidence")
    return sorted(set(reasons))


def dataset_key(proposal) -> str:
    return canonical_dataset_name(proposal.task.metadata.get("dataset", "")) or "unknown"


def extend_selection(selection, proposals, rollouts) -> None:
    if any(
        p.metadata.get("pool_id") is not None and not p.metadata.get("validated_pool_entry", False)
        for p in proposals
    ):
        raise ValueError("fixed-pool proposal lacks a validated task-pool attestation")
    proposal_by_id = {p.task.task_id: p for p in proposals}
    for group in selection["groups"]:
        evidence_ids = []
        for row in group["rollouts"]:
            reasons = frontier_evidence_exclusions(rollouts.get(row["rollout_id"]))
            row["frontier_exclusion_reasons"] = reasons
            row["frontier_eligible"] = not reasons
            if not reasons:
                evidence_ids.append(row["rollout_id"])
        # Preserve singleton evidence for audit, but it does not define a Frontier.
        eligible_evidence_count = len(evidence_ids)
        evidence_ids = evidence_ids if len(evidence_ids) >= 2 else []
        proposal = proposal_by_id[group["task_id"]]
        policy_reasons = policy_record_exclusions(proposal)
        group.update(
            dataset=dataset_key(proposal),
            frontier_eligible_rollout_count=eligible_evidence_count,
            frontier_rollout_ids=evidence_ids,
            frontier_rollout_count=len(evidence_ids),
            frontier_pair_count=math.comb(len(evidence_ids), 2),
            proposer_policy_exclusions=policy_reasons,
            proposer_exclusion=(
                "invalid_proposer_policy_record"
                if policy_reasons
                else "insufficient_frontier_evidence"
                if not evidence_ids
                else None
            ),
        )
    selection["frontier_normalization"] = NORMALIZATION
    selection["frontier_reward_source"] = "solver_task_reward"
    selection["frontier_rollout_ids"] = [
        rid for group in selection["groups"] for rid in group["frontier_rollout_ids"]
    ]


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


class DatasetBaselineStore:
    """Freeze before selection; publish completed cycles once, in cycle order.

    Concurrent collection can freeze an older baseline intentionally. Finishing
    that collection never changes its frozen advantages. Pending future cycles
    wait for earlier registered cycles before updating the shared EMA.
    """

    def __init__(self, path: Path, *, decay: float = 0.9):
        if not 0 <= decay < 1:
            raise ValueError("baseline decay must be in [0, 1)")
        self.path = Path(path)
        self.decay = decay

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _state(self, first_cycle: int, stability_policy=None):
        expected = frontier_stability_policy({"frontier_stability": stability_policy})
        if not self.path.exists():
            state = dict(
                version=BASELINE_VERSION,
                normalization=NORMALIZATION,
                decay=self.decay,
                next_cycle=first_cycle,
                values={},
                pending={},
                applied={},
            )
            if stability_policy is not None:
                state["frontier_stability"] = dict(stability_policy)
            return state
        state = json.loads(self.path.read_text())
        if (
            state.get("version") != BASELINE_VERSION
            or state.get("decay") != self.decay
            or state.get("normalization") != NORMALIZATION
        ):
            raise ValueError("baseline version/decay changed; rebuild or start a fresh baseline")
        if frontier_stability_policy(state) != expected:
            raise ValueError(
                "EMA Frontier stability policy mismatch; use a separate matching state"
            )
        return state

    def freeze(self, snapshot_path: Path, *, cycle: int, stability_policy=None):
        with self._locked():
            if snapshot_path.exists():
                snapshot = json.loads(snapshot_path.read_text())
                validate_baseline_snapshot(snapshot)
                if (
                    snapshot.get("version") != BASELINE_VERSION
                    or snapshot.get("cycle") != cycle
                    or snapshot.get("decay") != self.decay
                ):
                    raise ValueError("frozen Proposer baseline does not match this collection")
                if stability_policy is not None and frontier_stability_policy(
                    snapshot
                ) != frontier_stability_policy({"frontier_stability": stability_policy}):
                    raise ValueError("frozen Frontier stability policy mismatch")
                self._state(cycle, snapshot.get("frontier_stability"))
                return snapshot
            state = self._state(cycle, stability_policy)
            if cycle < state["next_cycle"]:
                raise ValueError("cannot refreeze an already committed baseline cycle")
            snapshot = dict(
                version=BASELINE_VERSION,
                normalization=NORMALIZATION,
                cycle=cycle,
                decay=self.decay,
                values=dict(state["values"]),
                ema_state_file=self.path.name,
            )
            if stability_policy is not None:
                snapshot["frontier_stability"] = dict(stability_policy)
                validate_baseline_snapshot(snapshot)
            _atomic_json(self.path, state)
            _atomic_json(snapshot_path, snapshot)
            return snapshot

    def commit(self, snapshot, means: dict[str, float]) -> None:
        validate_baseline_snapshot(snapshot)
        if (
            snapshot.get("version") != BASELINE_VERSION
            or snapshot.get("normalization") != NORMALIZATION
            or snapshot.get("decay") != self.decay
        ):
            raise ValueError("cannot commit an incompatible baseline snapshot")
        if not all(math.isfinite(value) and value >= 0 for value in means.values()):
            raise ValueError("baseline requires finite, trusted Frontier means")
        cycle = int(snapshot["cycle"])
        key = str(cycle)
        with self._locked():
            state = self._state(cycle, snapshot.get("frontier_stability"))
            old = state["applied"].get(key, state["pending"].get(key))
            if old is not None:
                if old != means:
                    raise ValueError("cannot change rewards of a committed baseline cycle")
                return
            if cycle < state["next_cycle"]:
                raise ValueError("baseline cycle was not registered")
            state["pending"][key] = means
            while str(state["next_cycle"]) in state["pending"]:
                current = str(state["next_cycle"])
                values = state["pending"].pop(current)
                for dataset, value in values.items():
                    previous = state["values"].get(dataset, 0.0)
                    state["values"][dataset] = self.decay * previous + (1 - self.decay) * value
                state["applied"][current] = values
                state["next_cycle"] += 1
            _atomic_json(self.path, state)


def assemble_independent_result(
    proposals,
    solver_rollouts,
    evidence_rollouts,
    snapshots,
    *,
    selection,
    baseline,
    graph_feature_extractor=None,
    frontier_reverification=None,
):
    from .features import execution_policy_features_many, graph_kernel_matrix
    from .graph_learning import build_graph_training_batch
    from .rollouts import TrainingBatch, TrainingSample
    from .selfplay import (
        DryRunSelfPlayResult,
        FrontierScore,
        graph_local_frontier,
        group_rollouts_by_task,
        scalar_frontier,
        stable_graph_local_frontier,
    )
    from .training_selection import trajectory_exclusions

    if [r.trajectory.rollout_id for r in solver_rollouts] != selection["selected_rollout_ids"]:
        raise ValueError("Solver membership differs from frozen selection")
    if [r.trajectory.rollout_id for r in evidence_rollouts] != selection["frontier_rollout_ids"]:
        raise ValueError("Frontier membership differs from frozen selection")
    if any(trajectory_exclusions(r.trajectory) for r in solver_rollouts):
        raise ValueError("Solver admission failed")
    if any(
        r.trajectory.metadata.get("task_reward", r.trajectory.reward) != r.trajectory.reward
        for r in solver_rollouts
    ):
        raise ValueError("trajectory reward differs from the audited task_reward field")
    if any(frontier_evidence_exclusions(r) for r in evidence_rollouts):
        raise ValueError("Frontier evidence admission failed")
    validate_baseline_snapshot(baseline)
    stability_policy = frontier_stability_policy(baseline)
    manifest = {g["task_id"]: g for g in selection["groups"]}
    if {p.task.task_id for p in proposals} != set(manifest):
        raise ValueError("Proposer task identities differ from frozen selection")

    def features(group):
        graphs = [r.graph for r in group]
        return (
            graph_feature_extractor.extract_many(graphs)
            if graph_feature_extractor
            else execution_policy_features_many(graphs)
        )

    # Solver kernels depend ONLY on Solver-selected siblings, as before.
    solver_features, solver_kernels = {}, {}
    for task_id, group in group_rollouts_by_task(solver_rollouts).items():
        vectors = features(group)
        solver_features.update(zip((r.trajectory.rollout_id for r in group), vectors, strict=True))
        solver_kernels[task_id] = graph_kernel_matrix(vectors)
    trajectories = [
        replace(
            r.trajectory,
            metadata={
                **r.trajectory.metadata,
                "training_selection_schema": selection["schema_version"],
                "selected_group_rollout_ids": manifest[r.trajectory.task_id][
                    "selected_rollout_ids"
                ],
                "selected_group_size": manifest[r.trajectory.task_id]["selected_rollout_count"],
                "rollout_group_complete": manifest[r.trajectory.task_id]["planned_group_complete"],
                "rollout_group_size_expected": manifest[r.trajectory.task_id][
                    "planned_rollout_count"
                ],
            },
        )
        for r in solver_rollouts
    ]
    solver_batch = build_graph_training_batch(
        trajectories, features_by_rollout=solver_features, kernels_by_task=solver_kernels
    )
    groups = group_rollouts_by_task(evidence_rollouts)
    frontiers, samples, means = [], [], defaultdict(list)
    final_exclusions = {}
    for proposal in proposals:
        task_id = proposal.task.task_id
        entry = manifest[task_id]
        group = groups.get(task_id, [])
        rewards = tuple(float(r.trajectory.reward) for r in group)
        vectors = features(group) if group else []
        kernel = graph_kernel_matrix(vectors)
        provisional = graph_local_frontier(
            rewards, vectors, kernel_matrix=kernel, normalization=NORMALIZATION
        )
        reverify = (frontier_reverification or {}).get(task_id)
        stable, pairs = None, ()
        exclusion = proposal.metadata.get("frontier_training_exclusion")
        if reverify is not None:
            if reverify.get("rollout_ids") != entry["frontier_rollout_ids"]:
                raise ValueError("Frontier reverify membership/order differs from primary evidence")
            other_rewards = tuple(float(v) for v in reverify["rewards"])
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in other_rewards):
                raise ValueError("untrusted Frontier reverify rewards")
            stable, pairs = stable_graph_local_frontier(
                rewards,
                other_rewards,
                vectors,
                kernel_matrix=kernel,
                normalization=NORMALIZATION,
                tie_weight=stability_policy["tie_weight"],
                reverify_trusted=reverify.get("reward_trusted", [False] * len(other_rewards)),
            )
        score = stable if stable is not None else provisional
        evidence_exclusion = exclusion or ("insufficient_frontier_evidence" if not group else None)
        final_exclusion = evidence_exclusion or entry["proposer_exclusion"]
        final_exclusions[task_id] = final_exclusion
        status = (
            "excluded_" + evidence_exclusion
            if evidence_exclusion
            else "completed"
            if reverify is not None
            else "not_triggered"
        )
        info = dict(
            normalization=NORMALIZATION,
            reward_source="solver_task_reward",
            rollout_ids=entry["frontier_rollout_ids"],
            actual_count=len(group),
            pair_count=entry["frontier_pair_count"],
            proposer_exclusion=final_exclusion,
            stability_policy=stability_policy,
        )
        frontiers.append(
            FrontierScore(
                task_id,
                1.0,
                scalar_frontier(rewards, normalization=NORMALIZATION),
                score,
                rewards,
                provisional,
                stable,
                status,
                pairs,
                info,
            )
        )
        dataset = entry["dataset"]
        if not evidence_exclusion:
            means[dataset].append(score)
        if final_exclusion:
            continue
        if policy_record_exclusions(proposal):
            raise ValueError("Proposer policy admission changed after freeze")
        value = float(baseline["values"].get(dataset, 0.0))
        samples.append(
            TrainingSample(
                rollout_id=f"proposal-{task_id}",
                task_id=task_id,
                token_ids=proposal.token_ids,
                action_mask=proposal.action_mask,
                reward=score,
                advantage=score - value,
                policy_calls=proposal.policy_calls,
                metadata={
                    **proposal.metadata,
                    "response": proposal.response,
                    "dataset": dataset,
                    "frontier": info,
                    "proposer_baseline": value,
                    "proposer_baseline_cycle": baseline["cycle"],
                    "proposer_advantage_version": baseline["version"],
                },
            )
        )
    metadata = dict(
        training_selection_schema=selection["schema_version"],
        rollout_group_policy="eligible_subset",
        groups=selection["groups"],
        selected_rollout_ids=selection["selected_rollout_ids"],
        frontier_rollout_ids=selection["frontier_rollout_ids"],
        frontier_normalization=NORMALIZATION,
        proposer_baseline=baseline,
        proposer_exclusions=final_exclusions,
        proposer_task_ids=[s.task_id for s in samples],
        frontier_dataset_means={key: sum(values) / len(values) for key, values in means.items()},
    )
    proposer_batch = TrainingBatch(
        role="proposer", samples=tuple(samples), objective="graph_local_frontier", metadata=metadata
    )
    solver_batch = replace(solver_batch, metadata=metadata)
    snapshots.advance()
    return DryRunSelfPlayResult(
        tasks=tuple(p.task for p in proposals),
        frontier_scores=tuple(frontiers),
        proposer_batch=proposer_batch,
        solver_batch=solver_batch,
        snapshots={
            "proposer": snapshots.proposer_snapshot,
            "solver": snapshots.solver_snapshot,
            "phase": snapshots.phase.value,
            "cycle": snapshots.cycle,
        },
    )


def validate_independent_batches(proposer_batch, solver_batch) -> None:
    from .training_selection import INDEPENDENT_SCHEMA, trajectory_exclusions

    manifest = solver_batch.metadata
    if (
        proposer_batch.metadata != manifest
        or manifest.get("training_selection_schema") != INDEPENDENT_SCHEMA
    ):
        raise ValueError("role batches have different independent selections")
    if manifest.get("frontier_normalization") != NORMALIZATION:
        raise ValueError("incompatible Frontier normalization")
    baseline = manifest.get("proposer_baseline", {})
    validate_baseline_snapshot(baseline)
    groups = {g["task_id"]: g for g in manifest["groups"]}
    if len(groups) != len(manifest["groups"]):
        raise ValueError("duplicate task groups")
    selected = manifest["selected_rollout_ids"]
    if (
        len(selected) != len(set(selected))
        or [s.rollout_id for s in solver_batch.samples] != selected
        or selected != [rid for g in groups.values() for rid in g["selected_rollout_ids"]]
    ):
        raise ValueError("Solver samples differ from selected trajectory IDs")
    for sample in solver_batch.samples:
        group = groups.get(sample.task_id)
        if group is None or trajectory_exclusions(sample):
            raise ValueError("selected Solver sample fails admission")
        if (
            sample.rollout_id not in group["selected_rollout_ids"]
            or sample.metadata.get("selected_group_rollout_ids") != group["selected_rollout_ids"]
            or sample.metadata.get("selected_group_size") != group["selected_rollout_count"]
            or sample.metadata.get("rollout_group_size_expected") != group["planned_rollout_count"]
            or sample.metadata.get("rollout_group_complete") != group["planned_group_complete"]
        ):
            raise ValueError("Solver sibling metadata differs from selection")
    evidence_ids = []
    expected_proposer = []
    exclusions = manifest["proposer_exclusions"]
    for task_id, group in groups.items():
        n = len(group["selected_rollout_ids"])
        count = group["planned_rollout_count"]
        ids = group["frontier_rollout_ids"]
        if n != group["selected_rollout_count"] or (n and not 2 <= n <= count):
            raise ValueError("invalid Solver group size")
        rows = {row["rollout_id"]: row for row in group["rollouts"]}
        all_solver_ids = [rid for rid, row in rows.items() if row["eligible"]]
        all_frontier_ids = [rid for rid, row in rows.items() if row["frontier_eligible"]]
        if (
            len(rows) != count
            or group["eligible_rollout_count"] != len(all_solver_ids)
            or group["frontier_eligible_rollout_count"] != len(all_frontier_ids)
            or group["selected_rollout_ids"] != (all_solver_ids if len(all_solver_ids) >= 2 else [])
            or ids != (all_frontier_ids if len(all_frontier_ids) >= 2 else [])
        ):
            raise ValueError("independent selection must retain every qualified sibling")
        if any(
            rid not in rows or not rows[rid]["eligible"] for rid in group["selected_rollout_ids"]
        ):
            raise ValueError("Solver selection contains excluded evidence")
        if (
            len(ids) != len(set(ids))
            or len(ids) != group["frontier_rollout_count"]
            or group["frontier_pair_count"] != math.comb(len(ids), 2)
            or (ids and not 2 <= len(ids) <= count)
            or any(rid not in rows or not rows[rid]["frontier_eligible"] for rid in ids)
        ):
            raise ValueError("invalid Frontier evidence membership")
        evidence_ids.extend(ids)
        if task_id not in exclusions:
            raise ValueError("missing Proposer admission outcome")
        if exclusions[task_id] is None:
            if not ids or group["proposer_exclusion"] is not None:
                raise ValueError("Proposer requires evidence and its own policy record")
            expected_proposer.append(task_id)
    if evidence_ids != manifest["frontier_rollout_ids"]:
        raise ValueError("Frontier evidence differs from selection")
    if [s.task_id for s in proposer_batch.samples] != expected_proposer or manifest[
        "proposer_task_ids"
    ] != expected_proposer:
        raise ValueError("Proposer samples differ from independent admission")
    for sample in proposer_batch.samples:
        group = groups[sample.task_id]
        value = float(baseline["values"].get(group["dataset"], 0.0))
        frontier = sample.metadata.get("frontier", {})
        if "frontier_stability" in baseline and frontier.get(
            "stability_policy"
        ) != frontier_stability_policy(baseline):
            raise ValueError("Proposer Frontier stability policy differs from frozen collection")
        if (
            policy_record_exclusions(sample)
            or not math.isfinite(value)
            or value < 0
            or not math.isfinite(sample.reward)
            or not 0 <= sample.reward <= REFERENCE_SCALE + 1e-12
            or not math.isclose(sample.advantage, sample.reward - value, rel_tol=0, abs_tol=1e-12)
            or sample.metadata.get("proposer_baseline") != value
            or sample.metadata.get("proposer_baseline_cycle") != baseline["cycle"]
            or sample.metadata.get("proposer_advantage_version") != baseline["version"]
            or frontier.get("rollout_ids") != group["frontier_rollout_ids"]
            or frontier.get("pair_count") != group["frontier_pair_count"]
            or frontier.get("normalization") != NORMALIZATION
            or frontier.get("reward_source") != "solver_task_reward"
        ):
            raise ValueError("Proposer policy, reward or frozen advantage is inconsistent")
