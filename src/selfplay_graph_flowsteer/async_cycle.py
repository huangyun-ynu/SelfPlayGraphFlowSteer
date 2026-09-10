from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .rollouts import TrainingBatch
from .selfplay import DryRunSelfPlayResult


@dataclass(frozen=True)
class AsyncPolicyLineage:
    """Immutable policy versions for one collected cycle.

    ``behavior_update_index`` is the number of committed updates visible to the
    rollout service.  ``target_cycle`` is the optimizer cycle that will consume
    the batch.  Their difference is the only supported staleness definition.
    """

    target_cycle: int
    behavior_update_index: int
    proposer_snapshot: str
    solver_snapshot: str
    collection_mode: str
    max_staleness_updates: int = 1

    @property
    def staleness_updates(self) -> int:
        return self.target_cycle - self.behavior_update_index

    def validate(self) -> None:
        if min(self.target_cycle, self.behavior_update_index) < 0:
            raise ValueError("async policy lineage indices must be non-negative")
        if self.max_staleness_updates != 1:
            raise ValueError("async cycle pipeline supports exactly one bounded stale update")
        if not 0 <= self.staleness_updates <= self.max_staleness_updates:
            raise ValueError(
                "rollout batch staleness must be zero or one update: "
                f"target={self.target_cycle}, behavior={self.behavior_update_index}"
            )
        if not self.proposer_snapshot or not self.solver_snapshot:
            raise ValueError("async policy lineage requires both behavior snapshots")
        if self.collection_mode not in {"synchronous", "async_one_step_stale"}:
            raise ValueError("unknown async collection mode")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "async_policy_lineage_v1",
            **asdict(self),
            "staleness_updates": self.staleness_updates,
        }


def bind_rollout_result_lineage(
    result: DryRunSelfPlayResult,
    *,
    output_dir: Path,
    target_cycle: int,
    behavior_update_index: int,
    collection_mode: str,
) -> DryRunSelfPlayResult:
    """Bind both role batches to the snapshots that actually generated them."""

    lineage_path = output_dir / "policy_lineage.json"
    if lineage_path.exists():
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "async_policy_lineage_v1"
            or int(payload.get("target_cycle", -1)) != target_cycle
            or str(payload.get("proposer_snapshot", ""))
            != str(result.snapshots.get("proposer", ""))
            or str(payload.get("solver_snapshot", "")) != str(result.snapshots.get("solver", ""))
        ):
            raise ValueError("saved policy lineage differs from durable rollout snapshots")
        AsyncPolicyLineage(
            target_cycle=target_cycle,
            behavior_update_index=int(payload["behavior_update_index"]),
            proposer_snapshot=str(payload["proposer_snapshot"]),
            solver_snapshot=str(payload["solver_snapshot"]),
            collection_mode=str(payload["collection_mode"]),
            max_staleness_updates=int(payload["max_staleness_updates"]),
        ).validate()
    else:
        lineage = AsyncPolicyLineage(
            target_cycle=target_cycle,
            behavior_update_index=behavior_update_index,
            proposer_snapshot=str(result.snapshots.get("proposer", "")),
            solver_snapshot=str(result.snapshots.get("solver", "")),
            collection_mode=collection_mode,
        )
        payload = lineage.to_dict()
    proposer_batch = replace(
        result.proposer_batch,
        metadata={**result.proposer_batch.metadata, "policy_lineage": payload},
    )
    solver_batch = replace(
        result.solver_batch,
        metadata={**result.solver_batch.metadata, "policy_lineage": payload},
    )
    bound = replace(result, proposer_batch=proposer_batch, solver_batch=solver_batch)
    _atomic_json(lineage_path, payload)
    _atomic_json(output_dir / "proposer_batch.json", proposer_batch.to_dict())
    _atomic_json(output_dir / "solver_batch.json", solver_batch.to_dict())
    selection_path = output_dir / "training_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("postcollection_complete"):
            hashes = selection.get("artifacts_sha256", {})
            for name in ("proposer_batch.json", "solver_batch.json"):
                if name not in hashes:
                    raise ValueError("training selection lacks a bound batch artifact")
                hashes[name] = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            selection["artifacts_sha256"] = hashes
            _atomic_json(selection_path, selection)
    return bound


def recover_interrupted_lineage_binding(output_dir: Path) -> bool:
    """Finish only an attested metadata-only partial lineage write.

    A crash can occur between the two batch replacements and selection commit.
    Never rebind arbitrary changed data: removing the newly added lineage must
    reproduce the exact previously frozen bytes for every changed batch.
    """
    selection_path = output_dir / "training_selection.json"
    lineage_path = output_dir / "policy_lineage.json"
    if not selection_path.exists() or not lineage_path.exists():
        return False
    selection = json.loads(selection_path.read_text())
    if not selection.get("postcollection_complete"):
        return False
    names = ("proposer_batch.json", "solver_batch.json")
    expected = selection.get("artifacts_sha256", {})
    raw = {name: (output_dir / name).read_bytes() for name in names}
    changed = [
        name for name in names if hashlib.sha256(raw[name]).hexdigest() != expected.get(name)
    ]
    if not changed:
        return False
    payload = json.loads(lineage_path.read_text())
    snapshots = json.loads((output_dir / "snapshots.json").read_text())
    if payload.get("schema_version") != "async_policy_lineage_v1" or any(
        payload.get(role + "_snapshot") != snapshots.get(role) for role in ("proposer", "solver")
    ):
        return False
    lineage = AsyncPolicyLineage(
        **{
            key: payload[key]
            for key in (
                "target_cycle",
                "behavior_update_index",
                "proposer_snapshot",
                "solver_snapshot",
                "collection_mode",
                "max_staleness_updates",
            )
        }
    )
    lineage.validate()
    if lineage.to_dict() != payload:
        return False
    if output_dir.name.startswith("cycle-") and int(output_dir.name[6:]) != lineage.target_cycle:
        return False
    batches = {name: json.loads(raw[name]) for name in names}
    for name, batch in batches.items():
        metadata = batch.setdefault("metadata", {})
        existing = metadata.get("policy_lineage")
        if existing is not None and existing != payload:
            return False
        if name in changed:
            if existing != payload:
                return False
            metadata.pop("policy_lineage")
            original = (json.dumps(batch, ensure_ascii=False, indent=2) + "\n").encode()
            if hashlib.sha256(original).hexdigest() != expected.get(name):
                return False
        metadata["policy_lineage"] = payload
    for name, batch in batches.items():
        _atomic_json(output_dir / name, batch)
        selection["artifacts_sha256"][name] = hashlib.sha256(
            (output_dir / name).read_bytes()
        ).hexdigest()
    _atomic_json(selection_path, selection)
    _atomic_json(
        output_dir / "lineage_binding_recovery.json",
        {
            "changed_batches": changed,
            "training_content_changed": False,
            "recovery": "completed_verified_metadata_only_lineage_binding",
        },
    )
    return True


def validate_batch_lineage(
    proposer_batch: TrainingBatch,
    solver_batch: TrainingBatch,
    *,
    learner_update_index: int,
    learner_proposer_snapshot: str,
    learner_solver_snapshot: str,
) -> dict[str, Any]:
    """Validate one bounded-stale batch immediately before optimizer creation."""

    proposer_lineage = proposer_batch.metadata.get("policy_lineage")
    solver_lineage = solver_batch.metadata.get("policy_lineage")
    if proposer_lineage is None and solver_lineage is None:
        return {
            "schema_version": "async_learner_binding_v1",
            "mode": "legacy_synchronous",
            "learner_update_index": learner_update_index,
            "staleness_updates": 0,
            "learner_proposer_snapshot": learner_proposer_snapshot,
            "learner_solver_snapshot": learner_solver_snapshot,
        }
    if proposer_lineage != solver_lineage or not isinstance(solver_lineage, dict):
        raise ValueError("Proposer and Solver batches have different policy lineage")
    if solver_lineage.get("schema_version") != "async_policy_lineage_v1":
        raise ValueError("unknown async policy lineage schema")
    target_cycle = int(solver_lineage.get("target_cycle", -1))
    behavior_index = int(solver_lineage.get("behavior_update_index", -1))
    staleness = learner_update_index - behavior_index
    if target_cycle != learner_update_index:
        raise ValueError(
            f"rollout target cycle {target_cycle} does not match learner {learner_update_index}"
        )
    if staleness not in {0, 1}:
        raise ValueError(
            "rollout batch exceeds the one-update staleness bound: "
            f"learner={learner_update_index}, behavior={behavior_index}"
        )
    if staleness == 0 and (
        str(solver_lineage.get("proposer_snapshot")) != learner_proposer_snapshot
        or str(solver_lineage.get("solver_snapshot")) != learner_solver_snapshot
    ):
        raise ValueError("synchronous rollout snapshots do not match learner snapshots")
    return {
        "schema_version": "async_learner_binding_v1",
        "mode": "bounded_stale_ppo" if staleness else "synchronous_ppo",
        "learner_update_index": learner_update_index,
        "behavior_update_index": behavior_index,
        "staleness_updates": staleness,
        "behavior_proposer_snapshot": solver_lineage["proposer_snapshot"],
        "behavior_solver_snapshot": solver_lineage["solver_snapshot"],
        "learner_proposer_snapshot": learner_proposer_snapshot,
        "learner_solver_snapshot": learner_solver_snapshot,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
