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
    skill_context: dict[str, Any] | None = None

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
        if self.skill_context is not None:
            _validate_pats_skill_context(self.skill_context)
            if (
                self.collection_mode == "async_one_step_stale"
                and self.skill_context["pats_step"] != self.target_cycle
            ):
                raise ValueError(
                    "async PATS collection must use the review committed for its target cycle"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        fields = asdict(self)
        if self.skill_context is None:
            fields.pop("skill_context")
        return {
            "schema_version": (
                "async_policy_lineage_v2"
                if self.skill_context is not None
                else "async_policy_lineage_v1"
            ),
            **fields,
            "staleness_updates": self.staleness_updates,
        }


def _validate_pats_skill_context(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "snapshot_id",
        "snapshot_sha256",
        "pats_snapshot_id",
        "pats_step",
        "contract_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != "pats_lineage_v1":
        raise ValueError("invalid PATS skill-context lineage")
    for key in required - {"pats_step"}:
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError("PATS skill-context lineage requires immutable identifiers")
    if not isinstance(payload.get("pats_step"), int) or payload["pats_step"] < -1:
        raise ValueError("PATS skill-context lineage has an invalid committed step")


def pats_skill_context_lineage(output_dir: Path) -> dict[str, Any] | None:
    """Return the immutable PATS view used by one collection cycle."""

    snapshot_path = output_dir / "director_skill_snapshot.v2.json"
    contract_path = output_dir / "skill_context_contract.json"
    if not snapshot_path.exists() and not contract_path.exists():
        return None
    if not contract_path.exists():
        raise ValueError("PATS collection snapshot lacks its context contract")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("pats_enabled") is not True or contract.get("context_enabled") is not True:
        return None
    if not snapshot_path.exists():
        raise ValueError("enabled PATS collection lacks its frozen snapshot")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    pats = snapshot.get("pats")
    if pats is None:
        raise ValueError("PATS snapshot differs from the collection context contract")
    payload = {
        "schema_version": "pats_lineage_v1",
        "snapshot_id": str(snapshot.get("snapshot_id", "")),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "pats_snapshot_id": str(pats.get("snapshot_id", "")),
        "pats_step": pats.get("step"),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    _validate_pats_skill_context(payload)
    return payload


def _lineage_from_payload(payload: dict[str, Any]) -> AsyncPolicyLineage:
    schema = payload.get("schema_version")
    if schema not in {"async_policy_lineage_v1", "async_policy_lineage_v2"}:
        raise ValueError("unknown async policy lineage schema")
    skill_context = payload.get("skill_context")
    if schema == "async_policy_lineage_v2" and not isinstance(skill_context, dict):
        raise ValueError("PATS-aware policy lineage requires a skill-context binding")
    if schema == "async_policy_lineage_v1" and skill_context is not None:
        raise ValueError("v1 policy lineage cannot contain a skill-context binding")
    lineage = AsyncPolicyLineage(
        target_cycle=int(payload["target_cycle"]),
        behavior_update_index=int(payload["behavior_update_index"]),
        proposer_snapshot=str(payload["proposer_snapshot"]),
        solver_snapshot=str(payload["solver_snapshot"]),
        collection_mode=str(payload["collection_mode"]),
        max_staleness_updates=int(payload["max_staleness_updates"]),
        skill_context=skill_context,
    )
    lineage.validate()
    if lineage.to_dict() != payload:
        raise ValueError("policy lineage contains unsupported or inconsistent fields")
    return lineage


def validate_async_queue_skill_context(
    state: dict[str, Any],
    solver_batch: TrainingBatch,
    *,
    pats_enabled: bool,
    require_persisted: bool = False,
) -> dict[str, Any] | None:
    """Keep the durable async queue bound to the collected PATS view."""

    lineage = solver_batch.metadata.get("policy_lineage", {})
    skill_context = lineage.get("skill_context") if isinstance(lineage, dict) else None
    if pats_enabled and skill_context is None:
        raise ValueError("PATS async collection lacks a frozen skill-context lineage")
    if "skill_context" in state:
        if state["skill_context"] != skill_context:
            raise ValueError("async queue has a different PATS snapshot")
    elif pats_enabled and require_persisted:
        raise ValueError("durable async queue lacks its PATS snapshot")
    return skill_context


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
    skill_context = pats_skill_context_lineage(output_dir)
    if lineage_path.exists():
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
        saved = _lineage_from_payload(payload)
        if (
            saved.target_cycle != target_cycle
            or saved.proposer_snapshot != str(result.snapshots.get("proposer", ""))
            or saved.solver_snapshot != str(result.snapshots.get("solver", ""))
            or saved.skill_context != skill_context
        ):
            raise ValueError("saved policy/PATS lineage differs from durable rollout snapshots")
    else:
        lineage = AsyncPolicyLineage(
            target_cycle=target_cycle,
            behavior_update_index=behavior_update_index,
            proposer_snapshot=str(result.snapshots.get("proposer", "")),
            solver_snapshot=str(result.snapshots.get("solver", "")),
            collection_mode=collection_mode,
            skill_context=skill_context,
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
    if payload.get("schema_version") not in {
        "async_policy_lineage_v1",
        "async_policy_lineage_v2",
    } or any(
        payload.get(role + "_snapshot") != snapshots.get(role) for role in ("proposer", "solver")
    ):
        return False
    try:
        lineage = _lineage_from_payload(payload)
        if lineage.skill_context != pats_skill_context_lineage(output_dir):
            return False
    except (KeyError, TypeError, ValueError):
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
    expected_skill_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one bounded-stale batch immediately before optimizer creation."""

    proposer_lineage = proposer_batch.metadata.get("policy_lineage")
    solver_lineage = solver_batch.metadata.get("policy_lineage")
    if proposer_lineage is None and solver_lineage is None:
        if expected_skill_context is not None:
            raise ValueError("PATS-aware training batch lacks skill-context lineage")
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
    lineage = _lineage_from_payload(solver_lineage)
    if lineage.skill_context != expected_skill_context:
        raise ValueError("rollout PATS lineage does not match its durable collection snapshot")
    target_cycle = lineage.target_cycle
    behavior_index = lineage.behavior_update_index
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
        lineage.proposer_snapshot != learner_proposer_snapshot
        or lineage.solver_snapshot != learner_solver_snapshot
    ):
        raise ValueError("synchronous rollout snapshots do not match learner snapshots")
    binding = {
        "schema_version": "async_learner_binding_v1",
        "mode": "bounded_stale_ppo" if staleness else "synchronous_ppo",
        "learner_update_index": learner_update_index,
        "behavior_update_index": behavior_index,
        "staleness_updates": staleness,
        "behavior_proposer_snapshot": lineage.proposer_snapshot,
        "behavior_solver_snapshot": lineage.solver_snapshot,
        "learner_proposer_snapshot": learner_proposer_snapshot,
        "learner_solver_snapshot": learner_solver_snapshot,
    }
    if lineage.skill_context is not None:
        binding["skill_context"] = lineage.skill_context
    return binding


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
