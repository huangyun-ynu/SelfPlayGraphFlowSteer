import hashlib
import json
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.async_cycle import (
    AsyncPolicyLineage,
    bind_rollout_result_lineage,
    pats_skill_context_lineage,
    validate_async_queue_skill_context,
    validate_batch_lineage,
)
from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.selfplay import DryRunSelfPlayResult
from selfplay_graph_flowsteer.services import ModelServiceSpec
from selfplay_graph_flowsteer.training import (
    AlternatingGRPOTrainer,
    AlternatingTrainingConfig,
    MockPolicyTrainer,
    PolicyTrainingConfig,
    UnsafeTrainingBatchError,
    _initial_ratio_diagnostics,
)


def _batch(role: str, metadata=None) -> TrainingBatch:
    sample = TrainingSample("r0", "t0", (1, 2), (0, 1), 1.0, 0.0)
    return TrainingBatch(role, (sample,), metadata=metadata or {})


def _write_pats_context(path: Path, *, step: int = 1) -> dict:
    (path / "director_skill_snapshot.v2.json").write_text(
        json.dumps(
            {
                "schema": "director_skill_v2",
                "snapshot_id": "bank-view-1",
                "collection_frozen": True,
                "cards": [],
                "pats": {"snapshot_id": "pats-view-1", "step": step},
            },
            indent=2,
        )
        + "\n"
    )
    (path / "skill_context_contract.json").write_text(
        json.dumps(
            {
                "schema": "skill_context_contract_v1",
                "context_enabled": True,
                "pats_enabled": True,
            },
            indent=2,
        )
        + "\n"
    )
    lineage = pats_skill_context_lineage(path)
    assert lineage is not None
    return lineage


def test_pats_disabled_context_does_not_require_a_snapshot(tmp_path) -> None:
    (tmp_path / "skill_context_contract.json").write_text(
        json.dumps(
            {
                "schema": "skill_context_contract_v1",
                "context_enabled": False,
                "pats_enabled": True,
            }
        )
    )
    assert pats_skill_context_lineage(tmp_path) is None


def test_async_lineage_accepts_exactly_one_stale_update() -> None:
    lineage = AsyncPolicyLineage(
        target_cycle=2,
        behavior_update_index=1,
        proposer_snapshot="/checkpoints/proposer/step-1",
        solver_snapshot="/checkpoints/solver/step-1",
        collection_mode="async_one_step_stale",
    )
    assert lineage.to_dict()["staleness_updates"] == 1

    with pytest.raises(ValueError, match="zero or one"):
        AsyncPolicyLineage(
            target_cycle=3,
            behavior_update_index=1,
            proposer_snapshot="p",
            solver_snapshot="s",
            collection_mode="async_one_step_stale",
        ).validate()


def test_bound_batches_preserve_identical_lineage_and_validate_learner(tmp_path) -> None:
    result = DryRunSelfPlayResult(
        tasks=(),
        frontier_scores=(),
        proposer_batch=_batch("proposer", {"selection": "same"}),
        solver_batch=_batch("solver", {"selection": "same"}),
        snapshots={"proposer": "p0", "solver": "s0"},
    )
    for name, batch in (
        ("proposer_batch.json", result.proposer_batch),
        ("solver_batch.json", result.solver_batch),
    ):
        (tmp_path / name).write_text(json.dumps(batch.to_dict()) + "\n")
    (tmp_path / "training_selection.json").write_text(
        json.dumps(
            {
                "postcollection_complete": True,
                "artifacts_sha256": {
                    name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                    for name in ("proposer_batch.json", "solver_batch.json")
                },
            }
        )
        + "\n"
    )
    bound = bind_rollout_result_lineage(
        result,
        output_dir=tmp_path,
        target_cycle=1,
        behavior_update_index=0,
        collection_mode="async_one_step_stale",
    )

    assert bound.proposer_batch.metadata == bound.solver_batch.metadata
    learner = validate_batch_lineage(
        bound.proposer_batch,
        bound.solver_batch,
        learner_update_index=1,
        learner_proposer_snapshot="p1",
        learner_solver_snapshot="s1",
    )
    assert learner["mode"] == "bounded_stale_ppo"
    assert learner["staleness_updates"] == 1
    assert (tmp_path / "policy_lineage.json").exists()
    resumed = bind_rollout_result_lineage(
        result,
        output_dir=tmp_path,
        target_cycle=1,
        behavior_update_index=1,
        collection_mode="synchronous",
    )
    assert resumed.solver_batch.metadata["policy_lineage"]["behavior_update_index"] == 0
    selection = json.loads((tmp_path / "training_selection.json").read_text())
    for name in ("proposer_batch.json", "solver_batch.json"):
        assert (
            selection["artifacts_sha256"][name]
            == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        )


def test_pats_async_lineage_binds_reviewed_view_and_rejects_changed_snapshot(tmp_path) -> None:
    expected_context = _write_pats_context(tmp_path)
    stale_context = {**expected_context, "pats_step": 0}
    with pytest.raises(ValueError, match="review committed for its target cycle"):
        AsyncPolicyLineage(
            target_cycle=1,
            behavior_update_index=0,
            proposer_snapshot="p0",
            solver_snapshot="s0",
            collection_mode="async_one_step_stale",
            skill_context=stale_context,
        ).validate()
    result = DryRunSelfPlayResult(
        tasks=(),
        frontier_scores=(),
        proposer_batch=_batch("proposer"),
        solver_batch=_batch("solver"),
        snapshots={"proposer": "p0", "solver": "s0"},
    )
    bound = bind_rollout_result_lineage(
        result,
        output_dir=tmp_path,
        target_cycle=1,
        behavior_update_index=0,
        collection_mode="async_one_step_stale",
    )

    lineage = bound.solver_batch.metadata["policy_lineage"]
    assert lineage["schema_version"] == "async_policy_lineage_v2"
    assert lineage["skill_context"] == expected_context
    learner = validate_batch_lineage(
        bound.proposer_batch,
        bound.solver_batch,
        learner_update_index=1,
        learner_proposer_snapshot="p1",
        learner_solver_snapshot="s1",
        expected_skill_context=expected_context,
    )
    assert learner["mode"] == "bounded_stale_ppo"
    assert learner["skill_context"]["pats_step"] == 1

    assert (
        validate_async_queue_skill_context({}, bound.solver_batch, pats_enabled=True)
        == expected_context
    )
    with pytest.raises(ValueError, match="durable async queue"):
        validate_async_queue_skill_context(
            {}, bound.solver_batch, pats_enabled=True, require_persisted=True
        )
    with pytest.raises(ValueError, match="different PATS snapshot"):
        validate_async_queue_skill_context(
            {"skill_context": {**expected_context, "pats_step": 0}},
            bound.solver_batch,
            pats_enabled=True,
        )

    snapshot_path = tmp_path / "director_skill_snapshot.v2.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["cards"].append({"tampered": True})
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    changed_context = pats_skill_context_lineage(tmp_path)
    with pytest.raises(ValueError, match="PATS lineage"):
        validate_batch_lineage(
            bound.proposer_batch,
            bound.solver_batch,
            learner_update_index=1,
            learner_proposer_snapshot="p1",
            learner_solver_snapshot="s1",
            expected_skill_context=changed_context,
        )
    with pytest.raises(ValueError, match="policy/PATS lineage"):
        bind_rollout_result_lineage(
            result,
            output_dir=tmp_path,
            target_cycle=1,
            behavior_update_index=0,
            collection_mode="async_one_step_stale",
        )


def test_lineage_rejects_second_stale_update() -> None:
    lineage = AsyncPolicyLineage(
        target_cycle=1,
        behavior_update_index=0,
        proposer_snapshot="p0",
        solver_snapshot="s0",
        collection_mode="async_one_step_stale",
    ).to_dict()
    with pytest.raises(ValueError, match="target cycle"):
        validate_batch_lineage(
            _batch("proposer", {"policy_lineage": lineage}),
            _batch("solver", {"policy_lineage": lineage}),
            learner_update_index=2,
            learner_proposer_snapshot="p2",
            learner_solver_snapshot="s2",
        )


def test_shared_vllm_service_exposes_base_alias_and_two_loras(tmp_path) -> None:
    p = tmp_path / "p"
    s = tmp_path / "s"
    p.mkdir()
    s.mkdir()
    spec = ModelServiceSpec(
        role="async_rollout",
        port=18013,
        served_model="base",
        base_model_path=Path("/models/qwen"),
        served_model_aliases=("base-alias",),
        lora_modules=(("proposer", p), ("solver", s)),
    )
    command = spec.command()

    served = command.index("--served-model-name")
    loras = command.index("--lora-modules")
    assert command[served + 1] == "base-alias"
    assert command[loras + 1 :] == [f"proposer={p}", f"solver={s}"]


def test_initial_ratio_diagnostics_reports_stale_clip_pressure() -> None:
    diagnostics = _initial_ratio_diagnostics(
        [0.0, 0.1, 0.3, -0.4], clip_range=0.2, staleness_updates=1
    )
    assert diagnostics["staleness_updates"] == 1
    assert diagnostics["initial_ratio_tokens"] == 4
    assert diagnostics["initial_ratio_clip_fraction"] == 0.5


def test_trainer_rejects_batch_addressed_to_a_later_cycle(tmp_path) -> None:
    lineage = AsyncPolicyLineage(
        target_cycle=2,
        behavior_update_index=1,
        proposer_snapshot="p1",
        solver_snapshot="s1",
        collection_mode="async_one_step_stale",
    ).to_dict()
    config = AlternatingTrainingConfig(
        PolicyTrainingConfig(tmp_path / "base", tmp_path / "p"),
        PolicyTrainingConfig(tmp_path / "base", tmp_path / "s"),
        tmp_path / "state.json",
    )
    trainer = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy, seed: MockPolicyTrainer(role, policy, seed=seed),
    )

    with pytest.raises(UnsafeTrainingBatchError, match="one-update boundary"):
        trainer.train_cycle(
            _batch("proposer", {"policy_lineage": lineage}),
            _batch("solver", {"policy_lineage": lineage}),
        )


def test_final_collection_boundary_keeps_prior_cycles_trainable():
    from argparse import Namespace

    from selfplay_graph_flowsteer.cli import _cycle_collection_only

    args = Namespace(cycles=3, collection_only=False, final_cycle_collection_only=True)
    assert [_cycle_collection_only(args, c) for c in range(3)] == [False, False, True]
    args.collection_only = True
    assert all(_cycle_collection_only(args, c) for c in range(3))


@pytest.mark.parametrize("bound_count", [1, 2])
@pytest.mark.parametrize("tamper", [False, True])
def test_partial_lineage_commit_recovers_only_unchanged_training_content(
    tmp_path, bound_count, tamper
):
    from selfplay_graph_flowsteer.async_cycle import (
        _atomic_json,
        recover_interrupted_lineage_binding,
    )

    out = tmp_path / "cycle-0000"
    out.mkdir()
    payload = AsyncPolicyLineage(0, 0, "base-p", "base-s", "synchronous").to_dict()
    _atomic_json(out / "snapshots.json", {"proposer": "base-p", "solver": "base-s"})
    _atomic_json(out / "policy_lineage.json", payload)
    hashes = {}
    names = ("proposer_batch.json", "solver_batch.json")
    for name in names:
        _atomic_json(out / name, {"samples": [{"reward": 1, "token_ids": [1, 2]}], "metadata": {}})
        hashes[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    _atomic_json(
        out / "training_selection.json",
        {"postcollection_complete": True, "artifacts_sha256": hashes},
    )
    for name in names[:bound_count]:
        batch = json.loads((out / name).read_text())
        batch["metadata"]["policy_lineage"] = payload
        if tamper:
            batch["samples"][0]["reward"] = 0
        _atomic_json(out / name, batch)
    before = {name: (out / name).read_bytes() for name in names}
    assert recover_interrupted_lineage_binding(out) is (not tamper)
    if tamper:
        assert all((out / name).read_bytes() == data for name, data in before.items())
    else:
        selection = json.loads((out / "training_selection.json").read_text())
        for name in names:
            assert json.loads((out / name).read_text())["metadata"]["policy_lineage"] == payload
            assert (
                hashlib.sha256((out / name).read_bytes()).hexdigest()
                == selection["artifacts_sha256"][name]
            )
        assert recover_interrupted_lineage_binding(out) is False


def test_managed_gpu_lease_recognizes_owner_and_rejects_stale_pid(tmp_path, monkeypatch):
    from selfplay_graph_flowsteer.services import (
        _write_gpu_reservation_leases,
        gpu_reservation_lease_active,
    )

    monkeypatch.setenv("SPGFS_GPU_LEASE_DIR", str(tmp_path))
    _write_gpu_reservation_leases((1, 3))
    assert gpu_reservation_lease_active(tmp_path / "gpu1-managed-lease.json")
    path = tmp_path / "gpu3-managed-lease.json"
    assert gpu_reservation_lease_active(path)
    lease = json.loads(path.read_text())
    lease["owner_start_ticks"] = str(int(lease["owner_start_ticks"]) + 1)
    path.write_text(json.dumps(lease))
    assert not gpu_reservation_lease_active(path)
    path.write_text("{")
    assert not gpu_reservation_lease_active(path)
    assert not gpu_reservation_lease_active(tmp_path / "missing.json")
