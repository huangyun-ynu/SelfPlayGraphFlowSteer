from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.application import create_selfplay_snapshots, load_adaptive_config
from selfplay_graph_flowsteer.benchmark import paired_summary, summarize
from selfplay_graph_flowsteer.cli import (
    _load_seeds,
    _pending_cycle_metrics_updates,
    _service_specs,
    _validate_attribution_classifier_repair_resume,
    _validate_infrastructure_repair_resume,
    _validate_proposal_selection_repair_resume,
    _validate_rollout_policy_snapshots,
    _validate_transient_backend_circuit_resume,
    _validate_uncertain_group_recollection_resume,
    main,
)
from selfplay_graph_flowsteer.counterfactual import RelationCredit
from selfplay_graph_flowsteer.distributed import ThreadRolloutPool
from selfplay_graph_flowsteer.evaluation import EvaluationRecord
from selfplay_graph_flowsteer.observability import (
    MultipleChoiceVerifier,
    NumericVerifier,
    TaskSpec,
)
from selfplay_graph_flowsteer.pool_audit import audit_fixed_task_pool
from selfplay_graph_flowsteer.rollouts import TokenizedPolicyCall, TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.services import (
    DEFAULT_SERVICE_START_WAIT_S,
    ModelServiceSpec,
    _spec_fingerprint,
    _state_matches_spec,
)
from selfplay_graph_flowsteer.training import (
    AlternatingGRPOTrainer,
    AlternatingTrainingConfig,
    MockPolicyTrainer,
    PolicyTrainingConfig,
    TransformersGRPOTrainer,
    UnsafeTrainingBatchError,
    _policy_target_count,
    _scheduler_lrs_at_step,
    _unit_policy_mask,
    dynamic_micro_batch_indices,
    token_advantages,
    training_device_for_gpu,
)


def test_policy_target_count_uses_per_call_next_token_masks() -> None:
    sample = TrainingSample(
        "rollout",
        "task",
        (1, 2, 3),
        (0, 1, 1),
        1.0,
        0.5,
        policy_calls=(
            TokenizedPolicyCall("call-0", (10, 11, 12), (0, 1, 1)),
            TokenizedPolicyCall("call-1", (20, 21), (0, 1)),
        ),
    )

    assert _policy_target_count(sample) == 3


def test_policy_mask_count_is_exact_when_logits_use_bfloat16() -> None:
    torch = pytest.importorskip("torch")
    log_probs = torch.zeros(5_299, dtype=torch.bfloat16)

    mask = _unit_policy_mask(log_probs, torch)

    assert mask.dtype == torch.float32
    assert int(mask.sum().item()) == 5_299


def test_training_defaults_use_kl_without_entropy_gradient(tmp_path) -> None:
    config = PolicyTrainingConfig(Path("/models/base"), tmp_path / "checkpoints")
    assert config.learning_rate == 1e-5
    assert config.clip_range == 0.2
    assert config.kl_coefficient == 0.005
    assert config.entropy_coefficient == 0.0
    assert config.loss_variant == "kl005_entropy0"
    assert config.weight_decay == 0.01
    assert (config.warmup_steps, config.total_optimizer_steps) == (10, 300)
    assert (config.lora_rank, config.lora_alpha, config.lora_dropout) == (64, 64, 0.05)
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert config.gradient_checkpointing
    assert not config.activation_cpu_offload
    assert config.activation_cpu_offload_min_tokens == 0
    assert config.max_micro_batch_tokens == 16_384


def test_activation_cpu_offload_threshold_is_non_negative(tmp_path) -> None:
    config = PolicyTrainingConfig(
        Path("/models/base"),
        tmp_path / "checkpoints",
        activation_cpu_offload=True,
        activation_cpu_offload_min_tokens=-1,
    )
    with pytest.raises(ValueError, match="activation_cpu_offload_min_tokens"):
        config.validate()


def test_activation_cpu_offload_threshold_selects_only_long_calls(tmp_path) -> None:
    trainer = object.__new__(TransformersGRPOTrainer)
    trainer.config = PolicyTrainingConfig(
        Path("/models/base"),
        tmp_path / "checkpoints",
        device="cuda:0",
        activation_cpu_offload=True,
        activation_cpu_offload_min_tokens=16_384,
    )

    assert not trainer._activation_cpu_offload_enabled(16_383)
    assert trainer._activation_cpu_offload_enabled(16_384)


def test_interrupted_solver_phase_validates_only_the_untrained_policy(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    adaptive = load_adaptive_config(config_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial = create_selfplay_snapshots(adaptive)
    (run_dir / "snapshots.json").write_text(
        json.dumps({"proposer": initial.proposer_snapshot, "solver": initial.solver_snapshot}),
        encoding="utf-8",
    )
    proposer_step = tmp_path / "proposer" / "step-00000001"
    proposer_step.mkdir(parents=True)
    (tmp_path / "proposer" / "latest.json").write_text(
        json.dumps({"step": 1, "path": str(proposer_step)}), encoding="utf-8"
    )
    updated = load_adaptive_config(config_path)

    _validate_rollout_policy_snapshots(run_dir, updated, allow_stale=False, roles=("solver",))
    with pytest.raises(ValueError, match="stale for proposer"):
        _validate_rollout_policy_snapshots(run_dir, updated, allow_stale=False)


def test_committed_cycle_recovers_missing_metrics_without_another_update(tmp_path) -> None:
    def update(role: str, step: int) -> dict[str, object]:
        return {
            "role": role,
            "step": step,
            "loss": 0.1,
            "policy_loss": 0.1,
            "kl": 0.0,
            "masked_tokens": 1,
            "checkpoint": str(tmp_path / role / f"step-{step:08d}"),
        }

    state = {
        "cycle": 1,
        "phase": "proposer",
        "proposer_step": 1,
        "solver_step": 2,
        "history": [update("proposer", 1), update("solver", 2)],
    }
    recovered = _pending_cycle_metrics_updates(state, tmp_path / "missing.json")

    assert recovered is not None
    assert [(item.role, item.step) for item in recovered] == [
        ("proposer", 1),
        ("solver", 2),
    ]
    metrics = tmp_path / "training_metrics_latest.json"
    metrics.write_text(json.dumps({"policies": {"solver": {"step": 2}}}), encoding="utf-8")
    assert _pending_cycle_metrics_updates(state, metrics) is None


def test_dynamic_micro_batches_use_padded_token_ceiling() -> None:
    samples = tuple(
        TrainingSample(
            f"short-{index}",
            "task",
            tuple(range(1_000)),
            (0,) * 999 + (1,),
            1.0,
            0.5,
        )
        for index in range(8)
    ) + tuple(
        TrainingSample(
            f"long-{index}",
            "task",
            tuple(range(4_096)),
            (0,) * 4_095 + (1,),
            1.0,
            0.5,
        )
        for index in range(4)
    )

    groups = dynamic_micro_batch_indices(
        samples,
        max_batch_size=8,
        max_padded_tokens=16_384,
    )

    assert tuple(len(group) for group in groups) == (8, 4)
    assert [index for group in groups for index in group] == list(range(12))


def test_dynamic_micro_batches_measure_actual_raw_calls() -> None:
    def sample(index, audit_length, call_length):
        return TrainingSample(
            str(index),
            "task",
            (1,) * audit_length,
            (1,) * audit_length,
            1.0,
            0.5,
            policy_calls=(
                TokenizedPolicyCall(
                    str(index),
                    (1,) * call_length,
                    (0,) + (1,) * (call_length - 1),
                ),
            ),
        )

    # A short generated answer can have a long actual conditioning context.
    assert dynamic_micro_batch_indices(
        [sample(0, 2, 10), sample(1, 2, 10)],
        max_batch_size=8,
        max_padded_tokens=16,
    ) == ((0,), (1,))
    # The audit concatenation is not fed to the model as one sequence.
    assert dynamic_micro_batch_indices(
        [sample(0, 30, 8), sample(1, 30, 8)],
        max_batch_size=8,
        max_padded_tokens=16,
    ) == ((0, 1),)
    with pytest.raises(ValueError, match="micro-batch token ceiling"):
        dynamic_micro_batch_indices(
            [sample(0, 2, 17)],
            max_batch_size=8,
            max_padded_tokens=16,
        )


def test_backend_resume_gate_keeps_permanent_and_recovered_quarantines(tmp_path) -> None:
    cycle = tmp_path / "cycle-0000"
    cycle.mkdir()
    quarantines = [
        {
            "task_id": "task-permanent",
            "status": "quarantined",
            "reason": "non_trainable_rollout_group",
            "non_trainable_rollout_ids": ["task-permanent-r1"],
        },
        {
            "task_id": "task-recovered",
            "status": "reopened_for_exact_resume",
            "prior_quarantine_reason": "backend_circuit_open",
            "missing_rollout_ids": ["task-recovered-r4"],
        },
        {
            "task_id": "task-pending",
            "status": "quarantined",
            "reason": "backend_circuit_open",
            "non_trainable_rollout_ids": [],
        },
    ]
    (cycle / "quarantined_groups.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in quarantines)
    )
    (cycle / "solver_rollouts.jsonl").write_text(
        json.dumps({"rollout_id": "task-recovered-r4"}) + "\n"
    )
    (cycle / "rollout_errors.jsonl").write_text(
        json.dumps({"error_type": "BackendCircuitOpenError"}) + "\n"
    )

    _validate_transient_backend_circuit_resume(cycle)


def test_backend_resume_gate_accepts_runtime_wrapper_and_inflight_group_error(tmp_path) -> None:
    cycle = tmp_path / "cycle-0000"
    cycle.mkdir()
    quarantines = [
        {
            "task_id": "task-backend",
            "status": "quarantined",
            "reason": "backend_circuit_open",
            "non_trainable_rollout_ids": [],
        },
        {
            "task_id": "task-inflight",
            "status": "quarantined",
            "reason": "backend_circuit_open",
            "non_trainable_rollout_ids": [],
        },
    ]
    (cycle / "quarantined_groups.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in quarantines)
    )
    (cycle / "solver_rollouts.jsonl").write_text("")
    (cycle / "rollout_errors.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "task_id": "task-backend",
                    "error_type": "BackendRetryExhaustedError",
                },
                {
                    "task_id": "task-backend",
                    "error_type": "BackendCircuitOpenError",
                },
                {
                    "task_id": "task-inflight",
                    "error_type": "NonTrainablePrimaryExhaustedError",
                },
            )
        )
    )

    _validate_transient_backend_circuit_resume(cycle)


def test_backend_resume_gate_rejects_unrelated_incomplete_quarantine(tmp_path) -> None:
    cycle = tmp_path / "cycle-0000"
    cycle.mkdir()
    (cycle / "quarantined_groups.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-pending",
                        "status": "quarantined",
                        "reason": "backend_circuit_open",
                        "non_trainable_rollout_ids": [],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-unrelated",
                        "status": "quarantined",
                        "reason": "incomplete_rollout_group",
                        "non_trainable_rollout_ids": [],
                    }
                ),
            ]
        )
        + "\n"
    )
    (cycle / "solver_rollouts.jsonl").write_text("")
    (cycle / "rollout_errors.jsonl").write_text(
        json.dumps({"error_type": "BackendCircuitOpenError"}) + "\n"
    )

    with pytest.raises(ValueError, match="task-unrelated"):
        _validate_transient_backend_circuit_resume(cycle)


def test_infrastructure_repair_gate_accepts_scoped_exact_missing_resume(tmp_path) -> None:
    cycle = tmp_path / "cycle-0001"
    cycle.mkdir()
    (cycle / "collection_repair_attestation.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "incident_class": "infrastructure",
                "recollection_mode": "exact_missing_rollout_ids",
                "source_tasks_sha256": "a" * 64,
                "affected_rollout_ids": ["task-6-r0"],
                "repaired_instance_ids": ["django__django-11603"],
                "repair_evidence": "preflight/repair.json",
                "persisted_rollouts_verified_unaffected": True,
            }
        )
        + "\n"
    )

    _validate_infrastructure_repair_resume(cycle)


def test_uncertain_group_gate_requires_every_sibling_to_be_absent(tmp_path) -> None:
    cycle = tmp_path / "cycle-0001"
    cycle.mkdir()
    (cycle / "tasks.jsonl").write_text(json.dumps({"task": {"task_id": "task-6"}}) + "\n")
    (cycle / "solver_rollouts.jsonl").write_text(json.dumps({"rollout_id": "task-7-r0"}) + "\n")
    attestation = {
        "status": "approved",
        "incident_class": "attribution_uncertain_recovery_exhausted",
        "recollection_mode": "exact_affected_task_groups",
        "group_recollection_scope": "all_siblings",
        "source_tasks_sha256": "a" * 64,
        "affected_task_ids": ["task-6"],
        "affected_rollout_ids": ["task-6-r0", "task-6-r1"],
        "policy_sampling_attempt_offsets": {"task-6": 2},
        "repair_evidence": {"selection_integrity": "all siblings discarded"},
        "persisted_rollouts_verified_unaffected": True,
    }
    path = cycle / "collection_repair_attestation.json"
    path.write_text(json.dumps(attestation) + "\n")

    assert _validate_uncertain_group_recollection_resume(cycle, rollouts_per_task=2) == {
        "task-6": 2
    }

    with (cycle / "solver_rollouts.jsonl").open("a") as handle:
        handle.write(json.dumps({"rollout_id": "task-6-r0"}) + "\n")
    with pytest.raises(ValueError, match="invalid uncertain-group"):
        _validate_uncertain_group_recollection_resume(cycle, rollouts_per_task=2)

    (cycle / "collection_incidents.jsonl").write_text(
        json.dumps(
            {"event": "collection_abort_reopened_for_full_group_recollection", "task_id": "task-6"}
        )
        + "\n"
    )
    # Reopening alone must never admit an old or unversioned sibling.
    with pytest.raises(ValueError, match="invalid uncertain-group"):
        _validate_uncertain_group_recollection_resume(cycle, rollouts_per_task=2)
    (cycle / "solver_rollouts.jsonl").write_text(
        json.dumps(
            {
                "rollout_id": "task-6-r0",
                "task_id": "task-6",
                "metadata": {"policy_sampling_attempt_offset": 2},
            }
        )
        + "\n"
    )
    assert _validate_uncertain_group_recollection_resume(cycle, rollouts_per_task=2) == {
        "task-6": 2
    }
    # A new incident invalidates the preceding epoch's reopening.
    with (cycle / "collection_incidents.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "collection_aborted", "task_id": "task-6"}) + "\n")
    with pytest.raises(ValueError, match="invalid uncertain-group"):
        _validate_uncertain_group_recollection_resume(cycle, rollouts_per_task=2)


def test_classifier_repair_gate_requires_one_absent_reclassified_slot(tmp_path) -> None:
    cycle = tmp_path / "cycle-0001"
    cycle.mkdir()
    (cycle / "tasks.jsonl").write_text(json.dumps({"task": {"task_id": "task-19"}}) + "\n")
    (cycle / "solver_rollouts.jsonl").write_text(json.dumps({"rollout_id": "task-19-r0"}) + "\n")
    attestation = {
        "status": "approved",
        "incident_class": "attribution_uncertain_recovery_exhausted",
        "recollection_mode": "exact_reclassified_rollout_ids",
        "source_tasks_sha256": "a" * 64,
        "reclassified_rollout_ids": ["task-19-r1"],
        "repair_evidence": {"bug": "BUG-046"},
        "persisted_rollouts_verified_unaffected": True,
    }
    path = cycle / "collection_repair_attestation.json"
    path.write_text(json.dumps(attestation) + "\n")

    _validate_attribution_classifier_repair_resume(cycle)

    with (cycle / "solver_rollouts.jsonl").open("a") as handle:
        handle.write(json.dumps({"rollout_id": "task-19-r1"}) + "\n")
    with pytest.raises(ValueError, match="invalid attribution-classifier"):
        _validate_attribution_classifier_repair_resume(cycle)


def test_proposal_selection_repair_gate_requires_a_typed_missing_task(tmp_path) -> None:
    cycle = tmp_path / "cycle-0001"
    cycle.mkdir()
    (cycle / "tasks.jsonl").write_text(json.dumps({"task": {"task_id": "task-11"}}) + "\n")
    (cycle / "solver_rollouts.jsonl").write_text(
        json.dumps({"task_id": "task-11", "rollout_id": "task-11-r0"}) + "\n"
    )
    (cycle / "proposal_attempts.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task-12",
                "success": False,
                "failure_kind": "format_or_validation_failure",
            }
        )
        + "\n"
    )
    (cycle / "collection_repair_attestation.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "incident_class": "proposal_selection_failure",
                "recollection_mode": "exact_missing_proposal_and_rollouts",
                "source_tasks_sha256": "a" * 64,
                "affected_task_ids": ["task-12"],
                "repair_evidence": {"bug": "anchor-only boundary"},
                "persisted_rollouts_verified_unaffected": True,
                "mace_snapshot_manifest_repaired": True,
            }
        )
        + "\n"
    )

    _validate_proposal_selection_repair_resume(cycle)

    with (cycle / "tasks.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": {"task_id": "task-12"}}) + "\n")
    with pytest.raises(ValueError, match="invalid proposal-selection"):
        _validate_proposal_selection_repair_resume(cycle)


def test_managed_service_refresh_allows_realistic_cold_start() -> None:
    assert DEFAULT_SERVICE_START_WAIT_S == 300.0


def test_extended_scheduler_horizon_restores_nonzero_current_learning_rate() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=30,
    )

    resumed_lrs = _scheduler_lrs_at_step(scheduler, 6)

    assert resumed_lrs == pytest.approx([6e-6])


def test_thread_rollout_pool_cancels_queued_jobs_when_iteration_stops() -> None:
    release_second = threading.Event()
    started: list[int] = []

    def run(value: int) -> int:
        started.append(value)
        if value == 1:
            release_second.wait(timeout=2)
        return value

    results = ThreadRolloutPool[int, int](workers=1).iter_map(run, [0, 1, 2])
    assert next(results) == 0
    release_second.set()
    results.close()

    assert started == [0]
    assert 2 not in started


def test_thread_rollout_pool_resumable_refills_trajectory_slots() -> None:
    slots = 24
    lock = threading.Lock()
    initial_slots_full = threading.Event()
    active = 0
    peak = 0
    started: list[int] = []

    def run(job: int):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(job)
            if active == slots:
                initial_slots_full.set()
        if job < slots:
            assert initial_slots_full.wait(timeout=2)
        if job == 0:
            yield 0.05
        with lock:
            active -= 1
        return job

    jobs = range(slots + 6)
    results = list(ThreadRolloutPool[int, int](workers=slots).iter_map_resumable(run, jobs))

    assert set(results) == set(jobs)
    assert peak == slots
    assert set(started[:slots]) == set(range(slots))


def test_thread_rollout_pool_can_launch_a_full_packed_window() -> None:
    """A 4-task x 5-rollout window reaches inference together."""

    target = 20
    release = threading.Event()
    all_started = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def run(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == target:
                all_started.set()
        release.wait(timeout=5)
        with lock:
            active -= 1
        return value

    with ThreadPoolExecutor(max_workers=1) as outer:
        future = outer.submit(
            ThreadRolloutPool[int, int](workers=target).map,
            run,
            list(range(target)),
        )
        assert all_started.wait(timeout=2)
        release.set()
        assert future.result(timeout=5) == list(range(target))

    assert peak == target


def test_selfplay_snapshot_resolves_exact_policy_checkpoint(tmp_path) -> None:
    adaptive = load_adaptive_config(_write_config(tmp_path))
    checkpoint = tmp_path / "proposer" / "step-00000003"
    checkpoint.mkdir(parents=True)
    (tmp_path / "proposer" / "latest.json").write_text(
        json.dumps({"path": str(checkpoint), "step": 3}), encoding="utf-8"
    )
    snapshots = create_selfplay_snapshots(adaptive)
    assert snapshots.proposer_snapshot == str(checkpoint.resolve())
    assert snapshots.solver_snapshot == str(adaptive.solver_model.base_model_path.resolve())


def test_seed_loader_accepts_existing_subset_schemas(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"problem": "math problem", "answer": "42"}),
                json.dumps({"problem": "zero answer", "answer": 0}),
                json.dumps({"instruction": "buy a red mug"}),
                json.dumps({"task_descriptions": ["put apple in fridge"]}),
                json.dumps({"problem_statement": "fix issue"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    seeds = _load_seeds(path)
    assert [seed.content for seed in seeds] == [
        "42",
        "0",
        "buy a red mug",
        '["put apple in fridge"]',
        "fix issue",
    ]
    assert seeds[0].target_answer == "42"
    assert seeds[1].target_answer == 0


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "adaptive.toml"
    path.write_text(
        f"""
[models.proposer]
base_url = "http://127.0.0.1:8001/v1"
served_model = "Qwen3.5-9B"
base_model_path = "models/Qwen3.5-9B"
checkpoint_path = "{tmp_path / "proposer"}"
trainable = true

[models.solver]
base_url = "http://127.0.0.1:8002/v1"
served_model = "Qwen3.5-9B"
base_model_path = "models/Qwen3.5-9B"
checkpoint_path = "{tmp_path / "solver"}"
trainable = true

[runtime]
base_url = "http://127.0.0.1:8003/v1"
served_model = "Qwen3.5-9B"
model_path = "models/Qwen3.5-9B"
frozen = true

[mace]
enabled = false


[trace]
path = "{tmp_path / "traces.jsonl"}"

[verifier]
mode = "none"
""".strip(),
        encoding="utf-8",
    )
    return path


def test_relation_credit_changes_only_selected_action_span() -> None:
    sample = TrainingSample("r1", "t1", (1, 2, 3, 4), (0, 1, 1, 0), 1.0, 0.5)
    credit = RelationCredit(
        "r1", 0, "a", "b", "bidirectional", 0.0, 1.0, -0.5, 0.5, 7, (1, 3), True
    )
    assert token_advantages(sample, [credit], relation_weight=2.0) == (
        0.5,
        1.0,
        1.0,
        0.5,
    )


def test_mock_full_experiment_updates_both_policies_and_resumes(tmp_path, capsys) -> None:
    config = _write_config(tmp_path)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text('{"seed":"answer"}\n', encoding="utf-8")
    output = tmp_path / "experiment"
    assert (
        main(
            [
                "selfplay-experiment",
                "--rollout-group-policy",
                "complete",
                "--config",
                str(config),
                "--seed-data",
                str(seeds),
                "--output",
                str(output),
                "--cycles",
                "2",
                "--rollouts",
                "2",
                "--verifier",
                "none",
                "--mock",
                "--mock-trainer",
            ]
        )
        == 0
    )
    capsys.readouterr()
    state = json.loads((output / "training_state.json").read_text())
    assert state["cycle"] == 2
    assert state["global_step"] == 4
    assert Path(state["proposer_checkpoint"]).exists()
    assert Path(state["solver_checkpoint"]).exists()
    assert len((output / "training_metrics.jsonl").read_text().splitlines()) == 2
    assert len((output / "training_curves.csv").read_text().splitlines()) == 3
    assert (output / "training_metrics_latest.json").exists()
    assert len((output / "training_steps.jsonl").read_text().splitlines()) == 4
    assert len((output / "training_step_curves.csv").read_text().splitlines()) == 5
    latest_step = json.loads((output / "training_steps_latest.json").read_text())
    assert latest_step["training_step"] == 4
    assert latest_step["role"] == "solver"
    assert (
        main(
            [
                "selfplay-experiment",
                "--rollout-group-policy",
                "complete",
                "--config",
                str(config),
                "--seed-data",
                str(seeds),
                "--output",
                str(output),
                "--cycles",
                "2",
                "--rollouts",
                "2",
                "--verifier",
                "none",
                "--mock",
                "--mock-trainer",
                "--resume",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert len((output / "training_metrics.jsonl").read_text().splitlines()) == 2


def test_mock_final_cycle_evaluation_only_does_not_update_policies(tmp_path, capsys) -> None:
    config = _write_config(tmp_path)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text('{"seed":"answer"}\n', encoding="utf-8")
    output = tmp_path / "experiment"

    assert (
        main(
            [
                "selfplay-experiment",
                "--rollout-group-policy",
                "complete",
                "--config",
                str(config),
                "--seed-data",
                str(seeds),
                "--output",
                str(output),
                "--cycles",
                "2",
                "--rollouts",
                "2",
                "--verifier",
                "none",
                "--mock",
                "--mock-trainer",
                "--final-cycle-evaluation-only",
            ]
        )
        == 0
    )
    capsys.readouterr()

    state = json.loads((output / "training_state.json").read_text())
    assert (state["cycle"], state["global_step"], state["phase"]) == (1, 2, "proposer")
    assert (state["proposer_step"], state["solver_step"]) == (1, 2)
    assert not (output / "checkpoints" / "proposer" / "step-00000003").exists()
    assert not (output / "checkpoints" / "solver" / "step-00000004").exists()

    progress = json.loads((output / "experiment_progress.json").read_text())
    assert progress["completed_cycles"] == 2
    assert progress["cycles"][0]["mode"] == "train"
    assert progress["cycles"][1]["mode"] == "evaluation_only"
    assert progress["cycles"][1]["updates"] == []

    metrics = [
        json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 2
    assert metrics[1]["policies"] == {}
    assert metrics[1]["experiment"]["evaluation_only"] is True
    assert len((output / "training_steps.jsonl").read_text().splitlines()) == 2
    assert len((output / "training_curves.csv").read_text().splitlines()) == 3
    assert (output / "training_metrics_latest.json").exists()


def test_audited_fixed_pool_experiment_does_not_require_seed_data(
    tmp_path, capsys, monkeypatch
) -> None:
    from .helpers import install_numeric_mock_worker

    install_numeric_mock_worker(monkeypatch)
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0,1")
    config = _write_config(tmp_path)
    pool = tmp_path / "aime.jsonl"
    rows = [
        {
            "id": f"aime:{index}",
            "dataset": "aime",
            "split": "train",
            "prompt": f"What is {index}+1?",
            "target_answers": [index + 1],
            "task_type": "math",
            "verifier": "numeric",
        }
        for index in (1, 2)
    ]
    pool.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    audited = audit_fixed_task_pool([pool], output_dir=tmp_path / "audited-pool")
    output = tmp_path / "fixed-experiment"

    assert (
        main(
            [
                "selfplay-experiment",
                "--rollout-group-policy",
                "complete",
                "--config",
                str(config),
                "--task-pool",
                str(audited.validated_path),
                "--tasks-per-cycle",
                "1",
                "--proposals-per-seed",
                "1",
                "--output",
                str(output),
                "--rollouts",
                "2",
                "--counterfactuals-per-rollout",
                "0",
                "--verifier",
                "numeric",
                "--mock",
                "--mock-trainer",
                "--proposer-gpu-id",
                "0",
                "--solver-gpu-id",
                "1",
                "--parallel-role-training",
                "--micro-batch-size",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    metrics = json.loads((output / "training_metrics_latest.json").read_text())
    assert metrics["experiment"]["seed_data"] is None
    assert metrics["experiment"]["task_pools"] == [str(audited.validated_path.resolve())]
    assert metrics["experiment"]["checkpoint_root"] == str((output / "checkpoints").resolve())
    assert metrics["experiment"]["runtime_state_root"] == str((output / "runtime_state").resolve())
    assert metrics["experiment"]["optimizer_schedule"] == {
        "epochs": 1,
        "mini_batch_size": 64,
        "micro_batch_size": 8,
        "max_micro_batch_tokens": 16_384,
        "activation_cpu_offload": False,
        "activation_cpu_offload_min_tokens": 0,
        "gradient_accumulation_steps": 8,
        "parallel_role_training": True,
        "proposer_total_optimizer_steps": 1,
        "solver_total_optimizer_steps": 1,
        "formula": ("cycles * epochs * ceil(role_samples_per_cycle / logical_mini_batch_size)"),
        "solver_global_trajectory_mean": False,
    }
    assert (output / "checkpoints" / "proposer" / "latest.json").is_file()
    assert (output / "checkpoints" / "solver" / "latest.json").is_file()


def test_realistic_verifiers() -> None:
    numeric = NumericVerifier(absolute_tolerance=1e-3)
    assert numeric.verify(TaskSpec("n", "", reference="3.1416"), "answer=3.142").passed
    assert numeric.verify(TaskSpec("z", "", reference=0), "answer=0").passed
    choice = MultipleChoiceVerifier()
    assert choice.verify(TaskSpec("m", "", reference="B"), "Final answer: (B)").passed


def test_benchmark_summary_and_pairing() -> None:
    candidate = [
        EvaluationRecord("a", "candidate", "", 1.0, True, seed=0),
        EvaluationRecord("b", "candidate", "", 0.0, False, seed=0),
    ]
    baseline = [
        EvaluationRecord("a", "flowsteer", "", 0.0, False, seed=0),
        EvaluationRecord("b", "flowsteer", "", 0.0, False, seed=0),
    ]
    assert summarize(candidate).pass_rate == 0.5
    paired = paired_summary(candidate, baseline)
    assert (paired.wins, paired.ties, paired.losses) == (1, 1, 0)


def test_benchmark_clips_only_the_final_mean() -> None:
    records = [
        EvaluationRecord("a", "candidate", "", -0.5, False),
        EvaluationRecord("b", "candidate", "", 0.25, False),
    ]
    summary = summarize(records)
    assert summary.unclipped_mean_score == -0.125
    assert summary.mean_score == 0.0


def test_service_command_loads_lora_checkpoint() -> None:
    spec = ModelServiceSpec(
        "solver",
        8002,
        "Qwen3.5-9B",
        Path("/models/base"),
        Path("/models/solver/step-1"),
    )
    command = spec.command()
    assert "--enable-lora" in command
    assert command[command.index("--max-lora-rank") + 1] == "64"
    assert "Qwen3.5-9B=/models/solver/step-1" in command


def test_service_command_reads_checkpoint_lora_rank(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-1"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"r": 128}', encoding="utf-8")
    command = ModelServiceSpec(
        "solver", 8002, "Qwen3.5-9B", Path("/models/base"), checkpoint
    ).command()

    assert command[command.index("--max-lora-rank") + 1] == "128"


def test_service_command_uses_configured_gpu_memory_utilization() -> None:
    command = ModelServiceSpec(
        "solver",
        8002,
        "Qwen3.5-9B",
        Path("/models/base"),
        gpu_memory_utilization=0.5,
    ).command()

    assert command[command.index("--gpu-memory-utilization") + 1] == "0.5"


def test_service_state_must_match_exact_policy_and_resources() -> None:
    checkpoint = Path("/models/solver/step-1")
    spec = ModelServiceSpec(
        "solver",
        8002,
        "Qwen3.5-9B",
        Path("/models/base"),
        checkpoint,
        gpu_memory_utilization=0.5,
        extra_env=(("VLLM_TEST", "1"),),
        gpu_ids=(3,),
    )
    state = {"spec_fingerprint": _spec_fingerprint(spec)}

    assert _state_matches_spec(state, spec)
    assert not _state_matches_spec(
        state, ModelServiceSpec("solver", 8002, "Qwen3.5-9B", Path("/models/base"))
    )
    assert not _state_matches_spec(
        state,
        ModelServiceSpec(
            "solver",
            8002,
            "Qwen3.5-9B",
            Path("/models/base"),
            checkpoint,
            gpu_memory_utilization=0.85,
            extra_env=(("VLLM_TEST", "1"),),
            gpu_ids=(3,),
        ),
    )


def test_qwen_services_use_python311_compatible_vllm_path(tmp_path) -> None:
    config = load_adaptive_config(_write_config(tmp_path))
    for spec in _service_specs(config).values():
        expected = [
            "--enforce-eager",
            "--max-model-len",
            "32768",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--gdn-prefill-backend",
            "triton",
        ]
        if spec.role == "runtime":
            expected += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]
        assert spec.command()[-len(expected) :] == expected
        assert spec.extra_env == (("VLLM_USE_FLASHINFER_SAMPLER", "0"),)


def test_managed_policy_ports_follow_configured_endpoints(tmp_path) -> None:
    from dataclasses import replace

    config = load_adaptive_config(_write_config(tmp_path))
    config = replace(
        config,
        proposer_model=replace(config.proposer_model, base_url="http://127.0.0.1:18011/v1"),
        solver_model=replace(config.solver_model, base_url="http://127.0.0.1:18012/v1"),
        proposer_gpu_id=1,
        solver_gpu_id=3,
    )
    specs = _service_specs(config)
    assert specs["proposer"].port == 18011
    assert specs["solver"].port == 18012
    assert specs["proposer"].gpu_ids == (1,)
    assert specs["solver"].gpu_ids == (3,)


def test_thread_rollout_pool_map_is_ordered_but_iter_map_uses_completion_order() -> None:
    import time

    assert ThreadRolloutPool[int, int](workers=2).map(lambda value: value * 2, [3, 1, 2]) == [
        6,
        2,
        4,
    ]
    assert list(
        ThreadRolloutPool[int, int](workers=2).iter_map(
            lambda value: (time.sleep({3: 0.20, 1: 0.01, 2: 0.02}[value]), value * 2)[1],
            [3, 1, 2],
        )
    ) == [2, 4, 6]


def test_training_devices_map_physical_ids_through_visible_devices() -> None:
    assert training_device_for_gpu(0, visible_devices="0,3,4") == "cuda:0"
    assert training_device_for_gpu(3, visible_devices="0,3,4") == "cuda:1"
    assert training_device_for_gpu(3, visible_devices="") == "cuda:3"


def test_parallel_trainer_defers_service_refresh_until_train_cycle_returns(tmp_path) -> None:
    events: list[tuple[str, str]] = []
    policy = lambda role: PolicyTrainingConfig(  # noqa: E731
        Path("/models/base"), tmp_path / role, device=f"cuda:{0 if role == 'proposer' else 3}"
    )
    trainer = AlternatingGRPOTrainer(
        AlternatingTrainingConfig(
            policy("proposer"),
            policy("solver"),
            tmp_path / "training_state.json",
            parallel_roles=True,
        ),
        trainer_factory=lambda role, config, seed: MockPolicyTrainer(role, config, seed=seed),
        before_role_callback=lambda role: events.append(("stop", role)),
        checkpoint_callback=lambda role, checkpoint: events.append(("refresh", role)),
    )
    sample = TrainingSample("rollout", "task", (1, 2), (0, 1), 1.0, 0.5)
    trainer.train_cycle(
        TrainingBatch("proposer", (sample,)),
        TrainingBatch("solver", (sample,)),
    )
    assert events == [("stop", "proposer"), ("stop", "solver")]


def test_committed_cycle_results_prevent_duplicate_resume_update(tmp_path) -> None:
    calls: list[str] = []

    class RecordingTrainer(MockPolicyTrainer):
        def update(self, batch, *, step, relation_credits=()):
            calls.append(self.role)
            return super().update(batch, step=step, relation_credits=relation_credits)

    policy = lambda role: PolicyTrainingConfig(  # noqa: E731
        Path("/models/base"), tmp_path / role, device=f"cuda:{0 if role == 'proposer' else 1}"
    )
    config = AlternatingTrainingConfig(
        policy("proposer"),
        policy("solver"),
        tmp_path / "training_state.json",
        parallel_roles=True,
    )
    sample = TrainingSample("rollout", "task", (1, 2), (0, 1), 1.0, 0.5)
    batch = TrainingBatch("proposer", (sample,))
    solver_batch = TrainingBatch("solver", (sample,))
    first = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy_config, seed: RecordingTrainer(
            role, policy_config, seed=seed
        ),
    )
    expected = first.train_cycle(batch, solver_batch)
    resumed = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy_config, seed: RecordingTrainer(
            role, policy_config, seed=seed
        ),
    )

    assert resumed.committed_cycle_results(0) == expected
    assert calls == ["proposer", "solver"]


def test_parallel_role_training_overlaps_updates_and_commits_both(tmp_path) -> None:
    barrier = threading.Barrier(2)
    threads: dict[str, int] = {}

    class CoordinatedTrainer(MockPolicyTrainer):
        def update(self, batch, *, step, relation_credits=()):
            threads[self.role] = threading.get_ident()
            barrier.wait(timeout=5)
            return super().update(batch, step=step, relation_credits=relation_credits)

    def policy(role: str) -> PolicyTrainingConfig:
        return PolicyTrainingConfig(
            Path("/models/base"),
            tmp_path / role,
            device=f"cuda:{0 if role == 'proposer' else 1}",
        )

    config = AlternatingTrainingConfig(
        policy("proposer"),
        policy("solver"),
        tmp_path / "training_state.json",
        parallel_roles=True,
    )
    trainer = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy_config, seed: CoordinatedTrainer(
            role, policy_config, seed=seed
        ),
    )
    sample = TrainingSample("rollout", "task", (1, 2), (0, 1), 1.0, 0.5)

    proposer, solver = trainer.train_cycle(
        TrainingBatch("proposer", (sample,)),
        TrainingBatch("solver", (sample,)),
    )

    assert threads["proposer"] != threads["solver"]
    assert (proposer.step, solver.step) == (1, 2)
    state = json.loads(config.state_path.read_text())
    assert (state["cycle"], state["global_step"], state["phase"]) == (1, 2, "proposer")
    assert json.loads((tmp_path / "proposer/latest.json").read_text())["step"] == 1
    assert json.loads((tmp_path / "solver/latest.json").read_text())["step"] == 2


def test_parallel_role_failure_does_not_publish_either_checkpoint(tmp_path) -> None:
    barrier = threading.Barrier(2)

    class FailSolver(MockPolicyTrainer):
        def update(self, batch, *, step, relation_credits=()):
            barrier.wait(timeout=5)
            if self.role == "solver":
                raise RuntimeError("simulated parallel solver failure")
            return super().update(batch, step=step, relation_credits=relation_credits)

    def policy(role: str) -> PolicyTrainingConfig:
        root = tmp_path / role
        old = root / "step-00000000"
        old.mkdir(parents=True)
        (root / "latest.json").write_text(json.dumps({"path": str(old), "step": 0}) + "\n")
        return PolicyTrainingConfig(
            Path("/models/base"),
            root,
            device=f"cuda:{0 if role == 'proposer' else 1}",
        )

    config = AlternatingTrainingConfig(
        policy("proposer"),
        policy("solver"),
        tmp_path / "training_state.json",
        parallel_roles=True,
    )
    trainer = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy_config, seed: FailSolver(
            role, policy_config, seed=seed
        ),
    )
    sample = TrainingSample("rollout", "task", (1, 2), (0, 1), 1.0, 0.5)

    with pytest.raises(RuntimeError, match="parallel solver failure"):
        trainer.train_cycle(
            TrainingBatch("proposer", (sample,)),
            TrainingBatch("solver", (sample,)),
        )

    assert not config.state_path.exists()
    assert json.loads((tmp_path / "proposer/latest.json").read_text())["step"] == 0
    assert json.loads((tmp_path / "solver/latest.json").read_text())["step"] == 0
    assert not config.state_path.with_suffix(".parallel-transaction.json").exists()


def test_parallel_publish_transaction_recovers_previous_joint_lineage(tmp_path) -> None:
    def policy(role: str) -> PolicyTrainingConfig:
        root = tmp_path / role
        old_step = 1 if role == "proposer" else 2
        old = root / f"step-{old_step:08d}"
        new = root / "step-00000003"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        (root / "latest.json").write_text(json.dumps({"path": str(new), "step": 3}) + "\n")
        return PolicyTrainingConfig(
            Path("/models/base"),
            root,
            device=f"cuda:{0 if role == 'proposer' else 1}",
        )

    proposer = policy("proposer")
    solver = policy("solver")
    state_path = tmp_path / "training_state.json"
    old_proposer = proposer.checkpoint_root / "step-00000001"
    old_solver = solver.checkpoint_root / "step-00000002"
    state_path.write_text(
        json.dumps(
            {
                "cycle": 1,
                "global_step": 2,
                "proposer_step": 1,
                "solver_step": 2,
                "proposer_checkpoint": str(old_proposer),
                "solver_checkpoint": str(old_solver),
                "phase": "proposer",
                "history": [],
            }
        )
        + "\n"
    )
    transaction_path = state_path.with_suffix(".parallel-transaction.json")
    transaction_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_cycle": 2,
                "previous_latest": {
                    "proposer": {"path": str(old_proposer), "step": 1},
                    "solver": {"path": str(old_solver), "step": 2},
                },
            }
        )
        + "\n"
    )

    AlternatingGRPOTrainer(
        AlternatingTrainingConfig(
            proposer,
            solver,
            state_path,
            parallel_roles=True,
        ),
        trainer_factory=lambda role, policy_config, seed: MockPolicyTrainer(
            role, policy_config, seed=seed
        ),
    )

    assert json.loads((proposer.checkpoint_root / "latest.json").read_text())["step"] == 1
    assert json.loads((solver.checkpoint_root / "latest.json").read_text())["step"] == 2
    assert not transaction_path.exists()


def test_alternating_trainer_rejects_role_specific_gradient_accumulation(tmp_path) -> None:
    proposer = PolicyTrainingConfig(
        Path("/models/base"),
        tmp_path / "proposer",
        gradient_accumulation_steps=1,
    )
    solver = PolicyTrainingConfig(
        Path("/models/base"),
        tmp_path / "solver",
        gradient_accumulation_steps=2,
    )

    with pytest.raises(ValueError, match="same mini-batch, micro-batch"):
        AlternatingGRPOTrainer(
            AlternatingTrainingConfig(
                proposer,
                solver,
                tmp_path / "training_state.json",
            )
        )


@pytest.mark.parametrize("excluded_count", [0, 1, 2])
def test_canary_proposer_exclusion_requires_complete_explicit_solver_group(excluded_count):
    trainer = object.__new__(AlternatingGRPOTrainer)
    proposer = TrainingBatch("proposer", (TrainingSample("p", "kept", (1, 2), (0, 1), 0, 0),))
    samples = []
    for task in ("kept", "excluded"):
        for index in range(2):
            metadata = {"rollout_group_complete": True, "rollout_group_size_expected": 2}
            if task == "excluded" and index < excluded_count:
                metadata["frontier_training_exclusion"] = "canary_executor_migration"
            samples.append(
                TrainingSample(f"{task}-{index}", task, (1, 2), (0, 1), 0, 0, metadata=metadata)
            )
    if excluded_count == 2:
        trainer._validate_cycle_batches(proposer, TrainingBatch("solver", tuple(samples)))
    else:
        with pytest.raises(UnsafeTrainingBatchError):
            trainer._validate_cycle_batches(proposer, TrainingBatch("solver", tuple(samples)))


def test_training_gate_rejects_incomplete_group_before_any_role_update(tmp_path) -> None:
    events: list[str] = []
    policy = lambda role: PolicyTrainingConfig(  # noqa: E731
        Path("/models/base"), tmp_path / role, device="cpu"
    )
    trainer = AlternatingGRPOTrainer(
        AlternatingTrainingConfig(
            policy("proposer"), policy("solver"), tmp_path / "training_state.json"
        ),
        trainer_factory=lambda role, config, seed: MockPolicyTrainer(role, config, seed=seed),
        before_role_callback=events.append,
    )
    proposer = TrainingSample("proposal", "task", (1, 2), (0, 1), 1.0, 0.5)
    incomplete = TrainingSample(
        "task-r0",
        "task",
        (1, 2),
        (0, 1),
        1.0,
        0.5,
        metadata={
            "rollout_group_complete": True,
            "rollout_group_size_expected": 2,
        },
    )

    with pytest.raises(UnsafeTrainingBatchError, match="same complete task groups"):
        trainer.train_cycle(
            TrainingBatch("proposer", (proposer,)),
            TrainingBatch("solver", (incomplete,)),
        )

    assert events == []
    assert not trainer.config.state_path.exists()


def test_training_gate_allows_short_final_mini_batch_with_complete_siblings(
    tmp_path,
) -> None:
    events: list[str] = []
    policy = lambda role: PolicyTrainingConfig(  # noqa: E731
        Path("/models/base"),
        tmp_path / role,
        device="cpu",
        mini_batch_size=64,
        micro_batch_size=1,
        gradient_accumulation_steps=64,
    )
    trainer = AlternatingGRPOTrainer(
        AlternatingTrainingConfig(
            policy("proposer"), policy("solver"), tmp_path / "training_state.json"
        ),
        trainer_factory=lambda role, config, seed: MockPolicyTrainer(role, config, seed=seed),
        before_role_callback=events.append,
    )
    proposer = TrainingSample("proposal", "task", (1, 2), (0, 1), 1.0, 0.5)
    solver = tuple(
        TrainingSample(
            f"task-r{index}",
            "task",
            (1, 2),
            (0, 1),
            1.0,
            0.5,
            metadata={
                "rollout_group_complete": True,
                "rollout_group_size_expected": 2,
            },
        )
        for index in range(2)
    )

    updates = trainer.train_cycle(
        TrainingBatch("proposer", (proposer,)),
        TrainingBatch("solver", solver),
    )

    assert [update.optimizer_steps for update in updates] == [1, 1]
    assert events == ["proposer", "solver"]
    assert trainer.config.state_path.exists()


def test_mock_trainer_uses_logical_mini_batches_for_optimizer_steps(tmp_path) -> None:
    config = PolicyTrainingConfig(
        Path("/models/base"),
        tmp_path / "solver",
        mini_batch_size=64,
        micro_batch_size=1,
        gradient_accumulation_steps=64,
    )
    samples = tuple(
        TrainingSample(f"r-{index}", f"task-{index}", (1, 2), (0, 1), 1.0, 0.5)
        for index in range(130)
    )

    result = MockPolicyTrainer("solver", config).update(TrainingBatch("solver", samples), step=1)

    assert result.samples == 130
    assert result.optimizer_steps == 3


def test_real_dynamic_micro_batch_matches_single_sample_accumulation(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class TinyCausalLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.projection = torch.nn.Linear(8, 32)

        def forward(self, *, input_ids, attention_mask=None):
            del attention_mask
            return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class Scheduler:
        def __init__(self, optimizer) -> None:
            self.optimizer = optimizer
            self.last_epoch = 0

        def step(self) -> None:
            self.last_epoch += 1

        def get_last_lr(self):
            return [self.optimizer.param_groups[0]["lr"]]

    def make_trainer(micro: int, name: str, state_dict) -> TransformersGRPOTrainer:
        config = PolicyTrainingConfig(
            Path("/models/tiny"),
            tmp_path / name,
            learning_rate=1e-2,
            mini_batch_size=4,
            micro_batch_size=micro,
            gradient_accumulation_steps=4 // micro,
            max_micro_batch_tokens=64,
            max_sequence_length=16,
            device="cpu",
            use_lora=False,
            gradient_checkpointing=False,
            kl_coefficient=0.0,
            entropy_coefficient=0.01,
        )
        trainer = object.__new__(TransformersGRPOTrainer)
        trainer.role = "solver"
        trainer.config = config
        trainer.seed = 0
        trainer.torch = torch
        trainer.model = TinyCausalLM()
        trainer.model.load_state_dict(state_dict)
        trainer.tokenizer = Tokenizer()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-2)
        trainer.scheduler = Scheduler(trainer.optimizer)
        trainer.optimizer_step_count = 0
        checkpoint = tmp_path / name / "step-00000001"
        checkpoint.mkdir(parents=True)
        trainer._save = lambda step: checkpoint  # type: ignore[method-assign]
        return trainer

    torch.manual_seed(7)
    initial = TinyCausalLM().state_dict()
    samples = tuple(
        TrainingSample(
            f"r-{index}",
            f"task-{index}",
            tuple((value + index) % 31 + 1 for value in range(5 + index)),
            (0,) + (1,) * (4 + index),
            1.0,
            (-0.75, -0.25, 0.25, 0.75)[index],
        )
        for index in range(4)
    )
    batch = TrainingBatch("solver", samples)
    single = make_trainer(1, "single", initial)
    dynamic = make_trainer(4, "dynamic", initial)
    fallback = make_trainer(4, "fallback", initial)
    fallback_token_log_probs = fallback._token_log_probs_batch

    def force_micro_two(token_id_sequences, **kwargs):
        if kwargs.get("require_grad") and len(token_id_sequences) > 2:
            raise RuntimeError("CUDA out of memory: simulated dynamic downgrade")
        return fallback_token_log_probs(token_id_sequences, **kwargs)

    fallback._token_log_probs_batch = force_micro_two  # type: ignore[method-assign]

    single_result = single.update(batch, step=1)
    dynamic_result = dynamic.update(batch, step=1)
    fallback_result = fallback.update(batch, step=1)

    assert single_result.optimizer_steps == dynamic_result.optimizer_steps == 1
    assert single_result.loss == pytest.approx(dynamic_result.loss, abs=1e-6)
    assert dynamic_result.step_metrics[0]["microbatch_sizes"] == [4]
    assert single_result.step_metrics[0]["microbatch_sizes"] == [1, 1, 1, 1]
    assert fallback_result.step_metrics[0]["microbatch_sizes"] == [2, 2]
    for result in (single_result, dynamic_result, fallback_result):
        metrics = result.step_metrics[0]
        assert metrics["parameter_update_l2"] > 0
        assert metrics["applied_learning_rate"] == pytest.approx(1e-2)
        assert metrics["nonfinite_ratio_count"] == 0
        assert metrics["ratio_count"] > 0
        assert metrics["ratio_min"] <= metrics["ratio_p50"] <= metrics["ratio_max"]
    assert dynamic_result.step_metrics[0]["oom_retry_count"] == 0
    assert fallback_result.step_metrics[0]["oom_retry_count"] == 1
    for single_parameter, dynamic_parameter in zip(
        single.model.parameters(), dynamic.model.parameters(), strict=True
    ):
        assert torch.allclose(single_parameter, dynamic_parameter, atol=1e-6, rtol=1e-6)
    for single_parameter, fallback_parameter in zip(
        single.model.parameters(), fallback.model.parameters(), strict=True
    ):
        assert torch.allclose(single_parameter, fallback_parameter, atol=1e-6, rtol=1e-6)


def test_training_gate_rejects_unsafe_terminal_rollout_before_optimizer(tmp_path) -> None:
    events: list[str] = []
    policy = lambda role: PolicyTrainingConfig(  # noqa: E731
        Path("/models/base"), tmp_path / role, device="cpu"
    )
    trainer = AlternatingGRPOTrainer(
        AlternatingTrainingConfig(
            policy("proposer"), policy("solver"), tmp_path / "training_state.json"
        ),
        trainer_factory=lambda role, config, seed: MockPolicyTrainer(role, config, seed=seed),
        before_role_callback=events.append,
    )
    proposer = TrainingSample("proposal", "task", (1, 2), (0, 1), 1.0, 0.5)
    solver = tuple(
        TrainingSample(
            f"task-r{index}",
            "task",
            (1, 2),
            (0, 1),
            -1.0,
            -0.5,
            metadata={
                "rollout_group_complete": True,
                "rollout_group_size_expected": 2,
                "training_eligible": index == 1,
                "terminal_graph_status": ("valid_finished" if index == 1 else "unsafe_partial"),
            },
        )
        for index in range(2)
    )

    with pytest.raises(UnsafeTrainingBatchError, match="training-ineligible"):
        trainer.train_cycle(
            TrainingBatch("proposer", (proposer,)),
            TrainingBatch("solver", solver),
        )

    assert events == []
    assert not trainer.config.state_path.exists()


def test_alternating_trainer_resumes_at_solver_without_repeating_proposer(tmp_path) -> None:
    attempts: list[str] = []
    failed = False

    class FailSolverOnce(MockPolicyTrainer):
        def update(self, batch, *, step, relation_credits=()):
            nonlocal failed
            attempts.append(self.role)
            if self.role == "solver" and not failed:
                failed = True
                raise RuntimeError("simulated solver interruption")
            return super().update(
                batch,
                step=step,
                relation_credits=relation_credits,
            )

    policy = lambda role: PolicyTrainingConfig(  # noqa: E731
        Path("/models/base"), tmp_path / role, device="cpu"
    )
    config = AlternatingTrainingConfig(
        policy("proposer"), policy("solver"), tmp_path / "training_state.json"
    )
    sample = TrainingSample("rollout", "task", (1, 2), (0, 1), 1.0, 0.5)
    proposer_batch = TrainingBatch("proposer", (sample,))
    solver_batch = TrainingBatch("solver", (sample,))
    trainer = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy_config, seed: FailSolverOnce(
            role, policy_config, seed=seed
        ),
    )
    try:
        trainer.train_cycle(proposer_batch, solver_batch)
    except RuntimeError as exc:
        assert "simulated solver interruption" in str(exc)
    else:
        raise AssertionError("solver interruption was not raised")
    interrupted = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert interrupted["phase"] == "solver"
    assert interrupted["global_step"] == 1

    resumed = AlternatingGRPOTrainer(
        config,
        trainer_factory=lambda role, policy_config, seed: FailSolverOnce(
            role, policy_config, seed=seed
        ),
    )
    proposer, solver = resumed.train_cycle(proposer_batch, solver_batch)
    assert (proposer.step, solver.step) == (1, 2)
    assert attempts == ["proposer", "solver", "solver"]
    completed = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert completed["cycle"] == 1
    assert completed["phase"] == "proposer"


def test_role_cuda_device_is_bound_before_model_initialization(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    from selfplay_graph_flowsteer import training

    events = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)

    def stop_after_load(*args, **kwargs):
        events.append("model_load")
        raise RuntimeError("initialization_order_probe")

    monkeypatch.setattr(training, "load_training_model", stop_after_load)
    config = PolicyTrainingConfig(tmp_path / "base", tmp_path / "checkpoints", device="cuda:3")
    with pytest.raises(RuntimeError, match="initialization_order_probe"):
        training.TransformersGRPOTrainer("solver", config)
    assert events == ["cuda:3", "model_load"]
