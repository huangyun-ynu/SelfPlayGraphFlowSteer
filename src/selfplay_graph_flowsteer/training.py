from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .counterfactual import RelationCredit
from .qwen_compat import load_training_model
from .rollouts import (
    TokenizedDirectorTrajectory,
    TokenizedPolicyCall,
    TrainingBatch,
    TrainingSample,
)


def _policy_call_sha256(call: TokenizedPolicyCall) -> str:
    payload = {
        "call_id": call.call_id,
        "token_ids": list(call.token_ids),
        "action_mask": list(call.action_mask),
        "relation_token_span": call.relation_token_span,
        "relation_policy": call.metadata.get("relation_policy"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PolicyTrainingConfig:
    base_model_path: Path
    checkpoint_root: Path
    learning_rate: float = 1e-5
    epochs: int = 1
    clip_range: float = 0.2
    kl_coefficient: float = 0.005
    entropy_coefficient: float = 0.0
    loss_variant: str = "kl005_entropy0"
    relation_credit_weight: float = 1.0
    # SESA/verl separates the logical PPO mini-batch from the device micro-batch.
    # Keeping the defaults at one preserves the small standalone trainer API; the
    # experiment CLI supplies the SESA-style 64/1 recipe.
    mini_batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_micro_batch_tokens: int = 16_384
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 10
    warmup_start_factor: float = 0.0
    total_optimizer_steps: int = 300
    learning_rate_schedule: str = "cosine"
    max_sequence_length: int = 4096
    dtype: str = "bfloat16"
    device: str = "cuda"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    gradient_checkpointing: bool = True
    # Optional memory-only execution policy.  Saved autograd tensors (including
    # tensors produced by gradient-checkpoint recomputation) are staged on CPU;
    # the objective, sample set, and optimizer step are unchanged.
    activation_cpu_offload: bool = False
    # Zero preserves all-call offload.  A positive threshold stages only
    # micro-batches containing at least one sequence of this length, allowing
    # short calls to remain entirely on GPU while protecting true long-call
    # peaks.
    activation_cpu_offload_min_tokens: int = 0
    # Release raw-call graphs promptly; logical optimizer boundaries are unchanged.
    raw_policy_backward_mode: str = "micro"
    # Parallel role training writes checkpoints first and publishes the two role
    # pointers only after both updates succeed.
    publish_latest_checkpoint: bool = True
    data_parallel_gpu_ids: tuple[int, ...] = ()
    # Optional frozen-snapshot probabilities computed while Solver collection is
    # still running.  Both fields are required together; the binding prevents a
    # cache from being reused after a checkpoint/tokenizer change.
    probability_cache_path: Path | None = None
    probability_cache_binding: str = ""
    # Short-call batching remains an explicit acceptance gate.  It is wired into
    # configuration and manifests but is not enabled by the formal launcher until
    # the long-context numerical gate has passed.
    short_call_batching: bool = False

    def validate(self) -> None:
        if self.learning_rate_schedule not in {"cosine", "constant"}:
            raise ValueError("learning_rate_schedule must be cosine or constant")
        if self.data_parallel_gpu_ids and (
            len(self.data_parallel_gpu_ids) != 2
            or len(set(self.data_parallel_gpu_ids)) != 2
            or min(self.data_parallel_gpu_ids) < 0
        ):
            raise ValueError("Solver data parallel requires two distinct physical GPUs")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        if not 0 < self.clip_range < 1:
            raise ValueError("clip_range must be in (0, 1)")
        if (
            min(
                self.mini_batch_size,
                self.micro_batch_size,
                self.gradient_accumulation_steps,
            )
            <= 0
        ):
            raise ValueError("mini-batch, micro-batch, and gradient accumulation must be positive")
        if self.micro_batch_size > self.mini_batch_size:
            raise ValueError("micro_batch_size may not exceed mini_batch_size")
        if self.max_micro_batch_tokens <= 0:
            raise ValueError("max_micro_batch_tokens must be positive")
        if self.max_micro_batch_tokens < self.max_sequence_length:
            raise ValueError(
                "max_micro_batch_tokens must fit at least one max_sequence_length sample"
            )
        if self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("weight_decay and warmup_steps must be non-negative")
        if not math.isfinite(self.warmup_start_factor) or not 0 <= self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be finite and in [0, 1]")
        if self.total_optimizer_steps <= 0 or self.max_sequence_length < 2:
            raise ValueError("total_optimizer_steps must be positive and sequence length >= 2")
        if self.activation_cpu_offload_min_tokens < 0:
            raise ValueError("activation_cpu_offload_min_tokens must be non-negative")
        if self.raw_policy_backward_mode not in ("micro", "call", "timeline"):
            raise ValueError("raw_policy_backward_mode must be micro, call or timeline")
        if bool(self.probability_cache_path) != bool(self.probability_cache_binding):
            raise ValueError("probability cache path and binding must be configured together")
        if min(self.kl_coefficient, self.entropy_coefficient, self.relation_credit_weight) < 0:
            raise ValueError("KL, entropy, and relation-credit coefficients must be non-negative")
        if self.use_lora and (self.lora_rank <= 0 or self.lora_alpha <= 0):
            raise ValueError("LoRA rank and alpha must be positive")
        if self.kl_coefficient > 0 and not self.use_lora:
            raise ValueError(
                "full fine-tuning with KL requires an explicit frozen reference checkpoint; "
                "the local trainer currently supports the immutable reference via LoRA only"
            )


@dataclass(frozen=True)
class AlternatingTrainingConfig:
    proposer: PolicyTrainingConfig
    solver: PolicyTrainingConfig
    state_path: Path
    seed: int = 0
    parallel_roles: bool = False

    def validate(self) -> None:
        self.proposer.validate()
        self.solver.validate()
        if self.proposer.data_parallel_gpu_ids:
            raise ValueError("data parallel is supported for Solver only")
        if self.parallel_roles and self.solver.data_parallel_gpu_ids:
            raise ValueError("two-card Solver requires sequential role updates")

        def batch_shape(policy: PolicyTrainingConfig) -> tuple[int, int, int]:
            return (
                policy.mini_batch_size,
                policy.micro_batch_size,
                policy.gradient_accumulation_steps,
                policy.max_micro_batch_tokens,
            )

        if batch_shape(self.proposer) != batch_shape(self.solver):
            raise ValueError(
                "proposer and solver must use the same mini-batch, micro-batch, and "
                "gradient-accumulation configuration; they consume different role batches, "
                "not different batch configurations"
            )
        if self.parallel_roles and self.proposer.device == self.solver.device:
            raise ValueError("parallel role training requires distinct proposer and solver devices")


@dataclass(frozen=True)
class PolicyUpdateResult:
    role: str
    step: int
    loss: float
    policy_loss: float
    kl: float
    masked_tokens: int
    checkpoint: str
    entropy: float = 0.0
    samples: int = 0
    optimizer_steps: int = 0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    clip_fraction: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    duration_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    step_metrics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    kl_loss_contribution: float = 0.0
    entropy_loss_contribution: float = 0.0
    loss_variant: str = ""
    reference_provenance: str = ""
    status: str = "updated"
    skip_reason: str = ""
    cycle: int | None = None


@dataclass
class AlternatingTrainingState:
    cycle: int = 0
    global_step: int = 0
    proposer_step: int = 0
    solver_step: int = 0
    proposer_checkpoint: str = ""
    solver_checkpoint: str = ""
    phase: str = "proposer"
    history: list[dict[str, Any]] = field(default_factory=list)
    active_batch_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AlternatingTrainingState:
        return cls(**payload)


class PolicyTrainer(Protocol):
    def update(
        self,
        batch: TrainingBatch,
        *,
        step: int,
        relation_credits: Iterable[RelationCredit] = (),
    ) -> PolicyUpdateResult: ...

    def close(self) -> None: ...


class UnsafeTrainingBatchError(ValueError):
    """Training was stopped before either optimizer because the batch is unsafe."""


def training_device_for_gpu(physical_gpu_id: int, *, visible_devices: str | None = None) -> str:
    """Resolve a configured physical GPU to its process-local CUDA index."""

    if physical_gpu_id < 0:
        raise ValueError("physical_gpu_id must be non-negative")
    value = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if value is None or not value.strip():
        return f"cuda:{physical_gpu_id}"
    try:
        visible = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain numeric GPU ids") from exc
    if physical_gpu_id not in visible:
        raise ValueError(
            f"configured physical GPU {physical_gpu_id} is not visible in CUDA_VISIBLE_DEVICES={value}"
        )
    return f"cuda:{visible.index(physical_gpu_id)}"


def training_batch_from_dict(payload: dict[str, Any]) -> TrainingBatch:
    samples = tuple(
        TrainingSample(
            rollout_id=str(item["rollout_id"]),
            task_id=str(item["task_id"]),
            token_ids=tuple(int(value) for value in item["token_ids"]),
            action_mask=tuple(int(value) for value in item["action_mask"]),
            reward=float(item["reward"]),
            advantage=float(item["advantage"]),
            density=float(item.get("density", 1.0)),
            canonical_graph_key=str(item.get("canonical_graph_key", "")),
            graph_features=tuple(float(value) for value in item.get("graph_features", [])),
            metadata=dict(item.get("metadata", {})),
            policy_calls=tuple(
                TokenizedPolicyCall(
                    call_id=str(call["call_id"]),
                    token_ids=tuple(int(value) for value in call["token_ids"]),
                    action_mask=tuple(int(value) for value in call["action_mask"]),
                    behavior_log_probs=tuple(
                        float(value) for value in call.get("behavior_log_probs", ())
                    ),
                    action_token_span=(
                        tuple(call["action_token_span"])
                        if call.get("action_token_span") is not None
                        else None
                    ),
                    relation_token_span=(
                        tuple(call["relation_token_span"])
                        if call.get("relation_token_span") is not None
                        else None
                    ),
                    metadata=dict(call.get("metadata", {})),
                )
                for call in item.get("policy_calls", ())
            ),
        )
        for item in payload.get("samples", [])
    )
    return TrainingBatch(
        role=str(payload["role"]),
        samples=samples,
        objective=str(payload.get("objective", "masked_graph_level_grpo")),
        optimizer_steps=0,
        metadata=dict(payload.get("metadata", {})),
    )


def load_training_batch(path: str | Path) -> TrainingBatch:
    return training_batch_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_relation_credits(path: str | Path | None) -> list[RelationCredit]:
    if path is None or not Path(path).exists():
        return []
    credits: list[RelationCredit] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("action_token_span") is not None:
                payload["action_token_span"] = tuple(payload["action_token_span"])
            credits.append(RelationCredit(**payload))
    return credits


def token_advantages(
    sample: TrainingSample,
    relation_credits: Iterable[RelationCredit] = (),
    *,
    relation_weight: float = 1.0,
) -> tuple[float, ...]:
    """Use graph credit globally and replace probed relation spans with local credit."""

    values = [float(sample.advantage)] * len(sample.token_ids)
    for credit in relation_credits:
        if credit.rollout_id != sample.rollout_id or credit.action_token_span is None:
            continue
        start, end = credit.action_token_span
        local = credit.advantage_present if credit.chosen_present else credit.advantage_absent
        for index in range(max(0, start), min(len(values), end)):
            if sample.action_mask[index]:
                values[index] = relation_weight * local
    return tuple(values)


def _relation_call_advantage(
    sample: TrainingSample,
    credits: Iterable[RelationCredit],
    call_index: int,
    relation_weight: float,
) -> float:
    for credit in credits:
        if credit.rollout_id != sample.rollout_id or credit.action_index != call_index:
            continue
        local = credit.advantage_present if credit.chosen_present else credit.advantage_absent
        return float(local) * float(relation_weight)
    return float(sample.advantage)


def _sampled_kl_terms(current_log_probs: Any, reference_log_probs: Any) -> Any:
    """Compute the sampled KL estimator in fp32 to avoid bf16 cancellation."""

    log_ratio = (reference_log_probs.float() - current_log_probs.float()).clamp(-20.0, 20.0)
    return log_ratio.exp() - log_ratio - 1.0


def _categorical_kl(current_log_probs: Any, reference_log_probs: Any) -> Any:
    """Compute exact finite-support KL in fp32 even when model logits are bf16."""

    current = current_log_probs.float()
    reference = reference_log_probs.float()
    return (current.exp() * (current - reference)).sum()


def _policy_identity_failed(
    *, max_delta: float, mean_delta: float, p95_delta: float, relation_call: bool
) -> bool:
    """Apply empirically calibrated vLLM/Transformers identity tolerances."""

    if relation_call:
        return max_delta > 0.05
    # Long Qwen/GDN calls can contain an isolated recurrent-kernel outlier even
    # when the exact provider prompt/completion IDs are replayed.  A maximum is
    # therefore useful telemetry, but is not a robust identity statistic for an
    # ordinary multi-token call.  Wrong/stale prefixes shift a material part of
    # the distribution and remain fail-closed under the mean and p95 bounds.
    return mean_delta > 0.03 or p95_delta > 0.10


def _binary_policy_identity_failed(current: Any, expected: Any) -> bool:
    """Bound the entire two-action distribution under mixed-engine bf16 replay.

    Qwen/GDN cache/prefill kernels are not bitwise identical across vLLM and
    Transformers. Real frozen-snapshot replay exhibited a sampled log-prob
    delta of .053 and a probability displacement of .040. Require BOTH
    actions to remain close, plus a sampled/support likelihood-ratio bound;
    never replace the rollout behavior probabilities with replay values.
    """
    current = current.float()
    expected = expected.float()
    if not bool(current.isfinite().all() and expected.isfinite().all()):
        return True
    current_p, expected_p = current.exp(), expected.exp()
    if abs(float(current_p.sum()) - 1.0) > 1e-5 or abs(float(expected_p.sum()) - 1.0) > 1e-5:
        return True
    total_variation = float((current_p - expected_p).abs().sum() / 2)
    kl = float((expected_p * (expected - current)).sum())
    # Full-support log-ratio stays within the PPO clip's log(1.2) scale.
    return (
        total_variation > 0.05
        or kl > 0.005
        or float((current - expected).abs().max()) > math.log(1.2)
    )


def _initial_ratio_diagnostics(
    log_ratios: list[float], *, clip_range: float, staleness_updates: int
) -> dict[str, Any]:
    """Summarize pre-update drift without inventing an uncalibrated reject threshold."""

    if not log_ratios:
        return {
            "staleness_updates": staleness_updates,
            "initial_ratio_tokens": 0,
            "initial_ratio_clip_fraction": 0.0,
        }
    ordered = sorted(float(value) for value in log_ratios)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    lower_log = math.log(1.0 - clip_range)
    upper_log = math.log(1.0 + clip_range)
    return {
        "staleness_updates": staleness_updates,
        "initial_ratio_tokens": len(ordered),
        "initial_log_ratio_mean": statistics.fmean(ordered),
        "initial_log_ratio_p50": quantile(0.50),
        "initial_log_ratio_p90": quantile(0.90),
        "initial_log_ratio_p95": quantile(0.95),
        "initial_log_ratio_abs_max": max(abs(ordered[0]), abs(ordered[-1])),
        "initial_ratio_clip_fraction": sum(
            value < lower_log or value > upper_log for value in ordered
        )
        / len(ordered),
    }


def _policy_target_count(sample: TrainingSample) -> int:
    """Count next-token policy targets from the durable rollout masks."""

    calls = sample.policy_calls or (
        TokenizedPolicyCall(
            call_id=f"{sample.rollout_id}:legacy",
            token_ids=sample.token_ids,
            action_mask=sample.action_mask,
        ),
    )
    return sum(sum(bool(value) for value in call.action_mask[1:]) for call in calls)


def _unit_policy_mask(log_probs: Any, torch_module: Any) -> Any:
    """Build an exact FP32 token mask even when policy logits use bfloat16."""

    return torch_module.ones(
        log_probs.shape,
        dtype=torch_module.float32,
        device=log_probs.device,
    )


def dynamic_micro_batch_indices(
    samples: tuple[TrainingSample, ...] | list[TrainingSample],
    *,
    max_batch_size: int,
    max_padded_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Pack samples stably under both a count and padded-token ceiling.

    Memory is governed by ``batch_size * max_sequence_length`` after padding,
    not by the unpadded sum. Raw trajectories use their longest actual policy
    call, not the concatenated completion-only audit sequence. Raw calls are
    evaluated separately; this is a packing heuristic, not a peak-memory
    guarantee (activation offload and OOM backoff still apply).
    Keeping input order stable also makes comparisons
    with the historical one-sample accumulation path reproducible.
    """

    if max_batch_size <= 0 or max_padded_tokens <= 0:
        raise ValueError("dynamic micro-batch limits must be positive")
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_max_length = 0
    for index, sample in enumerate(samples):
        sample_length = (
            max(len(call.token_ids) for call in sample.policy_calls)
            if sample.policy_calls
            else len(sample.token_ids)
        )
        if sample_length <= 0:
            raise ValueError(f"rollout {sample.rollout_id} has no tokens")
        if sample_length > max_padded_tokens:
            raise ValueError(
                f"rollout {sample.rollout_id} has {sample_length} tokens; "
                f"micro-batch token ceiling is {max_padded_tokens}"
            )
        candidate_max = max(current_max_length, sample_length)
        candidate_size = len(current) + 1
        if current and (
            candidate_size > max_batch_size or candidate_size * candidate_max > max_padded_tokens
        ):
            batches.append(tuple(current))
            current = []
            current_max_length = 0
            candidate_max = sample_length
        current.append(index)
        current_max_length = candidate_max
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def logical_mini_batch_size(batch: TrainingBatch, config: PolicyTrainingConfig) -> int:
    """Partial-group Solver uses the global trajectory mean, never an overweight tail."""
    if batch.role == "solver" and batch.metadata.get("training_selection_schema") in {
        "eligible_subset_v1",
        "independent_frontier_v2",
    }:
        return max(1, len(batch.samples))
    return config.mini_batch_size


class TransformersGRPOTrainer:
    """FlowSteer masked loss with SESA-style independent policy checkpoints."""

    def __init__(self, role: str, config: PolicyTrainingConfig, *, seed: int = 0) -> None:
        config.validate()
        self.role = role
        self.config = config
        self.seed = int(seed)
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'train' extra for parameter training") from exc
        self.torch = torch
        # Model/kernel initialization may consult the thread's current CUDA
        # device before model.to(). Bind it to the allocated role GPU rather
        # than accidentally creating a context on the default physical GPU 0.
        if str(config.device).startswith("cuda"):
            torch.cuda.set_device(config.device)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        latest = self._latest_checkpoint()
        source = config.base_model_path if config.use_lora else (latest or config.base_model_path)
        dtype = getattr(torch, config.dtype)
        self.model = load_training_model(source, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(latest or config.base_model_path), trust_remote_code=True
        )
        if config.use_lora:
            try:
                from peft import LoraConfig, PeftModel, get_peft_model
            except ImportError as exc:
                raise RuntimeError("PEFT is required when use_lora=true") from exc
            if latest:
                self.model = PeftModel.from_pretrained(self.model, str(latest), is_trainable=True)
            else:
                self.model = get_peft_model(
                    self.model,
                    LoraConfig(
                        r=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                        task_type="CAUSAL_LM",
                        target_modules=list(config.lora_target_modules),
                    ),
                )
        if config.gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            if config.use_lora and hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.to(config.device)
        if role == "solver" and config.raw_policy_backward_mode == "timeline":
            from .policy_timeline import stabilize_gated_delta_projections

            stabilize_gated_delta_projections(self.model)
            if config.dtype == "float32":
                from .policy_timeline import enable_expanded_fp32_sdpa

                enable_expanded_fp32_sdpa(self.model)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = _build_learning_rate_scheduler(self.optimizer, config)
        self.optimizer_step_count = 0
        self._restore_optimizer(latest or source)
        self._active_cuda_rng_state = (
            torch.cuda.get_rng_state(config.device)
            if torch.cuda.is_available() and str(config.device).startswith("cuda")
            else None
        )

    def activate_rng(self) -> None:
        """Restore this role's device RNG after the other role is initialized."""

        if self._active_cuda_rng_state is not None:
            self.torch.cuda.set_rng_state(self._active_cuda_rng_state, self.config.device)

    def precompute_probability_record(
        self, trajectory: TokenizedDirectorTrajectory
    ) -> dict[str, Any]:
        """Evaluate immutable current/reference probabilities without an update."""

        if not self.config.probability_cache_binding:
            raise ValueError("probability cache binding is required for precomputation")
        self.model.eval()
        batched: dict[int, tuple[Any, Any]] = {}
        if self.config.short_call_batching and trajectory.policy_calls:
            from .experimental_call_batching import evaluate_calls, plan_call_batches

            sample = TrainingSample(
                rollout_id=trajectory.rollout_id,
                task_id=trajectory.task_id,
                token_ids=trajectory.token_ids,
                action_mask=trajectory.action_mask,
                reward=trajectory.reward,
                advantage=0.0,
                policy_calls=trajectory.policy_calls,
            )
            for batch in plan_call_batches(
                (sample,),
                max_calls=max(1, self.config.micro_batch_size),
                short_call_tokens=min(4096, self.config.max_sequence_length),
                max_padded_tokens=self.config.max_micro_batch_tokens,
            ):
                if batch[0].call.relation_token_span is not None:
                    continue
                calls = [item.call for item in batch]
                current_rows = evaluate_calls(self, calls, include_entropy=False)
                reference_rows = (
                    evaluate_calls(self, calls, reference=True, include_entropy=False)
                    if self.config.kl_coefficient
                    else current_rows
                )
                for item, current_row, reference_row in zip(
                    batch, current_rows, reference_rows, strict=True
                ):
                    batched[item.call_index] = (current_row[0], reference_row[0])
        with self.torch.no_grad():
            timeline_current = (
                self._timeline_policy_outputs(trajectory.policy_calls, include_entropy=False)
                if self.config.raw_policy_backward_mode == "timeline"
                else None
            )
            timeline_reference = (
                self._timeline_policy_outputs(
                    trajectory.policy_calls, reference=True, include_entropy=False
                )
                if timeline_current is not None
                and (
                    self.config.kl_coefficient
                    or any(
                        c.relation_token_span is not None and c.metadata.get("relation_policy")
                        for c in trajectory.policy_calls
                    )
                )
                else timeline_current
            )
        rows = []
        for call_index, call in enumerate(trajectory.policy_calls):
            relation_policy = call.metadata.get("relation_policy", {})
            if timeline_current is not None:
                binary = call.relation_token_span is not None and relation_policy
                current = (
                    timeline_current[call_index] if binary else timeline_current[call_index][0]
                )
                reference = (
                    timeline_reference[call_index] if binary else timeline_reference[call_index][0]
                )
                kind = "binary_relation" if binary else "ordinary"
            elif call.relation_token_span is not None and relation_policy:
                current = self._binary_choice_log_probs(call, relation_policy)
                reference = self._binary_choice_log_probs(call, relation_policy, reference=True)
                kind = "binary_relation"
            else:
                cached_batch = batched.get(call_index)
                if cached_batch is not None:
                    current, reference = cached_batch
                else:
                    current = self._masked_policy_call_log_probs(call)
                    reference = (
                        self._masked_policy_call_log_probs(call, reference=True)
                        if self.config.kl_coefficient
                        else current.clone()
                    )
                kind = "ordinary"
            rows.append(
                {
                    "call_id": call.call_id,
                    "call_sha256": _policy_call_sha256(call),
                    "kind": kind,
                    "current": [float(value) for value in current.float().cpu().tolist()],
                    "reference": [float(value) for value in reference.float().cpu().tolist()],
                }
            )
        return {
            "schema_version": "solver_probability_cache_v1",
            "binding": self.config.probability_cache_binding,
            "rollout_id": trajectory.rollout_id,
            "calls": rows,
        }

    def _load_probability_cache(self) -> dict[tuple[str, str], dict[str, Any]]:
        path = self.config.probability_cache_path
        if path is None:
            return {}
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "solver_probability_cache_v1":
                raise ValueError("unknown Solver probability cache schema")
            if row.get("binding") != self.config.probability_cache_binding:
                raise ValueError("Solver probability cache binding does not match update snapshot")
            rollout_id = str(row.get("rollout_id", ""))
            for call in row.get("calls", ()):
                key = (rollout_id, str(call.get("call_id", "")))
                if key in records:
                    raise ValueError(f"duplicate Solver probability cache entry: {key}")
                records[key] = call
        return records

    def _cached_call_tensor(
        self,
        cache: dict[tuple[str, str], dict[str, Any]],
        rollout_id: str,
        call: TokenizedPolicyCall,
        field: str,
    ) -> Any:
        key = (rollout_id, call.call_id)
        try:
            row = cache[key]
        except KeyError as exc:
            raise ValueError(f"missing Solver probability cache entry: {key}") from exc
        if row.get("call_sha256") != _policy_call_sha256(call):
            raise ValueError(f"Solver probability cache call hash changed: {key}")
        values = row.get(field)
        if not isinstance(values, list):
            raise ValueError(f"Solver probability cache lacks {field}: {key}")
        return self.torch.tensor(values, dtype=self.torch.float32)

    def update(
        self,
        batch: TrainingBatch,
        *,
        step: int,
        relation_credits: Iterable[RelationCredit] = (),
    ) -> PolicyUpdateResult:
        started = time.perf_counter()
        if batch.role != self.role:
            raise ValueError(f"expected {self.role} batch, got {batch.role}")
        torch = self.torch
        cuda_metrics = torch.cuda.is_available() and str(self.config.device).startswith("cuda")
        if cuda_metrics:
            torch.cuda.reset_peak_memory_stats(self.config.device)
        credits = list(relation_credits)
        if not batch.samples:
            raise ValueError(f"cannot update {self.role} from an empty training batch")
        for sample in batch.samples:
            sequences = sample.policy_calls or (
                TokenizedPolicyCall(
                    call_id=f"{sample.rollout_id}:legacy",
                    token_ids=sample.token_ids,
                    action_mask=sample.action_mask,
                ),
            )
            if not sequences or not any(
                len(call.token_ids) >= 2 and any(call.action_mask[1:]) for call in sequences
            ):
                raise ValueError(f"rollout {sample.rollout_id} has no next-token action target")
            for call in sequences:
                if len(call.token_ids) > self.config.max_sequence_length:
                    raise ValueError(
                        f"policy call {call.call_id} has {len(call.token_ids)} tokens; "
                        f"limit is {self.config.max_sequence_length}"
                    )
                if sample.policy_calls and not call.behavior_log_probs:
                    raise ValueError(
                        f"raw policy call {call.call_id} lacks rollout-time behavior log-probs"
                    )
        probability_cache = self._load_probability_cache()
        lineage = batch.metadata.get("policy_lineage", {})
        staleness_updates = int(lineage.get("staleness_updates", 0)) if lineage else 0
        if staleness_updates not in {0, 1}:
            raise ValueError("policy batch staleness must be zero or one update")
        if staleness_updates and probability_cache:
            raise ValueError(
                "a stale behavior probability cache cannot supply learner-current probabilities"
            )
        self.model.eval()
        old_log_probs: list[list[Any]] = []
        reference_log_probs: list[list[Any]] = []
        for sample in batch.samples:
            calls = sample.policy_calls or (
                TokenizedPolicyCall(
                    call_id=f"{sample.rollout_id}:legacy",
                    token_ids=sample.token_ids,
                    action_mask=sample.action_mask,
                ),
            )
            if sample.policy_calls:
                old_log_probs.append(
                    [
                        torch.tensor(
                            [
                                value
                                for value, keep in zip(
                                    call.behavior_log_probs,
                                    call.action_mask[1:],
                                    strict=True,
                                )
                                if keep
                            ],
                            dtype=torch.float32,
                        )
                        for call in calls
                    ]
                )
            else:
                old_log_probs.append([self._token_log_probs(call.token_ids) for call in calls])
            timeline_reference = (
                self._timeline_policy_outputs(
                    sample.policy_calls, reference=True, include_entropy=False
                )
                if sample.policy_calls
                and self.config.raw_policy_backward_mode == "timeline"
                and self.config.kl_coefficient
                and not probability_cache
                else None
            )
            if self.config.kl_coefficient:
                if timeline_reference is not None:
                    reference_log_probs.append(
                        [
                            (
                                value
                                if call.relation_token_span is not None
                                and call.metadata.get("relation_policy")
                                else value[0]
                            )
                            .detach()
                            .cpu()
                            for call, value in zip(
                                sample.policy_calls, timeline_reference, strict=True
                            )
                        ]
                    )
                elif sample.policy_calls:
                    reference_log_probs.append(
                        [
                            self._cached_call_tensor(
                                probability_cache, sample.rollout_id, call, "reference"
                            )
                            if probability_cache
                            else (
                                self._binary_choice_log_probs(
                                    call,
                                    call.metadata["relation_policy"],
                                    reference=True,
                                )
                                if call.relation_token_span is not None
                                and call.metadata.get("relation_policy")
                                else self._masked_policy_call_log_probs(call, reference=True)
                            )
                            for call in calls
                        ]
                    )
                else:
                    reference_log_probs.append(
                        [self._token_log_probs(call.token_ids, reference=True) for call in calls]
                    )
            else:
                reference_log_probs.append([values.clone() for values in old_log_probs[-1]])
        initial_log_ratios: list[float] = []
        for sample_index, sample in enumerate(batch.samples):
            if not sample.policy_calls:
                continue
            with torch.no_grad():
                initial_timeline_outputs = (
                    self._timeline_policy_outputs(sample.policy_calls, include_entropy=False)
                    if self.config.raw_policy_backward_mode == "timeline" and not probability_cache
                    else None
                )
            for call_index, call in enumerate(sample.policy_calls):
                relation_policy = call.metadata.get("relation_policy", {})
                if call.relation_token_span is not None and relation_policy:
                    current = (
                        initial_timeline_outputs[call_index]
                        if initial_timeline_outputs is not None
                        else self._cached_call_tensor(
                            probability_cache, sample.rollout_id, call, "current"
                        ).to(self.config.device)
                        if probability_cache
                        else self._binary_choice_log_probs(call, relation_policy)
                    )
                    start, _end = call.relation_token_span
                    chosen_id = int(call.token_ids[start])
                    token_map = relation_policy.get("token_ids", {})
                    selected = 0 if chosen_id == int(token_map["off"]) else 1
                    observed = current[selected : selected + 1].cpu()
                    expected = old_log_probs[sample_index][call_index]
                else:
                    observed = (
                        initial_timeline_outputs[call_index][0].cpu()
                        if initial_timeline_outputs is not None
                        else self._cached_call_tensor(
                            probability_cache, sample.rollout_id, call, "current"
                        ).cpu()
                        if probability_cache
                        else self._masked_policy_call_log_probs(call).cpu()
                    )
                    expected = old_log_probs[sample_index][call_index]
                difference = (observed - expected).abs()
                log_ratio = observed.float() - expected.float()
                if observed.numel() and not bool(torch.isfinite(log_ratio).all()):
                    raise ValueError(
                        f"policy call {call.call_id} has non-finite initial PPO ratios"
                    )
                initial_log_ratios.extend(float(value) for value in log_ratio.tolist())
                max_delta = float(difference.max()) if observed.numel() else 0.0
                mean_delta = float(difference.mean()) if observed.numel() else 0.0
                p95_delta = (
                    float(torch.quantile(difference.float(), 0.95)) if observed.numel() else 0.0
                )
                # vLLM and Transformers use different Qwen/GDN kernels. On an
                # identical bf16 snapshot, isolated sampled tokens can differ
                # substantially while the distribution-level error remains
                # small. Gate ordinary calls by robust aggregate bounds. Binary
                # records with complete support use both probabilities below;
                # legacy records retain the strict sampled-token fallback.
                relation_call = bool(call.relation_token_span is not None and relation_policy)
                identity_failed = _policy_identity_failed(
                    max_delta=max_delta,
                    mean_delta=mean_delta,
                    p95_delta=p95_delta,
                    relation_call=relation_call,
                )
                if relation_call and "log_probabilities" in relation_policy:
                    expected_binary = torch.tensor(
                        [relation_policy["log_probabilities"][choice] for choice in ("off", "on")],
                        dtype=torch.float32,
                        device=current.device,
                    )
                    identity_failed = (
                        _binary_policy_identity_failed(current, expected_binary)
                        or abs(float(expected_binary[selected].cpu()) - float(expected[0])) > 1e-5
                    )
                if observed.numel() and identity_failed and staleness_updates == 0:
                    raise ValueError(
                        f"policy call {call.call_id} is not on-policy for the frozen update "
                        "snapshot (initial PPO ratio is not near one; "
                        f"max_abs_logprob_delta={max_delta:.6f}, "
                        f"mean_abs_logprob_delta={mean_delta:.6f}, "
                        f"p95_abs_logprob_delta={p95_delta:.6f})"
                    )
        initial_ratio_diagnostics = _initial_ratio_diagnostics(
            initial_log_ratios,
            clip_range=self.config.clip_range,
            staleness_updates=staleness_updates,
        )
        total_loss = total_policy = total_kl = total_entropy = 0.0
        masked_tokens = updates = clipped_tokens = optimizer_steps = 0
        gradient_norms: list[float] = []
        step_metrics: list[dict[str, Any]] = []
        self.model.train()
        logical_mini_size = logical_mini_batch_size(batch, self.config)
        for _epoch in range(self.config.epochs):
            # Keep all samples from a role update in one logical batch, then
            # perform PPO/GRPO updates over bounded mini-batches.  The final
            # mini-batch is intentionally allowed to be short: group integrity
            # is enforced above the trainer, not by padding with fake rollouts.
            for mini_start in range(0, len(batch.samples), logical_mini_size):
                mini_samples = batch.samples[mini_start : mini_start + logical_mini_size]
                mini_size = len(mini_samples)
                window_loss: list[float] = []
                window_policy: list[float] = []
                window_kl: list[float] = []
                window_entropy: list[float] = []
                window_rewards: list[float] = []
                window_masked = window_clipped = 0
                window_started = time.perf_counter()
                micro_size_limit = self.config.micro_batch_size
                oom_retries = 0
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(self.config.device) if cuda_metrics else None
                while True:
                    self.optimizer.zero_grad(set_to_none=True)
                    window_loss.clear()
                    window_policy.clear()
                    window_kl.clear()
                    window_entropy.clear()
                    window_rewards.clear()
                    window_masked = window_clipped = 0
                    ratio_values = []
                    micro_sizes: list[int] = []
                    try:
                        micro_batches = dynamic_micro_batch_indices(
                            mini_samples,
                            max_batch_size=micro_size_limit,
                            max_padded_tokens=self.config.max_micro_batch_tokens,
                        )
                        for micro_indices in micro_batches:
                            micro_max_tokens = max(
                                len(call.token_ids)
                                for index in micro_indices
                                for call in (
                                    mini_samples[index].policy_calls
                                    or (
                                        TokenizedPolicyCall(
                                            call_id=f"{mini_samples[index].rollout_id}:legacy",
                                            token_ids=mini_samples[index].token_ids,
                                            action_mask=mini_samples[index].action_mask,
                                        ),
                                    )
                                )
                            )
                            legacy_outputs: dict[int, tuple[Any, Any]] = {}
                            if all(not mini_samples[index].policy_calls for index in micro_indices):
                                batched = self._token_log_probs_batch(
                                    tuple(mini_samples[index].token_ids for index in micro_indices),
                                    require_grad=True,
                                    include_entropy=True,
                                    entropy_requires_grad=(self.config.entropy_coefficient > 0),
                                )
                                legacy_outputs = dict(zip(micro_indices, batched, strict=True))
                            micro_loss = None
                            for local_index in micro_indices:
                                sample = mini_samples[local_index]
                                index = mini_start + local_index
                                calls = sample.policy_calls or (
                                    TokenizedPolicyCall(
                                        call_id=f"{sample.rollout_id}:legacy",
                                        token_ids=sample.token_ids,
                                        action_mask=sample.action_mask,
                                    ),
                                )
                                policy_sum = kl_sum = entropy_sum = None
                                denominator = None
                                sample_clipped = 0
                                timeline_outputs = (
                                    self._timeline_policy_outputs(calls)
                                    if sample.policy_calls
                                    and self.config.raw_policy_backward_mode == "timeline"
                                    else None
                                )
                                stream_denominator = (
                                    sum(sum(c.action_mask[1:]) for c in calls)
                                    if sample.policy_calls
                                    and self.config.raw_policy_backward_mode in ("call", "timeline")
                                    and timeline_outputs is None
                                    else None
                                )
                                for call_index, call in enumerate(calls):
                                    relation_policy = call.metadata.get("relation_policy", {})
                                    if call.relation_token_span is not None and relation_policy:
                                        current_binary = (
                                            timeline_outputs[call_index]
                                            if timeline_outputs is not None
                                            else self._binary_choice_log_probs(
                                                call, relation_policy, require_grad=True
                                            )
                                        )
                                        reference_binary = (
                                            self._cached_call_tensor(
                                                probability_cache,
                                                sample.rollout_id,
                                                call,
                                                "reference",
                                            ).to(current_binary.device)
                                            if probability_cache
                                            else self._binary_choice_log_probs(
                                                call, relation_policy, reference=True
                                            )
                                        )
                                        start, end = call.relation_token_span
                                        if end - start != 1:
                                            raise ValueError(
                                                "relation policy span must be one token"
                                            )
                                        chosen_id = int(call.token_ids[start])
                                        token_map = relation_policy.get("token_ids", {})
                                        choices = ("off", "on")
                                        try:
                                            chosen_index = next(
                                                pos
                                                for pos, choice in enumerate(choices)
                                                if int(token_map[choice]) == chosen_id
                                            )
                                        except (
                                            KeyError,
                                            StopIteration,
                                            TypeError,
                                            ValueError,
                                        ) as exc:
                                            raise ValueError(
                                                "relation choice token does not match its binary support"
                                            ) from exc
                                        log_probs = current_binary[chosen_index].reshape(1)
                                        # For raw policy calls, old_log_probs is already
                                        # compacted to mask-1 targets. A binary relation has
                                        # exactly one such target regardless of its absolute
                                        # position in the provider prompt.
                                        old = old_log_probs[index][call_index][:1].to(
                                            log_probs.device
                                        )
                                        mask = _unit_policy_mask(log_probs, torch)
                                        local_advantage = _relation_call_advantage(
                                            sample,
                                            credits,
                                            call_index,
                                            self.config.relation_credit_weight,
                                        )
                                        advantages = torch.full_like(log_probs, local_advantage)
                                        exact_kl = _categorical_kl(current_binary, reference_binary)
                                        binary_entropy = -(
                                            current_binary.detach().float().exp()
                                            * current_binary.detach().float()
                                        ).sum()
                                        kl_terms = exact_kl.reshape(1)
                                        entropy_terms = binary_entropy.reshape(1)
                                    else:
                                        if not sample.policy_calls and call_index == 0:
                                            log_probs, token_entropy = legacy_outputs[local_index]
                                            mask = torch.tensor(
                                                call.action_mask[1 : 1 + log_probs.shape[0]],
                                                dtype=torch.float32,
                                                device=log_probs.device,
                                            )
                                        else:
                                            log_probs, token_entropy = (
                                                timeline_outputs[call_index]
                                                if timeline_outputs is not None
                                                else self._masked_policy_call_log_probs(
                                                    call,
                                                    require_grad=True,
                                                    include_entropy=True,
                                                    entropy_requires_grad=(
                                                        self.config.entropy_coefficient > 0
                                                    ),
                                                )
                                            )
                                            mask = _unit_policy_mask(log_probs, torch)
                                        advantages = torch.full_like(
                                            log_probs, float(sample.advantage)
                                        )
                                        old = old_log_probs[index][call_index][
                                            : log_probs.shape[0]
                                        ].to(log_probs.device)
                                        reference = reference_log_probs[index][call_index][
                                            : log_probs.shape[0]
                                        ].to(log_probs.device)
                                        kl_terms = _sampled_kl_terms(log_probs, reference) * mask
                                        entropy_terms = token_entropy * mask
                                    log_ratio = (log_probs - old).clamp(-20.0, 20.0)
                                    ratio = log_ratio.exp()
                                    ratio_values.append(ratio.detach()[mask.bool()].float().cpu())
                                    sample_clipped += int(
                                        ((ratio - 1.0).abs() > self.config.clip_range)
                                        .logical_and(mask.bool())
                                        .sum()
                                        .item()
                                    )
                                    unclipped = -advantages * ratio
                                    clipped = -advantages * ratio.clamp(
                                        1.0 - self.config.clip_range,
                                        1.0 + self.config.clip_range,
                                    )
                                    call_policy = (torch.maximum(unclipped, clipped) * mask).sum()
                                    call_kl = kl_terms.sum()
                                    call_entropy = entropy_terms.sum()
                                    call_denominator = mask.sum()
                                    if stream_denominator is not None:
                                        if stream_denominator <= 0:
                                            raise ValueError("raw rollout has no target tokens")
                                        contribution = (
                                            (
                                                call_policy
                                                + self.config.kl_coefficient * call_kl
                                                - self.config.entropy_coefficient * call_entropy
                                            )
                                            / stream_denominator
                                            / mini_size
                                        )
                                        call_context = (
                                            torch.autograd.graph.save_on_cpu(
                                                pin_memory=True, device_type="cuda"
                                            )
                                            if cuda_metrics
                                            and self._activation_cpu_offload_enabled(
                                                len(call.token_ids)
                                            )
                                            else nullcontext()
                                        )
                                        with call_context:
                                            contribution.backward()
                                        call_policy = call_policy.detach()
                                        call_kl = call_kl.detach()
                                        call_entropy = call_entropy.detach()
                                    policy_sum = (
                                        call_policy
                                        if policy_sum is None
                                        else policy_sum + call_policy
                                    )
                                    kl_sum = call_kl if kl_sum is None else kl_sum + call_kl
                                    entropy_sum = (
                                        call_entropy
                                        if entropy_sum is None
                                        else entropy_sum + call_entropy
                                    )
                                    denominator = (
                                        call_denominator
                                        if denominator is None
                                        else denominator + call_denominator
                                    )
                                if policy_sum is None or denominator is None:
                                    raise RuntimeError("rollout contained no policy calls")
                                denominator = denominator.clamp_min(1.0)
                                policy_loss = policy_sum / denominator
                                reference_kl = kl_sum / denominator
                                entropy = entropy_sum / denominator
                                loss = (
                                    policy_loss
                                    + self.config.kl_coefficient * reference_kl
                                    - self.config.entropy_coefficient * entropy
                                )
                                scaled = loss / mini_size
                                if timeline_outputs is not None:
                                    scaled.backward()
                                    scaled = scaled.detach()
                                    del timeline_outputs
                                micro_loss = scaled if micro_loss is None else micro_loss + scaled
                                sample_masked = int(denominator.item())
                                window_loss.append(float(loss.detach()))
                                window_policy.append(float(policy_loss.detach()))
                                window_kl.append(float(reference_kl.detach()))
                                window_entropy.append(float(entropy.detach()))
                                window_rewards.append(float(sample.reward))
                                window_masked += sample_masked
                                window_clipped += sample_clipped
                            if micro_loss is None:
                                raise RuntimeError(
                                    "dynamic micro-batch unexpectedly contained no samples"
                                )
                            backward_context = (
                                torch.autograd.graph.save_on_cpu(
                                    pin_memory=True, device_type="cuda"
                                )
                                if self._activation_cpu_offload_enabled(micro_max_tokens)
                                and cuda_metrics
                                else nullcontext()
                            )
                            with backward_context:
                                if micro_loss.requires_grad:
                                    micro_loss.backward()
                            micro_sizes.append(len(micro_indices))
                    except RuntimeError as exc:
                        if not self._is_cuda_oom(exc) or micro_size_limit <= 1:
                            raise
                        self.optimizer.zero_grad(set_to_none=True)
                        if cuda_metrics:
                            torch.cuda.empty_cache()
                            torch.cuda.set_rng_state(cuda_rng, self.config.device)
                        torch.set_rng_state(cpu_rng)
                        micro_size_limit = max(1, micro_size_limit // 2)
                        oom_retries += 1
                        continue
                    break
                updates += mini_size
                expected_window_masked = sum(
                    _policy_target_count(sample) for sample in mini_samples
                )
                if window_masked != expected_window_masked:
                    raise RuntimeError(
                        "trainer masked-token accounting diverged from durable policy masks "
                        f"({window_masked} != {expected_window_masked})"
                    )
                clipped_tokens += window_clipped
                total_loss += sum(window_loss)
                total_policy += sum(window_policy)
                total_kl += sum(window_kl)
                total_entropy += sum(window_entropy)
                masked_tokens += window_masked
                self._before_gradient_clip(mini_size)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                gradient_norms.append(float(norm))
                # Snapshot only trainable weights on CPU: no extra model-sized GPU allocation.
                parameter_before = [
                    (p, p.detach().to(device="cpu", dtype=torch.float32, copy=True))
                    for p in self.model.parameters()
                    if p.requires_grad
                ]
                applied_learning_rate = float(self.optimizer.param_groups[0]["lr"])
                self.optimizer.step()
                parameter_update_l2 = math.sqrt(
                    sum(
                        float((p.detach().float().cpu() - before).double().square().sum())
                        for p, before in parameter_before
                    )
                )
                del parameter_before
                self.scheduler.step()
                optimizer_steps += 1
                self.optimizer_step_count += 1
                step_metrics.append(
                    self._step_metrics(
                        alternating_update_step=step,
                        epoch=_epoch,
                        losses=window_loss,
                        policy_losses=window_policy,
                        kls=window_kl,
                        entropies=window_entropy,
                        rewards=window_rewards,
                        masked_tokens=window_masked,
                        clipped_tokens=window_clipped,
                        grad_norm=float(norm),
                        duration_seconds=time.perf_counter() - window_started,
                        microbatch_sizes=micro_sizes,
                    )
                )
                from .policy_timeline import timeline_positions

                layout_counts = {
                    "timeline_merged_samples": 0,
                    "timeline_fallback_samples": 0,
                    "per_call_forward_tokens": 0,
                    "learner_forward_tokens": 0,
                }
                for sample in mini_samples:
                    raw_tokens = (
                        sum(len(c.token_ids) for c in sample.policy_calls)
                        if sample.policy_calls
                        else len(sample.token_ids)
                    )
                    merged = (
                        self.config.raw_policy_backward_mode == "timeline"
                        and sample.policy_calls
                        and timeline_positions(sample.policy_calls, self.config.max_sequence_length)
                        is not None
                    )
                    layout_counts["per_call_forward_tokens"] += raw_tokens
                    layout_counts["learner_forward_tokens"] += (
                        len(sample.policy_calls[-1].token_ids) if merged else raw_tokens
                    )
                    if sample.policy_calls:
                        layout_counts[
                            "timeline_merged_samples" if merged else "timeline_fallback_samples"
                        ] += 1
                step_metrics[-1].update(layout_counts)
                step_metrics[-1].update(initial_ratio_diagnostics)
                observed_ratios = torch.cat(ratio_values) if ratio_values else torch.empty(0)
                finite_ratios = observed_ratios[torch.isfinite(observed_ratios)]
                step_metrics[-1].update(
                    parameter_update_l2=parameter_update_l2,
                    applied_learning_rate=applied_learning_rate,
                    oom_retry_count=oom_retries,
                    nonfinite_ratio_count=int((~torch.isfinite(observed_ratios)).sum()),
                    ratio_count=int(observed_ratios.numel()),
                    ratio_mean=float(finite_ratios.mean()) if finite_ratios.numel() else None,
                    ratio_min=float(finite_ratios.min()) if finite_ratios.numel() else None,
                    ratio_max=float(finite_ratios.max()) if finite_ratios.numel() else None,
                    ratio_p10=float(torch.quantile(finite_ratios, 0.1))
                    if finite_ratios.numel()
                    else None,
                    ratio_p50=float(torch.quantile(finite_ratios, 0.5))
                    if finite_ratios.numel()
                    else None,
                    ratio_p90=float(torch.quantile(finite_ratios, 0.9))
                    if finite_ratios.numel()
                    else None,
                )
        checkpoint = self._save(step)
        if cuda_metrics:
            torch.cuda.synchronize(self.config.device)
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(self.config.device) / 2**20
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(self.config.device) / 2**20
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(self.config.device) / 2**20
        else:
            peak_gpu_memory_mb = gpu_memory_allocated_mb = gpu_memory_reserved_mb = 0.0
        divisor = max(1, updates)
        rewards = [float(sample.reward) for sample in batch.samples]
        return PolicyUpdateResult(
            self.role,
            step,
            total_loss / divisor,
            total_policy / divisor,
            total_kl / divisor,
            masked_tokens,
            str(checkpoint),
            total_entropy / divisor,
            samples=len(batch.samples),
            optimizer_steps=optimizer_steps,
            learning_rate=float(self.scheduler.get_last_lr()[0]),
            grad_norm=(sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0),
            clip_fraction=clipped_tokens / max(1, masked_tokens),
            reward_mean=statistics.fmean(rewards),
            reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            duration_seconds=time.perf_counter() - started,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            step_metrics=tuple(step_metrics),
            kl_loss_contribution=self.config.kl_coefficient * total_kl / divisor,
            entropy_loss_contribution=(-self.config.entropy_coefficient * total_entropy / divisor),
            loss_variant=self.config.loss_variant,
            reference_provenance=(
                (
                    "async_frozen_probability_cache"
                    if self.config.probability_cache_path
                    else "lora_disabled_base"
                )
                if self.config.kl_coefficient
                else "disabled"
            ),
        )

    def _step_metrics(
        self,
        *,
        alternating_update_step: int,
        epoch: int,
        losses: list[float],
        policy_losses: list[float],
        kls: list[float],
        entropies: list[float],
        rewards: list[float],
        masked_tokens: int,
        clipped_tokens: int,
        grad_norm: float,
        duration_seconds: float,
        microbatch_sizes: list[int],
    ) -> dict[str, Any]:
        cuda_metrics = self.torch.cuda.is_available() and str(self.config.device).startswith("cuda")
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "role": self.role,
            "role_optimizer_step": self.optimizer_step_count,
            "alternating_update_step": alternating_update_step,
            "epoch": epoch,
            "microbatch_count": len(microbatch_sizes),
            "microbatch_sizes": list(microbatch_sizes),
            "max_microbatch_size": max(microbatch_sizes, default=0),
            "loss": statistics.fmean(losses),
            "policy_loss": statistics.fmean(policy_losses),
            "kl": statistics.fmean(kls),
            "entropy": statistics.fmean(entropies),
            "kl_loss_contribution": self.config.kl_coefficient * statistics.fmean(kls),
            "entropy_loss_contribution": (
                -self.config.entropy_coefficient * statistics.fmean(entropies)
            ),
            "loss_variant": self.config.loss_variant,
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "masked_tokens": masked_tokens,
            "clip_fraction": clipped_tokens / max(1, masked_tokens),
            "grad_norm": grad_norm,
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "duration_seconds": duration_seconds,
            "gpu_memory_allocated_mb": (
                self.torch.cuda.memory_allocated(self.config.device) / 2**20
                if cuda_metrics
                else 0.0
            ),
            "gpu_memory_reserved_mb": (
                self.torch.cuda.memory_reserved(self.config.device) / 2**20 if cuda_metrics else 0.0
            ),
        }

    def _token_log_probs(
        self,
        token_ids: tuple[int, ...],
        *,
        require_grad: bool = False,
        reference: bool = False,
        include_entropy: bool = False,
        entropy_requires_grad: bool = False,
    ):
        return self._token_log_probs_batch(
            (token_ids,),
            require_grad=require_grad,
            reference=reference,
            include_entropy=include_entropy,
            entropy_requires_grad=entropy_requires_grad,
        )[0]

    def _collect_token_log_probs(
        self,
        samples: tuple[TrainingSample, ...],
        *,
        reference: bool = False,
    ) -> list[Any]:
        """Compute frozen log-probs with the same bounded dynamic packing."""

        micro_size_limit = self.config.micro_batch_size
        while True:
            collected: list[Any | None] = [None] * len(samples)
            try:
                for indices in dynamic_micro_batch_indices(
                    samples,
                    max_batch_size=micro_size_limit,
                    max_padded_tokens=self.config.max_micro_batch_tokens,
                ):
                    outputs = self._token_log_probs_batch(
                        tuple(samples[index].token_ids for index in indices),
                        reference=reference,
                    )
                    for index, values in zip(indices, outputs, strict=True):
                        collected[index] = values
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or micro_size_limit <= 1:
                    raise
                if self.torch.cuda.is_available() and str(self.config.device).startswith("cuda"):
                    self.torch.cuda.empty_cache()
                micro_size_limit = max(1, micro_size_limit // 2)
                continue
            if any(values is None for values in collected):
                raise RuntimeError("dynamic log-prob collection left an unfilled sample")
            return [values for values in collected if values is not None]

    def _token_log_probs_batch(
        self,
        token_id_sequences: tuple[tuple[int, ...], ...],
        *,
        require_grad: bool = False,
        reference: bool = False,
        include_entropy: bool = False,
        entropy_requires_grad: bool = False,
    ) -> list[Any]:
        torch = self.torch
        if not token_id_sequences:
            raise ValueError("token log-prob batch cannot be empty")
        lengths = [len(token_ids) for token_ids in token_id_sequences]
        if min(lengths) < 2:
            raise ValueError("every token log-prob sequence must contain at least two tokens")
        maximum = max(lengths)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        inputs = torch.full(
            (len(token_id_sequences), maximum),
            int(pad_token_id),
            dtype=torch.long,
            device=self.config.device,
        )
        attention_mask = torch.zeros_like(inputs)
        for row, token_ids in enumerate(token_id_sequences):
            width = len(token_ids)
            inputs[row, :width] = torch.tensor(
                token_ids, dtype=torch.long, device=self.config.device
            )
            attention_mask[row, :width] = 1
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = nullcontext() if require_grad else torch.no_grad()
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if require_grad and self._activation_cpu_offload_enabled(maximum)
            else nullcontext()
        )
        with adapter_context, gradient_context, offload_context:
            logits = self.model(input_ids=inputs, attention_mask=attention_mask).logits[:, :-1, :]
            labels = inputs[:, 1:]
            all_log_probs = torch.log_softmax(logits, dim=-1)
            values = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            entropy_source = all_log_probs if entropy_requires_grad else all_log_probs.detach()
            entropy = (
                -(entropy_source.exp() * entropy_source).sum(dim=-1) if include_entropy else None
            )
        outputs: list[Any] = []
        for row, length in enumerate(lengths):
            row_values = values[row, : length - 1]
            row_entropy = entropy[row, : length - 1] if entropy is not None else None
            if not require_grad:
                row_values = row_values.detach().cpu()
                if row_entropy is not None:
                    row_entropy = row_entropy.detach().cpu()
            outputs.append((row_values, row_entropy) if include_entropy else row_values)
        return outputs

    def _binary_choice_log_probs(
        self,
        call: TokenizedPolicyCall,
        policy: dict[str, Any],
        *,
        require_grad: bool = False,
        reference: bool = False,
    ):
        """Evaluate current/reference policy on the same constrained off/on support."""

        torch = self.torch
        if call.relation_token_span is None:
            raise ValueError("binary relation call lacks a token span")
        start, end = call.relation_token_span
        if end - start != 1 or start <= 0:
            raise ValueError("binary relation call must contain one predicted token")
        token_map = policy.get("token_ids", {})
        try:
            support = [int(token_map[choice]) for choice in ("off", "on")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("binary relation policy lacks off/on tokenizer IDs") from exc
        inputs = torch.tensor([call.token_ids], dtype=torch.long, device=self.config.device)
        attention_mask = torch.ones_like(inputs)
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = nullcontext() if require_grad else torch.no_grad()
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if require_grad and self._activation_cpu_offload_enabled(len(call.token_ids))
            else nullcontext()
        )
        with adapter_context, gradient_context, offload_context:
            position = torch.tensor([start - 1], dtype=torch.long, device=self.config.device)
            logits = self._selected_sequence_logits(inputs, attention_mask, position)
            selected = logits[0, 0, support]
            values = torch.log_softmax(selected.float(), dim=-1)
        return values if require_grad else values.detach()

    def _timeline_policy_outputs(self, calls, *, reference=False, include_entropy=True):
        """Share exact causal prefixes; return None for per-call streaming fallback.

        All action targets retain their original positions. Reference/behavior
        probabilities, PPO weighting and optimizer boundaries remain unchanged.
        """
        from .policy_timeline import timeline_positions

        positions = timeline_positions(calls, self.config.max_sequence_length)
        if positions is None:
            return None
        torch = self.torch
        ids = calls[-1].token_ids
        inputs = torch.tensor([ids], dtype=torch.long, device=self.config.device)
        selected = sorted(p for row in positions for p in row)
        lookup = {p: i for i, p in enumerate(selected)}
        offload = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if self._activation_cpu_offload_enabled(len(ids))
            else nullcontext()
        )
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = torch.no_grad() if reference else nullcontext()
        with adapter_context, gradient_context, offload:
            logits = self._selected_sequence_logits(
                inputs,
                torch.ones_like(inputs),
                torch.tensor(selected, dtype=torch.long, device=inputs.device),
            )[0]
            outputs = []
            for call, row in zip(calls, positions, strict=True):
                local = logits[[lookup[p] for p in row]]
                relation = call.metadata.get("relation_policy", {})
                if call.relation_token_span is not None and relation:
                    support = [int(relation["token_ids"][k]) for k in ("off", "on")]
                    outputs.append(torch.log_softmax(local[0, support].float(), dim=-1))
                else:
                    lp = torch.log_softmax(local, dim=-1)
                    labels = torch.tensor(
                        [call.token_ids[p + 1] for p in row], dtype=torch.long, device=inputs.device
                    )
                    values = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    ent_source = lp if self.config.entropy_coefficient > 0 else lp.detach()
                    entropy = (
                        -(ent_source.exp() * ent_source).sum(dim=-1) if include_entropy else None
                    )
                    outputs.append((values, entropy))
        return outputs

    def _masked_policy_call_log_probs(
        self,
        call: TokenizedPolicyCall,
        *,
        require_grad: bool = False,
        reference: bool = False,
        include_entropy: bool = False,
        entropy_requires_grad: bool = False,
    ):
        """Evaluate only sampled action positions while retaining the full prefix."""

        torch = self.torch
        positions = [index for index, keep in enumerate(call.action_mask[1:]) if keep]
        if not positions:
            raise ValueError(f"policy call {call.call_id} has no next-token action target")
        inputs = torch.tensor([call.token_ids], dtype=torch.long, device=self.config.device)
        attention_mask = torch.ones_like(inputs)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=self.config.device)
        labels = torch.tensor(
            [call.token_ids[position + 1] for position in positions],
            dtype=torch.long,
            device=self.config.device,
        )
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = nullcontext() if require_grad else torch.no_grad()
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if require_grad and self._activation_cpu_offload_enabled(len(call.token_ids))
            else nullcontext()
        )
        with adapter_context, gradient_context, offload_context:
            logits = self._selected_sequence_logits(inputs, attention_mask, position_tensor)[0]
            all_log_probs = torch.log_softmax(logits, dim=-1)
            values = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            entropy_source = all_log_probs if entropy_requires_grad else all_log_probs.detach()
            entropy = (
                -(entropy_source.exp() * entropy_source).sum(dim=-1) if include_entropy else None
            )
        if not require_grad:
            values = values.detach().cpu()
            if entropy is not None:
                entropy = entropy.detach().cpu()
        return (values, entropy) if include_entropy else values

    def _before_gradient_clip(self, local_sample_count: int) -> None:
        """Optional distributed synchronization boundary, after all local backward calls."""

    def _selected_sequence_logits(self, inputs: Any, attention_mask: Any, positions: Any):
        """Use model-side logit slicing, with a narrow fallback for tiny test models."""

        try:
            return self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                logits_to_keep=positions,
            ).logits
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            return self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
            ).logits[:, positions, :]

    def _activation_cpu_offload_enabled(self, token_count: int) -> bool:
        threshold = self.config.activation_cpu_offload_min_tokens
        return bool(
            self.config.activation_cpu_offload
            and str(self.config.device).startswith("cuda")
            and (threshold == 0 or token_count >= threshold)
        )

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        oom_type = getattr(self.torch.cuda, "OutOfMemoryError", ())
        return (bool(oom_type) and isinstance(error, oom_type)) or "out of memory" in str(
            error
        ).casefold()

    def _save(self, step: int) -> Path:
        destination = self.config.checkpoint_root / f"step-{step:08d}"
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        self.torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_step_count": self.optimizer_step_count,
                "torch_rng": self.torch.get_rng_state(),
                "cuda_rng": (
                    self.torch.cuda.get_rng_state(self.config.device)
                    if self.torch.cuda.is_available() and str(self.config.device).startswith("cuda")
                    else None
                ),
                "cuda_rng_scope": "configured_device",
                "python_rng": random.getstate(),
                "step": step,
            },
            destination / "trainer_state.pt",
        )
        if self.config.publish_latest_checkpoint:
            _atomic_json(
                self.config.checkpoint_root / "latest.json",
                {"path": str(destination), "step": step},
            )
        _atomic_json(destination / "training_config.json", _jsonable(asdict(self.config)))
        return destination

    def _latest_checkpoint(self) -> Path | None:
        latest = self.config.checkpoint_root / "latest.json"
        if not latest.exists():
            return None
        path = Path(json.loads(latest.read_text(encoding="utf-8"))["path"])
        return path if path.exists() else None

    def _restore_optimizer(self, source: Path) -> None:
        state_path = source / "trainer_state.pt"
        if not state_path.exists():
            return
        saved_total_optimizer_steps = None
        saved_schedule = "cosine"
        saved_warmup_start_factor = 0.0
        saved_config_path = source / "training_config.json"
        if saved_config_path.exists():
            saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
            saved_total_optimizer_steps = int(saved_config["total_optimizer_steps"])
            saved_schedule = saved_config.get("learning_rate_schedule", "cosine")
            saved_warmup_start_factor = float(saved_config.get("warmup_start_factor", 0.0))
        if saved_schedule != self.config.learning_rate_schedule:
            raise ValueError("checkpoint learning-rate schedule differs from requested schedule")
        if saved_warmup_start_factor != self.config.warmup_start_factor:
            raise ValueError("checkpoint warmup start factor differs from requested schedule")
        state = self.torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer_step_count = int(
            state.get("optimizer_step_count", self.scheduler.last_epoch)
        )
        if (
            saved_total_optimizer_steps is not None
            and saved_total_optimizer_steps != self.config.total_optimizer_steps
        ):
            if self.config.total_optimizer_steps <= self.optimizer_step_count:
                raise ValueError(
                    "extended training requires total_optimizer_steps greater than the "
                    f"restored optimizer step {self.optimizer_step_count}"
                )
            resumed_lrs = _scheduler_lrs_at_step(self.scheduler, self.optimizer_step_count)
            for group, learning_rate in zip(self.optimizer.param_groups, resumed_lrs, strict=True):
                group["lr"] = learning_rate
            self.scheduler.last_epoch = self.optimizer_step_count
            self.scheduler._last_lr = resumed_lrs
        self.torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and self.torch.cuda.is_available():
            if state.get("cuda_rng_scope") == "configured_device":
                self.torch.cuda.set_rng_state(state["cuda_rng"], self.config.device)
            else:
                # Backward compatibility with checkpoints that stored all visible GPUs.
                self.torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])

    def close(self) -> None:
        del self.scheduler
        del self.optimizer
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _build_learning_rate_scheduler(optimizer: Any, config: PolicyTrainingConfig) -> Any:
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    if config.warmup_steps > 0 and config.warmup_start_factor > 0:
        from torch.optim.lr_scheduler import LambdaLR

        def factor(step: int) -> float:
            if step < config.warmup_steps:
                return (
                    config.warmup_start_factor
                    + (1.0 - config.warmup_start_factor) * step / config.warmup_steps
                )
            if config.learning_rate_schedule == "constant":
                return 1.0
            progress = (step - config.warmup_steps) / max(
                1, config.total_optimizer_steps - config.warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, factor)
    if config.learning_rate_schedule == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_optimizer_steps,
    )


def _scheduler_lrs_at_step(scheduler: Any, step: int) -> list[float]:
    """Rebase a LambdaLR when a resumed experiment explicitly extends its horizon."""

    return [
        float(base_lr) * float(schedule(step))
        for base_lr, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)
    ]


def create_policy_trainer(role: str, policy: PolicyTrainingConfig, seed: int) -> PolicyTrainer:
    if policy.data_parallel_gpu_ids:
        from .solver_parallel_trainer import SubprocessSolverTrainer

        return SubprocessSolverTrainer(role, policy, seed=seed)
    return TransformersGRPOTrainer(role, policy, seed=seed)


class AlternatingGRPOTrainer:
    def __init__(
        self,
        config: AlternatingTrainingConfig,
        *,
        trainer_factory: Callable[[str, PolicyTrainingConfig, int], PolicyTrainer] | None = None,
        before_role_callback: Callable[[str], None] | None = None,
        checkpoint_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.trainer_factory = trainer_factory or create_policy_trainer
        self.before_role_callback = before_role_callback
        self.checkpoint_callback = checkpoint_callback
        self.state = self._load_state()
        self._recover_interrupted_parallel_publish()

    def train_cycle(
        self,
        proposer_batch: TrainingBatch,
        solver_batch: TrainingBatch,
        *,
        relation_credits: Iterable[RelationCredit] = (),
    ) -> tuple[PolicyUpdateResult, PolicyUpdateResult]:
        self._validate_cycle_batches(proposer_batch, solver_batch)
        selected_ids = {s.rollout_id for s in solver_batch.samples}
        relation_credits = tuple(c for c in relation_credits if c.rollout_id in selected_ids)
        if solver_batch.metadata.get("training_selection_schema") in {
            "eligible_subset_v1",
            "independent_frontier_v2",
        }:
            digest = hashlib.sha256(
                json.dumps(
                    [
                        proposer_batch.to_dict(),
                        solver_batch.to_dict(),
                        [asdict(c) if is_dataclass(c) else vars(c) for c in relation_credits],
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if self.state.active_batch_sha256 and self.state.active_batch_sha256 != digest:
                raise UnsafeTrainingBatchError("cannot change a batch during an interrupted cycle")
            self.state.active_batch_sha256 = digest
            self._save_state()
        if self.config.parallel_roles and proposer_batch.samples and solver_batch.samples:
            return self._train_cycle_parallel(
                proposer_batch,
                solver_batch,
                relation_credits=tuple(relation_credits),
            )
        if self.state.phase not in {"proposer", "solver", "cycle_complete"}:
            raise ValueError(f"unknown alternating training phase: {self.state.phase}")
        proposer = (
            self._train_role(
                "proposer",
                proposer_batch,
                self.config.proposer,
                (),
                next_phase="solver",
            )
            if self.state.phase == "proposer"
            else self._latest_result("proposer")
        )
        solver = (
            self._train_role(
                "solver",
                solver_batch,
                self.config.solver,
                relation_credits,
                next_phase="cycle_complete",
            )
            if self.state.phase == "solver"
            else self._latest_result("solver")
        )
        self.state.cycle += 1
        self.state.active_batch_sha256 = ""
        self.state.phase = "proposer"
        self._save_state()
        return proposer, solver

    def committed_cycle_results(
        self, cycle: int
    ) -> tuple[PolicyUpdateResult, PolicyUpdateResult] | None:
        """Return a fully committed cycle without running either optimizer again.

        The experiment progress file is written after service restoration and
        metrics.  A failure in those steps can therefore leave durable policy
        checkpoints for a cycle which is not yet present in experiment_progress.
        Resume must use those checkpoints instead of interpreting the fresh
        ``phase=proposer`` boundary as permission to train the same batch twice.
        """

        if self.state.cycle <= cycle:
            return None
        results: dict[str, PolicyUpdateResult] = {}
        for payload in self.state.history:
            if payload.get("cycle") != cycle:
                continue
            normalized = dict(payload)
            normalized["step_metrics"] = tuple(normalized.get("step_metrics", ()))
            result = PolicyUpdateResult(**normalized)
            results[result.role] = result
        if set(results) != {"proposer", "solver"}:
            raise RuntimeError(
                f"training state marks cycle {cycle} committed but does not contain both roles"
            )
        return results["proposer"], results["solver"]

    def _train_cycle_parallel(
        self,
        proposer_batch: TrainingBatch,
        solver_batch: TrainingBatch,
        *,
        relation_credits: tuple[RelationCredit, ...],
    ) -> tuple[PolicyUpdateResult, PolicyUpdateResult]:
        if self.state.phase != "proposer":
            raise RuntimeError(
                "parallel role training can start only at a jointly committed cycle boundary"
            )
        proposer_step = self.state.global_step + 1
        solver_step = self.state.global_step + 2
        if self.before_role_callback:
            # Service mutation stays on the coordinator thread.  The expensive
            # model updates below are the only concurrent operations.
            self.before_role_callback("proposer")
            self.before_role_callback("solver")
        proposer_config = replace(self.config.proposer, publish_latest_checkpoint=False)
        solver_config = replace(self.config.solver, publish_latest_checkpoint=False)
        proposer_trainer = self.trainer_factory(
            "proposer", proposer_config, self.config.seed + proposer_step
        )
        try:
            solver_trainer = self.trainer_factory(
                "solver", solver_config, self.config.seed + solver_step
            )
        except BaseException:
            proposer_trainer.close()
            raise
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="policy-update") as pool:
            proposer_future = pool.submit(
                self._run_role_update,
                proposer_trainer,
                proposer_batch,
                (),
                proposer_step,
            )
            solver_future = pool.submit(
                self._run_role_update,
                solver_trainer,
                solver_batch,
                relation_credits,
                solver_step,
            )
            proposer = proposer_future.result()
            solver = solver_future.result()
        proposer = replace(proposer, cycle=self.state.cycle)
        solver = replace(solver, cycle=self.state.cycle)
        self._validate_parallel_results(proposer, solver)
        self._publish_parallel_results(proposer, solver)
        # Both trainers still have live local references until this method
        # returns.  Starting vLLM from here races their CUDA teardown and can
        # fail even though both checkpoints have already been committed.  The
        # experiment coordinator restores services after train_cycle returns.
        return proposer, solver

    def _run_role_update(
        self,
        trainer: PolicyTrainer,
        batch: TrainingBatch,
        credits: Iterable[RelationCredit],
        step: int,
    ) -> PolicyUpdateResult:
        activate_rng = getattr(trainer, "activate_rng", None)
        if activate_rng is not None:
            activate_rng()
        try:
            return trainer.update(batch, step=step, relation_credits=credits)
        finally:
            trainer.close()

    def _validate_parallel_results(
        self,
        proposer: PolicyUpdateResult,
        solver: PolicyUpdateResult,
    ) -> None:
        for expected_role, result in (("proposer", proposer), ("solver", solver)):
            if result.role != expected_role:
                raise RuntimeError(f"parallel trainer returned {result.role!r} for {expected_role}")
            if result.optimizer_steps <= 0 or not Path(result.checkpoint).exists():
                raise RuntimeError(f"{expected_role} produced an incomplete pending checkpoint")
            numeric = (result.loss, result.policy_loss, result.kl, result.grad_norm)
            if not all(math.isfinite(float(value)) for value in numeric):
                raise RuntimeError(f"{expected_role} produced non-finite training metrics")

    @property
    def _parallel_transaction_path(self) -> Path:
        return self.config.state_path.with_suffix(".parallel-transaction.json")

    def _latest_payload(self, config: PolicyTrainingConfig) -> dict[str, Any] | None:
        path = config.checkpoint_root / "latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _restore_latest_payload(
        self,
        config: PolicyTrainingConfig,
        payload: dict[str, Any] | None,
    ) -> None:
        path = config.checkpoint_root / "latest.json"
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_json(path, payload)

    def _publish_parallel_results(
        self,
        proposer: PolicyUpdateResult,
        solver: PolicyUpdateResult,
    ) -> None:
        previous = {
            "proposer": self._latest_payload(self.config.proposer),
            "solver": self._latest_payload(self.config.solver),
        }
        next_state = AlternatingTrainingState.from_dict(self.state.to_dict())
        next_state.global_step = solver.step
        next_state.proposer_step = proposer.step
        next_state.solver_step = solver.step
        next_state.proposer_checkpoint = proposer.checkpoint
        next_state.solver_checkpoint = solver.checkpoint
        next_state.history.extend((asdict(proposer), asdict(solver)))
        next_state.cycle += 1
        next_state.active_batch_sha256 = ""
        next_state.phase = "proposer"
        transaction = {
            "schema_version": 1,
            "target_cycle": next_state.cycle,
            "previous_latest": previous,
            "next_checkpoints": {
                "proposer": proposer.checkpoint,
                "solver": solver.checkpoint,
            },
        }
        _atomic_json(self._parallel_transaction_path, transaction)
        try:
            _atomic_json(
                self.config.proposer.checkpoint_root / "latest.json",
                {"path": proposer.checkpoint, "step": proposer.step},
            )
            _atomic_json(
                self.config.solver.checkpoint_root / "latest.json",
                {"path": solver.checkpoint, "step": solver.step},
            )
            self.state = next_state
            self._save_state()
        except BaseException:
            self._restore_latest_payload(self.config.proposer, previous["proposer"])
            self._restore_latest_payload(self.config.solver, previous["solver"])
            raise
        finally:
            self._parallel_transaction_path.unlink(missing_ok=True)

    def _recover_interrupted_parallel_publish(self) -> None:
        path = self._parallel_transaction_path
        if not path.exists():
            return
        transaction = json.loads(path.read_text(encoding="utf-8"))
        target_cycle = int(transaction.get("target_cycle", -1))
        if self.state.cycle < target_cycle:
            previous = dict(transaction.get("previous_latest", {}))
            self._restore_latest_payload(self.config.proposer, previous.get("proposer"))
            self._restore_latest_payload(self.config.solver, previous.get("solver"))
        path.unlink(missing_ok=True)

    def _validate_cycle_batches(
        self,
        proposer_batch: TrainingBatch,
        solver_batch: TrainingBatch,
    ) -> None:
        if proposer_batch.role != "proposer" or solver_batch.role != "solver":
            raise UnsafeTrainingBatchError("training batch roles must be proposer then solver")
        lineage = solver_batch.metadata.get("policy_lineage")
        if lineage is not None:
            if proposer_batch.metadata.get("policy_lineage") != lineage:
                raise UnsafeTrainingBatchError("role batches have different policy lineage")
            target_cycle = int(lineage.get("target_cycle", -1))
            behavior_index = int(lineage.get("behavior_update_index", -1))
            if target_cycle != self.state.cycle or self.state.cycle - behavior_index not in {0, 1}:
                raise UnsafeTrainingBatchError(
                    "rollout policy lineage exceeds the current one-update boundary"
                )
        schema = solver_batch.metadata.get("training_selection_schema")
        if schema and schema not in {"eligible_subset_v1", "independent_frontier_v2"}:
            raise UnsafeTrainingBatchError("unsupported training selection schema")
        if schema == "independent_frontier_v2":
            from .proposer_learning import validate_independent_batches

            try:
                validate_independent_batches(proposer_batch, solver_batch)
            except ValueError as exc:
                raise UnsafeTrainingBatchError(str(exc)) from exc
            return
        if schema == "eligible_subset_v1":
            self._validate_partial_batches(proposer_batch, solver_batch)
            return
        if not proposer_batch.samples or not solver_batch.samples:
            raise UnsafeTrainingBatchError(
                "safe training requires non-empty proposer and solver batches; "
                "no optimizer update was started"
            )

        if any(
            sample.metadata.get("training_eligible") is False
            or sample.metadata.get("terminal_graph_status") == "unsafe_partial"
            for sample in solver_batch.samples
        ):
            raise UnsafeTrainingBatchError(
                "solver batch contains a training-ineligible or unsafe terminal rollout; "
                "no optimizer update was started"
            )

        expected_values = {
            int(sample.metadata["rollout_group_size_expected"])
            for sample in solver_batch.samples
            if "rollout_group_size_expected" in sample.metadata
        }
        if expected_values:
            if len(expected_values) != 1:
                raise UnsafeTrainingBatchError("solver batch mixes different rollout group sizes")
            expected = next(iter(expected_values))
            if expected < 2 or any(
                int(sample.metadata.get("rollout_group_size_expected", 0)) != expected
                or not sample.metadata.get("rollout_group_complete")
                for sample in solver_batch.samples
            ):
                raise UnsafeTrainingBatchError(
                    "solver batch contains an unverified or incomplete rollout group"
                )
            solver_counts: dict[str, int] = {}
            for sample in solver_batch.samples:
                solver_counts[sample.task_id] = solver_counts.get(sample.task_id, 0) + 1
            incomplete = {
                task_id: count for task_id, count in solver_counts.items() if count != expected
            }
            proposer_counts: dict[str, int] = {}
            for sample in proposer_batch.samples:
                proposer_counts[sample.task_id] = proposer_counts.get(sample.task_id, 0) + 1
            allowed_proposer_exclusions = {
                "canary_executor_migration",
                "frontier_reverify_infrastructure_failure",
            }
            excluded_counts: dict[str, int] = {}
            for sample in solver_batch.samples:
                if (
                    sample.metadata.get("frontier_training_exclusion")
                    in allowed_proposer_exclusions
                ):
                    excluded_counts[sample.task_id] = excluded_counts.get(sample.task_id, 0) + 1
            excluded_tasks = {task for task, count in excluded_counts.items() if count == expected}
            if (
                incomplete
                or any(count != expected for count in excluded_counts.values())
                or set(proposer_counts) != set(solver_counts) - excluded_tasks
                or any(count != 1 for count in proposer_counts.values())
            ):
                raise UnsafeTrainingBatchError(
                    "proposer/solver batches do not contain the same complete task groups"
                )

        # A short final mini-batch is valid.  It contains only real, already
        # verified samples; adding synthetic padding would violate the sibling
        # group invariant above.

    def _validate_partial_batches(
        self, proposer_batch: TrainingBatch, solver_batch: TrainingBatch
    ) -> None:
        from .training_selection import trajectory_exclusions

        if proposer_batch.metadata != solver_batch.metadata:
            raise UnsafeTrainingBatchError("role batches have different selection manifests")
        manifest = solver_batch.metadata
        ids = [sample.rollout_id for sample in solver_batch.samples]
        if len(ids) != len(set(ids)) or ids != manifest.get("selected_rollout_ids"):
            raise UnsafeTrainingBatchError("Solver samples differ from selected rollout IDs")
        groups = {g["task_id"]: g for g in manifest.get("groups", ())}
        actual: dict[str, list[str]] = {}
        for sample in solver_batch.samples:
            if trajectory_exclusions(sample):
                raise UnsafeTrainingBatchError("selected Solver sample fails trajectory admission")
            actual.setdefault(sample.task_id, []).append(sample.rollout_id)
        expected_proposer = set()
        for task_id, group in groups.items():
            selected = group["selected_rollout_ids"]
            if actual.get(task_id, []) != selected:
                raise UnsafeTrainingBatchError("selected sibling membership differs from manifest")
            if selected and not 2 <= len(selected) <= group["planned_rollout_count"]:
                raise UnsafeTrainingBatchError("invalid selected sibling count")
            if not selected:
                continue
            samples = [s for s in solver_batch.samples if s.task_id == task_id]
            for sample in samples:
                if (
                    sample.metadata.get("selected_group_rollout_ids") != selected
                    or sample.metadata.get("selected_group_size") != len(selected)
                    or sample.metadata.get("rollout_group_size_expected")
                    != group["planned_rollout_count"]
                    or bool(sample.metadata.get("rollout_group_complete"))
                    != (len(selected) == group["planned_rollout_count"])
                ):
                    raise UnsafeTrainingBatchError("sample sibling metadata is inconsistent")
            exclusions = {s.metadata.get("frontier_training_exclusion") for s in samples}
            if len(exclusions) != 1:
                raise UnsafeTrainingBatchError("inconsistent Frontier exclusion within task")
            exclusion = next(iter(exclusions))
            if len(selected) < group["planned_rollout_count"]:
                if exclusion != "partial_solver_group":
                    raise UnsafeTrainingBatchError("partial group cannot train Proposer")
            elif exclusion is None:
                expected_proposer.add(task_id)
            elif exclusion not in {
                "canary_executor_migration",
                "frontier_reverify_infrastructure_failure",
            }:
                raise UnsafeTrainingBatchError("unknown Frontier exclusion")
        if set(actual) - set(groups):
            raise UnsafeTrainingBatchError("Solver contains unplanned tasks")
        proposer_ids = [s.task_id for s in proposer_batch.samples]
        if len(proposer_ids) != len(set(proposer_ids)) or set(proposer_ids) != expected_proposer:
            raise UnsafeTrainingBatchError("Proposer samples differ from admitted complete tasks")

    def _train_role(
        self,
        role: str,
        batch: TrainingBatch,
        config: PolicyTrainingConfig,
        credits: Iterable[RelationCredit],
        *,
        next_phase: str,
    ) -> PolicyUpdateResult:
        if not batch.samples:
            result = PolicyUpdateResult(
                role=role,
                step=self.state.global_step,
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                masked_tokens=0,
                checkpoint="",
                status="skipped",
                skip_reason="no_eligible_samples",
                cycle=self.state.cycle,
            )
            self.state.history.append(asdict(result))
            self.state.phase = next_phase
            self._save_state()
            return result
        step = self.state.global_step + 1
        credits = [c for c in credits if c.rollout_id in {s.rollout_id for s in batch.samples}]
        if self.before_role_callback:
            self.before_role_callback(role)
        trainer = self.trainer_factory(role, config, self.config.seed + step)
        try:
            result = trainer.update(batch, step=step, relation_credits=credits)
        finally:
            trainer.close()
        result = replace(result, cycle=self.state.cycle)
        self.state.global_step = step
        setattr(self.state, f"{role}_step", step)
        setattr(self.state, f"{role}_checkpoint", result.checkpoint)
        self.state.history.append(asdict(result))
        self.state.phase = next_phase
        self._save_state()
        if self.checkpoint_callback:
            self.checkpoint_callback(role, result.checkpoint)
        return result

    def _latest_result(self, role: str) -> PolicyUpdateResult:
        for payload in reversed(self.state.history):
            if str(payload.get("role", "")) == role:
                normalized = dict(payload)
                normalized["step_metrics"] = tuple(normalized.get("step_metrics", ()))
                return PolicyUpdateResult(**normalized)
        raise RuntimeError(f"training state says {role} is complete but contains no saved result")

    def _load_state(self) -> AlternatingTrainingState:
        if not self.config.state_path.exists():
            return AlternatingTrainingState()
        return AlternatingTrainingState.from_dict(
            json.loads(self.config.state_path.read_text(encoding="utf-8"))
        )

    def _save_state(self) -> None:
        _atomic_json(self.config.state_path, self.state.to_dict())


class MockPolicyTrainer:
    """Small deterministic backend used to exercise orchestration without Torch."""

    def __init__(self, role: str, config: PolicyTrainingConfig, *, seed: int = 0) -> None:
        self.role = role
        self.config = config
        self.seed = seed

    def update(
        self,
        batch: TrainingBatch,
        *,
        step: int,
        relation_credits: Iterable[RelationCredit] = (),
    ) -> PolicyUpdateResult:
        started = time.perf_counter()
        credits = list(relation_credits)
        masked = sum(sum(sample.action_mask) for sample in batch.samples)
        values = [
            value
            for sample in batch.samples
            for value, selected in zip(
                token_advantages(sample, credits), sample.action_mask, strict=True
            )
            if selected
        ]
        loss = -sum(values) / max(1, len(values))
        destination = self.config.checkpoint_root / f"step-{step:08d}"
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            destination / "mock_policy.json",
            {"role": self.role, "step": step, "loss": loss, "seed": self.seed},
        )
        if self.config.publish_latest_checkpoint:
            _atomic_json(
                self.config.checkpoint_root / "latest.json",
                {"path": str(destination), "step": step},
            )
        rewards = [float(sample.reward) for sample in batch.samples]
        duration = time.perf_counter() - started
        logical_mini_size = logical_mini_batch_size(batch, self.config)
        optimizer_steps = max(1, (len(batch.samples) + logical_mini_size - 1) // logical_mini_size)
        step_metric = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "role": self.role,
            "role_optimizer_step": step,
            "alternating_update_step": step,
            "epoch": 0,
            "microbatch_count": len(batch.samples),
            "loss": loss,
            "policy_loss": loss,
            "kl": 0.0,
            "entropy": 0.0,
            "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "masked_tokens": masked,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "learning_rate": self.config.learning_rate,
            "duration_seconds": duration,
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
        }
        return PolicyUpdateResult(
            self.role,
            step,
            loss,
            loss,
            0.0,
            masked,
            str(destination),
            samples=len(batch.samples),
            optimizer_steps=optimizer_steps,
            learning_rate=self.config.learning_rate,
            reward_mean=statistics.fmean(rewards) if rewards else 0.0,
            reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            duration_seconds=duration,
            step_metrics=(step_metric,),
        )

    def close(self) -> None:
        return None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
