from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .rollouts import TokenizedDirectorTrajectory
from .training import PolicyTrainingConfig, TransformersGRPOTrainer


def cached_identity_failures(
    trajectory: TokenizedDirectorTrajectory, record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the trainer's existing frozen-policy gate before group selection."""
    import torch
    from .training import (
        _binary_policy_identity_failed,
        _policy_identity_failed,
        _policy_call_sha256,
    )

    rows = {row["call_id"]: row for row in record["calls"]}
    failures = []
    for call in trajectory.policy_calls:
        row = rows[call.call_id]
        if row["call_sha256"] != _policy_call_sha256(call):
            raise ValueError(f"Solver probability cache call changed: {call.call_id}")
        current = torch.tensor(row["current"], dtype=torch.float32)
        expected = torch.tensor(
            [
                v
                for v, keep in zip(call.behavior_log_probs, call.action_mask[1:], strict=True)
                if keep
            ],
            dtype=torch.float32,
        )
        relation = call.metadata.get("relation_policy", {})
        binary = call.relation_token_span is not None and bool(relation)
        if binary:
            selected = (
                0
                if call.token_ids[call.relation_token_span[0]] == int(relation["token_ids"]["off"])
                else 1
            )
            observed = current[selected : selected + 1]
        else:
            observed = current
        if observed.shape != expected.shape:
            raise ValueError(f"Solver probability cache target shape changed: {call.call_id}")
        difference = (observed - expected).abs()
        if not difference.numel():
            continue
        finite = bool(torch.isfinite(difference).all())
        maximum, mean, p95 = (
            float(difference.max()),
            float(difference.mean()),
            float(torch.quantile(difference, 0.95)),
        )
        failed = not finite or _policy_identity_failed(
            max_delta=maximum, mean_delta=mean, p95_delta=p95, relation_call=binary
        )
        if binary and "log_probabilities" in relation:
            support = torch.tensor(
                [relation["log_probabilities"][choice] for choice in ("off", "on")],
                dtype=torch.float32,
            )
            failed = (
                not finite
                or _binary_policy_identity_failed(current, support)
                or abs(float(support[selected]) - float(expected[0])) > 1e-5
            )
        if failed:
            failures.append(
                {
                    "call_id": call.call_id,
                    "relation_call": binary,
                    "max_abs_logprob_delta": maximum,
                    "mean_abs_logprob_delta": mean,
                    "p95_abs_logprob_delta": p95,
                }
            )
    return failures


def probability_cache_binding(config: PolicyTrainingConfig) -> str:
    """Bind a cache to the exact immutable Solver source and tokenizer input."""

    latest_path = config.checkpoint_root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else None
    payload = {
        "base_model_path": str(config.base_model_path.resolve()),
        "latest": latest,
        "dtype": config.dtype,
        "use_lora": config.use_lora,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_target_modules": list(config.lora_target_modules),
        "max_sequence_length": config.max_sequence_length,
    }
    if config.raw_policy_backward_mode == "timeline":
        payload["timeline_numeric_policy"] = (
            "gated_delta_projection_float64_expanded_fp32_sdpa_merged_probabilities_v1"
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AsyncSolverProbabilityCache:
    """One-GPU FIFO evaluator fed immediately after each durable primary row."""

    def __init__(
        self,
        *,
        output_path: Path,
        config: PolicyTrainingConfig,
        trainer_factory: Callable[[PolicyTrainingConfig], Any] | None = None,
        prepare_callback: Callable[[], None] | None = None,
    ) -> None:
        if not config.probability_cache_binding:
            raise ValueError("async probability cache requires a snapshot binding")
        self.output_path = Path(output_path)
        self.config = config
        self.trainer_factory = trainer_factory or (
            lambda policy: TransformersGRPOTrainer("solver", policy)
        )
        self.prepare_callback = prepare_callback
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solver-probability")
        self._future: Future[Any] | None = None
        self._trainer: Any | None = None
        self._lock = threading.Lock()
        self._submitted: set[str] = set()
        self._closed = False
        if self.output_path.exists():
            for line in self.output_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("binding") != config.probability_cache_binding:
                    raise ValueError("existing probability cache belongs to another snapshot")
                rollout_id = str(row.get("rollout_id", ""))
                if not rollout_id or rollout_id in self._submitted:
                    raise ValueError("existing probability cache has duplicate/empty rollout IDs")
                self._submitted.add(rollout_id)

    def _start(self) -> None:
        if self._trainer is not None:
            return
        if self.prepare_callback is not None:
            self.prepare_callback()
        self._trainer = self.trainer_factory(self.config)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(self, trajectory: TokenizedDirectorTrajectory) -> None:
        if self._closed:
            raise RuntimeError("probability cache is closed")
        if not trajectory.policy_calls:
            return
        if trajectory.rollout_id in self._submitted:
            return
        self._submitted.add(trajectory.rollout_id)

        def evaluate(previous: Future[Any] | None) -> None:
            if previous is not None:
                previous.result()
            self._start()
            record = self._trainer.precompute_probability_record(trajectory)
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with self._lock, self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()

        self._future = self._executor.submit(evaluate, self._future)

    def finalize(self) -> dict[str, Any]:
        if self._future is not None:
            self._future.result()
        self.close()
        return {
            "schema_version": "solver_probability_cache_summary_v1",
            "binding": self.config.probability_cache_binding,
            "submitted_rollouts": len(self._submitted),
            "path": str(self.output_path),
        }

    def abort(self) -> None:
        self.close(cancel=True)

    def identity_exclusions(
        self, trajectories: list[TokenizedDirectorTrajectory]
    ) -> dict[str, list[dict[str, Any]]]:
        records = {
            row["rollout_id"]: row
            for line in self.output_path.read_text().splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        return {
            trajectory.rollout_id: failures
            for trajectory in trajectories
            if trajectory.policy_calls
            for failures in [cached_identity_failures(trajectory, records[trajectory.rollout_id])]
            if failures
        }

    def close(self, *, cancel: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=cancel)
        if self._trainer is not None:
            self._trainer.close()
            self._trainer = None
