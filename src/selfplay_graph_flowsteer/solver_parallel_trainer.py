"""Opt-in, single-global-mini Solver updates in isolated distributed workers."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

from .rollouts import TrainingBatch
from .solver_data_parallel import balanced_rollout_shards, synchronize_mean_gradients
from .training import (
    PolicyTrainingConfig,
    PolicyUpdateResult,
    TransformersGRPOTrainer,
    _atomic_json,
    load_relation_credits,
    load_training_batch,
)


def aggregate_results(rows, batch, duration):
    """Rollout-weighted objective; token-weighted clip fraction; one logical step."""
    results = [row["result"] for row in rows]
    visited = [i for row in rows for i in row["indices"]]
    if sorted(visited) != list(range(len(batch.samples))):
        raise ValueError("missing or duplicate distributed samples")
    if any(r["optimizer_steps"] != 1 for r in results):
        raise ValueError("expected one global optimizer step")
    if any(r["samples"] != len(row["indices"]) for r, row in zip(results, rows, strict=True)):
        raise ValueError("rank sample accounting mismatch")
    merged = dict(results[0])
    for key in (
        "loss",
        "policy_loss",
        "kl",
        "entropy",
        "kl_loss_contribution",
        "entropy_loss_contribution",
    ):
        merged[key] = sum(r[key] * r["samples"] for r in results) / len(batch.samples)
    tokens = sum(r["masked_tokens"] for r in results)
    merged.update(
        samples=len(batch.samples),
        masked_tokens=tokens,
        duration_seconds=duration,
        clip_fraction=sum(r["clip_fraction"] * r["masked_tokens"] for r in results)
        / max(1, tokens),
        reward_mean=statistics.fmean(s.reward for s in batch.samples),
        reward_std=statistics.pstdev(s.reward for s in batch.samples),
    )
    for key in ("peak_gpu_memory_mb", "gpu_memory_allocated_mb", "gpu_memory_reserved_mb"):
        merged[key] = max(r[key] for r in results)
    metric = dict(results[0]["step_metrics"][0])
    for key in (
        "loss",
        "policy_loss",
        "kl",
        "entropy",
        "reward_mean",
        "reward_std",
        "masked_tokens",
        "clip_fraction",
        "duration_seconds",
        "kl_loss_contribution",
        "entropy_loss_contribution",
        "gpu_memory_allocated_mb",
        "gpu_memory_reserved_mb",
    ):
        metric[key] = merged[key]
    metric.update(
        samples=len(batch.samples),
        data_parallel_world_size=len(rows),
        rank_metrics=[r["step_metrics"] for r in results],
        microbatch_count=sum(r["step_metrics"][0]["microbatch_count"] for r in results),
        microbatch_sizes=[
            size for r in results for size in r["step_metrics"][0]["microbatch_sizes"]
        ],
    )
    local_metrics = [r["step_metrics"][0] for r in results]
    for key in ("oom_retry_count", "nonfinite_ratio_count", "ratio_count"):
        metric[key] = sum(m.get(key, 0) for m in local_metrics)
    metric["max_microbatch_size"] = max(metric["microbatch_sizes"], default=0)
    for key in ("ratio_p10", "ratio_p50", "ratio_p90"):
        metric.pop(key, None)  # Rank quantiles cannot be merged into exact global quantiles.
    count = metric["ratio_count"]
    metric["ratio_mean"] = (
        sum((m.get("ratio_mean") or 0) * m.get("ratio_count", 0) for m in local_metrics) / count
        if count
        else None
    )
    for key, operation in (("ratio_min", min), ("ratio_max", max)):
        values = [m[key] for m in local_metrics if m.get(key) is not None]
        metric[key] = operation(values) if values else None
    metric["ranks"] = {
        str(i): {
            "samples": r["samples"],
            "masked_tokens": r["masked_tokens"],
            "peak_gpu_memory_mb": r["peak_gpu_memory_mb"],
            "duration_seconds": r["duration_seconds"],
            "oom_retry_count": local_metrics[i].get("oom_retry_count", 0),
        }
        for i, r in enumerate(results)
    }
    merged["step_metrics"] = (metric,)
    return PolicyUpdateResult(**merged)


class SubprocessSolverTrainer:
    def __init__(self, role, config, *, seed=0):
        if role != "solver":
            raise ValueError("distributed trainer is Solver-only")
        config.validate()
        self.config, self.seed = config, seed

    def update(self, batch, *, step, relation_credits=()):
        subset = batch.metadata.get("training_selection_schema") in {
            "eligible_subset_v1",
            "independent_frontier_v2",
        }
        if (
            self.config.epochs != 1
            or len(batch.samples) < 2
            or (not subset and len(batch.samples) > self.config.mini_batch_size)
        ):
            raise ValueError(
                "distributed Solver currently requires one complete global mini, epoch=1"
            )
        root = self.config.checkpoint_root
        if (root / f"step-{step:08d}").exists():
            raise FileExistsError("refusing to overwrite an existing distributed checkpoint")
        run = root / "distributed-updates" / f"step-{step:08d}-{uuid.uuid4().hex[:8]}"
        run.mkdir(parents=True)
        _atomic_json(run / "batch.json", batch.to_dict())
        (run / "credits.jsonl").write_text(
            "".join(json.dumps(asdict(c)) + "\n" for c in relation_credits), encoding="utf-8"
        )
        payload = asdict(self.config)
        payload["base_model_path"] = str(self.config.base_model_path.resolve())
        payload["checkpoint_root"] = str(root.resolve())
        _atomic_json(run / "request.json", dict(config=payload, seed=self.seed, step=step))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.data_parallel_gpu_ids))
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "selfplay_graph_flowsteer.solver_parallel_trainer",
            str(run.resolve()),
        ]
        with (run / "workers.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = PolicyUpdateResult(**json.loads((run / "result.json").read_text()))
        destination = Path(result.checkpoint)
        for name in (
            "trainer_state.pt",
            "training_config.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            if not (destination / name).is_file():
                raise RuntimeError(f"incomplete distributed checkpoint: {name}")
        _atomic_json(destination / "training_config.json", payload)
        if self.config.publish_latest_checkpoint:
            _atomic_json(root / "latest.json", dict(path=str(destination), step=step))
        return result

    def close(self):
        pass


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def worker(run):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=50))
    trainer = None
    try:
        request = json.loads((run / "request.json").read_text())
        policy = request["config"]
        for key in ("checkpoint_root", "base_model_path"):
            policy[key] = Path(policy[key])
        config = PolicyTrainingConfig(**policy)
        batch = load_training_batch(run / "batch.json")
        shards = balanced_rollout_shards(batch.samples, dist.get_world_size())
        indices = shards[rank]

        def agree(value):
            values = [None] * dist.get_world_size()
            dist.all_gather_object(values, value)
            if any(v != values[0] for v in values):
                raise RuntimeError("distributed policy state mismatch")

        class Trainer(TransformersGRPOTrainer):
            def _before_gradient_clip(self, count):
                if count != len(indices):
                    raise RuntimeError("unexpected local optimizer boundary")
                print(
                    json.dumps(
                        dict(
                            event="gradient_sync_started",
                            rank=rank,
                            local_samples=count,
                            timestamp=time.time(),
                        )
                    ),
                    flush=True,
                )
                synchronize_mean_gradients(self.model, count, len(batch.samples))
                print(
                    json.dumps(
                        dict(event="gradient_sync_completed", rank=rank, timestamp=time.time())
                    ),
                    flush=True,
                )

            def _masked_policy_call_log_probs(self, call, **kwargs):
                values = super()._masked_policy_call_log_probs(call, **kwargs)
                if kwargs.get("require_grad") and not kwargs.get("reference"):
                    self._distributed_call_count = getattr(self, "_distributed_call_count", 0) + 1
                    if self._distributed_call_count % 20 == 0:
                        print(
                            json.dumps(
                                dict(
                                    event="ordinary_policy_calls_forwarded",
                                    rank=rank,
                                    calls=self._distributed_call_count,
                                    timestamp=time.time(),
                                )
                            ),
                            flush=True,
                        )
                return values

            def _save(self, step):
                if rank == 0:
                    return super()._save(step)
                return self.config.checkpoint_root / f"step-{step:08d}"

        local_config = replace(
            config,
            device=f"cuda:{rank}",
            data_parallel_gpu_ids=(),
            mini_batch_size=len(indices),
            micro_batch_size=min(config.micro_batch_size, len(indices)),
            raw_policy_backward_mode="call",
            publish_latest_checkpoint=False,
        )
        trainer = Trainer("solver", local_config, seed=request["seed"])
        initial = parameter_hash(trainer.model)
        agree((initial, trainer.optimizer_step_count, shards))
        before_step = trainer.optimizer_step_count
        dist.barrier()
        started = time.monotonic()
        result = trainer.update(
            TrainingBatch(
                batch.role, tuple(batch.samples[i] for i in indices), objective=batch.objective
            ),
            step=request["step"],
            relation_credits=load_relation_credits(run / "credits.jsonl"),
        )
        torch.cuda.synchronize()
        final = parameter_hash(trainer.model)
        agree((final, trainer.optimizer_step_count))
        if trainer.optimizer_step_count != before_step + 1:
            raise RuntimeError("optimizer state did not advance exactly once")
        row = dict(
            indices=indices,
            result=asdict(result),
            initial_hash=initial,
            final_hash=final,
            optimizer_step_before=before_step,
            optimizer_step_after=trainer.optimizer_step_count,
            elapsed=time.monotonic() - started,
        )
        _atomic_json(run / f"rank-{rank}.json", row)
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, row)
        if rank == 0:
            merged = aggregate_results(rows, batch, max(r["elapsed"] for r in rows))
            _atomic_json(run / "result.json", asdict(merged))
    finally:
        if trainer is not None:
            trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
