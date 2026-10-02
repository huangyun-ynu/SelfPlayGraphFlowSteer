"""Whole-rollout data parallelism; advantages are never recomputed per rank."""

from __future__ import annotations

from typing import Any


def balanced_rollout_shards(samples: tuple[Any, ...], world_size: int) -> list[list[int]]:
    if world_size < 1 or len(samples) < world_size:
        raise ValueError("each rank needs at least one rollout")
    ids = [s.rollout_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rollout IDs")

    def cost(index: int) -> float:
        lengths = [len(c.token_ids) for c in samples[index].policy_calls]
        if not lengths:
            lengths = [len(samples[index].token_ids)]
        # A policy call has substantial fixed launch/checkpoint/offload overhead in
        # addition to token-dependent attention work.  The 5k-token equivalent was
        # calibrated from the 70-rollout H800 replay: the old token-only split had
        # near-identical estimated loads, but 131 calls reached synchronization
        # 236 seconds before 232 calls.  Including call overhead balances both the
        # amount of sequence work and the number of independently replayed calls.
        return sum(5_000 + n + n * n / 8192 for n in lengths)

    shards: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for index in sorted(range(len(samples)), key=lambda i: (-cost(i), i)):
        rank = min(range(world_size), key=lambda r: (loads[r], len(shards[r]), r))
        shards[rank].append(index)
        loads[rank] += cost(index)
    return [sorted(shard) for shard in shards]


def synchronize_mean_gradients(model: Any, local_count: int, global_count: int) -> None:
    """SUM local mean gradients weighted by actual rollout count, before clipping."""
    import torch
    import torch.distributed as dist

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or local_count <= 0 or global_count <= 0:
        raise ValueError("invalid distributed gradient inputs")
    device = parameters[0].device
    count = torch.tensor(local_count, device=device, dtype=torch.int64)
    dist.all_reduce(count)
    if int(count) != global_count:
        raise ValueError("global sample count differs from rank sum")
    used = torch.tensor([p.grad is not None for p in parameters], device=device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, flag in zip(parameters, used.tolist(), strict=True) if flag]
    if not active:
        return
    flat = torch.cat(
        [
            (
                p.grad.detach().float() if p.grad is not None else torch.zeros_like(p).float()
            ).reshape(-1)
            for p in active
        ]
    )
    flat.mul_(local_count / global_count)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if not torch.isfinite(flat).all():
        raise RuntimeError("nonfinite synchronized gradient")
    offset = 0
    for p in active:
        gradient = flat[offset : offset + p.numel()].view_as(p).to(p.dtype)
        if p.grad is None:
            p.grad = gradient.clone()
        else:
            p.grad.copy_(gradient)
        offset += p.numel()
