"""Opt-in numerical test candidate; not connected to formal training entrypoints.

Only schedules/evaluates already admitted raw calls. Admission, PPO terms,
optimizer boundaries and checkpoint publication remain the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .rollouts import TokenizedPolicyCall, TrainingSample


@dataclass(frozen=True)
class CallItem:
    sample_index: int
    call_index: int
    call: TokenizedPolicyCall
    trajectory_targets: int
    trajectory_count: int

    @property
    def weight(self) -> float:
        return 1.0 / (self.trajectory_count * self.trajectory_targets)


def plan_call_batches(
    samples: Iterable[TrainingSample],
    *,
    max_calls: int = 2,
    short_call_tokens: int = 4096,
    max_padded_tokens: int = 8192,
    max_length_ratio: float = 1.25,
) -> tuple[tuple[CallItem, ...], ...]:
    """Length-sort short calls; keep long and constrained calls singleton.

    Token limits bound batching, not input length: singleton inputs are never
    truncated. Counts must come from the full selected logical optimizer batch.
    """
    if min(max_calls, short_call_tokens, max_padded_tokens) < 1 or max_length_ratio < 1:
        raise ValueError("invalid call batching limits")
    samples = tuple(samples)
    if len({sample.rollout_id for sample in samples}) != len(samples):
        raise ValueError("duplicate trajectory")
    short, singleton = [], []
    for sample_index, sample in enumerate(samples):
        if not sample.policy_calls:
            raise ValueError("candidate requires raw policy calls")
        targets = [sum(call.action_mask[1:]) for call in sample.policy_calls]
        if not all(targets):
            raise ValueError("every call must have next-token action targets")
        for call_index, call in enumerate(sample.policy_calls):
            item = CallItem(sample_index, call_index, call, sum(targets), len(samples))
            if call.relation_token_span is not None or len(call.token_ids) > short_call_tokens:
                singleton.append((item,))
            else:
                short.append(item)
    short.sort(key=lambda item: (len(item.call.token_ids), item.sample_index, item.call_index))
    batches, pending = [], []
    for item in short:
        width = len(item.call.token_ids)
        if pending and (
            len(pending) >= max_calls
            or width * (len(pending) + 1) > max_padded_tokens
            or width / len(pending[0].call.token_ids) > max_length_ratio
        ):
            batches.append(tuple(pending))
            pending = []
        pending.append(item)
    if pending:
        batches.append(tuple(pending))
    return tuple(batches + singleton)


def evaluate_calls(
    trainer: Any,
    calls: Iterable[TokenizedPolicyCall],
    *,
    require_grad: bool = False,
    reference: bool = False,
    include_entropy: bool = True,
    entropy_requires_grad: bool = False,
) -> list[tuple[Any, Any]]:
    """Right-pad independent calls and project only the union of action positions."""
    calls = tuple(calls)
    if not calls or any(call.relation_token_span is not None for call in calls):
        raise ValueError("expected ordinary calls; binary relations require original evaluator")
    if len(calls) == 1:
        values, entropy = trainer._masked_policy_call_log_probs(
            calls[0],
            require_grad=require_grad,
            reference=reference,
            include_entropy=True,
            entropy_requires_grad=entropy_requires_grad,
        )
        return [(values, entropy if include_entropy else None)]
    torch = trainer.torch
    positions = [[i for i, keep in enumerate(call.action_mask[1:]) if keep] for call in calls]
    if any(not selected for selected in positions):
        raise ValueError("each call must have action targets")
    union = sorted({position for selected in positions for position in selected})
    indices = {position: index for index, position in enumerate(union)}
    width = max(len(call.token_ids) for call in calls)
    pad = trainer.tokenizer.pad_token_id
    if pad is None:
        pad = trainer.tokenizer.eos_token_id or 0
    inputs = torch.full((len(calls), width), pad, dtype=torch.long, device=trainer.config.device)
    mask = torch.zeros_like(inputs)
    for row, call in enumerate(calls):
        inputs[row, : len(call.token_ids)] = torch.tensor(call.token_ids, device=inputs.device)
        mask[row, : len(call.token_ids)] = 1
    adapter = trainer.model.disable_adapter() if reference else nullcontext()
    gradient = nullcontext() if require_grad else torch.no_grad()
    offload = (
        torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
        if require_grad and trainer._activation_cpu_offload_enabled(width)
        else nullcontext()
    )
    with adapter, gradient, offload:
        logits = trainer._selected_sequence_logits(
            inputs, mask, torch.tensor(union, dtype=torch.long, device=inputs.device)
        )
        results = []
        for row, (call, selected) in enumerate(zip(calls, positions, strict=True)):
            log_probs = torch.log_softmax(logits[row, [indices[p] for p in selected]], dim=-1)
            labels = torch.tensor([call.token_ids[p + 1] for p in selected], device=inputs.device)
            values = log_probs.gather(-1, labels[:, None]).squeeze(-1)
            source = log_probs if entropy_requires_grad else log_probs.detach()
            entropy = -(source.exp() * source).sum(-1) if include_entropy else None
            if not require_grad:
                values = values.detach().cpu()
                entropy = entropy.detach().cpu() if entropy is not None else None
            results.append((values, entropy))
    return results


def accumulate_call_gradients(
    trainer: Any,
    plan: tuple[tuple[CallItem, ...], ...],
    loss_terms: Callable[[CallItem, Any, Any], Any],
    *,
    binary_evaluator: Callable[[CallItem], tuple[Any, Any]] | None = None,
    entropy_requires_grad: bool = False,
) -> dict[str, Any]:
    """Backprop each micro-batch; never zero, clip, step or publish parameters.

    loss_terms returns the SUM of existing token loss terms for one call, before
    trajectory normalization. A failure propagates; caller must discard partial
    gradients and replay the full logical batch before stepping.
    """
    if binary_evaluator is None and any(
        item.call.relation_token_span is not None for batch in plan for item in batch
    ):
        raise ValueError("binary evaluator required before backward")
    value, backward_count = 0.0, 0
    for batch in plan:
        if len(batch) == 1 and batch[0].call.relation_token_span is not None:
            outputs = [binary_evaluator(batch[0])]
        else:
            outputs = evaluate_calls(
                trainer,
                [item.call for item in batch],
                require_grad=True,
                entropy_requires_grad=entropy_requires_grad,
            )
        loss = sum(
            loss_terms(item, log_probs, entropy) * item.weight
            for item, (log_probs, entropy) in zip(batch, outputs, strict=True)
        )
        context = (
            trainer.torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if trainer._activation_cpu_offload_enabled(
                max(len(item.call.token_ids) for item in batch)
            )
            else nullcontext()
        )
        with context:
            loss.backward()
        value += float(loss.detach())
        backward_count += 1
        del outputs, loss
    return {
        "loss": value,
        "backward_count": backward_count,
        "batch_sizes": [len(batch) for batch in plan],
    }
