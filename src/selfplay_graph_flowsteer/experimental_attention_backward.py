"""Isolated FP32 SDPA derivative candidate. Not installed in formal training."""

from __future__ import annotations

import math


def fp32_attention_backward(
    grad_out,
    query,
    key,
    value,
    *,
    bias=None,
    causal=False,
    scale=None,
    query_chunk=128,
    gqa_round_before_sum=False,
):
    """Recompute softmax and its exact mathematical derivative in FP32.

    Only dropout-free dense attention, including grouped-query attention.
    Query chunks bound score memory; key/value gradients accumulate in fixed order.
    Forward remains the original caller's operation. No gradient is cached.
    """
    import torch

    if (
        query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or query.shape[0] != key.shape[0]
        or key.shape[:3] != value.shape[:3]
        or key.shape[1] < 1
        or query.shape[1] % key.shape[1]
    ):
        raise ValueError("requires dense attention with compatible head groups")
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    if query.shape[-1] != key.shape[-1] or grad_out.shape != (*query.shape[:3], value.shape[-1]):
        raise ValueError("incompatible attention dimensions")
    q, k, v, go = (x.float() for x in (query, key, value, grad_out))
    key_heads = key.shape[1]
    groups = query.shape[1] // key_heads
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    factor = 1 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    mask = None
    if bias is not None and bias.numel():
        mask = torch.broadcast_to(bias, (*q.shape[:3], k.shape[-2]))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            for start in range(0, q.shape[-2], query_chunk):
                end = min(start + query_chunk, q.shape[-2])
                qc, gc = q[:, :, start:end], go[:, :, start:end]
                logits = (qc @ k.transpose(-1, -2)) * factor
                if mask is not None:
                    selected = mask[:, :, start:end]
                    if selected.dtype == torch.bool:
                        logits.masked_fill_(~selected, -torch.inf)
                    else:
                        logits.add_(selected.float())
                if causal:
                    allowed = (
                        torch.arange(k.shape[-2], device=q.device)[None, :]
                        <= torch.arange(start, end, device=q.device)[:, None]
                    )
                    logits.masked_fill_(~allowed, -torch.inf)
                probabilities = torch.softmax(logits, dim=-1).nan_to_num(0.0)
                dp = gc @ v.transpose(-1, -2)
                ds = probabilities * (dp - (probabilities * dp).sum(-1, keepdim=True))
                dq[:, :, start:end] = (ds @ k) * factor
                dk.add_((ds.transpose(-1, -2) @ qc) * factor)
                dv.add_(probabilities.transpose(-1, -2) @ gc)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    if groups > 1:
        if gqa_round_before_sum:
            # Match the expanded-KV graph: each head's returned gradient crosses
            # the input dtype boundary BEFORE repeat_kv's grouped reduction.
            dk = dk.to(key.dtype).float()
            dv = dv.to(value.dtype).float()
        dk = dk.reshape(dk.shape[0], key_heads, groups, *dk.shape[2:]).sum(2)
        dv = dv.reshape(dv.shape[0], key_heads, groups, *dv.shape[2:]).sum(2)
    return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype)


def make_fp32_backward_mode(*, observer=None, gqa_round_before_sum=False):
    """Intercept only cuDNN SDPA backward; reject unsupported signatures."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class FP32Backward(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) != "aten._scaled_dot_product_cudnn_attention_backward.default":
                return func(*args, **(kwargs or {}))
            arguments = {
                argument.name: value
                for argument, value in zip(func._schema.arguments, args, strict=False)
            }
            arguments.update(kwargs or {})
            if arguments["dropout_p"] != 0:
                raise ValueError("FP32 candidate does not support attention dropout")
            if arguments["cum_seq_q"] is not None or arguments["cum_seq_k"] is not None:
                raise ValueError("FP32 candidate does not support packed variable-length attention")
            self.calls += 1
            result = fp32_attention_backward(
                arguments["grad_out"],
                arguments["query"],
                arguments["key"],
                arguments["value"],
                bias=arguments["attn_bias"],
                causal=arguments["is_causal"],
                scale=arguments.get("scale"),
                gqa_round_before_sum=gqa_round_before_sum,
            )
            if observer is not None:
                observer(arguments, result)
            return result

    return FP32Backward()
