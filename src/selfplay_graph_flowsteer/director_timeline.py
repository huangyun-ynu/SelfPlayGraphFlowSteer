"""Opt-in chronological Director input for collection/accuracy experiments.

This does not merge PPO calls. Provider token-prefix checks attest whether a
future timeline trainer may reuse the saved behavior probabilities.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT_MODE = "snapshot_dedup"
APPEND_CONTEXT_MODE = "append_only"
DELTA_CONTEXT_MODE = "delta_timeline"
TIMELINE_CONTEXT_MODES = frozenset({APPEND_CONTEXT_MODE, DELTA_CONTEXT_MODE})


def persist_context_policy(output_dir: Path, *, resume: bool) -> None:
    """Do not mix context policies or template versions in one collection."""
    mode = director_context_mode()
    template = Path(__file__).resolve().parents[2] / "configs/templates/director_append_only.jinja"
    policy = {
        "mode": mode,
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest()
        if mode in TIMELINE_CONTEXT_MODES
        else None,
        "ppo_call_layout": "per_call",
    }
    marker = output_dir / "director_context_policy.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != policy:
            raise ValueError(
                "Director context policy differs from saved collection; use a new output directory"
            )
        return
    if mode in TIMELINE_CONTEXT_MODES and resume:
        raise ValueError("cannot resume an unmarked collection with append-only Director context")
    marker.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def director_context_mode() -> str:
    mode = os.environ.get("SPGFS_DIRECTOR_CONTEXT_MODE", DEFAULT_CONTEXT_MODE)
    if mode not in {DEFAULT_CONTEXT_MODE, *TIMELINE_CONTEXT_MODES}:
        raise ValueError(f"unsupported SPGFS_DIRECTOR_CONTEXT_MODE: {mode!r}")
    return mode


def timeline_assistant_content(tokenizer: Any, prompt_ids, completion_ids) -> str:
    """Keep the actual prefill and sampled bytes, including thinking and stops."""
    if tokenizer is None or not prompt_ids or not completion_ids:
        raise ValueError("append-only Director requires provider token IDs and a tokenizer")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    decode = lambda ids: tokenizer.decode(  # noqa: E731
        list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    prompt = decode(prompt_ids)
    delimiter = "<|im_start|>assistant\n"
    if delimiter not in prompt:
        raise ValueError("append-only Director prompt lacks an assistant generation prefix")
    prefill = prompt.rsplit(delimiter, 1)[1]
    return prefill + decode(completion_ids)


def timeline_prefix_audit(previous_ids, prompt_ids, completion_ids) -> dict[str, Any]:
    available = bool(prompt_ids and completion_ids)
    matches = (
        tuple(prompt_ids[: len(previous_ids)]) == tuple(previous_ids)
        if available and previous_ids
        else None
    )
    return {
        "provider_tokens_available": available,
        "previous_policy_token_count": len(previous_ids),
        "previous_policy_is_exact_prefix": matches,
        "timeline_merge_candidate": available and (not previous_ids or matches is True),
    }
