"""Separate trusted environment limits from the application's Action budget."""
from __future__ import annotations

from typing import Any

PINNED_WEBSHOP_REVISION = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"


def environment_step_capacity(state: object) -> tuple[str, int | None]:
    payload: dict[str, Any] = state if isinstance(state, dict) else {}
    if any(payload.get(key) is True for key in ("terminal", "done", "purchased")):
        return "finite", 0
    remaining = payload.get("remaining_steps")
    if type(remaining) is int and remaining >= 0:
        return "finite", remaining
    capability = payload.get("environment_step_limit")
    if (
        "remaining_steps" not in payload
        and isinstance(capability, dict)
        and capability.get("kind") == "unbounded"
        and capability.get("source_revision") == PINNED_WEBSHOP_REVISION
    ):
        return "unbounded", None
    return "unknown", None
