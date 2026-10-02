"""Pure accounting helpers; never choose Actions or change environment rewards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def execution_accounting(
    *,
    events: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    token_in: int,
    token_out: int,
    action_attempts: int,
) -> dict[str, Any]:
    dispatched = sum(
        e.get("event") == "backend_request_success"
        or (e.get("event") == "backend_request_failure" and e.get("stage") == "backend_request")
        for e in events
    )
    blocked = [d for d in diagnostics if d.get("no_request_dispatched") is True]
    zero_confirmed = bool(
        blocked and not dispatched and not token_in and not token_out and not action_attempts
    )
    # Missing provider usage alone never proves that no request was made.
    count = dispatched if dispatched or zero_confirmed else None
    return {
        "model_request_count": count,
        "model_request_count_known": count is not None,
        "action_attempt_count": action_attempts,
        "stop_stage": (
            "request_admission_before_first_call"
            if zero_confirmed
            else "request_admission_after_calls"
            if blocked and dispatched
            else "request_admission_unknown_prior_calls"
            if blocked
            else "worker_returned"
        ),
        "last_request_budget": dict(blocked[-1].get("request_token_budget", {})) if blocked else {},
    }
