"""Pure accounting helpers; never choose Actions or change environment rewards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

BUDGET_SEMANTICS = "utf8_request_upper_bound_v1"


def request_budget_quote(request: Mapping[str, Any]) -> dict[str, Any]:
    """The same conservative quote used immediately before an OpenAI send."""
    body = {key: request[key] for key in ("messages", "tools") if key in request}
    input_bound = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 2048
    output_bound = int(request.get("max_completion_tokens", request.get("max_tokens", 2048)))
    if output_bound < 0:
        raise ValueError("negative request output budget")
    return {
        "semantics": BUDGET_SEMANTICS,
        "input_bound": input_bound,
        "output_bound": output_bound,
        "required_tokens": input_bound + output_bound,
    }


def closure_reserve_floor(total_limit: int, configured_minimum: int = 0) -> int:
    return min(
        max(0, total_limit), max(configured_minimum, min(32768, max(8192, total_limit // 4)))
    )


def observed_request_bounds(artifacts: Sequence[Any]) -> list[int]:
    """Read budget quotes, not provider token usage (a different estimator)."""
    bounds = []
    for artifact in artifacts:
        for event in artifact.backend_request_events:
            quote = event.get("request_token_budget", {})
            if quote.get("semantics") == BUDGET_SEMANTICS:
                bounds.append(int(quote["required_tokens"]))
        for diagnostic in artifact.protocol_diagnostics:
            quote = diagnostic.get("request_token_budget", {})
            if quote.get("semantics") == BUDGET_SEMANTICS:
                bounds.append(int(quote["required_tokens"]))
    return bounds


def budget_partition(
    *,
    total_limit: int,
    spent: int,
    configured_minimum: int,
    call_count: int,
    closure: bool,
    observed_bounds: Sequence[int] = (),
) -> dict[str, Any]:
    remaining = max(0, total_limit - spent)
    # Two recent request upper bounds are a planning reserve for an Action and
    # a report, not a promise that two future requests finish the task.
    reserve = (
        0
        if closure
        else min(
            remaining,
            max(
                closure_reserve_floor(total_limit, configured_minimum),
                2 * max(observed_bounds, default=0),
            ),
        )
    )
    spendable = remaining - reserve
    return {
        "budget_schema": "webshop_partition_v1",
        "phase": "closure" if closure else "exploration",
        "remaining_worker_tokens": remaining,
        "reserved_closure_tokens": reserve,
        "spendable_tokens": spendable,
        "call_count": max(1, call_count),
        "per_execution_credit": spendable // max(1, call_count),
        "request_admission": "authoritative_after_request_serialization",
    }


def request_admission(
    quote: Mapping[str, Any],
    *,
    credit: int,
    spent: int = 0,
    pre_reserved_closure_tokens: int | None = None,
) -> dict[str, Any]:
    # Canvas already withheld the baseline reserve. Protect only the additional
    # amount revealed by the ACTUAL request; never subtract that baseline twice.
    extra = (
        max(0, 2 * int(quote["required_tokens"]) - pre_reserved_closure_tokens)
        if pre_reserved_closure_tokens is not None
        else 0
    )
    required = int(quote["required_tokens"]) + extra
    return {
        **quote,
        "credit": credit,
        "spent": spent,
        "pre_reserved_closure_tokens": pre_reserved_closure_tokens,
        "additional_closure_reserve": extra,
        "required_with_reserve": required,
        "admitted": spent + required <= credit,
    }


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
