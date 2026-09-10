from __future__ import annotations

import ssl
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.backend_failures import (
    BackendRequestError,
    classify_backend_failure,
    classify_legacy_backend_event,
    sanitize_backend_detail,
)
from selfplay_graph_flowsteer.deadline import WorkerWallClockLimitExceeded


class _StatusError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.body = {"error": {"message": message}}
        self.response = SimpleNamespace(
            status_code=status_code,
            headers={"x-request-id": "request-123"},
        )
        super().__init__(message)


@pytest.mark.parametrize(
    ("status", "kind", "retryable", "circuit", "disable"),
    [
        (400, "invalid_request", False, False, False),
        (401, "auth_failure", False, True, True),
        (422, "invalid_request", False, False, False),
        (424, "upstream_424", True, True, False),
        (429, "rate_limit", True, True, False),
        (500, "upstream_5xx", True, True, False),
        (502, "upstream_5xx", True, True, False),
        (503, "upstream_5xx", True, True, False),
    ],
)
def test_http_failure_taxonomy(
    status: int,
    kind: str,
    retryable: bool,
    circuit: bool,
    disable: bool,
) -> None:
    result = classify_backend_failure(_StatusError(status, "relay failure"), route="grok")

    assert result.backend_failure is True
    assert result.kind == kind
    assert result.retryable is retryable
    assert result.counts_toward_route_circuit is circuit
    assert result.disable_route is disable
    assert result.route == "grok"
    assert result.request_id == "request-123"


def test_backend_queue_timeout_does_not_poison_provider_circuit() -> None:
    result = classify_backend_failure(
        TimeoutError("queue full"),
        stage="backend_queue",
        route="gemini",
        queue_wait_s=4.0,
        request_budget_s=4.0,
    )

    assert result.origin == "local_queue"
    assert result.kind == "queue_timeout"
    assert result.retryable is True
    assert result.counts_toward_route_circuit is False
    routed = classify_backend_failure(BackendRequestError(result), route="gemini")
    assert routed.stage == "backend_queue"
    assert routed.origin == "local_queue"


def test_generic_tls_record_failure_is_bounded_retryable() -> None:
    result = classify_backend_failure(
        ssl.SSLError("[SSL] record layer failure (_ssl.c:2590)"),
        route="deepseek",
    )

    assert result.kind == "tls_failure"
    assert result.retryable is True
    assert result.counts_toward_route_circuit is True
    assert result.disable_route is False


def test_tls_certificate_failure_still_disables_route() -> None:
    result = classify_backend_failure(
        ssl.SSLCertVerificationError("certificate verify failed"),
        route="deepseek",
    )

    assert result.kind == "tls_failure"
    assert result.retryable is False
    assert result.counts_toward_route_circuit is True
    assert result.disable_route is True


def test_rollout_deadline_is_not_a_backend_failure() -> None:
    result = classify_backend_failure(
        WorkerWallClockLimitExceeded(
            "hard stop",
            reason="hard_deadline",
            stage="application_solve_complete",
            elapsed_s=600.0,
            idle_s=0.0,
        ),
        route="grok",
    )

    assert result.backend_failure is False
    assert result.origin == "rollout_controller"
    assert result.kind == "hard_deadline"
    assert result.counts_toward_route_circuit is False


def test_model_not_found_disables_route_without_retry() -> None:
    result = classify_backend_failure(
        _StatusError(422, "model not found: invalid-name"), route="kiro"
    )

    assert result.kind == "model_not_found"
    assert result.retryable is False
    assert result.disable_route is True


def test_deepseek_official_insufficient_balance_disables_route() -> None:
    result = classify_backend_failure(_StatusError(402, "Insufficient Balance"), route="deepseek")

    assert result.origin == "provider_account"
    assert result.kind == "quota_exhausted"
    assert result.retryable is False
    assert result.counts_toward_route_circuit is True
    assert result.disable_route is True


def test_wrapped_account_disabled_error_disables_route() -> None:
    result = classify_backend_failure(
        _StatusError(
            400,
            "request invalid: upstream 403: This service has been disabled in this account",
        ),
        route="gemini",
    )

    assert result.origin == "provider_account"
    assert result.kind == "account_disabled"
    assert result.retryable is False
    assert result.counts_toward_route_circuit is True
    assert result.disable_route is True


@pytest.mark.parametrize(
    ("provider_code", "kind", "retryable", "circuit", "disable"),
    [
        (1001, "request_timeout", True, True, False),
        (1002, "rate_limit", True, True, False),
        (1004, "auth_failure", False, True, True),
        (1008, "quota_exhausted", False, True, True),
        (1024, "provider_internal", True, True, False),
        (1026, "content_policy", False, False, False),
        (1039, "invalid_request", False, False, False),
        (1041, "connection_limit", True, True, False),
        (2049, "auth_failure", False, True, True),
        (2056, "quota_exhausted", False, True, True),
    ],
)
def test_minimax_official_business_error_taxonomy(
    provider_code: int,
    kind: str,
    retryable: bool,
    circuit: bool,
    disable: bool,
) -> None:
    error = RuntimeError("MiniMax provider failure")
    error.provider_code = provider_code
    error.request_id = "trace-123"

    result = classify_backend_failure(error, route="minimax")

    assert result.provider_code == provider_code
    assert result.kind == kind
    assert result.retryable is retryable
    assert result.counts_toward_route_circuit is circuit
    assert result.disable_route is disable
    assert result.request_id == "trace-123"


def test_backend_error_preserves_typed_safe_payload() -> None:
    classification = classify_backend_failure(
        _StatusError(424, "upstream unavailable"), route="gemini"
    )
    error = BackendRequestError(
        classification,
        request_events=[{"event": "backend_request_failure", "kind": "upstream_424"}],
    )

    assert error.to_dict()["kind"] == "upstream_424"
    assert error.request_events[0]["kind"] == "upstream_424"


def test_backend_diagnostics_redact_credentials() -> None:
    detail = sanitize_backend_detail(
        'Authorization: Bearer secret-token api_key=topsecret "auth_token":"hidden"'
    )

    assert "secret-token" not in detail
    assert "topsecret" not in detail
    assert '"hidden"' not in detail
    assert "[REDACTED]" in detail


@pytest.mark.parametrize(
    ("failure_type", "message", "kind"),
    [
        ("InternalServerError", "service unavailable", "upstream_5xx"),
        ("RuntimeError", "Gemini request failed with HTTP 424", "upstream_424"),
        ("RateLimitError", "rate limited", "rate_limit"),
        ("APITimeoutError", "request timed out", "request_timeout"),
    ],
)
def test_historical_backend_event_replay(
    failure_type: str,
    message: str,
    kind: str,
) -> None:
    classifications = classify_legacy_backend_event(
        {
            "stage": "primary_worker_execution",
            "message": message,
            "backend_failure": {
                "routes": ["grok"],
                "failure_types": [failure_type],
            },
        }
    )

    assert len(classifications) == 1
    assert classifications[0].kind == kind
    assert classifications[0].route == "grok"


@pytest.mark.parametrize("status", [400, 403, 429, 503])
def test_account_pool_exhaustion_is_temporary_capacity(status):
    result = classify_backend_failure(
        _StatusError(status, "All available accounts exhausted"),
        route="gemini_uuapi_group1",
    )
    assert result.kind == "account_pool_exhausted"
    assert result.origin == "provider_capacity"
    assert result.retryable is True
    assert result.disable_route is False
    assert result.counts_toward_route_circuit is False


def test_real_forbidden_credentials_remain_auth_failure():
    result = classify_backend_failure(_StatusError(403, "Invalid API key"))
    assert result.kind == "auth_failure"
    assert result.retryable is False
