from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError

from .deadline import WorkerWallClockLimitExceeded

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(
        r"(?i)(\b(?:authorization|api[_ -]?key|auth[_ -]?token|access[_ -]?token)"
        r"\b\s*[:=]\s*)[^\s,;]+"
    ),
    re.compile(
        r'(?i)("(?:authorization|api[_ -]?key|auth[_ -]?token|access[_ -]?token)"'
        r"\s*:\s*)\"[^\"]*\""
    ),
)


def sanitize_backend_detail(value: object, *, max_chars: int = 1200) -> str:
    """Return bounded diagnostic text without credentials or authorization values."""

    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)\\bBearer"):
            text = pattern.sub("Bearer [REDACTED]", text)
        elif pattern.pattern.startswith("(?i)\\bsk-"):
            text = pattern.sub("sk-[REDACTED]", text)
        else:
            text = pattern.sub(r"\1[REDACTED]", text)
    return text[:max_chars]


@dataclass(frozen=True)
class BackendFailureClassification:
    """Controller-owned classification of one backend or local scheduling failure."""

    backend_failure: bool
    origin: str
    kind: str
    retryable: bool
    counts_toward_route_circuit: bool
    disable_route: bool = False
    stage: str = "backend_request"
    route: str = ""
    exception_type: str = ""
    status_code: int | None = None
    provider_code: int | None = None
    message: str = ""
    request_id: str = ""
    queue_wait_s: float | None = None
    upstream_elapsed_s: float | None = None
    total_elapsed_s: float | None = None
    request_budget_s: float | None = None
    retry_after_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value for key, value in asdict(self).items() if value is not None and value != ""
        }

    def with_context(self, **changes: Any) -> BackendFailureClassification:
        return replace(self, **changes)


class EnvironmentServiceError(RuntimeError):
    """Environment transport failure, never an LLM provider failure."""

    def __init__(
        self, message: str, *, service: str, operation: str, status_code: int | None = None
    ) -> None:
        self.service = service
        self.operation = operation
        self.status_code = status_code
        super().__init__(message)


class BackendRequestError(RuntimeError):
    """A typed request failure retaining safe timing and route attribution."""

    def __init__(
        self,
        classification: BackendFailureClassification,
        *,
        request_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.classification = classification
        self.request_events = tuple(dict(event) for event in (request_events or ()))
        route = classification.route or "unassigned"
        status = (
            f" HTTP {classification.status_code}" if classification.status_code is not None else ""
        )
        super().__init__(
            f"backend {classification.kind}{status} at {classification.stage} "
            f"on route {route}: {classification.message or classification.exception_type}"
        )

    def to_dict(self) -> dict[str, Any]:
        return self.classification.to_dict()


def classify_backend_failure(
    exc: BaseException,
    *,
    stage: str = "",
    route: str = "",
    detail: object | None = None,
    queue_wait_s: float | None = None,
    upstream_elapsed_s: float | None = None,
    total_elapsed_s: float | None = None,
    request_budget_s: float | None = None,
) -> BackendFailureClassification:
    """Classify exception chains without treating controller deadlines as route faults."""

    if isinstance(exc, BackendRequestError):
        existing = exc.classification
        return existing.with_context(
            stage=stage or existing.stage,
            route=route or existing.route,
            queue_wait_s=(queue_wait_s if queue_wait_s is not None else existing.queue_wait_s),
            upstream_elapsed_s=(
                upstream_elapsed_s
                if upstream_elapsed_s is not None
                else existing.upstream_elapsed_s
            ),
            total_elapsed_s=(
                total_elapsed_s if total_elapsed_s is not None else existing.total_elapsed_s
            ),
            request_budget_s=(
                request_budget_s if request_budget_s is not None else existing.request_budget_s
            ),
        )
    if isinstance(exc, WorkerWallClockLimitExceeded):
        return BackendFailureClassification(
            backend_failure=False,
            origin="rollout_controller",
            kind=str(exc.reason),
            retryable=False,
            counts_toward_route_circuit=False,
            stage=str(exc.stage),
            route=route,
            exception_type=type(exc).__name__,
            message=sanitize_backend_detail(exc),
            total_elapsed_s=float(exc.elapsed_s),
        )

    chain = _exception_chain(exc)
    environment_error = next(
        (item for item in chain if isinstance(item, EnvironmentServiceError)), None
    )
    if environment_error is not None:
        return BackendFailureClassification(
            backend_failure=False,
            origin="environment_service",
            kind="environment_service_failure",
            retryable=False,
            counts_toward_route_circuit=False,
            disable_route=False,
            stage=environment_error.service + ":" + environment_error.operation,
            exception_type=type(environment_error).__name__,
            status_code=environment_error.status_code,
            message=sanitize_backend_detail(environment_error),
        )
    status_code = _status_code(chain)
    provider_code = _provider_code(chain)
    request_id = _request_id(chain)
    retry_after_s = _retry_after_s(chain)
    exception_type = type(exc).__name__
    safe_detail = sanitize_backend_detail(
        detail if detail is not None else _diagnostic_detail(chain)
    )
    folded = safe_detail.casefold()
    names = {type(item).__name__ for item in chain}

    resolved_stage = stage or "backend_request"
    common = {
        "stage": resolved_stage,
        "route": route,
        "exception_type": exception_type,
        "status_code": status_code,
        "provider_code": provider_code,
        "message": safe_detail,
        "request_id": request_id,
        "queue_wait_s": queue_wait_s,
        "upstream_elapsed_s": upstream_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "request_budget_s": request_budget_s,
        "retry_after_s": retry_after_s,
    }
    if resolved_stage == "backend_queue" and _is_timeout_or_connection(chain):
        return BackendFailureClassification(
            backend_failure=True,
            origin="local_queue",
            kind="queue_timeout",
            retryable=True,
            counts_toward_route_circuit=False,
            **common,
        )
    if route == "minimax" and provider_code is not None and provider_code != 0:
        return _classify_minimax_provider_code(provider_code, common)
    if "model not found" in folded or "model_not_found" in folded:
        return BackendFailureClassification(
            backend_failure=True,
            origin="route_configuration",
            kind="model_not_found",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    # Some intermediary routes wrap an upstream account-disable response in an
    # outer HTTP 400.  This is not a malformed model request: continuing to
    # admit Worker calls to that route cannot recover and would contaminate a
    # collection with infrastructure failures.  Detect the provider statement
    # before the generic 400/422 branch and permanently open this route's
    # circuit, just as an ordinary 401/403 account failure would do.
    if any(
        marker in folded
        for marker in (
            "service has been disabled",
            "service is disabled in this account",
            "account has been disabled",
            "account disabled",
        )
    ):
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_account",
            kind="account_disabled",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if "ssrf_blocked" in folded or "目标地址不可达" in safe_detail:
        return BackendFailureClassification(
            backend_failure=True,
            origin="relay",
            kind="relay_network_policy",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if _is_tls_configuration_failure(names, folded):
        return BackendFailureClassification(
            backend_failure=True,
            origin="client_network",
            kind="tls_failure",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if _is_tls_transport_failure(names, folded):
        # A generic TLS transport reset (for example OpenSSL's "record layer
        # failure") is commonly an intermittent connection failure.  It is
        # materially different from a certificate verification/configuration
        # failure: retain circuit accounting, but allow the caller's bounded
        # backend retry policy to recover it instead of permanently disabling
        # a route after one otherwise healthy request stream.
        return BackendFailureClassification(
            backend_failure=True,
            origin="client_network",
            kind="tls_failure",
            retryable=True,
            counts_toward_route_circuit=True,
            disable_route=False,
            **common,
        )
    # Relays may report temporary upstream account-pool capacity as HTTP 403.
    # This specific provider statement is not evidence of invalid credentials.
    # Permit request-local failover without poisoning persistent route health.
    if "all available accounts exhausted" in folded:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_capacity",
            kind="account_pool_exhausted",
            retryable=True,
            counts_toward_route_circuit=False,
            disable_route=False,
            **common,
        )
    if status_code in {401, 403}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="route_configuration",
            kind="auth_failure",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if status_code == 402:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_account",
            kind="quota_exhausted",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if status_code == 404:
        return BackendFailureClassification(
            backend_failure=True,
            origin="route_configuration",
            kind="route_not_found",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if status_code in {400, 422}:
        return BackendFailureClassification(
            backend_failure=True,
            origin=("provider_request" if route in {"deepseek", "minimax"} else "relay"),
            kind="invalid_request",
            retryable=False,
            counts_toward_route_circuit=False,
            **common,
        )
    if status_code == 424:
        return BackendFailureClassification(
            backend_failure=True,
            origin="relay",
            kind="upstream_424",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if status_code == 429 or "RateLimitError" in names:
        return BackendFailureClassification(
            backend_failure=True,
            origin=("provider_capacity" if route in {"deepseek", "minimax"} else "relay"),
            kind="rate_limit",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if (status_code is not None and status_code >= 500) or names & {
        "InternalServerError",
        "ServiceUnavailableError",
    }:
        return BackendFailureClassification(
            backend_failure=True,
            origin="upstream_model",
            kind="upstream_5xx",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if names & {
        "APITimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutError",
    } or any(isinstance(item, TimeoutError) for item in chain):
        return BackendFailureClassification(
            backend_failure=True,
            origin="client_network",
            kind="request_timeout",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if names & {
        "APIConnectionError",
        "ConnectError",
        "ReadError",
        "ConnectionError",
    } or any(isinstance(item, (ConnectionError, URLError)) for item in chain):
        return BackendFailureClassification(
            backend_failure=True,
            origin="client_network",
            kind="connection_error",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    return BackendFailureClassification(
        backend_failure=False,
        origin="unknown",
        kind="unknown",
        retryable=False,
        counts_toward_route_circuit=False,
        **common,
    )


def _classify_minimax_provider_code(
    provider_code: int,
    common: dict[str, Any],
) -> BackendFailureClassification:
    if provider_code in {1004, 2049}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="route_configuration",
            kind="auth_failure",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if provider_code in {1008, 2056}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_account",
            kind="quota_exhausted",
            retryable=False,
            counts_toward_route_circuit=True,
            disable_route=True,
            **common,
        )
    if provider_code in {1001}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="upstream_model",
            kind="request_timeout",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if provider_code in {1002, 2045}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_capacity",
            kind="rate_limit",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if provider_code == 1041:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_capacity",
            kind="connection_limit",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if provider_code in {1000, 1024, 1033}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="upstream_model",
            kind="provider_internal",
            retryable=True,
            counts_toward_route_circuit=True,
            **common,
        )
    if provider_code in {1026, 1027}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_policy",
            kind="content_policy",
            retryable=False,
            counts_toward_route_circuit=False,
            **common,
        )
    if provider_code in {1039, 1042, 2013}:
        return BackendFailureClassification(
            backend_failure=True,
            origin="provider_request",
            kind="invalid_request",
            retryable=False,
            counts_toward_route_circuit=False,
            **common,
        )
    return BackendFailureClassification(
        backend_failure=True,
        origin="provider",
        kind="provider_error",
        retryable=False,
        counts_toward_route_circuit=False,
        **common,
    )


def classify_legacy_backend_event(
    event: dict[str, Any],
) -> list[BackendFailureClassification]:
    """Replay historical JSONL events through the current typed taxonomy."""

    failure = event.get("backend_failure", {})
    if not isinstance(failure, dict):
        failure = {}
    structured = failure.get("failure_details", ())
    if isinstance(structured, list) and structured:
        return [_classification_from_dict(value) for value in structured if isinstance(value, dict)]
    message = str(event.get("message", ""))
    routes = failure.get("routes", event.get("routes", ()))
    route = ""
    if isinstance(routes, (list, tuple)) and routes:
        route = str(routes[0])
    elif isinstance(event.get("route"), str):
        route = str(event["route"])
    if not route:
        route_match = re.search(r"\broutes=([^;)\s]+)", message)
        if route_match:
            route = route_match.group(1).split(",", maxsplit=1)[0]
    raw_types = failure.get("failure_types", ())
    if not isinstance(raw_types, (list, tuple)):
        raw_types = ()
    error_types = [str(value) for value in raw_types if str(value).strip()]
    if not error_types:
        types_match = re.search(r"\bfailure_types=([^;)\s]+)", message)
        if types_match:
            error_types = [value for value in types_match.group(1).split(",") if value]
        else:
            error_types = [str(event.get("error_type", "RuntimeError"))]
    status_match = re.search(r"(?:HTTP|Error code:)\s*(\d{3})", message, re.I)
    explicit_status = int(status_match.group(1)) if status_match else None
    classifications: list[BackendFailureClassification] = []
    for error_type in error_types:
        normalized_name = re.sub(r"[^A-Za-z0-9_]", "", error_type) or "RuntimeError"
        legacy_type = type(normalized_name, (RuntimeError,), {})
        error = legacy_type(message or normalized_name)
        status_code = explicit_status or _legacy_status_for_type(normalized_name)
        if status_code is not None:
            error.status_code = status_code
        classifications.append(
            classify_backend_failure(
                error,
                stage=str(event.get("stage", "primary_worker_execution")),
                route=route,
                detail=f"{normalized_name}: {message}",
            )
        )
    return classifications


def _classification_from_dict(value: dict[str, Any]) -> BackendFailureClassification:
    fields = BackendFailureClassification.__dataclass_fields__
    payload = {key: item for key, item in value.items() if key in fields}
    payload.setdefault("backend_failure", True)
    payload.setdefault("origin", "unknown")
    payload.setdefault("kind", "unknown")
    payload.setdefault("retryable", False)
    payload.setdefault("counts_toward_route_circuit", False)
    return BackendFailureClassification(**payload)


def _legacy_status_for_type(error_type: str) -> int | None:
    if error_type == "RateLimitError":
        return 429
    if error_type in {"InternalServerError", "ServiceUnavailableError"}:
        return 500
    return None


def _exception_chain(exc: BaseException) -> list[BaseException]:
    values: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        values.append(current)
        current = current.__cause__ or current.__context__
    return values


def _status_code(chain: list[BaseException]) -> int | None:
    for item in chain:
        value = item.code if isinstance(item, HTTPError) else getattr(item, "status_code", None)
        if isinstance(value, int):
            return value
        response = getattr(item, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def _provider_code(chain: list[BaseException]) -> int | None:
    for item in chain:
        value = getattr(item, "provider_code", None)
        if isinstance(value, int):
            return value
    return None


def _request_id(chain: list[BaseException]) -> str:
    for item in chain:
        direct = getattr(item, "request_id", None)
        if direct:
            return sanitize_backend_detail(direct, max_chars=160)
        for source in (getattr(item, "headers", None), getattr(item, "response", None)):
            headers = getattr(source, "headers", source)
            if headers is None:
                continue
            for key in (
                "x-request-id",
                "request-id",
                "x-goog-request-id",
                "trace-id",
                "trace_id",
                "x-trace-id",
            ):
                try:
                    value = headers.get(key)
                except AttributeError:
                    value = None
                if value:
                    return sanitize_backend_detail(value, max_chars=160)
    return ""


def _retry_after_s(chain: list[BaseException]) -> float | None:
    for item in chain:
        for source in (getattr(item, "headers", None), getattr(item, "response", None)):
            headers = getattr(source, "headers", source)
            if headers is None:
                continue
            try:
                value = headers.get("Retry-After")
            except AttributeError:
                value = None
            if value is None:
                continue
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                try:
                    date = parsedate_to_datetime(str(value))
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=UTC)
                    seconds = (date - datetime.now(UTC)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    return None
            return max(0.0, seconds) if math.isfinite(seconds) else None
    return None


def _diagnostic_detail(chain: list[BaseException]) -> str:
    values: list[str] = []
    for item in chain:
        values.append(str(item))
        body = getattr(item, "body", None)
        if body:
            values.append(sanitize_backend_detail(body))
    return " | caused by: ".join(value for value in values if value)


def _is_timeout_or_connection(chain: list[BaseException]) -> bool:
    return any(isinstance(item, (TimeoutError, ConnectionError, URLError)) for item in chain)


def _is_tls_configuration_failure(names: set[str], folded: str) -> bool:
    return bool(
        names & {"CertificateError", "SSLCertVerificationError"}
        or "certificate verify failed" in folded
        or "ssl" in folded
        and "certificate" in folded
    )


def _is_tls_transport_failure(names: set[str], folded: str) -> bool:
    return bool(
        names & {"SSLError"}
        or "record layer failure" in folded
        or "tls" in folded
        and "handshake" in folded
    )
