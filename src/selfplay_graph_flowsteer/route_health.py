from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PersistentRouteCircuitOpenError(RuntimeError):
    """No configured Worker route is eligible under persisted health state."""


class RouteHealthStore:
    """Atomic, process-safe route circuit state shared by rollout processes."""

    VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        cooldown_s: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if cooldown_s <= 0:
            raise ValueError("route health cooldown_s must be positive")
        self.path = Path(path)
        self.cooldown_s = float(cooldown_s)
        self.clock = clock
        self._thread_lock = threading.Lock()

    def available_routes(
        self, routes: Sequence[str]
    ) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
        payload = self.snapshot()
        now = self.clock()
        available: list[str] = []
        blocked: dict[str, dict[str, Any]] = {}
        route_states = payload["routes"]
        for raw_route in routes:
            route = str(raw_route).strip()
            if not route:
                continue
            state = route_states.get(route, {})
            if state.get("status") != "open":
                available.append(route)
                continue
            cooldown_until = float(state.get("cooldown_until_unix_s", 0.0))
            reason = "cooldown" if now < cooldown_until else "complex_probe_required"
            blocked[route] = {**state, "block_reason": reason}
        return tuple(available), blocked

    def available_scoped_routes(
        self,
        routes: Sequence[str],
        *,
        dataset: str,
        request_role: str,
    ) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
        """Filter early circuits without globally disabling an otherwise healthy route."""

        payload = self.snapshot()
        now = self.clock()
        available: list[str] = []
        blocked: dict[str, dict[str, Any]] = {}
        scopes = payload["scopes"]
        for raw_route in routes:
            route = str(raw_route).strip()
            if not route:
                continue
            key = _scope_key(route, dataset, request_role)
            state = scopes.get(key, {})
            if state.get("status") != "open":
                available.append(route)
                continue
            cooldown_until = float(state.get("cooldown_until_unix_s", 0.0))
            reason = "cooldown" if now < cooldown_until else "complex_probe_required"
            blocked[route] = {**state, "block_reason": reason}
        return tuple(available), blocked

    def record_scoped_outcome(
        self,
        route: str,
        *,
        dataset: str,
        request_role: str,
        success: bool,
        total_elapsed_s: float,
        slow_threshold_s: float,
        failure_counts_toward_circuit: bool,
        threshold: int,
        upstream_elapsed_s: float | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a route × dataset × request-role early-admission circuit."""

        route = str(route).strip() or "unknown"
        dataset = str(dataset).strip().casefold() or "unknown"
        request_role = str(request_role).strip().casefold() or "primary"
        elapsed = float(total_elapsed_s)
        upstream_elapsed = elapsed if upstream_elapsed_s is None else float(upstream_elapsed_s)
        slow_threshold = float(slow_threshold_s)
        if elapsed < 0 or upstream_elapsed < 0:
            raise ValueError("scoped route elapsed time must be non-negative")
        if slow_threshold <= 0:
            raise ValueError("scoped route slow threshold must be positive")
        if threshold <= 0:
            raise ValueError("scoped route failure threshold must be positive")
        # Provider circuits use upstream time, not local request-gate queue time.
        slow = bool(success and upstream_elapsed >= slow_threshold)
        unhealthy = slow or (not success and failure_counts_toward_circuit)
        now = self.clock()
        key = _scope_key(route, dataset, request_role)
        if not unhealthy:
            previous = dict(self.snapshot()["scopes"].get(key, {}))
            if (
                previous.get("status") != "open"
                and int(previous.get("consecutive_unhealthy", 0)) == 0
            ):
                # The overwhelmingly common fast-success path needs no durable
                # state transition and therefore avoids an fsync per API request.
                return {
                    **previous,
                    "scope_key": key,
                    "route": route,
                    "dataset": dataset,
                    "request_role": request_role,
                    "status": "closed",
                    "probe_required": False,
                    "consecutive_unhealthy": 0,
                    "last_outcome": "success",
                }
        with self._locked_payload() as payload:
            scopes = payload["scopes"]
            previous = dict(scopes.get(key, {}))
            consecutive = int(previous.get("consecutive_unhealthy", 0)) + 1 if unhealthy else 0
            opened = previous.get("status") == "open" or consecutive >= threshold
            state = {
                **previous,
                "scope_key": key,
                "route": route,
                "dataset": dataset,
                "request_role": request_role,
                "status": "open" if opened else "closed",
                "probe_required": bool(opened),
                "consecutive_unhealthy": consecutive,
                "total_unhealthy": int(previous.get("total_unhealthy", 0)) + int(unhealthy),
                "last_outcome": ("slow_success" if slow else ("success" if success else "failure")),
                "last_total_elapsed_s": elapsed,
                "last_upstream_elapsed_s": upstream_elapsed,
                "last_outcome_at": _iso8601(now),
                "last_outcome_unix_s": now,
            }
            if evidence:
                state["last_outcome_evidence"] = dict(evidence)
            if opened:
                state.setdefault("opened_at", _iso8601(now))
                state.setdefault("opened_at_unix_s", now)
                next_cooldown = (
                    now + self.cooldown_s
                    if unhealthy
                    else float(previous.get("cooldown_until_unix_s", 0.0))
                )
                state["cooldown_until_unix_s"] = max(
                    float(previous.get("cooldown_until_unix_s", 0.0)),
                    next_cooldown,
                )
                state["cooldown_until"] = _iso8601(float(state["cooldown_until_unix_s"]))
            scopes[key] = state
            return dict(state)

    def record_scoped_probe_success(
        self,
        route: str,
        *,
        dataset: str,
        request_role: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Close one cooled scoped circuit after a bounded complex Worker probe."""

        if str(evidence.get("probe_kind", "")) != "complex_worker":
            raise ValueError("scoped route recovery requires probe_kind=complex_worker")
        if evidence.get("success") is not True:
            raise ValueError("scoped route recovery requires a successful probe")
        key = _scope_key(route, dataset, request_role)
        now = self.clock()
        with self._locked_payload() as payload:
            scopes = payload["scopes"]
            previous = dict(scopes.get(key, {}))
            if previous.get("status") != "open":
                raise ValueError(f"scoped route {key!r} is not open")
            cooldown_until = float(previous.get("cooldown_until_unix_s", 0.0))
            if now < cooldown_until:
                raise RuntimeError(
                    f"scoped route {key!r} cooldown remains active for {cooldown_until - now:.1f}s"
                )
            state = {
                **previous,
                "status": "closed",
                "probe_required": False,
                "consecutive_unhealthy": 0,
                "last_probe_at": _iso8601(now),
                "last_probe_unix_s": now,
                "last_probe_evidence": dict(evidence),
            }
            scopes[key] = state
            return dict(state)

    def record_failure(
        self,
        route: str,
        *,
        failure_type: str,
        threshold: int,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        route = route.strip()
        if not route:
            route = "unknown"
        if threshold <= 0:
            raise ValueError("route failure threshold must be positive")
        now = self.clock()
        with self._locked_payload() as payload:
            routes = payload["routes"]
            previous = dict(routes.get(route, {}))
            consecutive = int(previous.get("consecutive_failures", 0)) + 1
            opened = previous.get("status") == "open" or consecutive >= threshold
            state = {
                **previous,
                "route": route,
                "status": "open" if opened else "closed",
                "probe_required": bool(opened),
                "consecutive_failures": consecutive,
                "total_failures": int(previous.get("total_failures", 0)) + 1,
                "last_failure_type": str(failure_type),
                "last_failure_at": _iso8601(now),
                "last_failure_unix_s": now,
            }
            if evidence:
                state["last_failure_evidence"] = dict(evidence)
            if opened:
                state.setdefault("opened_at", _iso8601(now))
                state.setdefault("opened_at_unix_s", now)
                state["cooldown_until_unix_s"] = max(
                    float(previous.get("cooldown_until_unix_s", 0.0)),
                    now + self.cooldown_s,
                )
                state["cooldown_until"] = _iso8601(float(state["cooldown_until_unix_s"]))
            routes[route] = state
            return dict(state)

    def record_runtime_success(
        self, route: str, *, evidence: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Reset a closed route's consecutive count; never close an open route."""
        route = route.strip()
        if not route:
            return {}
        now = self.clock()
        with self._locked_payload() as payload:
            routes = payload["routes"]
            previous = dict(routes.get(route, {}))
            state = {
                **previous,
                "route": route,
                "last_runtime_success_at": _iso8601(now),
                "last_runtime_success_unix_s": now,
            }
            if evidence:
                state["last_runtime_success_evidence"] = dict(evidence)
            if previous.get("status") != "open":
                state.update(
                    status="closed",
                    probe_required=False,
                    consecutive_failures=0,
                )
            routes[route] = state
            return dict(state)

    def record_complex_probe_success(
        self, route: str, *, evidence: dict[str, Any]
    ) -> dict[str, Any]:
        """Close a cooled circuit only after an explicit complex Worker probe."""
        route = route.strip()
        if not route:
            raise ValueError("probe route cannot be empty")
        if str(evidence.get("probe_kind", "")) != "complex_worker":
            raise ValueError("route recovery requires probe_kind=complex_worker")
        if evidence.get("success") is not True:
            raise ValueError("route recovery requires a successful complex Worker probe")
        now = self.clock()
        with self._locked_payload() as payload:
            routes = payload["routes"]
            previous = dict(routes.get(route, {}))
            if previous.get("status") != "open":
                raise ValueError(f"route {route!r} is not open")
            cooldown_until = float(previous.get("cooldown_until_unix_s", 0.0))
            if now < cooldown_until:
                raise RuntimeError(
                    f"route {route!r} cooldown remains active for {cooldown_until - now:.1f}s"
                )
            state = {
                **previous,
                "route": route,
                "status": "closed",
                "probe_required": False,
                "consecutive_failures": 0,
                "last_probe_at": _iso8601(now),
                "last_probe_unix_s": now,
                "last_probe_evidence": dict(evidence),
            }
            routes[route] = state
            return dict(state)

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock:
            if not self.path.exists():
                return self._empty_payload()
            return self._read_payload()

    @contextmanager
    def _locked_payload(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self._thread_lock, lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                payload = self._read_payload() if self.path.exists() else self._empty_payload()
                yield payload
                payload["updated_at"] = _iso8601(self.clock())
                self._atomic_write(payload)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_payload(self) -> dict[str, Any]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            raise ValueError(f"unsupported route health state: {self.path}")
        routes = payload.get("routes")
        if not isinstance(routes, dict):
            raise ValueError(f"route health state has no routes object: {self.path}")
        scopes = payload.setdefault("scopes", {})
        if not isinstance(scopes, dict):
            raise ValueError(f"route health state has no scopes object: {self.path}")
        return payload

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def _empty_payload(cls) -> dict[str, Any]:
        return {"version": cls.VERSION, "routes": {}, "scopes": {}}


def _scope_key(route: str, dataset: str, request_role: str) -> str:
    return "::".join(
        (
            str(route).strip() or "unknown",
            str(dataset).strip().casefold() or "unknown",
            str(request_role).strip().casefold() or "primary",
        )
    )


def _iso8601(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
