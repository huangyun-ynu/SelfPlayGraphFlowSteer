from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import threading
import time
import uuid
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .backend_failures import EnvironmentServiceError
from .observability import TaskSpec, VerificationResult

_DIRECT_OPENER = build_opener(ProxyHandler({}))

_PUBLIC_SEARCH_EVIDENCE_SEMANTICS = {
    "title_variant_scope": "default_preview_only",
    "title_option_values_can_exclude_product": False,
    "option_availability": "unknown_until_product_page",
    "plain_language_rule": (
        "A search title shows only one preview variant. Do not accept or reject a "
        "product's requested size, color, or other selectable options from that title."
    ),
    "option_verification_action_kind": "open_product",
    "candidate_selection": "agent_decides; no candidate is forced or preselected",
}


class WebShopClient(Protocol):
    def create_session(self, goal_id: str, *, seed: int) -> dict[str, Any]: ...

    def search(self, session_id: str, query: str) -> dict[str, Any]: ...

    def click(self, session_id: str, target_id: str) -> dict[str, Any]: ...

    def commit(
        self,
        session_id: str,
        target_id: str,
        *,
        commit_id: str,
    ) -> dict[str, Any]: ...

    def close_session(self, session_id: str) -> None: ...


@dataclass(frozen=True)
class WebShopHTTPClient:
    service_url: str
    timeout_s: float = 10.0
    _deadline: ContextVar = field(
        default_factory=lambda: ContextVar("webshop_deadline", default=None),
        compare=False,
        repr=False,
    )
    _epoch: str | None = field(default=None, init=False, compare=False, repr=False)
    _capability_checked: bool = field(default=False, init=False, compare=False, repr=False)
    retry_events: list = field(default_factory=list, init=False, compare=False, repr=False)

    def set_deadline_context(self, deadline):
        self._deadline.set(deadline)

    def __post_init__(self) -> None:
        if not self.service_url.startswith(("http://", "https://")):
            raise ValueError("webshop service_url must be an HTTP(S) URL")
        if self.timeout_s <= 0:
            raise ValueError("webshop timeout_s must be positive")

    def create_session(self, goal_id: str, *, seed: int) -> dict[str, Any]:
        return self._request("POST", "/sessions", {"goal_id": goal_id, "seed": seed})

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", None)

    def search(self, session_id: str, query: str) -> dict[str, Any]:
        return self._request("POST", f"/sessions/{session_id}/search", {"query": query})

    def click(self, session_id: str, target_id: str) -> dict[str, Any]:
        return self._request("POST", f"/sessions/{session_id}/click", {"target_id": target_id})

    def commit(
        self,
        session_id: str,
        target_id: str,
        *,
        commit_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/sessions/{session_id}/commit",
            {"target_id": target_id, "commit_id": commit_id},
        )

    def close_session(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}", None)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        mutation = method != "GET"
        if mutation and not self._capability_checked:
            health = self._request("GET", "/health", None)
            if health.get("idempotency_protocol") == "webshop-request-v1":
                object.__setattr__(self, "_epoch", health.get("request_epoch"))
            object.__setattr__(self, "_capability_checked", True)
        key = uuid.uuid4().hex
        headers = (
            {"Idempotency-Key": key, "X-Webshop-Epoch": self._epoch}
            if mutation and self._epoch
            else {}
        )
        limit = 3 if not mutation or self._epoch else 1
        deadline = self._deadline.get()
        pause = deadline.pause_no_progress("webshop_request") if deadline else nullcontext()
        with pause:
            for attempt in range(1, limit + 1):
                started = time.monotonic()
                timeout = self.timeout_s
                if deadline:
                    timeout = min(timeout, deadline.hard_remaining_s("webshop_request"))
                try:
                    result = self._request_once(method, path, payload, headers, timeout)
                except EnvironmentServiceError as exc:
                    if deadline:
                        deadline.record_failed_request(started, time.monotonic())
                    retryable = exc.status_code is None or exc.status_code in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    event = {
                        "service": "webshop",
                        "operation": method + " " + path,
                        "request_id": key,
                        "attempt": attempt,
                        "elapsed_s": time.monotonic() - started,
                        "status": "failed",
                        "status_code": exc.status_code,
                        "will_retry": retryable and attempt < limit,
                    }
                    self.retry_events.append(event)
                    exc.request_events = list(self.retry_events)
                    if not event["will_retry"]:
                        raise
                    delay = 0.25 * attempt
                    if deadline:
                        delay = min(delay, deadline.hard_remaining_s("webshop_retry_wait"))
                    time.sleep(delay)
                    continue
                self.retry_events.append(
                    {
                        "service": "webshop",
                        "operation": method + " " + path,
                        "request_id": key,
                        "attempt": attempt,
                        "status": "success",
                        "elapsed_s": time.monotonic() - started,
                    }
                )
                if deadline:
                    deadline.check("webshop_request_complete")
                return result
        raise AssertionError("unreachable retry loop")

    def _request_once(self, method, path, payload, extra_headers, timeout):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.service_url.rstrip("/") + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", **extra_headers},
        )
        try:
            # The sidecar is a local service.  Never let inherited proxy
            # variables redirect these requests through an external gateway.
            with _DIRECT_OPENER.open(request, timeout=timeout) as response:  # noqa: S310
                text = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise EnvironmentServiceError(
                f"WebShop service HTTP {exc.code}: {detail}",
                service="webshop",
                operation=method + " " + path,
                status_code=exc.code,
            ) from exc
        except (URLError, OSError, HTTPException) as exc:
            raise EnvironmentServiceError(
                f"WebShop service unavailable: {exc}",
                service="webshop",
                operation=method + " " + path,
            ) from exc
        if not text.strip():
            return {}
        result = json.loads(text)
        if not isinstance(result, dict):
            raise RuntimeError("WebShop service response must be a JSON object")
        return result


@dataclass(frozen=True)
class _PendingPurchase:
    session_id: str
    target_id: str
    staged_at: float


@dataclass
class WebShopSessionLifecycle:
    """Own one continuous WebShop episode for the complete rollout.

    The first Worker that enters the environment becomes its sole mutable-state
    owner.  Later executions of that same Worker resume the live session;
    other Workers may still contribute as stateless planners but cannot open a
    competing shopping episode.  A staged purchase is retained until Canvas
    selects the owner as output and commits it.
    """

    client: WebShopClient
    max_observation_chars: int = 12_000
    max_pending_sessions: int = 8
    pending_ttl_s: float = 900.0
    stage_purchases: bool = True
    search_observation_mode: str = "structured_only"
    _task: TaskSpec | None = None
    _owner_agent: str | None = None
    _active_agent: str | None = None
    _active_session: str | None = None
    _execution_open: bool = False
    _active_pending_target: str | None = None
    _pending_sessions: dict[str, _PendingPurchase] = field(default_factory=dict)
    _results: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Runtime-owned, bounded decision memory for the one continuous rollout.
    # The environment page already survives same-owner Worker revisions; this
    # journal keeps the public search/inspection/evidence history continuous as
    # well.  It is cleared with the bound task and is never sent to the sidecar.
    _transaction_journals: dict[str, dict[str, Any]] = field(default_factory=dict)
    _committer_agent: str | None = None
    _purchase_committed_by: str | None = None
    _environment_fingerprint: str = ""
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def bind_task(self, task: TaskSpec) -> None:
        with self._lock:
            if self.search_observation_mode not in {
                "legacy",
                "retain_page_text",
                "structured_only",
            }:
                raise ValueError(
                    "WebShop search_observation_mode must be legacy, "
                    "retain_page_text, or structured_only"
                )
            if self.max_pending_sessions <= 0:
                raise ValueError("WebShop max_pending_sessions must be positive")
            if self.pending_ttl_s <= 0:
                raise ValueError("WebShop pending_ttl_s must be positive")
            self.close_all()
            goal_id = _trusted_goal_id(task)
            if not goal_id:
                raise ValueError("WebShop task requires trusted metadata.goal_id")
            health = getattr(self.client, "health", None)
            status: dict[str, Any] = {}
            if callable(health):
                status = health()
                if status.get("status") != "ok":
                    raise RuntimeError("WebShop service health check failed")
            self._task = task
            self._results = {}
            self._transaction_journals = {}
            self._pending_sessions = {}
            self._owner_agent = None
            self._committer_agent = None
            self._purchase_committed_by = None
            self._environment_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "goal_id": goal_id,
                        "goal_fingerprint": status.get("goal_fingerprint", ""),
                        "search_observation_mode": self.search_observation_mode,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()

    @property
    def environment_fingerprint(self) -> str:
        return self._environment_fingerprint

    def reserve_full_graph_owner(self, agent_id: str) -> None:
        """Pin the recorded output owner before a fresh intervention graph starts.

        This grants no Actions or purchase: non-owners remain stateless and the
        selected Worker must independently stage a legal purchase.
        """
        with self._lock:
            if self._task is None or self._active_session or self._owner_agent is not None:
                raise RuntimeError("full-graph owner reservation requires a fresh bound episode")
            self._owner_agent = str(agent_id)

    def set_committer(self, agent_id: str) -> None:
        """Atomically transfer purchase authority before the first successful commit."""

        with self._lock:
            agent_id = str(agent_id).strip()
            if not agent_id:
                raise ValueError("WebShop committer agent_id cannot be empty")
            if self._purchase_committed_by and self._purchase_committed_by != agent_id:
                # A successful purchase is an irreversible environment terminal
                # state. A late Canvas output reassignment must not invalidate
                # that outcome or crash an otherwise completed trajectory.
                return
            self._committer_agent = agent_id

    @property
    def owner_agent(self) -> str | None:
        with self._lock:
            return self._owner_agent

    def allows_agent(self, agent_id: str) -> bool:
        """Return whether ``agent_id`` may mutate this rollout's episode."""

        with self._lock:
            owner = self.owner_agent
            return owner is None or owner == str(agent_id)

    def closure_budget_context(self, agent_id: str) -> dict[str, Any]:
        """Runtime-only authority/limit check; never opens or replaces a session."""
        with self._lock:
            state = self._results.get(agent_id, {})
            if (
                self._task is None
                or self._owner_agent != agent_id
                or self._committer_agent != agent_id
                or self._active_session is None
                or self._execution_open
                or agent_id in self._pending_sessions
                or self._active_pending_target is not None
                or self._purchase_committed_by is not None
                or any(
                    state.get(key) for key in ("commit_pending", "purchased", "done", "terminal")
                )
            ):
                return {"eligible": False, "reason": "no_live_output_owner_session"}
            remaining = state.get("remaining_steps")
            if type(remaining) is not int:
                return {"eligible": False, "reason": "official_remaining_steps_unknown"}
            return {
                "eligible": True,
                "reason": "same_session_output_owner",
                "session_id": self._active_session,
                "official_remaining_steps": max(0, remaining),
            }

    def begin_execution(self, *, agent_id: str, seed: int, revision: bool) -> dict[str, Any]:
        with self._lock:
            if self._task is None:
                raise RuntimeError("WebShop lifecycle has no bound task")
            self._cleanup_expired_pending()
            agent_id = str(agent_id)
            owner = self.owner_agent
            if owner is not None and owner != agent_id:
                raise PermissionError(
                    f"WebShop episode is owned by {owner}; {agent_id} is stateless"
                )
            pending = self._pending_sessions.get(agent_id)
            if pending is not None:
                raise RuntimeError(
                    "WebShop purchase is already staged; Canvas must select the owner "
                    "as output instead of revising the episode"
                )
            if self._execution_open:
                raise RuntimeError("WebShop lifecycle already has an active Worker execution")
            if self._active_session is not None:
                self._execution_open = True
                state = dict(self._results.get(agent_id, {}))
                state.update(
                    {
                        "commit_pending": False,
                        "commit_ready": False,
                        "commit_authorized": agent_id == self._committer_agent,
                        "purchase_committed": self._purchase_committed_by is not None,
                        "environment_owner": agent_id,
                        "session_reused": True,
                        "execution_revision": bool(revision),
                    }
                )
                state["_runtime_transaction_journal"] = self._transaction_journals.setdefault(
                    agent_id, {}
                )
                return state
            payload = self.client.create_session(_trusted_goal_id(self._task), seed=seed)
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                raise RuntimeError("WebShop session creation returned no session_id")
            self._active_agent = agent_id
            self._owner_agent = agent_id
            self._active_session = session_id
            self._execution_open = True
            self._active_pending_target = None
            state = self._bounded(payload)
            state["commit_pending"] = False
            state["commit_ready"] = False
            self._results[agent_id] = dict(state)
            state["commit_authorized"] = agent_id == self._committer_agent
            state["purchase_committed"] = self._purchase_committed_by is not None
            state["environment_owner"] = agent_id
            state["session_reused"] = False
            state["execution_revision"] = bool(revision)
            state["_runtime_transaction_journal"] = self._transaction_journals.setdefault(
                agent_id, {}
            )
            return state

    def end_execution(self) -> None:
        with self._lock:
            self._cleanup_expired_pending()
            if self._active_agent and self._active_session and self._active_pending_target:
                agent_id = self._active_agent
                replacement = _PendingPurchase(
                    session_id=self._active_session,
                    target_id=self._active_pending_target,
                    staged_at=time.monotonic(),
                )
                previous = self._pending_sessions.pop(agent_id, None)
                if previous is not None:
                    self.client.close_session(previous.session_id)
                self._pending_sessions[agent_id] = replacement
                self._active_agent = None
                self._active_session = None
                self._active_pending_target = None
                self._execution_open = False
                self._enforce_pending_limit()
                return
            # The Worker turn ended, not the WebShop episode.  Preserve the
            # live page so a same-owner prompt revision or output closure pass
            # can continue from the exact public environment state.
            self._execution_open = False

    def search(self, query: str) -> dict[str, Any]:
        with self._lock:
            if self._purchase_committed_by is not None:
                return self._committed_result()
            session_id, agent_id = self._active()
            self._active_pending_target = None
            result = self._bounded(self.client.search(session_id, query))
            self._results[agent_id] = result
            return result

    def click(
        self,
        target_id: str,
        *,
        purchase_evidence: object = None,
        state_version: object = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._purchase_committed_by is not None:
                return self._committed_result()
            session_id, agent_id = self._active()
            current = self._results.get(agent_id, {})
            version_rejection = _webshop_state_version_rejection(
                current,
                state_version,
            )
            if version_rejection is not None:
                raise ValueError(version_rejection[1])
            requested_target_id = str(target_id)
            target_id, transport_repair_kind = _resolve_visible_target_id(
                requested_target_id,
                current.get("valid_subactions", []),
            )
            target = next(
                (
                    item
                    for item in current.get("valid_subactions", [])
                    if isinstance(item, dict) and str(item.get("target_id", "")) == str(target_id)
                ),
                None,
            )
            is_purchase = bool(target and str(target.get("kind", "")) == "purchase")
            validated_purchase_evidence: dict[str, Any] | None = None
            if is_purchase:
                validated_purchase_evidence, rejection = _validate_purchase_evidence(
                    purchase_evidence
                )
                if rejection is not None:
                    rejected = dict(current)
                    rejected.pop("action_effect", None)
                    rejected.update(
                        {
                            "commit_pending": False,
                            "commit_ready": False,
                            "purchase_executed": False,
                            "purchase_evidence_status": {
                                "accepted": False,
                                **rejection,
                            },
                            "termination_reason": "active",
                        }
                    )
                    self._results[agent_id] = rejected
                    return dict(rejected)
            if is_purchase and agent_id != self._committer_agent:
                if not self.stage_purchases:
                    raise PermissionError(
                        "purchase requires the Canvas-selected output Agent; report the "
                        "candidate artifact and wait for commit authorization"
                    )
                self._active_pending_target = str(target_id)
                staged = dict(current)
                staged.pop("target_id_transport_repair", None)
                # ``current`` describes the preceding environment transition.
                # Staging Buy Now does not execute a sidecar action, so carrying
                # its action_effect forward would falsely attribute (for example)
                # the previous option selection to the purchase-stage request.
                staged.pop("action_effect", None)
                staged.update(
                    {
                        "commit_pending": True,
                        "commit_ready": True,
                        "purchase_executed": False,
                        "purchase_evidence_status": {
                            "accepted": True,
                            "evidence": validated_purchase_evidence,
                        },
                        "termination_reason": "purchase_staged",
                    }
                )
                if transport_repair_kind:
                    staged["target_id_transport_repair"] = {
                        "applied": True,
                        "kind": transport_repair_kind,
                        "requested_target_id": requested_target_id,
                        "resolved_target_id": target_id,
                    }
                self._results[agent_id] = staged
                return dict(staged)
            self._active_pending_target = None
            result = self._bounded(self.client.click(session_id, target_id))
            result["commit_pending"] = False
            result["commit_ready"] = False
            if validated_purchase_evidence is not None:
                result["purchase_evidence_status"] = {
                    "accepted": True,
                    "evidence": validated_purchase_evidence,
                }
            if transport_repair_kind:
                result["target_id_transport_repair"] = {
                    "applied": True,
                    "kind": transport_repair_kind,
                    "requested_target_id": requested_target_id,
                    "resolved_target_id": target_id,
                }
            self._results[agent_id] = result
            if bool(result.get("purchased")):
                self._purchase_committed_by = agent_id
            return result

    def preflight_click(
        self,
        target_id: object,
        *,
        state_version: object = None,
    ) -> dict[str, Any] | None:
        """Validate the current public action surface without mutating it."""

        with self._lock:
            _session_id, agent_id = self._active()
            current = self._results.get(agent_id, {})
            version_rejection = _webshop_state_version_rejection(
                current,
                state_version,
            )
            if version_rejection is not None:
                code, message = version_rejection
                return {
                    "code": code,
                    "message": message,
                    "details": _webshop_current_action_surface(current),
                }
            requested = str(target_id or "").strip()
            resolved, _repair_kind = _resolve_visible_target_id(
                requested,
                current.get("valid_subactions", []),
            )
            visible = {
                str(item.get("target_id", ""))
                for item in current.get("valid_subactions", [])
                if isinstance(item, dict) and str(item.get("target_id", ""))
            }
            if resolved not in visible:
                return {
                    "code": "webshop_target_not_in_current_state",
                    "message": (
                        "target_id is stale or not valid on the current page; choose an Action "
                        "from the latest valid_subactions list."
                    ),
                    "details": _webshop_current_action_surface(current),
                }
            return None

    def commit_ready_agents(self) -> tuple[str, ...]:
        with self._lock:
            self._cleanup_expired_pending()
            return tuple(sorted(self._pending_sessions))

    def discard_pending(self, agent_id: str) -> None:
        with self._lock:
            pending = self._pending_sessions.pop(str(agent_id), None)
            if pending is not None:
                self.client.close_session(pending.session_id)

    def commit_pending(self, agent_id: str) -> dict[str, Any]:
        """Commit one staged Buy Now action without another Worker execution."""

        with self._lock:
            agent_id = str(agent_id).strip()
            if not agent_id:
                raise ValueError("WebShop commit agent_id cannot be empty")
            if self._purchase_committed_by is not None:
                if self._purchase_committed_by != agent_id:
                    raise RuntimeError("WebShop purchase was already committed by another Agent")
                return self._committed_result()
            self._cleanup_expired_pending()
            pending = self._pending_sessions.get(agent_id)
            if pending is None:
                raise ValueError(f"WebShop Agent {agent_id} has no staged purchase")
            self._committer_agent = agent_id
            commit_id = hashlib.sha256(
                (
                    f"{self._environment_fingerprint}\0{agent_id}\0"
                    f"{pending.session_id}\0{pending.target_id}"
                ).encode()
            ).hexdigest()
            commit = getattr(self.client, "commit", None)
            if callable(commit):
                payload = commit(
                    pending.session_id,
                    pending.target_id,
                    commit_id=commit_id,
                )
            else:
                # Compatibility for in-process test clients and older local
                # sidecars. The official HTTP client uses the idempotent path.
                payload = self.client.click(pending.session_id, pending.target_id)
            result = self._bounded(payload)
            # Keep the Agent's uncertainty declaration with the committed outcome.
            # It is audit evidence, never an override of the environment reward.
            purchase_evidence_status = self._results.get(agent_id, {}).get(
                "purchase_evidence_status"
            )
            if isinstance(purchase_evidence_status, dict):
                result["purchase_evidence_status"] = copy.deepcopy(purchase_evidence_status)
            result["commit_pending"] = False
            result["commit_ready"] = False
            result["purchase_executed"] = True
            result["purchase_committed"] = bool(result.get("purchased", False))
            result["terminal"] = bool(result.get("purchased", False) or result.get("done", False))
            if bool(result.get("purchased", False)):
                self._purchase_committed_by = agent_id
                result["purchase_committed_by"] = agent_id
            self._results[agent_id] = result
            try:
                self._close_pending_sessions()
            except Exception as exc:  # noqa: BLE001 - outcome survives cleanup telemetry
                result["session_cleanup_error"] = type(exc).__name__
                result["session_cleanup_detail"] = str(exc)[:500]
                self._results[agent_id] = result
            return dict(result)

    def result_for(self, agent_id: str | None) -> dict[str, Any]:
        with self._lock:
            self._cleanup_expired_pending()
            if self._purchase_committed_by is not None:
                return self._committed_result()
            if not agent_id:
                return _empty_result("missing_output_agent")
            return dict(self._results.get(agent_id, _empty_result("agent_never_executed")))

    def _committed_result(self) -> dict[str, Any]:
        committed_by = self._purchase_committed_by
        if committed_by is None:
            raise RuntimeError("WebShop purchase has not been committed")
        result = dict(self._results.get(committed_by, {}))
        result["purchase_committed"] = True
        result["purchase_committed_by"] = committed_by
        result["terminal"] = True
        return result

    def close_all(self) -> None:
        with self._lock:
            self._close_active()
            self._close_pending_sessions()
            self._owner_agent = None
            self._active_agent = None
            self._execution_open = False
            self._transaction_journals = {}
            self._task = None

    def _active(self) -> tuple[str, str]:
        if not self._execution_open or not self._active_session or not self._active_agent:
            raise RuntimeError("WebShop Action called outside an active Worker execution")
        return self._active_session, self._active_agent

    def _close_active(self) -> None:
        if self._active_session:
            try:
                self.client.close_session(self._active_session)
            finally:
                self._active_session = None
                self._active_agent = None
                self._active_pending_target = None
                self._execution_open = False

    def _close_pending_sessions(self) -> None:
        pending = list(self._pending_sessions.values())
        self._pending_sessions = {}
        first_error: Exception | None = None
        for item in pending:
            try:
                self.client.close_session(item.session_id)
            except Exception as exc:  # noqa: BLE001 - close every retained session
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def _cleanup_expired_pending(self) -> None:
        now = time.monotonic()
        expired = [
            agent_id
            for agent_id, pending in self._pending_sessions.items()
            if now - pending.staged_at >= self.pending_ttl_s
        ]
        for agent_id in expired:
            pending = self._pending_sessions.pop(agent_id)
            self.client.close_session(pending.session_id)
            result = dict(self._results.get(agent_id, {}))
            result.update(
                {
                    "commit_pending": False,
                    "commit_ready": False,
                    "termination_reason": "pending_purchase_expired",
                }
            )
            self._results[agent_id] = result

    def _enforce_pending_limit(self) -> None:
        while len(self._pending_sessions) > self.max_pending_sessions:
            agent_id, pending = min(
                self._pending_sessions.items(),
                key=lambda item: item[1].staged_at,
            )
            del self._pending_sessions[agent_id]
            self.client.close_session(pending.session_id)
            result = dict(self._results.get(agent_id, {}))
            result.update(
                {
                    "commit_pending": False,
                    "commit_ready": False,
                    "termination_reason": "pending_purchase_capacity_evicted",
                }
            )
            self._results[agent_id] = result

    def _bounded(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result.pop("session_id", None)
        subactions = result.get("valid_subactions", [])
        if not isinstance(subactions, list):
            raise RuntimeError("WebShop valid_subactions must be a list")
        if self.search_observation_mode == "legacy":
            result["valid_subactions"] = [
                _legacy_webshop_subaction(item) for item in subactions
            ]
            result.pop("search_evidence_semantics", None)
        else:
            structured_subactions: list[object] = []
            for item in subactions:
                if isinstance(item, dict):
                    copied = dict(item)
                    if copied.get("kind") == "open_product":
                        copied.setdefault(
                            "option_availability_evidence",
                            "unchecked_product_page",
                        )
                        copied.setdefault(
                            "option_constraints_status",
                            "unknown_until_opened",
                        )
                        copied.setdefault(
                            "preview_mismatch_is_exclusion_evidence",
                            False,
                        )
                        copied.setdefault(
                            "inspection_required_for_option_verification",
                            True,
                        )
                    structured_subactions.append(copied)
                else:
                    structured_subactions.append(item)
            result["valid_subactions"] = structured_subactions
            if str(result.get("page_type", "")) == "search_results":
                result.setdefault(
                    "search_evidence_semantics",
                    dict(_PUBLIC_SEARCH_EVIDENCE_SEMANTICS),
                )
        page_text = str(result.get("page_text", ""))
        if (
            self.search_observation_mode == "structured_only"
            and str(result.get("page_type", "")) == "search_results"
        ):
            page_text = _compact_search_results_text(page_text)
        if self.max_observation_chars > 0:
            result["page_text"] = page_text[: self.max_observation_chars]
            result["observation_truncated"] = len(page_text) > self.max_observation_chars
        else:
            result["page_text"] = page_text
            result["observation_truncated"] = False
        return result


_SEARCH_PRODUCT_MARKER = re.compile(
    r"^\[button\]\s+[A-Z0-9]{10}\s+\[button_\]\s*$",
    re.IGNORECASE,
)
_STRUCTURED_SEARCH_FIELDS = frozenset(
    {
        "asin",
        "title",
        "price",
        "evidence_scope",
        "variant_status",
        "options_authoritative_on",
        "option_availability_evidence",
        "option_constraints_status",
        "preview_mismatch_is_exclusion_evidence",
        "inspection_required_for_option_verification",
    }
)


def _legacy_webshop_subaction(item: object) -> object:
    if not isinstance(item, dict) or item.get("kind") != "open_product":
        return item
    return {key: value for key, value in item.items() if key not in _STRUCTURED_SEARCH_FIELDS}


def _compact_search_results_text(page_text: str) -> str:
    """Keep the public search header/navigation and move product rows to Actions."""

    lines = str(page_text).splitlines()
    for index, line in enumerate(lines):
        if _SEARCH_PRODUCT_MARKER.match(line.strip()):
            compact = "\n".join(lines[:index]).strip()
            return compact or "Search results"
    # Never discard an unfamiliar renderer format. Structured-only is a layout
    # transformation, not authority to hide public evidence.
    return str(page_text)


def _resolve_visible_target_id(
    requested: str,
    visible_subactions: object,
) -> tuple[str, str | None]:
    """Undo representation escaping only on a unique currently legal target.

    Some native tool-call providers return a copied target_id containing the
    JSON display backslashes around embedded quotes or HTML entities around
    punctuation. This is a transport-level representation error, not authority
    to choose another target: repair is allowed only when the decoded value
    exactly matches one currently visible target_id and the requested value
    itself does not.
    """

    items = (
        [item for item in visible_subactions if isinstance(item, dict)]
        if isinstance(visible_subactions, list)
        else []
    )
    visible = {str(item.get("target_id", "")) for item in items if str(item.get("target_id", ""))}
    if requested in visible:
        return requested, None
    decoded = html.unescape(requested).replace(r"\"", '"').replace(r"\'", "'")
    matches = [candidate for candidate in visible if candidate == decoded]
    if len(matches) == 1:
        return matches[0], "visible_target_representation_escape"

    # An open_product ID contains both a renderer-local ordinal and the public
    # ASIN selected by the Agent. Some providers correctly copy the ASIN while
    # combining it with a neighboring ordinal. Resolve only that representation
    # mismatch, and only when the ASIN identifies exactly one currently visible
    # open_product Action. This neither ranks candidates nor chooses an ASIN.
    requested_parts = decoded.split(":", 2)
    if len(requested_parts) == 3 and requested_parts[0] == "open_product":
        requested_asin = requested_parts[2].strip().casefold()
        asin_matches = [
            str(item.get("target_id", ""))
            for item in items
            if str(item.get("kind", "")) == "open_product"
            and str(item.get("asin", "")).strip().casefold() == requested_asin
            and str(item.get("target_id", "")) in visible
        ]
        if len(asin_matches) == 1:
            return asin_matches[0], "visible_open_product_asin_canonicalization"

    # A select_option ID embeds a renderer spelling of the public option value.
    # Models occasionally copy the displayed value (for example ``70\" x 90\"``)
    # instead of the renderer's punctuation-heavy ID suffix
    # (``70\"-x-90\"``).  Resolve only when the Agent-requested public value
    # identifies exactly one currently legal option.  This repairs transport
    # representation; it does not select an option or infer a desired value.
    if len(requested_parts) == 3 and requested_parts[0] == "select_option":
        requested_value = _canonical_public_option_value(requested_parts[2])
        option_matches = [
            str(item.get("target_id", ""))
            for item in items
            if str(item.get("kind", "")) == "select_option"
            and str(item.get("target_id", "")) in visible
            and requested_value
            in {
                _canonical_public_option_value(item.get("option_value", "")),
                _canonical_public_option_value(item.get("label", "")),
                _canonical_public_option_value(str(item.get("target_id", "")).split(":", 2)[-1]),
            }
        ]
        if len(option_matches) == 1:
            return option_matches[0], "visible_select_option_value_canonicalization"
    return requested, None


def _webshop_state_version_rejection(
    current: object,
    requested_version: object,
) -> tuple[str, str] | None:
    payload = current if isinstance(current, dict) else {}
    current_version = payload.get("state_version")
    # Older in-process fixtures and legacy sidecars have no version.  Keep them
    # compatible; official sidecar observations always expose the field.
    if not isinstance(current_version, int) or isinstance(current_version, bool):
        return None
    if not isinstance(requested_version, int) or isinstance(requested_version, bool):
        return (
            "webshop_state_version_required",
            f"state_version is required for the current WebShop state ({current_version}).",
        )
    if requested_version != current_version:
        return (
            "webshop_stale_state_version",
            (
                f"state_version {requested_version} is stale; the current WebShop state_version "
                f"is {current_version}. Choose an Action from the latest observation."
            ),
        )
    return None


def _webshop_current_action_surface(current: object) -> dict[str, Any]:
    payload = current if isinstance(current, dict) else {}
    actions = payload.get("valid_subactions", [])
    actions = actions if isinstance(actions, list) else []
    return {
        "budget_consumed": False,
        "retry_allowed": True,
        "current_state_version": payload.get("state_version"),
        "current_actions": [
            {
                key: item[key]
                for key in ("target_id", "kind", "label", "asin", "option_name", "option_value")
                if key in item
            }
            for item in actions
            if isinstance(item, dict)
        ],
    }


def _canonical_public_option_value(value: object) -> str:
    """Normalize only presentation separators used by public WebShop options."""

    normalized = html.unescape(str(value)).casefold().strip()
    normalized = normalized.translate(str.maketrans({"“": '"', "”": '"', "″": '"', "×": "x"}))
    normalized = re.sub(r"\s*[-–—]?\s*x\s*[-–—]?\s*", "x", normalized)
    return re.sub(r"\s+", " ", normalized)


def _validate_purchase_evidence(
    value: object,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(value, dict):
        return None, {
            "code": "purchase_evidence_required",
            "message": (
                "Buy Now requires public purchase_evidence. Report supported requirements and "
                "remaining uncertainty honestly; you decide whether to purchase. You may "
                "correct this declaration using already observed evidence without another Action."
            ),
        }
    verified = value.get("verified_requirements")
    unresolved = value.get("unresolved_constraints")
    if not isinstance(verified, list) or not verified:
        return None, {
            "code": "verified_requirements_required",
            "message": "Buy Now requires at least one verified public requirement.",
        }
    if not isinstance(unresolved, list):
        return None, {
            "code": "unresolved_constraints_required",
            "message": "Buy Now requires an explicit unresolved_constraints list.",
        }
    normalized_verified = [
        " ".join(str(item).split())[:240] for item in verified[:12] if " ".join(str(item).split())
    ]
    normalized_unresolved = [
        " ".join(str(item).split())[:240] for item in unresolved[:12] if " ".join(str(item).split())
    ]
    if not normalized_verified:
        return None, {
            "code": "verified_requirements_required",
            "message": "Buy Now requires at least one non-empty verified requirement.",
        }
    return {
        "verified_requirements": normalized_verified,
        "unresolved_constraints": normalized_unresolved,
    }, None


@dataclass(frozen=True)
class WebShopSearchTool:
    lifecycle: WebShopSessionLifecycle
    max_query_chars: int = 500
    name: str = "webshop_search"
    stateful: bool = True
    description: str = (
        "Search the WebShop catalog to obtain candidates only; search titles cannot verify or "
        "exclude selectable size/color variants. If the same candidates recur, another search "
        "adds no option evidence. Use webshop_click with an open_product target_id for a "
        "candidate you choose to inspect; no candidate is preselected or forced."
    )

    def set_deadline_context(self, deadline):
        setter = getattr(self.lifecycle.client, "set_deadline_context", None)
        if setter:
            setter(deadline)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("webshop_search requires a non-empty query")
        query = query.strip()
        if len(query) > self.max_query_chars:
            raise ValueError(f"webshop_search query exceeds {self.max_query_chars} characters")
        return json.dumps(self.lifecycle.search(query), ensure_ascii=False)


@dataclass(frozen=True)
class WebShopClickTool:
    lifecycle: WebShopSessionLifecycle
    name: str = "webshop_click"
    stateful: bool = True
    description: str = (
        "Execute one currently valid WebShop subaction. Copy target_id exactly from the latest "
        "valid_subactions list and copy state_version from that same latest observation; stale "
        "or invented targets are rejected before they consume Action budget. Buy Now additionally "
        "requires purchase_evidence recording supported requirements and remaining uncertainty "
        "honestly. Uncertainty does not block your purchase decision. An open_product action "
        "is how selectable variants are verified; a "
        "conflicting default variant in a search title is not exclusion evidence."
    )

    def set_deadline_context(self, deadline):
        setter = getattr(self.lifecycle.client, "set_deadline_context", None)
        if setter:
            setter(deadline)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "minLength": 1},
                "state_version": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Copy state_version from the latest WebShop observation. It binds "
                        "target_id to the current public action surface."
                    ),
                },
                "purchase_evidence": {
                    "type": "object",
                    "properties": {
                        "verified_requirements": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1, "maxLength": 240},
                            "minItems": 1,
                            "maxItems": 12,
                        },
                        "unresolved_constraints": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1, "maxLength": 240},
                            "maxItems": 12,
                        },
                    },
                    "required": ["verified_requirements", "unresolved_constraints"],
                    "additionalProperties": False,
                },
            },
            "required": ["target_id"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        target_id = arguments.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("webshop_click requires a non-empty target_id")
        return json.dumps(
            self.lifecycle.click(
                target_id.strip(),
                purchase_evidence=arguments.get("purchase_evidence"),
                state_version=arguments.get("state_version"),
            ),
            ensure_ascii=False,
        )

    def preflight(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Return a structured, non-mutating current-state rejection if any."""

        return self.lifecycle.preflight_click(
            arguments.get("target_id"),
            state_version=arguments.get("state_version"),
        )


class WebShopEnvironmentVerifier:
    name = "webshop_environment"
    supports_intermediate_scoring = False

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        del prediction
        result = task.metadata.get("webshop_environment_result")
        if not isinstance(result, dict):
            raise ValueError("WebShop verification requires a trusted environment result")
        score = float(result.get("reward", 0.0))
        score = min(1.0, max(0.0, score))
        # WebShop's standard harsh success metric is a completed purchase with
        # reward 1.0.  Exact target-ASIN matching is a stricter diagnostic that
        # the upstream benchmark records separately and must not replace the
        # standard success rate.
        passed = bool(result.get("purchased", False)) and score >= 1.0
        detail = {
            key: result.get(key)
            for key in (
                "purchased",
                "exact_success",
                "resolved_action",
                "steps",
                "reward_components",
                "termination_reason",
            )
            if key in result
        }
        return VerificationResult(
            score=score,
            passed=passed,
            verifier=self.name,
            detail=json.dumps(detail, ensure_ascii=False),
        )


def webshop_lifecycles(tools: dict[str, Any]) -> tuple[WebShopSessionLifecycle, ...]:
    values: dict[int, WebShopSessionLifecycle] = {}
    for tool in tools.values():
        lifecycle = getattr(tool, "lifecycle", None)
        if isinstance(lifecycle, WebShopSessionLifecycle):
            values[id(lifecycle)] = lifecycle
    return tuple(values.values())


def _trusted_goal_id(task: TaskSpec) -> str:
    for key in ("goal_id", "source_id", "source_task_id"):
        value = str(task.metadata.get(key, "")).strip()
        if re.fullmatch(r"(?:webshop/)?goal[-/:]\d+", value, re.IGNORECASE):
            return value
    return ""


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "purchased": False,
        "exact_success": False,
        "done": False,
        "termination_reason": reason,
    }
