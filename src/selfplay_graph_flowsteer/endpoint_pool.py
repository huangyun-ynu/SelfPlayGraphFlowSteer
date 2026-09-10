"""Same-model endpoint sharding and bounded, request-local failover."""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .backend_failures import classify_backend_failure
from .llm import _logical_request_budget_s, endpoint_failover_scope


class EndpointPoolBackend:
    def __init__(self, name: str, members: dict[str, Any], state_dir: Path):
        if not members:
            raise ValueError("endpoint pool must not be empty")
        self.name = name
        self.members = tuple(members.items())
        self.config = self.members[0][1].config
        signature = repr([(key, backend.config.base_url) for key, backend in self.members])
        self.counter_path = state_dir / (
            hashlib.sha256(signature.encode()).hexdigest() + ".counter"
        )
        self._deadline = ContextVar(f"endpoint_pool_deadline_{id(self)}", default=None)

    def _audit(self, event):
        # Persist failures even when no completed Worker result reaches the runner.
        with self.counter_path.with_suffix(".events.jsonl").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def generate(self, messages, **kwargs):
        self.counter_path.parent.mkdir(parents=True, exist_ok=True)
        with self.counter_path.open("a+") as counter:
            fcntl.flock(counter, fcntl.LOCK_EX)
            counter.seek(0)
            value = int(counter.read() or "0")
            counter.seek(0)
            counter.truncate()
            counter.write(str(value + 1))
            counter.flush()
        ordered = (
            self.members[value % len(self.members) :] + self.members[: value % len(self.members)]
        )
        # DeepSeek thinking history is endpoint-specific. Balance new sessions,
        # but keep an existing conversation on its previous member first.
        # Bounded failover remains available if that member actually fails.
        if self.name == "deepseek":
            previous = next(
                (
                    m.get("_endpoint_pool_member")
                    for m in reversed(messages)
                    if m.get("_endpoint_pool_member")
                ),
                None,
            )
            if previous is not None:
                ordered = tuple(item for item in ordered if item[0] == previous) + tuple(
                    item for item in ordered if item[0] != previous
                )
        deadline = self._deadline.get()
        started = time.monotonic()
        first_config = ordered[0][1].config
        request_deadline = deadline
        if deadline is not None:
            deadline.check("endpoint_pool_start")
            scope_request = getattr(deadline, "for_request", None)
            if callable(scope_request):
                request_deadline = scope_request(
                    kwargs.get("role", "worker"), first_config.timeout_s
                )
        budget = _logical_request_budget_s(first_config, request_deadline)
        end = started + budget
        request_id = uuid.uuid4().hex
        attempts = []
        last_error = None
        for key, backend in ordered:
            # Failed waits do not consume the next member's request budget.
            # Each member is still tried once and the rollout has an absolute bound.
            if getattr(deadline, "exclude_failed_request_time", False):
                end = time.monotonic() + _logical_request_budget_s(backend.config, request_deadline)
            if deadline is not None:
                deadline.check("endpoint_pool_failover")
            if time.monotonic() >= end and last_error is not None:
                raise last_error
            # Only completed canonical messages cross endpoints. Tools execute
            # outside generate(), so failover cannot replay an environment action.
            replay = [
                {
                    k: v
                    for k, v in message.items()
                    if k not in {"_provider_payloads", "_endpoint_pool_member"}
                }
                if message.get("_endpoint_pool_member") != key
                else {k: v for k, v in message.items() if k != "_endpoint_pool_member"}
                for message in messages
            ]
            attempt_started = time.monotonic()
            event = dict(
                request_id=request_id,
                logical_route=self.name,
                endpoint_pool_member=key,
                attempt=len(attempts) + 1,
                request_role=kwargs.get("role", ""),
                timestamp=time.time(),
                request_remaining_s=max(0.0, end - attempt_started),
            )
            try:
                with endpoint_failover_scope(end):
                    response = backend.generate(replay, **kwargs)
            except Exception as exc:
                failure = classify_backend_failure(exc, stage="endpoint_pool", route=key)
                if failure.backend_failure and hasattr(deadline, "record_failed_request"):
                    deadline.record_failed_request(attempt_started, time.monotonic())
                event.update(
                    status="failed",
                    elapsed_s=time.monotonic() - attempt_started,
                    failure=failure.to_dict(),
                )
                attempts.append(event)
                self._audit(event)
                exc.endpoint_pool_attempts = list(attempts)
                last_error = exc
                if not failure.backend_failure or not failure.retryable:
                    raise
                continue
            event.update(status="success", elapsed_s=time.monotonic() - attempt_started)
            attempts.append(event)
            self._audit(event)
            if getattr(response, "assistant_message", None) is not None:
                response.assistant_message["_endpoint_pool_member"] = key
            response.metadata.update(
                logical_route=self.name,
                endpoint_pool_member=key,
                endpoint_pool_attempts=attempts,
                endpoint_pool_failovers=len(attempts) - 1,
            )
            return response
        assert last_error is not None
        raise last_error

    def set_deadline_context(self, deadline):
        self._deadline.set(deadline)
        for _, backend in self.members:
            backend.set_deadline_context(deadline)
