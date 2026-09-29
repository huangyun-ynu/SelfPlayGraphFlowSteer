"""Durable, question-scoped accounting for reported Worker model usage."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


POLICY = "reported_usage_threshold_v1"


class UsageDispatchStopped(RuntimeError):
    def __init__(self, reason: str, status: dict[str, Any]):
        self.reason = reason
        self.status = status
        super().__init__(reason)


_ACTIVE_LEDGER: ContextVar[tuple[WorkerUsageLedger, str, str] | None] = ContextVar(
    "worker_usage_ledger", default=None
)


@contextmanager
def worker_usage_scope(ledger: WorkerUsageLedger | None, *, agent_id: str, execution_id: str):
    token = _ACTIVE_LEDGER.set((ledger, agent_id, execution_id) if ledger else None)
    try:
        yield
    finally:
        _ACTIVE_LEDGER.reset(token)


def active_worker_usage() -> tuple[WorkerUsageLedger, str, str] | None:
    return _ACTIVE_LEDGER.get()


class WorkerUsageLedger:
    """Each physical dispatch is persisted before it reaches the provider."""

    def __init__(self, path: str | Path, *, question_attempt_id: str,
                 threshold: int, max_unsettled_attempts: int = 2) -> None:
        if not question_attempt_id or threshold <= 0 or max_unsettled_attempts <= 0:
            raise ValueError("invalid Worker usage account configuration")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = int(threshold)
        self.max_unsettled_attempts = int(max_unsettled_attempts)
        self.question_attempt_id = question_attempt_id
        self._lock = threading.RLock()
        self._lock_fd: int | None = None
        self._db: sqlite3.Connection | None = None
        try:
            self._lock_fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Worker usage account is already active") from exc
            self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self._initialize_account()
        except BaseException:
            self.close()
            raise

    def _initialize_account(self) -> None:
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL, threshold INTEGER NOT NULL, max_unsettled INTEGER NOT NULL, policy TEXT NOT NULL)")
        self._db.execute("CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT PRIMARY KEY, route TEXT NOT NULL, model TEXT NOT NULL, api_surface TEXT NOT NULL, agent_id TEXT NOT NULL, execution_id TEXT NOT NULL, request_fingerprint TEXT NOT NULL, state TEXT NOT NULL, confirmed_before INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER, provider_response_id TEXT, started_at REAL NOT NULL, settled_at REAL)")
        attempt_columns = {row[1] for row in self._db.execute("PRAGMA table_info(attempts)")}
        for name in ("model", "api_surface"):
            if name not in attempt_columns:
                self._db.execute(f"ALTER TABLE attempts ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        self._db.execute("CREATE TABLE IF NOT EXISTS reconciliation (attempt_id TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER, reconciled_at REAL NOT NULL)")
        with self._transaction():
            row = self._db.execute("SELECT identity, threshold, max_unsettled, policy FROM account WHERE id=1").fetchone()
            expected = (self.question_attempt_id, self.threshold, self.max_unsettled_attempts, POLICY)
            if row is None:
                self._db.execute("INSERT INTO account VALUES (1, ?, ?, ?, ?)", expected)
            elif tuple(row) != expected:
                raise ValueError("Worker usage ledger belongs to another question or policy")
            # A crash after dispatch intent cannot be treated as a free request.
            self._db.execute("UPDATE attempts SET state='unknown', settled_at=? WHERE state='pending'", (time.time(),))

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    def _status_unlocked(self) -> dict[str, Any]:
        known_in, known_out, unknown, pending, attempts = self._db.execute(
            "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            "SUM(CASE WHEN state='unknown' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN state='pending' THEN 1 ELSE 0 END), COUNT(*) FROM attempts"
        ).fetchone()
        used = int(known_in) + int(known_out)
        return {
            "policy": POLICY, "threshold": self.threshold,
            "confirmed_input_tokens": int(known_in), "confirmed_output_tokens": int(known_out),
            "confirmed_used": used, "confirmed_overshoot": max(0, used - self.threshold),
            "threshold_reached": used >= self.threshold, "attempt_count": int(attempts),
            "unsettled_attempt_count": int(unknown or 0), "inflight_request_count": int(pending or 0),
            "usage_complete": not (unknown or pending),
            "can_dispatch": used < self.threshold and int(unknown or 0) < self.max_unsettled_attempts and not pending,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def stop_reason(self) -> str | None:
        state = self.status()
        if state["confirmed_used"] >= self.threshold:
            return "worker_usage_threshold_reached"
        if state["unsettled_attempt_count"] >= self.max_unsettled_attempts:
            return "worker_usage_unsettled_limit"
        if state["inflight_request_count"]:
            return "worker_usage_request_inflight"
        return None

    def begin(self, *, route: str, agent_id: str, execution_id: str,
              model: str = "", api_surface: str = "",
              request: dict[str, Any]) -> str:
        fingerprint = hashlib.sha256(json.dumps(
            request, ensure_ascii=False, sort_keys=True, default=str,
        ).encode()).hexdigest()
        with self._transaction():
            state = self._status_unlocked()
            if not state["can_dispatch"]:
                reason = self.stop_reason()
                raise UsageDispatchStopped(reason or "worker_usage_request_inflight", state)
            attempt_id = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO attempts (attempt_id, route, model, api_surface, agent_id, execution_id, request_fingerprint, state, confirmed_before, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (attempt_id, route, model, api_surface, agent_id, execution_id, fingerprint,
                 state["confirmed_used"], time.time()),
            )
            return attempt_id

    @staticmethod
    def _usage_value(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    def settle(self, attempt_id: str, *, input_tokens: object, output_tokens: object,
               provider_response_id: str | None = None) -> dict[str, Any]:
        token_in = self._usage_value(input_tokens)
        token_out = self._usage_value(output_tokens)
        state = "complete" if token_in is not None and token_out is not None else "unknown"
        with self._transaction():
            row = self._db.execute(
                "SELECT state, input_tokens, output_tokens FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown Worker usage attempt")
            if row[0] != "pending":
                if row[1:] != (token_in, token_out):
                    raise ValueError("conflicting Worker usage settlement")
                return self._status_unlocked()
            self._db.execute(
                "UPDATE attempts SET state=?, input_tokens=?, output_tokens=?, provider_response_id=?, settled_at=? WHERE attempt_id=?",
                (state, token_in, token_out, provider_response_id, time.time(), attempt_id),
            )
            return self._status_unlocked()

    def mark_unknown(self, attempt_id: str) -> dict[str, Any]:
        with self._transaction():
            self._db.execute(
                "UPDATE attempts SET state='unknown', settled_at=? WHERE attempt_id=? AND state='pending'",
                (time.time(), attempt_id),
            )
            return self._status_unlocked()

    def reconcile(self, attempt_id: str, *, input_tokens: object,
                  output_tokens: object) -> dict[str, Any]:
        token_in = self._usage_value(input_tokens)
        token_out = self._usage_value(output_tokens)
        with self._transaction():
            row = self._db.execute(
                "SELECT state, input_tokens, output_tokens FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None or row[0] == "pending":
                raise ValueError("Worker usage attempt is not available for reconciliation")
            if ((row[1] is not None and row[1] != token_in)
                    or (row[2] is not None and row[2] != token_out)):
                raise ValueError("conflicting Worker usage reconciliation")
            if row[1:] == (token_in, token_out):
                return self._status_unlocked()
            if token_in is None or token_out is None:
                raise ValueError("reconciliation requires complete provider usage")
            self._db.execute(
                "UPDATE attempts SET state='complete', input_tokens=?, output_tokens=?, settled_at=? WHERE attempt_id=?",
                (token_in, token_out, time.time(), attempt_id),
            )
            self._db.execute(
                "INSERT INTO reconciliation VALUES (?, ?, ?, ?)",
                (attempt_id, token_in, token_out, time.time()),
            )
            return self._status_unlocked()

    def digest(self) -> str:
        with self._lock:
            rows = self._db.execute(
                "SELECT attempt_id, route, model, api_surface, state, confirmed_before, input_tokens, output_tokens FROM attempts ORDER BY started_at, attempt_id"
            ).fetchall()
            return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    def dispatches_valid(self) -> bool:
        with self._lock:
            rows = self._db.execute("SELECT confirmed_before FROM attempts").fetchall()
            return all(before < self.threshold for (before,) in rows)

    def close(self) -> None:
        """Release the connection and flock once, including partial initialization."""
        with self._lock:
            connection, self._db = self._db, None
            lock_fd, self._lock_fd = self._lock_fd, None
            try:
                if connection is not None:
                    connection.close()
            finally:
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lock_fd)
