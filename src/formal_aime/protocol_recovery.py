"""Question-owned protocol state; internal bookkeeping, not model tools."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import time

from .result_candidate import digest

WORK = ContextVar("protocol_work", default=None)
RECOVERY = ContextVar("protocol_recovery_request", default=None)
VERSION = "protocol_recovery_v1"


class ProtocolNoProgress(RuntimeError):
    def __init__(self, reason, details=None):
        self.reason, self.details = reason, details or {}
        super().__init__(reason)


class ProtocolStore:
    def __init__(self, ledger):
        self.ledger = ledger
        with ledger._transaction():
            db = ledger._db
            db.execute("CREATE TABLE IF NOT EXISTS protocol_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS protocol_failures (request_key TEXT, evidence_version INTEGER, wire_request_sha256 TEXT, attempt_id TEXT, failure_kind TEXT, result_sha256 TEXT, work_key TEXT, PRIMARY KEY(request_key,evidence_version))")
            db.execute("CREATE TABLE IF NOT EXISTS recovery_claims (work_key TEXT, evidence_version INTEGER, recovery_kind TEXT, candidate_id TEXT, claimed_at REAL, attempt_id TEXT, state TEXT, PRIMARY KEY(work_key,evidence_version,recovery_kind,candidate_id))")
            db.execute("CREATE TABLE IF NOT EXISTS candidate_refs (candidate_id TEXT PRIMARY KEY, work_key TEXT, source_sha256 TEXT, record_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS protocol_events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, data TEXT NOT NULL)")
            stored = db.execute("SELECT value FROM protocol_state WHERE key='version'").fetchone()
            if stored and json.loads(stored[0]) != VERSION:
                raise ValueError("incompatible protocol recovery ledger")
            self._set("version", VERSION)

    def _get(self, key, default=None):
        row = self.ledger._db.execute("SELECT value FROM protocol_state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set(self, key, value):
        self.ledger._db.execute("INSERT OR REPLACE INTO protocol_state VALUES (?,?)",
                                (key, json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def load(self, key, default=None):
        with self.ledger._lock:
            return self._get(key, default)

    def save(self, key, value):
        with self.ledger._transaction():
            self._set(key, value)

    def event(self, name, data):
        with self.ledger._transaction():
            self.ledger._db.execute("INSERT INTO protocol_events(event,data) VALUES (?,?)",
                                   (name, json.dumps(data, ensure_ascii=False, default=str)))

    def work_status(self, key, version):
        return self.load(f"work:{key}:{version}", {})

    def exhaust(self, key, version, reason):
        with self.ledger._transaction():
            state = self._get(f"work:{key}:{version}", {})
            state.update(exhausted=True, reason=reason)
            self._set(f"work:{key}:{version}", state)

    def deny(self, key, version, reason):
        with self.ledger._transaction():
            previous = self._get("last_denial", {})
            same = previous.get("work_key") == key and previous.get("evidence_version") == version
            details = {"work_key": key, "evidence_version": version, "reason": reason,
                       "repeat_count": previous.get("repeat_count", 0) + 1 if same else 1,
                       "no_request_dispatched": True}
            self._set("last_denial", details)
        self.event("dispatch_denied", details)
        return details

    def progress(self):
        self.save("last_denial", {})

    def status(self):
        data = self.load("last_denial", {})
        current_version = self.ledger.output_recovery.snapshot()["evidence_version"]
        return {**data, "terminal": data.get("repeat_count", 0) >= 2
                and data.get("evidence_version") == current_version, "version": VERSION,
                **({"guidance": "The repeated failed request was not sent. Use a current valid candidate or change the actual responsibility, input or evidence. Renaming nodes does not renew recovery."}
                   if data else {})}

    def bind_artifact(self, artifact):
        ids = list(dict.fromkeys(d["candidate_id"] for d in artifact.protocol_diagnostics
                                 if d.get("candidate_id") and d.get("accepted") is True))
        if ids:
            self.event("candidate_artifact_bound", {"candidate_ids": ids,
                "artifact_id": artifact.artifact_id, "agent_id": artifact.agent_id,
                "input_signature": getattr(artifact, "input_signature", ""),
                "answer_sha256": digest(artifact.answer), "issues": artifact.unresolved_issues})

    def submit(self, receipt):
        self.event("candidate_submitted", {"artifact_id": receipt.artifact_id,
            "agent_id": receipt.output_agent_id, "receipt_ref": receipt.receipt_ref,
            "answer_sha256": digest(receipt.raw_answer_snapshot),
            "director_call_id": receipt.director_call_id})

    def claim(self, key, version, kind, candidate="", *, attempt_id=None):
        with self.ledger._transaction():
            cursor = self.ledger._db.execute(
                "INSERT OR IGNORE INTO recovery_claims VALUES (?,?,?,?,?,?,?)",
                (key, version, kind, candidate, time.time(), attempt_id,
                 "dispatched" if attempt_id else "reserved"))
            return cursor.rowcount == 1

    def bind_claim(self, claim, attempt_id):
        if not claim:
            return
        with self.ledger._transaction():
            self.ledger._db.execute(
                "UPDATE recovery_claims SET attempt_id=?,state='dispatched' WHERE work_key=? AND evidence_version=? AND recovery_kind=? AND candidate_id=?",
                (attempt_id, *claim))

    def candidate(self, candidate, *, observations=(), status=None, parent=None, assembly=None):
        key = WORK.get() or "unscoped"
        record = candidate.record()
        if assembly is not None:
            record["assembly"] = assembly
        with self.ledger._transaction():
            db = self.ledger._db
            previous = db.execute("SELECT candidate_id,record_json FROM candidate_refs WHERE work_key=? ORDER BY rowid DESC LIMIT 1", (key,)).fetchone()
            if previous and previous[0] != candidate.candidate_id:
                old = json.loads(previous[1])
                parent = parent or previous[0]
                record["revision_kind"] = "solve_revision" if old["answer"] != candidate.answer else "same_answer_observation"
                record["revision_source"] = candidate.source
                # Prior dissent remains a historical claim, never a verified fact.
                record["prior_issue_ids"] = old.get("issue_ids")
            record.update(question_attempt_id=self.ledger.question_attempt_id,
                          work_key=key, parent_candidate_id=parent,
                          evidence_refs=[{"observation_sha256": digest(item), "observation": item}
                                         for item in observations], state=status or record["parse_status"])
            db.execute("INSERT OR IGNORE INTO candidate_refs VALUES (?,?,?,?)",
                       (candidate.candidate_id, key, digest(candidate.raw), json.dumps(record, ensure_ascii=False)))
        return record

    @staticmethod
    def request_key(request):
        # Ignore only transport/output controls. Do not rewrite mathematical
        # prose, code, tool results or model configuration.
        filtered = {k: v for k, v in request.items() if k not in
                    {"max_tokens", "max_completion_tokens", "timeout"}}
        return digest(filtered)

    def before_dispatch(self, request):
        version = self.ledger.output_recovery.snapshot()["evidence_version"]
        key = WORK.get() or self.request_key(request)
        with self.ledger._lock:
            failed = self.ledger._db.execute(
                "SELECT failure_kind FROM protocol_failures WHERE request_key=? AND evidence_version=?",
                (self.request_key(request), version)).fetchone()
        if failed:
            raise ProtocolNoProgress("duplicate_failed_request", self.deny(key, version, failed[0]))
        intent = RECOVERY.get()
        if intent and not intent.get("claimed"):
            claim = (("__question_solve_revision__", 0, intent["kind"], "")
                     if intent["kind"] == "solve_revision" else
                     (key, version, intent["kind"], intent.get("candidate_id", "")))
            if not self.claim(*claim):
                raise ProtocolNoProgress("protocol_recovery_exhausted", self.deny(key, version, intent["kind"]))
            intent.update(claimed=True, claim=claim)
        if intent:
            return intent.get("claim")
        with self.ledger._lock:
            native = self.ledger._db.execute("SELECT state FROM recovery_claims WHERE work_key=? AND evidence_version=? AND recovery_kind='tool_generation' AND candidate_id=''", (key, version)).fetchone()
        return (key, version, "tool_generation", "") if native and native[0] == "reserved" else None

    def completion(self, request, response, attempt_id):
        from .output_recovery import generation_stop
        from .result_candidate import capture_candidate
        from jsonschema import Draft202012Validator
        reason = generation_stop(response, request)
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        if not reason and calls:
            specs = {s["function"]["name"]: s["function"]["parameters"] for s in request.get("tools", [])}
            from .artifact_protocol import _unique_object
            for call in calls:
                try:
                    args = json.loads(call.function.arguments, object_pairs_hook=_unique_object)
                    if call.function.name not in specs or not Draft202012Validator(specs[call.function.name]).is_valid(args):
                        reason = "native_tool_invalid_arguments"
                except (ValueError, TypeError):
                    reason = "native_tool_invalid_json"
        content = getattr(message, "content", None) or ""
        text_action = False
        if (not reason and not calls and not RECOVERY.get()
                and request.get("extra_body", {}).get("structured_outputs")):
            # MATH-Hard's existing protocol puts Actions in content. Use the
            # runtime's parser; a recognized Action is not a failed final report.
            # Validation/execution and tool errors still belong to the runtime.
            from .runtime import _text_action_calls
            text_action = bool(_text_action_calls(content))
        from .math_completion import parse_decision
        compact = request.get("response_format", {}).get("json_schema", {}).get("name") == "math_completion"
        if not reason and not calls and not text_action:
            if compact:
                if parse_decision(content) is None:
                    from .artifact_protocol import check_artifact
                    legacy = capture_candidate(content)
                    fields = {"answer", "summary", "evidence", "unresolved_issues", "tool_summary", "confidence"}
                    if (legacy is None or not fields.issubset(legacy.payload)
                            or check_artifact(legacy.payload_json)[0] is None):
                        reason = "invalid_completion_decision"
            elif capture_candidate(content) is None:
                reason = "invalid_final_data"
        if not reason:
            return
        version = self.ledger.output_recovery.snapshot()["evidence_version"]
        with self.ledger._transaction():
            self.ledger._db.execute("INSERT OR IGNORE INTO protocol_failures VALUES (?,?,?,?,?,?,?)",
                (self.request_key(request), version, digest(request), attempt_id, reason,
                 digest(message.model_dump() if hasattr(message, "model_dump") else content), WORK.get()))


@contextmanager
def recovery_request(kind, candidate_id=""):
    intent = {"kind": kind, "candidate_id": candidate_id, "claimed": False}
    token = RECOVERY.set(intent)
    try:
        yield intent
    finally:
        RECOVERY.reset(token)


def current_store():
    from .worker_usage_ledger import active_worker_usage
    active = active_worker_usage()
    return active[0].protocol if active else None


def work_key(executor, kwargs):
    node = kwargs["node"]
    config = getattr(executor.backend, "config", None)
    role = getattr(config, "roles", {}).get("worker")
    def packet(p):
        if getattr(p, "answer", None) in {"WORKER_PROTOCOL_FAILURE", "WORKER_BACKEND_FAILURE"}:
            return None
        return {field: getattr(p, field, None) for field in
                ("answer", "summary", "evidence", "unresolved_issues", "confidence", "tool_summary")}
    return digest({"version": VERSION, "public_task": kwargs["task"], "responsibility": node.prompt,
                   "tools": list(node.allowed_tools), "tool_limits": [node.initial_tool_budget, node.revision_tool_budget, node.total_tool_budget],
                   "result_scope": node.metadata.get("result_scope"),
                   "upstream": [value for p in kwargs.get("upstream", []) if (value := packet(p)) is not None],
                   "peers": [value for p in kwargs.get("peers", []) if (value := packet(p)) is not None],
                   "prior": packet(kwargs["prior"]) if kwargs.get("prior") else None,
                   "model": getattr(role, "model", None), "route": getattr(config, "route_name", None),
                   "temperature": getattr(role, "temperature", None), "top_p": getattr(role, "top_p", None),
                   "top_k": getattr(role, "top_k", None), "thinking": getattr(role, "enable_thinking", None),
                   "seed": getattr(config, "sampling_seed", None), "request_profile": getattr(config, "request_profile", None)})


def protocol_execution(function):
    @wraps(function)
    def execute(self, **kwargs):
        node = kwargs["node"]
        if node.metadata.get("action_adapter") not in {"aime", "math_hard"}:
            return function(self, **kwargs)
        from .output_recovery import current_output_recovery
        from .contracts import AgentArtifact
        key = work_key(self, kwargs)
        token = WORK.set(key)
        store = current_store()
        state = current_output_recovery()
        version = state.snapshot()["evidence_version"]
        try:
            if store and store.work_status(key, version).get("exhausted"):
                raise ProtocolNoProgress("protocol_recovery_exhausted", store.deny(key, version, "same_failed_work"))
            artifact = function(self, **kwargs)
            if store:
                if artifact.answer == "WORKER_PROTOCOL_FAILURE":
                    store.exhaust(key, state.snapshot()["evidence_version"], "final_report_failed")
                elif artifact.answer != "WORKER_BACKEND_FAILURE":
                    store.progress()
            return artifact
        except ProtocolNoProgress as exc:
            used = getattr(exc, "worker_token_usage", (0, 0))
            return AgentArtifact(artifact_id="pending", agent_id=node.agent_id,
                answer="WORKER_PROTOCOL_FAILURE", summary=exc.reason, confidence=0,
                token_in=used[0], token_out=used[1], unresolved_issues=[exc.reason],
                react_trace=getattr(exc, "worker_react_trace", []),
                backend_request_events=getattr(exc, "request_events", []),
                protocol_diagnostics=[*getattr(exc, "worker_protocol_diagnostics", []),
                    {"stage": "question_protocol_guard", "accepted": False, "rejection_reason": exc.reason,
                     "local_recovery_exhausted": True, "no_request_dispatched": True, **exc.details}])
        finally:
            WORK.reset(token)
    return execute
