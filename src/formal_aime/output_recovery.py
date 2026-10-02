"""Bound Qwen output recovery by real, question-scoped tool evidence."""
from __future__ import annotations

import hashlib
import json
import threading

from jsonschema import Draft202012Validator
from .artifact_protocol import _unique_object


RECOVERY_MAX_TOKENS = 4096
REPETITION_DETECTION = {
    "min_pattern_size": 24, "max_pattern_size": 512, "min_count": 4,
}
RECOVERY_INSTRUCTION = (
    "The previous generation was incomplete or repetitive; no rejected tool call executed. "
    "Use the actual observations already provided. Emit one concise complete tool call in the configured protocol "
    "or the required final JSON. Put only executable computation in code. "
    "Do not write Python comments or explanatory docstrings, or hide discussion in strings or unused variables. "
    "Put any necessary brief explanation before the tool call, outside its arguments. "
    "Do not repeat analysis or claim a tool result before receiving its observation."
)


class OutputRecoveryState:
    """Agent recreation cannot renew recovery without a new successful observation."""

    def __init__(self, store=None):
        self._lock = threading.Lock()
        self._store = store
        saved = store.load("output_recovery", {}) if store else {}
        self._evidence_version = saved.get("evidence_version", 0)
        self._failed_versions = set(saved.get("failed_versions", []))
        self._claimed_versions = set(saved.get("claimed_versions", []))
        self._work_claims = set(tuple(item) for item in saved.get("work_claims", []))
        self._observed = set(saved.get("observed", []))

    def _persist(self):
        if self._store:
            self._store.save("output_recovery", {"evidence_version": self._evidence_version,
                "failed_versions": sorted(self._failed_versions),
                "claimed_versions": sorted(self._claimed_versions), "observed": sorted(self._observed),
                "work_claims": sorted(self._work_claims)})

    def snapshot(self):
        from .protocol_recovery import WORK
        with self._lock:
            version = self._evidence_version
            return {
                "evidence_version": version,
                "restricted": version in self._failed_versions,
                "recovery_used": ((WORK.get(), version) in self._work_claims
                                  if WORK.get() else version in self._claimed_versions),
                "recovery_max_tokens": RECOVERY_MAX_TOKENS,
            }

    def mark_failed(self, version):
        with self._lock:
            self._failed_versions.add(version)
            self._persist()

    def claim(self, version):
        from .protocol_recovery import WORK
        with self._lock:
            key = WORK.get()
            if key:
                if (key, version) in self._work_claims:
                    return False
                if self._store and not self._store.claim(key, version, "tool_generation"):
                    return False
                self._work_claims.add((key, version))
                self._persist()
                return True
            if version in self._claimed_versions:
                return False
            self._claimed_versions.add(version)
            self._persist()
            return True

    def observe(self, name, observation):
        # Called only by the runtime after executing a real mathematical Action.
        if name not in {"python_exec", "symbolic_compute", "finite_search"}:
            return
        output = observation.get("output")
        if (observation.get("status") != "ok" or not output
                or isinstance(output, dict) and output.get("status", "ok") != "ok"):
            return
        if isinstance(output, dict):
            output = {k: v for k, v in output.items()
                      if k not in {"duration_s", "elapsed_s", "elapsed_ms", "duration_ms",
                                   "execution_time_s", "call_id", "purpose"}}
        signature = hashlib.sha256(json.dumps(
            {"name": name, "output": output}, sort_keys=True, default=str,
        ).encode()).hexdigest()
        with self._lock:
            if signature not in self._observed:
                self._observed.add(signature)
                self._evidence_version += 1
                self._persist()


def current_output_recovery():
    from .worker_usage_ledger import active_worker_usage
    active = active_worker_usage()
    if active is not None:
        return active[0].output_recovery
    # Executors may supply an execution-local guard. Formal evaluations use
    # the shared ledger above, across all graph nodes.
    from .generation_audit import execution_scope
    scope = execution_scope()
    return getattr(scope, "output_recovery", None) or OutputRecoveryState()


def generation_stop(completed, request):
    choice = completed.choices[0]
    if (choice.finish_reason == "repetition"
            or getattr(choice, "stop_reason", None) == "repetition_detected"):
        return "repetition"
    usage = getattr(completed, "usage", None)
    cap = request.get("max_tokens", request.get("max_completion_tokens", 0))
    if (choice.finish_reason == "length"
            or cap and int(getattr(usage, "completion_tokens", 0) or 0) >= cap):
        return "output_cap"
    return None


def native_rejection(completed, request, actions):
    """Keep provider finish reasons intact; validate the whole native call."""
    calls = getattr(completed.choices[0].message, "tool_calls", None) or []
    if not calls:
        return None
    stop = generation_stop(completed, request)
    if stop:
        return {"reason": "native_tool_" + stop, "path": []}
    specs = {action.name: action for action in actions}
    for call in calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", "")
        if name not in specs:
            return {"reason": "native_tool_unknown_action", "path": []}
        arguments = getattr(function, "arguments", None)
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_object) if isinstance(arguments, str) else arguments
        except (TypeError, ValueError):
            return {"reason": "native_tool_invalid_json", "path": []}
        error = next(Draft202012Validator(specs[name].parameters).iter_errors(arguments), None)
        if error is not None:
            return {"reason": "native_tool_invalid_arguments", "path": list(error.path),
                    "message": error.message[:500]}
    return None
