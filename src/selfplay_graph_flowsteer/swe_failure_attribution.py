"""Runtime evidence that prevents attributing an unfinished SWE run to policy alone.

Only runtime-owned fields are projected. Worker prose, thinking and self-reported
causes are never evidence. These records do not restore or submit any candidate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SOURCE = "swe_runtime_failure_evidence_v1"
CREDIT_STAGES = frozenset({
    "swe_request_token_credit_exhausted", "swe_finalization_token_credit_exhausted",
})


def project_swe_step(step: Any) -> dict[str, Any]:
    """Keep small, trusted evidence instead of copying full Worker trajectories."""
    def artifact(value: Any) -> dict[str, Any]:
        return {
            "artifact_id": value.artifact_id,
            "agent_id": value.agent_id,
            "failed": value.answer in {"WORKER_PROTOCOL_FAILURE", "WORKER_BACKEND_FAILURE"}
                      or "terminal_protocol_failure" in value.integrity_risks,
            "token_in": value.token_in, "token_out": value.token_out,
            "code_artifact_ref": value.code_artifact_ref.to_dict() if value.code_artifact_ref else None,
            "swe_progress": {key: value.swe_progress.get(key) for key in
                             ("trusted", "test_after_latest_edit", "result_scope")},
            "protocol_diagnostics": [
                {key: d[key] for key in ("stage", "execution_credit", "spent_tokens",
                                         "request_token_budget") if key in d}
                for d in value.protocol_diagnostics if d.get("stage") in CREDIT_STAGES
            ],
            "changed_workspace": any(
                (turn.get("action") or {}).get("name") in {"swe_edit", "swe_apply_artifact"}
                and (turn.get("observation") or {}).get("status") == "ok"
                and ((turn.get("observation") or {}).get("output") or {}).get("status") == "ok"
                for turn in value.react_trace
            ),
        }

    report = step.execution
    return {
        "event_id": step.event_id,
        "round_index": step.round_index,
        "node_ids": [node["agent_id"] for node in step.graph.get("nodes", ())],
        "token_budget": dict(step.control_snapshot.get("token_budget", {})),
        "recoverable_patch_hashes": [
            item["code_artifact_ref"]["artifact_sha256"]
            for item in step.control_snapshot.get("recoverable_code_artifacts", ())
        ],
        "execution": None if report is None else {
            "artifacts": {key: artifact(value) for key, value in report.artifacts.items()},
            "attempt_artifacts": [artifact(value) for value in report.attempt_artifacts],
            "execution_events": [
                {key: event[key] for key in
                 ("artifact_id", "agent_id", "cache_hit", "token_in", "token_out") if key in event}
                for event in report.execution_events
            ],
            "token_in": report.token_in, "token_out": report.token_out,
        },
    }


def swe_failure_attribution(
    history: Iterable[Mapping[str, Any]], *, worker_token_limit: int,
) -> dict[str, Any]:
    """Diagnose allocation refusals and failed replacement of a visible patch.

    Input is the runtime projection above (also usable by offline replay). Real
    shared-balance exhaustion alone is not an engineering attribution blocker.
    A restored patch resolves its replacement incident. A valid submission is
    still scored normally by the caller, regardless of historical incidents.
    """
    incidents: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] = {}
    seen_executions: set[str] = set()
    total_spent = 0
    for step in history:
        node_ids = set(step["node_ids"])
        previous = {key: value for key, value in previous.items() if key in node_ids}
        report = step.get("execution")
        if not report:
            continue
        current = report.get("artifacts", {})
        artifacts = {a["artifact_id"]: a for a in current.values()}
        artifacts.update({a["artifact_id"]: a for a in report.get("attempt_artifacts", ())})
        candidates = dict(previous)
        budget = step.get("token_budget", {})
        # The snapshot is taken after this report is charged to the question.
        # It includes costs of deleted nodes and preceding execution waves.
        before = (int(budget["used"]) - int(report.get("token_in", 0))
                  - int(report.get("token_out", 0))) if "used" in budget else total_spent
        for event in report.get("execution_events", ()):
            artifact_id = event.get("artifact_id")
            if event.get("cache_hit") or artifact_id in seen_executions:
                continue
            seen_executions.add(artifact_id)
            artifact = artifacts.get(artifact_id, {})
            # Initial generation and peer revision can occur in one Canvas
            # step. Keep its last candidate until a healthy replacement or an
            # intentional workspace change, including across failed attempts.
            if artifact and (artifact.get("code_artifact_ref")
                             or not artifact.get("failed") or artifact.get("changed_workspace")):
                candidates[artifact["agent_id"]] = artifact
            for diagnostic in artifact.get("protocol_diagnostics", ()):
                if diagnostic.get("stage") not in CREDIT_STAGES:
                    continue
                quote = diagnostic.get("request_token_budget", {})
                # Finalization diagnostics in old traces omit per-execution
                # spend. Total artifact usage is a conservative fallback.
                spent = int(diagnostic.get("spent_tokens", int(artifact.get("token_in", 0))
                                           + int(artifact.get("token_out", 0))))
                remaining = max(0, worker_token_limit - before - spent)
                details = {
                    "event_id": step.get("event_id", ""), "artifact_id": artifact_id,
                    "agent_id": event.get("agent_id"), "stage": diagnostic["stage"],
                    "question_remaining": remaining, "resolved": False,
                }
                if not all(key in quote for key in ("input_bound", "output_bound", "credit")):
                    incidents.append({**details, "code": "swe_request_budget_evidence_incomplete"})
                    continue
                minimum = int(quote["input_bound"]) + min(128, int(quote["output_bound"]))
                request_remaining = max(0, int(quote["credit"]) - int(quote.get("spent", 0)))
                if request_remaining < minimum <= remaining:
                    incidents.append({
                        **details, "code": "swe_request_credit_allocation_blocked",
                        "minimum_request_tokens": minimum,
                        "request_remaining": request_remaining,
                        "execution_credit": diagnostic.get("execution_credit"),
                    })
            before += int(event.get("token_in", 0)) + int(event.get("token_out", 0))
        total_spent = int(budget.get("used", before))

        # Only compare the committed current results. An unsuccessful attempt
        # whose incumbent was preserved must not create a false loss incident.
        for agent_id, artifact in current.items():
            ref = artifact.get("code_artifact_ref") or {}
            old = candidates.get(agent_id, {})
            old_ref = old.get("code_artifact_ref") or {}
            if (old_ref.get("patch_bytes", 0) > 0
                    and old.get("swe_progress", {}).get("trusted") is True
                    and old.get("artifact_id") != artifact.get("artifact_id")
                    and old_ref.get("artifact_sha256") not in step.get("recoverable_patch_hashes", ())
                    and not ref and artifact.get("failed")
                    and not artifact.get("changed_workspace")):
                incidents.append({
                    "code": "swe_candidate_recovery_failed", "resolved": False,
                    "event_id": step.get("event_id", ""), "agent_id": agent_id,
                    "candidate_artifact_id": old["artifact_id"],
                    "failed_artifact_id": artifact["artifact_id"],
                    "patch_sha256": old_ref["artifact_sha256"],
                    "candidate_tested": old["swe_progress"].get("test_after_latest_edit") is True,
                })
        # Resolve only from a runtime-exported, healthy current result. Scan
        # after recording losses so retained references on another node work
        # regardless of the iteration order of the graph's agents.
        for artifact in current.values():
            ref = artifact.get("code_artifact_ref") or {}
            if not ref or artifact.get("failed") or artifact.get("swe_progress", {}).get("trusted") is not True:
                continue
            for incident in incidents:
                if (incident["code"] == "swe_candidate_recovery_failed"
                        and incident["patch_sha256"] == ref.get("artifact_sha256")):
                    incident.update(resolved=True, recovered_artifact_id=artifact["artifact_id"])
        previous = {key: value for key, value in current.items() if key in node_ids}
    codes = sorted({incident["code"] for incident in incidents if not incident["resolved"]})
    return {
        "source": SOURCE, "attribution": "runtime_or_mixed" if codes else None,
        "blocks_policy_failure": bool(codes), "reason_codes": codes, "incidents": incidents,
    }
