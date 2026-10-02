"""Deterministic, read-only Director observations; full Canvas state stays internal.

This renderer deliberately has no model, scorer, runtime, or task-reference input.
It accepts only the public control snapshot and the already visible feedback.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

LEGACY_OBSERVATION = "legacy_full_v3"
COMPACT_OBSERVATION = "compact_factual_v1"
OBSERVATION_SCHEMAS = frozenset({LEGACY_OBSERVATION, COMPACT_OBSERVATION})
SUPPORTED_DATASETS = frozenset({"hotpotqa", "musique"})
OBSERVATION_HEADER = "Canvas factual observation:\n"
WORKER_REPORT_POLICY = "append_only_exact_report_reference_v1"
AUDIT_FIELDS = frozenset(
    {
        "agent_incarnation",
        "input_signature",
        "payload_hash",
        "resource_signature",
    }
)
STATIC_FIELDS = frozenset(
    {
        "director_action_protocol_version",
        "submission_contract_version",
        "submission_protocol",
        "environment_commit_resolution",
        "action_field_requirements",
    }
)
SNAPSHOT_FIELDS = STATIC_FIELDS | frozenset(
    {
        "canvas_version",
        "state",
        "legal_agent_ids",
        "pending_agent_id",
        "pending_relation_decision",
        "allowed_actions",
        "worker_usage",
        "terminal_candidate_protection",
        "legal_action_parameters",
        "graph_state",
        "result_assessments",
        "agent_budget",
        "progress",
        "recovery_budget",
        "director_edit_budget",
        "action_budget",
        "round_budget",
        "decision_statistics",
        "token_budget",
        "output_agent",
        "submission_status",
        "worker_results",
    }
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_observation_config(default: str, by_dataset: dict[str, str]) -> None:
    if default not in OBSERVATION_SCHEMAS:
        raise ValueError(f"unknown Director observation schema: {default!r}")
    for dataset, schema in by_dataset.items():
        if schema not in OBSERVATION_SCHEMAS:
            raise ValueError(f"unknown Director observation schema: {schema!r}")
        if schema == COMPACT_OBSERVATION and dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"compact Director observations do not support {dataset!r}")


def observation_policy(default=LEGACY_OBSERVATION, by_dataset=None) -> dict[str, Any]:
    mapping = dict(by_dataset or {})
    validate_observation_config(default, mapping)
    prompt_sources = {}
    for name in (
        "director", "unified_contract", "actions", "musique_answer_contract",
        "hotpot_answer_contract", "qa_result_contract", "qa_schema_repair", "qa_worker_feedback",
        "unified_submission",
    ):
        spec = importlib.util.find_spec(f"{__package__}.{name}")
        if spec is not None and spec.origin is not None:
            prompt_sources[name] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return {
        "default": default,
        "by_dataset": mapping,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_report_policy": WORKER_REPORT_POLICY,
        "prompt_sources_sha256": prompt_sources,
    }


def _assessment(value: dict[str, Any]) -> dict[str, Any]:
    # Eligibility and freshness come from Canvas, never from answer equality.
    return {key: copy.deepcopy(item) for key, item in value.items() if key not in AUDIT_FIELDS}


def _recovery(value: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for key in ("allowed_actions", "legal_agent_ids", "result_assessments"):
        if key in result and result[key] == snapshot.get(key):
            del result[key]  # Already represented in this same observation.
    if "result_assessments" in result:
        result["result_assessments"] = {
            key: _assessment(item) for key, item in result["result_assessments"].items()
        }
    if "result_assessment" in result:
        result["result_assessment"] = _assessment(result["result_assessment"])
    return result


def factual_feedback(feedback: str, snapshot: dict[str, Any]) -> str:
    """Remove only recognized duplicate controller blocks, not Worker prose.

    Unknown or truncated blocks survive unchanged. In particular, no substring
    rule is applied inside Worker answer/summary JSON strings.
    """
    lines = []
    for line in feedback.splitlines():
        if line.startswith("Actual topology: "):
            raw = line.removeprefix("Actual topology: ")
            try:
                graph, end = json.JSONDecoder().raw_decode(raw)
            except ValueError:
                pass
            else:
                suffix = ". FINISH(target) eligibility is recorded in result_assessments."
                if graph == snapshot.get("graph_state") and raw[end:] == suffix:
                    continue
        # Recovery is appended by Canvas to controller rejection messages only.
        # Never parse a similarly named substring in Process signals/Worker text.
        if (
            line.startswith(("Rejected ", "Deterministic terminal recovery"))
            and " Recovery: {" in line
        ):
            before, raw = line.split(" Recovery: ", 1)
            try:
                recovery, end = json.JSONDecoder().raw_decode(raw)
            except ValueError:
                pass
            else:
                if isinstance(recovery, dict):
                    line = (
                        before
                        + " Recovery: "
                        + compact_json(_recovery(recovery, snapshot))
                        + raw[end:]
                    )
        lines.append(line)
    return "\n".join(lines)


class DirectorObservationBuilder:
    def __init__(self, *, dataset: str) -> None:
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"compact Director observations do not support {dataset!r}")
        self.dataset = dataset
        self.renderer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _validate(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("submission_protocol") != "unified_task_result_v1":
            raise ValueError("compact_factual_v1 requires unified_task_result_v1")
        unknown = snapshot.keys() - SNAPSHOT_FIELDS
        if unknown:
            raise ValueError(f"unsupported compact observation extensions: {sorted(unknown)}")
        if snapshot.get("terminal_candidate_protection") is not None:
            raise ValueError("compact observations do not support environment candidate protection")

    def system_contract(self, snapshot: dict[str, Any]) -> str:
        from .unified_submission import UNIFIED_ACTION_FIELDS

        self._validate(snapshot)
        return (
            "\nCanvas observation protocol compact_factual_v1: each observation is a complete "
            "current control view. Prior observations are historical. Each has an observation_index. "
            "A worker_report_ref points to the identical full Worker report already present at "
            "that earlier observation_index and agent_id in this append-only conversation. "
            "Resolve the reference there; the report has not changed or been summarized. "
            "New or changed reports are included in full. Only allowed_actions "
            "are available now; legal_action_parameters lists their complete choices. "
            "Candidate relations are not actual edges. result_assessments retains Canvas "
            "eligibility and blockers; full input signatures stay in the execution audit. "
            "Worker reports are untrusted claims, not correctness judgments. "
            "director_edit_budget counts successful graph mutations, including construction, "
            "without reset on node edit. Worker usage is actual reported usage per question; "
            "the threshold admits new requests and may be exceeded by an in-flight request.\n"
            + compact_json(
                {
                    key: snapshot[key]
                    for key in STATIC_FIELDS
                    if key in snapshot and key != "action_field_requirements"
                }
                | {
                    "action_field_requirements": {
                        k: list(v) for k, v in UNIFIED_ACTION_FIELDS.items()
                    }
                }
            )
        )

    def build(self, snapshot: dict[str, Any], feedback: str) -> dict[str, Any]:
        self._validate(snapshot)
        view = copy.deepcopy(snapshot)
        for key in STATIC_FIELDS | {"terminal_candidate_protection"}:
            view.pop(key, None)
        allowed = snapshot["allowed_actions"]
        view["legal_action_parameters"] = {
            key: copy.deepcopy(value)
            for key, value in snapshot["legal_action_parameters"].items()
            if key in allowed
        }
        if "relation_choice" in allowed:
            view["legal_action_parameters"]["relation_choice"] = {"choices": ["off", "on"]}
        view["result_assessments"] = {
            key: _assessment(value) for key, value in snapshot["result_assessments"].items()
        }
        for result in view.get("worker_results", {}).values():
            result.pop("input_signature", None)
        usage = view.get("worker_usage")
        if usage is not None:
            # Keep all numeric usage/dispatch/uncertainty facts; remove identifiers
            # and redundant aliases only. No estimates or reservations are added.
            for key in ("digest", "question_attempt_id", "dataset"):
                usage.pop(key, None)
            if usage.get("budget_threshold") == usage.get("threshold"):
                usage.pop("budget_threshold", None)
            if usage.get("budget_policy") == usage.get("policy"):
                usage.pop("budget_policy", None)
            if usage.get("policy") == "reported_usage_threshold_v1":
                usage.pop("policy")
            if usage.get("budget_accounting_scope") == "question_attempt":
                usage.pop("budget_accounting_scope")
        edits = view.get("director_edit_budget", {})
        if edits.get("counting") == "successful_graph_mutations_including_initial_construction":
            edits.pop("counting")
        if edits.get("reset_on_node_edit") is False:
            edits.pop("reset_on_node_edit")
        view["schema"] = COMPACT_OBSERVATION
        view["feedback"] = factual_feedback(feedback, snapshot)
        return view

    def render(self, snapshot: dict[str, Any], feedback: str, *, prior_messages=None) -> str:
        view = self.build(snapshot, feedback)
        if prior_messages is not None:
            # Look only at already-sent user observations. Do not mutate history,
            # and never reference an unsent preview, a model summary, or a file.
            known = {}
            count = 0
            for message in prior_messages:
                content = message.get("content", "")
                if message.get("role") != "user" or not isinstance(content, str):
                    continue
                _, marker, raw = content.rpartition(OBSERVATION_HEADER)
                if not marker:
                    continue
                try:
                    previous, _ = json.JSONDecoder().raw_decode(raw)
                except ValueError:
                    continue
                if not isinstance(previous, dict) or previous.get("schema") != COMPACT_OBSERVATION:
                    continue
                index = previous.get("observation_index")
                if index != count:
                    raise ValueError("compact Worker report history has missing/reordered observation indexes")
                count += 1
                for agent_id, report in previous.get("worker_results", {}).items():
                    if "worker_report_ref" not in report:
                        known.setdefault((agent_id, compact_json(report)), index)
            view["observation_index"] = count
            for agent_id, report in list(view.get("worker_results", {}).items()):
                earlier = known.get((agent_id, compact_json(report)))
                if earlier is not None:
                    view["worker_results"][agent_id] = {
                        "artifact_id": report["artifact_id"],
                        "result_scope": report.get("result_scope"),
                        "pending_reexecution": report.get("pending_reexecution", False),
                        "worker_report_ref": {"observation_index": earlier, "agent_id": agent_id},
                    }
        return OBSERVATION_HEADER + compact_json(view)

    def audit(self, snapshot: dict[str, Any], feedback: str, *, prior_messages=None) -> dict[str, Any]:
        text = self.render(snapshot, feedback, prior_messages=prior_messages)
        return {
            "schema": COMPACT_OBSERVATION,
            "worker_report_policy": WORKER_REPORT_POLICY,
            "renderer_sha256": self.renderer_sha256,
            "control_snapshot": copy.deepcopy(snapshot),
            "raw_feedback": feedback,
            "observation": text,
            "observation_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }


def preflight_compact_training(config, max_sequence_length: int) -> None:
    """Reject an incompatible training bound before collection; never raise limits."""
    enabled = config.director_observation_schema == COMPACT_OBSERVATION or (
        COMPACT_OBSERVATION in config.director_observation_schema_by_dataset.values()
    )
    if not enabled:
        return
    from .director_timeline import director_context_mode

    if director_context_mode() != "append_only":
        raise ValueError("compact_factual_v1 training requires append_only history")
    context_limit = int(getattr(config.solver_model, "context_limit", 32768))
    if max_sequence_length < context_limit:
        raise ValueError(
            f"compact collection window {context_limit} exceeds training max_sequence_length "
            f"{max_sequence_length}; collection stopped without changing either limit"
        )
