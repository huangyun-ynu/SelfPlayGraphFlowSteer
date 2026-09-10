from __future__ import annotations

import json
import math
import re
import string
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .aime_submission import is_aime_dataset, parse_aime_answer
from .canvas import GraphCanvas
from .dataset_actions import DatasetActionAdapter
from .graph import MultiAgentGraph
from .runtime import MultiAgentRuntime


class EnvironmentResultIncompleteError(ValueError):
    """A stateful environment ended without a trustworthy terminal result."""


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    reference: Any = None
    task_type: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)
    private_verifier_payload: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        private_payload = dict(self.private_verifier_payload)
        dataset = str(metadata.get("dataset", "")).strip().casefold()
        verifier = str(metadata.get("verifier", "")).strip().casefold()
        if dataset == "healthbench_professional" or verifier == "healthbench_rubric":
            for key in tuple(metadata):
                if _is_healthbench_private_key(key):
                    private_payload.setdefault(key, metadata.pop(key))
            # Closed-book HealthBench never consumes generic retrieval context.
            metadata.pop("context_documents", None)
            metadata = _sanitize_healthbench_public_value(metadata)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "private_verifier_payload", private_payload)


def _is_healthbench_private_key(key: object) -> bool:
    normalized = str(key).strip().casefold()
    return normalized in {"physician_response", "rubric_items"} or "canary" in normalized


def _sanitize_healthbench_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_healthbench_public_value(item)
            for key, item in value.items()
            if not _is_healthbench_private_key(key)
        }
    if isinstance(value, list):
        return [_sanitize_healthbench_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_healthbench_public_value(item) for item in value)
    return value


def task_to_public_dict(task: TaskSpec) -> dict[str, Any]:
    """Serialize a TaskSpec without verifier-only HealthBench payloads."""

    dataset = str(task.metadata.get("dataset", "")).strip().casefold()
    verifier = str(task.metadata.get("verifier", "")).strip().casefold()
    is_healthbench = (
        dataset == "healthbench_professional"
        or verifier == "healthbench_rubric"
        or bool(task.private_verifier_payload)
    )
    metadata = (
        _sanitize_healthbench_public_value(task.metadata) if is_healthbench else dict(task.metadata)
    )
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "reference": task.reference,
        "task_type": task.task_type,
        "metadata": metadata,
    }


@dataclass(frozen=True)
class VerificationResult:
    score: float
    passed: bool
    verifier: str
    detail: str = ""

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("verification score must be finite")
        object.__setattr__(self, "score", score)


class Verifier(Protocol):
    name: str

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult: ...


class ExactMatchVerifier:
    name = "exact_match"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        references = (
            task.reference if isinstance(task.reference, (list, tuple, set)) else [task.reference]
        )
        expected = [
            _normalize_answer("" if reference is None else str(reference))
            for reference in references
        ]
        actual_candidates = _normalized_qa_candidates(prediction)
        passed = any(actual and actual in expected for actual in actual_candidates)
        return VerificationResult(
            float(passed),
            passed,
            self.name,
            f"actual_candidates={actual_candidates!r}",
        )


class MultiAnswerExactMatchVerifier(ExactMatchVerifier):
    """Match one prediction against dataset-provided aliases, without adding aliases."""

    name = "multi_answer_exact_match"


class NumericVerifier:
    name = "numeric"

    def __init__(self, *, absolute_tolerance: float = 1e-6) -> None:
        self.absolute_tolerance = absolute_tolerance

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        expected = _final_number(str("" if task.reference is None else task.reference))
        dataset = str(task.metadata.get("dataset", "")).casefold()
        if is_aime_dataset(dataset):
            parsed = parse_aime_answer(prediction)
            if not parsed.valid:
                return VerificationResult(
                    0.0, False, self.name, f"invalid_answer_submission:{parsed.reason}"
                )
            actual = Decimal(parsed.answer)
            passed = (
                expected is not None
                and actual is not None
                and expected == actual
                and expected == expected.to_integral_value()
                and Decimal(0) <= actual <= Decimal(999)
            )
        else:
            actual = _final_number(prediction)
            passed = (
                expected is not None
                and actual is not None
                and math.isclose(
                    float(expected),
                    float(actual),
                    rel_tol=0.0,
                    abs_tol=self.absolute_tolerance,
                )
            )
        return VerificationResult(float(passed), passed, self.name, f"actual={actual!r}")


class MultipleChoiceVerifier:
    name = "multiple_choice"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        expected = _choice(str("" if task.reference is None else task.reference))
        actual = _choice(prediction)
        passed = expected is not None and actual == expected
        return VerificationResult(float(passed), passed, self.name, f"actual={actual!r}")


class AutoVerifier:
    """Choose a deterministic verifier from per-task metadata and task type."""

    name = "auto"

    def __init__(
        self,
        adapters: Mapping[str, Verifier] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.adapters = {
            str(name).strip().casefold(): verifier for name, verifier in (adapters or {}).items()
        }
        self.aliases = {
            str(name).strip().casefold(): str(target).strip().casefold()
            for name, target in (aliases or {}).items()
        }

    def resolve(self, task: TaskSpec) -> Verifier:
        """Resolve the concrete verifier for a task.

        Capability checks must use the concrete adapter.  Treating the outer
        ``AutoVerifier`` as universally capable of scoring arbitrary text can
        accidentally enable intermediate rewards for stateful, final-only
        environments.
        """

        requested = str(task.metadata.get("verifier", "")).strip().casefold()
        if not requested:
            task_type = task.task_type.casefold()
            if any(value in task_type for value in ("math", "numeric", "number")):
                requested = "numeric"
            elif any(value in task_type for value in ("multiple_choice", "choice", "mcq")):
                requested = "multiple_choice"
            else:
                requested = "exact_match"
        requested = self.aliases.get(requested, requested)
        verifier: Verifier | None = {
            "exact_match": ExactMatchVerifier(),
            "multi_answer_exact_match": MultiAnswerExactMatchVerifier(),
            "numeric": NumericVerifier(),
            "multiple_choice": MultipleChoiceVerifier(),
            **self.adapters,
        }.get(requested)
        if verifier is None:
            raise ValueError(
                f"verifier {requested!r} requires a dataset-specific environment adapter"
            )
        return verifier

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        return self.resolve(task).verify(task, prediction)

    def supports_intermediate_scoring_for(self, task: TaskSpec) -> bool:
        return bool(getattr(self.resolve(task), "supports_intermediate_scoring", True))

    def score_intermediate(self, task: TaskSpec, prediction: str) -> float:
        verifier = self.resolve(task)
        if not bool(getattr(verifier, "supports_intermediate_scoring", True)):
            raise ValueError(f"verifier {verifier.name!r} does not support intermediate scoring")
        scorer = getattr(verifier, "score_intermediate", None)
        if callable(scorer):
            return float(scorer(task, prediction))
        return float(verifier.verify(task, prediction).score)


def task_requires_reference(task: TaskSpec) -> bool:
    """Only answer-comparison verifiers require a textual reference answer."""

    requested = str(task.metadata.get("verifier", "")).strip().casefold()
    if requested:
        return requested in {
            "exact_match",
            "multi_answer_exact_match",
            "numeric",
            "multiple_choice",
        }
    task_type = task.task_type.casefold()
    return not any(
        value in task_type
        for value in ("webshop", "alfworld", "swe", "health", "environment", "rubric")
    )


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    kind: str
    payload: dict[str, Any]


@dataclass
class ExecutionTrace:
    run_id: str
    task: TaskSpec
    events: list[TraceEvent]
    final_graph: dict[str, Any]
    output: str = ""
    verification: VerificationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": task_to_public_dict(self.task),
            "events": [asdict(event) for event in self.events],
            "final_graph": self.final_graph,
            "output": self.output,
            "verification": asdict(self.verification) if self.verification else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionTrace:
        verification = payload.get("verification")
        return cls(
            run_id=str(payload["run_id"]),
            task=TaskSpec(**payload["task"]),
            events=[TraceEvent(**event) for event in payload.get("events", [])],
            final_graph=dict(payload["final_graph"]),
            output=str(payload.get("output", "")),
            verification=VerificationResult(**verification) if verification else None,
        )


def trace_from_canvas(
    *,
    run_id: str,
    task: TaskSpec,
    canvas: GraphCanvas,
    verification: VerificationResult | None = None,
) -> ExecutionTrace:
    events: list[TraceEvent] = []
    director_turn_index = 0
    graph_before = MultiAgentGraph(max_agents=canvas.graph.max_agents).to_dict()
    for sequence, step in enumerate(canvas.history):
        # A protocol-recovery step is an environment mutation, not a sampled
        # Director action.  Persist the mapping explicitly so Canvas event
        # sequence numbers are never confused with trainable model turns.
        model_turn = None if step.protocol_recovery else director_turn_index
        if model_turn is not None:
            director_turn_index += 1
        events.append(
            TraceEvent(
                sequence=sequence,
                kind="canvas_step",
                payload={
                    "raw_action": step.action.raw_text,
                    "accepted": step.accepted,
                    "feedback": step.feedback,
                    "graph_version": step.graph.get("version"),
                    "graph_before": graph_before,
                    "graph": step.graph,
                    "dirty_agents": list(step.dirty_agents),
                    "invalidated_agents": list(step.invalidated_agents),
                    "scheduled_agents": list(step.scheduled_agents),
                    "executed_agents": list(step.executed_agents),
                    "reused_agents": list(step.reused_agents),
                    "remaining_dirty": list(step.remaining_dirty),
                    "invalidation_reasons": {
                        agent_id: list(reasons)
                        for agent_id, reasons in step.invalidation_reasons.items()
                    },
                    "prompt_revision": dict(step.prompt_revision),
                    "execution": step.execution.to_dict() if step.execution else None,
                    "rejection_code": step.rejection_code,
                    "protocol_recovery": step.protocol_recovery,
                    "director_turn_index": model_turn,
                    "final_execution": step.final_execution,
                    "topology_audit": dict(step.topology_audit),
                    "structural_repair": dict(step.structural_repair),
                    "responsibility_issue": dict(step.responsibility_issue),
                    "responsibility_overlap_check": dict(step.responsibility_overlap_check),
                    "delegation_field_repairs": list(step.delegation_field_repairs),
                    "time_admission": dict(step.time_admission),
                    "token_admission": dict(step.token_admission),
                    "control_snapshot": dict(step.control_snapshot),
                    "rejection_details": dict(step.rejection_details),
                    "relation_decision": dict(step.relation_decision),
                    "invalid_repeat_count": step.invalid_repeat_count,
                    "topology_edits_frozen": step.topology_edits_frozen,
                },
            )
        )
        graph_before = step.graph
    output = ""
    if canvas.graph.output_agent:
        artifact = canvas.runtime.artifacts.get(canvas.graph.output_agent)
        output = artifact.answer if artifact else ""
    return ExecutionTrace(
        run_id=run_id,
        task=task,
        events=events,
        final_graph=canvas.graph.to_dict(),
        output=output,
        verification=verification,
    )


class JSONLTraceStore:
    _append_lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trace: ExecutionTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def load(self, run_id: str) -> ExecutionTrace:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if str(payload.get("run_id")) == run_id:
                    return ExecutionTrace.from_dict(payload)
        raise KeyError(f"unknown trace run_id: {run_id}")

    def list_run_ids(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [str(json.loads(line)["run_id"]) for line in handle if line.strip()]


def replay_trace(trace: ExecutionTrace, *, runtime: MultiAgentRuntime) -> GraphCanvas:
    # Local import avoids the TaskSpec/dataset-adapter module cycle while ensuring
    # replayed Workers receive the same trusted public q as live execution.
    from .dataset_adapters import solver_task_text

    raw_nodes = trace.final_graph.get("nodes", [])
    configured_nodes = [
        node
        for node in raw_nodes
        if isinstance(node, dict) and bool(node.get("operation_policy_configured"))
    ]
    action_adapter = None
    if configured_nodes:
        template = configured_nodes[0]
        metadata = template.get("metadata", {})
        adapter_id = (
            str(metadata.get("action_adapter", "")).strip() if isinstance(metadata, dict) else ""
        ) or "trace_replay"
        dataset = str(trace.task.metadata.get("dataset", adapter_id)).strip() or adapter_id
        action_adapter = DatasetActionAdapter(
            adapter_id=adapter_id,
            datasets=(dataset,),
            action_names=tuple(str(value) for value in template.get("allowed_tools", [])),
            initial_action_budget=int(template.get("initial_tool_budget", 0)),
            revision_action_budget=int(template.get("revision_tool_budget", 0)),
            total_action_budget=int(template.get("total_tool_budget", 0)),
        )
    canvas = GraphCanvas(
        task=solver_task_text(trace.task),
        runtime=runtime,
        action_adapter=action_adapter,
        dataset=str(trace.task.metadata.get("dataset", "")),
        managed_delegation_contracts=any(
            bool(node.get("metadata", {}).get("system_managed_contract", {}).get("version"))
            for node in raw_nodes
            if isinstance(node, dict)
        ),
        allow_legacy_prompts=True,
    )
    for event in trace.events:
        if event.kind != "canvas_step" or not event.payload.get("accepted"):
            continue
        raw_action = str(event.payload["raw_action"])
        if '"set_operation_policy"' in raw_action:
            continue
        step = canvas.step(raw_action)
        if not step.accepted:
            raise RuntimeError(
                f"trace replay diverged at sequence {event.sequence}: {step.feedback}"
            )
    replayed = canvas.graph.to_dict()
    expected = dict(trace.final_graph)
    replayed.pop("version", None)
    expected.pop("version", None)
    if replayed != expected:
        raise RuntimeError("trace replay produced a different final graph")
    return canvas


def _normalize_answer(value: str) -> str:
    """SQuAD/NQ-style exact-match normalization with Unicode punctuation."""

    lowered = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if char in string.punctuation or unicodedata.category(char).startswith("P") else char
        for char in lowered
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return re.sub(r"\s+", " ", without_articles).strip()


def _normalized_qa_candidates(value: str) -> list[str]:
    """Compatibility API returning at most one reference-blind candidate."""
    from .qa_submission import extract_qa_answer

    candidate = _normalize_answer(extract_qa_answer(value))
    return [candidate] if candidate else []


_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _final_number(value: str) -> Decimal | None:
    """Prefer an explicit final/boxed number, then fall back to the last number."""

    cleaned = value.replace(",", "")
    candidates: list[str] = []
    for pattern in (
        rf"\\boxed\s*\{{\s*({_NUMBER_PATTERN})\s*\}}",
        rf"(?:final\s+answer|answer|答案)\s*(?:is|=|:|：)?\s*({_NUMBER_PATTERN})",
    ):
        candidates.extend(re.findall(pattern, cleaned, flags=re.IGNORECASE))
    if not candidates:
        candidates = re.findall(_NUMBER_PATTERN, cleaned)
    if not candidates:
        return None
    try:
        return Decimal(candidates[-1])
    except InvalidOperation:
        return None


def _last_number(value: str) -> float | None:
    """Compatibility helper retained for callers outside this module."""

    number = _final_number(value)
    return float(number) if number is not None else None


def _choice(value: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", value.upper())
    return matches[-1] if matches else None
