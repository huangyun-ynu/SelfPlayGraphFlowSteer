from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .delegation import graph_delegation_issues
from .graph import MultiAgentGraph

LEGACY_REWARD_VERSION = "legacy_v1"
PROTOCOL_GATE_REWARD_VERSION = "protocol_gate_v1"
TASK_FIDELITY_REWARD_VERSION = "protocol_gate_v2"
SUPPORTED_REWARD_VERSIONS = frozenset(
    {
        LEGACY_REWARD_VERSION,
        PROTOCOL_GATE_REWARD_VERSION,
        TASK_FIDELITY_REWARD_VERSION,
    }
)


@dataclass(frozen=True)
class DirectorRewardConfig:
    """Versioned Director reward so resumed groups never mix reward semantics."""

    version: str = PROTOCOL_GATE_REWARD_VERSION

    def validate(self) -> None:
        if self.version not in SUPPORTED_REWARD_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_REWARD_VERSIONS))
            raise ValueError(f"director_reward.version must be one of: {supported}")


@dataclass(frozen=True)
class ProtocolReward:
    version: str
    delegation_complete: bool
    delegation_fidelity: bool
    delegation_issues: tuple[str, ...]
    graph_complete: bool
    finish_complete: bool
    execution_complete: bool
    protocol_score: float
    qualified: bool
    answer_reward_released: bool
    answer_score: float
    reward: float
    excluded_external_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.version != TASK_FIDELITY_REWARD_VERSION:
            payload.pop("delegation_fidelity", None)
            payload.pop("delegation_issues", None)
        return payload


def calculate_director_reward(
    *,
    version: str,
    graph: dict[str, Any],
    events: list[Any] | tuple[Any, ...],
    finished: bool,
    output: str,
    answer_score: float,
    task: str = "",
    submission_valid: bool = True,
    worker_backend_failure: bool = False,
    environment_commit_complete: bool = False,
) -> ProtocolReward:
    if version not in SUPPORTED_REWARD_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_REWARD_VERSIONS))
        raise ValueError(f"unsupported Director reward version {version!r}; use {supported}")
    score = float(answer_score)
    if not math.isfinite(score):
        raise ValueError("answer_score must be finite")

    components = _protocol_components(
        version=version,
        task=task,
        graph=graph,
        events=events,
        finished=finished,
        output=output,
        submission_valid=submission_valid,
        worker_backend_failure=worker_backend_failure,
        environment_commit_complete=environment_commit_complete,
    )
    if version == LEGACY_REWARD_VERSION:
        released = bool(finished and not worker_backend_failure)
        reward = 0.0 if worker_backend_failure else (score if released else -1.0)
        return ProtocolReward(
            version=version,
            **components,
            answer_reward_released=released,
            answer_score=score,
            reward=reward,
            excluded_external_failure=bool(worker_backend_failure),
        )

    protocol_score = float(components["protocol_score"])
    qualified = bool(components["qualified"])
    released = bool(qualified and not worker_backend_failure)
    reward = 0.0 if worker_backend_failure else -1.0 + protocol_score + (score if released else 0.0)
    return ProtocolReward(
        version=version,
        **components,
        answer_reward_released=released,
        answer_score=score,
        reward=reward,
        excluded_external_failure=bool(worker_backend_failure),
    )


def _protocol_components(
    *,
    version: str,
    task: str,
    graph: dict[str, Any],
    events: list[Any] | tuple[Any, ...],
    finished: bool,
    output: str,
    submission_valid: bool,
    worker_backend_failure: bool,
    environment_commit_complete: bool,
) -> dict[str, bool | float]:
    parsed = MultiAgentGraph.from_dict(graph)
    configured = bool(parsed.nodes) and all(node.configured for node in parsed.nodes.values())
    delegation_issues = (
        graph_delegation_issues(task, parsed)
        if version == TASK_FIDELITY_REWARD_VERSION and configured
        else ()
    )
    if version == TASK_FIDELITY_REWARD_VERSION and not str(task or "").strip():
        delegation_issues = ("missing task for protocol_gate_v2 delegation audit",)
    delegation_fidelity = configured and not delegation_issues
    delegation_complete = (
        delegation_fidelity if version == TASK_FIDELITY_REWARD_VERSION else configured
    )
    graph_complete = not parsed.validate(final=True)
    final_events = [event for event in events if _event_value(event, "final_execution")]
    finish_complete = len(final_events) == 1
    final_event = final_events[0] if finish_complete else None
    execution_complete = bool(
        finished
        and finish_complete
        and _event_value(final_event, "accepted")
        and (_event_value(final_event, "execution") is not None or environment_commit_complete)
        and str(output or "").strip()
        and not _is_failure_output(output)
        and submission_valid
        and not worker_backend_failure
    )
    checks = (
        delegation_complete,
        graph_complete,
        finish_complete,
        execution_complete,
    )
    protocol_score = sum(int(value) for value in checks) / len(checks)
    return {
        "delegation_complete": delegation_complete,
        "delegation_fidelity": delegation_fidelity,
        "delegation_issues": tuple(delegation_issues),
        "graph_complete": graph_complete,
        "finish_complete": finish_complete,
        "execution_complete": execution_complete,
        "protocol_score": protocol_score,
        "qualified": all(checks),
    }


def _event_value(event: Any, key: str) -> Any:
    if event is None:
        return None
    if isinstance(event, dict):
        payload = event.get("payload", event)
        return payload.get(key) if isinstance(payload, dict) else None
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(event, key, None)


def _is_failure_output(output: str) -> bool:
    normalized = str(output or "").strip().upper()
    return normalized in {
        "WORKER_BACKEND_FAILURE",
        "WORKER_PROTOCOL_FAILURE",
    }
