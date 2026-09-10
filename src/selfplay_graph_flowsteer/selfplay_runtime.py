from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any
from .application import (
    AdaptiveApplicationResult,
    AdaptiveSolverApplication,
    GraphEvaluationBackendError,
    GraphEvaluationIncompleteError,
)
from .backend_failures import classify_backend_failure
from .config import canonical_dataset_name
from .counterfactual import evaluate_relation_decision, schedule_relation_decisions
from .deadline import RolloutDeadline, WorkerWallClockLimitExceeded
from .director_timeline import persist_context_policy
from .distributed import RolloutPool, ThreadRolloutPool
from .execution_audit import audit_cross_agent_read_overlap
from .features import execution_policy_features_many
from .graph import MultiAgentGraph
from .graph_learning import canonical_graph_key
from .llm import RequestTokenCreditExceeded, request_priority
from .observability import (
    EnvironmentResultIncompleteError,
    TaskSpec,
    task_requires_reference,
    task_to_public_dict,
)
from .proposer_learning import NORMALIZATION, freeze_proposer_baseline
from .protocol_reward import (
    LEGACY_REWARD_VERSION,
    PROTOCOL_GATE_REWARD_VERSION,
    calculate_director_reward,
)
from .qwen_compat import encode_chat_trajectory
from .rollouts import (
    DirectorTurn,
    TokenizedDirectorTrajectory,
    TokenizedPolicyCall,
    Tokenizer,
    tokenize_director_policy_calls,
)
from .route_health import PersistentRouteCircuitOpenError, RouteHealthStore
from .selfplay import (
    AlternatingSnapshots,
    DryRunSelfPlayResult,
    FrontierScore,
    ProposedTask,
    SeedInput,
    SolverRollout,
    TaskProposer,
    assemble_selfplay_result,
    graph_local_frontier,
    group_rollouts_by_task,
    select_frontier_reverification,
)
from .swebench import SWEWorkspaceProvisioningError, swe_task_is_training_split
from .training_selection import INDEPENDENT_SCHEMA, SELECTION_SCHEMAS, build_training_selection

OUTCOME_ONLY_REWARD_VERSION = "outcome_only_v1"
PRIMARY_JOB_ORDER_ROUND_ROBIN = "round_robin"
PRIMARY_JOB_ORDER_LONG_TAIL_FIRST = "long_tail_first"
PRIMARY_JOB_ORDER_CHOICES = frozenset(
    {PRIMARY_JOB_ORDER_ROUND_ROBIN, PRIMARY_JOB_ORDER_LONG_TAIL_FIRST}
)
PRIMARY_DATASET_DURATION_ESTIMATES_S = {
    "alfworld": 485.0,
    "swe_bench": 321.0,
    "webshop": 232.0,
    "aime": 182.0,
    "healthbench_professional": 129.0,
    "nq_open": 122.0,
    "hotpotqa": 88.0,
}
PRIMARY_DURATION_ESTIMATE_VERSION = "20260909-cycle1-14x5-v1"
_BINARY_OUTCOME_DATASETS = frozenset({"aime", "nq_open", "hotpotqa", "alfworld", "swe_bench"})


def _primary_job_dataset(job: tuple[ProposedTask, int]) -> str:
    (proposal, _rollout_index) = job
    return canonical_dataset_name(proposal.task.metadata.get("dataset", proposal.task.task_type))


def _primary_job_rollout_id(job: tuple[ProposedTask, int]) -> str:
    (proposal, rollout_index) = job
    return _rollout_id(str(proposal.task.task_id), rollout_index)


def _primary_job_duration_estimate_s(job: tuple[ProposedTask, int]) -> float:
    return PRIMARY_DATASET_DURATION_ESTIMATES_S.get(_primary_job_dataset(job), 182.0)


def _order_primary_jobs(
    jobs: Iterable[tuple[ProposedTask, int]], order: str
) -> list[tuple[ProposedTask, int]]:
    indexed = list(enumerate(jobs))
    if order == PRIMARY_JOB_ORDER_ROUND_ROBIN:
        return [job for (_index, job) in indexed]
    if order != PRIMARY_JOB_ORDER_LONG_TAIL_FIRST:
        raise ValueError(f"unknown primary job order: {order}")
    initial: list[tuple[int, tuple[ProposedTask, int]]] = []
    remaining: list[tuple[int, tuple[ProposedTask, int]]] = []
    admitted_tasks: set[str] = set()
    for original_index, job in indexed:
        task_id = str(job[0].task.task_id)
        if task_id not in admitted_tasks:
            admitted_tasks.add(task_id)
            initial.append((original_index, job))
        else:
            remaining.append((original_index, job))
    remaining.sort(key=lambda item: (-_primary_job_duration_estimate_s(item[1]), item[0]))
    return [job for (_index, job) in (*initial, *remaining)]


def _freeze_primary_job_schedule(
    output_dir: Path,
    *,
    window_start: int,
    jobs: Iterable[tuple[ProposedTask, int]],
    requested_order: str,
) -> tuple[list[tuple[ProposedTask, int]], dict[str, Any]]:
    """Persist and replay the exact job order for one physical task window."""
    path = output_dir / "rollout_job_schedule.json"
    job_list = list(jobs)
    jobs_by_id = {_primary_job_rollout_id(job): job for job in job_list}
    if len(jobs_by_id) != len(job_list):
        raise ValueError("primary rollout schedule contains duplicate rollout IDs")
    payload: dict[str, Any] = {
        "schema_version": "primary_job_schedule_v1",
        "duration_estimate_version": PRIMARY_DURATION_ESTIMATE_VERSION,
        "duration_estimates_s": dict(PRIMARY_DATASET_DURATION_ESTIMATES_S),
        "windows": [],
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "primary_job_schedule_v1":
            raise ValueError("unsupported primary rollout schedule schema")
    existing = [
        row
        for row in payload.get("windows", [])
        if int(row.get("window_start", -1)) == window_start
    ]
    if len(existing) > 1:
        raise ValueError("primary rollout schedule has duplicate physical windows")
    if existing:
        window = existing[0]
        ordered_ids = [str(value) for value in window.get("ordered_rollout_ids", [])]
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("frozen primary rollout schedule contains duplicate IDs")
        if set(ordered_ids) != set(jobs_by_id):
            raise ValueError("frozen primary rollout schedule does not match planned jobs")
        return ([jobs_by_id[rollout_id] for rollout_id in ordered_ids], window)
    ordered = _order_primary_jobs(jobs_by_id.values(), requested_order)
    ordered_ids = [_primary_job_rollout_id(job) for job in ordered]
    entries = [
        {
            "rank": rank,
            "rollout_id": rollout_id,
            "task_id": str(job[0].task.task_id),
            "rollout_index": int(job[1]),
            "dataset": _primary_job_dataset(job),
            "estimated_duration_s": _primary_job_duration_estimate_s(job),
        }
        for (rank, (rollout_id, job)) in enumerate(zip(ordered_ids, ordered, strict=True))
    ]
    window = {
        "window_start": window_start,
        "order": requested_order,
        "ordered_rollout_ids": ordered_ids,
        "order_sha256": hashlib.sha256(
            json.dumps(ordered_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "jobs": entries,
    }
    payload.setdefault("windows", []).append(window)
    _write_json(path, payload)
    return (ordered, window)


def _executor_compatibility_signature(manifest: dict[str, Any]) -> str:
    """Hash Executor semantics while ignoring attested transport-only changes.

    Dataset credential selection never changes the requested model. The
    HealthBench judge's Responses/Chat-Completions switch likewise changes the
    supported wire surface for the same grader model, and its scores are
    reverified separately. Keeping either field in the primary graph bundle
    makes an exact infrastructure repair look like a policy migration.
    """
    semantic = copy.deepcopy(manifest)
    runtime_environment = semantic.get("runtime_environment")
    if isinstance(runtime_environment, dict):
        runtime_environment.pop("dataset_api_key_overrides", None)
        runtime_environment.pop("dataset_concurrency_overrides", None)
    judge_route = str(semantic.get("healthbench_judge_audit", {}).get("runtime_route", ""))
    runtime_environments = semantic.get("runtime_environments", {})
    if isinstance(runtime_environments, dict):
        for route_name, environment in runtime_environments.items():
            if not isinstance(environment, dict):
                continue
            environment.pop("dataset_api_key_overrides", None)
            environment.pop("dataset_concurrency_overrides", None)
            if route_name == judge_route:
                environment.pop("api_surface", None)
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rollout_executor_compatibility_signature(rollout: SolverRollout) -> str:
    manifest = rollout.trajectory.metadata.get("model_roles")
    if isinstance(manifest, dict):
        return _executor_compatibility_signature(manifest)
    return str(rollout.trajectory.metadata.get("executor_bundle_signature", ""))


def _stable_execution_seed(base_seed: int, task_id: str, *, phase: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}\x00{task_id}\x00{phase}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 2147483647


def _rollout_sampling_seed(
    base_seed: int, task_id: str, rollout_index: int, replacement_attempt: int
) -> int:
    """Keep primary seeds stable while giving a recovery a new policy sample."""
    if replacement_attempt == 0:
        return int(base_seed) + int(rollout_index)
    return _stable_execution_seed(
        base_seed,
        f"{task_id}\x00{int(rollout_index)}\x00{int(replacement_attempt)}",
        phase="policy_recovery",
    )


def _outcome_task_reward(
    dataset: str, verification: Any, *, prediction: str
) -> tuple[float, dict[str, Any]]:
    if verification is None:
        return (0.0, {"source": "empty_or_missing_verification"})
    dataset_key = canonical_dataset_name(dataset)
    if dataset_key == "healthbench_professional":
        try:
            detail = json.loads(str(verification.detail or "{}"))
            breakdown = dict(detail["training_reward_breakdown"])
            reward = float(breakdown["training_reward"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "HealthBench verification lacks the versioned training reward breakdown"
            ) from exc
        if breakdown.get("version") != "healthbench_theoretical_bounds_length_v1":
            raise ValueError("unexpected HealthBench training reward adapter version")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("HealthBench training reward must be in [0, 1]")
        return (reward, {"source": "healthbench_training_adapter", **breakdown})
    reward = float(verification.score)
    if not 0.0 <= reward <= 1.0:
        raise ValueError(f"trusted task outcome for {dataset_key or dataset!r} must be in [0, 1]")
    if dataset_key in _BINARY_OUTCOME_DATASETS and reward not in {0.0, 1.0}:
        raise ValueError(
            f"trusted task outcome for binary dataset {dataset_key!r} must be exactly 0 or 1"
        )
    return (reward, {"source": "trusted_verifier_outcome", "evaluation_score": reward})


_SCOPED_ROUTE_CIRCUIT_DATASETS = frozenset(
    {"aime", "nq_open", "hotpotqa", "healthbench_professional"}
)
_JSONL_APPEND_LOCK = threading.Lock()
_MODEL_ATTRIBUTED_ACTION_FAILURE_CODES = frozenset(
    {
        "action_arguments_must_be_an_object",
        "invalid_action_arguments",
        "invalid_action_arguments_encoding",
        "action_not_visible",
        "tool_action_failed",
        "SecurityError",
        "ImportError",
        "SyntaxError",
        "NameError",
        "TypeError",
        "ValueError",
        "action_execution_failed",
    }
)


class WorkerBackendUnavailableError(RuntimeError):
    """A primary rollout is invalid because a required Worker backend failed."""

    def __init__(self, failure: dict[str, Any]) -> None:
        self.failure = dict(failure)
        self.routes = tuple((str(value) for value in failure.get("routes", ())))
        self.failure_types = tuple((str(value) for value in failure.get("failure_types", ())))
        self.agents = tuple((str(value) for value in failure.get("agents", ())))
        self.failure_details = tuple(
            (dict(value) for value in failure.get("failure_details", ()) if isinstance(value, dict))
        )
        self.request_events = tuple(
            (dict(value) for value in failure.get("request_events", ()) if isinstance(value, dict))
        )
        legacy = not self.failure_details
        self.retryable = bool(failure.get("retryable", legacy))
        self.counts_toward_route_circuit = bool(failure.get("counts_toward_route_circuit", legacy))
        self.disable_route = bool(failure.get("disable_route", False))
        super().__init__(
            f"required Worker backend failed; routes={','.join(self.routes) or 'unknown'}; failure_types={','.join(self.failure_types) or 'unknown'}"
        )

    def route_policy(self, route: str) -> tuple[bool, bool]:
        matching = [
            detail
            for detail in self.failure_details
            if str(detail.get("route", "") or route) == route
        ]
        if not matching:
            return (self.counts_toward_route_circuit, self.disable_route)
        return (
            any((bool(detail.get("counts_toward_route_circuit")) for detail in matching)),
            any((bool(detail.get("disable_route")) for detail in matching)),
        )


class InsufficientCompleteRolloutGroupsError(RuntimeError):
    """No atomic K-rollout task group is available for safe batch assembly."""


class NonTrainablePrimaryExhaustedError(RuntimeError):
    """A fixed slot exhausted recovery without a trainable terminal result."""


class UncertainAttributionExhaustedError(RuntimeError):
    """One permitted recovery did not resolve an attribution-uncertain failure."""


class CollectionInfrastructureIncidentError(RuntimeError):
    """A backend, tool, or environment incident invalidated the active cycle."""


class BackendRetryExhaustedError(RuntimeError):
    """An opt-in test run skipped a bounded backend-failure slot.

    This covers either exhausted retryable failures or a failure explicitly
    classified as permanent.  It is deliberately not an infrastructure
    incident: diagnostic runs may continue with the remaining complete groups.
    """

    def __init__(self, failure: dict[str, Any]) -> None:
        self.failure = dict(failure)
        self.routes = tuple((str(value) for value in failure.get("routes", ())))
        self.failure_types = tuple((str(value) for value in failure.get("failure_types", ())))
        self.failure_details = tuple(
            (dict(value) for value in failure.get("failure_details", ()) if isinstance(value, dict))
        )
        self.request_events = tuple(
            (dict(value) for value in failure.get("request_events", ()) if isinstance(value, dict))
        )
        legacy = not self.failure_details
        self.retryable = bool(failure.get("retryable", legacy))
        self.counts_toward_route_circuit = bool(failure.get("counts_toward_route_circuit", legacy))
        self.disable_route = bool(failure.get("disable_route", False))
        super().__init__("retryable Worker backend failure exhausted configured retries")

    def route_policy(self, route: str) -> tuple[bool, bool]:
        matching = [
            detail
            for detail in self.failure_details
            if str(detail.get("route", "") or route) == route
        ]
        if not matching:
            return (self.counts_toward_route_circuit, self.disable_route)
        return (
            any((bool(detail.get("counts_toward_route_circuit")) for detail in matching)),
            any((bool(detail.get("disable_route")) for detail in matching)),
        )


def _is_healthbench_judge_backend_failure(failure: dict[str, Any]) -> bool:
    return bool(failure.get("backend_failure")) and str(failure.get("message", "")).startswith(
        "HealthBench Judge failed closed at rubric "
    )


class RecoveryScope(StrEnum):
    """Smallest safe unit that can repair an invalid planned rollout slot."""

    NONE = "none"
    FULL_PRIMARY = "full_primary"
    FRESH_STATEFUL_SESSION = "fresh_stateful_session"


@dataclass(frozen=True)
class RecoveryDecision:
    scope: RecoveryScope
    reason: str
    infrastructure_incident: bool = False


def _recovery_decision(
    dataset: str, *, error: Exception | None = None, rollout: SolverRollout | None = None
) -> RecoveryDecision:
    """Classify invalid results without retrying ordinary task-level negatives."""
    dataset_key = canonical_dataset_name(dataset)
    if error is not None:
        if isinstance(error, EnvironmentResultIncompleteError):
            return RecoveryDecision(
                RecoveryScope.NONE,
                "stateful_environment_terminal_result_missing",
                infrastructure_incident=True,
            )
        if isinstance(error, WorkerBackendUnavailableError):
            return RecoveryDecision(
                RecoveryScope.NONE,
                "transient_worker_backend_failure"
                if error.retryable
                else "permanent_worker_backend_failure",
                infrastructure_incident=True,
            )
        if isinstance(error, WorkerWallClockLimitExceeded):
            return RecoveryDecision(RecoveryScope.FULL_PRIMARY, "rollout_wall_clock_limit")
        return RecoveryDecision(RecoveryScope.NONE, "non_recoverable_exception")
    if rollout is None:
        return RecoveryDecision(RecoveryScope.NONE, "missing_outcome")
    metadata = rollout.trajectory.metadata
    reasons = {str(value) for value in metadata.get("training_exclusion_reasons", ())}
    failure_mode = str(metadata.get("failure_mode", ""))
    swe_failure = metadata.get("swe_infrastructure_failure") or {}
    if (
        dataset_key == "swe_bench"
        and isinstance(swe_failure, dict)
        and (swe_failure.get("detail") in {"ssh_exit_255", "ssh_request_timeout"})
        and (not metadata.get("infrastructure_failure"))
    ):
        return RecoveryDecision(
            RecoveryScope.NONE,
            "isolated_swe_verifier_transport_failure",
            infrastructure_incident=True,
        )
    if bool(metadata.get("infrastructure_failure")) or bool(
        metadata.get("swe_infrastructure_failure")
    ):
        return RecoveryDecision(
            RecoveryScope.NONE, "runtime_environment_or_tool_failure", infrastructure_incident=True
        )
    if dataset_key == "swe_bench":
        return RecoveryDecision(RecoveryScope.FULL_PRIMARY, "swe_terminal_contract")
    if dataset_key in {"webshop", "alfworld"} and (
        "worker_artifact_integrity_failure" in reasons
        or failure_mode == "worker_artifact_integrity_failure"
    ):
        return RecoveryDecision(
            RecoveryScope.FRESH_STATEFUL_SESSION, "stateful_environment_or_action_integrity_failure"
        )
    if (
        reasons
        & {
            "not_finished",
            "invalid_final_graph",
            "execution_incomplete",
            "execution_budget_exceeded",
        }
        or failure_mode == "director_protocol_failure"
    ):
        return RecoveryDecision(RecoveryScope.FULL_PRIMARY, "director_protocol_failure")
    if dataset_key == "aime" and (
        "worker_artifact_integrity_failure" in reasons
        or failure_mode == "worker_artifact_integrity_failure"
    ):
        return RecoveryDecision(
            RecoveryScope.FULL_PRIMARY, "aime_selected_output_recovery_exhausted"
        )
    return RecoveryDecision(RecoveryScope.NONE, "dataset_terminal_contract_failed")


def _rollout_is_training_eligible(rollout: SolverRollout) -> bool:
    """Treat explicit current-protocol exclusions as unsafe; preserve legacy rows."""
    metadata = rollout.trajectory.metadata
    if "training_eligible" not in metadata:
        return True
    return bool(metadata.get("training_eligible"))


def _uncertain_failure_zero(rollout: SolverRollout, reason: str) -> SolverRollout:
    """Apply an explicit scoring convention, without inventing policy attribution.

    Preserve trusted scores and all policy/training exclusions. A scored slot is
    not necessarily a trainable slot, and raw verification remains audit evidence.
    """
    metadata = rollout.trajectory.metadata
    if metadata.get("reward_known") is True or any(
        (
            metadata.get(key)
            for key in (
                "worker_backend_failure",
                "infrastructure_failure",
                "swe_infrastructure_failure",
                "swe_synthetic_evaluation",
                "swe_non_train_split",
            )
        )
    ):
        return rollout
    failure = {
        "attribution": "unknown",
        "reason": reason,
        "scoring_policy": "uncertain_attribution_zero_v1",
        "original_reward": rollout.trajectory.reward,
        "original_training_exclusion_reasons": list(metadata.get("training_exclusion_reasons", ())),
        "original_reward_admission_reason": metadata.get("reward_admission_reason"),
    }
    metadata = {
        **metadata,
        "reward_known": True,
        "task_outcome_passed": False,
        "task_reward": 0.0,
        "director_reward": 0.0,
        "base_director_reward": 0.0,
        "reward_admission_reason": "uncertain_attribution_zero",
        "uncertain_attribution_zero": failure,
        "task_reward_breakdown": {"source": "uncertain_attribution_zero", "failure": failure},
    }
    return replace(rollout, trajectory=replace(rollout.trajectory, reward=0.0, metadata=metadata))


def _rollout_has_trusted_score(rollout: SolverRollout) -> bool:
    """Outcome admission is independent of whether the policy record can train."""
    metadata = rollout.trajectory.metadata
    return (
        metadata.get("reward_known") is True
        and metadata.get("reward_admission_reason")
        in {"trusted_task_result", "explicit_policy_terminal"}
        and math.isfinite(rollout.trajectory.reward)
    )


def _rollout_training_exclusion(rollout: SolverRollout) -> dict[str, Any]:
    metadata = rollout.trajectory.metadata
    return {
        "rollout_id": rollout.trajectory.rollout_id,
        "task_id": rollout.trajectory.task_id,
        "training_eligible": False,
        "terminal_graph_status": metadata.get("terminal_graph_status"),
        "training_exclusion_reasons": list(metadata.get("training_exclusion_reasons", ())),
        "finished": bool(metadata.get("finished")),
        "failure_mode": metadata.get("failure_mode"),
    }


def _runtime_owned_model_policy_failure(
    dataset: str,
    *,
    training_exclusion_reasons: list[str],
    verification: object | None,
    worker_backend_failure: object,
    worker_artifact_integrity: object,
    worker_artifact_integrity_failure: object,
    swe_output_progress: object,
    alfworld_output_progress: object,
    webshop_output_progress: object = None,
    swe_environment_result: object,
    swe_infrastructure_failure: object,
    swe_synthetic_evaluation: bool,
    swe_non_train_split: bool,
    stateful_environment_result: object,
    admit_stateful_policy_failure_terminal: bool,
) -> dict[str, Any] | None:
    """Turn model-attributed terminal failures into auditable negative samples.

    Infrastructure and backend failures remain excluded. Admitted cases have a
    complete runtime-owned reward signal and enough trusted evidence to assign
    the failure to model policy rather than to the execution environment.
    """
    if (
        verification is None
        or worker_backend_failure
        or swe_infrastructure_failure
        or swe_synthetic_evaluation
        or swe_non_train_split
    ):
        return None
    exclusions = set(training_exclusion_reasons)
    failure = (
        worker_artifact_integrity_failure
        if isinstance(worker_artifact_integrity_failure, dict)
        else {}
    )
    dataset_key = canonical_dataset_name(dataset)
    artifacts = worker_artifact_integrity if isinstance(worker_artifact_integrity, dict) else {}
    if dataset_key == "aime" and (not failure):
        terminal_candidates = [
            (str(agent_id), artifact)
            for (agent_id, artifact) in artifacts.items()
            if isinstance(artifact, dict)
            and "terminal_protocol_failure"
            in {str(value) for value in artifact.get("integrity_risks", ())}
        ]
        if len(terminal_candidates) == 1:
            (agent_id, artifact) = terminal_candidates[0]
            failure = {
                "output_agent": agent_id,
                "risks": list(artifact.get("integrity_risks", ())),
                "inferred_unselected_output_agent": True,
            }
    risks = {str(value) for value in failure.get("risks", ())}
    stateful_environment = (
        stateful_environment_result if isinstance(stateful_environment_result, dict) else {}
    )
    alfworld_progress = (
        alfworld_output_progress if isinstance(alfworld_output_progress, dict) else {}
    )
    webshop_progress = webshop_output_progress if isinstance(webshop_output_progress, dict) else {}
    staged_agents = []
    if (
        dataset_key == "webshop"
        and exclusions
        and (exclusions <= {"not_finished", "execution_incomplete"})
        and (not worker_artifact_integrity_failure)
        and (getattr(verification, "verifier", "") == "webshop_environment")
        and (float(getattr(verification, "score", -1.0)) == 0.0)
        and (getattr(verification, "passed", True) is False)
    ):
        for agent_id, artifact in artifacts.items():
            if not isinstance(artifact, dict):
                continue
            progress = artifact.get("webshop_progress", {})
            evidence = artifact.get("runtime_tool_evidence", {})
            if (
                progress.get("trusted") is True
                and progress.get("state") == "purchase_staged"
                and (progress.get("commit_ready") is True)
                and (progress.get("commit_protocol_status") == "awaiting_canvas_output_selection")
                and (evidence.get("trusted") is True)
                and (int(evidence.get("successful_count", 0)) > 0)
                and (int(evidence.get("failed_count", 0)) == 0)
                and (not evidence.get("failure_codes"))
            ):
                staged_agents.append(str(agent_id))
        if staged_agents:
            return {
                "status": "typed_policy_failure",
                "dataset": dataset_key,
                "code": "webshop_director_staged_purchase_not_committed",
                "attribution": "director_policy",
                "official_score": 0.0,
                "commit_ready_agents": sorted(staged_agents),
                "source": "runtime_staged_purchase_evidence",
                "original_training_exclusion_reasons": sorted(exclusions),
            }
    allowed_stateful_exclusions = {"not_finished", "invalid_final_graph", "execution_incomplete"}
    allowed_exhausted_missing_output_exclusions = {
        *allowed_stateful_exclusions,
        "execution_budget_exceeded",
    }
    allowed_webshop_closure_exclusions = {
        *allowed_stateful_exclusions,
        "webshop_output_closure_incomplete",
        "worker_artifact_integrity_failure",
    }
    allowed_webshop_semantic_stall_exclusions = {
        *allowed_stateful_exclusions,
        "webshop_output_closure_incomplete",
    }
    policy_failure = alfworld_progress.get("policy_failure", {})
    if (
        dataset_key == "alfworld"
        and exclusions <= allowed_stateful_exclusions
        and (not worker_artifact_integrity_failure)
        and (alfworld_progress.get("trusted") is True)
        and (alfworld_progress.get("state") == "typed_policy_failure")
        and isinstance(policy_failure, dict)
        and (policy_failure.get("status") == "typed_policy_failure")
        and (policy_failure.get("attribution") == "model_policy")
        and (policy_failure.get("code") == "alfworld_semantic_no_progress")
        and (
            int(policy_failure.get("semantic_no_progress_streak", 0) or 0)
            >= int(policy_failure.get("fuse_threshold", 4) or 4)
        )
        and (int(policy_failure.get("repeated_transition_count", 0) or 0) > 0)
        and (not bool(stateful_environment.get("done", False)))
        and (not bool(stateful_environment.get("budget_truncated", False)))
    ):
        return {
            "status": "typed_policy_failure",
            "code": "alfworld_semantic_no_progress",
            "dataset": dataset_key,
            "attribution": "model_policy",
            "official_environment_terminal": False,
            "runtime_terminal": True,
            "original_training_exclusion_reasons": sorted(exclusions),
            "semantic_no_progress_count": int(
                policy_failure.get("semantic_no_progress_count", 0) or 0
            ),
            "semantic_no_progress_streak": int(
                policy_failure.get("semantic_no_progress_streak", 0) or 0
            ),
            "repeated_transition_count": int(
                policy_failure.get("repeated_transition_count", 0) or 0
            ),
            "last_commands": list(policy_failure.get("last_commands", ())),
        }
    if (
        dataset_key == "webshop"
        and exclusions <= allowed_webshop_semantic_stall_exclusions
        and (not worker_artifact_integrity_failure)
        and (not bool(stateful_environment.get("done", False)))
        and (not bool(stateful_environment.get("budget_truncated", False)))
    ):
        stalled_candidates: list[tuple[str, dict[str, Any]]] = []
        for agent_id, artifact in artifacts.items():
            if not isinstance(artifact, dict):
                continue
            progress = artifact.get("webshop_progress", {})
            if not isinstance(progress, dict) or progress.get("trusted") is not True:
                continue
            candidate = progress.get("policy_failure", {})
            if (
                progress.get("state") == "typed_policy_failure"
                and isinstance(candidate, dict)
                and (candidate.get("status") == "typed_policy_failure")
                and (candidate.get("attribution") == "model_policy")
                and (candidate.get("code") == "webshop_semantic_no_progress")
                and (
                    int(candidate.get("semantic_no_progress_streak", 0) or 0)
                    >= int(candidate.get("fuse_threshold", 4) or 4)
                )
            ):
                stalled_candidates.append((str(agent_id), candidate))
        if stalled_candidates:
            (agent_id, policy_failure) = max(
                stalled_candidates,
                key=lambda item: int(item[1].get("semantic_no_progress_streak", 0) or 0),
            )
            return {
                "status": "typed_policy_failure",
                "code": "webshop_semantic_no_progress",
                "dataset": dataset_key,
                "attribution": "model_policy",
                "worker_agent": agent_id,
                "official_environment_terminal": False,
                "runtime_terminal": True,
                "original_training_exclusion_reasons": sorted(exclusions),
                "semantic_no_progress_count": int(
                    policy_failure.get("semantic_no_progress_count", 0) or 0
                ),
                "semantic_no_progress_streak": int(
                    policy_failure.get("semantic_no_progress_streak", 0) or 0
                ),
                "duplicate_state_action_count": int(
                    policy_failure.get("duplicate_state_action_count", 0) or 0
                ),
                "unique_public_evidence_count": int(
                    policy_failure.get("unique_public_evidence_count", 0) or 0
                ),
            }
    closure_integrity_risks = {
        str(value)
        for value in (
            worker_artifact_integrity_failure.get("risks", ())
            if isinstance(worker_artifact_integrity_failure, dict)
            else ()
        )
    }
    closure_ended_at_official_step_limit = bool(
        stateful_environment.get("done", False)
        and str(stateful_environment.get("termination_reason", "")) == "step_limit"
        and closure_integrity_risks
        and (closure_integrity_risks <= {"terminal_tool_failure"})
    )
    closure_integrity_is_trusted = bool(
        not worker_artifact_integrity_failure or closure_ended_at_official_step_limit
    )
    if (
        dataset_key == "webshop"
        and "webshop_output_closure_incomplete" in exclusions
        and (exclusions <= allowed_webshop_closure_exclusions)
        and closure_integrity_is_trusted
        and (getattr(verification, "verifier", "") == "webshop_environment")
        and (float(getattr(verification, "score", -1.0)) == 0.0)
        and (getattr(verification, "passed", True) is False)
        and (webshop_progress.get("trusted") is True)
        and (webshop_progress.get("state") == "completed")
        and bool(str(webshop_progress.get("environment_owner", "")).strip())
        and (webshop_progress.get("environment_access") == "mutable_owner")
        and ("reward" in stateful_environment)
        and (float(stateful_environment.get("reward", -1.0)) == 0.0)
        and (stateful_environment.get("purchased") is False)
        and (not bool(stateful_environment.get("purchase_committed", False)))
        and (str(stateful_environment.get("termination_reason", "")) in {"active", "step_limit"})
    ):
        return {
            "status": "typed_policy_failure",
            "code": "webshop_output_closure_incomplete",
            "dataset": dataset_key,
            "attribution": "model_policy",
            "official_environment_terminal": bool(stateful_environment.get("done", False)),
            "runtime_terminal": True,
            "official_score": 0.0,
            "environment_owner": str(webshop_progress["environment_owner"]),
            "environment_steps": int(stateful_environment.get("steps", 0) or 0),
            "termination_reason": str(stateful_environment.get("termination_reason", "")),
            "original_training_exclusion_reasons": sorted(exclusions),
            "official_step_limit": closure_ended_at_official_step_limit,
        }
    if (
        dataset_key in {"webshop", "alfworld"}
        and admit_stateful_policy_failure_terminal
        and exclusions
        and (
            exclusions
            <= (
                allowed_exhausted_missing_output_exclusions
                if dataset_key == "webshop"
                else allowed_stateful_exclusions
            )
        )
        and (not worker_artifact_integrity_failure)
        and (str(stateful_environment.get("termination_reason", "")) == "missing_output_agent")
        and (not bool(stateful_environment.get("done", False)))
        and (not bool(stateful_environment.get("budget_truncated", False)))
        and (int(stateful_environment.get("steps", 0) or 0) == 0)
    ):
        if dataset_key == "webshop":
            trusted_progress = {
                str(agent_id): artifact.get("webshop_progress", {})
                for (agent_id, artifact) in artifacts.items()
                if isinstance(artifact, dict)
                and isinstance(artifact.get("webshop_progress"), dict)
                and (artifact["webshop_progress"].get("trusted") is True)
            }
            if not trusted_progress:
                return None
            commit_ready_agents = sorted(
                (
                    agent_id
                    for (agent_id, progress) in trusted_progress.items()
                    if progress.get("commit_ready") is True
                )
            )
            if commit_ready_agents:
                return {
                    "status": "typed_policy_failure",
                    "code": "webshop_director_commit_ready_not_selected",
                    "dataset": dataset_key,
                    "attribution": "director_policy",
                    "termination_reason": "missing_output_agent",
                    "commit_ready_agents": commit_ready_agents,
                    "original_training_exclusion_reasons": sorted(exclusions),
                    "recovery_exhausted": True,
                }
            return {
                "status": "typed_policy_failure",
                "code": "webshop_worker_no_staged_purchase",
                "dataset": dataset_key,
                "attribution": "worker_policy",
                "termination_reason": "missing_output_agent",
                "worker_agents": sorted(trusted_progress),
                "worker_progress_states": {
                    agent_id: str(progress.get("state", ""))
                    for (agent_id, progress) in trusted_progress.items()
                },
                "original_training_exclusion_reasons": sorted(exclusions),
                "recovery_exhausted": True,
            }
        return {
            "status": "typed_policy_failure",
            "code": "alfworld_director_missing_output_policy_failure",
            "dataset": dataset_key,
            "termination_reason": "missing_output_agent",
            "original_training_exclusion_reasons": sorted(exclusions),
            "recovery_exhausted": True,
        }
    if dataset_key == "aime" and exclusions <= {
        "worker_artifact_integrity_failure",
        "execution_incomplete",
        "invalid_final_graph",
        "not_finished",
    }:
        output_agent = str(failure.get("output_agent", ""))
        artifact = artifacts.get(output_agent, {})
        tool_evidence = (
            artifact.get("runtime_tool_evidence", {}) if isinstance(artifact, dict) else {}
        )
        failure_codes = {str(value) for value in tool_evidence.get("failure_codes", ())}
        if (
            int(tool_evidence.get("attempted_count", 0)) > 0
            and failure_codes
            and (failure_codes <= _MODEL_ATTRIBUTED_ACTION_FAILURE_CODES)
            and risks
            and (
                risks
                <= {
                    "all_tool_actions_failed",
                    "terminal_tool_failure",
                    "unsupported_tool_verification_claim",
                }
            )
        ):
            return {
                "status": "typed_policy_failure",
                "code": "aime_model_tool_policy_failure",
                "dataset": dataset_key,
                "original_training_exclusion_reasons": sorted(exclusions),
                "integrity_risks": sorted(risks),
                "tool_failure_codes": sorted(failure_codes),
            }
        if risks == {"terminal_protocol_failure"}:
            return {
                "status": "typed_policy_failure",
                "code": "aime_worker_final_protocol_policy_failure",
                "dataset": dataset_key,
                "original_training_exclusion_reasons": sorted(exclusions),
                "integrity_risks": sorted(risks),
            }
    allowed_swe_exclusions = {
        "not_finished",
        "invalid_final_graph",
        "execution_incomplete",
        "worker_artifact_integrity_failure",
    }
    progress = swe_output_progress if isinstance(swe_output_progress, dict) else {}
    environment = swe_environment_result if isinstance(swe_environment_result, dict) else {}
    if (
        dataset_key == "swe_bench"
        and exclusions
        and (exclusions <= allowed_swe_exclusions)
        and risks
        and (risks <= {"terminal_protocol_failure"})
        and (progress.get("trusted") is True)
        and (progress.get("selected_as_output") is True)
        and (progress.get("commit_ready") is True)
        and (progress.get("workspace_changed") is True)
        and (progress.get("test_after_latest_edit") is True)
        and (environment.get("official") is True)
        and (environment.get("synthetic") is False)
        and (environment.get("environment_completed") is True)
    ):
        return {
            "status": "typed_policy_failure",
            "code": "swe_post_commit_protocol_failure",
            "dataset": dataset_key,
            "official_status": str(environment.get("status", "")),
            "original_training_exclusion_reasons": sorted(exclusions),
            "integrity_risks": sorted(risks),
        }
    return None


@dataclass
class _PrimaryCollection:
    """Live state retained only after the primary rollout is durable."""

    proposal: ProposedTask
    rollout_index: int
    rollout_seed: int
    executor_seed: int
    application: AdaptiveSolverApplication
    result: AdaptiveApplicationResult
    rollout: SolverRollout
    primary_duration_s: float
    decisions: tuple[Any, ...] = ()
    counterfactual_cancellation_event: threading.Event | None = None
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        close = getattr(self.application, "close", None)
        if callable(close):
            close()
        self.closed = True


@dataclass
class _WindowPipelineState:
    window_start: int
    window_entries: list[tuple[str, ProposedTask]]
    window_complete_entries: list[tuple[str, ProposedTask]]
    collection_errors: list[dict[str, Any]]
    counterfactual_jobs: list[_PrimaryCollection]
    circuit_open: bool
    counterfactual_future: Future[list[tuple[_PrimaryCollection, Any, Exception | None]]] | None = (
        None
    )


class ByteTokenizer:
    approximate_token_count = True
    "Deterministic tokenizer for smoke tests only."

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "off":
            return [256]
        if text == "on":
            return [257]
        return list(text.encode("utf-8"))

    def token_span_for_char_span(self, text: str, span: tuple[int, int]) -> tuple[int, int]:
        (start, end) = span
        return (len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8")))


class HuggingFaceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'selfplay' extra to use a real tokenizer") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def token_span_for_char_span(self, text: str, span: tuple[int, int]) -> tuple[int, int] | None:
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0][0], list):
            offsets = offsets[0]
        (start, end) = span
        selected = [
            index
            for (index, (left, right)) in enumerate(offsets)
            if int(right) > start and int(left) < end
        ]
        if not selected:
            return None
        (first, last) = (selected[0], selected[-1])
        if int(offsets[first][0]) != start or int(offsets[last][1]) != end:
            return None
        return (first, last + 1)

    def encode_chat_trajectory(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        return encode_chat_trajectory(self.tokenizer, messages, enable_thinking=False)


def adaptive_result_to_rollout(
    result: AdaptiveApplicationResult,
    tokenizer: Tokenizer,
    *,
    rollout_index: int,
    seed: int,
    max_tokens: int = 4096,
    duration_s: float | None = None,
    reward_version: str = PROTOCOL_GATE_REWARD_VERSION,
) -> SolverRollout:
    run = result.solver_result.director_run
    prefixes = {
        int(event.payload["director_turn_index"]): event.payload.get("graph_before", {})
        for event in result.solver_result.trace.events
        if event.kind == "canvas_step" and event.payload.get("director_turn_index") is not None
    }
    turns = [
        DirectorTurn(
            model_response=turn.model_action,
            feedback=turn.feedback,
            accepted=turn.accepted,
            graph_prefix=prefixes.get(index, {}),
            prompt_messages=tuple(turn.prompt_messages),
            trainable=turn.trainable,
            call_id=turn.call_id,
            completion_token_ids=tuple(turn.completion_token_ids),
            behavior_log_probs=tuple(turn.behavior_log_probs),
            raw_reasoning_text=turn.raw_reasoning_text,
            raw_action_text=turn.raw_action_text,
            prompt_token_ids=tuple(turn.prompt_token_ids),
            action_character_span=turn.action_character_span,
            turn_kind=turn.turn_kind,
            metadata={
                "model_id": turn.model_id,
                "route_name": turn.route_name,
                "thinking_requested": turn.thinking_requested,
                "thinking_effective": turn.thinking_effective,
                "token_provenance": turn.token_provenance,
                "trajectory_schema": turn.trajectory_schema,
                "relation_policy": dict(turn.relation_decision.get("policy", {})),
            },
        )
        for (index, turn) in enumerate(run.turns)
    ]
    policy_encoding_error = None
    try:
        policy_calls = tokenize_director_policy_calls(turns, tokenizer, max_tokens=max_tokens)
    except ValueError as exc:
        policy_calls = ()
        policy_encoding_error = str(exc)
    flattened_ids: list[int] = []
    flattened_mask: list[int] = []
    action_spans: list[tuple[int, int] | None] = []
    for turn in turns:
        completion_ids = list(
            turn.completion_token_ids
            or tokenizer.encode(
                turn.raw_reasoning_text + turn.raw_action_text, add_special_tokens=False
            )
        )
        start = len(flattened_ids)
        flattened_ids.extend(completion_ids)
        flattened_mask.extend([int(turn.trainable)] * len(completion_ids))
        action_spans.append((start, len(flattened_ids)) if completion_ids else None)
    (token_ids, mask) = (tuple(flattened_ids), tuple(flattened_mask))
    relation_choice_spans: list[tuple[int, int] | None] = list(action_spans)
    for index, turn in enumerate(run.turns):
        if getattr(turn, "turn_kind", "graph_action") != "relation_choice":
            relation_choice_spans[index] = None
            continue
        span = action_spans[index]
        policy = turn.relation_decision.get("policy", {})
        token_map = policy.get("token_ids", {}) if isinstance(policy, dict) else {}
        try:
            expected_id = int(token_map[turn.model_action])
        except (KeyError, TypeError, ValueError):
            relation_choice_spans[index] = None
            continue
        positions = (
            [position for position in range(span[0], span[1]) if token_ids[position] == expected_id]
            if span is not None
            else []
        )
        relation_choice_spans[index] = (
            (positions[0], positions[0] + 1) if len(positions) == 1 else None
        )
    verification = result.solver_result.verification
    structure = result.solver_result.flowsteer_structure
    worker_backend_failure = result.task.metadata.get("worker_backend_failure")
    worker_artifact_integrity = result.task.metadata.get("worker_artifact_integrity", {})
    worker_output_integrity_risks = result.task.metadata.get("worker_output_integrity_risks", [])
    worker_artifact_integrity_failure = result.task.metadata.get(
        "worker_artifact_integrity_failure"
    )
    swe_output_progress = result.task.metadata.get("swe_output_progress", {})
    alfworld_output_progress = result.task.metadata.get("alfworld_output_progress", {})
    webshop_output_progress = result.task.metadata.get("webshop_output_progress", {})
    swe_infrastructure_failure = result.task.metadata.get("swe_infrastructure_failure")
    swe_environment_result = result.task.metadata.get("swe_environment_result")
    swe_synthetic_evaluation = bool(
        isinstance(swe_environment_result, dict) and swe_environment_result.get("synthetic", False)
    )
    is_swe_task = str(result.task.metadata.get("dataset", "")).strip().casefold() in {
        "swe_bench",
        "swe-bench",
        "swebench",
    }
    swe_non_train_split = bool(
        is_swe_task and (not swe_task_is_training_split(result.task.metadata))
    )
    dataset_key = canonical_dataset_name(result.task.metadata.get("dataset", ""))
    stateful_environment_result = (
        result.task.metadata.get("alfworld_environment_result", {})
        if dataset_key == "alfworld"
        else result.task.metadata.get("webshop_environment_result", {})
        if dataset_key == "webshop"
        else {}
    )
    environment_commit_execution_complete = bool(
        dataset_key == "webshop"
        and isinstance(stateful_environment_result, dict)
        and (stateful_environment_result.get("purchase_executed") is True)
        and (stateful_environment_result.get("purchase_committed") is True)
        and (stateful_environment_result.get("purchased") is True)
        and (stateful_environment_result.get("terminal") is True)
    )
    answer_score = float(verification.score) if verification else 0.0
    result_payload = result.to_dict()
    submission = result.solver_result.answer_submission
    submitted_output = submission.submitted_answer if submission else run.output
    protocol_reward = calculate_director_reward(
        version=reward_version,
        task=result.task.prompt,
        graph=run.graph,
        events=result.solver_result.trace.events,
        finished=run.finished,
        output=submitted_output,
        answer_score=answer_score,
        submission_valid=submission.valid if submission else bool(run.output.strip()),
        worker_backend_failure=bool(worker_backend_failure),
        environment_commit_complete=environment_commit_execution_complete,
    )
    answer_reward_released = protocol_reward.answer_reward_released
    reward_semantics = "outcome_only"
    director_reward = 0.0
    task_reward_breakdown: dict[str, Any] = {"source": "pending_eligibility"}
    graph = MultiAgentGraph.from_dict(run.graph)
    canvas_events = [
        event for event in result.solver_result.trace.events if event.kind == "canvas_step"
    ]
    cross_agent_read_audit = (
        audit_cross_agent_read_overlap(canvas_events, run.graph)
        if is_swe_task
        else {
            "schema_version": 1,
            "policy": "record_only",
            "pair_count": 0,
            "flagged_pair_count": 0,
            "duplicate_read_only_exploration": False,
            "pairs": [],
        }
    )
    execution_payloads = [
        execution
        for event in canvas_events
        if isinstance((execution := event.payload.get("execution")), dict)
        and (
            bool(execution.get("execution_events"))
            or bool(execution.get("executed_agents"))
            or bool(execution.get("reused_agents"))
        )
    ]
    bidirectional_revision_decisions = [
        dict(decision)
        for payload in execution_payloads
        for decision in payload.get("revision_decisions", ())
        if isinstance(decision, dict)
    ]
    bidirectional_revision_reason_codes = sorted(
        {
            str(reason)
            for decision in bidirectional_revision_decisions
            for reason in decision.get("reason_codes", ())
            if str(reason)
        }
    )
    prompt_revision_payloads = [
        dict(event.payload["prompt_revision"])
        for event in canvas_events
        if isinstance(event.payload.get("prompt_revision"), dict)
        and event.payload["prompt_revision"]
    ]
    prompt_revision_bases = sorted(
        {str(payload.get("basis", "")) for payload in prompt_revision_payloads} - {""}
    )
    responsibility_overlap_checks = [
        dict(event.payload["responsibility_overlap_check"])
        for event in canvas_events
        if isinstance(event.payload.get("responsibility_overlap_check"), dict)
        and event.payload["responsibility_overlap_check"]
    ]
    duplicate_responsibility_detection_count = sum(
        (
            int(
                any(
                    (
                        bool(comparison.get("high_confidence_duplicate"))
                        for comparison in check.get("comparisons", ())
                        if isinstance(comparison, dict)
                    )
                )
            )
            for check in responsibility_overlap_checks
        )
    )
    duplicate_responsibility_decision_counts = {
        decision: sum(
            (
                int(str(check.get("decision", "")) == decision)
                for check in responsibility_overlap_checks
            )
        )
        for decision in ("record_only", "rewrite_requested", "accepted_after_warning", "rejected")
    }
    executed_agent_count = sum(
        (len(payload.get("executed_agents", ())) for payload in execution_payloads)
    )
    reused_agent_count = sum(
        (len(payload.get("reused_agents", ())) for payload in execution_payloads)
    )
    execution_agent_count = executed_agent_count + reused_agent_count
    worker_model_call_count = sum(
        (
            int(payload.get("worker_model_calls_total", len(payload.get("executed_agents", ()))))
            for payload in execution_payloads
        )
    )
    worker_initial_model_call_count = sum(
        (
            int(payload.get("initial_model_calls", len(payload.get("executed_agents", ()))))
            for payload in execution_payloads
        )
    )
    worker_revision_model_call_count = sum(
        (int(payload.get("revision_model_calls", 0)) for payload in execution_payloads)
    )
    worker_cache_hit_count = sum(
        (int(payload.get("cache_hits", 0)) for payload in execution_payloads)
    )
    worker_component_execution_count = sum(
        (int(payload.get("component_execution_count", 0)) for payload in execution_payloads)
    )
    worker_scheduled_agent_count = sum(
        (len(payload.get("scheduled_agents", ())) for payload in execution_payloads)
    )
    worker_attempt_count = worker_model_call_count + worker_cache_hit_count
    repair_payloads = [
        event.payload.get("structural_repair")
        for event in canvas_events
        if isinstance(event.payload.get("structural_repair"), dict)
    ]
    audit_payloads = [
        event.payload.get("topology_audit")
        for event in canvas_events
        if isinstance(event.payload.get("topology_audit"), dict)
    ]
    final_graph_errors = graph.validate(final=True)
    bounded_director_terminal = next(
        (
            str(event.payload.get("rejection_code"))
            for event in reversed(canvas_events)
            if event.payload.get("rejection_code")
            in {"director_no_legal_continuation", "director_no_progress_exhausted"}
        ),
        None,
    )
    terminal_execution_codes = sorted(
        {
            str(event.payload.get("rejection_code"))
            for event in canvas_events
            if event.payload.get("rejection_code")
            in {
                "execution_budget_exceeded",
                "worker_backend_unavailable",
                "execution_failure",
                "webshop_output_closure_incomplete",
            }
        }
    )
    training_exclusion_reasons: list[str] = []
    if not run.turns or any((not turn.trainable for turn in run.turns)):
        training_exclusion_reasons.append("director_policy_call_ineligible")
    if policy_encoding_error:
        training_exclusion_reasons.append("director_policy_call_ineligible")
    if any(
        (
            turn.token_provenance != "mock_text"
            and (
                not turn.prompt_token_ids
                or not turn.completion_token_ids
                or len(turn.behavior_log_probs) != len(turn.completion_token_ids)
                or any((not math.isfinite(value) or value > 0 for value in turn.behavior_log_probs))
            )
            for turn in run.turns
        )
    ):
        training_exclusion_reasons.append("director_policy_call_ineligible")
    if not run.finished:
        training_exclusion_reasons.append("not_finished")
    if final_graph_errors:
        training_exclusion_reasons.append("invalid_final_graph")
    if not protocol_reward.execution_complete:
        training_exclusion_reasons.append("execution_incomplete")
    if worker_backend_failure:
        training_exclusion_reasons.append("worker_backend_failure")
    if worker_artifact_integrity_failure:
        training_exclusion_reasons.append("worker_artifact_integrity_failure")
    if (
        is_swe_task
        and isinstance(swe_output_progress, dict)
        and swe_output_progress.get("commit_required")
        and (not swe_output_progress.get("commit_ready"))
    ):
        training_exclusion_reasons.append("swe_output_commit_incomplete")
    if swe_infrastructure_failure:
        training_exclusion_reasons.append("swe_infrastructure_failure")
    if swe_synthetic_evaluation:
        training_exclusion_reasons.append("swe_synthetic_evaluation")
    if swe_non_train_split:
        training_exclusion_reasons.append("swe_non_train_split")
    training_exclusion_reasons.extend(terminal_execution_codes)
    if bounded_director_terminal:
        training_exclusion_reasons.append(bounded_director_terminal)
    training_exclusion_reasons = list(dict.fromkeys(training_exclusion_reasons))
    policy_data_exclusions = [
        reason
        for reason in training_exclusion_reasons
        if reason == "director_policy_call_ineligible"
    ]
    typed_policy_failure = _runtime_owned_model_policy_failure(
        str(result.task.metadata.get("dataset", "")),
        training_exclusion_reasons=[
            reason for reason in training_exclusion_reasons if reason not in policy_data_exclusions
        ],
        verification=verification,
        worker_backend_failure=worker_backend_failure,
        worker_artifact_integrity=worker_artifact_integrity,
        worker_artifact_integrity_failure=worker_artifact_integrity_failure,
        swe_output_progress=swe_output_progress,
        alfworld_output_progress=alfworld_output_progress,
        webshop_output_progress=webshop_output_progress,
        swe_environment_result=swe_environment_result,
        swe_infrastructure_failure=swe_infrastructure_failure,
        swe_synthetic_evaluation=swe_synthetic_evaluation,
        swe_non_train_split=swe_non_train_split,
        stateful_environment_result=stateful_environment_result,
        admit_stateful_policy_failure_terminal=bool(
            result.task.metadata.get("admit_stateful_policy_failure_terminal", False)
        ),
    )
    from .outcome_admission import model_tool_errors, trusted_environment_outcome

    environment = swe_environment_result if is_swe_task else stateful_environment_result
    terminal = result.task.metadata.get("runtime_terminal_policy_failure")
    runtime_terminal_score = bool(
        verification and verification.verifier == "runtime_policy_terminal"
    )
    unresolved_tool_failure = any(
        (
            bool(
                set(artifact.get("runtime_tool_evidence", {}).get("failure_codes", ()))
                - model_tool_errors(dataset_key)
            )
            and (not artifact.get("runtime_tool_evidence", {}).get("recovered_failure"))
            for artifact in worker_artifact_integrity.values()
            if isinstance(artifact, dict)
        )
    )
    if unresolved_tool_failure:
        typed_policy_failure = None
        training_exclusion_reasons.append("tool_failure_attribution_unresolved")
    trusted_result = bool(
        verification is not None
        and (not runtime_terminal_score)
        and (not swe_infrastructure_failure)
        and (
            trusted_environment_outcome(dataset_key, environment)
            or (not worker_backend_failure and (not unresolved_tool_failure))
        )
        and (
            trusted_environment_outcome(dataset_key, environment)
            or (
                dataset_key in {"aime", "nq_open", "hotpotqa", "healthbench_professional"}
                and (submission.valid if submission else bool(run.output.strip()))
                and (not worker_artifact_integrity_failure)
                and (typed_policy_failure is None)
            )
            or (
                run.finished
                and protocol_reward.execution_complete
                and (not worker_artifact_integrity_failure)
                and (
                    not (
                        is_swe_task
                        and swe_output_progress.get("commit_required")
                        and (not swe_output_progress.get("commit_ready"))
                    )
                )
            )
        )
    )
    clearable_terminal_reasons = {
        "not_finished",
        "invalid_final_graph",
        "execution_incomplete",
        "worker_artifact_integrity_failure",
        "swe_output_commit_incomplete",
        "webshop_output_closure_incomplete",
        "execution_budget_exceeded",
    }
    if (
        is_swe_task
        and isinstance(environment, dict)
        and (environment.get("status") == "typed_policy_failure")
        and (environment.get("runtime_owned") is True)
        and (environment.get("attribution") == "model_policy")
        and (environment.get("environment_completed") is True)
        and (verification is not None)
        and (float(verification.score) == 0.0)
        and (not worker_backend_failure)
        and (not swe_infrastructure_failure)
        and (not unresolved_tool_failure)
        and (not swe_synthetic_evaluation)
        and (
            set(training_exclusion_reasons)
            <= clearable_terminal_reasons
            | {"director_policy_call_ineligible", "swe_non_train_split"}
        )
    ):
        typed_policy_failure = {
            "status": "typed_policy_failure",
            "dataset": dataset_key,
            "code": str(environment.get("failure_code", "read_only_policy_stall")),
            "attribution": "model_policy",
            "source": "runtime_owned_swe_action_ledger",
            "original_training_exclusion_reasons": list(training_exclusion_reasons),
        }
        trusted_result = False
    if trusted_result and (
        not (
            dataset_key == "alfworld"
            and typed_policy_failure
            and (float(verification.score) == 0.0)
        )
    ):
        typed_policy_failure = None
        training_exclusion_reasons = [
            reason
            for reason in training_exclusion_reasons
            if reason not in clearable_terminal_reasons
        ]
    elif (
        isinstance(terminal, dict)
        and terminal.get("source") == "runtime_terminal_ledger_v1"
        and (not worker_backend_failure)
        and (not swe_infrastructure_failure)
        and (not unresolved_tool_failure)
        and (
            set(training_exclusion_reasons)
            <= clearable_terminal_reasons
            | {"director_policy_call_ineligible", "swe_non_train_split"}
        )
        and (runtime_terminal_score or not (submission.valid if submission else run.output.strip()))
    ):
        typed_policy_failure = terminal
    if typed_policy_failure is not None:
        training_exclusion_reasons = list(policy_data_exclusions)
        if swe_non_train_split:
            training_exclusion_reasons.append("swe_non_train_split")
        director_reward = 0.0
    if (
        verification is None
        and typed_policy_failure is None
        and (
            dataset_key
            in {
                "aime",
                "nq_open",
                "hotpotqa",
                "healthbench_professional",
                "alfworld",
                "webshop",
                "swe_bench",
            }
        )
    ):
        training_exclusion_reasons.append("missing_trusted_verification")
    elif (
        verification is not None
        and (not trusted_result)
        and (typed_policy_failure is None)
        and (not training_exclusion_reasons)
    ):
        training_exclusion_reasons.append("outcome_attribution_unresolved")
    training_eligible = not training_exclusion_reasons
    if typed_policy_failure is not None:
        task_reward_breakdown = {
            "source": "typed_model_policy_failure",
            "failure": typed_policy_failure,
        }
    elif trusted_result:
        (director_reward, task_reward_breakdown) = _outcome_task_reward(
            str(result.task.metadata.get("dataset", "")), verification, prediction=submitted_output
        )
    else:
        task_reward_breakdown = {
            "source": "training_ineligible",
            "reasons": list(training_exclusion_reasons),
        }
    last_safe_graph: dict[str, Any] = {}
    for event in canvas_events:
        payload = event.payload
        raw_graph = payload.get("graph")
        audit = payload.get("topology_audit")
        repair = payload.get("structural_repair")
        if (
            not payload.get("accepted")
            or not isinstance(raw_graph, dict)
            or (not isinstance(audit, dict))
            or (not isinstance(repair, dict))
            or repair.get("required")
            or (int(audit.get("agent_count", 0)) <= 0)
            or (int(audit.get("configured_agent_count", 0)) != int(audit.get("agent_count", 0)))
            or (int(audit.get("weak_component_count", 0)) > 1)
        ):
            continue
        last_safe_graph = dict(raw_graph)

    def final_repair_counter(name: str) -> int:
        return max((int(payload.get(name, 0) or 0) for payload in repair_payloads), default=0)

    return SolverRollout(
        TokenizedDirectorTrajectory(
            rollout_id=f"{result.task.task_id}-r{rollout_index}",
            task_id=result.task.task_id,
            token_ids=token_ids,
            action_mask=mask,
            reward=director_reward,
            graph=run.graph,
            seed=seed,
            executor_version="adaptive-v1",
            metadata={
                "run_id": result.run_id,
                "finished": run.finished,
                "interactive_turns": len(run.turns),
                "accepted_turns": sum((int(turn.accepted) for turn in run.turns)),
                "rejected_turns": sum((int(not turn.accepted) for turn in run.turns)),
                "responsibility_violation_count": sum(
                    (int(turn.rejection_code == "responsibility_violation") for turn in run.turns)
                ),
                "duplicate_responsibility_rejection_count": sum(
                    (int(turn.rejection_code == "duplicate_responsibility") for turn in run.turns)
                ),
                "duplicate_responsibility_detection_count": duplicate_responsibility_detection_count,
                "duplicate_responsibility_decision_counts": duplicate_responsibility_decision_counts,
                "duplicate_responsibility_policy": next(
                    (
                        str(check.get("policy"))
                        for check in reversed(responsibility_overlap_checks)
                        if check.get("policy")
                    ),
                    "",
                ),
                "cross_agent_exact_read_overlap": cross_agent_read_audit,
                "duplicate_read_only_exploration": bool(
                    cross_agent_read_audit["duplicate_read_only_exploration"]
                ),
                "delegation_field_repair_count": sum(
                    (
                        len(event.payload.get("delegation_field_repairs", ()))
                        for event in canvas_events
                    )
                ),
                "managed_delegation_contract_count": sum(
                    (
                        int(
                            bool(
                                node.get("metadata", {})
                                .get("system_managed_contract", {})
                                .get("version")
                            )
                        )
                        for node in run.graph.get("nodes", ())
                    )
                ),
                "director_action_diagnostics": [
                    dict(turn.action_diagnostics) for turn in run.turns
                ],
                "postsolve_deadline_exceeded": result.task.metadata.get(
                    "postsolve_deadline_exceeded"
                ),
                "director_action_repairs": sum(
                    (
                        int(bool(turn.action_diagnostics.get("repair_attempted")))
                        for turn in run.turns
                    )
                ),
                "director_action_repair_successes": sum(
                    (
                        int(bool(turn.action_diagnostics.get("repair_succeeded")))
                        for turn in run.turns
                    )
                ),
                "director_discarded_output_chars": sum(
                    (
                        int(turn.action_diagnostics.get("discarded_output_chars", 0) or 0)
                        for turn in run.turns
                    )
                ),
                "protocol_recovery_count": sum(
                    (
                        int(bool(step.payload.get("protocol_recovery")))
                        for step in result.solver_result.trace.events
                        if step.kind == "canvas_step"
                    )
                ),
                "verification": asdict(verification) if verification else None,
                "answer_score": answer_score,
                "evaluation_score": answer_score,
                "flowsteer_structure": structure.to_dict(),
                "answer_reward_released": answer_reward_released,
                "director_reward": director_reward,
                "base_director_reward": director_reward,
                "task_reward": director_reward,
                "task_reward_breakdown": task_reward_breakdown,
                "reward_semantics": reward_semantics,
                "director_reward_version": OUTCOME_ONLY_REWARD_VERSION,
                "protocol_audit_version": protocol_reward.version,
                "protocol_reward_diagnostic": protocol_reward.to_dict(),
                "protocol_reward": protocol_reward.to_dict(),
                "protocol_score": protocol_reward.protocol_score,
                "protocol_qualified": protocol_reward.qualified,
                "environment_commit_execution_complete": environment_commit_execution_complete,
                "delegation_fidelity": protocol_reward.delegation_fidelity,
                "delegation_issues": list(protocol_reward.delegation_issues),
                "action_protocol": "director_model_v1",
                "environment_request_events": result.task.metadata.get(
                    "environment_request_events", []
                ),
                "deadline_accounting": result.task.metadata.get("deadline_accounting", {}),
                "model_roles": result.task.metadata.get("model_roles", {}),
                "canvas_prefixes": prefixes,
                "action_token_spans": action_spans,
                "relation_choice_token_spans": tuple(relation_choice_spans),
                "trajectory_schema": "director_trajectory_v2_raw_policy_calls",
                "director_sampling_seed": seed,
                "primary_executor_seed": seed,
                "executor_bundle_signature": result.solver_result.trace.task.metadata.get(
                    "executor_bundle_signature"
                ),
                "solver_answer": run.output,
                "raw_solver_answer": run.output,
                "submitted_answer": submission.submitted_answer if submission else run.output,
                "answer_submission": submission.to_dict() if submission else None,
                "qa_token_f1": result.task.metadata.get("qa_token_f1"),
                "qa_official_metrics": result.task.metadata.get("qa_official_metrics"),
                "environment_result_metrics": {
                    key: stateful_environment_result[key]
                    for key in (
                        "purchased",
                        "purchase_executed",
                        "purchase_committed",
                        "steps",
                        "success",
                        "terminal",
                        "reward",
                    )
                    if key in stateful_environment_result
                    and isinstance(stateful_environment_result[key], (bool, int, float))
                },
                "solver_trace": result.solver_result.trace.to_dict(),
                "backend_request_events": result.task.metadata.get("backend_request_events", []),
                "judge_request_events": result.task.metadata.get("judge_request_events", []),
                "director_request_events": [
                    event
                    for turn in run.turns
                    for event in turn.action_diagnostics.get("backend_request_events", [])
                ],
                "worker_backend_failure": worker_backend_failure,
                "worker_artifact_integrity": worker_artifact_integrity,
                "worker_output_integrity_risks": worker_output_integrity_risks,
                "worker_artifact_integrity_failure": worker_artifact_integrity_failure,
                "swe_output_progress": swe_output_progress,
                "alfworld_output_progress": alfworld_output_progress,
                "webshop_output_progress": webshop_output_progress,
                "swe_infrastructure_failure": swe_infrastructure_failure,
                "swe_synthetic_evaluation": swe_synthetic_evaluation,
                "swe_non_train_split": swe_non_train_split,
                "training_eligible": training_eligible,
                "reward_known": trusted_result or typed_policy_failure is not None,
                "task_outcome_passed": bool(trusted_result and verification.passed),
                "reward_admission_reason": "trusted_task_result"
                if trusted_result
                else "explicit_policy_terminal"
                if typed_policy_failure is not None
                else "infrastructure_or_unresolved_attribution",
                "policy_data_exclusion_reasons": policy_data_exclusions,
                "policy_encoding_error": policy_encoding_error,
                "runtime_terminal_policy_failure": terminal,
                "bounded_director_terminal": bounded_director_terminal,
                "training_exclusion_reasons": training_exclusion_reasons,
                "typed_policy_failure": typed_policy_failure,
                "terminal_status": "typed_policy_failure"
                if typed_policy_failure is not None
                else "completed"
                if training_eligible
                else "excluded",
                "terminal_graph_status": "typed_policy_failure"
                if typed_policy_failure is not None
                else "unfinished_with_trusted_result"
                if trusted_result and (not run.finished)
                else "valid_finished"
                if training_eligible
                else "unsafe_partial",
                "final_graph_validation_errors": final_graph_errors,
                "last_safe_graph": last_safe_graph,
                "failure_mode": "worker_backend_failure"
                if worker_backend_failure
                else "typed_policy_failure"
                if typed_policy_failure is not None
                else "worker_artifact_integrity_failure"
                if worker_artifact_integrity_failure
                else None
                if director_reward > 0.0
                else "director_protocol_failure"
                if not protocol_reward.qualified
                else "task_verification_failure",
                "duration_s": duration_s,
                "token_in": int(result_payload.get("token_in", 0)),
                "token_out": int(result_payload.get("token_out", 0)),
                "final_execution_round": next(
                    (
                        int(event.payload.get("director_turn_index", 0)) + 1
                        for event in result.solver_result.trace.events
                        if event.kind == "canvas_step"
                        and event.payload.get("final_execution")
                        and (event.payload.get("director_turn_index") is not None)
                    ),
                    None,
                ),
                "incremental_execution_count": sum(
                    (
                        int(
                            event.payload.get("execution") is not None
                            and (not bool(event.payload.get("final_execution")))
                        )
                        for event in result.solver_result.trace.events
                        if event.kind == "canvas_step"
                    )
                ),
                "worker_executed_agent_count": executed_agent_count,
                "worker_reused_agent_count": reused_agent_count,
                "worker_scheduled_agent_count": worker_scheduled_agent_count,
                "worker_model_call_count": worker_model_call_count,
                "worker_initial_model_call_count": worker_initial_model_call_count,
                "worker_revision_model_call_count": worker_revision_model_call_count,
                "worker_cache_hit_count": worker_cache_hit_count,
                "worker_initial_cache_hit_count": sum(
                    (int(payload.get("initial_cache_hits", 0)) for payload in execution_payloads)
                ),
                "worker_revision_cache_hit_count": sum(
                    (int(payload.get("revision_cache_hits", 0)) for payload in execution_payloads)
                ),
                "worker_mandatory_revision_call_count": sum(
                    (
                        int(payload.get("mandatory_revision_calls", 0))
                        for payload in execution_payloads
                    )
                ),
                "worker_incomplete_bidirectional_components": [
                    dict(item)
                    for payload in execution_payloads
                    for item in payload.get("incomplete_bidirectional_components", ())
                    if isinstance(item, dict)
                ],
                "worker_component_execution_count": worker_component_execution_count,
                "worker_bidirectional_revision_gate_count": len(bidirectional_revision_decisions),
                "worker_bidirectional_revision_required_count": sum(
                    (
                        int(bool(decision.get("revision_required")))
                        for decision in bidirectional_revision_decisions
                    )
                ),
                "worker_bidirectional_revision_skipped_agent_count": sum(
                    (
                        len(payload.get("revision_skipped_agents", ()))
                        for payload in execution_payloads
                    )
                ),
                "worker_bidirectional_revision_wave_count": sum(
                    (int(payload.get("revision_wave_count", 0)) for payload in execution_payloads)
                ),
                "worker_bidirectional_revision_reason_counts": {
                    reason: sum(
                        (
                            int(reason in decision.get("reason_codes", ()))
                            for decision in bidirectional_revision_decisions
                        )
                    )
                    for reason in bidirectional_revision_reason_codes
                },
                "worker_bidirectional_revision_decisions": bidirectional_revision_decisions,
                "worker_attempt_cache_hit_rate": worker_cache_hit_count / worker_attempt_count
                if worker_attempt_count
                else 0.0,
                "prompt_revision_count": len(prompt_revision_payloads),
                "prompt_revision_basis_counts": {
                    basis: sum(
                        (
                            int(str(payload.get("basis", "")) == basis)
                            for payload in prompt_revision_payloads
                        )
                    )
                    for basis in prompt_revision_bases
                },
                "prompt_revision_target_model_call_count": sum(
                    (
                        int(payload.get("target_model_calls", 0))
                        for payload in prompt_revision_payloads
                    )
                ),
                "prompt_revision_worker_model_call_count": sum(
                    (
                        int(payload.get("worker_model_calls_total", 0))
                        for payload in prompt_revision_payloads
                    )
                ),
                "prompt_revision_evidence_rejection_count": sum(
                    (
                        int(
                            event.payload.get("rejection_code")
                            == "prompt_revision_evidence_required"
                        )
                        for event in canvas_events
                    )
                ),
                "prompt_revision_events": prompt_revision_payloads,
                "worker_cache_reuse_rate": reused_agent_count / execution_agent_count
                if execution_agent_count
                else 0.0,
                "feedback_truncation_count": sum(
                    (
                        int("[feedback truncated]" in str(event.payload.get("feedback", "")))
                        for event in canvas_events
                    )
                ),
                "structural_repair_entry_count": final_repair_counter("entries_total"),
                "structural_repair_resolution_count": final_repair_counter("resolutions_total"),
                "structural_repair_blocked_action_count": final_repair_counter(
                    "blocked_actions_total"
                ),
                "semantic_no_progress_recovery_count": final_repair_counter(
                    "semantic_no_progress_recoveries_total"
                ),
                "output_switch_without_progress_count": final_repair_counter(
                    "output_switches_without_progress_total"
                ),
                "output_lifecycle_recovery_count": final_repair_counter(
                    "output_lifecycle_recoveries_total"
                ),
                "finish_reachability_rejection_count": final_repair_counter(
                    "finish_reachability_rejections_total"
                ),
                "relation_layer_rejection_count": final_repair_counter(
                    "relation_layer_rejections_total"
                ),
                "duplicate_agent_rejection_count": final_repair_counter(
                    "duplicate_agent_rejections_total"
                ),
                "disconnected_multi_agent_step_count": sum(
                    (
                        int(bool(payload.get("disconnected_multi_agent")))
                        for payload in audit_payloads
                    )
                ),
                "token_admission_rejection_count": sum(
                    (
                        int(bool(event.payload.get("token_admission", {}).get("blocked")))
                        for event in canvas_events
                        if isinstance(event.payload.get("token_admission"), dict)
                    )
                ),
                "token_admission_events": [
                    dict(event.payload["token_admission"])
                    for event in canvas_events
                    if isinstance(event.payload.get("token_admission"), dict)
                    and event.payload["token_admission"].get("blocked")
                ],
                "final_execution_count": sum(
                    (
                        int(bool(event.payload.get("final_execution")))
                        for event in result.solver_result.trace.events
                        if event.kind == "canvas_step"
                    )
                ),
            },
            policy_calls=policy_calls,
        ),
        graph,
    )


@dataclass(frozen=True)
class SelfPlayRunConfig:
    rollouts_per_task: int = 5
    base_seed: int = 0
    max_tokens: int = 4096
    counterfactuals_per_rollout: int = 1
    workers: int = 1
    proposals_per_seed: int = 1
    task_window: int = 1
    require_all_proposals: bool = False
    task_scheduling_policy: str = "logical_windows"
    max_active_task_groups: int = 8
    graph_diversity_bonus: float = 0.0
    structural_exploration_policy: str = "off"
    collapse_single_agent_threshold: float = 0.8
    collapse_unique_graph_threshold: float = 0.4
    collapse_task_relation_threshold: float = 0.4
    collapse_disconnected_multi_agent_threshold: float = 0.0
    collapse_patience_windows: int = 2
    initial_collapse_alert_streak: int = 0
    rollout_wall_time_s: float = 900.0
    stateful_rollout_wall_time_s: float = 900.0
    swe_rollout_wall_time_s: float = 900.0
    rollout_slot_wall_time_s: float = 900.0
    stateful_slot_wall_time_s: float = 900.0
    swe_slot_wall_time_s: float = 900.0
    rollout_no_progress_time_s: float = 120.0
    request_wall_time_s: float = 180.0
    aime_request_wall_time_s: float = 240.0
    grok_aime_request_wall_time_s: float = 300.0
    replacement_rollouts_per_task: int = 2
    swe_non_trainable_recovery_attempts: int = 1
    non_swe_recovery_attempts: int = 1
    allow_legacy_whole_rollout_recovery: bool = False
    backend_failure_retry_attempts: int = 0
    backend_failure_route_threshold: int = 3
    backend_failure_total_threshold: int = 3
    continue_after_backend_failure_circuit: bool = False
    scoped_route_circuit_enabled: bool = True
    scoped_route_unhealthy_threshold: int = 2
    scoped_route_slow_request_s: float = 180.0
    allow_exact_rollout_resume: bool = False
    allow_attribution_classifier_repair_resume: bool = False
    allow_infrastructure_repair_resume: bool = False
    allow_uncertain_group_recollection_resume: bool = False
    policy_sampling_attempt_offsets: dict[str, int] = field(default_factory=dict)
    route_health_path: Path | None = None
    route_health_cooldown_s: float = 3600.0
    worker_runtime_routes: tuple[str, ...] = ()
    rollout_group_policy: str = "complete"
    minimum_complete_task_groups: int = 1
    proposer_learning_mode: str = "legacy"
    proposer_baseline_mode: str = "ema"
    proposer_baseline_decay: float = 0.9
    proposer_baseline_path: Path | None = None
    proposer_baseline_cycle: int = 0
    require_all_planned_task_groups_for_training: bool = False
    continue_on_uncertain_attribution_exhausted: bool = False
    uncertain_attribution_zero_reward: bool = False
    pipeline_counterfactuals: bool = False
    counterfactual_workers: int = 2
    counterfactual_pair_wall_time_s: float = 900.0
    task_execution_window: int | None = None
    mace_peer_state_path: Path | None = None
    mace_model_state_path: Path | None = None
    frontier_reverify_fraction: float = 0.25
    pipeline_frontier_by_dataset: bool = False
    canary_exclude_migrated_frontier: bool = False
    evaluation_only: bool = False
    primary_job_order: str = PRIMARY_JOB_ORDER_ROUND_ROBIN


class SelfPlayRolloutRunner:
    """SESA ordering: propose tasks, collect K Solver graphs, stop at batches."""

    def __init__(
        self,
        *,
        proposer: TaskProposer,
        application_factory: Any,
        tokenizer: Tokenizer,
        snapshots: AlternatingSnapshots,
        output_dir: str | Path,
        config: SelfPlayRunConfig | None = None,
        rollout_pool: RolloutPool[Any, Any] | None = None,
        graph_feature_extractor: Any | None = None,
        primary_probability_observer: Any | None = None,
    ) -> None:
        self.proposer = proposer
        self.application_factory = application_factory
        self.tokenizer = tokenizer
        self.snapshots = snapshots
        self.output_dir = Path(output_dir)
        self.config = config or SelfPlayRunConfig()
        if (
            self.config.mace_peer_state_path is not None
            or self.config.mace_model_state_path is not None
        ):
            raise ValueError("MACE state paths are retired; start a new Director SET_MODEL run")
        self.rollout_pool = rollout_pool or ThreadRolloutPool(self.config.workers)
        self.graph_feature_extractor = graph_feature_extractor
        self.primary_probability_observer = primary_probability_observer
        if self.config.rollouts_per_task < 1 or (
            self.config.rollouts_per_task < 2 and (not self.config.evaluation_only)
        ):
            raise ValueError("rollouts_per_task must be at least two")
        if self.config.proposals_per_seed <= 0:
            raise ValueError("proposals_per_seed must be positive")
        if self.config.task_window <= 0:
            raise ValueError("task_window must be positive")
        if self.config.task_execution_window is not None and self.config.task_execution_window <= 0:
            raise ValueError("task_execution_window must be positive")
        if not 0.0 <= self.config.frontier_reverify_fraction <= 1.0:
            raise ValueError("frontier_reverify_fraction must be in [0, 1]")
        if self.config.task_scheduling_policy not in {"logical_windows", "frozen_manifest_dynamic"}:
            raise ValueError(
                "task_scheduling_policy must be 'logical_windows' or 'frozen_manifest_dynamic'"
            )
        if self.config.max_active_task_groups <= 0:
            raise ValueError("max_active_task_groups must be positive")
        if self.config.primary_job_order not in PRIMARY_JOB_ORDER_CHOICES:
            raise ValueError("primary_job_order must be 'round_robin' or 'long_tail_first'")
        if self.config.task_scheduling_policy == "frozen_manifest_dynamic" and (
            not self.config.require_all_proposals
        ):
            raise ValueError(
                "frozen_manifest_dynamic scheduling requires require_all_proposals=true"
            )
        if self.config.counterfactual_workers <= 0:
            raise ValueError("counterfactual_workers must be positive")
        if self.config.counterfactual_pair_wall_time_s <= 0:
            raise ValueError("counterfactual_pair_wall_time_s must be positive")
        if self.config.graph_diversity_bonus != 0.0:
            raise ValueError("outcome_only_v1 requires graph_diversity_bonus = 0")
        if not 0.0 <= self.config.collapse_task_relation_threshold <= 1.0:
            raise ValueError("collapse_task_relation_threshold must be in [0, 1]")
        if not 0.0 <= self.config.collapse_disconnected_multi_agent_threshold <= 1.0:
            raise ValueError("collapse_disconnected_multi_agent_threshold must be in [0, 1]")
        if self.config.collapse_patience_windows <= 0:
            raise ValueError("collapse_patience_windows must be positive")
        if self.config.initial_collapse_alert_streak < 0:
            raise ValueError("initial_collapse_alert_streak must be non-negative")
        if self.config.rollout_wall_time_s <= 0:
            raise ValueError("rollout_wall_time_s must be positive")
        if self.config.stateful_rollout_wall_time_s <= 0:
            raise ValueError("stateful_rollout_wall_time_s must be positive")
        if self.config.swe_rollout_wall_time_s <= 0:
            raise ValueError("swe_rollout_wall_time_s must be positive")
        if self.config.rollout_slot_wall_time_s < self.config.rollout_wall_time_s:
            raise ValueError("rollout_slot_wall_time_s must be at least rollout_wall_time_s")
        if self.config.stateful_slot_wall_time_s < self.config.stateful_rollout_wall_time_s:
            raise ValueError(
                "stateful_slot_wall_time_s must be at least stateful_rollout_wall_time_s"
            )
        if self.config.swe_slot_wall_time_s <= 0:
            raise ValueError("swe_slot_wall_time_s must be positive")
        if self.config.swe_slot_wall_time_s < self.config.swe_rollout_wall_time_s:
            raise ValueError("swe_slot_wall_time_s must be at least swe_rollout_wall_time_s")
        if self.config.rollout_no_progress_time_s <= 0:
            raise ValueError("rollout_no_progress_time_s must be positive")
        if self.config.request_wall_time_s <= 0:
            raise ValueError("request_wall_time_s must be positive")
        if self.config.aime_request_wall_time_s <= 0:
            raise ValueError("aime_request_wall_time_s must be positive")
        if self.config.grok_aime_request_wall_time_s <= 0:
            raise ValueError("grok_aime_request_wall_time_s must be positive")
        if self.config.replacement_rollouts_per_task < 0:
            raise ValueError("replacement_rollouts_per_task must be non-negative")
        if self.config.swe_non_trainable_recovery_attempts < 0:
            raise ValueError("swe_non_trainable_recovery_attempts must be non-negative")
        if self.config.non_swe_recovery_attempts < 0:
            raise ValueError("non_swe_recovery_attempts must be non-negative")
        if self.config.backend_failure_retry_attempts < 0:
            raise ValueError("backend_failure_retry_attempts must be non-negative")
        if self.config.backend_failure_route_threshold <= 0:
            raise ValueError("backend_failure_route_threshold must be positive")
        if self.config.backend_failure_total_threshold <= 0:
            raise ValueError("backend_failure_total_threshold must be positive")
        if self.config.scoped_route_unhealthy_threshold <= 0:
            raise ValueError("scoped_route_unhealthy_threshold must be positive")
        if self.config.scoped_route_slow_request_s <= 0:
            raise ValueError("scoped_route_slow_request_s must be positive")
        if self.config.route_health_cooldown_s <= 0:
            raise ValueError("route_health_cooldown_s must be positive")
        if self.config.rollout_group_policy not in {"complete", "eligible_subset"}:
            raise ValueError("unknown rollout_group_policy")
        if self.config.rollout_group_policy == "eligible_subset" and (
            self.config.require_all_planned_task_groups_for_training
            or self.config.minimum_complete_task_groups != 1
            or self.config.rollouts_per_task < 2
            or self.config.allow_uncertain_group_recollection_resume
        ):
            raise ValueError(
                "eligible_subset requires K>=2, minimum groups=1 and no strict/recollection gate"
            )
        if self.config.minimum_complete_task_groups <= 0:
            raise ValueError("minimum_complete_task_groups must be positive")
        if any(
            (
                not str(task_id) or not isinstance(offset, int) or offset < 0
                for (task_id, offset) in self.config.policy_sampling_attempt_offsets.items()
            )
        ):
            raise ValueError(
                "policy_sampling_attempt_offsets must map task IDs to non-negative integers"
            )
        if self.config.structural_exploration_policy not in {"off", "stratified"}:
            raise ValueError("structural_exploration_policy must be 'off' or 'stratified'")

    def run(self, seeds: Iterable[SeedInput], *, resume: bool = False) -> DryRunSelfPlayResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        persist_context_policy(self.output_dir, resume=resume)
        mode_path = self.output_dir / "proposer_learning.json"
        requested = (
            self.config.proposer_learning_mode
            if self.config.rollout_group_policy == "eligible_subset"
            and (not self.config.evaluation_only)
            else "legacy"
        )
        if requested not in {"legacy", INDEPENDENT_SCHEMA}:
            raise ValueError("unknown Proposer learning mode")
        mode = (
            json.loads(mode_path.read_text())["mode"]
            if resume and mode_path.exists()
            else "legacy"
            if resume and (self.output_dir / "tasks.jsonl").exists()
            else requested
        )
        self._independent_frontier = mode == INDEPENDENT_SCHEMA
        _write_json(mode_path, {"mode": mode})
        self._proposer_baseline = None
        self._baseline_store = None
        if self._independent_frontier:
            baseline_path = (
                self.config.proposer_baseline_path
                or (
                    self.output_dir.parent
                    if self.output_dir.name.startswith("cycle-")
                    else self.output_dir
                )
                / "proposer_baseline_state.json"
            )
            (self._proposer_baseline, self._baseline_store) = freeze_proposer_baseline(
                self.output_dir / "proposer_baseline_snapshot.json",
                state_path=baseline_path,
                cycle=self.config.proposer_baseline_cycle,
                decay=self.config.proposer_baseline_decay,
                mode=self.config.proposer_baseline_mode,
            )
        self._validate_action_protocol()
        seed_values = tuple(seeds)
        completed_primary_collection_resume = False
        completed_collection_resume = False
        subset_resume = None
        if resume:
            expected_task_ids = {
                f"task-{index}"
                if self.config.proposals_per_seed == 1
                else f"task-{index}-p{proposer_index}"
                for (index, _seed) in enumerate(seed_values, start=1)
                for proposer_index in range(self.config.proposals_per_seed)
            }
            expected_rollout_ids = {
                _rollout_id(task_id, rollout_index)
                for task_id in expected_task_ids
                for rollout_index in range(self.config.rollouts_per_task)
            }
            try:
                persisted_rows = _read_jsonl(self.output_dir / "solver_rollouts.jsonl")
                persisted_ids = [str(row.get("rollout_id", "")) for row in persisted_rows]
                batch_gate = json.loads(
                    (self.output_dir / "batch_gate.json").read_text(encoding="utf-8")
                )
                progress = json.loads(
                    (self.output_dir / "progress.json").read_text(encoding="utf-8")
                )
                completed_primary_collection_resume = bool(
                    expected_rollout_ids
                    and len(persisted_ids) == len(expected_rollout_ids)
                    and (len(set(persisted_ids)) == len(persisted_ids))
                    and (set(persisted_ids) == expected_rollout_ids)
                    and all(
                        (
                            row.get("metadata", {}).get("training_eligible") is True
                            and row.get("metadata", {}).get("terminal_graph_status")
                            != "unsafe_partial"
                            for row in persisted_rows
                        )
                    )
                    and (batch_gate.get("status") == "ready")
                    and (batch_gate.get("all_planned_task_groups_complete") is True)
                    and (
                        int(batch_gate.get("planned_task_group_count", -1))
                        == len(expected_task_ids)
                    )
                    and (
                        int(batch_gate.get("complete_task_group_count", -1))
                        == len(expected_task_ids)
                    )
                    and (not batch_gate.get("incomplete_task_ids"))
                    and (not batch_gate.get("quarantined_task_ids"))
                    and (int(batch_gate.get("training_excluded_rollout_count", -1)) == 0)
                    and (
                        int(batch_gate.get("rollouts_per_task", -1))
                        == self.config.rollouts_per_task
                    )
                    and (int(progress.get("completed_tasks", -1)) == len(expected_task_ids))
                    and (int(progress.get("rollouts", -1)) == len(expected_rollout_ids))
                    and (
                        int(progress.get("training_eligible_rollouts", -1))
                        == len(expected_rollout_ids)
                    )
                    and (int(progress.get("training_excluded_rollouts", -1)) == 0)
                )
                completed_collection_resume = bool(
                    completed_primary_collection_resume
                    and all(
                        (
                            (self.output_dir / name).is_file()
                            for name in (
                                "frontier_scores.json",
                                "proposer_batch.json",
                                "solver_batch.json",
                                "snapshots.json",
                            )
                        )
                    )
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                completed_primary_collection_resume = False
                completed_collection_resume = False
        selection_path = self.output_dir / "training_selection.json"
        if resume and selection_path.exists():
            from .async_cycle import recover_interrupted_lineage_binding

            recover_interrupted_lineage_binding(self.output_dir)
            saved_selection = json.loads(selection_path.read_text())
            if self.config.rollout_group_policy != "eligible_subset":
                raise CollectionInfrastructureIncidentError(
                    "resume cannot change the saved selection policy"
                )
            if saved_selection.get("postcollection_complete"):
                if (
                    saved_selection.get("schema_version") not in SELECTION_SCHEMAS
                    or {g["task_id"] for g in saved_selection["groups"]} != expected_task_ids
                    or any(
                        (
                            g["planned_rollout_count"] != self.config.rollouts_per_task
                            for g in saved_selection["groups"]
                        )
                    )
                ):
                    raise CollectionInfrastructureIncidentError(
                        "saved selection does not match requested tasks/K"
                    )
                for name, digest in {
                    **saved_selection["artifacts_sha256"],
                    **saved_selection.get("source_artifacts_sha256", {}),
                }.items():
                    if hashlib.sha256((self.output_dir / name).read_bytes()).hexdigest() != digest:
                        raise CollectionInfrastructureIncidentError(
                            "frozen training artifact changed: " + name
                        )
                (loaded_proposals, loaded_rollouts) = self._load()
                rebuilt = build_training_selection(
                    loaded_proposals,
                    {r.trajectory.rollout_id: r for r in loaded_rollouts},
                    self.config.rollouts_per_task,
                    schema=saved_selection["schema_version"],
                )
                if rebuilt["groups"] != saved_selection["groups"]:
                    raise CollectionInfrastructureIncidentError(
                        "source trajectories or admission changed after batch freeze"
                    )
                subset_resume = saved_selection
                completed_primary_collection_resume = True
                completed_collection_resume = True
        abort_path = self.output_dir / "collection_abort.json"
        if resume and abort_path.exists():
            abort = json.loads(abort_path.read_text(encoding="utf-8"))
            incident_class = str(abort.get("incident_class", ""))
            diagnostic_skip_resume = bool(
                self.config.continue_on_uncertain_attribution_exhausted
                and incident_class == "attribution_uncertain_recovery_exhausted"
            )
            infrastructure_repair_resume = bool(
                self.config.allow_infrastructure_repair_resume
                and incident_class == "infrastructure"
            )
            uncertain_group_recollection_resume = bool(
                self.config.allow_uncertain_group_recollection_resume
                and incident_class == "attribution_uncertain_recovery_exhausted"
            )
            if completed_collection_resume:
                _append_jsonl(
                    self.output_dir / "collection_incidents.jsonl",
                    {
                        "event": "completed_collection_reopened_for_training_only_resume",
                        "incident_class": incident_class,
                        "persisted_rollouts": len(expected_rollout_ids),
                        "preserved_abort": str(abort_path),
                    },
                )
            elif completed_primary_collection_resume:
                _append_jsonl(
                    self.output_dir / "collection_incidents.jsonl",
                    {
                        "event": "completed_primary_collection_reopened_for_postcollection_resume",
                        "incident_class": incident_class,
                        "persisted_rollouts": len(expected_rollout_ids),
                        "preserved_abort": str(abort_path),
                    },
                )
            elif diagnostic_skip_resume:
                _append_jsonl(
                    self.output_dir / "collection_incidents.jsonl",
                    {
                        "event": "collection_abort_reopened_for_diagnostic_group_skip",
                        "incident_class": incident_class,
                        "rollout_id": abort.get("rollout_id"),
                        "task_id": abort.get("task_id"),
                        "preserved_abort": str(abort_path),
                    },
                )
            elif infrastructure_repair_resume:
                _append_jsonl(
                    self.output_dir / "collection_incidents.jsonl",
                    {
                        "event": "collection_abort_reopened_after_infrastructure_repair",
                        "incident_class": incident_class,
                        "rollout_id": abort.get("rollout_id"),
                        "task_id": abort.get("task_id"),
                        "preserved_abort": str(abort_path),
                    },
                )
            elif uncertain_group_recollection_resume:
                _append_jsonl(
                    self.output_dir / "collection_incidents.jsonl",
                    {
                        "event": "collection_abort_reopened_for_full_group_recollection",
                        "incident_class": incident_class,
                        "rollout_id": abort.get("rollout_id"),
                        "task_id": abort.get("task_id"),
                        "preserved_abort": str(abort_path),
                    },
                )
            elif not self.config.allow_attribution_classifier_repair_resume:
                raise CollectionInfrastructureIncidentError(
                    "exact rollout resume is forbidden after a collection incident; start a fresh collection epoch from the last committed checkpoint; incident="
                    + str(abort.get("incident_class", "unknown"))
                )
            if (
                completed_collection_resume
                or completed_primary_collection_resume
                or diagnostic_skip_resume
                or infrastructure_repair_resume
                or uncertain_group_recollection_resume
            ):
                pass
            elif incident_class != "attribution_uncertain_recovery_exhausted":
                raise CollectionInfrastructureIncidentError(
                    "attribution-classifier repair resume is limited to attribution_uncertain_recovery_exhausted incidents; incident="
                    + str(abort.get("incident_class", "unknown"))
                )
            repair_path = self.output_dir / "collection_repair_attestation.json"
            if (
                not completed_collection_resume
                and (not completed_primary_collection_resume)
                and (not diagnostic_skip_resume)
                and (not infrastructure_repair_resume)
                and (not uncertain_group_recollection_resume)
                and (not repair_path.exists())
            ):
                raise CollectionInfrastructureIncidentError(
                    "attribution-classifier repair resume requires collection_repair_attestation.json"
                )
            if (
                not completed_collection_resume
                and (not completed_primary_collection_resume)
                and (not diagnostic_skip_resume)
                and (not infrastructure_repair_resume)
                and (not uncertain_group_recollection_resume)
            ):
                repair = json.loads(repair_path.read_text(encoding="utf-8"))
                if (
                    repair.get("status") != "approved"
                    or repair.get("incident_class") != "attribution_uncertain_recovery_exhausted"
                    or (not repair.get("reclassified_rollout_ids"))
                ):
                    raise CollectionInfrastructureIncidentError(
                        "invalid attribution-classifier repair attestation"
                    )
        if completed_collection_resume:
            from .training import load_training_batch

            (proposals, _rollouts) = self._load()
            all_resumed_tasks = tuple((p.task for p in proposals))
            proposer_batch = load_training_batch(self.output_dir / "proposer_batch.json")
            solver_batch = load_training_batch(self.output_dir / "solver_batch.json")
            frontier_rows = json.loads(
                (self.output_dir / "frontier_scores.json").read_text(encoding="utf-8")
            )
            frontier_scores = tuple(
                (
                    FrontierScore(
                        task_id=str(row["task_id"]),
                        validity=float(row["validity"]),
                        scalar=float(row["scalar"]),
                        graph_local=float(row["graph_local"]),
                        rewards=tuple((float(value) for value in row["rewards"])),
                        provisional_graph_local=float(row["provisional_graph_local"])
                        if row.get("provisional_graph_local") is not None
                        else None,
                        stable_graph_local=float(row["stable_graph_local"])
                        if row.get("stable_graph_local") is not None
                        else None,
                        reverify_status=str(row.get("reverify_status", "not_triggered")),
                        pair_stability=tuple(
                            (dict(value) for value in row.get("pair_stability", ()))
                        ),
                        metadata=dict(row.get("metadata", {})),
                    )
                    for row in frontier_rows
                )
            )
            if subset_resume is not None:
                if [s.rollout_id for s in solver_batch.samples] != subset_resume[
                    "selected_rollout_ids"
                ] or [s.task_id for s in proposer_batch.samples] != subset_resume[
                    "proposer_task_ids"
                ]:
                    raise CollectionInfrastructureIncidentError(
                        "saved batches differ from selection"
                    )
                expected_task_ids = set(subset_resume["solver_task_ids"])
                expected_rollout_ids = set(subset_resume["selected_rollout_ids"])
                proposals = [p for p in proposals if p.task.task_id in expected_task_ids]
            if (
                {proposal.task.task_id for proposal in proposals} != expected_task_ids
                or {sample.task_id for sample in proposer_batch.samples}
                != (set(subset_resume["proposer_task_ids"]) if subset_resume else expected_task_ids)
                or {sample.rollout_id for sample in solver_batch.samples} != expected_rollout_ids
                or (
                    {score.task_id for score in frontier_scores}
                    != (
                        {g["task_id"] for g in subset_resume["groups"]}
                        if subset_resume and subset_resume["schema_version"] == INDEPENDENT_SCHEMA
                        else expected_task_ids
                    )
                )
            ):
                raise CollectionInfrastructureIncidentError(
                    "durable post-collection artifacts do not match the requested task/K manifest"
                )
            if self.primary_probability_observer is not None:
                for sample in solver_batch.samples:
                    self.primary_probability_observer.submit(
                        TokenizedDirectorTrajectory(
                            rollout_id=sample.rollout_id,
                            task_id=sample.task_id,
                            token_ids=sample.token_ids,
                            action_mask=sample.action_mask,
                            reward=sample.reward,
                            graph={},
                            metadata=dict(sample.metadata),
                            policy_calls=sample.policy_calls,
                        )
                    )
                cache_summary = self.primary_probability_observer.finalize()
                _write_json(
                    self.output_dir / "solver_probability_cache_summary.json", cache_summary
                )
            _append_jsonl(
                self.output_dir / "pipeline_events.jsonl",
                {
                    "event": "completed_postcollection_artifacts_reused",
                    "task_groups": len(expected_task_ids),
                    "rollouts": len(expected_rollout_ids),
                    "frontier_reverification_reexecuted": False,
                    "timestamp": time.time(),
                },
            )
            if self._baseline_store is not None:
                self._baseline_store.commit(
                    self._proposer_baseline, proposer_batch.metadata["frontier_dataset_means"]
                )
            extraction_path = self.output_dir / "proposal_extraction.json"
            return DryRunSelfPlayResult(
                tasks=all_resumed_tasks,
                frontier_scores=frontier_scores,
                proposer_batch=proposer_batch,
                solver_batch=solver_batch,
                snapshots=json.loads(
                    (self.output_dir / "snapshots.json").read_text(encoding="utf-8")
                ),
                proposal_extraction=json.loads(extraction_path.read_text(encoding="utf-8"))
                if extraction_path.is_file()
                else {},
            )
        route_health_store = (
            RouteHealthStore(
                self.config.route_health_path, cooldown_s=self.config.route_health_cooldown_s
            )
            if self.config.route_health_path is not None
            else None
        )
        if route_health_store is not None and self.config.worker_runtime_routes:
            (available_routes, blocked_routes) = route_health_store.available_routes(
                self.config.worker_runtime_routes
            )
            if blocked_routes:
                _append_jsonl(
                    self.output_dir / "backend_api_events.jsonl",
                    {
                        "stage": "collection_preflight",
                        "event": "persistent_backend_routes_blocked",
                        "available_routes": list(available_routes),
                        "blocked_routes": blocked_routes,
                    },
                )
            if not available_routes:
                blocked_summary = ", ".join(
                    (
                        f"{route}({state['block_reason']})"
                        for (route, state) in sorted(blocked_routes.items())
                    )
                )
                raise PersistentRouteCircuitOpenError(
                    "all configured Worker routes are blocked by persisted health state: "
                    + blocked_summary
                )
        (proposals, rollouts) = self._load() if resume else ([], [])
        if resume:
            rehydrate = getattr(self.proposer, "rehydrate_resumed_proposal", None)
            if callable(rehydrate):
                proposals = [rehydrate(proposal) for proposal in proposals]
        terminal_failed_ids = (
            {
                str(row.get("rollout_id", ""))
                for row in _read_jsonl(self.output_dir / "rollout_errors.jsonl")
                if row.get("error_type") != "CancelledError"
            }
            if resume
            and self.config.rollout_group_policy == "eligible_subset"
            and (not self.config.allow_exact_rollout_resume)
            else set()
        )
        attempts = _read_jsonl(self.output_dir / "proposal_attempts.jsonl") if resume else []
        observations = (
            _read_jsonl(self.output_dir / "curriculum_observations.jsonl") if resume else []
        )
        collapse_windows = _read_jsonl(self.output_dir / "collapse_monitor.jsonl") if resume else []
        quarantined_groups = (
            _read_jsonl(self.output_dir / "quarantined_groups.jsonl") if resume else []
        )
        quarantine_status_by_task: dict[str, dict[str, Any]] = {}
        for item in quarantined_groups:
            task_id = str(item.get("task_id", ""))
            if task_id:
                quarantine_status_by_task[task_id] = item
        quarantined_task_ids = {
            task_id
            for (task_id, item) in quarantine_status_by_task.items()
            if item.get("status") == "quarantined"
            and (
                not (
                    self.config.rollout_group_policy == "eligible_subset"
                    and item.get("reason")
                    in {"non_trainable_rollout_group", "incomplete_rollout_group"}
                    and (not self.config.allow_exact_rollout_resume)
                )
            )
        }
        inherited_collapse_streak = (
            0 if collapse_windows else self.config.initial_collapse_alert_streak
        )
        if inherited_collapse_streak >= self.config.collapse_patience_windows:
            raise RuntimeError(
                f"structural collapse stop inherited from the previous cycle for {inherited_collapse_streak} consecutive windows"
            )
        last_collapse = collapse_windows[-1] if collapse_windows else None
        last_reasons = set(last_collapse.get("alert_reasons", ())) if last_collapse else set()
        legacy_only_stop_disabled = bool(
            last_collapse
            and self.config.structural_exploration_policy == "off"
            and (last_reasons == {"legacy_low_diversity"})
        )
        if legacy_only_stop_disabled:
            _append_jsonl(
                self.output_dir / "collapse_policy_events.jsonl",
                {
                    "event": "resume_legacy_stop_reclassified",
                    "structural_exploration_policy": "off",
                    "prior_window_key": last_collapse.get("window_key"),
                    "prior_alert_reasons": sorted(last_reasons),
                    "prior_consecutive_alert_windows": int(
                        last_collapse.get("consecutive_alert_windows", 0)
                    ),
                    "preserved_safety_gates": ["disconnected_multi_agent"],
                },
            )
        if (
            last_collapse
            and (not legacy_only_stop_disabled)
            and (
                int(last_collapse.get("consecutive_alert_windows", 0))
                >= self.config.collapse_patience_windows
            )
        ):
            raise RuntimeError(
                f"structural collapse stop remains active on resume: single-Agent rate {float(last_collapse.get('single_agent_rate', 0.0)):.3f}, within-task unique graph ratio {float(last_collapse.get('within_task_unique_graph_ratio', 0.0)):.3f} for {int(last_collapse.get('consecutive_alert_windows', 0))} consecutive windows"
            )
        observed_task_ids = {str(item["task_id"]) for item in observations}
        state_path = self.output_dir / "curriculum_state.json"
        load_state = getattr(self.proposer, "load_state_dict", None)
        if resume and state_path.exists() and callable(load_state):
            load_state(json.loads(state_path.read_text(encoding="utf-8")))
        if resume and proposals and (not attempts):
            attempts = [
                {
                    "task_id": proposal.task.task_id,
                    "success": True,
                    "failure_kind": None,
                    "resumed_legacy_record": True,
                }
                for proposal in proposals
            ]
        existing_proposals = {proposal.task.task_id: proposal for proposal in proposals}
        rollouts_by_id = {rollout.trajectory.rollout_id: rollout for rollout in rollouts}
        task_specs = [
            (
                index,
                seed_text,
                proposer_index,
                f"task-{index}"
                if self.config.proposals_per_seed == 1
                else f"task-{index}-p{proposer_index}",
            )
            for (index, seed_text) in enumerate(seed_values, start=1)
            for proposer_index in range(self.config.proposals_per_seed)
        ]
        if resume and self.config.allow_exact_rollout_resume:
            recoverable_reasons = {"backend_circuit_open", "incomplete_rollout_group"}
            planned_task_ids = {task_id for (*_prefix, task_id) in task_specs}
            for task_id in sorted(quarantined_task_ids & planned_task_ids):
                quarantine = quarantine_status_by_task[task_id]
                expected_ids = {
                    _rollout_id(task_id, rollout_index)
                    for rollout_index in range(self.config.rollouts_per_task)
                }
                missing_ids = sorted(expected_ids - rollouts_by_id.keys())
                evaluation_missing_slot_reopen = bool(self.config.evaluation_only and missing_ids)
                if not evaluation_missing_slot_reopen and (
                    quarantine.get("reason") not in recoverable_reasons
                    or not missing_ids
                    or quarantine.get("non_trainable_rollout_ids")
                ):
                    continue
                reopen_event = {
                    "task_id": task_id,
                    "status": "reopened_for_exact_resume",
                    "reason": "evaluation_missing_slot_recovery"
                    if evaluation_missing_slot_reopen
                    else "transient_backend_recovery",
                    "missing_rollout_ids": missing_ids,
                    "prior_quarantine_reason": quarantine.get("reason"),
                }
                quarantined_groups.append(reopen_event)
                quarantine_status_by_task[task_id] = reopen_event
                quarantined_task_ids.remove(task_id)
                terminal_failed_ids.difference_update(missing_ids)
                _append_jsonl(self.output_dir / "quarantined_groups.jsonl", reopen_event)
        if resume and (not self.config.allow_exact_rollout_resume):
            planned_task_ids = {task_id for (*_prefix, task_id) in task_specs}
            expected_persisted_ids = {
                _rollout_id(task_id, rollout_index)
                for task_id in existing_proposals
                if task_id in planned_task_ids and task_id not in quarantined_task_ids
                for rollout_index in range(self.config.rollouts_per_task)
            }
            missing_persisted_ids = sorted(
                expected_persisted_ids - rollouts_by_id.keys() - terminal_failed_ids
            )
            if missing_persisted_ids:
                raise RuntimeError(
                    "exact rollout resume is disabled; preserved incomplete collection without recollecting missing rollout ids: "
                    + ", ".join(missing_persisted_ids)
                )
        failed_task_ids: list[str] = []
        dynamic_counterfactual_executor = (
            ThreadPoolExecutor(
                max_workers=self.config.counterfactual_workers,
                thread_name_prefix="counterfactual-stream",
            )
            if self.config.pipeline_counterfactuals
            else None
        )
        dynamic_counterfactual_futures: list[tuple[Future[None], _PrimaryCollection]] = []
        counterfactual_cancellation = threading.Event()
        counterfactual_commit_lock = threading.Lock()
        counterfactual_status: dict[str, str] = {}
        counterfactual_credit_count = 0

        def submit_dynamic_counterfactual(
            primary: _PrimaryCollection, *, window_index: int
        ) -> None:
            assert dynamic_counterfactual_executor is not None
            rollout_id = primary.rollout.trajectory.rollout_id
            task_id = primary.proposal.task.task_id
            primary.counterfactual_cancellation_event = counterfactual_cancellation
            with counterfactual_commit_lock:
                counterfactual_status[rollout_id] = "queued"

            def collect_and_persist() -> None:
                nonlocal counterfactual_credit_count
                with counterfactual_commit_lock:
                    if counterfactual_cancellation.is_set():
                        primary.close()
                        return
                    counterfactual_status[rollout_id] = "running"
                _append_jsonl(
                    self.output_dir / "pipeline_events.jsonl",
                    {
                        "event": "counterfactual_rollout_started",
                        "task_id": task_id,
                        "rollout_id": rollout_id,
                        "timestamp": time.time(),
                        "priority": "counterfactual",
                        "scheduling": "after_primary_rollout",
                        "window_index": window_index,
                    },
                )
                try:
                    with request_priority("counterfactual"):
                        outcome = self._collect_counterfactual(primary)
                    result = (primary, outcome, None)
                except Exception as exc:
                    result = (primary, None, exc)
                with counterfactual_commit_lock:
                    if counterfactual_cancellation.is_set():
                        counterfactual_status[rollout_id] = "discarded_after_update_boundary"
                        primary.close()
                        return
                    self._persist_counterfactual_batch([result])
                    if result[1] is not None:
                        counterfactual_credit_count += len(result[1][1])
                    counterfactual_status[rollout_id] = (
                        "completed" if result[1] is not None else "failed"
                    )
                    _append_jsonl(
                        self.output_dir / "pipeline_events.jsonl",
                        {
                            "event": "counterfactual_rollout_completed",
                            "task_id": task_id,
                            "rollout_id": rollout_id,
                            "timestamp": time.time(),
                            "priority": "counterfactual",
                            "scheduling": "after_primary_rollout",
                            "window_index": window_index,
                        },
                    )

            future = dynamic_counterfactual_executor.submit(collect_and_persist)
            dynamic_counterfactual_futures.append((future, primary))

        def finish_dynamic_counterfactuals(*, reason: str, wait_for_completion: bool) -> None:
            nonlocal dynamic_counterfactual_executor
            if dynamic_counterfactual_executor is None:
                return
            cancelled: list[dict[str, Any]] = []
            with counterfactual_commit_lock:
                status_at_boundary = dict(counterfactual_status)
            unfinished_at_boundary = sum(
                (status not in {"completed", "failed"} for status in status_at_boundary.values())
            )
            wait_started = time.monotonic()
            if wait_for_completion:
                dynamic_counterfactual_executor.shutdown(wait=True, cancel_futures=False)
            else:
                with counterfactual_commit_lock:
                    counterfactual_cancellation.set()
                    for future, primary in dynamic_counterfactual_futures:
                        rollout_id = primary.rollout.trajectory.rollout_id
                        status = counterfactual_status.get(rollout_id, "queued")
                        if status in {"completed", "failed"}:
                            continue
                        was_queued = future.cancel()
                        if was_queued:
                            primary.close()
                        counterfactual_status[rollout_id] = "cancelled_after_collection_abort"
                        event = {
                            "rollout_id": rollout_id,
                            "task_id": primary.proposal.task.task_id,
                            "stage": "relation_counterfactual",
                            "reason": reason,
                            "prior_status": status,
                            "future_cancelled_before_start": was_queued,
                            "credit_admitted": False,
                            "timestamp": time.time(),
                        }
                        cancelled.append(event)
                        _append_jsonl(
                            self.output_dir / "relation_counterfactual_cancellations.jsonl", event
                        )
                dynamic_counterfactual_executor.shutdown(wait=False, cancel_futures=True)
            with counterfactual_commit_lock:
                completed = sum(
                    (status == "completed" for status in counterfactual_status.values())
                )
                failed = sum((status == "failed" for status in counterfactual_status.values()))
            _write_json(
                self.output_dir / "counterfactual_update_boundary.json",
                {
                    "schema_version": "counterfactual_update_boundary_v2",
                    "reason": reason,
                    "submitted": len(dynamic_counterfactual_futures),
                    "completed_before_boundary": sum(
                        (status == "completed" for status in status_at_boundary.values())
                    ),
                    "unfinished_at_boundary": unfinished_at_boundary,
                    "completed_after_wait": completed,
                    "failed_after_wait": failed,
                    "settled_after_wait": completed + failed,
                    "cancelled_or_discarded": len(cancelled),
                    "admitted_credit_count": counterfactual_credit_count,
                    "waited_for_incomplete": bool(wait_for_completion and unfinished_at_boundary),
                    "completion_policy": "wait_for_bounded_completion"
                    if wait_for_completion
                    else "cancel_after_collection_abort",
                    "boundary_wait_s": time.monotonic() - wait_started,
                    "primary_budget_shared": False,
                    "pair_wall_budget_s": self.config.counterfactual_pair_wall_time_s,
                    "timestamp": time.time(),
                },
            )
            _append_jsonl(
                self.output_dir / "pipeline_events.jsonl",
                {
                    "event": "counterfactual_update_boundary",
                    "timestamp": time.time(),
                    "reason": reason,
                    "submitted": len(dynamic_counterfactual_futures),
                    "completed": completed,
                    "failed": failed,
                    "cancelled": len(cancelled),
                    "admitted_credit_count": counterfactual_credit_count,
                    "waited_for_incomplete": bool(wait_for_completion and unfinished_at_boundary),
                },
            )
            dynamic_counterfactual_executor = None

        pending_window: _WindowPipelineState | None = None
        deferred_collapse_error: RuntimeError | None = None
        if self.config.task_execution_window is not None:
            physical_window_size = self.config.task_execution_window
        elif self.config.task_scheduling_policy == "frozen_manifest_dynamic":
            physical_window_size = max(1, len(task_specs))
        else:
            physical_window_size = self.config.task_window
        if self.config.pipeline_frontier_by_dataset and physical_window_size < len(task_specs):
            raise ValueError(
                "dataset-early Frontier requires one physical manifest window so dataset completion is known before submission"
            )
        frontier_pipeline_started = time.monotonic()
        frontier_pipeline_executor = (
            ThreadPoolExecutor(
                max_workers=self.config.counterfactual_workers,
                thread_name_prefix="frontier-dataset",
            )
            if self.config.pipeline_frontier_by_dataset
            and self.config.frontier_reverify_fraction > 0
            else None
        )
        frontier_dataset_futures: dict[str, Future[dict[str, dict[str, Any]]]] = {}
        frontier_graph_semaphore = (
            threading.Semaphore(self.config.counterfactual_workers)
            if frontier_pipeline_executor is not None
            else None
        )

        def submit_ready_frontier_datasets() -> None:
            if frontier_pipeline_executor is None:
                return
            datasets = sorted(
                {
                    canonical_dataset_name(proposal.task.metadata.get("dataset", ""))
                    for proposal in proposals
                }
            )
            for dataset in datasets:
                if dataset in frontier_dataset_futures:
                    continue
                dataset_proposals = [
                    proposal
                    for proposal in proposals
                    if canonical_dataset_name(proposal.task.metadata.get("dataset", "")) == dataset
                ]
                expected_ids = {
                    _rollout_id(proposal.task.task_id, rollout_index)
                    for proposal in dataset_proposals
                    for rollout_index in range(self.config.rollouts_per_task)
                }
                completed_ids = rollouts_by_id.keys() | (
                    terminal_failed_ids if self._independent_frontier else set()
                )
                if not expected_ids or not expected_ids <= completed_ids:
                    continue
                if self._independent_frontier:
                    evidence_selection = build_training_selection(
                        dataset_proposals,
                        {
                            rid: rollouts_by_id[rid]
                            for rid in sorted(expected_ids)
                            if rid in rollouts_by_id
                        },
                        self.config.rollouts_per_task,
                        schema=INDEPENDENT_SCHEMA,
                    )
                    admitted = {
                        g["task_id"]
                        for g in evidence_selection["groups"]
                        if g["frontier_rollout_ids"]
                    }
                    dataset_proposals = [p for p in dataset_proposals if p.task.task_id in admitted]
                    dataset_rollouts = [
                        rollouts_by_id[rid] for rid in evidence_selection["frontier_rollout_ids"]
                    ]
                else:
                    dataset_rollouts = [
                        rollouts_by_id[rollout_id] for rollout_id in sorted(expected_ids)
                    ]
                    if not all((_rollout_is_training_eligible(item) for item in dataset_rollouts)):
                        continue
                dataset_dir = self.output_dir / "frontier_by_dataset" / (dataset or "unknown")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                dataset_runner = copy.copy(self)
                dataset_runner.output_dir = dataset_dir
                dataset_runner.config = replace(self.config, pipeline_frontier_by_dataset=False)
                dataset_runner._frontier_graph_semaphore = frontier_graph_semaphore

                def collect(runner, selected_proposals, selected_rollouts):
                    with request_priority("counterfactual"):
                        return runner._collect_frontier_reverification(
                            selected_proposals, selected_rollouts
                        )

                frontier_dataset_futures[dataset] = frontier_pipeline_executor.submit(
                    collect, dataset_runner, tuple(dataset_proposals), tuple(dataset_rollouts)
                )
                _append_jsonl(
                    self.output_dir / "pipeline_events.jsonl",
                    {
                        "event": "frontier_dataset_submitted",
                        "dataset": dataset,
                        "task_count": len(dataset_proposals),
                        "graph_count": len(dataset_rollouts),
                        "timestamp": time.time(),
                    },
                )

        for window_start in range(0, len(task_specs), physical_window_size):
            window_entries: list[tuple[str, ProposedTask]] = []
            for index, seed_text, proposer_index, task_id in task_specs[
                window_start : window_start + physical_window_size
            ]:
                proposal = existing_proposals.get(task_id)
                if proposal is None:
                    try:
                        generated = self.proposer.propose(seed_text, task_id=task_id)
                        if task_requires_reference(generated.task) and generated.task.reference in (
                            None,
                            "",
                            [],
                            {},
                        ):
                            raise ValueError("proposer response has no reference answer")
                    except ValueError as exc:
                        event = {
                            "task_id": task_id,
                            "seed_group": f"seed-{index}",
                            "proposer_index": proposer_index,
                            "success": False,
                            "failure_kind": _proposal_failure_kind(str(exc)),
                            "detail": str(exc),
                        }
                        attempts.append(event)
                        _append_jsonl(self.output_dir / "proposal_attempts.jsonl", event)
                        failed_task_ids.append(task_id)
                        continue
                    selection_group = str(
                        generated.metadata.get("selection_group", f"seed-{index}")
                    )
                    event = {
                        "task_id": task_id,
                        "seed_group": selection_group,
                        "proposer_index": proposer_index,
                        "success": True,
                        "failure_kind": None,
                    }
                    attempts.append(event)
                    _append_jsonl(self.output_dir / "proposal_attempts.jsonl", event)
                    proposal = replace(
                        generated,
                        metadata={
                            **generated.metadata,
                            "seed_group": selection_group,
                            "proposer_index": proposer_index,
                        },
                    )
                    proposals.append(proposal)
                    existing_proposals[task_id] = proposal
                    _append_jsonl(self.output_dir / "tasks.jsonl", _proposal_dict(proposal))
                elif task_id not in observed_task_ids:
                    reserve = getattr(self.proposer, "reserve", None)
                    if callable(reserve):
                        reserve(proposal)
                window_entries.append((task_id, proposal))
            state_dict = getattr(self.proposer, "state_dict", None)
            if callable(state_dict):
                _write_json(state_path, state_dict())
            expected_rollout_ids = {
                _rollout_id(task_id, rollout_index)
                for (task_id, _proposal) in window_entries
                for rollout_index in range(self.config.rollouts_per_task)
            }
            planned_jobs = [
                (proposal, rollout_index)
                for rollout_index in range(self.config.rollouts_per_task)
                for (_task_id, proposal) in window_entries
            ]
            (ordered_jobs, frozen_job_schedule) = _freeze_primary_job_schedule(
                self.output_dir,
                window_start=window_start,
                jobs=planned_jobs,
                requested_order=self.config.primary_job_order,
            )
            jobs = [
                job
                for job in ordered_jobs
                if str(job[0].task.task_id) not in quarantined_task_ids
                if _primary_job_rollout_id(job) not in rollouts_by_id
                if _primary_job_rollout_id(job) not in terminal_failed_ids
            ]
            schedule_rank = {
                str(row["rollout_id"]): int(row["rank"]) for row in frozen_job_schedule["jobs"]
            }
            schedule_estimate_s = {
                str(row["rollout_id"]): float(row["estimated_duration_s"])
                for row in frozen_job_schedule["jobs"]
            }
            effective_primary_job_order = str(frozen_job_schedule["order"])
            scheduler_path = self.output_dir / "rollout_scheduler_events.jsonl"
            scheduler_state_lock = threading.Lock()
            scheduler_started_groups: set[str] = set()
            scheduler_finished_by_group: dict[str, int] = {}
            scheduler_expected_by_group: dict[str, int] = {}
            for proposal, _rollout_index in jobs:
                task_id = str(proposal.task.task_id)
                scheduler_expected_by_group[task_id] = (
                    scheduler_expected_by_group.get(task_id, 0) + 1
                )

            def scheduler_event(
                event: str, *, scheduler_path: Path = scheduler_path, **payload: Any
            ) -> None:
                _append_jsonl(
                    scheduler_path,
                    {
                        "schema_version": 1,
                        "event": event,
                        "timestamp": time.time(),
                        "monotonic_s": time.monotonic(),
                        **payload,
                    },
                )

            for proposal, rollout_index in jobs:
                rollout_id = _rollout_id(proposal.task.task_id, rollout_index)
                scheduler_event(
                    "job_submitted",
                    task_id=proposal.task.task_id,
                    rollout_id=rollout_id,
                    rollout_index=rollout_index,
                    dataset=canonical_dataset_name(proposal.task.metadata.get("dataset", "")),
                    primary_job_order=effective_primary_job_order,
                    schedule_rank=schedule_rank[rollout_id],
                    estimated_duration_s=schedule_estimate_s[rollout_id],
                )
            if self.config.pipeline_counterfactuals:
                _append_jsonl(
                    self.output_dir / "pipeline_events.jsonl",
                    {
                        "event": "primary_window_started",
                        "window_index": window_start // max(1, self.config.task_window),
                        "timestamp": time.time(),
                        "jobs": len(jobs),
                    },
                )
            collection_cancelled = threading.Event()

            def collect_primary_steps(
                job,
                collection_cancelled=collection_cancelled,
                scheduler_event=scheduler_event,
                scheduler_state_lock=scheduler_state_lock,
                scheduler_started_groups=scheduler_started_groups,
                scheduler_finished_by_group=scheduler_finished_by_group,
                scheduler_expected_by_group=scheduler_expected_by_group,
            ):
                (proposal, rollout_index) = job
                attempt_events: list[dict[str, Any]] = []
                dataset = canonical_dataset_name(proposal.task.metadata.get("dataset", ""))
                task_id = str(proposal.task.task_id)
                rollout_id = _rollout_id(task_id, rollout_index)
                worker_name = threading.current_thread().name
                with scheduler_state_lock:
                    if task_id not in scheduler_started_groups:
                        scheduler_started_groups.add(task_id)
                        scheduler_event(
                            "group_admitted",
                            task_id=task_id,
                            dataset=dataset,
                            expected_jobs=scheduler_expected_by_group[task_id],
                            worker=worker_name,
                        )
                scheduler_event(
                    "job_started",
                    task_id=task_id,
                    rollout_id=rollout_id,
                    rollout_index=rollout_index,
                    dataset=dataset,
                    worker=worker_name,
                )

                def finish_scheduler_job(status: str) -> None:
                    scheduler_event(
                        "job_finished",
                        task_id=task_id,
                        rollout_id=rollout_id,
                        rollout_index=rollout_index,
                        dataset=dataset,
                        worker=worker_name,
                        status=status,
                    )
                    with scheduler_state_lock:
                        finished = scheduler_finished_by_group.get(task_id, 0) + 1
                        scheduler_finished_by_group[task_id] = finished
                        if finished == scheduler_expected_by_group[task_id]:
                            scheduler_event(
                                "group_released",
                                task_id=task_id,
                                dataset=dataset,
                                completed_jobs=finished,
                                worker=worker_name,
                            )

                maximum_attempt = (
                    self.config.swe_non_trainable_recovery_attempts
                    if dataset == "swe_bench"
                    else self.config.non_swe_recovery_attempts
                )
                if not self.config.allow_legacy_whole_rollout_recovery:
                    maximum_attempt = 0
                slot_timeout_s = self._slot_wall_time_s(proposal)
                attempt_timeout_s = self._deadline_profile(proposal)[0]
                slot_started = time.monotonic()
                last_exclusions: list[str] = []
                semantic_recoveries_used = 0
                backend_retries_used = 0
                replacement_attempt = 0
                while True:
                    if collection_cancelled.is_set():
                        finish_scheduler_job("cancelled")
                        return (
                            job,
                            None,
                            RuntimeError("same-slot recovery cancelled by collection circuit"),
                            attempt_events,
                        )
                    slot_elapsed_s = time.monotonic() - slot_started
                    slot_remaining_s = slot_timeout_s - slot_elapsed_s
                    if slot_remaining_s <= 0:
                        exc = WorkerWallClockLimitExceeded(
                            f"same-slot recovery exhausted its shared wall-clock budget ({slot_elapsed_s:.1f}/{slot_timeout_s:.1f}s)",
                            reason="hard_deadline",
                            stage="same_slot_recovery_admission",
                            elapsed_s=slot_elapsed_s,
                            idle_s=0.0,
                        )
                        event = self._rollout_attempt_event(
                            proposal,
                            rollout_index,
                            replacement_attempt=replacement_attempt,
                            error=exc,
                            duration_s=0.0,
                        )
                        event.update(
                            {
                                "recovery_scope": RecoveryScope.NONE.value,
                                "recovery_reason": "slot_wall_clock_exhausted",
                            }
                        )
                        attempt_events.append(event)
                        scheduler_event(
                            "attempt_finished",
                            task_id=task_id,
                            rollout_id=rollout_id,
                            rollout_index=rollout_index,
                            dataset=dataset,
                            worker=worker_name,
                            attempt=replacement_attempt,
                            status="slot_wall_clock_exhausted",
                        )
                        finish_scheduler_job("slot_wall_clock_exhausted")
                        return (job, None, exc, attempt_events)
                    attempt_started = time.monotonic()
                    effective_attempt_timeout_s = min(attempt_timeout_s, slot_remaining_s)
                    scheduler_event(
                        "attempt_started",
                        task_id=task_id,
                        rollout_id=rollout_id,
                        rollout_index=rollout_index,
                        dataset=dataset,
                        worker=worker_name,
                        attempt=replacement_attempt,
                        effective_attempt_timeout_s=effective_attempt_timeout_s,
                        slot_remaining_s=slot_remaining_s,
                        whole_rollout_recovery_limit=maximum_attempt,
                    )
                    try:
                        outcome = self._collect_primary(
                            job,
                            replacement_attempt=replacement_attempt,
                            cancellation_event=collection_cancelled,
                            total_timeout_s_override=effective_attempt_timeout_s,
                        )
                    except Exception as exc:
                        decision = _recovery_decision(dataset, error=exc)
                        direct_backend_failure = classify_backend_failure(exc)
                        retryable_backend = (
                            isinstance(exc, WorkerBackendUnavailableError)
                            and exc.retryable
                            or (
                                direct_backend_failure.backend_failure
                                and direct_backend_failure.retryable
                            )
                        )
                        backend_retry_available = (
                            retryable_backend
                            and self.config.allow_legacy_whole_rollout_recovery
                            and (backend_retries_used < self.config.backend_failure_retry_attempts)
                            and (not collection_cancelled.is_set())
                        )
                        backend_failure_skipped = (
                            (
                                isinstance(exc, WorkerBackendUnavailableError)
                                or direct_backend_failure.backend_failure
                            )
                            and self.config.backend_failure_retry_attempts > 0
                            and (not backend_retry_available)
                        )
                        event = self._rollout_attempt_event(
                            proposal,
                            rollout_index,
                            replacement_attempt=replacement_attempt,
                            error=exc,
                            duration_s=time.monotonic() - attempt_started,
                        )
                        event.update(
                            {
                                "recovery_scope": decision.scope.value,
                                "recovery_reason": decision.reason,
                                "effective_attempt_timeout_s": effective_attempt_timeout_s,
                                "slot_timeout_s": slot_timeout_s,
                                "infrastructure_incident": decision.infrastructure_incident
                                and (not backend_retry_available)
                                and (not backend_failure_skipped),
                                "backend_retries_used": backend_retries_used,
                                "backend_retry_limit": self.config.backend_failure_retry_attempts
                                if self.config.allow_legacy_whole_rollout_recovery
                                else 0,
                                "backend_rollout_replay_disabled": not self.config.allow_legacy_whole_rollout_recovery,
                            }
                        )
                        attempt_events.append(event)
                        if not (backend_retry_available or backend_failure_skipped):
                            scheduler_event(
                                "attempt_finished",
                                task_id=task_id,
                                rollout_id=rollout_id,
                                rollout_index=rollout_index,
                                dataset=dataset,
                                worker=worker_name,
                                attempt=replacement_attempt,
                                status="error",
                                error_type=type(exc).__name__,
                                recovery_scope=decision.scope.value,
                                recovery_reason=decision.reason,
                            )
                        if backend_retry_available:
                            backend_retries_used += 1
                            event["recovery_scope"] = RecoveryScope.FULL_PRIMARY.value
                            event["recovery_reason"] = "retryable_backend_failure"
                            event["backend_retries_used"] = backend_retries_used
                            scheduler_event(
                                "attempt_finished",
                                task_id=task_id,
                                rollout_id=rollout_id,
                                rollout_index=rollout_index,
                                dataset=dataset,
                                worker=worker_name,
                                attempt=replacement_attempt,
                                status="backend_retry",
                                error_type=type(exc).__name__,
                                recovery_scope=RecoveryScope.FULL_PRIMARY.value,
                                recovery_reason="retryable_backend_failure",
                            )
                            replacement_attempt += 1
                            delay_s = min(30.0, 2.0 ** min(backend_retries_used, 5))
                            delay_s = min(
                                delay_s,
                                max(0.0, slot_timeout_s - (time.monotonic() - slot_started)),
                            )
                            scheduler_event(
                                "retry_queued",
                                task_id=task_id,
                                rollout_id=rollout_id,
                                attempt=replacement_attempt,
                                delay_s=delay_s,
                                reason="retryable_backend_failure",
                            )
                            yield delay_s
                            continue
                        if backend_failure_skipped:
                            exhausted_reason = (
                                (
                                    "retryable_backend_retries_exhausted"
                                    if self.config.allow_legacy_whole_rollout_recovery
                                    else "backend_request_failed_no_rollout_replay"
                                )
                                if retryable_backend
                                else "permanent_backend_failure_skipped"
                            )
                            event["recovery_reason"] = exhausted_reason
                            event["backend_retries_exhausted"] = retryable_backend
                            scheduler_event(
                                "attempt_finished",
                                task_id=task_id,
                                rollout_id=rollout_id,
                                rollout_index=rollout_index,
                                dataset=dataset,
                                worker=worker_name,
                                attempt=replacement_attempt,
                                status="backend_failure_skipped",
                                error_type=type(exc).__name__,
                                recovery_scope=RecoveryScope.NONE.value,
                                recovery_reason=exhausted_reason,
                            )
                            finish_scheduler_job("backend_failure_skipped")
                            return (
                                job,
                                None,
                                BackendRetryExhaustedError(
                                    getattr(exc, "failure", direct_backend_failure.to_dict())
                                ),
                                attempt_events,
                            )
                        if decision.infrastructure_incident:
                            finish_scheduler_job("infrastructure_incident")
                            return (job, None, exc, attempt_events)
                        if (
                            decision.scope is not RecoveryScope.NONE
                            and semantic_recoveries_used < maximum_attempt
                            and (not collection_cancelled.is_set())
                        ):
                            semantic_recoveries_used += 1
                            replacement_attempt += 1
                            yield 0.0
                            continue
                        finish_scheduler_job("error")
                        if decision.scope is not RecoveryScope.NONE:
                            return (
                                job,
                                None,
                                UncertainAttributionExhaustedError(
                                    "attribution-uncertain failure exhausted its one same-slot recovery; reason="
                                    + decision.reason
                                ),
                                attempt_events,
                            )
                        return (job, None, exc, attempt_events)
                    bounded_terminal = outcome.rollout.trajectory.metadata.get(
                        "bounded_director_terminal"
                    ) in {"director_no_legal_continuation", "director_no_progress_exhausted"} and (
                        not any(
                            (
                                outcome.rollout.trajectory.metadata.get(key)
                                for key in (
                                    "worker_backend_failure",
                                    "infrastructure_failure",
                                    "swe_infrastructure_failure",
                                    "swe_synthetic_evaluation",
                                    "swe_non_train_split",
                                )
                            )
                        )
                    )
                    if bounded_terminal and self.config.uncertain_attribution_zero_reward:
                        outcome.rollout = _uncertain_failure_zero(
                            outcome.rollout,
                            str(outcome.rollout.trajectory.metadata["bounded_director_terminal"]),
                        )
                    training_eligible = _rollout_is_training_eligible(outcome.rollout)
                    if (
                        training_eligible
                        or _rollout_has_trusted_score(outcome.rollout)
                        or bounded_terminal
                    ):
                        admission_reason = (
                            "terminal_contract_satisfied"
                            if training_eligible
                            else "trusted_score_preserved_training_excluded"
                            if _rollout_has_trusted_score(outcome.rollout)
                            else "bounded_director_terminal_no_recollection"
                        )
                        if replacement_attempt or not training_eligible:
                            event = self._rollout_attempt_event(
                                proposal,
                                rollout_index,
                                replacement_attempt=replacement_attempt,
                                outcome=outcome,
                                accepted_as_primary=True,
                            )
                            event.update(
                                {
                                    "recovery_scope": "accepted_primary",
                                    "recovery_reason": admission_reason,
                                    "training_eligible": training_eligible,
                                    "effective_attempt_timeout_s": effective_attempt_timeout_s,
                                    "slot_timeout_s": slot_timeout_s,
                                }
                            )
                            attempt_events.append(event)
                        scheduler_event(
                            "attempt_finished",
                            task_id=task_id,
                            rollout_id=rollout_id,
                            rollout_index=rollout_index,
                            dataset=dataset,
                            worker=worker_name,
                            attempt=replacement_attempt,
                            status="accepted_primary",
                            training_eligible=training_eligible,
                            recovery_reason=admission_reason,
                        )
                        finish_scheduler_job("accepted_primary")
                        return (job, outcome, None, attempt_events)
                    last_exclusions = list(
                        outcome.rollout.trajectory.metadata.get("training_exclusion_reasons", ())
                    )
                    decision = _recovery_decision(dataset, rollout=outcome.rollout)
                    event = self._rollout_attempt_event(
                        proposal,
                        rollout_index,
                        replacement_attempt=replacement_attempt,
                        outcome=outcome,
                        accepted_as_primary=False,
                    )
                    event.update(
                        {
                            "recovery_scope": decision.scope.value,
                            "recovery_reason": decision.reason,
                            "effective_attempt_timeout_s": effective_attempt_timeout_s,
                            "slot_timeout_s": slot_timeout_s,
                            "infrastructure_incident": decision.infrastructure_incident,
                        }
                    )
                    attempt_events.append(event)
                    scheduler_event(
                        "attempt_finished",
                        task_id=task_id,
                        rollout_id=rollout_id,
                        rollout_index=rollout_index,
                        dataset=dataset,
                        worker=worker_name,
                        attempt=replacement_attempt,
                        status="non_trainable",
                        recovery_scope=decision.scope.value,
                        recovery_reason=decision.reason,
                    )
                    if (
                        self.config.uncertain_attribution_zero_reward
                        and (not decision.infrastructure_incident)
                        and (
                            decision.scope is not RecoveryScope.NONE
                            or "outcome_attribution_unresolved" in last_exclusions
                            or "tool_failure_attribution_unresolved" in last_exclusions
                        )
                        and (
                            decision.scope is RecoveryScope.NONE
                            or semantic_recoveries_used >= maximum_attempt
                        )
                    ):
                        scored = _uncertain_failure_zero(outcome.rollout, decision.reason)
                        if scored.trajectory.metadata.get("reward_known") is True:
                            outcome.rollout = scored
                            finish_scheduler_job("scored_terminal_failure")
                            return (job, outcome, None, attempt_events)
                    if decision.reason == "isolated_swe_verifier_transport_failure":
                        outcome.rollout.trajectory.metadata[
                            "isolated_verifier_transport_failure"
                        ] = True
                        finish_scheduler_job("isolated_verifier_transport_failure")
                        return (job, outcome, None, attempt_events)
                    outcome.close()
                    if decision.infrastructure_incident:
                        finish_scheduler_job("infrastructure_incident")
                        return (
                            job,
                            None,
                            CollectionInfrastructureIncidentError(
                                "runtime reported a backend, tool, or environment incident; reason="
                                + decision.reason
                            ),
                            attempt_events,
                        )
                    if (
                        decision.scope is not RecoveryScope.NONE
                        and semantic_recoveries_used < maximum_attempt
                    ):
                        semantic_recoveries_used += 1
                        replacement_attempt += 1
                        yield 0.0
                        continue
                    break
                finish_scheduler_job("non_trainable_exhausted")
                if decision.scope is not RecoveryScope.NONE:
                    return (
                        job,
                        None,
                        UncertainAttributionExhaustedError(
                            "attribution-uncertain terminal result exhausted its one same-slot recovery; reason="
                            + decision.reason
                        ),
                        attempt_events,
                    )
                return (
                    job,
                    None,
                    NonTrainablePrimaryExhaustedError(
                        f"{dataset or 'unknown'} same-slot recovery exhausted without a trainable terminal result; exclusions="
                        + ",".join(last_exclusions or ["unknown"])
                    ),
                    attempt_events,
                )

            def collect_primary_outcome(
                job,
                collect_primary_steps=collect_primary_steps,
                collection_cancelled=collection_cancelled,
            ):
                steps = collect_primary_steps(job)
                while True:
                    try:
                        delay_s = next(steps)
                    except StopIteration as done:
                        return done.value
                    collection_cancelled.wait(delay_s)

            resumable_map = getattr(self.rollout_pool, "iter_map_resumable", None)
            iter_map = getattr(self.rollout_pool, "iter_map", None)
            grouped_iter_map = getattr(self.rollout_pool, "iter_map_grouped", None)
            if callable(resumable_map):
                results = resumable_map(
                    collect_primary_steps,
                    jobs,
                    group_key=lambda job: str(job[0].task.task_id),
                    max_active_groups=self.config.max_active_task_groups
                    if self.config.task_scheduling_policy == "frozen_manifest_dynamic"
                    else max(1, len(jobs)),
                )
            elif self.config.task_scheduling_policy == "frozen_manifest_dynamic":
                if not callable(grouped_iter_map):
                    raise TypeError(
                        "frozen_manifest_dynamic scheduling requires a rollout pool with iter_map_grouped"
                    )
                results = grouped_iter_map(
                    collect_primary_outcome,
                    jobs,
                    group_key=lambda job: str(job[0].task.task_id),
                    max_active_groups=self.config.max_active_task_groups,
                )
            else:
                results = (
                    iter_map(collect_primary_outcome, jobs)
                    if callable(iter_map)
                    else self.rollout_pool.map(collect_primary_outcome, jobs)
                )
            collection_errors: list[dict[str, Any]] = []
            counterfactual_jobs: list[_PrimaryCollection] = []
            completed_job_ids: set[str] = set()
            backend_failure_total = 0
            backend_failure_by_route: dict[str, int] = {}
            backend_failure_by_kind: dict[str, int] = {}
            backend_failure_by_origin: dict[str, int] = {}
            consecutive_backend_failures = 0
            consecutive_backend_failures_by_route: dict[str, int] = {}
            circuit_open = False
            collection_abort: dict[str, Any] | None = None
            for job, outcome, error, attempt_events in results:
                (proposal, rollout_index) = job
                rollout_id = _rollout_id(proposal.task.task_id, rollout_index)
                completed_job_ids.add(rollout_id)
                for attempt_event in attempt_events:
                    persisted_attempt = {
                        key: value
                        for (key, value) in attempt_event.items()
                        if key not in {"partial_state", "trajectory"}
                    }
                    if bool(persisted_attempt.get("infrastructure_incident")):
                        _append_jsonl(
                            self.output_dir / "collection_incidents.jsonl",
                            {
                                **persisted_attempt,
                                "event": "infrastructure_incident",
                                "counted_as_recovery_attempt": False,
                                "requires_cycle_recollection": True,
                                "checkpoint_resume_policy": "last_committed_checkpoint_only",
                            },
                        )
                        continue
                    _append_jsonl(self.output_dir / "rollout_attempts.jsonl", persisted_attempt)
                    attempt_trajectory = attempt_event.get("trajectory")
                    if isinstance(attempt_trajectory, dict):
                        _append_jsonl(
                            self.output_dir / "rollout_attempt_trajectories.jsonl",
                            {
                                "task_id": proposal.task.task_id,
                                "rollout_id": rollout_id,
                                "rollout_index": rollout_index,
                                "replacement_attempt": attempt_event["replacement_attempt"],
                                "accepted_as_primary": False,
                                "trajectory": attempt_trajectory,
                            },
                        )
                    partial_state = attempt_event.get("partial_state")
                    if isinstance(partial_state, dict):
                        _append_jsonl(
                            self.output_dir / "rollout_timeout_snapshots.jsonl",
                            {
                                "task_id": proposal.task.task_id,
                                "rollout_id": rollout_id,
                                "rollout_index": rollout_index,
                                "replacement_attempt": attempt_event["replacement_attempt"],
                                **partial_state,
                            },
                        )
                if error is not None:
                    circuit_relevant = False
                    error_event = {
                        "task_id": proposal.task.task_id,
                        "rollout_id": rollout_id,
                        "rollout_index": rollout_index,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                    classification = classify_backend_failure(error)
                    if classification.origin == "environment_service":
                        error_event["environment_failure"] = classification.to_dict()
                        error_event["environment_request_events"] = list(
                            getattr(error, "request_events", ())
                        )
                    timeout_diagnostics = getattr(error, "to_dict", None)
                    if callable(timeout_diagnostics):
                        error_event["timeout"] = timeout_diagnostics()
                    if attempt_events:
                        error_event["attempt_count"] = len(attempt_events)
                    if isinstance(
                        error, (WorkerBackendUnavailableError, BackendRetryExhaustedError)
                    ):
                        error_event["backend_failure"] = dict(error.failure)
                        for request_event in error.request_events:
                            scoped_state = self._record_scoped_route_event(
                                route_health_store,
                                proposal=proposal,
                                rollout_id=rollout_id,
                                event=request_event,
                            )
                            _append_jsonl(
                                self.output_dir / "backend_api_events.jsonl",
                                {
                                    "task_id": proposal.task.task_id,
                                    "rollout_id": rollout_id,
                                    "rollout_index": rollout_index,
                                    **request_event,
                                    **(
                                        {"scoped_route_health": scoped_state}
                                        if scoped_state
                                        else {}
                                    ),
                                },
                            )
                        backend_failure_total += 1
                        isolate_transient_failure = (
                            isinstance(error, BackendRetryExhaustedError)
                            and error.retryable
                            and (not error.disable_route)
                        )
                        isolate_judge_failure = canonical_dataset_name(
                            proposal.task.metadata.get("dataset", "")
                        ) == "healthbench_professional" and _is_healthbench_judge_backend_failure(
                            error.failure
                        )
                        error_event["isolated_transient_failure"] = isolate_transient_failure
                        error_event["isolated_judge_failure"] = isolate_judge_failure
                        circuit_relevant = not (
                            isolate_transient_failure or isolate_judge_failure
                        ) and (error.counts_toward_route_circuit or error.disable_route)
                        if circuit_relevant:
                            consecutive_backend_failures += 1
                        persistent_circuit_open = False
                        persistent_states: dict[str, dict[str, Any]] = {}
                        for detail in error.failure_details:
                            kind = str(detail.get("kind", "unknown"))
                            origin = str(detail.get("origin", "unknown"))
                            backend_failure_by_kind[kind] = backend_failure_by_kind.get(kind, 0) + 1
                            backend_failure_by_origin[origin] = (
                                backend_failure_by_origin.get(origin, 0) + 1
                            )
                        for route in error.routes or ("unknown",):
                            backend_failure_by_route[route] = (
                                backend_failure_by_route.get(route, 0) + 1
                            )
                            (counts_for_route, disable_route) = error.route_policy(route)
                            if isolate_transient_failure or isolate_judge_failure:
                                counts_for_route = False
                            if counts_for_route:
                                increment = (
                                    self.config.backend_failure_route_threshold
                                    if disable_route
                                    else 1
                                )
                                consecutive_backend_failures_by_route[route] = (
                                    consecutive_backend_failures_by_route.get(route, 0) + increment
                                )
                            if route_health_store is not None and counts_for_route:
                                state = route_health_store.record_failure(
                                    route,
                                    failure_type=",".join(error.failure_types) or "unknown",
                                    threshold=1
                                    if disable_route
                                    else self.config.backend_failure_route_threshold,
                                    evidence={
                                        "output_dir": str(self.output_dir),
                                        "rollout_id": rollout_id,
                                        "stage": "primary_worker_execution",
                                        "failure_details": list(error.failure_details),
                                    },
                                )
                                persistent_states[route] = state
                                persistent_circuit_open = (
                                    persistent_circuit_open or state.get("status") == "open"
                                )
                        if persistent_states:
                            error_event["persistent_route_health"] = persistent_states
                        _append_jsonl(
                            self.output_dir / "backend_api_events.jsonl",
                            {
                                **error_event,
                                "stage": "primary_worker_execution",
                                "event": "terminal_backend_failure",
                            },
                        )
                    collection_errors.append(error_event)
                    _append_jsonl(self.output_dir / "rollout_errors.jsonl", error_event)
                    terminal_failed_ids.add(str(error_event["rollout_id"]))
                    attempt_marked_infrastructure = any(
                        (bool(event.get("infrastructure_incident")) for event in attempt_events)
                    )
                    backend_incident = (
                        isinstance(
                            error, (WorkerBackendUnavailableError, BackendRetryExhaustedError)
                        )
                        or classification.backend_failure
                    )
                    is_infrastructure_incident = isinstance(
                        error,
                        (CollectionInfrastructureIncidentError, SWEWorkspaceProvisioningError),
                    ) or (
                        (
                            isinstance(error, WorkerBackendUnavailableError)
                            or attempt_marked_infrastructure
                        )
                        and (
                            not (
                                self.config.continue_after_backend_failure_circuit
                                and backend_incident
                            )
                        )
                    )
                    is_uncertain_exhausted = isinstance(error, UncertainAttributionExhaustedError)
                    if (
                        (is_uncertain_exhausted or isinstance(error, WorkerWallClockLimitExceeded))
                        and (not is_infrastructure_incident)
                        and self.config.uncertain_attribution_zero_reward
                    ):
                        _append_jsonl(
                            self.output_dir / "uncertain_failure_outcomes.jsonl",
                            {
                                **error_event,
                                "reward_known": True,
                                "task_reward": 0.0,
                                "task_outcome_passed": False,
                                "training_eligible": False,
                                "reward_admission_reason": "uncertain_attribution_zero",
                                "uncertain_attribution_zero": {
                                    "attribution": "unknown",
                                    "scoring_policy": "uncertain_attribution_zero_v1",
                                    "reason": str(error),
                                },
                            },
                        )
                    if is_infrastructure_incident or (
                        is_uncertain_exhausted
                        and (not self.config.continue_on_uncertain_attribution_exhausted)
                        and (self.config.rollout_group_policy != "eligible_subset")
                        and (not self.config.uncertain_attribution_zero_reward)
                    ):
                        collection_abort = {
                            **error_event,
                            "event": "collection_aborted",
                            "incident_class": "infrastructure"
                            if is_infrastructure_incident
                            else "attribution_uncertain_recovery_exhausted",
                            "counted_as_recovery_attempt": False,
                            "requires_cycle_recollection": True,
                            "checkpoint_resume_policy": "last_committed_checkpoint_only",
                            "resume_instruction": "repair the incident, start a fresh collection epoch, and continue training from the most recent committed checkpoint; do not reuse this partial cycle",
                        }
                        collection_cancelled.set()
                        _append_jsonl(
                            self.output_dir / "collection_incidents.jsonl", collection_abort
                        )
                        close_results = getattr(results, "close", None)
                        if callable(close_results):
                            close_results()
                        break
                    circuit_open = (
                        isinstance(error, WorkerBackendUnavailableError)
                        and persistent_circuit_open
                        or (
                            circuit_relevant
                            and consecutive_backend_failures
                            >= self.config.backend_failure_total_threshold
                        )
                        or any(
                            (
                                count >= self.config.backend_failure_route_threshold
                                for count in consecutive_backend_failures_by_route.values()
                            )
                        )
                    )
                    if circuit_open:
                        if self.config.continue_after_backend_failure_circuit:
                            circuit_open = False
                            continue
                        collection_cancelled.set()
                        close_results = getattr(results, "close", None)
                        if callable(close_results):
                            close_results()
                        break
                    continue
                assert outcome is not None
                consecutive_backend_failures = 0
                primary = outcome
                rollout = primary.rollout
                rollout_id = rollout.trajectory.rollout_id
                for request_event in rollout.trajectory.metadata.get("backend_request_events", ()):
                    if isinstance(request_event, dict):
                        scoped_state = self._record_scoped_route_event(
                            route_health_store,
                            proposal=proposal,
                            rollout_id=rollout_id,
                            event=request_event,
                        )
                        _append_jsonl(
                            self.output_dir / "backend_api_events.jsonl",
                            {
                                "task_id": proposal.task.task_id,
                                "rollout_id": rollout_id,
                                "rollout_index": rollout_index,
                                **request_event,
                                **({"scoped_route_health": scoped_state} if scoped_state else {}),
                            },
                        )
                try:
                    if rollout_id not in expected_rollout_ids:
                        raise ValueError(f"collector returned unexpected rollout_id {rollout_id!r}")
                    existing = rollouts_by_id.get(rollout_id)
                    if existing is not None:
                        if existing.trajectory.to_dict() != rollout.trajectory.to_dict():
                            raise ValueError(f"conflicting duplicate rollout_id: {rollout_id}")
                        primary.close()
                        continue
                    rollouts.append(rollout)
                    rollouts_by_id[rollout_id] = rollout
                    successful_routes = {
                        str(node.metadata.get("runtime_route", "")).strip()
                        for node in primary.rollout.graph.nodes.values()
                    }
                    for route in sorted(successful_routes - {""}):
                        consecutive_backend_failures_by_route[route] = 0
                        if route_health_store is not None:
                            route_health_store.record_runtime_success(
                                route,
                                evidence={
                                    "output_dir": str(self.output_dir),
                                    "rollout_id": rollout_id,
                                },
                            )
                    _append_jsonl(
                        self.output_dir / "solver_rollouts.jsonl", rollout.trajectory.to_dict()
                    )
                    self._write_progress(proposals=proposals, rollouts_by_id=rollouts_by_id)
                    if not _rollout_is_training_eligible(rollout):
                        _append_jsonl(
                            self.output_dir / "rollout_training_exclusions.jsonl",
                            _rollout_training_exclusion(rollout),
                        )
                        primary.close()
                        continue
                    if self.primary_probability_observer is not None:
                        self.primary_probability_observer.submit(rollout.trajectory)
                    self._prepare_counterfactual(primary)
                    self._persist_counterfactual_metadata(primary.rollout)
                    if primary.decisions:
                        if dynamic_counterfactual_executor is not None:
                            submit_dynamic_counterfactual(
                                primary,
                                window_index=window_start // max(1, self.config.task_window),
                            )
                        else:
                            counterfactual_jobs.append(primary)
                    else:
                        for event in primary.rollout.trajectory.metadata.get(
                            "relation_counterfactual_errors", ()
                        ):
                            _append_jsonl(
                                self.output_dir / "relation_counterfactual_errors.jsonl",
                                {"rollout_id": rollout_id, **event},
                            )
                        primary.close()
                    submit_ready_frontier_datasets()
                except BaseException:
                    primary.close()
                    raise
            if collection_abort is not None:
                if frontier_pipeline_executor is not None:
                    frontier_pipeline_executor.shutdown(wait=False, cancel_futures=True)
                if self.primary_probability_observer is not None:
                    self.primary_probability_observer.abort()
                finish_dynamic_counterfactuals(
                    reason="collection_aborted", wait_for_completion=False
                )
                for primary in counterfactual_jobs:
                    primary.close()
                counterfactual_jobs.clear()
                _write_json(self.output_dir / "collection_abort.json", collection_abort)
                raise CollectionInfrastructureIncidentError(
                    "collection aborted before any optimizer update; "
                    + str(collection_abort["incident_class"])
                    + "; resume from the last committed checkpoint after repair"
                )
            if self.config.pipeline_counterfactuals:
                _append_jsonl(
                    self.output_dir / "pipeline_events.jsonl",
                    {
                        "event": "primary_window_completed",
                        "window_index": window_start // max(1, self.config.task_window),
                        "timestamp": time.time(),
                        "persisted_rollouts": len(expected_rollout_ids & rollouts_by_id.keys()),
                    },
                )
            if circuit_open:
                for proposal, rollout_index in jobs:
                    rollout_id = _rollout_id(proposal.task.task_id, rollout_index)
                    if rollout_id in completed_job_ids:
                        continue
                    error_event = {
                        "task_id": proposal.task.task_id,
                        "rollout_id": rollout_id,
                        "rollout_index": rollout_index,
                        "error_type": "BackendCircuitOpenError",
                        "message": "rollout not started or result discarded after the backend failure circuit opened",
                    }
                    collection_errors.append(error_event)
                    _append_jsonl(self.output_dir / "rollout_errors.jsonl", error_event)
                    terminal_failed_ids.add(str(error_event["rollout_id"]))
                _append_jsonl(
                    self.output_dir / "backend_api_events.jsonl",
                    {
                        "stage": "primary_worker_execution",
                        "event": "backend_circuit_open",
                        "failure_total": backend_failure_total,
                        "failure_by_route": backend_failure_by_route,
                        "failure_by_kind": backend_failure_by_kind,
                        "failure_by_origin": backend_failure_by_origin,
                        "route_threshold": self.config.backend_failure_route_threshold,
                        "total_threshold": self.config.backend_failure_total_threshold,
                    },
                )
                for primary in counterfactual_jobs:
                    primary.close()
                counterfactual_jobs.clear()
            window_complete_entries: list[tuple[str, ProposedTask]] = []
            for task_id, proposal in window_entries:
                expected_ids = {
                    _rollout_id(task_id, rollout_index)
                    for rollout_index in range(self.config.rollouts_per_task)
                }
                valid_ids = sorted(
                    (
                        rollout_id
                        for rollout_id in expected_ids & rollouts_by_id.keys()
                        if _rollout_is_training_eligible(rollouts_by_id[rollout_id])
                    )
                )
                non_trainable_ids = sorted(
                    (
                        rollout_id
                        for rollout_id in expected_ids & rollouts_by_id.keys()
                        if not _rollout_is_training_eligible(rollouts_by_id[rollout_id])
                    )
                )
                if (
                    len(valid_ids) == self.config.rollouts_per_task
                    and task_id not in quarantined_task_ids
                ):
                    window_complete_entries.append((task_id, proposal))
                    continue
                if self.config.rollout_group_policy == "eligible_subset":
                    continue
                if task_id in quarantined_task_ids:
                    continue
                task_errors = [
                    item for item in collection_errors if str(item.get("task_id", "")) == task_id
                ]
                quarantine = {
                    "task_id": task_id,
                    "status": "quarantined",
                    "reason": "backend_circuit_open"
                    if circuit_open
                    else "non_trainable_rollout_group"
                    if non_trainable_ids
                    else "incomplete_rollout_group",
                    "expected_rollout_count": self.config.rollouts_per_task,
                    "valid_rollout_count": len(valid_ids),
                    "valid_rollout_ids": valid_ids,
                    "missing_rollout_ids": sorted(expected_ids - rollouts_by_id.keys()),
                    "non_trainable_rollout_ids": non_trainable_ids,
                    "training_exclusions": [
                        _rollout_training_exclusion(rollouts_by_id[rollout_id])
                        for rollout_id in non_trainable_ids
                    ],
                    "error_types": sorted(
                        {str(item.get("error_type", "unknown")) for item in task_errors}
                    ),
                    "failure_records": task_errors,
                    "training_excluded": True,
                    "exact_resume_allowed": False,
                    "window_index": window_start // max(1, self.config.task_window),
                }
                quarantined_groups.append(quarantine)
                quarantined_task_ids.add(task_id)
                _append_jsonl(self.output_dir / "quarantined_groups.jsonl", quarantine)
            current_window = _WindowPipelineState(
                window_start=window_start,
                window_entries=window_entries,
                window_complete_entries=window_complete_entries,
                collection_errors=collection_errors,
                counterfactual_jobs=counterfactual_jobs,
                circuit_open=circuit_open,
            )
            if self.config.task_scheduling_policy == "frozen_manifest_dynamic":
                complete_by_task = dict(current_window.window_complete_entries)
                logical_windows: list[_WindowPipelineState] = []
                for logical_offset in range(
                    0, len(current_window.window_entries), self.config.task_window
                ):
                    logical_entries = current_window.window_entries[
                        logical_offset : logical_offset + self.config.task_window
                    ]
                    logical_task_ids = {task_id for (task_id, _proposal) in logical_entries}
                    logical_windows.append(
                        _WindowPipelineState(
                            window_start=window_start + logical_offset,
                            window_entries=logical_entries,
                            window_complete_entries=[
                                (task_id, complete_by_task[task_id])
                                for (task_id, _proposal) in logical_entries
                                if task_id in complete_by_task
                            ],
                            collection_errors=[
                                error
                                for error in current_window.collection_errors
                                if str(error.get("task_id", "")) in logical_task_ids
                            ],
                            counterfactual_jobs=[
                                primary
                                for primary in current_window.counterfactual_jobs
                                if primary.proposal.task.task_id in logical_task_ids
                            ],
                            circuit_open=current_window.circuit_open,
                        )
                    )
                for logical_index, logical_window in enumerate(logical_windows):
                    if logical_window.counterfactual_jobs and (not circuit_open):
                        if self.config.pipeline_counterfactuals:
                            _append_jsonl(
                                self.output_dir / "pipeline_events.jsonl",
                                {
                                    "event": "counterfactual_window_started",
                                    "window_index": logical_window.window_start
                                    // max(1, self.config.task_window),
                                    "timestamp": time.time(),
                                    "jobs": len(logical_window.counterfactual_jobs),
                                    "workers": self.config.counterfactual_workers,
                                    "priority": "counterfactual",
                                    "scheduling": "after_dynamic_primary_collection",
                                },
                            )
                        counterfactual_pool = (
                            ThreadRolloutPool(self.config.counterfactual_workers)
                            if self.config.pipeline_counterfactuals
                            else self.rollout_pool
                        )
                        counterfactual_results = self._collect_counterfactual_batch(
                            logical_window.counterfactual_jobs,
                            pool=counterfactual_pool,
                            low_priority=self.config.pipeline_counterfactuals,
                        )
                        self._persist_counterfactual_batch(counterfactual_results)
                        if self.config.pipeline_counterfactuals:
                            _append_jsonl(
                                self.output_dir / "pipeline_events.jsonl",
                                {
                                    "event": "counterfactual_window_completed",
                                    "window_index": logical_window.window_start
                                    // max(1, self.config.task_window),
                                    "timestamp": time.time(),
                                    "jobs": len(logical_window.counterfactual_jobs),
                                    "scheduling": "after_dynamic_primary_collection",
                                },
                            )
                        logical_window.counterfactual_jobs = []
                    collapse_error = self._finalize_window(
                        logical_window,
                        proposals=proposals,
                        rollouts=rollouts,
                        rollouts_by_id=rollouts_by_id,
                        observations=observations,
                        observed_task_ids=observed_task_ids,
                        collapse_windows=collapse_windows,
                        inherited_collapse_streak=inherited_collapse_streak,
                        quarantined_task_ids=quarantined_task_ids,
                    )
                    if collapse_error is not None:
                        for remaining_window in logical_windows[logical_index + 1 :]:
                            for primary in remaining_window.counterfactual_jobs:
                                primary.close()
                        raise collapse_error
                if circuit_open:
                    break
                continue
            if self.config.pipeline_counterfactuals:
                collapse_error = self._finalize_window(
                    current_window,
                    proposals=proposals,
                    rollouts=rollouts,
                    rollouts_by_id=rollouts_by_id,
                    observations=observations,
                    observed_task_ids=observed_task_ids,
                    collapse_windows=collapse_windows,
                    inherited_collapse_streak=inherited_collapse_streak,
                    quarantined_task_ids=quarantined_task_ids,
                )
                deferred_collapse_error = deferred_collapse_error or collapse_error
                if circuit_open or deferred_collapse_error is not None:
                    break
            else:
                if current_window.counterfactual_jobs:
                    counterfactual_results = self._collect_counterfactual_batch(
                        current_window.counterfactual_jobs,
                        pool=self.rollout_pool,
                        low_priority=False,
                    )
                    self._persist_counterfactual_batch(counterfactual_results)
                collapse_error = self._finalize_window(
                    current_window,
                    proposals=proposals,
                    rollouts=rollouts,
                    rollouts_by_id=rollouts_by_id,
                    observations=observations,
                    observed_task_ids=observed_task_ids,
                    collapse_windows=collapse_windows,
                    inherited_collapse_streak=inherited_collapse_streak,
                    quarantined_task_ids=quarantined_task_ids,
                )
                if collapse_error is not None:
                    raise collapse_error
                if circuit_open:
                    break
        if pending_window is not None:
            collapse_error = self._finalize_window(
                pending_window,
                proposals=proposals,
                rollouts=rollouts,
                rollouts_by_id=rollouts_by_id,
                observations=observations,
                observed_task_ids=observed_task_ids,
                collapse_windows=collapse_windows,
                inherited_collapse_streak=inherited_collapse_streak,
                quarantined_task_ids=quarantined_task_ids,
            )
            deferred_collapse_error = deferred_collapse_error or collapse_error
            pending_window = None
        finish_dynamic_counterfactuals(reason="training_batch_boundary", wait_for_completion=True)
        if deferred_collapse_error is not None:
            raise deferred_collapse_error
        if self.config.require_all_proposals and failed_task_ids:
            raise ValueError(
                "fixed-pool cycle has failed proposal slots; resume after correcting them: "
                + ", ".join(failed_task_ids)
            )
        extraction = _proposal_extraction_summary(attempts)
        _write_json(self.output_dir / "proposal_extraction.json", extraction)
        if not proposals:
            raise ValueError(
                "all proposer extraction attempts failed; see proposal_extraction.json"
            )
        if self.primary_probability_observer is not None:
            candidates = [r for r in rollouts_by_id.values() if _rollout_is_training_eligible(r)]
            for rollout in candidates:
                self.primary_probability_observer.submit(rollout.trajectory)
            cache_summary = self.primary_probability_observer.finalize()
            _write_json(self.output_dir / "solver_probability_cache_summary.json", cache_summary)
            checker = getattr(self.primary_probability_observer, "identity_exclusions", None)
            exclusions = checker([r.trajectory for r in candidates]) if checker else {}
            for rollout_id, failures in exclusions.items():
                rollout = rollouts_by_id[rollout_id]
                metadata = rollout.trajectory.metadata
                metadata["training_eligible"] = False
                metadata["training_exclusion_reasons"] = sorted(
                    set(metadata.get("training_exclusion_reasons", ()))
                    | {"frozen_policy_probability_mismatch"}
                )
                metadata["probability_identity_failures"] = failures
                self._persist_counterfactual_metadata(rollout)
                _append_jsonl(
                    self.output_dir / "rollout_training_exclusions.jsonl",
                    _rollout_training_exclusion(rollout),
                )
            _write_json(self.output_dir / "probability_identity_exclusions.json", exclusions)
        selection = None
        if self.config.rollout_group_policy == "eligible_subset":
            (_, durable_rollouts) = self._load()
            rollouts_by_id = {r.trajectory.rollout_id: r for r in durable_rollouts}
            selection = build_training_selection(
                proposals,
                rollouts_by_id,
                self.config.rollouts_per_task,
                schema=INDEPENDENT_SCHEMA if self._independent_frontier else "eligible_subset_v1",
            )
            _write_json(self.output_dir / "training_selection.json", selection)
        eligible_proposals = [
            proposal
            for proposal in proposals
            if proposal.task.task_id not in quarantined_task_ids
            and all(
                (
                    _rollout_id(proposal.task.task_id, rollout_index) in rollouts_by_id
                    and _rollout_is_training_eligible(
                        rollouts_by_id[_rollout_id(proposal.task.task_id, rollout_index)]
                    )
                    for rollout_index in range(self.config.rollouts_per_task)
                )
            )
        ]
        eligible_rollouts = [
            rollouts_by_id[_rollout_id(proposal.task.task_id, rollout_index)]
            for proposal in eligible_proposals
            for rollout_index in range(self.config.rollouts_per_task)
        ]
        if selection is not None:
            selected_tasks = {
                g["task_id"] for g in selection["groups"] if g["solver_group_eligible"]
            }
            eligible_proposals = [p for p in proposals if p.task.task_id in selected_tasks]
            eligible_rollouts = [rollouts_by_id[rid] for rid in selection["selected_rollout_ids"]]
        all_planned_groups_complete = len(eligible_proposals) == len(proposals) and (
            not quarantined_task_ids
        )
        if selection is not None:
            all_planned_groups_complete = all(
                (g["planned_group_complete"] for g in selection["groups"])
            )
        training_ready = len(eligible_proposals) >= self.config.minimum_complete_task_groups and (
            not self.config.require_all_planned_task_groups_for_training
            or all_planned_groups_complete
        )
        incomplete_task_ids = sorted(
            (proposal.task.task_id for proposal in proposals if proposal not in eligible_proposals)
        )
        batch_gate = {
            "status": "ready" if training_ready else "blocked",
            "minimum_complete_task_groups": self.config.minimum_complete_task_groups,
            "require_all_planned_task_groups_for_training": self.config.require_all_planned_task_groups_for_training,
            "planned_task_group_count": len(proposals),
            "all_planned_task_groups_complete": all_planned_groups_complete,
            "complete_task_group_count": len(eligible_proposals),
            "complete_task_ids": [proposal.task.task_id for proposal in eligible_proposals],
            "incomplete_task_ids": incomplete_task_ids,
            "quarantined_task_group_count": len(quarantined_task_ids),
            "quarantined_task_ids": sorted(quarantined_task_ids),
            "training_excluded_rollout_count": sum(
                (
                    int(not _rollout_is_training_eligible(rollout))
                    for rollout in rollouts_by_id.values()
                )
            ),
            "rollouts_per_task": self.config.rollouts_per_task,
            "optimizer_steps": 0,
        }
        if selection is not None:
            batch_gate.update(
                status="ready" if eligible_rollouts else "skipped",
                rollout_group_policy="eligible_subset",
                selected_task_group_count=len(eligible_proposals),
                selected_rollout_count=len(eligible_rollouts),
                eligible_rollout_count=sum(
                    (g["eligible_rollout_count"] for g in selection["groups"])
                ),
                training_excluded_rollout_count=sum(
                    (
                        not row["eligible"] and row["source_sha256"] is not None
                        for g in selection["groups"]
                        for row in g["rollouts"]
                    )
                ),
                complete_task_group_count=sum(
                    (g["planned_group_complete"] for g in selection["groups"])
                ),
                complete_task_ids=[
                    g["task_id"] for g in selection["groups"] if g["planned_group_complete"]
                ],
                incomplete_task_ids=[
                    g["task_id"] for g in selection["groups"] if not g["planned_group_complete"]
                ],
                skip_reason=None if eligible_rollouts else "no_eligible_task_groups",
            )
        _write_json(self.output_dir / "batch_gate.json", batch_gate)
        if circuit_open:
            raise RuntimeError(
                "backend failure circuit opened before the planned cycle was fully collected; preserving independent successes and complete groups is not sufficient to authorize a truncated optimizer update; freshly re-probe routes and exact-resume the missing rollout ids"
            )
        if not training_ready and selection is None:
            if self.config.require_all_planned_task_groups_for_training:
                raise InsufficientCompleteRolloutGroupsError(
                    f"all planned task groups are required before training; complete={len(eligible_proposals)}/{len(proposals)}; incomplete="
                    + ",".join(incomplete_task_ids or ["unknown"])
                    + "; no optimizer update was started"
                )
            raise InsufficientCompleteRolloutGroupsError(
                f"insufficient complete rollout groups after preserving independent successes for safe batch assembly ({len(eligible_proposals)}/{self.config.minimum_complete_task_groups}); no optimizer update was started"
            )
        if self.snapshots.phase.value == "proposer_collection":
            self.snapshots.advance()
        frontier_proposals = eligible_proposals
        if selection is not None:
            complete_ids = {
                g["task_id"] for g in selection["groups"] if g["planned_group_complete"]
            }
            frontier_proposals = [p for p in eligible_proposals if p.task.task_id in complete_ids]
        frontier_rollouts = [
            r
            for r in eligible_rollouts
            if r.trajectory.task_id in {p.task.task_id for p in frontier_proposals}
        ]
        if self._independent_frontier:
            frontier_ids = {g["task_id"] for g in selection["groups"] if g["frontier_rollout_ids"]}
            frontier_proposals = [p for p in proposals if p.task.task_id in frontier_ids]
            frontier_rollouts = [rollouts_by_id[rid] for rid in selection["frontier_rollout_ids"]]
        if frontier_pipeline_executor is None:
            frontier_reverification = self._collect_frontier_reverification(
                frontier_proposals, frontier_rollouts
            )
        else:
            submit_ready_frontier_datasets()
            allowed_ids = {proposal.task.task_id for proposal in frontier_proposals}
            frontier_reverification = {}
            frontier_failures: list[dict[str, Any]] = []
            selection_datasets: dict[str, Any] = {}
            excluded_frontiers: dict[str, Any] = {}
            graph_partial_rows: list[dict[str, Any]] = []
            partial_rows: list[dict[str, Any]] = []
            try:
                for dataset, future in sorted(frontier_dataset_futures.items()):
                    dataset_records = future.result()
                    unexpected = set(dataset_records) - allowed_ids
                    if unexpected:
                        probability_excluded_tasks = {
                            r.trajectory.task_id
                            for r in rollouts_by_id.values()
                            if "frozen_policy_probability_mismatch"
                            in r.trajectory.metadata.get("training_exclusion_reasons", ())
                        }
                        if unexpected - probability_excluded_tasks:
                            raise RuntimeError(
                                "early Frontier selected a task excluded by the final training gate: "
                                + ",".join(sorted(unexpected - probability_excluded_tasks))
                            )
                        _append_jsonl(
                            self.output_dir / "frontier_late_admission_exclusions.jsonl",
                            {
                                "dataset": dataset,
                                "task_ids": sorted(unexpected),
                                "reason": "final_training_admission_excluded_task",
                            },
                        )
                    frontier_reverification.update(
                        {
                            key: value
                            for (key, value) in dataset_records.items()
                            if key in allowed_ids
                        }
                    )
                    dataset_dir = self.output_dir / "frontier_by_dataset" / (dataset or "unknown")
                    selection_path = dataset_dir / "frontier_reverify_selection.json"
                    if selection_path.exists():
                        dataset_selection = json.loads(selection_path.read_text(encoding="utf-8"))
                        selection_datasets.update(dataset_selection.get("datasets", {}))
                        excluded_frontiers.update(dataset_selection.get("excluded_frontiers", {}))
                    frontier_failures.extend(
                        json.loads(
                            (dataset_dir / "frontier_reverification_failures.json").read_text()
                        )
                        if (dataset_dir / "frontier_reverification_failures.json").exists()
                        else []
                    )
                    graph_partial_rows.extend(
                        _read_jsonl(dataset_dir / "frontier_reverification_graph_partial.jsonl")
                    )
                    partial_rows.extend(
                        _read_jsonl(dataset_dir / "frontier_reverification_partial.jsonl")
                    )
            finally:
                frontier_pipeline_executor.shutdown(wait=True, cancel_futures=False)
            _write_json(
                self.output_dir / "frontier_reverify_selection.json",
                {
                    "schema_version": "frontier_reverify_selection_by_dataset_v1",
                    "fraction": self.config.frontier_reverify_fraction,
                    "minimum_frontier": 0.0,
                    "strict_positive": True,
                    "datasets": selection_datasets,
                    "excluded_frontiers": excluded_frontiers,
                },
            )
            _write_json(
                self.output_dir / "frontier_reverification.json",
                [frontier_reverification[key] for key in sorted(frontier_reverification)],
            )
            _write_json(
                self.output_dir / "frontier_reverification_failures.json", frontier_failures
            )
            for path, rows in (
                (
                    self.output_dir / "frontier_reverification_graph_partial.jsonl",
                    graph_partial_rows,
                ),
                (self.output_dir / "frontier_reverification_partial.jsonl", partial_rows),
            ):
                path.unlink(missing_ok=True)
                for row in rows:
                    _append_jsonl(path, row)
            _write_json(
                self.output_dir / "frontier_phase_metrics.json",
                {
                    "duration_s": time.monotonic() - frontier_pipeline_started,
                    "selected_task_count": sum(
                        (
                            len(row.get("selected_task_ids", ()))
                            for row in selection_datasets.values()
                        )
                    ),
                    "completed_task_count": len(frontier_reverification),
                    "excluded_task_count": len(
                        {str(row.get("task_id", "")) for row in frontier_failures} - {""}
                    ),
                    "scheduling": "dataset_early_global_pool",
                    "workers": self.config.counterfactual_workers,
                    "submitted_datasets": sorted(frontier_dataset_futures),
                },
            )
        result = replace(
            assemble_selfplay_result(
                proposals if self._independent_frontier else eligible_proposals,
                eligible_rollouts,
                self.snapshots,
                expected_rollouts_per_task=self.config.rollouts_per_task,
                graph_feature_extractor=self.graph_feature_extractor,
                frontier_reverification=frontier_reverification,
                evaluation_only=self.config.evaluation_only,
                training_selection=selection,
                frontier_evidence=frontier_rollouts if self._independent_frontier else None,
                proposer_baseline=self._proposer_baseline,
            ),
            proposal_extraction=extraction,
        )
        if selection is not None:
            result = replace(result, tasks=tuple((p.task for p in proposals)))
        self._persist_result(result)
        if selection is not None:
            selection["proposer_task_ids"] = [s.task_id for s in result.proposer_batch.samples]
            selection["solver_task_ids"] = [p.task.task_id for p in eligible_proposals]
            selected_ids = set(selection["selected_rollout_ids"])
            credit_rows = _read_jsonl(self.output_dir / "relation_counterfactuals.jsonl")
            selected_credits = [r for r in credit_rows if r.get("rollout_id") in selected_ids]
            (self.output_dir / "training_relation_credits.jsonl").write_text(
                "".join((json.dumps(r, ensure_ascii=False) + "\n" for r in selected_credits))
            )
            _write_json(
                self.output_dir / "credit_selection.json",
                [
                    {
                        "rollout_id": r.get("rollout_id"),
                        "selected": r.get("rollout_id") in selected_ids,
                        "reason": None
                        if r.get("rollout_id") in selected_ids
                        else "rollout_not_selected",
                    }
                    for r in credit_rows
                ],
            )
            batch_gate["solver_status"] = "ready" if result.solver_batch.samples else "skipped"
            batch_gate["proposer_status"] = "ready" if result.proposer_batch.samples else "skipped"
            batch_gate["status"] = (
                "ready"
                if result.proposer_batch.samples or result.solver_batch.samples
                else "skipped"
            )
            batch_gate["skip_reason"] = (
                None if batch_gate["status"] == "ready" else "no_eligible_role_samples"
            )
            _write_json(self.output_dir / "batch_gate.json", batch_gate)
            selection["artifacts_sha256"] = {
                name: hashlib.sha256((self.output_dir / name).read_bytes()).hexdigest()
                for name in (
                    "solver_batch.json",
                    "proposer_batch.json",
                    "frontier_scores.json",
                    "snapshots.json",
                    "training_relation_credits.jsonl",
                )
            }
            selection["source_artifacts_sha256"] = {
                name: hashlib.sha256((self.output_dir / name).read_bytes()).hexdigest()
                for name in (
                    "tasks.jsonl",
                    "solver_rollouts.jsonl",
                    "rollout_metadata_updates.jsonl",
                    "rollout_errors.jsonl",
                    "relation_counterfactuals.jsonl",
                )
                if (self.output_dir / name).exists()
            }
            selection["proposer_exclusions"] = {
                g["task_id"]: next(
                    (
                        p.metadata.get("frontier_training_exclusion")
                        for p in proposals
                        if p.task.task_id == g["task_id"]
                    ),
                    None,
                )
                or g["proposer_exclusion"]
                for g in selection["groups"]
                if g["task_id"] not in selection["proposer_task_ids"]
            }
            selection["postcollection_complete"] = True
            _write_json(self.output_dir / "training_selection.json", selection)
        if self._baseline_store is not None:
            self._baseline_store.commit(
                self._proposer_baseline, result.proposer_batch.metadata["frontier_dataset_means"]
            )
        return result

    def _collect_frontier_reverification(
        self, proposals: list[ProposedTask], rollouts: list[SolverRollout]
    ) -> dict[str, dict[str, Any]]:
        phase_started = time.monotonic()
        override_path = self.output_dir / "frontier_execution_override.json"
        override = json.loads(override_path.read_text()) if override_path.exists() else {}
        allow_current_executor = bool(override.get("allow_current_executor", False))
        fraction = self.config.frontier_reverify_fraction
        if fraction <= 0.0 or not proposals:
            return {}
        groups = group_rollouts_by_task(rollouts)
        candidates: list[tuple[str, str, float]] = []
        provisional: dict[str, float] = {}
        current_bundle = None
        if self.config.canary_exclude_migrated_frontier:
            application = self.application_factory(self.config.base_seed)
            try:
                current_bundle = _executor_compatibility_signature(
                    application.config.model_manifest()
                )
            finally:
                application.close()
        excluded_frontiers: dict[str, Any] = {}
        for proposal in proposals:
            group = groups[proposal.task.task_id]
            rewards = [
                float(item.trajectory.metadata.get("task_reward", item.trajectory.reward))
                for item in group
            ]
            features = (
                self.graph_feature_extractor.extract_many([item.graph for item in group])
                if self.graph_feature_extractor is not None
                else execution_policy_features_many([item.graph for item in group])
            )
            score = graph_local_frontier(
                rewards,
                features,
                validity=1.0,
                normalization=NORMALIZATION
                if getattr(self, "_independent_frontier", False)
                else "legacy",
            )
            signatures = {_rollout_executor_compatibility_signature(item) for item in group}
            if current_bundle is not None and (
                len(signatures) != 1 or (score > 0 and signatures != {current_bundle})
            ):
                proposal.metadata["frontier_training_exclusion"] = "canary_executor_migration"
                excluded_frontiers[proposal.task.task_id] = {
                    "reason": "canary_executor_migration",
                    "provisional_frontier": score,
                    "primary_bundles": sorted(signatures),
                    "current_bundle": current_bundle,
                    "proposer_training_eligible": False,
                }
                continue
            dataset = canonical_dataset_name(proposal.task.metadata.get("dataset", ""))
            provisional[proposal.task.task_id] = score
            candidates.append((proposal.task.task_id, dataset, score))
        selected_by_dataset = select_frontier_reverification(
            candidates, fraction=fraction, minimum_frontier=0.0
        )
        selected_ids = {
            task_id for task_ids in selected_by_dataset.values() for task_id in task_ids
        }
        manifest_datasets: dict[str, Any] = {}
        for dataset in sorted({dataset for (_task, dataset, _score) in candidates}):
            layer = [(task_id, score) for (task_id, key, score) in candidates if key == dataset]
            positive = sorted(
                ((task_id, score) for (task_id, score) in layer if score > 0.0),
                key=lambda item: (-item[1], item[0]),
            )
            selected = selected_by_dataset.get(dataset, ())
            manifest_datasets[dataset] = {
                "eligible_task_count": len(layer),
                "positive_candidate_count": len(positive),
                "quota": math.ceil(fraction * len(layer)),
                "selected_task_ids": list(selected),
                "actual_cutoff": provisional[selected[-1]] if selected else None,
                "tie_break": "task_id",
            }
        selection_manifest = {
            "schema_version": "frontier_reverify_selection_v1",
            "fraction": fraction,
            "minimum_frontier": 0.0,
            "strict_positive": True,
            "datasets": manifest_datasets,
            "excluded_frontiers": excluded_frontiers,
        }
        _write_json(self.output_dir / "frontier_reverify_selection.json", selection_manifest)
        selection_sha256 = hashlib.sha256(
            json.dumps(selection_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        partial_path = self.output_dir / "frontier_reverification_partial.jsonl"
        partial_rows = _read_jsonl(partial_path)
        partial_ids = [str(row.get("task_id", "")) for row in partial_rows]
        if len(set(partial_ids)) != len(partial_ids) or any(
            (task_id not in selected_ids for task_id in partial_ids)
        ):
            raise RuntimeError("Frontier reverify partial journal does not match selection")
        records: dict[str, dict[str, Any]] = {str(row["task_id"]): row for row in partial_rows}
        graph_partial_path = self.output_dir / "frontier_reverification_graph_partial.jsonl"
        graph_partial_rows = _read_jsonl(graph_partial_path)
        graph_partial_keys = [
            (str(row.get("task_id", "")), int(row.get("graph_index", -1)))
            for row in graph_partial_rows
        ]
        if len(set(graph_partial_keys)) != len(graph_partial_keys) or any(
            (task_id not in selected_ids for (task_id, _index) in graph_partial_keys)
        ):
            raise RuntimeError("Frontier graph partial journal does not match selection")
        graph_partial_records = {
            key: row for (key, row) in zip(graph_partial_keys, graph_partial_rows, strict=True)
        }
        proposals_by_id = {proposal.task.task_id: proposal for proposal in proposals}
        pending_graph_futures: dict[Future[dict[str, Any]], tuple[str, int]] = {}
        task_contexts: dict[str, dict[str, Any]] = {}
        graph_results_by_task: dict[str, dict[int, dict[str, Any]]] = {}
        executor = ThreadPoolExecutor(max_workers=self.config.counterfactual_workers)
        nonreplayable_failures: dict[str, list[dict[str, Any]]] = {}
        for task_id in sorted(selected_ids):
            proposal = proposals_by_id[task_id]
            group = groups[task_id]
            if getattr(self, "_independent_frontier", False):
                incomplete = [
                    i
                    for (i, r) in enumerate(group)
                    if not r.graph.output_agent
                    or any((not node.configured for node in r.graph.nodes.values()))
                ]
                if incomplete:
                    proposal.metadata["frontier_training_exclusion"] = (
                        "frontier_reverify_not_replayable"
                    )
                    nonreplayable_failures[task_id] = [
                        dict(
                            task_id=task_id,
                            graph_index=i,
                            error_type="NonReplayablePrimaryGraph",
                            failure_origin="primary_graph_configuration",
                            message="required fixed-graph reverify has no configured output",
                        )
                        for i in incomplete
                    ]
                    continue
            reverify_seed = _stable_execution_seed(
                self.config.base_seed, task_id, phase="frontier_reverify"
            )
            bundle_signatures = {_rollout_executor_compatibility_signature(item) for item in group}
            if len(bundle_signatures) != 1:
                if not allow_current_executor:
                    raise RuntimeError("primary graph group used inconsistent Executor bundles")
                bundle_signatures = {
                    hashlib.sha256(json.dumps(sorted(bundle_signatures)).encode()).hexdigest()
                }
            expected_graph_ids = [
                hashlib.sha256(
                    canonical_graph_key(item.graph, include_prompts=True).encode("utf-8")
                ).hexdigest()
                for item in group
            ]
            existing_record = records.get(task_id)
            if existing_record is not None:
                if (
                    existing_record.get("graph_ids") != expected_graph_ids
                    or len(existing_record.get("rewards", ())) != len(group)
                    or existing_record.get("reverify_executor_seed") != reverify_seed
                    or (
                        not allow_current_executor
                        and existing_record.get("executor_bundle_signature")
                        != next(iter(bundle_signatures))
                    )
                ):
                    raise RuntimeError(
                        "Frontier reverify partial record does not match primary graph group"
                    )
                if existing_record.get("selection_sha256") != selection_sha256:
                    records[task_id] = {**existing_record, "selection_sha256": selection_sha256}
                continue
            journal_lock = threading.Lock()

            def evaluate_one(
                graph_index,
                item,
                *,
                task_id=task_id,
                proposal=proposal,
                reverify_seed=reverify_seed,
                expected_graph_ids=expected_graph_ids,
                bundle_signatures=bundle_signatures,
                journal_lock=journal_lock,
            ):
                existing_graph_record = graph_partial_records.get((task_id, graph_index))
                if existing_graph_record is not None:
                    if (
                        existing_graph_record.get("graph_id") != expected_graph_ids[graph_index]
                        or existing_graph_record.get("reverify_executor_seed") != reverify_seed
                        or (
                            not allow_current_executor
                            and existing_graph_record.get("executor_bundle_signature")
                            != next(iter(bundle_signatures))
                        )
                    ):
                        raise RuntimeError(
                            "Frontier graph partial record does not match primary graph"
                        )
                    return {**existing_graph_record, "selection_sha256": selection_sha256}
                graph_started = time.monotonic()
                graph_semaphore = getattr(self, "_frontier_graph_semaphore", None)
                if graph_semaphore is not None:
                    graph_semaphore.acquire()
                application = None
                try:
                    application = self.application_factory(reverify_seed)
                    application.runtime.seed = reverify_seed
                    current_bundle = _executor_compatibility_signature(
                        application.config.model_manifest()
                    )
                    if current_bundle not in bundle_signatures:
                        override_path = self.output_dir / "frontier_execution_override.json"
                        override = (
                            json.loads(override_path.read_text()) if override_path.exists() else {}
                        )
                        if not override.get("allow_current_executor", False):
                            raise CollectionInfrastructureIncidentError(
                                "Executor bundle changed before Frontier reverify; exclude affected Proposer sample"
                            )
                        with journal_lock:
                            _append_jsonl(
                                self.output_dir / "frontier_executor_changes.jsonl",
                                {
                                    "event": "authorized_current_executor_reverification",
                                    "task_id": task_id,
                                    "graph_index": graph_index,
                                    "original_executor_bundle_signature": next(
                                        iter(bundle_signatures)
                                    ),
                                    "actual_executor_bundle_signature": current_bundle,
                                    "actual_model_manifest": application.config.model_manifest(),
                                    "reason": override.get("reason", ""),
                                    "timestamp": time.time(),
                                },
                            )
                    evaluation_task = TaskSpec(
                        proposal.task.task_id,
                        proposal.task.prompt,
                        reference=proposal.task.reference,
                        task_type=proposal.task.task_type,
                        metadata=dict(proposal.task.metadata),
                        private_verifier_payload=dict(proposal.task.private_verifier_payload),
                    )
                    evaluation_task.metadata["graph_evaluation_phase"] = (
                        f"frontier:{selection_sha256}:{graph_index}"
                    )
                    try:
                        outcome = application.evaluate_graph(
                            evaluation_task,
                            item.graph,
                            seed=reverify_seed,
                            return_verification=True,
                        )
                    except GraphEvaluationBackendError as exc:
                        incident = {
                            "schema_version": "frontier_reverify_incident_v1",
                            "event": "frontier_reverify_backend_failure",
                            "incident_class": "infrastructure",
                            "task_id": task_id,
                            "graph_index": graph_index,
                            "graph_id": expected_graph_ids[graph_index],
                            "selection_sha256": selection_sha256,
                            "backend_failure": exc.failure,
                            "optimizer_steps": 0,
                            "timestamp": time.time(),
                        }
                        with journal_lock:
                            _append_jsonl(
                                self.output_dir / "frontier_reverification_incidents.jsonl",
                                incident,
                            )
                        raise CollectionInfrastructureIncidentError(
                            f"Frontier reverify aborted before optimizer update; task_id={task_id}; graph_index={graph_index}; {exc}"
                        ) from exc
                    if not isinstance(outcome, dict):
                        raise RuntimeError("Frontier reverify did not return verification details")
                    if outcome.get("verification") is None:
                        raise GraphEvaluationIncompleteError(
                            "Frontier reverify has no verified outcome; statistical zero is not a trusted tie"
                        )
                    (reward, _breakdown) = _outcome_task_reward(
                        str(proposal.task.metadata.get("dataset", "")),
                        outcome.get("verification"),
                        prediction=str(outcome.get("prediction", "")),
                    )
                    graph_record = {
                        "schema_version": "frontier_reverify_graph_partial_v1",
                        "duration_s": time.monotonic() - graph_started,
                        "task_id": task_id,
                        "dataset": canonical_dataset_name(
                            proposal.task.metadata.get("dataset", "")
                        ),
                        "graph_index": graph_index,
                        "graph_id": expected_graph_ids[graph_index],
                        "reward": reward,
                        "reward_trusted": True,
                        "reward_source": _breakdown["source"],
                        "actual_executor_bundle_signature": current_bundle,
                        "reverify_executor_seed": reverify_seed,
                        "executor_bundle_signature": next(iter(bundle_signatures)),
                        "selection_sha256": selection_sha256,
                    }
                    return graph_record
                finally:
                    close = getattr(application, "close", None)
                    if callable(close):
                        close()
                    if graph_semaphore is not None:
                        graph_semaphore.release()

            task_contexts[task_id] = {
                "proposal": proposal,
                "group": group,
                "reverify_seed": reverify_seed,
                "executor_bundle_signature": next(iter(bundle_signatures)),
            }
            graph_results_by_task[task_id] = {}
            for index, item in enumerate(group):
                pending_graph_futures[executor.submit(evaluate_one, index, item)] = (task_id, index)
        errors: list[Exception] = []
        infrastructure_failures_by_task: dict[str, list[dict[str, Any]]] = dict(
            nonreplayable_failures
        )
        try:
            for future in as_completed(pending_graph_futures):
                (task_id, index) = pending_graph_futures[future]
                try:
                    graph_record = future.result()
                    graph_results_by_task[task_id][index] = graph_record
                    if (task_id, index) not in graph_partial_records:
                        graph_partial_records[task_id, index] = graph_record
                        _append_jsonl(graph_partial_path, graph_record)
                except (
                    CollectionInfrastructureIncidentError,
                    RequestTokenCreditExceeded,
                    GraphEvaluationIncompleteError,
                ) as exc:
                    infrastructure_failures_by_task.setdefault(task_id, []).append(
                        {
                            "task_id": task_id,
                            "graph_index": index,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                except Exception as exc:
                    errors.append(exc)
                    for pending in pending_graph_futures:
                        if pending is not future:
                            pending.cancel()
            if errors:
                raise errors[0]
        finally:
            executor.shutdown(wait=True, cancel_futures=bool(errors))
        for task_id in sorted(task_contexts):
            context = task_contexts[task_id]
            proposal = context["proposal"]
            group = context["group"]
            if task_id in infrastructure_failures_by_task:
                proposal.metadata["frontier_training_exclusion"] = (
                    "frontier_reverify_infrastructure_failure"
                )
                continue
            graph_results = graph_results_by_task[task_id]
            record = {
                "task_id": task_id,
                "dataset": canonical_dataset_name(proposal.task.metadata.get("dataset", "")),
                "primary_executor_seed": group[0].trajectory.metadata.get("primary_executor_seed"),
                "reverify_executor_seed": context["reverify_seed"],
                "rewards": [float(graph_results[i]["reward"]) for i in range(len(group))],
                "reward_trusted": [
                    graph_results[i].get("reward_trusted") is True for i in range(len(group))
                ],
                "graph_ids": [graph_results[i]["graph_id"] for i in range(len(group))],
                "rollout_ids": [item.trajectory.rollout_id for item in group],
                "executor_bundle_signature": context["executor_bundle_signature"],
                "persistent_mace_updates": False,
                "selection_sha256": selection_sha256,
            }
            records[task_id] = record
            _append_jsonl(partial_path, record)
        _write_json(
            self.output_dir / "frontier_reverification.json",
            [records[task_id] for task_id in sorted(records)],
        )
        _write_json(
            self.output_dir / "frontier_reverification_failures.json",
            [
                failure
                for task_id in sorted(infrastructure_failures_by_task)
                for failure in infrastructure_failures_by_task[task_id]
            ],
        )
        _write_json(
            self.output_dir / "frontier_phase_metrics.json",
            {
                "duration_s": time.monotonic() - phase_started,
                "selected_task_count": len(selected_ids),
                "completed_task_count": len(records),
                "excluded_task_count": len(infrastructure_failures_by_task),
                "scheduling": "global_graph_pool",
                "workers": self.config.counterfactual_workers,
            },
        )
        return records

    def _validate_action_protocol(self) -> None:
        """Never resume old sampled actions or MACE state under the new policy."""
        path = self.output_dir / "action_protocol.json"
        expected = {
            "action_protocol": "director_model_v1",
            "counterfactual_execution": "full_graph_v1",
        }
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise RuntimeError("incompatible action protocol; start a fresh run")
        else:
            if any(
                (
                    (self.output_dir / name).exists()
                    for name in (
                        "solver_rollouts.jsonl",
                        "progress.json",
                        "manifest.json",
                        "mace_snapshots",
                    )
                )
            ):
                raise RuntimeError("legacy run lacks Director SET_MODEL protocol; read-only only")
            _write_json(path, expected)

    def _collect_primary(
        self,
        job: tuple[ProposedTask, int],
        *,
        replacement_attempt: int = 0,
        cancellation_event: threading.Event | None = None,
        total_timeout_s_override: float | None = None,
    ) -> _PrimaryCollection:
        (proposal, rollout_index) = job
        policy_attempt_offset = self.config.policy_sampling_attempt_offsets.get(
            proposal.task.task_id, 0
        )
        policy_sampling_attempt = replacement_attempt + policy_attempt_offset
        rollout_seed = _rollout_sampling_seed(
            self.config.base_seed, proposal.task.task_id, rollout_index, policy_sampling_attempt
        )
        executor_seed = _stable_execution_seed(
            self.config.base_seed, proposal.task.task_id, phase="primary"
        )
        started = time.monotonic()
        (total_timeout_s, request_timeout_s, request_overrides) = self._deadline_profile(proposal)
        if total_timeout_s_override is not None:
            if total_timeout_s_override <= 0:
                raise ValueError("total_timeout_s_override must be positive")
            total_timeout_s = min(total_timeout_s, total_timeout_s_override)
        deadline = RolloutDeadline(
            total_timeout_s=total_timeout_s,
            no_progress_timeout_s=min(self.config.rollout_no_progress_time_s, total_timeout_s),
            request_timeout_s=min(request_timeout_s, total_timeout_s),
            request_timeout_overrides_s=request_overrides,
            started_monotonic=started,
            cancellation_event=cancellation_event,
            exclude_failed_request_time=True,
        )
        application: AdaptiveSolverApplication = self.application_factory(rollout_seed)
        runtime_for_seed = getattr(application, "runtime", None)
        if runtime_for_seed is not None:
            runtime_for_seed.seed = executor_seed
        solve_metadata = dict(proposal.task.metadata)
        dataset_key = canonical_dataset_name(solve_metadata.get("dataset", ""))
        solve_metadata["same_slot_recovery_attempt"] = replacement_attempt
        solve_metadata["admit_stateful_policy_failure_terminal"] = bool(
            dataset_key in {"webshop", "alfworld"}
            and replacement_attempt >= self.config.non_swe_recovery_attempts
        )
        configure_routes = getattr(application, "configure_scoped_worker_routes", None)
        try:
            if (
                self.config.scoped_route_circuit_enabled
                and dataset_key in _SCOPED_ROUTE_CIRCUIT_DATASETS
                and callable(configure_routes)
            ):
                solve_metadata["scoped_route_admission"] = configure_routes(
                    dataset=dataset_key, request_role="primary"
                )
        except BaseException:
            close = getattr(application, "close", None)
            if callable(close):
                close()
            raise
        install_deadline = getattr(application, "set_rollout_deadline", None)
        if callable(install_deadline):
            install_deadline(deadline)
        runtime = getattr(application, "runtime", None)
        set_deadline_context = getattr(
            getattr(runtime, "executor", None), "set_deadline_context", None
        )
        if not callable(install_deadline) and callable(set_deadline_context):
            set_deadline_context(deadline)
        elif not callable(install_deadline):
            set_deadline = getattr(getattr(runtime, "executor", None), "set_deadline", None)
            if callable(set_deadline):
                set_deadline(total_timeout_s)
        try:
            solve_kwargs: dict[str, Any] = {
                "task_id": proposal.task.task_id,
                "task_type": proposal.task.task_type,
                "reference": proposal.task.reference,
                "run_id": f"{proposal.task.task_id}-r{rollout_index}"
                if replacement_attempt == 0
                else f"{proposal.task.task_id}-r{rollout_index}-replacement-{replacement_attempt}",
                "metadata": solve_metadata,
            }
            if proposal.task.private_verifier_payload:
                solve_kwargs["private_verifier_payload"] = proposal.task.private_verifier_payload
            result = application.solve(proposal.task.prompt, **solve_kwargs)
            environment_events = []
            seen_clients = set()
            for tool in getattr(
                getattr(getattr(application, "runtime", None), "executor", None), "tools", {}
            ).values():
                client = getattr(getattr(tool, "lifecycle", None), "client", None)
                if client is not None and id(client) not in seen_clients:
                    seen_clients.add(id(client))
                    environment_events.extend(getattr(client, "retry_events", ()))
            result.task.metadata["environment_request_events"] = environment_events
            result.task.metadata["deadline_accounting"] = deadline.diagnostics()
            backend_failure = result.task.metadata.get("worker_backend_failure")
            if backend_failure:
                raise WorkerBackendUnavailableError(backend_failure)
            duration_s = time.monotonic() - started
            application_config = getattr(application, "config", None)
            director_reward_config = getattr(application_config, "director_reward", None)
            rollout = adaptive_result_to_rollout(
                result,
                self.tokenizer,
                rollout_index=rollout_index,
                seed=rollout_seed,
                max_tokens=self.config.max_tokens,
                duration_s=duration_s,
                reward_version=str(
                    getattr(director_reward_config, "version", PROTOCOL_GATE_REWARD_VERSION)
                ),
            )
            rollout.trajectory.metadata.update(
                {
                    "replacement_attempt": replacement_attempt,
                    "policy_sampling_attempt_offset": policy_attempt_offset,
                    "policy_sampling_attempt": policy_sampling_attempt,
                    "original_rollout_seed": self.config.base_seed + rollout_index,
                    "effective_rollout_seed": rollout_seed,
                    "director_sampling_seed": rollout_seed,
                    "primary_executor_seed": executor_seed,
                    "same_slot_seed_preserved": True,
                    "same_slot_executor_seed_preserved": True,
                    "policy_resampled_on_replacement": policy_sampling_attempt > 0,
                    "deadline_profile": self._deadline_profile_payload(proposal),
                }
            )
            return _PrimaryCollection(
                proposal=proposal,
                rollout_index=rollout_index,
                rollout_seed=rollout_seed,
                executor_seed=executor_seed,
                application=application,
                result=result,
                rollout=rollout,
                primary_duration_s=duration_s,
            )
        except BaseException as exc:
            if isinstance(exc, WorkerWallClockLimitExceeded):
                exc.partial_state = {
                    **(getattr(exc, "partial_state", None) or {}),
                    **self._partial_timeout_state(application, deadline),
                }
            close = getattr(application, "close", None)
            if callable(close):
                close()
            raise

    def _record_scoped_route_event(
        self,
        store: RouteHealthStore | None,
        *,
        proposal: ProposedTask,
        rollout_id: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.scoped_route_circuit_enabled or store is None:
            return {}
        dataset = canonical_dataset_name(proposal.task.metadata.get("dataset", ""))
        if dataset not in _SCOPED_ROUTE_CIRCUIT_DATASETS:
            return {}
        request_role = str(event.get("priority", "primary")).strip().casefold()
        if request_role != "primary":
            return {}
        route = str(event.get("route", "")).strip()
        event_kind = str(event.get("event", ""))
        if not route or event_kind not in {"backend_request_success", "backend_request_failure"}:
            return {}
        success = event_kind == "backend_request_success"
        counts_toward_circuit = bool(event.get("counts_toward_route_circuit", False))
        if (
            self.config.backend_failure_retry_attempts > 0
            and event.get("retryable")
            and (not event.get("disable_route"))
        ):
            counts_toward_circuit = False
        if not success and (not counts_toward_circuit):
            return {}
        state = store.record_scoped_outcome(
            route,
            dataset=dataset,
            request_role=request_role,
            success=success,
            total_elapsed_s=float(event.get("total_elapsed_s", 0.0) or 0.0),
            upstream_elapsed_s=float(event["upstream_elapsed_s"])
            if event.get("upstream_elapsed_s") is not None
            else None,
            slow_threshold_s=self.config.scoped_route_slow_request_s,
            failure_counts_toward_circuit=counts_toward_circuit,
            threshold=self.config.scoped_route_unhealthy_threshold,
            evidence={
                "output_dir": str(self.output_dir),
                "rollout_id": rollout_id,
                "event": event_kind,
                "kind": str(event.get("kind", "")),
                "attempt": int(event.get("attempt", 1) or 1),
            },
        )
        return {
            key: state[key]
            for key in (
                "scope_key",
                "status",
                "consecutive_unhealthy",
                "last_outcome",
                "probe_required",
            )
        }

    def _deadline_profile(self, proposal: ProposedTask) -> tuple[float, float, dict[str, float]]:
        """Return the compact timeout profile selected from public task metadata."""
        dataset = str(proposal.task.metadata.get("dataset", "")).strip().lower()
        if dataset in {"swe_bench", "swe-bench", "swebench"}:
            total_timeout_s = self.config.swe_rollout_wall_time_s
        elif dataset in {"webshop", "alfworld"}:
            total_timeout_s = self.config.stateful_rollout_wall_time_s
        else:
            total_timeout_s = self.config.rollout_wall_time_s
        if dataset == "aime":
            return (
                total_timeout_s,
                self.config.aime_request_wall_time_s,
                {
                    "grok": self.config.grok_aime_request_wall_time_s,
                    **({"": 360.0} if self.config.evaluation_only else {}),
                },
            )
        return (total_timeout_s, self.config.request_wall_time_s, {})

    def _slot_wall_time_s(self, proposal: ProposedTask) -> float:
        dataset = canonical_dataset_name(proposal.task.metadata.get("dataset", ""))
        if dataset == "swe_bench":
            return self.config.swe_slot_wall_time_s
        if dataset in {"webshop", "alfworld"}:
            return self.config.stateful_slot_wall_time_s
        return self.config.rollout_slot_wall_time_s

    def _deadline_profile_payload(self, proposal: ProposedTask) -> dict[str, Any]:
        (total_timeout_s, request_timeout_s, overrides) = self._deadline_profile(proposal)
        return {
            "dataset": str(proposal.task.metadata.get("dataset", "")),
            "total_timeout_s": total_timeout_s,
            "control_idle_timeout_s": min(self.config.rollout_no_progress_time_s, total_timeout_s),
            "request_timeout_s": min(request_timeout_s, total_timeout_s),
            "request_timeout_overrides_s": dict(overrides),
        }

    def _rollout_attempt_event(
        self,
        proposal: ProposedTask,
        rollout_index: int,
        *,
        replacement_attempt: int,
        error: Exception | None = None,
        outcome: _PrimaryCollection | None = None,
        accepted_as_primary: bool | None = None,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        original_seed = self.config.base_seed + rollout_index
        policy_attempt_offset = self.config.policy_sampling_attempt_offsets.get(
            proposal.task.task_id, 0
        )
        policy_sampling_attempt = replacement_attempt + policy_attempt_offset
        seed = (
            outcome.rollout_seed
            if outcome is not None
            else _rollout_sampling_seed(
                self.config.base_seed, proposal.task.task_id, rollout_index, policy_sampling_attempt
            )
        )
        outcome_eligible = bool(
            outcome is not None and _rollout_is_training_eligible(outcome.rollout)
        )
        event: dict[str, Any] = {
            "task_id": proposal.task.task_id,
            "rollout_id": _rollout_id(proposal.task.task_id, rollout_index),
            "rollout_index": rollout_index,
            "replacement_attempt": replacement_attempt,
            "policy_sampling_attempt_offset": policy_attempt_offset,
            "policy_sampling_attempt": policy_sampling_attempt,
            "seed": seed,
            "original_rollout_seed": original_seed,
            "primary_executor_seed": _stable_execution_seed(
                self.config.base_seed, proposal.task.task_id, phase="primary"
            ),
            "success": error is None and (outcome is None or outcome_eligible),
            "accepted_as_primary": bool(accepted_as_primary)
            if accepted_as_primary is not None
            else error is None and outcome_eligible,
            "same_slot_seed_preserved": True,
            "same_slot_executor_seed_preserved": True,
            "policy_resampled_on_replacement": policy_sampling_attempt > 0,
            "deadline_profile": self._deadline_profile_payload(proposal),
        }
        if outcome is not None:
            event["duration_s"] = outcome.primary_duration_s
            event["training_eligible"] = outcome_eligible
            event["training_exclusion_reasons"] = list(
                outcome.rollout.trajectory.metadata.get("training_exclusion_reasons", ())
            )
            if not bool(event["accepted_as_primary"]):
                event["trajectory"] = outcome.rollout.trajectory.to_dict()
        elif duration_s is not None:
            event["duration_s"] = max(0.0, float(duration_s))
        if error is not None:
            event.update({"error_type": type(error).__name__, "message": str(error)})
            timeout_diagnostics = getattr(error, "to_dict", None)
            if callable(timeout_diagnostics):
                event["timeout"] = timeout_diagnostics()
            partial_state = getattr(error, "partial_state", None)
            if isinstance(partial_state, dict):
                event["partial_state"] = partial_state
        return event

    @staticmethod
    def _partial_timeout_state(
        application: AdaptiveSolverApplication, deadline: RolloutDeadline
    ) -> dict[str, Any]:
        solver = getattr(application, "solver", None)
        canvas = getattr(solver, "active_canvas", None)
        if canvas is None:
            return {"deadline": deadline.diagnostics()}
        return {
            "deadline": deadline.diagnostics(),
            "canvas_state": canvas.state.value,
            "round_index": canvas.round_index,
            "total_worker_tokens": canvas.total_tokens,
            "pending_agent_id": canvas.pending_agent_id,
            "dirty_agents": sorted(canvas.dirty_agents),
            "graph": canvas.graph.to_dict(),
            "history": [step.to_dict() for step in canvas.history],
        }

    def _prepare_counterfactual(self, primary: _PrimaryCollection) -> None:
        rollout = primary.rollout
        try:
            check = getattr(primary.application, "supports_graph_counterfactual", None)
            capability = (
                check(primary.proposal.task)
                if callable(check)
                else {"supported": False, "reason": "full_graph_capability_unavailable"}
            )
            rollout.trajectory.metadata["graph_counterfactual_capability"] = capability
            action_spans = rollout.trajectory.metadata.get(
                "relation_choice_token_spans",
                rollout.trajectory.metadata.get("action_token_spans", ()),
            )
            primary.decisions = tuple(
                schedule_relation_decisions(
                    primary.result.solver_result.trace,
                    limit=self.config.counterfactuals_per_rollout,
                    seed=primary.rollout_seed,
                    action_token_spans=action_spans,
                )
                if capability["supported"] and self.config.counterfactuals_per_rollout > 0
                else ()
            )
            rollout.trajectory.metadata["relation_counterfactual_candidate_count"] = len(
                primary.decisions
            )
            if not primary.decisions:
                rollout.trajectory.metadata["relation_counterfactual_skip_reason"] = (
                    "disabled_by_configuration"
                    if self.config.counterfactuals_per_rollout <= 0
                    else capability["reason"]
                    if not capability["supported"]
                    else "no_auditable_effective_relation_choice"
                )
        except Exception as exc:
            rollout.trajectory.metadata["relation_counterfactual_candidate_count"] = 0
            rollout.trajectory.metadata.setdefault("relation_counterfactual_errors", []).append(
                {"stage": "schedule", "error_type": type(exc).__name__, "message": str(exc)}
            )
            primary.decisions = ()

    def _collect_counterfactual(
        self, primary: _PrimaryCollection
    ) -> tuple[SolverRollout, list[dict[str, Any]]]:
        rollout = replace(
            primary.rollout,
            trajectory=replace(
                primary.rollout.trajectory,
                metadata=copy.deepcopy(primary.rollout.trajectory.metadata),
            ),
        )
        proposal = primary.proposal
        action_spans = rollout.trajectory.metadata.get(
            "relation_choice_token_spans", rollout.trajectory.metadata.get("action_token_spans", ())
        )
        credits: list[dict[str, Any]] = []
        counterfactual_errors = list(
            rollout.trajectory.metadata.get("relation_counterfactual_errors", ())
        )
        (_primary_timeout_s, request_timeout_s, request_overrides) = self._deadline_profile(
            proposal
        )
        pair_budget = self.config.counterfactual_pair_wall_time_s
        pair_deadline = RolloutDeadline(
            exclude_failed_request_time=True,
            absolute_wall_timeout_s=pair_budget,
            total_timeout_s=pair_budget,
            no_progress_timeout_s=min(self.config.rollout_no_progress_time_s, pair_budget),
            request_timeout_s=min(request_timeout_s, pair_budget),
            request_timeout_overrides_s={
                route: min(value, pair_budget) for (route, value) in request_overrides.items()
            },
            cancellation_event=primary.counterfactual_cancellation_event,
            cancellation_reason="counterfactual_collection_aborted",
        )
        branch_audits: list[dict[str, Any]] = []
        rollout.trajectory.metadata["relation_counterfactual_execution_mode"] = "full_graph_v1"
        rollout.trajectory.metadata["relation_counterfactual_budget"] = {
            "scope": "independent_off_on_pair",
            "pair_wall_budget_s": pair_budget,
            "primary_duration_deducted": False,
        }

        def evaluate_branch(graph: MultiAgentGraph, same_seed: int) -> float:
            pair_deadline.check("counterfactual_branch_start")
            branch = self.application_factory(same_seed)
            started = time.monotonic()
            branch_name = "off" if len(branch_audits) % 2 == 0 else "on"
            try:
                expected_bundle = _rollout_executor_compatibility_signature(rollout)
                actual_bundle = _executor_compatibility_signature(branch.config.model_manifest())
                if expected_bundle and expected_bundle != actual_bundle:
                    raise RuntimeError(
                        "Executor bundle changed between primary and full graph branch"
                    )
                branch.set_rollout_deadline(pair_deadline)
                score = float(
                    branch.evaluate_graph(copy.deepcopy(proposal.task), graph, seed=same_seed)
                )
                branch_audits.append(
                    {
                        **getattr(branch, "last_graph_evaluation", {}),
                        "score": score,
                        "pair_wall_budget_s": pair_budget,
                        "branch": branch_name,
                        "completed": True,
                        "duration_s": time.monotonic() - started,
                        "pair_deadline": pair_deadline.diagnostics(),
                    }
                )
                return score
            except Exception as exc:
                branch_audits.append(
                    {
                        **getattr(branch, "last_graph_evaluation", {}),
                        "branch": branch_name,
                        "completed": False,
                        "pair_wall_budget_s": pair_budget,
                        "duration_s": time.monotonic() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "request_budget": getattr(exc, "budget", {}),
                        "request_events": getattr(exc, "request_events", []),
                        "pair_deadline": pair_deadline.diagnostics(),
                    }
                )
                raise
            finally:
                branch.close()

        try:
            if pair_budget <= 0.5:
                rollout.trajectory.metadata["relation_counterfactual_skip_reason"] = (
                    "insufficient_paired_probe_budget"
                )
                return (rollout, [])
            for decision in primary.decisions:
                try:
                    credit = evaluate_relation_decision(
                        decision,
                        rollout_id=rollout.trajectory.rollout_id,
                        seed=primary.executor_seed,
                        action_token_span=action_spans[decision.action_index],
                        evaluate=evaluate_branch,
                    )
                except Exception as exc:
                    counterfactual_errors.append(
                        {
                            "action_index": decision.action_index,
                            "source": decision.source,
                            "target": decision.target,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    continue
                if credit:
                    credits.append(credit.to_dict())
            rollout.trajectory.metadata["relation_counterfactual_branches"] = branch_audits
            rollout.trajectory.metadata["relation_counterfactual_success_count"] = len(credits)
            if counterfactual_errors:
                rollout.trajectory.metadata["relation_counterfactual_errors"] = (
                    counterfactual_errors
                )
            return (rollout, credits)
        finally:
            primary.close()

    def _collect_counterfactual_batch(
        self,
        primaries: list[_PrimaryCollection],
        *,
        pool: RolloutPool[Any, Any],
        low_priority: bool,
    ) -> list[tuple[_PrimaryCollection, Any, Exception | None]]:

        def collect(primary: _PrimaryCollection):
            try:
                if low_priority:
                    with request_priority("counterfactual"):
                        outcome = self._collect_counterfactual(primary)
                else:
                    outcome = self._collect_counterfactual(primary)
                return (primary, outcome, None)
            except Exception as exc:
                return (primary, None, exc)

        iter_map = getattr(pool, "iter_map", None)
        results = (
            iter_map(collect, primaries) if callable(iter_map) else pool.map(collect, primaries)
        )
        return list(results)

    def _persist_counterfactual_batch(
        self, results: list[tuple[_PrimaryCollection, Any, Exception | None]]
    ) -> None:
        for primary, outcome, error in results:
            rollout_id = primary.rollout.trajectory.rollout_id
            if error is not None:
                event = {
                    "rollout_id": rollout_id,
                    "stage": "relation_counterfactual",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                primary.rollout.trajectory.metadata.setdefault(
                    "relation_counterfactual_errors", []
                ).append(event)
                _append_jsonl(self.output_dir / "relation_counterfactual_errors.jsonl", event)
                primary.close()
            else:
                assert outcome is not None
                (rollout, credit_payloads) = outcome
                outcome_metadata = dict(rollout.trajectory.metadata)
                primary.rollout.trajectory.metadata.clear()
                primary.rollout.trajectory.metadata.update(outcome_metadata)
                for payload in credit_payloads:
                    _append_jsonl(self.output_dir / "relation_counterfactuals.jsonl", payload)
                for event in rollout.trajectory.metadata.get("relation_counterfactual_errors", ()):
                    _append_jsonl(
                        self.output_dir / "relation_counterfactual_errors.jsonl",
                        {"rollout_id": rollout_id, **event},
                    )
            self._persist_counterfactual_metadata(primary.rollout)

    def _collect(self, job: tuple[ProposedTask, int]) -> tuple[SolverRollout, list[dict[str, Any]]]:
        """Compatibility helper for direct unit callers; run() uses two durable phases."""
        primary = self._collect_primary(job)
        self._prepare_counterfactual(primary)
        return self._collect_counterfactual(primary)

    def _persist_counterfactual_metadata(self, rollout: SolverRollout) -> None:
        _append_jsonl(
            self.output_dir / "rollout_metadata_updates.jsonl",
            {"rollout_id": rollout.trajectory.rollout_id, "metadata": rollout.trajectory.metadata},
        )

    def _write_progress(
        self,
        *,
        proposals: list[ProposedTask],
        rollouts_by_id: dict[str, SolverRollout],
        failed_rollout_ids: list[str] | None = None,
        quarantined_task_ids: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "completed_tasks": sum(
                (
                    int(
                        all(
                            (
                                (rollout_id := _rollout_id(proposal.task.task_id, rollout_index))
                                in rollouts_by_id
                                and _rollout_is_training_eligible(rollouts_by_id[rollout_id])
                                for rollout_index in range(self.config.rollouts_per_task)
                            )
                        )
                    )
                    for proposal in proposals
                )
            ),
            "proposed_tasks": len(proposals),
            "rollouts": len(rollouts_by_id),
            "training_eligible_rollouts": sum(
                (int(_rollout_is_training_eligible(rollout)) for rollout in rollouts_by_id.values())
            ),
            "training_excluded_rollouts": sum(
                (
                    int(not _rollout_is_training_eligible(rollout))
                    for rollout in rollouts_by_id.values()
                )
            ),
        }
        if failed_rollout_ids:
            payload["failed_rollout_ids"] = failed_rollout_ids
        if quarantined_task_ids:
            payload["quarantined_task_ids"] = quarantined_task_ids
            payload["quarantined_task_groups"] = len(quarantined_task_ids)
        _write_json(self.output_dir / "progress.json", payload)

    def _finalize_window(
        self,
        state: _WindowPipelineState,
        *,
        proposals: list[ProposedTask],
        rollouts: list[SolverRollout],
        rollouts_by_id: dict[str, SolverRollout],
        observations: list[dict[str, Any]],
        observed_task_ids: set[str],
        collapse_windows: list[dict[str, Any]],
        inherited_collapse_streak: int,
        quarantined_task_ids: set[str],
    ) -> RuntimeError | None:
        if state.counterfactual_future is not None:
            results = state.counterfactual_future.result()
            self._persist_counterfactual_batch(results)
            _append_jsonl(
                self.output_dir / "pipeline_events.jsonl",
                {
                    "event": "counterfactual_window_completed",
                    "window_index": state.window_start // max(1, self.config.task_window),
                    "timestamp": time.time(),
                    "jobs": len(state.counterfactual_jobs),
                },
            )
        pending_observations: list[tuple[ProposedTask, list[float]]] = []
        for task_id, proposal in state.window_complete_entries:
            expected_ids = {
                _rollout_id(task_id, rollout_index)
                for rollout_index in range(self.config.rollouts_per_task)
            }
            if task_id in observed_task_ids or not expected_ids <= rollouts_by_id.keys():
                continue
            task_rewards = [
                rollouts_by_id[_rollout_id(task_id, rollout_index)].trajectory.reward
                for rollout_index in range(self.config.rollouts_per_task)
            ]
            pending_observations.append((proposal, task_rewards))
        observe_many = getattr(self.proposer, "observe_many", None)
        observe = getattr(self.proposer, "observe", None)
        if pending_observations and callable(observe_many):
            observe_many(pending_observations)
        elif callable(observe):
            for proposal, task_rewards in pending_observations:
                observe(proposal, task_rewards)
        for proposal, task_rewards in pending_observations:
            task_id = proposal.task.task_id
            observation = {
                "task_id": task_id,
                "pool_id": proposal.metadata.get("pool_id"),
                "rewards": task_rewards,
            }
            observations.append(observation)
            observed_task_ids.add(task_id)
            _append_jsonl(self.output_dir / "curriculum_observations.jsonl", observation)
        state_dict = getattr(self.proposer, "state_dict", None)
        if callable(state_dict):
            _write_json(self.output_dir / "curriculum_state.json", state_dict())
        self._write_progress(
            proposals=proposals,
            rollouts_by_id=rollouts_by_id,
            failed_rollout_ids=[item["rollout_id"] for item in state.collection_errors],
            quarantined_task_ids=sorted(quarantined_task_ids),
        )
        collapse_event = self._record_collapse_window(
            window_start=state.window_start,
            window_entries=state.window_complete_entries,
            rollouts_by_id=rollouts_by_id,
            prior_events=collapse_windows,
            inherited_streak=inherited_collapse_streak,
        )
        collapse_error = None
        if collapse_event is not None:
            collapse_windows.append(collapse_event)
            _append_jsonl(self.output_dir / "collapse_monitor.jsonl", collapse_event)
        if self.config.pipeline_counterfactuals:
            _append_jsonl(
                self.output_dir / "pipeline_events.jsonl",
                {
                    "event": "window_finalized",
                    "window_index": state.window_start // max(1, self.config.task_window),
                    "timestamp": time.time(),
                    "observations": len(pending_observations),
                },
            )
        return collapse_error

    def _record_collapse_window(
        self,
        *,
        window_start: int,
        window_entries: list[tuple[str, ProposedTask]],
        rollouts_by_id: dict[str, SolverRollout],
        prior_events: list[dict[str, Any]],
        inherited_streak: int = 0,
    ) -> dict[str, Any] | None:
        task_ids = [task_id for (task_id, _proposal) in window_entries]
        window_key = ",".join(task_ids)
        if not task_ids or any((event.get("window_key") == window_key for event in prior_events)):
            return None
        groups: list[list[SolverRollout]] = []
        for task_id in task_ids:
            rollout_ids = [
                _rollout_id(task_id, index) for index in range(self.config.rollouts_per_task)
            ]
            if not all((rollout_id in rollouts_by_id for rollout_id in rollout_ids)):
                return None
            groups.append([rollouts_by_id[rollout_id] for rollout_id in rollout_ids])
        flat = [item for group in groups for item in group]
        single_agent_rate = sum((len(item.graph.nodes) == 1 for item in flat)) / len(flat)
        within_ratios = [
            len({canonical_graph_key(item.graph) for item in group}) / len(group)
            for group in groups
        ]
        within_unique = sum(within_ratios) / len(within_ratios)
        task_relation_rates = [
            sum(
                (
                    bool(item.graph.directed_edges or item.graph.bidirectional_edges)
                    for item in group
                )
            )
            / len(group)
            for group in groups
        ]
        max_task_relation_rate = max(task_relation_rates)
        relation_rate = sum(
            (bool(item.graph.directed_edges or item.graph.bidirectional_edges) for item in flat)
        ) / len(flat)
        disconnected_multi_agent_rate = sum(
            (
                len(item.graph.nodes) > 1
                and (not (item.graph.directed_edges or item.graph.bidirectional_edges))
                for item in flat
            )
        ) / len(flat)
        counterfactual_eligible_rate = sum(
            (
                int(item.trajectory.metadata.get("relation_counterfactual_candidate_count", 0)) > 0
                for item in flat
            )
        ) / len(flat)
        legacy_alert = (
            self.config.structural_exploration_policy == "stratified"
            and single_agent_rate >= self.config.collapse_single_agent_threshold
            and (within_unique <= self.config.collapse_unique_graph_threshold)
            and (max_task_relation_rate <= self.config.collapse_task_relation_threshold)
        )
        disconnected_alert = (
            disconnected_multi_agent_rate > self.config.collapse_disconnected_multi_agent_threshold
        )
        alert_reasons = []
        if legacy_alert:
            alert_reasons.append("legacy_low_diversity")
        if disconnected_alert:
            alert_reasons.append("disconnected_multi_agent")
        alert = bool(alert_reasons)
        prior_streak = (
            int(prior_events[-1].get("consecutive_alert_windows", 0))
            if prior_events and prior_events[-1].get("alert")
            else inherited_streak
            if not prior_events
            else 0
        )
        return {
            "window_index": window_start // max(1, self.config.task_window),
            "window_key": window_key,
            "task_ids": task_ids,
            "rollouts": len(flat),
            "single_agent_rate": single_agent_rate,
            "within_task_unique_graph_ratio": within_unique,
            "relation_graph_rate": relation_rate,
            "disconnected_multi_agent_rate": disconnected_multi_agent_rate,
            "task_relation_rates": dict(zip(task_ids, task_relation_rates, strict=True)),
            "max_task_relation_rate": max_task_relation_rate,
            "counterfactual_eligible_rate": counterfactual_eligible_rate,
            "alert": alert,
            "alert_reasons": alert_reasons,
            "consecutive_alert_windows": prior_streak + 1 if alert else 0,
            "thresholds": {
                "structural_exploration_policy": self.config.structural_exploration_policy,
                "single_agent_rate": self.config.collapse_single_agent_threshold,
                "within_task_unique_graph_ratio": self.config.collapse_unique_graph_threshold,
                "max_task_relation_rate": self.config.collapse_task_relation_threshold,
                "disconnected_multi_agent_rate": self.config.collapse_disconnected_multi_agent_threshold,
                "patience_windows": self.config.collapse_patience_windows,
            },
        }

    def _load(self) -> tuple[list[ProposedTask], list[SolverRollout]]:
        proposals = [
            _proposal_from_dict(item) for item in _read_jsonl(self.output_dir / "tasks.jsonl")
        ]
        rehydrate = getattr(self.proposer, "rehydrate_private_payload", None)
        if callable(rehydrate):
            proposals = [rehydrate(proposal) for proposal in proposals]
        trajectories_by_id: dict[str, TokenizedDirectorTrajectory] = {}
        for item in _read_jsonl(self.output_dir / "solver_rollouts.jsonl"):
            trajectory = TokenizedDirectorTrajectory(
                rollout_id=str(item["rollout_id"]),
                task_id=str(item["task_id"]),
                token_ids=tuple(item["token_ids"]),
                action_mask=tuple(item["action_mask"]),
                reward=float(item["reward"]),
                graph=dict(item["graph"]),
                seed=int(item.get("seed", 0)),
                executor_version=str(item.get("executor_version", "adaptive-v1")),
                metadata=dict(item.get("metadata", {})),
                policy_calls=tuple(
                    (
                        TokenizedPolicyCall(
                            call_id=str(call["call_id"]),
                            token_ids=tuple(call["token_ids"]),
                            action_mask=tuple(call["action_mask"]),
                            behavior_log_probs=tuple(call.get("behavior_log_probs", ())),
                            action_token_span=tuple(call["action_token_span"])
                            if call.get("action_token_span") is not None
                            else None,
                            relation_token_span=tuple(call["relation_token_span"])
                            if call.get("relation_token_span") is not None
                            else None,
                            metadata=dict(call.get("metadata", {})),
                        )
                        for call in item.get("policy_calls", ())
                    )
                ),
            )
            existing = trajectories_by_id.get(trajectory.rollout_id)
            if existing is not None:
                if existing.to_dict() != trajectory.to_dict():
                    raise ValueError(f"conflicting duplicate rollout_id: {trajectory.rollout_id}")
                continue
            trajectories_by_id[trajectory.rollout_id] = trajectory
        for update in _read_jsonl(self.output_dir / "rollout_metadata_updates.jsonl"):
            rollout_id = str(update.get("rollout_id", ""))
            existing = trajectories_by_id.get(rollout_id)
            if existing is None:
                continue
            trajectories_by_id[rollout_id] = replace(
                existing, metadata=dict(update.get("metadata", existing.metadata))
            )
        return (
            proposals,
            [
                SolverRollout(item, MultiAgentGraph.from_dict(item.graph))
                for item in trajectories_by_id.values()
            ],
        )

    def _persist_result(self, result: DryRunSelfPlayResult) -> None:
        _write_json(
            self.output_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "trainable_roles": ["proposer", "solver"],
                "optimizer_steps": 0,
                "rollouts_per_task": self.config.rollouts_per_task,
                "rollout_group_policy": self.config.rollout_group_policy,
                "proposer_learning_mode": INDEPENDENT_SCHEMA
                if getattr(self, "_independent_frontier", False)
                else "legacy",
                "frontier_normalization": NORMALIZATION
                if getattr(self, "_independent_frontier", False)
                else "legacy",
                "frontier_reward_source": "solver_task_reward",
                "minimum_complete_task_groups": self.config.minimum_complete_task_groups,
                "proposals_per_seed": self.config.proposals_per_seed,
                "task_window": self.config.task_window,
                "rollout_workers": self.config.workers,
                "long_tail_control": {
                    "deadline_profiles_seconds": {
                        "stateless": self.config.rollout_wall_time_s,
                        "swe": self.config.swe_rollout_wall_time_s,
                        "webshop_alfworld": self.config.stateful_rollout_wall_time_s,
                    },
                    "rollout_wall_time_s": self.config.rollout_wall_time_s,
                    "stateful_rollout_wall_time_s": self.config.stateful_rollout_wall_time_s,
                    "swe_rollout_wall_time_s": self.config.swe_rollout_wall_time_s,
                    "swe_slot_wall_time_s": self.config.swe_slot_wall_time_s,
                    "rollout_slot_wall_time_s": self.config.rollout_slot_wall_time_s,
                    "stateful_slot_wall_time_s": self.config.stateful_slot_wall_time_s,
                    "rollout_no_progress_time_s": self.config.rollout_no_progress_time_s,
                    "request_wall_time_s": self.config.request_wall_time_s,
                    "aime_request_wall_time_s": self.config.aime_request_wall_time_s,
                    "grok_aime_request_wall_time_s": self.config.grok_aime_request_wall_time_s,
                    "pause_no_progress_during_active_request": True,
                    "replacement_rollouts_per_task": self.config.replacement_rollouts_per_task,
                    "swe_non_trainable_recovery_attempts": self.config.swe_non_trainable_recovery_attempts,
                    "non_swe_recovery_attempts": self.config.non_swe_recovery_attempts,
                    "backend_circuit_semantics": "consecutive_failures",
                    "backend_failure_route_threshold": self.config.backend_failure_route_threshold,
                    "scoped_route_circuit": {
                        "enabled": self.config.scoped_route_circuit_enabled,
                        "scope": "route_x_dataset_x_request_role",
                        "datasets": sorted(_SCOPED_ROUTE_CIRCUIT_DATASETS),
                        "unhealthy_threshold": self.config.scoped_route_unhealthy_threshold,
                        "slow_request_s": self.config.scoped_route_slow_request_s,
                        "recovery": "bounded_complex_worker_probe",
                    },
                },
                "rollout_batching": {
                    "mode": "independent_requests_continuous_server_batch",
                    "task_scheduling_policy": self.config.task_scheduling_policy,
                    "primary_job_order": self.config.primary_job_order,
                    "primary_duration_estimate_version": PRIMARY_DURATION_ESTIMATE_VERSION,
                    "max_active_task_groups": self.config.max_active_task_groups
                    if self.config.task_scheduling_policy == "frozen_manifest_dynamic"
                    else self.config.task_window,
                    "curriculum_observation_order": "logical_manifest_windows",
                    "logical_window_size": self.config.task_window * self.config.rollouts_per_task,
                    "max_active_rollouts": min(
                        self.config.workers, self.config.task_window * self.config.rollouts_per_task
                    ),
                    "incomplete_group_policy": "quarantine_without_exact_resume",
                    "training_eligibility_policy": "valid_finished_execution_only",
                    **(
                        {
                            "counterfactual_pipeline": "global_queue_as_each_primary_rollout_completes",
                            "counterfactual_workers": self.config.counterfactual_workers,
                            "counterfactual_pair_wall_time_s": self.config.counterfactual_pair_wall_time_s,
                            "counterfactual_pair_budget": "independent_from_primary_shared_by_off_on",
                            "request_priority": "primary_before_queued_counterfactual",
                            "final_training_barrier": "wait_for_bounded_counterfactual_completion",
                        }
                        if self.config.pipeline_counterfactuals
                        else {}
                    ),
                },
                "graph_diversity_bonus": self.config.graph_diversity_bonus,
                "frontier_reverification": {
                    "schema_version": "frontier_reverify_selection_v1",
                    "fraction": self.config.frontier_reverify_fraction,
                    "minimum_frontier": 0.0,
                    "strict_positive": True,
                    "dataset_stratified": True,
                    "group_scope": "complete_fixed_graph_group",
                    "aggregation": "same_pair_sign_gate_mean_delta_squared",
                    "persistent_mace_updates": False,
                },
                "director_reward_versions": sorted(
                    {
                        str(sample.metadata.get("director_reward_version", LEGACY_REWARD_VERSION))
                        for sample in result.solver_batch.samples
                    }
                ),
                "structural_exploration_policies": sorted(
                    {
                        str(
                            sample.metadata.get("model_roles", {})
                            .get("canvas_execution", {})
                            .get("structural_exploration_policy", "legacy_unspecified")
                        )
                        for sample in result.solver_batch.samples
                    }
                ),
                "collapse_monitor": {
                    "single_agent_threshold": self.config.collapse_single_agent_threshold,
                    "unique_graph_threshold": self.config.collapse_unique_graph_threshold,
                    "task_relation_threshold": self.config.collapse_task_relation_threshold,
                    "disconnected_multi_agent_threshold": self.config.collapse_disconnected_multi_agent_threshold,
                    "patience_windows": self.config.collapse_patience_windows,
                },
            },
        )
        _write_json(
            self.output_dir / "frontier_scores.json", [asdict(x) for x in result.frontier_scores]
        )
        _write_json(self.output_dir / "proposer_batch.json", result.proposer_batch.to_dict())
        _write_json(self.output_dir / "solver_batch.json", result.solver_batch.to_dict())
        _write_json(self.output_dir / "snapshots.json", result.snapshots)


def _proposal_dict(value: ProposedTask) -> dict[str, Any]:
    return {
        "task": task_to_public_dict(value.task),
        "response": value.response,
        "token_ids": list(value.token_ids),
        "action_mask": list(value.action_mask),
        "metadata": value.metadata,
        "policy_calls": [asdict(call) for call in value.policy_calls],
    }


def _rollout_id(task_id: str, rollout_index: int) -> str:
    return f"{task_id}-r{rollout_index}"


def _proposal_from_dict(value: dict[str, Any]) -> ProposedTask:
    return ProposedTask(
        TaskSpec(**value["task"]),
        str(value["response"]),
        tuple(value["token_ids"]),
        tuple(value["action_mask"]),
        dict(value.get("metadata", {})),
        tuple(
            (
                TokenizedPolicyCall(
                    call_id=str(call["call_id"]),
                    token_ids=tuple(call["token_ids"]),
                    action_mask=tuple(call["action_mask"]),
                    behavior_log_probs=tuple(call.get("behavior_log_probs", ())),
                    action_token_span=tuple(call["action_token_span"])
                    if call.get("action_token_span") is not None
                    else None,
                    relation_token_span=tuple(call["relation_token_span"])
                    if call.get("relation_token_span") is not None
                    else None,
                    metadata=dict(call.get("metadata", {})),
                )
                for call in value.get("policy_calls", ())
            )
        ),
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    with _JSONL_APPEND_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _proposal_failure_kind(detail: str) -> str:
    normalized = detail.casefold()
    if "json" in normalized:
        return "json_parse_failure"
    if "prompt" in normalized:
        return "missing_prompt"
    if "reference" in normalized or "answer" in normalized:
        return "missing_reference"
    return "format_or_validation_failure"


def _proposal_extraction_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum((int(bool(item.get("success"))) for item in attempts))
    failure_kinds: dict[str, int] = {}
    for item in attempts:
        kind = item.get("failure_kind")
        if not item.get("success") and kind:
            failure_kinds[str(kind)] = failure_kinds.get(str(kind), 0) + 1
    return {
        "attempts": len(attempts),
        "successes": successes,
        "failures": len(attempts) - successes,
        "success_rate": successes / max(1, len(attempts)),
        "failure_kinds": failure_kinds,
    }
