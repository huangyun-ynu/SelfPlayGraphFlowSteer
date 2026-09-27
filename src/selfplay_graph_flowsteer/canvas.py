from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .actions import (
    ACTION_FIELDS,
    DIRECTOR_ACTION_PROTOCOL_VERSION,
    ActionParser,
    ActionType,
    CanvasAction,
    PromptRevisionBasis,
)
from .aime_submission import is_aime_dataset, parse_aime_answer
from .artifact_protocol import WORKER_PROTOCOL_STATUS_VERSION, summarize_worker_protocol
from .config import CanvasConfig
from .contracts import ExecutionReport, RelationType
from .dataset_actions import DatasetActionAdapter
from .deadline import RolloutDeadline
from .delegation import (
    DUPLICATE_RESPONSIBILITY_POLICIES,
    DelegationCompilation,
    DelegationIssue,
    DelegationValidationError,
    compare_responsibilities,
    compile_delegation,
    delegation_task_alignment_issue,
    responsibility_signature,
)
from .graph import GraphValidationError, MultiAgentGraph, MutationResult
from .output_contract import OUTPUT_CONTRACT_VERSION, worker_output_role_changes_input
from .qa_submission import is_short_qa_dataset
from .runtime import (
    SWE_SHARED_TOKEN_BUDGET,
    WORKER_BACKEND_FAILURE_SENTINEL,
    WORKER_PROTOCOL_FAILURE_SENTINEL,
    MultiAgentRuntime,
    _enforce_artifact_integrity,
    artifact_backend_failure_records,
    artifact_integrity_failure_risks,
)
from .submission_contract import (
    SUBMISSION_CONTRACT_VERSION,
    DirectorCallContext,
    SubmissionReceipt,
    _issue_receipt,
    answer_hash,
    is_text_submission_dataset,
    snapshot_hash,
)
from .webshop_budget import budget_partition, observed_request_bounds


def _merge_submission_reports(first, later):
    """Keep all real execution usage when bounded recovery precedes submission."""
    import copy
    from dataclasses import fields

    reports = []
    for report in [first, *later]:
        if report is not None and all(report is not prior for prior in reports):
            reports.append(report)
    if not reports:
        return None
    merged = ExecutionReport()
    for report in reports:
        for item in fields(ExecutionReport):
            value = getattr(report, item.name)
            current = getattr(merged, item.name)
            if isinstance(current, dict):
                current.update(copy.deepcopy(value))
            elif isinstance(current, list):
                current.extend(copy.deepcopy(value))
            elif isinstance(current, int):
                setattr(merged, item.name, current + value)
            else:
                setattr(merged, item.name, value)
    return merged


class CanvasState(StrEnum):
    """FlowSteer's edit state with an explicit prompt-completion barrier."""

    BUILDING = "building"
    AWAITING_PROMPT = "awaiting_prompt"
    AWAITING_MODEL = "awaiting_model"
    AWAITING_RELATION_CHOICE = "awaiting_relation_choice"
    FINISHED = "finished"
    FAILED = "failed"


class TokenBudgetExceeded(RuntimeError):
    """Terminal aggregate-cost failure retaining the committed execution report."""

    def __init__(self, report: ExecutionReport, total: int, limit: int) -> None:
        self.report = report
        self.total = int(total)
        self.limit = int(limit)
        super().__init__(f"token budget exceeded ({self.total}/{self.limit})")


class RequiredWorkerBackendFailure(RuntimeError):
    """Terminal Worker outage retaining the execution that exposed it."""

    def __init__(self, report: ExecutionReport) -> None:
        self.report = report
        failures = [
            artifact
            for artifact in report.artifacts.values()
            if artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
        ]
        self.agents = tuple(sorted(artifact.agent_id for artifact in failures))
        self.routes = tuple(sorted({artifact.model_route or "unassigned" for artifact in failures}))
        self.failure_details = tuple(
            record for artifact in failures for record in artifact_backend_failure_records(artifact)
        )
        self.failure_types = tuple(
            sorted({str(record.get("kind", "unknown")) for record in self.failure_details})
        )
        super().__init__(
            "required Worker backend unavailable "
            f"(agents={','.join(self.agents) or 'unknown'}; "
            f"routes={','.join(self.routes) or 'unknown'}; "
            f"failure_types={','.join(self.failure_types) or 'unknown'})"
        )


@dataclass
class CanvasStep:
    round_index: int
    action: CanvasAction
    accepted: bool
    active: bool
    feedback: str
    graph: dict[str, Any]
    dirty_agents: list[str] = field(default_factory=list)
    invalidated_agents: list[str] = field(default_factory=list)
    scheduled_agents: list[str] = field(default_factory=list)
    executed_agents: list[str] = field(default_factory=list)
    reused_agents: list[str] = field(default_factory=list)
    remaining_dirty: list[str] = field(default_factory=list)
    invalidation_reasons: dict[str, list[str]] = field(default_factory=dict)
    prompt_revision: dict[str, Any] = field(default_factory=dict)
    execution: ExecutionReport | None = None
    rejection_code: str | None = None
    protocol_recovery: bool = False
    final_execution: bool = False
    topology_audit: dict[str, Any] = field(default_factory=dict)
    structural_repair: dict[str, Any] = field(default_factory=dict)
    responsibility_issue: dict[str, Any] = field(default_factory=dict)
    responsibility_overlap_check: dict[str, Any] = field(default_factory=dict)
    delegation_field_repairs: list[dict[str, Any]] = field(default_factory=list)
    time_admission: dict[str, Any] = field(default_factory=dict)
    token_admission: dict[str, Any] = field(default_factory=dict)
    control_snapshot: dict[str, Any] = field(default_factory=dict)
    rejection_details: dict[str, Any] = field(default_factory=dict)
    relation_decision: dict[str, Any] = field(default_factory=dict)
    invalid_repeat_count: int = 0
    topology_edits_frozen: bool = False
    event_id: str = ""
    director_call_id: str = ""
    submission_receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "event_id": self.event_id,
            "director_call_id": self.director_call_id,
            "submission_receipt": self.submission_receipt,
            "action": self.action.to_dict(),
            "accepted": self.accepted,
            "active": self.active,
            "feedback": self.feedback,
            "graph": self.graph,
            "dirty_agents": list(self.dirty_agents),
            "invalidated_agents": list(self.invalidated_agents),
            "scheduled_agents": list(self.scheduled_agents),
            "executed_agents": list(self.executed_agents),
            "reused_agents": list(self.reused_agents),
            "remaining_dirty": list(self.remaining_dirty),
            "invalidation_reasons": {
                key: list(value) for key, value in self.invalidation_reasons.items()
            },
            "prompt_revision": dict(self.prompt_revision),
            "execution": self.execution.to_dict() if self.execution else None,
            "rejection_code": self.rejection_code,
            "protocol_recovery": self.protocol_recovery,
            "final_execution": self.final_execution,
            "topology_audit": dict(self.topology_audit),
            "structural_repair": dict(self.structural_repair),
            "responsibility_issue": dict(self.responsibility_issue),
            "responsibility_overlap_check": dict(self.responsibility_overlap_check),
            "delegation_field_repairs": list(self.delegation_field_repairs),
            "time_admission": dict(self.time_admission),
            "token_admission": dict(self.token_admission),
            "control_snapshot": dict(self.control_snapshot),
            "rejection_details": dict(self.rejection_details),
            "relation_decision": dict(self.relation_decision),
            "invalid_repeat_count": self.invalid_repeat_count,
            "topology_edits_frozen": self.topology_edits_frozen,
        }


@dataclass(frozen=True)
class RepairProgressSignature:
    """Version-independent measure of whether a repair became easier to finish."""

    reason: str
    output_health: str
    final_error_codes: tuple[str, ...]
    unconfigured_count: int
    unreachable_count: int
    dirty_count: int
    final_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "output_health": self.output_health,
            "final_error_codes": list(self.final_error_codes),
            "unconfigured_count": self.unconfigured_count,
            "unreachable_count": self.unreachable_count,
            "dirty_count": self.dirty_count,
            "final_ready": self.final_ready,
        }


@dataclass(frozen=True)
class PendingRelationDecision:
    source: str
    target: str
    relation_type: RelationType
    graph_version: int
    proposal_round_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "graph_version": self.graph_version,
            "proposal_round_index": self.proposal_round_index,
        }


@dataclass(frozen=True)
class RuntimeRecoveryReport:
    """Non-policy execution recovery; never serialized as a Director action."""

    accepted: bool
    execution: ExecutionReport
    protocol_recovery: bool = True


from .unified_submission import UnifiedSubmissionMixin
from .unified_contract import PROTOCOL, RESULT_SCOPES


class GraphCanvas(UnifiedSubmissionMixin):
    """Transactional graph editor plus factual execution feedback.

    Invalid mutations never alter the accepted graph. Like FlowSteer, adding an
    agent enters a mandatory AWAITING_PROMPT state so empty placeholder agents
    cannot accumulate on the canvas.
    """

    def __init__(
        self,
        *,
        task: str,
        worker_task: str | None = None,
        director_task: str | None = None,
        runtime: MultiAgentRuntime,
        config: CanvasConfig | None = None,
        parser: ActionParser | None = None,
        runtime_routes: tuple[str, ...] = (),
        task_type: str = "general",
        model_router: None = None,
        action_adapter: DatasetActionAdapter | None = None,
        dataset: str = "",
        duplicate_responsibility_policy: str = "record_only",
        managed_delegation_contracts: bool = True,
        allow_legacy_prompts: bool = False,
        rollout_deadline: RolloutDeadline | None = None,
        structural_exploration_required: bool = False,
        binary_relation_policy: bool = False,
        run_id: str = "",
        prepare_text_submission: Callable[[Any], Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.prepare_text_submission = prepare_text_submission
        self.submission_receipt: SubmissionReceipt | None = None
        self._director_call_context: DirectorCallContext | None = None
        self._submission_recovery_reports: list[ExecutionReport] = []
        self._text_recovery_attempted = False
        self.task = task
        self.worker_task = worker_task if worker_task is not None else task
        self.director_task = director_task if director_task is not None else task
        self.runtime = runtime
        self.config = config or CanvasConfig()
        self.parser = parser or ActionParser(unified=self.unified)
        self._init_unified()
        self.runtime_routes = tuple(runtime_routes)
        self.task_type = str(task_type)
        if model_router is not None:
            raise ValueError("MACE is retired; Director must use SET_MODEL")
        self.action_adapter = action_adapter
        inferred_dataset = (
            action_adapter.datasets[0]
            if action_adapter is not None and len(action_adapter.datasets) == 1
            else ""
        )
        self.dataset = str(dataset or inferred_dataset).strip().casefold()
        self.duplicate_responsibility_policy = (
            str(duplicate_responsibility_policy).strip().casefold()
        )
        if self.duplicate_responsibility_policy not in DUPLICATE_RESPONSIBILITY_POLICIES:
            raise ValueError(
                "duplicate_responsibility_policy must be one of: "
                + ", ".join(sorted(DUPLICATE_RESPONSIBILITY_POLICIES))
            )
        self.managed_delegation_contracts = bool(managed_delegation_contracts)
        self.allow_legacy_prompts = allow_legacy_prompts
        self.rollout_deadline = rollout_deadline
        self.structural_exploration_required = bool(structural_exploration_required)
        self.binary_relation_policy = bool(binary_relation_policy)
        self.structural_exploration_waived = False
        self.graph = MultiAgentGraph(
            max_agents=self.config.max_agents, runtime_routes=self.runtime_routes, submission_protocol=self.config.submission_protocol
        )
        self.state = CanvasState.BUILDING
        self.pending_agent_id: str | None = None
        self.pending_relation_decision: PendingRelationDecision | None = None
        self._last_relation_decision: dict[str, Any] = {}
        self._considered_relation_pairs: set[tuple[int, tuple[str, str]]] = set()
        self.dirty_agents: set[str] = set()
        self.dirty_reasons: dict[str, set[str]] = {}
        self._step_invalidated_agents: set[str] = set()
        self._step_scheduled_agents: set[str] = set()
        self._step_invalidation_reasons: dict[str, set[str]] = {}
        self._prompt_revision_checkpoints: dict[str, dict[str, Any]] = {}
        self._prompt_revision_consumed: dict[str, set[str]] = {}
        self.prompt_revision_counts: dict[str, int] = {}
        self.prompt_revision_history: list[dict[str, Any]] = []
        self._step_prompt_revision: dict[str, Any] = {}
        self._step_responsibility_overlap_check: dict[str, Any] = {}
        self._duplicate_responsibility_warnings: set[tuple[int, str, str]] = set()
        self.round_index = 0
        self.total_tokens = 0
        self.history: list[CanvasStep] = []
        self.structural_repair_reason: str | None = None
        self.structural_repair_agents: tuple[str, ...] = ()
        self.structural_repair_pair: tuple[str, str] | None = None
        self.structural_repair_relation: str | None = None
        self.structural_repair_entries = 0
        self.structural_repair_resolutions = 0
        self.structural_repair_blocked_actions = 0
        self.finish_reachability_rejections = 0
        self.relation_layer_rejections = 0
        self.duplicate_agent_rejections = 0
        self.time_budget_rejections = 0
        self.token_budget_admission_rejections = 0
        self._time_admission_event: dict[str, Any] = {}
        self._token_admission_event: dict[str, Any] = {}
        self._repair_transition: str | None = None
        # Rejection fusion is keyed by the authoritative Canvas state, not by
        # the model-authored action. Different invalid actions against the
        # same unchanged state are one semantic no-progress streak.
        self._last_invalid_signature: str | None = None
        self._invalid_repeat_count = 0
        self.topology_edits_frozen = False
        self.structural_exploration_waived = False
        self.repair_epoch = 0
        self.semantic_no_progress_streak = 0
        self.semantic_no_progress_recovery_count = 0
        self.output_switches_without_progress = 0
        self.output_lifecycle_recovery_count = 0
        self._last_repair_signature: RepairProgressSignature | None = None
        self._repair_entry_pending = False
        self._recent_repair_actions: list[dict[str, Any]] = []
        # Connectivity-repair state is tracked separately from output-lifecycle
        # recovery. Its history survives intermediate structural reasons (for
        # example a layer edit before the relation is added).
        self._connectivity_repair_active = False
        self._connectivity_repair_states: set[str] = set()
        self._connectivity_cycle_blocks = 0
        self._consolidation_output_locked = False
        self._output_selection_budget_remaining = self.config.output_selection_budget
        self._output_incumbent: dict[str, Any] = {}

    @property
    def director_round_limit(self) -> int | None:
        return None if self.config.director_budget_policy == "edits_v1" else self.config.max_rounds

    @property
    def active(self) -> bool:
        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return False
        return self.director_round_limit is None or self.round_index < self.director_round_limit

    def reset(self) -> None:
        self._init_unified()
        self.submission_receipt = None
        self._director_call_context = None
        self._submission_recovery_reports = []
        self._text_recovery_attempted = False
        self.graph = MultiAgentGraph(
            max_agents=self.config.max_agents, runtime_routes=self.runtime_routes, submission_protocol=self.config.submission_protocol
        )
        self.runtime.reset()
        self.state = CanvasState.BUILDING
        self.pending_agent_id = None
        self.pending_relation_decision = None
        self._last_relation_decision = {}
        self._considered_relation_pairs.clear()
        self.dirty_agents.clear()
        self.dirty_reasons.clear()
        self._step_invalidated_agents.clear()
        self._step_scheduled_agents.clear()
        self._step_invalidation_reasons.clear()
        self._prompt_revision_checkpoints.clear()
        self._prompt_revision_consumed.clear()
        self.prompt_revision_counts.clear()
        self.prompt_revision_history.clear()
        self._step_prompt_revision.clear()
        self._step_responsibility_overlap_check.clear()
        self.round_index = 0
        self.total_tokens = 0
        self.history.clear()
        self.structural_repair_reason = None
        self.structural_repair_agents = ()
        self.structural_repair_pair = None
        self.structural_repair_relation = None
        self.structural_repair_entries = 0
        self.structural_repair_resolutions = 0
        self.structural_repair_blocked_actions = 0
        self.finish_reachability_rejections = 0
        self.relation_layer_rejections = 0
        self.duplicate_agent_rejections = 0
        self.time_budget_rejections = 0
        self.token_budget_admission_rejections = 0
        self._time_admission_event = {}
        self._token_admission_event = {}
        self._repair_transition = None
        self._last_invalid_signature = None
        self._invalid_repeat_count = 0
        self.topology_edits_frozen = False
        self.repair_epoch = 0
        self.semantic_no_progress_streak = 0
        self.semantic_no_progress_recovery_count = 0
        self.output_switches_without_progress = 0
        self.output_lifecycle_recovery_count = 0
        self._last_repair_signature = None
        self._repair_entry_pending = False
        self._recent_repair_actions = []
        self._connectivity_repair_active = False
        self._connectivity_repair_states.clear()
        self._connectivity_cycle_blocks = 0
        self._consolidation_output_locked = False
        self._output_selection_budget_remaining = self.config.output_selection_budget
        self._output_incumbent = {}

    def step(
        self,
        raw_action: str | CanvasAction,
        *,
        count_round: bool = True,
        authoritative_director: bool = False,
        director_context: DirectorCallContext | None = None,
    ) -> CanvasStep:
        self._director_call_context = director_context
        self._submission_recovery_reports = []
        self._time_admission_event = {}
        self._token_admission_event = {}
        self._step_invalidated_agents = set()
        self._step_scheduled_agents = set()
        self._step_invalidation_reasons = {}
        self._step_prompt_revision = {}
        self._step_responsibility_overlap_check = {}
        action = (
            raw_action if isinstance(raw_action, CanvasAction) else self.parser.parse(raw_action)
        )
        if self.unified:
            self._restore_submission_lock()
        if self.rollout_deadline is not None:
            self.rollout_deadline.check("canvas_step_start")
        # Internal deterministic recovery actions do not consume Director
        # rounds and must remain possible after the model-turn budget is
        # exhausted. Terminal Canvas states are never reopened.
        if self.state in {CanvasState.FINISHED, CanvasState.FAILED} or (
            count_round and self.director_round_limit is not None
            and self.round_index >= self.director_round_limit
        ):
            return self._record(
                action,
                accepted=False,
                feedback="Canvas is not active.",
                rejection_code="canvas_inactive",
            )

        if count_round:
            self.round_index += 1
        if (
            self.runtime_routes
            and action.action_type is ActionType.ADD_AGENT
            and self.director_round_limit is not None
            and self.config.max_rounds - self.round_index < (3 if self.unified else 4)
        ):
            return self._record(
                action,
                accepted=False,
                feedback="Insufficient Director turns for prompt, model, output and finish.",
                rejection_code="configuration_turn_budget_exhausted",
            )
        if not action.valid:
            if action.parse_error_code:
                return self._record(
                    action, accepted=False,
                    feedback=f"Rejected action encoding: {action.parse_error}",
                    rejection_code=action.parse_error_code,
                    rejection_details={"action_protocol_error": dict(action.parse_error_details)},
                )
            awaiting = self.state is CanvasState.AWAITING_PROMPT
            return self._record(
                action,
                accepted=False,
                feedback=f"Rejected action: {action.parse_error}",
                rejection_code=("responsibility_violation" if awaiting else "invalid_action"),
                responsibility_issue=(
                    {
                        "code": "prompt_action_schema",
                        "field": None,
                        "message": str(action.parse_error or "invalid action"),
                    }
                    if awaiting
                    else None
                ),
            )
        if self.unified:
            if action.action_type is ActionType.SET_OUTPUT:
                return self._reject_graph_action(action, code="retired_action", message="Use result_scope in SET_PROMPT and FINISH(target).")
            if action.action_type is ActionType.SET_PROMPT and action.result_scope not in RESULT_SCOPES:
                return self._reject_graph_action(action, code="result_scope_required", message="SET_PROMPT requires result_scope: subtask or task_result.")
            if action.action_type in {ActionType.RUN_AGENT, ActionType.FINISH} and not action.target:
                return self._reject_graph_action(action, code="target_required", message="An explicit target is required.")
            edit_rejection = self._director_edit_admission(action)
            if edit_rejection is not None:
                return edit_rejection
        if authoritative_director:
            # The Director chooses graph semantics, not concurrency metadata.
            # External Canvas callers retain the normal optimistic-version
            # check; internal Director actions always bind to current state.
            action.expected_version = self.graph.version
        if action.expected_version is not None and action.expected_version != self.graph.version:
            return self._reject_graph_action(
                action,
                code="stale_canvas_version",
                message=(
                    f"expected Canvas version {action.expected_version}, but current version is "
                    f"{self.graph.version}"
                ),
            )
        # FINISH may submit an existing result while abandoning an isolated draft.
        # Admission, provenance and transaction checks still run in _unified_finish.
        if (self.unified and action.action_type is ActionType.FINISH
                and self._isolated_submission_draft(str(action.target))):
            return self._unified_finish(action)
        pending_output_binding = bool(
            self.state is CanvasState.AWAITING_MODEL
            and action.action_type is ActionType.SET_OUTPUT
            and action.target == self.pending_agent_id
            and self.pending_agent_id in self.graph.nodes
            and self.graph.nodes[self.pending_agent_id].prompt_configured
            and not self._uses_staged_environment_commit()
        )
        if self.state is CanvasState.AWAITING_MODEL and not (
            action.action_type is ActionType.SET_MODEL and action.target == self.pending_agent_id
        ) and not pending_output_binding and not (
            self.unified and action.action_type is ActionType.DELETE_AGENT and action.target == self.pending_agent_id
        ):
            return self._record(
                action,
                accepted=False,
                feedback=f"{self.pending_agent_id} requires SET_MODEL before another graph edit.",
                rejection_code="model_selection_required",
            )
        if self.state is CanvasState.AWAITING_RELATION_CHOICE:
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: the pending relation proposal requires one constrained "
                    "off/on policy choice before another graph action."
                ),
                rejection_code="relation_choice_required",
                relation_decision=(
                    self.pending_relation_decision.to_dict()
                    if self.pending_relation_decision is not None
                    else {}
                ),
            )
        commit_ready_agents = (
            set(self.runtime.environment_commit_ready_agents())
            if self._uses_staged_environment_commit()
            else set()
        )
        staged_environment_cleanup = bool(
            commit_ready_agents
            and action.action_type is ActionType.DELETE_AGENT
            and str(action.target or action.agent_id)
            in set(self._staged_commit_safe_delete_targets())
        )
        environment_owner_agents = (
            set(self.runtime.environment_owner_agents())
            if self._uses_staged_environment_commit()
            else set()
        )
        if (
            action.action_type is ActionType.DELETE_AGENT
            and str(action.target or action.agent_id) in environment_owner_agents
        ):
            return self._reject_graph_action(
                action,
                code="webshop_environment_owner_immutable",
                message=(
                    "the sole WebShop environment owner cannot be deleted or replaced within "
                    "one official episode; revise or select that owner for bounded output closure"
                ),
            )
        if (
            commit_ready_agents
            and action.action_type is not ActionType.SET_OUTPUT
            and not staged_environment_cleanup
        ):
            return self._reject_graph_action(
                action,
                code="webshop_commit_ready_selection_required",
                message=(
                    "a staged WebShop purchase is ready; select a commit-ready Agent as output "
                    "or remove a non-contributing Agent exposed by Canvas. The trusted staged "
                    "transaction is latched and cannot be revised or discarded"
                ),
            )
        frozen_recovery_actions = {ActionType.SET_OUTPUT, ActionType.FINISH}
        if ActionType.DELETE_AGENT in self._structural_repair_allowed_action_types():
            frozen_recovery_actions.add(ActionType.DELETE_AGENT)
        if self.topology_edits_frozen and self.pending_agent_id:
            frozen_recovery_actions.add(ActionType.DELETE_AGENT)
        if self.unified:
            frozen_recovery_actions.update({ActionType.RUN_AGENT, ActionType.SET_PROMPT, ActionType.SET_MODEL, ActionType.SET_LAYER, ActionType.CONSIDER_RELATION, ActionType.SET_RELATION, ActionType.REMOVE_RELATION, ActionType.DELETE_AGENT})
        if self.topology_edits_frozen and action.action_type not in frozen_recovery_actions:
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: topology edits are frozen after repeated invalid "
                    "mutations; use one of the bounded recovery actions exposed "
                    "by the control snapshot."
                ),
                rejection_code="topology_edits_frozen",
                rejection_details=self._rejection_details("topology_edits_frozen"),
            )
        discards_pending_agent = bool(
            (self.topology_edits_frozen or self.unified)
            and self.state is CanvasState.AWAITING_PROMPT
            and action.action_type is ActionType.DELETE_AGENT
            and str(action.target or action.agent_id) == self.pending_agent_id
        )
        if (
            self.state is CanvasState.AWAITING_PROMPT
            and not self._completes_prompt(action)
            and not discards_pending_agent
            and not (
                commit_ready_agents
                and action.action_type in {ActionType.SET_OUTPUT, ActionType.DELETE_AGENT}
            )
        ):
            if (
                action.action_type is ActionType.SET_PROMPT
                and action.target not in self.graph.nodes
            ):
                return self._reject_graph_action(
                    action,
                    code="unknown_agent",
                    message=(
                        f"unknown agent: {action.target}; the pending runtime-allocated Agent is "
                        f"{self.pending_agent_id}"
                    ),
                )
            return self._record(
                action,
                accepted=False,
                feedback=(
                    f"Rejected action: {self.pending_agent_id} requires SET_PROMPT before "
                    "another graph edit."
                ),
                rejection_code="responsibility_violation",
                responsibility_issue={
                    "code": "prompt_order",
                    "field": None,
                    "message": (
                        f"{self.pending_agent_id} requires SET_PROMPT before another graph edit"
                    ),
                },
            )
        if (
            self.structural_exploration_required
            and not self.structural_exploration_waived
            and action.action_type in {ActionType.SET_OUTPUT, ActionType.FINISH}
            and not self._structural_exploration_satisfied()
            and not commit_ready_agents
        ):
            if self.structural_repair_reason in {
                "token_budget_consolidation",
                "time_budget_consolidation",
            }:
                self.structural_exploration_waived = True
            else:
                return self._record(
                    action,
                    accepted=False,
                    feedback=(
                        "Rejected action: STRUCTURAL_EXPLORATION_REQUIRED. This rollout "
                        "must test a compact connected multi-Agent graph before selecting "
                        "the output or finishing. Add and configure a second task-relevant "
                        "Agent, then establish a legal directed or bidirectional relation. "
                        "Roles and topology remain your task-conditioned choice."
                    ),
                    rejection_code="structural_exploration_required",
                    rejection_details=self._rejection_details("structural_exploration_required"),
                )
        growth_admission = self._new_agent_time_admission(action)
        if growth_admission is not None:
            return self._reject_for_time_admission(action, growth_admission)
        if (
            self.config.structural_repair_enabled
            and action.action_type is ActionType.ADD_AGENT
            and self.graph.nodes
            and self.config.max_total_tokens - self.total_tokens
            < self.config.graph_growth_token_reserve
        ):
            self._enter_structural_repair(
                "token_budget_consolidation",
                agents=tuple(self.graph.nodes),
            )
            self.structural_repair_blocked_actions += 1
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: STRUCTURAL_REPAIR_REQUIRED "
                    "(reason=token_budget_consolidation; remaining Worker tokens "
                    f"{self.config.max_total_tokens - self.total_tokens} are reserved "
                    "for selecting an existing output and FINISH)"
                ),
                rejection_code="token_budget_consolidation_required",
            )
        token_growth_admission = self._new_agent_token_admission(action)
        if token_growth_admission is not None:
            return self._reject_for_token_admission(action, token_growth_admission)
        if (
            action.action_type is ActionType.SET_OUTPUT
            and action.target in self.graph.nodes
            and self._connectivity_repair_active
            and self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}
            and not (
                self._uses_staged_environment_commit()
                and self.runtime.environment_commit_ready_agents()
            )
        ):
            cycle = self._output_switch_cycle(str(action.target))
            if cycle["blocked_as_cycle"]:
                return self._reject_graph_action(
                    action,
                    code="connectivity_output_cycle",
                    message=(
                        f"SET_OUTPUT to {action.target} returns to a previously visited graph/output "
                        "state and does not improve connectivity. Current actual edges and "
                        "unreachable Agents are in graph_state. Continue with a legal layer or "
                        "relation edit, or choose an output that improves reachability."
                    ),
                    rejection_details={"cycle_preview": cycle},
                )
        repair_error = None if self.unified else self._structural_repair_gate_error(action)
        if repair_error is not None:
            self.structural_repair_blocked_actions += 1
            return self._reject_graph_action(
                action,
                code="structural_repair_required",
                message=repair_error,
            )
        is_prompt_revision = bool(
            action.action_type is ActionType.SET_PROMPT
            and str(action.target) in self.graph.nodes
            and self.graph.nodes[str(action.target)].configured
        )
        prompt_revision_evidence: dict[str, Any] | None = None
        if authoritative_director and not is_prompt_revision:
            legality_error = self._director_action_legality_error(action)
            if legality_error is not None:
                code, message = legality_error
                return self._reject_graph_action(action, code=code, message=message)
        if action.action_type is ActionType.CONSIDER_RELATION:
            return self._begin_relation_decision(action)
        if action.action_type is ActionType.RUN_AGENT and self.unified:
            return self._unified_run(action)
        if action.action_type is ActionType.FINISH:
            return self._unified_finish(action) if self.unified else self._finish(action)
        if (
            action.action_type is ActionType.SET_OUTPUT
            and str(action.target) == self.graph.output_agent
        ):
            return self._reject_graph_action(
                action,
                code="output_already_selected",
                message=(
                    f"{action.target} is already the output Agent; choose a graph-changing "
                    "action or FINISH"
                ),
            )
        staged_environment_commit = bool(
            action.action_type is ActionType.SET_OUTPUT and self._uses_staged_environment_commit()
        )
        webshop_ready_agents = set(self.runtime.environment_commit_ready_agents())
        webshop_owner_agents = set(self.runtime.environment_owner_agents())
        webshop_closure_pass = bool(
            staged_environment_commit
            and str(action.target) in webshop_owner_agents
            and str(action.target) not in webshop_ready_agents
            and self._artifact_is_usable_output(str(action.target))
        )
        if staged_environment_commit and not (
            str(action.target) in webshop_ready_agents or webshop_closure_pass
        ):
            return self._reject_graph_action(
                action,
                code="webshop_output_not_commit_ready",
                message=(
                    f"{action.target} is not the mutable WebShop episode owner and has no "
                    "trusted staged purchase"
                ),
            )

        graph_before = self.graph.to_dict()
        graph_before.pop("version", None)
        candidate = self.graph.clone()
        before = set(candidate.nodes)
        rejection_code: str | None = None
        compilation: DelegationCompilation | None = None
        responsibility_issue: dict[str, Any] = {}
        try:
            if action.action_type is ActionType.SET_PROMPT:
                compilation = self._validate_delegation(action)
            mutation = self._apply(candidate, action, compilation=compilation)
            if not mutation.changed:
                field_repairs = (
                    [repair.to_dict() for repair in compilation.field_repairs]
                    if compilation is not None
                    else []
                )
                return self._reject_graph_action(
                    action,
                    code="no_effective_change",
                    message=(
                        f"{mutation.message}; the Canvas version and Worker inputs "
                        "were left unchanged"
                    ),
                    rejection_details={
                        "change_reason": mutation.change_reason,
                        "mutation_changed": False,
                    },
                    delegation_field_repairs=field_repairs,
                )
            if is_prompt_revision and not self.unified:
                prompt_revision_evidence, revision_error = self._validate_prompt_revision_evidence(
                    action
                )
                if revision_error is not None:
                    return self._reject_graph_action(
                        action,
                        code="prompt_revision_evidence_required",
                        message=revision_error,
                        rejection_details={
                            "target": str(action.target),
                            "eligible_revision_evidence": (
                                self._eligible_prompt_revision_evidence(str(action.target))[
                                    "public"
                                ]
                            ),
                        },
                    )
                if authoritative_director:
                    legality_error = self._director_action_legality_error(action)
                    if legality_error is not None:
                        code, message = legality_error
                        return self._reject_graph_action(
                            action,
                            code=code,
                            message=message,
                        )
            candidate.assert_valid(final=False)
            if staged_environment_commit:
                final_errors = candidate.validate(final=True)
                if final_errors:
                    return self._reject_graph_action(
                        action,
                        code="webshop_output_graph_not_finalizable",
                        message=(
                            "staged WebShop purchase cannot be committed until the selected "
                            "output graph is finalizable: " + "; ".join(final_errors)
                        ),
                    )
            # One accepted Canvas action is one committed topology version even
            # when its implementation performs several internal graph updates.
            candidate.version = self.graph.version + 1
        except (GraphValidationError, ValueError, TypeError) as exc:
            message = str(exc)
            if isinstance(exc, DelegationValidationError):
                responsibility_issue = {
                    "code": exc.issue.code,
                    "field": exc.issue.field,
                    "message": exc.issue.message,
                }
                if exc.issue.details:
                    responsibility_issue["details"] = dict(exc.issue.details)
            rejection_code = (
                "duplicate_responsibility"
                if isinstance(exc, DelegationValidationError)
                and exc.issue.code == "duplicate_responsibility"
                else self._classify_mutation_error(action, message)
            )
            if action.action_type is ActionType.SET_RELATION and (
                "requires layer" in message or "requires both agents" in message
            ):
                self.relation_layer_rejections += 1
                self._enter_structural_repair(
                    "relation_layer_mismatch",
                    agents=(str(action.source), str(action.target)),
                    pair=(str(action.source), str(action.target)),
                    relation=str(action.relation),
                )
                rejection_code = "relation_layer_mismatch"
            elif action.action_type is ActionType.ADD_AGENT and "already exists" in message:
                self.duplicate_agent_rejections += 1
                rejection_code = "duplicate_agent_id"
            return self._reject_graph_action(
                action,
                code=rejection_code,
                message=message,
                responsibility_issue=responsibility_issue,
            )

        execution_admission = self._execution_time_admission(
            action,
            candidate,
            mutation,
        )
        if execution_admission is not None:
            return self._reject_for_time_admission(action, execution_admission)
        token_admission = self._execution_token_admission(
            action,
            candidate,
            mutation,
            webshop_closure_pass=webshop_closure_pass,
        )
        if token_admission is not None:
            if webshop_closure_pass:
                # Selecting output is a legal Director decision, but it must not
                # purchase a new Worker execution on credit. Preserve the existing
                # trusted unpurchased episode as a zero-score terminal outcome.
                self.graph = candidate
                self.state = CanvasState.FAILED
                self._token_admission_event = dict(token_admission)
                self.token_budget_admission_rejections += 1
                return self._record(
                    action,
                    accepted=True,
                    feedback=(
                        "The selected WebShop owner has no staged purchase and insufficient "
                        "remaining tokens for a same-session closure execution. No Worker "
                        "request or environment Action was made; the official episode "
                        "remains incomplete. Existing evidence is preserved."
                    ),
                    rejection_code="webshop_output_closure_incomplete",
                    final_execution=True,
                )
            return self._reject_for_token_admission(action, token_admission)

        new_agent = next(iter(set(candidate.nodes) - before), None)
        previous_graph = self.graph
        self.graph = candidate
        if self.unified and action.action_type in self._DIRECTOR_EDIT_ACTIONS:
            # Charge the committed mutation even when its subsequent Worker
            # execution fails. Validation failures and no-ops never reach here.
            self.director_edits_used += 1
        if action.action_type is ActionType.DELETE_AGENT:
            self.runtime.discard_environment_candidate(str(action.target or action.agent_id))
        if prompt_revision_evidence is not None:
            self._commit_prompt_revision_evidence(
                action,
                prompt_revision_evidence,
            )
        graph_after = self.graph.to_dict()
        graph_after.pop("version", None)
        if self.rollout_deadline is not None and graph_after != graph_before:
            self.rollout_deadline.mark_progress(f"canvas_{action.action_type.value}")
        # Text-task output selection changes the selected Agent's Worker input
        # and invalidates its downstream closure. Environment-backed adapters
        # keep their own output/commit semantics; SWE may need one bounded final-
        # fix pass if the selected node is not already patch+test ready.
        swe_commit_pass = bool(
            action.action_type is ActionType.SET_OUTPUT
            and self.action_adapter is not None
            and self.action_adapter.adapter_id == "swe_bench"
            and not self._swe_artifact_commit_ready(str(action.target))
        )
        if (
            action.action_type is ActionType.SET_OUTPUT
            and self.action_adapter is not None
            and self.action_adapter.adapter_id == "swe_bench"
            and not swe_commit_pass
        ):
            existing_artifact = self.runtime.artifacts.get(str(action.target))
            if existing_artifact is not None:
                existing_artifact.swe_progress = {
                    **existing_artifact.swe_progress,
                    "commit_required": True,
                    "selected_as_output": True,
                    "final_fix_pass_count": 0,
                    "output_commit_ready": True,
                }
                # Worker integrity is initially evaluated before Canvas knows
                # which Agent will own the final code commit.  Re-evaluate the
                # selected, already patch+test-ready Artifact after attaching
                # that runtime-owned fact so a terminal rejected duplicate read
                # remains telemetry instead of invalidating a usable patch.
                _enforce_artifact_integrity(existing_artifact)
        commit_output = (
            action.action_type is ActionType.SET_OUTPUT
            and self.action_adapter is not None
            and (
                self.action_adapter.commit_activation.value == "execute_on_output"
                or swe_commit_pass
                or webshop_closure_pass
            )
        )
        output_contract_changed = bool(
            action.action_type is ActionType.SET_OUTPUT
            and worker_output_role_changes_input(
                dataset=self.dataset,
                action_adapter=(
                    self.action_adapter.adapter_id if self.action_adapter is not None else ""
                ),
            )
        )
        if (
            not staged_environment_cleanup
            and (
                action.action_type is not ActionType.SET_OUTPUT
                or commit_output
                or output_contract_changed
            )
        ):
            invalidated = (
                {str(action.target)}
                if swe_commit_pass or webshop_closure_pass
                else set(mutation.dirty_agents)
            )
            self.dirty_agents.update(invalidated)
            self._step_invalidated_agents.update(invalidated)
            invalidation_reasons = (
                {str(action.target): {"swe_output_commit_required"}}
                if swe_commit_pass
                else {str(action.target): {"webshop_output_closure_required"}}
                if webshop_closure_pass
                else {
                    agent_id: {"output_contract_changed"}
                    for agent_id in invalidated
                }
                if output_contract_changed and not commit_output
                else self._mutation_invalidation_reasons(
                    action,
                    mutation,
                    previous_graph=previous_graph,
                    current_graph=self.graph,
                )
            )
            for agent_id, reasons in invalidation_reasons.items():
                self.dirty_reasons.setdefault(agent_id, set()).update(reasons)
                self._step_invalidation_reasons.setdefault(agent_id, set()).update(reasons)
        self.dirty_reasons = {
            agent_id: reasons
            for agent_id, reasons in self.dirty_reasons.items()
            if agent_id in self.graph.nodes
        }
        if action.action_type is ActionType.ADD_AGENT:
            self.pending_agent_id = new_agent
            self.state = CanvasState.AWAITING_PROMPT
        elif (
            self._completes_prompt(action)
            and self.runtime_routes
            and not self.graph.nodes[str(action.target)].configured
        ):
            self.state = CanvasState.AWAITING_MODEL
        elif (
            self._completes_prompt(action)
            or discards_pending_agent
            or (
                self.state is CanvasState.AWAITING_MODEL
                and action.action_type is ActionType.SET_MODEL
                and action.target == self.pending_agent_id
            )
        ):
            self.pending_agent_id = None
            self.state = CanvasState.BUILDING

        self._refresh_structural_repair(action)

        field_repairs = (
            [repair.to_dict() for repair in compilation.field_repairs]
            if compilation is not None
            else []
        )
        report: ExecutionReport | None = None
        pending_output_waits_for_model = bool(
            action.action_type is ActionType.SET_OUTPUT
            and self.state is CanvasState.AWAITING_MODEL
            and str(action.target) == self.pending_agent_id
        )
        should_execute = (
            action.action_type is not ActionType.ADD_AGENT
            and not pending_output_waits_for_model
            and not staged_environment_cleanup
            and (
                action.action_type is not ActionType.SET_OUTPUT
                or commit_output
                or output_contract_changed
            )
        )
        if should_execute:
            try:
                # DELETE_AGENT may leave no dirty configured nodes, but Runtime
                # still needs a zero-cost synchronization pass to drop deleted
                # artifacts.  Other edits execute only their dirty closure.
                report = self._execute_dirty(force=action.action_type is ActionType.DELETE_AGENT)
            except TokenBudgetExceeded as exc:
                report = exc.report
                self._finalize_prompt_assignment(action, report)
                commit_ready_agents = (
                    tuple(self.runtime.environment_commit_ready_agents())
                    if self._uses_staged_environment_commit()
                    else ()
                )
                if commit_ready_agents:
                    # The Worker execution and its staged environment transaction are
                    # already durable. Preserve that same session long enough for the
                    # zero-Worker-token SET_OUTPUT protocol step; failing first would
                    # strand a completed purchase candidate and misattribute the result.
                    self._enter_structural_repair("token_budget_consolidation")
                else:
                    self.state = CanvasState.FAILED
                return self._record(
                    action,
                    # The graph edit itself was legal and is already committed;
                    # only the cumulative execution budget ended the rollout.
                    accepted=True,
                    feedback=self._feedback(
                        (
                            "Execution crossed the Worker token budget after producing a "
                            "trusted staged WebShop purchase. The staged session is preserved "
                            "only for bounded SET_OUTPUT/FINISH recovery: "
                            f"{exc}"
                            if commit_ready_agents
                            else f"Execution stopped after the accepted graph edit: {exc}"
                        ),
                        report,
                    ),
                    execution=report,
                    rejection_code="execution_budget_exceeded",
                    delegation_field_repairs=field_repairs,
                )
            except RequiredWorkerBackendFailure as exc:
                report = exc.report
                self._finalize_prompt_assignment(action, report)
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    # The graph mutation and diagnostic execution are durable,
                    # but the outage must end the trajectory before the Director
                    # can react by adding unrelated Agents.
                    accepted=True,
                    feedback=self._feedback(
                        f"Execution stopped after the accepted graph edit: {exc}",
                        report,
                    ),
                    execution=report,
                    rejection_code="worker_backend_unavailable",
                    delegation_field_repairs=field_repairs,
                )
            except ValueError as exc:
                self._finalize_prompt_assignment(action, report)
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    accepted=True,
                    feedback=self._feedback(
                        f"Execution failed after the accepted graph edit: {exc}",
                        report,
                    ),
                    execution=report,
                    rejection_code="execution_failure",
                    delegation_field_repairs=field_repairs,
                )
        self._finalize_prompt_assignment(action, report)
        if staged_environment_commit:
            selected_artifact = self.runtime.artifacts.get(str(action.target))
            selected_result = (
                dict(selected_artifact.environment_result)
                if selected_artifact is not None
                and isinstance(selected_artifact.environment_result, dict)
                else {}
            )
            if bool(selected_result.get("purchased", False)):
                self.state = CanvasState.FINISHED
                if self.rollout_deadline is not None:
                    self.rollout_deadline.mark_progress("canvas_environment_closure_finish")
                return self._record(
                    action,
                    accepted=True,
                    feedback=self._feedback(
                        (
                            "The selected WebShop owner completed the purchase during the bounded "
                            "same-session output closure pass "
                            f"(reward={float(selected_result.get('reward', 0.0) or 0.0):.6g}); "
                            "finished graph."
                        ),
                        report,
                    ),
                    execution=report,
                    final_execution=True,
                    delegation_field_repairs=field_repairs,
                )
            if str(action.target) not in set(self.runtime.environment_commit_ready_agents()):
                self.state = CanvasState.FAILED
                accounting = dict(
                    selected_artifact.webshop_progress.get("execution_accounting", {})
                    if selected_artifact is not None
                    and report is not None
                    and str(action.target) in report.executed_agents
                    else {}
                )
                self._token_admission_event["closure_execution"] = accounting
                zero_calls = accounting.get("model_request_count") == 0
                closure_status = (
                    "The selected WebShop owner's closure request was blocked before the first "
                    "model call; no model request or environment Action was made in this closure. "
                    if zero_calls
                    else "The selected WebShop owner's bounded same-session output closure ended "
                    "without staging or completing a purchase. "
                )
                return self._record(
                    action,
                    accepted=True,
                    feedback=self._feedback(
                        (
                            closure_status + "The official "
                            "episode is incomplete; no replacement session will be started."
                        ),
                        report,
                    ),
                    execution=report,
                    rejection_code="webshop_output_closure_incomplete",
                    final_execution=True,
                    delegation_field_repairs=field_repairs,
                )
            try:
                result = self.runtime.commit_environment_output(str(action.target))
            except Exception as exc:  # noqa: BLE001 - external commit is fail-closed
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    accepted=True,
                    feedback=(
                        "Selected staged WebShop output, but the deterministic environment "
                        f"commit failed: {exc}"
                    ),
                    rejection_code="webshop_environment_commit_failed",
                    final_execution=True,
                    delegation_field_repairs=field_repairs,
                )
            if not bool(result.get("purchased", False)):
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    accepted=True,
                    feedback=(
                        "The selected staged WebShop action executed but did not produce a "
                        "terminal purchase; refusing to claim a completed output."
                    ),
                    rejection_code="webshop_environment_commit_incomplete",
                    final_execution=True,
                    delegation_field_repairs=field_repairs,
                )
            self.state = CanvasState.FINISHED
            if self.rollout_deadline is not None:
                self.rollout_deadline.mark_progress("canvas_environment_commit_finish")
            return self._record(
                action,
                accepted=True,
                feedback=(
                    "Committed the selected staged WebShop purchase without another Worker "
                    f"execution (purchased={bool(result.get('purchased', False))}, "
                    f"reward={float(result.get('reward', 0.0) or 0.0):.6g}); finished graph."
                ),
                final_execution=True,
                delegation_field_repairs=field_repairs,
            )
        return self._record(
            action,
            accepted=True,
            feedback=self._feedback(mutation.message, report),
            execution=report,
            delegation_field_repairs=field_repairs,
        )

    def recover_pending_prompt(self) -> list[CanvasStep]:
        """Safely configure a twice-rejected Agent without ending the graph."""

        if self.state is not CanvasState.AWAITING_PROMPT or not self.pending_agent_id:
            return []
        target = self.pending_agent_id
        recovered: list[CanvasStep] = []

        def apply(raw_action: str) -> bool:
            step = self.step(raw_action, count_round=False)
            step.protocol_recovery = True
            recovered.append(step)
            return step.accepted

        if not apply(
            json.dumps(
                {
                    "action": "set_prompt",
                    "target": target,
                    "role": "Independent solver",
                    "objective": "Solve the assigned task independently.",
                    "scope": "Reason carefully and verify the result.",
                    "expected_output": "Return a concise direct final answer.",
                }
            )
        ):
            return recovered
        return recovered

    def recover_unknown_pending_target(self, action: CanvasAction) -> CanvasStep | None:
        """Retarget a rejected SET_PROMPT to the sole runtime-known pending Agent."""

        if (
            self.state is not CanvasState.AWAITING_PROMPT
            or not self.pending_agent_id
            or action.action_type is not ActionType.SET_PROMPT
        ):
            return None
        payload = {
            "action": "set_prompt",
            "target": self.pending_agent_id,
            "role": action.role,
            "objective": action.objective,
            "scope": action.scope,
            "expected_output": action.expected_output,
            "expected_version": self.graph.version,
        }
        recovered = self.step(json.dumps(payload, ensure_ascii=False), count_round=False)
        recovered.protocol_recovery = True
        return recovered

    def recover_output_lifecycle(self) -> list[CanvasStep]:
        if self.unified:
            return []
        """Deterministically close protocol-only output repair states.

        The Director retains one semantic choice when several usable artifacts
        exist. Once an output is selected, pruning and FINISH are controller
        protocol steps and do not consume another model turn.
        """

        if not self._is_output_lifecycle_repair() or self.state in {
            CanvasState.FINISHED,
            CanvasState.FAILED,
        }:
            return []
        recovered: list[CanvasStep] = []
        current = self.graph.output_agent
        current_usable = bool(current is not None and self._artifact_is_usable_output(current))
        if self._is_consolidation_repair() and current_usable:
            self._lock_consolidation_output()

        if not current_usable:
            candidates = [
                agent_id for agent_id in self._eligible_output_agents() if agent_id != current
            ]
            if not candidates:
                self.output_lifecycle_recovery_count += 1
                recovered.append(
                    self._fail_terminal_recovery(
                        code="no_usable_output_artifact",
                        message="no configured Agent has a usable answer artifact",
                    )
                )
                return recovered
            if len(candidates) > 1:
                if self._is_consolidation_repair() and self._output_selection_budget_remaining <= 0:
                    self.output_lifecycle_recovery_count += 1
                    recovered.append(
                        self._fail_terminal_recovery(
                            code="output_selection_budget_exhausted",
                            message="the one semantic output selection was already consumed",
                        )
                    )
                    return recovered
                # This is the one remaining trainable decision. The legal
                # surface exposes only these candidates to the Director.
                return []
            self.output_lifecycle_recovery_count += 1
            selected_step = self.step(
                json.dumps(
                    {
                        "action": "set_output",
                        "target": candidates[0],
                        "expected_version": self.graph.version,
                    }
                ),
                count_round=False,
            )
            selected_step.protocol_recovery = True
            recovered.append(selected_step)
            if not selected_step.accepted:
                recovered.append(
                    self._fail_terminal_recovery(
                        code="output_lifecycle_recovery_failed",
                        message="the sole usable output candidate could not be selected",
                    )
                )
                return recovered
            if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
                # A staged environment SET_OUTPUT is itself the irreversible
                # commit and terminal transition. Do not append a redundant
                # FINISH step after that transaction has already completed.
                return recovered
        else:
            self.output_lifecycle_recovery_count += 1

        for agent_id in sorted(self._unreachable_to_output()):
            if (
                agent_id
                not in self._legal_action_parameters()[ActionType.DELETE_AGENT.value]["targets"]
            ):
                break
            deleted_step = self.step(
                json.dumps(
                    {
                        "action": "delete_agent",
                        "target": agent_id,
                        "expected_version": self.graph.version,
                    }
                ),
                count_round=False,
            )
            deleted_step.protocol_recovery = True
            recovered.append(deleted_step)
            if not deleted_step.accepted:
                break

        if (
            self.graph.output_agent is not None
            and self._artifact_is_usable_output(self.graph.output_agent)
            and not self.graph.validate(final=True)
        ):
            finish_step = self.step(
                json.dumps({"action": "finish", "expected_version": self.graph.version}),
                count_round=False,
            )
            finish_step.protocol_recovery = True
            recovered.append(finish_step)

        if self.state not in {CanvasState.FINISHED, CanvasState.FAILED}:
            recovered.append(
                self._fail_terminal_recovery(
                    code="output_lifecycle_recovery_failed",
                    message="locked output did not yield a final-valid graph",
                )
            )
        return recovered

    def recover_frozen_topology(
        self,
        *,
        failure_code: str = "topology_recovery_exhausted",
    ) -> list[CanvasStep]:
        if self.unified:
            return []
        """Boundedly finalize the best usable existing graph after edit-loop fusion."""

        if not self.topology_edits_frozen:
            return []
        recovered: list[CanvasStep] = []
        if self.state is CanvasState.AWAITING_PROMPT:
            pending = self.pending_agent_id
            if pending is not None:
                discarded = self.step(
                    json.dumps(
                        {
                            "action": "delete_agent",
                            "target": pending,
                            "expected_version": self.graph.version,
                        }
                    ),
                    count_round=False,
                )
                discarded.protocol_recovery = True
                recovered.append(discarded)
            if self.state is CanvasState.AWAITING_PROMPT:
                recovered.append(
                    self._fail_terminal_recovery(
                        code=failure_code,
                        message="could not configure the pending Agent during recovery",
                    )
                )
                return recovered
        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return recovered
        if self._is_output_lifecycle_repair():
            lifecycle_recovery = self.recover_output_lifecycle()
            recovered.extend(lifecycle_recovery)
            if lifecycle_recovery or self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
                return recovered
            recovered.append(
                self._fail_terminal_recovery(
                    code=failure_code,
                    message=(
                        "the Director exhausted output selection repair without choosing "
                        "one of multiple usable artifacts"
                    ),
                )
            )
            return recovered
        candidates = [
            artifact
            for agent_id, artifact in self.runtime.artifacts.items()
            if agent_id in self.graph.nodes and self._artifact_is_usable_output(agent_id)
        ]
        if self.graph.output_agent is None and candidates:
            selected = max(
                candidates,
                key=lambda artifact: (
                    artifact.confidence,
                    len(artifact.evidence),
                    artifact.agent_id,
                ),
            )
            selected_step = self.step(
                json.dumps(
                    {
                        "action": "set_output",
                        "target": selected.agent_id,
                        "expected_version": self.graph.version,
                    }
                ),
                count_round=False,
            )
            selected_step.protocol_recovery = True
            recovered.append(selected_step)
        if (
            self.graph.output_agent is not None
            and ActionType.DELETE_AGENT in self._structural_repair_allowed_action_types()
        ):
            # A consolidation lock can coexist with a disconnected extra Agent.
            # Repeatedly selecting the existing output freezes open-ended topology
            # edits, but bounded deletion is still required before FINISH can pass
            # final graph validation. Remove only Agents that cannot reach the
            # already selected output; never mutate the chosen output itself.
            for agent_id in sorted(self._unreachable_to_output()):
                deleted_step = self.step(
                    json.dumps(
                        {
                            "action": "delete_agent",
                            "target": agent_id,
                            "expected_version": self.graph.version,
                        }
                    ),
                    count_round=False,
                )
                deleted_step.protocol_recovery = True
                recovered.append(deleted_step)
                if not deleted_step.accepted:
                    break
        if self.graph.output_agent is not None:
            finish_step = self.step(
                json.dumps({"action": "finish", "expected_version": self.graph.version}),
                count_round=False,
            )
            finish_step.protocol_recovery = True
            recovered.append(finish_step)
        if self.state not in {CanvasState.FINISHED, CanvasState.FAILED}:
            recovered.append(
                self._fail_terminal_recovery(
                    code=failure_code,
                    message="no finishable existing graph remained after bounded recovery",
                )
            )
        return recovered

    def recover_selected_output_agent(self, *, reason_code: str) -> RuntimeRecoveryReport:
        """Rerun only the selected output Agent without fabricating a policy action."""

        if self.submission_receipt is not None:
            raise ValueError("selected-output recovery must precede FINISH acceptance")

        target = self.graph.output_agent
        if not target or target not in self.graph.nodes:
            raise ValueError("selected-output recovery requires an output Agent")
        if not self.graph.nodes[target].configured:
            raise ValueError("selected-output recovery requires a configured Agent")
        if target not in self.runtime.artifacts:
            raise ValueError("selected-output recovery requires a prior Artifact")
        self._time_admission_event = {}
        self._token_admission_event = {}
        self._step_invalidated_agents = {target}
        self._step_scheduled_agents = set()
        self._step_invalidation_reasons = {
            target: {"selected_output_recovery_required", str(reason_code)}
        }
        self._step_prompt_revision = {}
        self._step_responsibility_overlap_check = {}
        self.dirty_agents.add(target)
        self.dirty_reasons[target] = set(self._step_invalidation_reasons[target])
        try:
            report = self._execute_dirty()
        except (TokenBudgetExceeded, RequiredWorkerBackendFailure) as exc:
            self._submission_recovery_reports.append(exc.report)
            raise
        if report is not None:
            self._submission_recovery_reports.append(report)
        return RuntimeRecoveryReport(accepted=True, execution=report)

    def recover_trusted_alfworld_success(self) -> list[CanvasStep]:
        if self.unified:
            return []
        """Lock and finalize an Agent whose ALFWorld episode officially succeeded."""

        if (
            self.state in {CanvasState.FINISHED, CanvasState.FAILED}
            or self.action_adapter is None
            or self.action_adapter.adapter_id != "alfworld"
        ):
            return []
        winners = [
            artifact
            for artifact in self.runtime.artifacts.values()
            if artifact.environment_result.get("environment_completed") is True
            and artifact.environment_result.get("won") is True
            and artifact.agent_id in self.graph.nodes
        ]
        if not winners:
            return []
        current = self.graph.output_agent
        winner = next(
            (artifact for artifact in winners if artifact.agent_id == current),
            min(
                winners,
                key=lambda artifact: (
                    int(artifact.environment_result.get("attempt_index", 0) or 0),
                    artifact.agent_id,
                ),
            ),
        )
        recovered: list[CanvasStep] = []
        if current != winner.agent_id:
            selected = self.step(
                json.dumps(
                    {
                        "action": "set_output",
                        "target": winner.agent_id,
                        "expected_version": self.graph.version,
                    }
                ),
                count_round=False,
            )
            selected.protocol_recovery = True
            selected.rejection_details = {
                "recovery_scope": "trusted_alfworld_success",
                "reason_code": "official_environment_success",
            }
            recovered.append(selected)
            if not selected.accepted:
                return recovered

        for agent_id in sorted(self._unreachable_to_output()):
            legal = self._legal_action_parameters()[ActionType.DELETE_AGENT.value]["targets"]
            if agent_id not in legal:
                break
            deleted = self.step(
                json.dumps(
                    {
                        "action": "delete_agent",
                        "target": agent_id,
                        "expected_version": self.graph.version,
                    }
                ),
                count_round=False,
            )
            deleted.protocol_recovery = True
            deleted.rejection_details = {
                "recovery_scope": "trusted_alfworld_success",
                "reason_code": "prune_nonwinning_branch",
            }
            recovered.append(deleted)
            if not deleted.accepted:
                return recovered

        if self.graph.output_agent == winner.agent_id and not self.graph.validate(final=True):
            finished = self.step(
                json.dumps({"action": "finish", "expected_version": self.graph.version}),
                count_round=False,
            )
            finished.protocol_recovery = True
            finished.rejection_details = {
                "recovery_scope": "trusted_alfworld_success",
                "reason_code": "official_environment_success",
            }
            recovered.append(finished)
        return recovered

    def recover_finish_only(self) -> CanvasStep | None:
        if self.unified:
            return None
        """Finish a ready graph without spending another model turn."""

        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return None
        snapshot = self.control_snapshot()
        finish_is_only_action = snapshot["allowed_actions"] == [ActionType.FINISH.value]
        if self.dataset == "alfworld" and not finish_is_only_action:
            artifact = self.runtime.artifacts.get(self.graph.output_agent)
            result = artifact.environment_result if artifact is not None else {}
            # Selecting output is not a decision to abandon an unsolved episode.
            # Preserve the next Director turn whenever other legal actions remain.
            if not (result.get("environment_completed") is True and result.get("won") is True):
                return None
        graph_is_ready = bool(
            self.graph.output_agent
            and not self.graph.validate(final=True)
            and ActionType.FINISH.value in snapshot["allowed_actions"]
        )
        if not (finish_is_only_action or graph_is_ready):
            return None
        step = self.step(
            json.dumps({"action": "finish", "expected_version": self.graph.version}),
            count_round=False,
        )
        step.protocol_recovery = True
        if not step.accepted and self.control_snapshot()["allowed_actions"] == [
            ActionType.FINISH.value
        ]:
            # Frozen topology alone is insufficient: SET_OUTPUT/DELETE_AGENT
            # can still be legal. Stop only when FINISH is the sole action.
            return self._fail_terminal_recovery(
                code=step.rejection_code or "finish_recovery_rejected",
                message="automatic FINISH was rejected and no graph-edit recovery remains",
            )
        return step

    def recover_round_limit(self) -> list[CanvasStep]:
        if self.unified:
            return []
        """Close an exhausted Director loop with a final graph or a typed failure."""

        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return []
        if self.round_index < self.config.max_rounds:
            return []
        self.topology_edits_frozen = True
        return self.recover_frozen_topology(failure_code="max_rounds_exhausted")

    def terminate_round_limit_without_graph_repair(self) -> CanvasStep | None:
        """Terminate an exhausted policy loop without editing the Director's graph."""

        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return None
        if self.director_round_limit is None:
            return None
        if self.round_index < self.config.max_rounds:
            return None
        self.topology_edits_frozen = True
        return self._fail_terminal_recovery(
            code="max_rounds_exhausted",
            message="Director round budget ended before a final-valid graph was finished",
        )

    def terminate_context_limit_without_graph_repair(self) -> CanvasStep:
        """Record exact-context exhaustion without selecting output or inventing tokens."""
        return self._fail_terminal_recovery(
            code="director_context_budget_exhausted",
            message="exact Director history exhausted the 32768-token context budget",
        )

    def fail_if_no_usable_output_artifact(self) -> CanvasStep | None:
        """Fail a terminal consolidation that offers no possible output choice.

        This is deliberately not a graph repair: it never selects an output,
        deletes an Agent, or changes a relation.  It merely avoids spending the
        remaining Director budget on a control state with no legal successful
        continuation.
        """

        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return None
        if not self._is_output_lifecycle_repair():
            return None
        current = self.graph.output_agent
        if current is not None and self._artifact_is_usable_output(current):
            return None
        if self._eligible_output_agents():
            return None
        self.output_lifecycle_recovery_count += 1
        return self._fail_terminal_recovery(
            code="no_usable_output_artifact",
            message="no configured Agent has a usable answer artifact",
        )

    def _fail_terminal_recovery(self, *, code: str, message: str) -> CanvasStep:
        self.state = CanvasState.FAILED
        return self._record(
            self.parser.parse(
                json.dumps({"action": "finish", "expected_version": self.graph.version})
            ),
            accepted=False,
            feedback=f"Deterministic terminal recovery failed: {message}.",
            rejection_code=code,
            protocol_recovery=True,
        )

    def _finish(self, action: CanvasAction) -> CanvasStep:
        text_submission = is_text_submission_dataset(self.dataset)
        context = self._director_call_context
        if text_submission and (
            not isinstance(context, DirectorCallContext) or not context.runtime_owned
            or context.run_id != self.run_id
        ):
            return self._record(
                action, accepted=False, rejection_code="director_finish_source_required",
                feedback="Text submission requires FINISH from an actual Director call.",
            )
        if self.total_tokens > self.config.max_total_tokens:
            self.state = CanvasState.FAILED
            return self._record(
                action, accepted=False, rejection_code="execution_budget_exceeded",
                feedback="Rejected finish: accumulated Worker tokens exceed the task budget.",
                final_execution=True,
            )
        if self.state in {CanvasState.AWAITING_PROMPT, CanvasState.AWAITING_MODEL}:
            return self._record(
                action,
                accepted=False,
                feedback=f"Rejected action: {self.pending_agent_id} is incompletely configured.",
            )
        errors = self.graph.validate(final=True)
        if errors:
            if self.config.structural_repair_enabled:
                if "output agent is not set" in errors and self.graph.nodes:
                    self._enter_structural_repair("output_not_set")
                unreachable = self._unreachable_to_output()
                if unreachable:
                    self.finish_reachability_rejections += 1
                    self._enter_structural_repair(
                        "output_reachability",
                        agents=unreachable,
                    )
            return self._record(
                action,
                accepted=False,
                feedback="Rejected finish: " + "; ".join(errors),
                rejection_code=(
                    "output_reachability_required"
                    if self.structural_repair_reason == "output_reachability"
                    else "output_selection_required"
                    if self.structural_repair_reason == "output_not_set"
                    else None
                ),
            )

        if self.structural_repair_reason in {
            "token_budget_consolidation",
            "time_budget_consolidation",
        }:
            self._clear_structural_repair()

        report: ExecutionReport | None = None
        prepared_submission = None
        try:
            self._prepare_native_output_materialization()
            for agent_id in self.dirty_agents:
                self.dirty_reasons.setdefault(agent_id, set()).add("residual_finish")
            report = self._execute_dirty(force=True)
            if text_submission and self.prepare_text_submission is not None:
                prepared_submission = self.prepare_text_submission(self)
                report = _merge_submission_reports(report, self._submission_recovery_reports)
        except TokenBudgetExceeded as exc:
            report = _merge_submission_reports(report, [exc.report])
            self.state = CanvasState.FAILED
            return self._record(
                action,
                accepted=False,
                feedback=f"Rejected finish: {exc}",
                execution=report,
                final_execution=True,
                rejection_code="execution_budget_exceeded",
            )
        except RequiredWorkerBackendFailure as exc:
            report = _merge_submission_reports(report, [exc.report])
            self.state = CanvasState.FAILED
            return self._record(
                action,
                accepted=False,
                feedback=self._feedback(
                    f"Final execution stopped: {exc}",
                    report,
                ),
                execution=report,
                rejection_code="worker_backend_unavailable",
                final_execution=True,
            )
        except ValueError as exc:
            self.state = CanvasState.FAILED
            return self._record(
                action,
                accepted=False,
                feedback=f"Final execution failed: {exc}",
                final_execution=True,
                rejection_code="execution_failure",
            )
        if self.graph.output_agent is not None and not self._artifact_is_usable_output(
            self.graph.output_agent
        ):
            unusable_output = self.graph.output_agent
            self._enter_structural_repair(
                "output_artifact_unusable",
                agents=(unusable_output,),
            )
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected finish: selected output Agent did not produce a usable result "
                    "for its current inputs"
                ),
                execution=report,
                rejection_code="output_artifact_unusable",
            )
        if (
            self.graph.output_agent is not None
            and self.action_adapter is not None
            and self.action_adapter.adapter_id == "swe_bench"
            and not self._swe_artifact_commit_ready(self.graph.output_agent)
        ):
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected finish: selected SWE output did not produce a tested patch or a "
                    "structured typed failure during its bounded final-fix pass"
                ),
                execution=report,
                rejection_code="swe_output_commit_incomplete",
            )
        selected_output = self.graph.output_agent
        input_signature_check = getattr(
            self.runtime, "artifact_matches_current_input_signature", None
        )
        if (
            selected_output is not None
            and worker_output_role_changes_input(
                dataset=self.dataset,
                action_adapter=(
                    self.action_adapter.adapter_id if self.action_adapter is not None else ""
                ),
            )
            and callable(input_signature_check)
            and not input_signature_check(selected_output, task=self.worker_task, graph=self.graph)
        ):
            self.dirty_agents.add(selected_output)
            self.dirty_reasons.setdefault(selected_output, set()).add(
                "output_contract_signature_stale"
            )
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected finish: selected output Artifact does not match a recorded input "
                    "signature for its current output contract; it remains dirty for execution"
                ),
                execution=report,
                rejection_code="stale_output_artifact",
            )
        output_artifact = (
            self.runtime.artifacts.get(self.graph.output_agent) if self.graph.output_agent else None
        )
        output = output_artifact.answer if output_artifact is not None else ""
        trusted_alfworld_output = bool(
            (
                self.dataset == "alfworld"
                or (
                    self.action_adapter is not None
                    and self.action_adapter.adapter_id == "alfworld"
                )
            )
            and output_artifact is not None
            and self._artifact_is_usable_output(output_artifact.agent_id)
        )
        if not output and not trusted_alfworld_output:
            self.state = CanvasState.FAILED
            return self._record(
                action,
                accepted=False,
                feedback="Final execution failed: output agent did not produce an answer.",
                execution=report,
                final_execution=True,
            )
        if self.runtime.native_webshop and self.dataset == "webshop":
            selected = self.graph.output_agent
            if selected in self.runtime.environment_commit_ready_agents():
                try:
                    result = self.runtime.commit_environment_output(selected)
                    if not result.get("purchased"):
                        raise ValueError("selected candidate did not reach a terminal purchase")
                    if report is not None:
                        report.artifacts[selected] = self.runtime.artifacts[selected]
                except Exception as exc:
                    self.state = CanvasState.FAILED
                    return self._record(
                        action,
                        accepted=False,
                        feedback=f"Selected WebShop candidate commit failed: {exc}",
                        execution=report,
                        final_execution=True,
                        rejection_code="webshop_environment_commit_failed",
                    )
            # A graph may legitimately finish without a purchase (score zero).
            # Never select a different Agent or a candidate using hidden reward.
        if text_submission:
            if prepared_submission is None:
                from .answer_submission import AnswerFinalizer, AnswerSubmissionConfig
                from .observability import TaskSpec

                prepared_submission = AnswerFinalizer(
                    AnswerSubmissionConfig(enabled=True)
                ).finalize(
                    TaskSpec(task_id=self.run_id, prompt=self.worker_task,
                             metadata={"dataset": self.dataset}),
                    output,
                )
            risks = artifact_integrity_failure_risks(output_artifact) if output_artifact else []
            if not prepared_submission.valid or risks:
                return self._record(
                    action, accepted=False, execution=report,
                    rejection_code="output_artifact_integrity_failure" if risks else "output_answer_missing",
                    feedback="Rejected finish: selected output is incomplete or has no answer payload.",
                    rejection_details={"integrity_risks": risks,
                                       "submission_detail": prepared_submission.detail},
                )
            if (self.total_tokens > self.config.max_total_tokens
                    or self.graph.validate(final=True) or not self.selected_output_is_current()):
                return self._record(
                    action, accepted=False, execution=report,
                    rejection_code="submission_final_check_failed",
                    feedback="Rejected finish: final graph, input binding or budget check failed.",
                )
            binding = self.runtime.artifact_input_binding(str(self.graph.output_agent))
            if not binding.get("input_hash") or binding.get("artifact_id") != output_artifact.artifact_id:
                return self._record(
                    action, accepted=False, execution=report,
                    rejection_code="submission_input_binding_missing",
                    feedback="Rejected finish: selected output lacks its runtime input binding.",
                )
            self.submission_receipt = _issue_receipt(
                context=context, dataset=self.dataset,
                accepted_event_id=f"{self.run_id}:canvas:{len(self.history)}",
                output_agent_id=str(self.graph.output_agent),
                graph_snapshot_hash=snapshot_hash(self.graph.to_dict()),
                artifact_id=output_artifact.artifact_id,
                input_signature=str(binding.get("input_hash", "")),
                execution_generation=int(binding.get("generation", 0)),
                raw_answer_snapshot=output,
                submitted_answer_snapshot=prepared_submission.submitted_answer,
                answer_hash=answer_hash(prepared_submission.submitted_answer),
                normalization_version=prepared_submission.method,
                worker_tokens_used=self.total_tokens,
                worker_token_limit=self.config.max_total_tokens,
            )
        self.state = CanvasState.FINISHED
        if self.rollout_deadline is not None:
            self.rollout_deadline.mark_progress("canvas_finish")
        return self._record(
            action,
            accepted=True,
            feedback=self._feedback("Finished graph execution.", report),
            execution=report,
            final_execution=True,
        )

    def evaluate_flowsteer_structure(self):
        """Compatibility projection for old traces; no role/topology gate is active."""

        return self.graph.evaluate_flowsteer_structure(enabled=False)

    def selected_output_is_current(self) -> bool:
        """One submission gate shared by Director, finalization and scoring."""
        selected = self.graph.output_agent
        if not selected or selected not in self.runtime.artifacts:
            return False
        if self.unified and self.submission_receipt is not None and self.state is CanvasState.FINISHED:
            return self.runtime.artifacts[selected].artifact_id == self.submission_receipt.artifact_id
        if not self.unified and not worker_output_role_changes_input(
            dataset=self.dataset,
            action_adapter=self.action_adapter.adapter_id if self.action_adapter else "",
        ):
            return True  # Environment/code provenance is checked by the adapter.
        return selected not in self.dirty_agents and self.runtime.artifact_matches_current_input_signature(
            selected, task=self.worker_task, graph=self.graph
        )

    def _execute_dirty(self, *, force: bool = False) -> ExecutionReport | None:
        # A deleted Agent can remain in the mutation's historical dirty set but
        # must never survive as executable state.
        self.dirty_agents.intersection_update(self.graph.nodes)
        self.dirty_reasons = {
            agent_id: reasons
            for agent_id, reasons in self.dirty_reasons.items()
            if agent_id in self.graph.nodes
        }
        configured = {key for key, node in self.graph.nodes.items() if node.configured}
        executable_dirty = self.dirty_agents & configured
        if not executable_dirty and not force:
            return None
        if self.dataset == "swe_bench" and configured:
            remaining = max(0, self.config.max_total_tokens - self.total_tokens)
            self._token_admission_event.update(
                budget_schema=SWE_SHARED_TOKEN_BUDGET,
                remaining_worker_tokens=remaining,
                budget_scope="per_question",
                allocation="shared_remaining",
                reserved_closure_tokens=0,
                request_admission="authoritative_after_request_serialization",
            )
            for agent_id in configured:
                metadata = self.graph.nodes[agent_id].metadata
                metadata.update(
                    _runtime_token_credit=remaining,
                    _runtime_budget_kind=SWE_SHARED_TOKEN_BUDGET,
                )
                # Restored nodes must not retain the old finalization reserve.
                metadata.pop("_runtime_finalization_output_reserve", None)
        elif (is_short_qa_dataset(self.dataset)
                or (self.unified and self.dataset != "webshop")) and executable_dirty:
            # The estimator is a scheduling hint, not permission to spend past
            # the budget. Give each scheduled execution/revision a bounded share
            # and enforce it against serialized requests in the gateway.
            calls = max(
                1,
                int(
                    self.runtime.estimate_execution_tokens(
                        self.graph,
                        executable_dirty,
                        quantile=self.config.worker_token_quantile,
                        minimum_samples=self.config.worker_token_min_samples,
                        cold_start_tokens=self.config.worker_token_cold_start,
                    )["call_count"]
                ),
            )
            remaining = max(0, self.config.max_total_tokens - self.total_tokens)
            credit = remaining // calls
            budget_kind = (
                "short_qa_request_credit_v1" if is_short_qa_dataset(self.dataset)
                else "unified_request_credit_v1"
            )
            self._token_admission_event.update(
                {
                    "budget_schema": budget_kind,
                    "remaining_worker_tokens": remaining,
                    "finalization_output_reserve_minimum": self.config.finalization_token_reserve,
                    "call_count": calls,
                    "per_execution_credit": credit,
                    "request_admission": "authoritative_after_request_serialization",
                }
            )
            for agent_id in configured:
                self.graph.nodes[agent_id].metadata.update(
                    {
                        "_runtime_token_credit": credit,
                        "_runtime_budget_kind": budget_kind,
                        "_runtime_finalization_output_reserve": self.config.finalization_token_reserve,
                    }
                )
        if self.dataset == "webshop" and executable_dirty:
            admission_enabled = self.config.remaining_token_admission_enabled
            partition = None
            if admission_enabled:
                calls = self.runtime.estimate_execution_tokens(
                    self.graph,
                    executable_dirty,
                    quantile=self.config.worker_token_quantile,
                    minimum_samples=self.config.worker_token_min_samples,
                    cold_start_tokens=self.config.worker_token_cold_start,
                )["call_count"]
                closure = any(
                    "webshop_output_closure_required" in self.dirty_reasons.get(agent_id, ())
                    for agent_id in executable_dirty
                )
                partition = self._webshop_budget_partition(call_count=int(calls), closure=closure)
                self._token_admission_event.update(partition)
            else:
                self._token_admission_event.update(
                    request_admission="disabled",
                    remaining_worker_tokens=max(
                        0, self.config.max_total_tokens - self.total_tokens
                    ),
                )
            for agent_id in executable_dirty:
                metadata = self.graph.nodes[agent_id].metadata
                metadata.update(
                    {
                        "_runtime_webshop_request_admission_enabled": admission_enabled,
                        "_runtime_webshop_output_closure": (
                            "webshop_output_closure_required"
                            in self.dirty_reasons.get(agent_id, ())
                        ),
                    }
                )
                if partition is not None:
                    metadata.update(
                        _runtime_token_credit=partition["per_execution_credit"],
                        _runtime_reserved_closure_tokens=partition["reserved_closure_tokens"],
                        _runtime_budget_phase=partition["phase"],
                    )
                else:
                    # A restored graph may still carry an earlier execution's credit.
                    for key in (
                        "_runtime_token_credit",
                        "_runtime_reserved_closure_tokens",
                        "_runtime_budget_phase",
                    ):
                        metadata.pop(key, None)
        report = self.runtime.execute(
            task=self.worker_task,
            graph=self.graph,
            dirty_agents=executable_dirty if executable_dirty else set(),
            invalidation_reasons={
                agent_id: set(self.dirty_reasons.get(agent_id, ())) for agent_id in executable_dirty
            },
            token_credit=(max(0, self.config.max_total_tokens - self.total_tokens)
                          if self.dataset != "webshop" or self.config.remaining_token_admission_enabled
                          else None),
        )
        self._step_scheduled_agents.update(report.scheduled_agents)
        self.total_tokens += report.token_in + report.token_out
        # Execution has already committed artifacts to the runtime. Clear the
        # corresponding dirty bits before reporting a budget violation; otherwise
        # every later graph edit re-executes the same Agents and compounds the
        # over-budget trajectory without producing new feedback.
        completed = set(report.scheduled_agents) - report.blocked_agents.keys()
        self.dirty_agents -= completed
        for agent_id in completed:
            self.dirty_reasons.pop(agent_id, None)
        self.dirty_agents.update(report.blocked_agents)
        for agent_id, blocker in report.blocked_agents.items():
            self.dirty_reasons.setdefault(agent_id, set()).add(blocker)
        if any(
            artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
            and not (self.unified and self.runtime.artifact_matches_current_input_signature(
                artifact.agent_id, task=self.worker_task, graph=self.graph))
            for artifact in report.artifacts.values()
        ):
            raise RequiredWorkerBackendFailure(report)
        if self.total_tokens > self.config.max_total_tokens:
            raise TokenBudgetExceeded(
                report,
                self.total_tokens,
                self.config.max_total_tokens,
            )
        return report

    def _new_agent_time_admission(self, action: CanvasAction) -> dict[str, Any] | None:
        if (
            action.action_type is not ActionType.ADD_AGENT
            or not self.graph.nodes
            or not self._time_admission_active()
            or not self._has_usable_artifact()
        ):
            return None
        estimate = self.runtime.estimate_new_agent_s(
            self.runtime_routes,
            quantile=self.config.worker_latency_quantile,
            minimum_samples=self.config.worker_latency_min_samples,
            cold_start_s=self.config.worker_latency_cold_start_s,
            workload_scope=(
                self.action_adapter.adapter_id if self.action_adapter is not None else ""
            ),
        )
        return self._insufficient_time_event(action, estimate)

    def _new_agent_token_admission(self, action: CanvasAction) -> dict[str, Any] | None:
        if (
            self.dataset == "swe_bench"
            or action.action_type is not ActionType.ADD_AGENT
            or not self.graph.nodes
            or not self.config.remaining_token_admission_enabled
            or not self._has_usable_artifact()
        ):
            return None
        estimate = self.runtime.estimate_new_agent_tokens(
            self.runtime_routes,
            quantile=self.config.worker_token_quantile,
            minimum_samples=self.config.worker_token_min_samples,
            cold_start_tokens=self.config.worker_token_cold_start,
            workload_scope=(
                self.action_adapter.adapter_id if self.action_adapter is not None else ""
            ),
        )
        # ADD_AGENT itself is zero execution, but it creates a mandatory
        # SET_PROMPT barrier. Reserve one call to configure the node and one
        # conservative call to integrate it into the graph before finalization.
        estimate = {
            **estimate,
            "estimated_worker_tokens": int(estimate["estimated_worker_tokens"]) * 2,
            "call_count": int(estimate["call_count"]) * 2,
            "estimation_mode": "new_agent_plus_structural_completion",
        }
        return self._insufficient_token_event(action, estimate)

    def _execution_time_admission(
        self,
        action: CanvasAction,
        candidate: MultiAgentGraph,
        mutation: MutationResult,
    ) -> dict[str, Any] | None:
        if (
            not self._time_admission_active()
            or not self._has_usable_artifact()
            or action.action_type
            in {
                # SET_PROMPT completes an ADD_AGENT that already passed the
                # admission gate. Rejecting it here would strand the Canvas at
                # its mandatory prompt barrier.
                ActionType.SET_PROMPT,
                ActionType.SET_MODEL,
                ActionType.ADD_AGENT,
                ActionType.DELETE_AGENT,
                ActionType.SET_OUTPUT,
                ActionType.FINISH,
            }
        ):
            return None
        estimate = self.runtime.estimate_execution_s(
            candidate,
            set(mutation.dirty_agents),
            quantile=self.config.worker_latency_quantile,
            minimum_samples=self.config.worker_latency_min_samples,
            cold_start_s=self.config.worker_latency_cold_start_s,
        )
        if int(estimate["call_count"]) <= 0:
            return None
        return self._insufficient_time_event(action, estimate)

    def _webshop_budget_partition(self, *, call_count: int, closure: bool) -> dict[str, Any]:
        return budget_partition(
            total_limit=self.config.max_total_tokens,
            spent=self.total_tokens,
            configured_minimum=self.config.finalization_token_reserve,
            call_count=call_count,
            closure=closure,
            observed_bounds=observed_request_bounds(list(self.runtime.artifacts.values())),
        )

    def _execution_token_admission(
        self,
        action: CanvasAction,
        candidate: MultiAgentGraph,
        mutation: MutationResult,
        *,
        webshop_closure_pass: bool = False,
    ) -> dict[str, Any] | None:
        if (
            # SWE uses the serialized request gate against the shared balance;
            # historical execution estimates must not withhold future credit.
            self.dataset == "swe_bench"
            or not self.config.remaining_token_admission_enabled
            or (self.dataset != "webshop" and not self._has_usable_artifact())
            or action.action_type
            in {
                ActionType.ADD_AGENT,
                ActionType.FINISH,
            }
            or (action.action_type is ActionType.SET_OUTPUT and not webshop_closure_pass)
            or (
                action.action_type is ActionType.SET_PROMPT
                and action.target == self.pending_agent_id
            )
        ):
            return None
        estimate = self.runtime.estimate_execution_tokens(
            candidate,
            ({str(action.target)} if webshop_closure_pass else set(mutation.dirty_agents)),
            quantile=self.config.worker_token_quantile,
            minimum_samples=self.config.worker_token_min_samples,
            cold_start_tokens=(
                max(self.config.worker_token_cold_start, 8_192)
                if webshop_closure_pass
                else self.config.worker_token_cold_start
            ),
            workload_scope_override=("webshop_closure" if webshop_closure_pass else ""),
            use_observed_agent_floor=not webshop_closure_pass,
        )
        structural_completion: dict[str, object] | None = None
        if webshop_closure_pass:
            partition = self._webshop_budget_partition(call_count=1, closure=True)
            self._token_admission_event.update(partition)
            # Historical usage and cold floors are advisory estimates, not the
            # serialized closure request. The gateway owns the sole hard quote.
            if partition["remaining_worker_tokens"] > 0:
                return None
            return {
                **partition,
                "blocked": True,
                "action": action.action_type.value,
                "estimated_worker_tokens": 0,
                "finalization_token_reserve": 0,
                "required_tokens": 1,
                "closure_execution": {
                    "model_request_count": 0,
                    "model_request_count_known": True,
                    "action_attempt_count": 0,
                    "stop_stage": "canvas_total_budget_exhausted",
                },
            }
        if (
            action.action_type is ActionType.SET_LAYER
            and self.structural_repair_reason == "relation_layer_mismatch"
            and self.structural_repair_pair is not None
            and self.structural_repair_relation is not None
        ):
            source, target = self.structural_repair_pair
            relation_graph = candidate.clone()
            try:
                relation_mutation = relation_graph.set_relation(
                    source,
                    target,
                    self.structural_repair_relation,
                )
            except GraphValidationError:
                relation_mutation = None
            if relation_mutation is not None:
                structural_completion = self.runtime.estimate_execution_tokens(
                    relation_graph,
                    set(relation_mutation.dirty_agents),
                    quantile=self.config.worker_token_quantile,
                    minimum_samples=self.config.worker_token_min_samples,
                    cold_start_tokens=self.config.worker_token_cold_start,
                )
                estimate = {
                    **estimate,
                    "estimated_worker_tokens": int(estimate["estimated_worker_tokens"])
                    + int(structural_completion["estimated_worker_tokens"]),
                    "call_count": int(estimate["call_count"])
                    + int(structural_completion["call_count"]),
                    "estimation_mode": "current_edit_plus_pending_relation",
                    "structural_completion": structural_completion,
                }
        if int(estimate["call_count"]) <= 0:
            return None
        return self._insufficient_token_event(
            action,
            estimate,
            reserve_tokens=(
                0
                if webshop_closure_pass
                else self._webshop_budget_partition(
                    call_count=int(estimate["call_count"]), closure=False
                )["reserved_closure_tokens"]
                if self.action_adapter is not None and self.action_adapter.adapter_id == "webshop"
                else None
            ),
        )

    def _time_admission_active(self) -> bool:
        return bool(
            self.config.remaining_time_admission_enabled and self.rollout_deadline is not None
        )

    def _has_usable_artifact(self) -> bool:
        return bool(self._usable_output_agents())

    def _uses_staged_environment_commit(self) -> bool:
        return bool(
            not self.unified and
            self.action_adapter is not None
            and self.action_adapter.commit_activation.value == "commit_pending_on_output"
        )

    def _artifact_is_usable_output(self, agent_id: str) -> bool:
        """Return whether an existing configured Agent has a real answer artifact."""

        node = self.graph.nodes.get(agent_id)
        artifact = self.runtime.artifacts.get(agent_id)
        if node is None or not node.configured or artifact is None:
            return False
        if (
            (
                self.dataset == "alfworld"
                or (
                    self.action_adapter is not None
                    and self.action_adapter.adapter_id == "alfworld"
                )
            )
            and isinstance(artifact.environment_result, dict)
            and isinstance(artifact.environment_result.get("environment_completed"), bool)
            and "won" in artifact.environment_result
        ):
            return True
        answer = str(artifact.answer or "").strip()
        return bool(
            answer
            and answer
            not in {
                WORKER_BACKEND_FAILURE_SENTINEL,
                WORKER_PROTOCOL_FAILURE_SENTINEL,
            }
        )

    def _swe_artifact_commit_ready(self, agent_id: str) -> bool:
        if self.action_adapter is None or self.action_adapter.adapter_id != "swe_bench":
            return True
        artifact = self.runtime.artifacts.get(agent_id)
        if artifact is None:
            return False
        progress = artifact.swe_progress
        if isinstance(progress, dict) and bool(progress.get("trusted")):
            return bool(progress.get("commit_ready"))
        # Compatibility for custom executors and historical trace replay: a
        # runtime-owned non-empty patch is commit-ready even when old artifacts
        # predate the Stage-5 progress schema.
        return artifact.code_artifact_ref is not None

    def _webshop_staged_revision_allowed(self, agent_id: str) -> bool:
        """A trusted staged WebShop transaction is an irreversible protocol latch."""

        del agent_id
        return False

    def _usable_output_agents(self) -> tuple[str, ...]:
        return tuple(
            agent_id
            for agent_id in sorted(self.graph.nodes)
            if self._artifact_is_usable_output(agent_id)
        )

    def _eligible_output_agents(self) -> tuple[str, ...]:
        usable = self._usable_output_agents()
        if (
            self.state is CanvasState.AWAITING_MODEL
            and self.pending_agent_id in self.graph.nodes
            and self.graph.nodes[self.pending_agent_id].prompt_configured
            and not self._uses_staged_environment_commit()
            and (
                not self.structural_exploration_required
                or self.structural_exploration_waived
                or self._structural_exploration_satisfied()
            )
            and (
                self.structural_repair_reason is None
                or ActionType.SET_OUTPUT in self._structural_repair_allowed_action_types()
            )
            and worker_output_role_changes_input(
                dataset=self.dataset,
                action_adapter=(
                    self.action_adapter.adapter_id if self.action_adapter is not None else ""
                ),
            )
        ):
            usable = (*usable, self.pending_agent_id)
        if not self._uses_staged_environment_commit():
            return usable
        ready = set(self.runtime.environment_commit_ready_agents())
        owners = set(self.runtime.environment_owner_agents())
        return tuple(agent_id for agent_id in usable if agent_id in ready or agent_id in owners)

    def _staged_commit_safe_delete_targets(self) -> tuple[str, ...]:
        """Nodes that cannot contribute to any latched environment candidate."""

        if not self._uses_staged_environment_commit():
            return ()
        candidates = set(self.runtime.environment_commit_ready_agents())
        owners = set(self.runtime.environment_owner_agents())
        if not candidates:
            return ()
        return tuple(
            agent_id
            for agent_id in sorted(self.graph.nodes)
            if agent_id not in owners
            and (
                not self.graph.nodes[agent_id].configured
                or agent_id not in self.runtime.artifacts
                or not any(
                    candidate in self.graph.reachable_from(agent_id) for candidate in candidates
                )
            )
        )

    def _output_health(self) -> str:
        output = self.graph.output_agent
        if output is None:
            return "missing"
        return "usable" if self._artifact_is_usable_output(output) else "unusable"

    def _native_webshop_task_status(self) -> dict[str, Any] | None:
        """Live public completion facts, independent of model claims and stale artifacts."""
        if not (self.runtime.native_webshop and self.dataset == "webshop"):
            return None
        ready = set(self.runtime.environment_commit_ready_agents())
        selected = self.graph.output_agent
        artifact = self.runtime.artifacts.get(selected)
        purchased = bool(
            artifact is not None
            and isinstance(artifact.environment_result, dict)
            and artifact.environment_result.get("purchased")
        )
        ledger = getattr(self.runtime.executor, "budget_ledger", None)
        scope = f"webshop-rollout:{self.runtime.environment_fingerprint}:whole-graph"
        if self.config.action_budget_policy == "shared_total_v1":
            scope = "tool-rollout:whole-graph"
        budgets = {}
        if ledger is not None:
            for agent_id, node in sorted(self.graph.nodes.items()):
                initial = ledger.remaining(node, revision=False, scope=scope)
                revision = ledger.remaining(node, revision=True, scope=scope)
                budgets[agent_id] = {
                    "initial_actions_remaining": min(initial["phase"], initial["total"]),
                    "revision_actions_remaining": min(revision["phase"], revision["total"]),
                    "total_actions_remaining": initial["total"],
                }
        return {
            "selected_output": selected,
            "selected_candidate_staged": selected in ready,
            "candidate_agents": sorted(ready),
            "purchase_committed": purchased,
            "completion_state": (
                "purchased"
                if purchased
                else "candidate_staged"
                if selected in ready
                else "no_candidate"
            ),
            "action_budget_scope": "whole_graph_shared_not_additive",
            "remaining_actions_by_agent": budgets,
        }

    def _prepare_native_output_materialization(self) -> None:
        """A single selected-output revision within the existing shared allowance.

        Ordinary graph edits and dirty-closure execution retain their semantics.
        No alternative Agent or purchase is chosen by the scheduler.
        """
        if not self.config.native_webshop_output_materialization or self.dirty_agents:
            return
        status = self._native_webshop_task_status()
        if status is None or status["selected_candidate_staged"] or status["purchase_committed"]:
            return
        selected = self.graph.output_agent
        if selected not in self.runtime.artifacts:
            return
        remaining = status["remaining_actions_by_agent"].get(selected, {})
        if remaining.get("revision_actions_remaining", 0) <= 0:
            return
        node = self.graph.nodes[selected]
        if node.metadata.get("_runtime_native_output_materialization_attempted"):
            return
        node.metadata["_runtime_native_output_materialization_attempted"] = True
        self.dirty_agents.add(selected)
        self.dirty_reasons.setdefault(selected, set()).add("selected_output_recovery_required")
        self._step_invalidated_agents.add(selected)
        self._step_invalidation_reasons.setdefault(selected, set()).add(
            "selected_output_recovery_required"
        )

    def _final_validation_error_codes(self) -> tuple[str, ...]:
        codes: set[str] = set()
        for error in self.graph.validate(final=True):
            if error == "graph is empty":
                codes.add("empty_graph")
            elif error.startswith("agents missing prompts:"):
                codes.add("unconfigured_agents")
            elif error == "output agent is not set":
                codes.add("output_not_set")
            elif error.startswith("agents cannot influence output:"):
                codes.add("output_unreachable")
            elif error.startswith("agent budget exceeded"):
                codes.add("agent_budget_exceeded")
            elif "relation" in error or "layer" in error:
                codes.add("invalid_relation")
            else:
                codes.add("other_graph_error")
        if self.graph.output_agent is not None and self._output_health() == "unusable":
            codes.add("output_artifact_unusable")
        return tuple(sorted(codes))

    def _repair_progress_signature(self) -> RepairProgressSignature | None:
        reason = self.structural_repair_reason
        if reason is None:
            return None
        unconfigured_count = sum(int(not node.configured) for node in self.graph.nodes.values())
        final_error_codes = self._final_validation_error_codes()
        return RepairProgressSignature(
            reason=reason,
            output_health=self._output_health(),
            final_error_codes=final_error_codes,
            unconfigured_count=unconfigured_count,
            unreachable_count=len(self._unreachable_to_output()),
            dirty_count=len(self.dirty_agents),
            final_ready=bool(
                not final_error_codes
                and not self.dirty_agents
                and self.state is CanvasState.BUILDING
            ),
        )

    @staticmethod
    def _repair_progress_score(signature: RepairProgressSignature) -> tuple[int, ...]:
        return (
            int(signature.final_ready),
            int(signature.output_health == "usable"),
            -len(signature.final_error_codes),
            -signature.unconfigured_count,
            -signature.unreachable_count,
            -signature.dirty_count,
        )

    def _record_repair_progress(
        self,
        action: CanvasAction,
        *,
        accepted: bool,
        rejection_code: str | None,
    ) -> None:
        connectivity_repair = (
            self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}
            or (self._connectivity_repair_active and not self._is_consolidation_repair())
        )
        if connectivity_repair:
            self._connectivity_repair_active = True
            signature = self._connectivity_state_signature()
            was_visited = signature in self._connectivity_repair_states
            if accepted:
                if was_visited:
                    self.semantic_no_progress_streak += 1
                else:
                    # A novel state may be a legitimate intermediate step, such
                    # as SET_LAYER before relation selection. Do not freeze it.
                    self.semantic_no_progress_streak = 0
                    self._connectivity_repair_states.add(signature)
            elif rejection_code == "connectivity_output_cycle":
                self._connectivity_cycle_blocks += 1
                self.semantic_no_progress_streak += 1
            self._recent_repair_actions.append({
                "action": action.action_type.value,
                "accepted": accepted,
                "rejection_code": rejection_code,
                "semantic_progress": not was_visited if accepted else False,
                "state_revisited": was_visited,
                "signature": {"state_hash": signature},
                "unreachable_agents": list(self._unreachable_to_output()),
            })
            self._recent_repair_actions = self._recent_repair_actions[
                -self.config.repair_recent_action_limit :
            ]
            if self.structural_repair_reason is None:
                self._connectivity_repair_active = False
                self._connectivity_repair_states.clear()
                self.semantic_no_progress_streak = 0
            return

        # Other repair classes retain their existing progress policy.
        if not self._is_output_lifecycle_repair():
            return
        current = self._repair_progress_signature()
        if current is None:
            return
        previous = self._last_repair_signature
        progress: bool | None
        if self._repair_entry_pending or previous is None:
            progress = None
            self._repair_entry_pending = False
            self.semantic_no_progress_streak = 0
        else:
            progress = self._repair_progress_score(current) > self._repair_progress_score(previous)
            if progress:
                self.semantic_no_progress_streak = 0
            else:
                self.semantic_no_progress_streak += 1
                if accepted and action.action_type is ActionType.SET_OUTPUT:
                    self.output_switches_without_progress += 1
        self._last_repair_signature = current
        self._recent_repair_actions.append(
            {
                "action": action.action_type.value,
                "accepted": accepted,
                "rejection_code": rejection_code,
                "semantic_progress": progress,
                "signature": current.to_dict(),
            }
        )
        self._recent_repair_actions = self._recent_repair_actions[
            -self.config.repair_recent_action_limit :
        ]
        if (
            self.semantic_no_progress_streak >= self.config.semantic_no_progress_limit
            and self.state is CanvasState.BUILDING
            and not self.topology_edits_frozen
        ):
            self.topology_edits_frozen = True
            self.semantic_no_progress_recovery_count += 1

    def _insufficient_time_event(
        self,
        action: CanvasAction,
        estimate: dict[str, object],
    ) -> dict[str, Any] | None:
        assert self.rollout_deadline is not None
        remaining_s = self.rollout_deadline.hard_remaining_s("canvas_time_admission")
        worker_s = float(estimate["estimated_worker_s"])
        reserve_s = max(
            self.config.finalization_time_reserve_s,
            self.config.finalization_time_reserve_by_dataset.get(self.dataset, 0.0),
        )
        required_s = worker_s + reserve_s
        if remaining_s >= required_s:
            return None
        return {
            "blocked": True,
            "action": action.action_type.value,
            "hard_remaining_s": remaining_s,
            "estimated_worker_s": worker_s,
            "finalization_reserve_s": reserve_s,
            "required_s": required_s,
            "quantile": self.config.worker_latency_quantile,
            **estimate,
        }

    def _insufficient_token_event(
        self,
        action: CanvasAction,
        estimate: dict[str, object],
        *,
        reserve_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        remaining_tokens = self.config.max_total_tokens - self.total_tokens
        worker_tokens = int(estimate["estimated_worker_tokens"])
        reserve_tokens = (
            self.config.finalization_token_reserve
            if reserve_tokens is None
            else max(0, int(reserve_tokens))
        )
        required_tokens = worker_tokens + reserve_tokens
        if remaining_tokens >= required_tokens:
            return None
        return {
            "blocked": True,
            "action": action.action_type.value,
            "remaining_worker_tokens": remaining_tokens,
            "estimated_worker_tokens": worker_tokens,
            "finalization_token_reserve": reserve_tokens,
            "required_tokens": required_tokens,
            "quantile": self.config.worker_token_quantile,
            **estimate,
        }

    def _reject_for_time_admission(
        self,
        action: CanvasAction,
        event: dict[str, Any],
    ) -> CanvasStep:
        self._time_admission_event = dict(event)
        self.time_budget_rejections += 1
        self._enter_structural_repair(
            "time_budget_consolidation",
            agents=tuple(self.graph.nodes),
        )
        self.structural_repair_blocked_actions += 1
        return self._record(
            action,
            accepted=False,
            feedback=(
                "Rejected action: STRUCTURAL_REPAIR_REQUIRED "
                "(reason=time_budget_consolidation; hard remaining "
                f"{event['hard_remaining_s']:.1f}s is below estimated Worker time "
                f"{event['estimated_worker_s']:.1f}s plus finalization reserve "
                f"{event['finalization_reserve_s']:.1f}s; use an existing output and FINISH)"
            ),
            rejection_code="time_budget_consolidation_required",
        )

    def _reject_for_token_admission(
        self,
        action: CanvasAction,
        event: dict[str, Any],
    ) -> CanvasStep:
        self._token_admission_event = dict(event)
        self.token_budget_admission_rejections += 1
        self._enter_structural_repair(
            "token_budget_consolidation",
            agents=tuple(self.graph.nodes),
        )
        self.structural_repair_blocked_actions += 1
        return self._record(
            action,
            accepted=False,
            feedback=(
                "Rejected action: STRUCTURAL_REPAIR_REQUIRED "
                "(reason=token_budget_consolidation; remaining Worker tokens "
                f"{event['remaining_worker_tokens']} are below estimated dirty execution "
                f"{event['estimated_worker_tokens']} plus finalization reserve "
                f"{event['finalization_token_reserve']}; use the existing graph and FINISH)"
            ),
            rejection_code="token_budget_admission_required",
        )

    def _begin_relation_decision(self, action: CanvasAction) -> CanvasStep:
        source, target, relation_type = self._canonical_relation_candidate(
            str(action.source), str(action.target)
        )
        key = (self.graph.version, tuple(sorted((source, target))))
        if key in self._considered_relation_pairs:
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: this Agent pair was already considered at the current "
                    "Canvas version; make another graph change before considering it again."
                ),
                rejection_code="relation_pair_already_considered",
            )
        pending = PendingRelationDecision(
            source=source,
            target=target,
            relation_type=relation_type,
            graph_version=self.graph.version,
            proposal_round_index=self.round_index,
        )
        self._considered_relation_pairs.add(key)
        self.pending_relation_decision = pending
        self.state = CanvasState.AWAITING_RELATION_CHOICE
        payload = {"phase": "proposal", **pending.to_dict()}
        self._last_relation_decision = dict(payload)
        return self._record(
            action,
            accepted=True,
            feedback=(
                f"Accepted relation proposal for {source} and {target}; Canvas inferred "
                f"{relation_type.value}. A constrained off/on policy choice is now required."
            ),
            relation_decision=payload,
        )

    def resolve_relation_choice(
        self,
        choice: str,
        *,
        policy_audit: dict[str, Any] | None = None,
        count_round: bool = True,
    ) -> CanvasStep:
        """Resolve one pending proposal without treating a no-op choice as failure."""

        pending = self.pending_relation_decision
        normalized = str(choice).strip().casefold()
        if pending is None or self.state is not CanvasState.AWAITING_RELATION_CHOICE:
            return self._record(
                CanvasAction(
                    ActionType.INVALID, raw_text=str(choice), parse_error="no pending relation"
                ),
                accepted=False,
                feedback="Rejected relation choice: no relation proposal is pending.",
                rejection_code="no_pending_relation_choice",
            )
        if count_round:
            if self.director_round_limit is not None and self.round_index >= self.director_round_limit:
                return self._record(
                    CanvasAction(
                        ActionType.INVALID,
                        raw_text=str(choice),
                        parse_error="relation choice exceeds Director round budget",
                    ),
                    accepted=False,
                    feedback="Rejected relation choice: Director round budget is exhausted.",
                    rejection_code="canvas_inactive",
                    relation_decision={"phase": "choice", **pending.to_dict()},
                )
            self.round_index += 1
        if normalized not in {"off", "on"}:
            return self._record(
                CanvasAction(
                    ActionType.INVALID,
                    raw_text=str(choice),
                    parse_error="relation choice must be exactly off or on",
                ),
                accepted=False,
                feedback="Rejected relation choice: expected exactly off or on.",
                rejection_code="invalid_relation_choice",
                relation_decision={"phase": "choice", **pending.to_dict()},
            )
        if pending.graph_version != self.graph.version:
            self.pending_relation_decision = None
            self.state = CanvasState.BUILDING
            return self._record(
                CanvasAction(
                    ActionType.INVALID, raw_text=normalized, parse_error="stale relation choice"
                ),
                accepted=False,
                feedback="Rejected relation choice: the Canvas version changed after proposal.",
                rejection_code="stale_relation_choice",
                relation_decision={"phase": "choice", **pending.to_dict()},
            )

        desired_present = normalized == "on"
        pair = tuple(sorted((pending.source, pending.target)))
        currently_present = (
            pair in self.graph.bidirectional_edges
            if pending.relation_type is RelationType.BIDIRECTIONAL
            else (pending.source, pending.target) in self.graph.directed_edges
        )
        decision_payload = {
            "phase": "choice",
            **pending.to_dict(),
            "choice": normalized,
            "chosen_present": desired_present,
            "previously_present": currently_present,
            "policy": dict(policy_audit or {}),
        }
        self._last_relation_decision = dict(decision_payload)
        self.pending_relation_decision = None
        self.state = CanvasState.BUILDING
        if desired_present == currently_present:
            action = CanvasAction(
                ActionType.SET_RELATION if desired_present else ActionType.REMOVE_RELATION,
                source=pending.source,
                target=pending.target,
                relation=pending.relation_type,
                expected_version=pending.graph_version,
                raw_text=normalized,
            )
            return self._record(
                action,
                accepted=True,
                feedback=(
                    f"Accepted relation choice {normalized}; the inferred "
                    f"{pending.relation_type.value} relation already had that state, so the "
                    "Canvas version and Worker inputs are unchanged."
                ),
                relation_decision=decision_payload,
            )

        action_name = "set_relation" if desired_present else "remove_relation"
        raw_action = json.dumps(
            {
                "action": action_name,
                "source": pending.source,
                "target": pending.target,
                "relation": pending.relation_type.value,
                "expected_version": pending.graph_version,
            },
            separators=(",", ":"),
        )
        step = self.step(raw_action, count_round=False, authoritative_director=False)
        step.action.raw_text = normalized
        step.relation_decision = decision_payload
        return step

    def abandon_relation_choice(self, reason: str) -> CanvasStep:
        """Fail closed when a backend cannot provide auditable binary probabilities."""

        pending = self.pending_relation_decision
        if pending is None:
            raise RuntimeError("no pending relation decision to abandon")
        self.pending_relation_decision = None
        self.state = CanvasState.BUILDING
        return self._record(
            CanvasAction(ActionType.INVALID, raw_text="", parse_error=str(reason)),
            accepted=False,
            feedback=(
                "Relation proposal was abandoned without changing the graph because the backend "
                "could not provide an auditable constrained off/on policy choice."
            ),
            rejection_code="relation_binary_policy_unavailable",
            protocol_recovery=True,
            relation_decision={"phase": "abandoned", **pending.to_dict(), "reason": str(reason)},
        )

    def _canonical_relation_candidate(
        self, source: str, target: str
    ) -> tuple[str, str, RelationType]:
        source_node = self.graph.require_node(source)
        target_node = self.graph.require_node(target)
        if source == target:
            raise GraphValidationError("self relations are not allowed")
        if source_node.layer == target_node.layer:
            first, second = sorted((source, target))
            return first, second, RelationType.BIDIRECTIONAL
        if source_node.layer < target_node.layer:
            return source, target, RelationType.DIRECTED
        return target, source, RelationType.DIRECTED

    def _apply(
        self,
        graph: MultiAgentGraph,
        action: CanvasAction,
        *,
        compilation: DelegationCompilation | None = None,
    ) -> MutationResult:
        kind = action.action_type
        if kind is ActionType.ADD_AGENT:
            mutation = graph.add_agent(action.agent_id)
            agent_id = next(iter(mutation.dirty_agents))
            if self.unified:
                from uuid import uuid4
                graph.nodes[agent_id].metadata.update(submission_protocol=PROTOCOL, result_scope="subtask", incarnation_id=uuid4().hex, task_dataset=self.dataset)
            if self.action_adapter is not None:
                configured = graph.configure_action_environment(
                    agent_id,
                    adapter_id=self.action_adapter.adapter_id,
                    action_names=self.action_adapter.action_names,
                    initial_action_budget=self.action_adapter.initial_action_budget,
                    revision_action_budget=self.action_adapter.revision_action_budget,
                    total_action_budget=self.action_adapter.total_action_budget,
                    capability_policy=self.action_adapter.capability_policy(),
                )
                mutation.dirty_agents.update(configured.dirty_agents)
                mutation.message += "; " + configured.message
            return mutation
        if kind is ActionType.SET_PROMPT:
            prompt = (
                compilation.prompt if compilation is not None else self._delegation_prompt(action)
            )
            if action.runtime_route is not None:
                raise GraphValidationError("use SET_MODEL to select a Worker model")
            metadata_updates = compilation.metadata() if compilation is not None else {}
            if self.unified:
                metadata_updates["result_scope"] = action.result_scope
            mutation = graph.set_prompt(
                str(action.target),
                prompt,
                metadata_updates=metadata_updates,
            )
            if compilation is not None and compilation.field_repairs:
                repaired_fields = ", ".join(
                    sorted({repair.field for repair in compilation.field_repairs})
                )
                mutation.message += f"; deterministically repaired fields: {repaired_fields}"
            if compilation is not None and compilation.contract_version:
                mutation.message += (
                    f"; appended {compilation.contract_version} for {compilation.dataset}"
                )
            return mutation
        if kind is ActionType.SET_MODEL:
            if action.runtime_route not in self.runtime_routes:
                raise GraphValidationError(f"unknown Worker runtime_route: {action.runtime_route}")
            return graph.set_model(str(action.target), str(action.runtime_route))
        if kind is ActionType.SET_LAYER:
            return graph.set_layer(str(action.target), int(action.layer))
        if kind is ActionType.SET_RELATION:
            return graph.set_relation(str(action.source), str(action.target), action.relation)
        if kind is ActionType.REMOVE_RELATION:
            return graph.remove_relation(str(action.source), str(action.target), action.relation)
        if kind is ActionType.DELETE_AGENT:
            return graph.delete_agent(str(action.target or action.agent_id))
        if kind is ActionType.SET_OUTPUT:
            mutation = graph.set_output(str(action.target))
            if (
                self.action_adapter is not None
                and self.action_adapter.commit_policy.value == "single_committer"
                and not (self.runtime.native_webshop and self.dataset == "webshop")
            ):
                capability_name = (
                    "environment_commit"
                    if self.action_adapter.commit_activation.value
                    in {"execute_on_output", "commit_pending_on_output"}
                    else "code_commit"
                )
                capability = graph.assign_exclusive_capability(str(action.target), capability_name)
                mutation.message += "; " + capability.message
                if self.action_adapter.commit_activation.value == "execute_on_output":
                    mutation.dirty_agents.update(capability.dirty_agents)
            return mutation
        raise GraphValidationError(f"unsupported canvas action: {kind.value}")

    @staticmethod
    def _mutation_invalidation_reasons(
        action: CanvasAction,
        mutation: MutationResult,
        *,
        previous_graph: MultiAgentGraph,
        current_graph: MultiAgentGraph,
    ) -> dict[str, set[str]]:
        dirty = set(mutation.dirty_agents) & set(current_graph.nodes)
        if not dirty:
            return {}
        if action.action_type is ActionType.ADD_AGENT:
            return {agent_id: {"new_agent_initial"} for agent_id in dirty}
        if action.action_type is ActionType.SET_PROMPT:
            target = str(action.target)
            component = current_graph.bidirectional_component(target)
            reasons: dict[str, set[str]] = {}
            for agent_id in dirty:
                if agent_id == target:
                    reasons[agent_id] = {"prompt_changed"}
                elif agent_id in component:
                    reasons[agent_id] = {"peer_changed"}
                else:
                    reasons[agent_id] = {"upstream_changed"}
            return reasons
        if action.action_type is ActionType.SET_LAYER:
            target = str(action.target)
            peer_agents: set[str] = set()
            for graph in (previous_graph, current_graph):
                if target in graph.nodes:
                    peer_agents.update(graph.bidirectional_component(target))
            return {
                agent_id: {
                    "layer_changed"
                    if agent_id == target
                    else "peer_changed"
                    if agent_id in peer_agents
                    else "upstream_changed"
                }
                for agent_id in dirty
            }
        if action.action_type in {
            ActionType.SET_RELATION,
            ActionType.REMOVE_RELATION,
        }:
            if action.relation is RelationType.BIDIRECTIONAL:
                peer_agents: set[str] = set()
                for graph in (previous_graph, current_graph):
                    for endpoint in (str(action.source), str(action.target)):
                        if endpoint in graph.nodes:
                            peer_agents.update(graph.bidirectional_component(endpoint))
                return {
                    agent_id: {"peer_changed" if agent_id in peer_agents else "upstream_changed"}
                    for agent_id in dirty
                }
            return {agent_id: {"upstream_changed"} for agent_id in dirty}
        if action.action_type is ActionType.DELETE_AGENT:
            target = str(action.target or action.agent_id)
            former_peers = (
                previous_graph.bidirectional_component(target) - {target}
                if target in previous_graph.nodes
                else set()
            )
            return {
                agent_id: {"peer_changed" if agent_id in former_peers else "upstream_changed"}
                for agent_id in dirty
            }
        if action.action_type is ActionType.SET_OUTPUT:
            return {agent_id: {"commit_activation"} for agent_id in dirty}
        return {agent_id: {mutation.change_reason or "environment_changed"} for agent_id in dirty}

    def _prompt_revision_checkpoint(self, target: str) -> dict[str, Any]:
        upstream_ids = sorted(self.graph.directed_predecessors(target))
        peer_ids = sorted(self.graph.bidirectional_component(target) - {target})

        def artifact_ids(agent_ids: list[str]) -> dict[str, str]:
            return {
                agent_id: (
                    self.runtime.artifacts[agent_id].artifact_id
                    if agent_id in self.runtime.artifacts
                    else ""
                )
                for agent_id in agent_ids
            }

        return {
            "upstream_artifacts": artifact_ids(upstream_ids),
            "peer_artifacts": artifact_ids(peer_ids),
            "topology": {
                "layer": self.graph.nodes[target].layer,
                "upstream_agent_ids": upstream_ids,
                "peer_agent_ids": peer_ids,
            },
            "repair_epoch": self.repair_epoch,
        }

    @staticmethod
    def _revision_evidence_signature(kind: str, payload: object) -> str:
        encoded = json.dumps(
            {"kind": kind, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode()
        return f"{kind}:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def _stable_tool_evidence(cls, value: object) -> object:
        volatile = {
            "call_id",
            "elapsed_ms",
            "remaining_budget",
            "round_index",
            "token_in",
            "token_out",
            "raw_response",
        }
        if isinstance(value, dict):
            return {
                str(key): cls._stable_tool_evidence(item)
                for key, item in value.items()
                if str(key) not in volatile
            }
        if isinstance(value, (list, tuple)):
            return [cls._stable_tool_evidence(item) for item in value]
        return value

    def _eligible_prompt_revision_evidence(self, target: str) -> dict[str, Any]:
        if target not in self.graph.nodes or not self.graph.nodes[target].configured:
            return {"public": {}, "internal": {}}
        current = self._prompt_revision_checkpoint(target)
        checkpoint = self._prompt_revision_checkpoints.get(target, current)
        consumed = self._prompt_revision_consumed.get(target, set())
        internal: dict[str, dict[str, list[str]]] = {}

        def add(kind: PromptRevisionBasis, agent_id: str, payload: object) -> None:
            signature = self._revision_evidence_signature(kind.value, payload)
            if signature in consumed:
                return
            internal.setdefault(kind.value, {}).setdefault(agent_id, []).append(signature)

        for kind, checkpoint_field in (
            (PromptRevisionBasis.UPSTREAM_ARTIFACT_CHANGED, "upstream_artifacts"),
            (PromptRevisionBasis.PEER_ARTIFACT_CHANGED, "peer_artifacts"),
        ):
            before = checkpoint[checkpoint_field]
            after = current[checkpoint_field]
            for agent_id, artifact_id in after.items():
                if before.get(agent_id) != artifact_id:
                    add(
                        kind,
                        agent_id,
                        {"agent_id": agent_id, "artifact_id": artifact_id},
                    )

        artifact = self.runtime.artifacts.get(target)
        if artifact is not None:
            for trace_item in artifact.react_trace:
                if not isinstance(trace_item, dict):
                    continue
                observation = trace_item.get("observation")
                if not isinstance(observation, dict):
                    continue
                output = observation.get("output")
                output_error = bool(
                    isinstance(output, dict) and str(output.get("status", "")).casefold() == "error"
                )
                if str(observation.get("status", "")).casefold() == "error" or output_error:
                    add(
                        PromptRevisionBasis.TOOL_ERROR,
                        target,
                        {
                            "action": self._stable_tool_evidence(trace_item.get("action")),
                            "observation": self._stable_tool_evidence(observation),
                        },
                    )
            for issue in artifact.unresolved_issues:
                add(PromptRevisionBasis.UNRESOLVED_ISSUE, target, str(issue))
            protocol_evidence: list[object] = []
            if is_aime_dataset(self.dataset) and target == self.graph.output_agent:
                submitted = parse_aime_answer(artifact.answer)
                if not submitted.valid:
                    # Public answer-format evidence permits an optional revision;
                    # it is not a Worker protocol failure or a FINISH prerequisite.
                    add(
                        PromptRevisionBasis.UNRESOLVED_ISSUE,
                        target,
                        {
                            "stage": "answer_format",
                            "artifact_id": artifact.artifact_id,
                            "format_reason": submitted.reason,
                        }
                    )
            protocol_status = summarize_worker_protocol(
                answer=artifact.answer,
                raw_response=artifact.raw_response,
                diagnostics=artifact.protocol_diagnostics,
            )
            if protocol_status["status"] == "failed":
                protocol_evidence.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "stage": protocol_status["stage"],
                        "error_code": protocol_status["error_code"],
                    }
                )
            for evidence in protocol_evidence:
                add(PromptRevisionBasis.PROTOCOL_FAILURE, target, evidence)

        if current["topology"] != checkpoint["topology"]:
            add(
                PromptRevisionBasis.STRUCTURAL_ROLE_CHANGE,
                target,
                current["topology"],
            )
        if int(current["repair_epoch"]) > int(checkpoint["repair_epoch"]):
            add(
                PromptRevisionBasis.CONTROLLER_REPAIR,
                target,
                {
                    "repair_epoch": current["repair_epoch"],
                    "target": target,
                },
            )

        public = {
            basis: {
                "evidence_agent_ids": sorted(by_agent),
                "revision_count": self.prompt_revision_counts.get(target, 0),
            }
            for basis, by_agent in sorted(internal.items())
            if by_agent
        }
        return {"public": public, "internal": internal}

    def _validate_prompt_revision_evidence(
        self,
        action: CanvasAction,
    ) -> tuple[dict[str, Any] | None, str | None]:
        target = str(action.target)
        if action.revision_basis is None:
            return None, (
                f"SET_PROMPT for configured Agent {target} requires revision_basis and "
                "Canvas-verifiable evidence_agent_ids"
            )
        eligibility = self._eligible_prompt_revision_evidence(target)
        basis = action.revision_basis.value
        eligible = eligibility["internal"].get(basis, {})
        if not eligible:
            return None, (
                f"revision_basis={basis} has no new Canvas-verifiable evidence for {target}"
            )
        evidence_agent_ids = tuple(action.evidence_agent_ids)
        if not evidence_agent_ids:
            return None, "prompt revision requires at least one evidence_agent_id"
        unknown = sorted(set(evidence_agent_ids) - set(eligible))
        if unknown:
            return None, (
                f"evidence_agent_ids are not eligible for revision_basis={basis}: "
                + ", ".join(unknown)
            )
        signatures = sorted(
            {signature for agent_id in evidence_agent_ids for signature in eligible[agent_id]}
        )
        return (
            {
                "target": target,
                "basis": basis,
                "evidence_agent_ids": list(evidence_agent_ids),
                "evidence_signatures": signatures,
            },
            None,
        )

    def _commit_prompt_revision_evidence(
        self,
        action: CanvasAction,
        evidence: dict[str, Any],
    ) -> None:
        target = str(action.target)
        self._prompt_revision_consumed.setdefault(target, set()).update(
            str(value) for value in evidence["evidence_signatures"]
        )
        revision_index = self.prompt_revision_counts.get(target, 0) + 1
        self.prompt_revision_counts[target] = revision_index
        self._step_prompt_revision = {
            "target": target,
            "revision_index": revision_index,
            "basis": str(evidence["basis"]),
            "evidence_agent_ids": list(evidence["evidence_agent_ids"]),
            "evidence_signature_count": len(evidence["evidence_signatures"]),
            "target_model_calls": 0,
            "target_cache_hits": 0,
            "worker_model_calls_total": 0,
        }

    def _finalize_prompt_assignment(
        self,
        action: CanvasAction,
        report: ExecutionReport | None,
    ) -> None:
        if action.action_type not in {ActionType.SET_PROMPT, ActionType.SET_MODEL}:
            return
        target = str(action.target)
        if target not in self.graph.nodes or not self.graph.nodes[target].configured:
            return
        self._prompt_revision_checkpoints[target] = self._prompt_revision_checkpoint(target)
        if not self._step_prompt_revision:
            return
        events = report.execution_events if report is not None else []
        self._step_prompt_revision.update(
            {
                "target_model_calls": sum(
                    int(event.get("agent_id") == target and not bool(event.get("cache_hit")))
                    for event in events
                ),
                "target_cache_hits": sum(
                    int(event.get("agent_id") == target and bool(event.get("cache_hit")))
                    for event in events
                ),
                "worker_model_calls_total": (
                    report.worker_model_calls_total if report is not None else 0
                ),
            }
        )
        self.prompt_revision_history.append(dict(self._step_prompt_revision))

    def _completes_prompt(self, action: CanvasAction) -> bool:
        return (
            action.action_type is ActionType.SET_PROMPT and action.target == self.pending_agent_id
        )

    def _delegation_prompt(self, action: CanvasAction) -> str:
        if action.prompt is not None:
            return action.prompt
        return "\n".join(
            (
                f"Role: {action.role}",
                f"Objective: {action.objective}",
                f"Scope: {action.scope}",
                f"Expected output: {action.expected_output}",
            )
        )

    def _validate_delegation(self, action: CanvasAction) -> DelegationCompilation | None:
        if action.prompt is not None:
            if self.allow_legacy_prompts:
                return None
            raise GraphValidationError(
                "SET_PROMPT must use short role, objective, scope, and expected_output fields; "
                "free-form prompt is forbidden"
            )
        fields = {
            "role": action.role,
            "objective": action.objective,
            "scope": action.scope,
            "expected_output": action.expected_output,
        }
        if action.structural_operator is not None:
            raise GraphValidationError(
                "SET_PROMPT must not assign a fixed structural_operator; use the free-text role"
            )
        action_names = set(self.action_adapter.action_names if self.action_adapter else ())
        compilation, issue = compile_delegation(
            fields,
            public_task=self.task,
            dataset=self.dataset if self.managed_delegation_contracts else "",
            action_names=action_names,
            webshop_native=self.runtime.native_webshop and self.dataset == "webshop",
        )
        if issue is not None:
            raise DelegationValidationError(issue)
        assert compilation is not None
        alignment_issue = delegation_task_alignment_issue(
            compilation.director_fields,
            public_task=self.task,
            dataset=self.dataset,
        )
        if alignment_issue is not None:
            raise DelegationValidationError(alignment_issue)
        overlap_issue = self._duplicate_responsibility_issue(
            target=str(action.target),
            fields=compilation.director_fields,
        )
        if overlap_issue is not None:
            raise DelegationValidationError(overlap_issue)
        # Keep the accepted Canvas action aligned with the compiled responsibility.
        # ``raw_text`` remains the exact model output for observability.
        for name, value in compilation.director_fields.items():
            setattr(action, name, value)
        return compilation

    def _duplicate_responsibility_issue(
        self,
        *,
        target: str,
        fields: dict[str, str],
    ) -> DelegationIssue | None:
        """Reject only high-confidence duplicate SWE implementation ownership."""

        if self.duplicate_responsibility_policy == "off":
            self._step_responsibility_overlap_check = {
                "target": target,
                "policy": "off",
                "decision": "not_checked",
                "comparisons": [],
            }
            return None

        candidate_signature = responsibility_signature(fields)
        comparisons: list[dict[str, Any]] = []
        for agent_id, node in sorted(self.graph.nodes.items()):
            if agent_id == target:
                continue
            existing_fields = node.metadata.get("director_delegation")
            if not isinstance(existing_fields, dict):
                continue
            comparison = compare_responsibilities(existing_fields, fields)
            comparisons.append(
                {
                    "agent_id": agent_id,
                    **comparison.to_dict(),
                }
            )
        comparisons.sort(key=lambda item: float(item["similarity"]), reverse=True)
        strongest = comparisons[0] if comparisons else None
        is_swe = self.dataset in {"swe_bench", "swe-bench", "swebench"} or bool(
            self.action_adapter is not None and self.action_adapter.adapter_id == "swe_bench"
        )
        high_confidence_duplicate = bool(
            is_swe and strongest and strongest["high_confidence_duplicate"]
        )
        warning_key = (
            self.graph.version,
            target,
            str(strongest["agent_id"]) if strongest else "",
        )
        warned_before = warning_key in self._duplicate_responsibility_warnings
        if not high_confidence_duplicate:
            decision = (
                "overlap_warning"
                if strongest and float(strongest["scope_overlap"]) >= 0.55
                else "accepted"
            )
        elif self.duplicate_responsibility_policy == "record_only":
            decision = "record_only"
        elif self.duplicate_responsibility_policy == "warn_once" and warned_before:
            decision = "accepted_after_warning"
        elif self.duplicate_responsibility_policy == "warn_once":
            decision = "rewrite_requested"
        else:
            decision = "rejected"
        self._step_responsibility_overlap_check = {
            "target": target,
            "policy": self.duplicate_responsibility_policy,
            "candidate_primary_type": candidate_signature.primary_type,
            "decision": decision,
            "comparisons": comparisons[:8],
        }
        if not high_confidence_duplicate or self.duplicate_responsibility_policy in {
            "record_only",
        }:
            return None
        if self.duplicate_responsibility_policy == "warn_once" and warned_before:
            return None
        if self.duplicate_responsibility_policy == "warn_once":
            self._duplicate_responsibility_warnings.add(warning_key)
        assert strongest is not None
        conflicting_agent = str(strongest["agent_id"])
        details = {
            "target": target,
            "conflicting_agent_id": conflicting_agent,
            "candidate_primary_type": candidate_signature.primary_type,
            **{key: value for key, value in strongest.items() if key != "agent_id"},
        }
        return DelegationIssue(
            code="duplicate_responsibility",
            field="scope",
            message=(
                f"SET_PROMPT duplicates the implementation responsibility owned by "
                f"{conflicting_agent}; narrow the scope, assign a distinct diagnosis, "
                "testing, review, or synthesis contribution, or reuse the existing Agent"
            ),
            details=details,
        )

    def _feedback(self, event: str, report: ExecutionReport | None) -> str:
        facts = [event, self.graph.describe()]
        if self.dirty_agents:
            facts.append("Dirty agents: " + ", ".join(sorted(self.dirty_agents)))
        if report is not None:
            if report.blocked_agents:
                facts.append("Execution blocked: " + "; ".join(
                    f"{agent_id}={reason}" for agent_id, reason in sorted(report.blocked_agents.items())
                ) + ". No new Worker request or workspace was started for the blocked phase.")
            facts.append(
                "Execution: ran=[{}], reused=[{}], tokens={}/{}.".format(
                    ", ".join(report.executed_agents),
                    ", ".join(report.reused_agents),
                    self.total_tokens,
                    self.config.max_total_tokens,
                )
            )
            if report.errors:
                facts.append(
                    "Execution errors: "
                    + "; ".join(
                        f"{key}={self._one_line(value, 160)}"
                        for key, value in sorted(report.errors.items())
                    )
                    + "."
                )
            if report.revision_decisions:
                revision_facts = []
                for decision in report.revision_decisions:
                    agents = ",".join(decision.get("component_agents", []))
                    outcome = "ran" if bool(decision.get("revision_required")) else "skipped"
                    reasons = ",".join(decision.get("reason_codes", []))
                    revision_facts.append(f"[{agents}]={outcome}({reasons})")
                facts.append("Peer revision decisions: " + "; ".join(revision_facts) + ".")
            signals = []
            for agent_id in sorted(set(report.executed_agents)):
                artifact = report.artifacts.get(agent_id)
                if artifact is None:
                    continue
                if self.dataset == "alfworld":
                    public_task = (artifact.environment_result or {}).get("public_task_statement")
                    if public_task:
                        facts.append(
                            f"ALFWorld public reset task for {agent_id}: "
                            + json.dumps(public_task, ensure_ascii=False)
                            + ". This is the environment's public task statement; "
                            "the original dataset task remains unchanged."
                        )
                tool_errors = int(artifact.runtime_tool_evidence.get("failed_count", 0) or 0)
                summary = self._one_line(
                    artifact.summary or artifact.answer,
                    self.config.artifact_summary_max_chars,
                )
                unresolved = [
                    self._one_line(value, 120) for value in artifact.unresolved_issues[:2]
                ]
                swe_progress = artifact.swe_progress or {}
                webshop_progress = artifact.webshop_progress or {}
                webshop_action_budget = webshop_progress.get("action_budget") or {}
                webshop_recovery = None
                if webshop_action_budget.get("transfer_status") not in (None, "not_requested"):
                    webshop_recovery = (
                        "The once-only same-session output-closure pass has ended. "
                        "Any unspent transferred Action allowance is no longer spendable."
                        if webshop_action_budget.get("transferred")
                        else "The output-closure attempt has ended; no further Worker Action "
                        "pass is available."
                    )
                    budget_facts = {
                        key: webshop_action_budget.get(key)
                        for key in (
                            "transferred",
                            "transfer_status",
                            "initial_used",
                            "revision_used",
                            "initial_transferred",
                            "revision_transferred",
                            "closure_limit",
                            "closure_used",
                            "closure_unspent",
                            "total_used",
                            "total_limit",
                            "total_remaining",
                            "environment_remaining",
                            "closure_active",
                        )
                    }
                    facts.append(
                        f"WebShop Action budget for {agent_id}: "
                        + json.dumps(budget_facts, ensure_ascii=False)
                        + "."
                    )
                elif webshop_progress.get("state") == "typed_policy_failure":
                    webshop_recovery = (
                        "This Agent remains the sole owner of the current WebShop episode. "
                        "Use only a grounded same-owner revision or select it for the one "
                        "bounded output-closure pass; no replacement session is legal."
                    )
                elif (
                    self.dataset == "webshop"
                    and not self.runtime.native_webshop
                    and not bool(webshop_progress.get("commit_ready", False))
                ):
                    webshop_recovery = (
                        "No purchase is staged. The sole owner may be revised in the same "
                        "session, or SET_OUTPUT may select it for one bounded same-session "
                        "closure pass using only the remaining episode budget."
                    )
                if self.unified and self.dataset == "webshop":
                    webshop_recovery = (
                        "Worker reports do not terminate a live shopping session. "
                        "Use RUN_AGENT for admissible bounded continuation; FINISH(target) "
                        "submits only a current valid candidate or a runtime-confirmed terminal result."
                    )
                signals.append(
                    f"{agent_id}(summary={json.dumps(summary, ensure_ascii=False)}, "
                    f"confidence={artifact.confidence:.2f}, "
                    f"claimed_confidence={float(artifact.claimed_confidence or 0.0):.2f}, "
                    f"unresolved={len(artifact.unresolved_issues)}, "
                    f"unresolved_preview={json.dumps(unresolved, ensure_ascii=False)}, "
                    f"evidence={len(artifact.evidence)}, tool_errors={tool_errors}, "
                    f"integrity_risks={json.dumps(artifact.integrity_risks)}, "
                    f"swe_state={json.dumps(swe_progress.get('state'))}, "
                    f"swe_commit_ready={bool(swe_progress.get('commit_ready', False))}, "
                    f"webshop_state={json.dumps(webshop_progress.get('state'))}, "
                    f"webshop_commit_ready="
                    f"{bool(webshop_progress.get('commit_ready', False))}, "
                    f"webshop_recovery={json.dumps(webshop_recovery)}, "
                    f"backend_failure={artifact.answer == 'WORKER_BACKEND_FAILURE'})"
                )
            if signals:
                facts.append("Process signals: " + "; ".join(signals) + ".")
        return "\n".join(facts)

    def _unreachable_to_output(self) -> tuple[str, ...]:
        output = self.graph.output_agent
        if output is None or output not in self.graph.nodes:
            return ()
        return tuple(
            sorted(
                agent_id
                for agent_id in self.graph.nodes
                if agent_id != output and output not in self.graph.reachable_from(agent_id)
            )
        )

    def _weak_component_count(self) -> int:
        remaining = set(self.graph.nodes)
        count = 0
        adjacency = {agent_id: set() for agent_id in remaining}
        for source, target in self.graph.directed_edges | self.graph.bidirectional_edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        while remaining:
            count += 1
            frontier = [remaining.pop()]
            while frontier:
                current = frontier.pop()
                unseen = adjacency[current] & remaining
                remaining -= unseen
                frontier.extend(unseen)
        return count

    @staticmethod
    def _unreachable_for_graph(graph: MultiAgentGraph, output: str | None) -> tuple[str, ...]:
        if output is None or output not in graph.nodes:
            return ()
        return tuple(sorted(
            agent_id for agent_id in graph.nodes
            if agent_id != output and output not in graph.reachable_from(agent_id)
        ))

    def _connectivity_state_signature(self, *, output: str | None = None) -> str:
        """Stable repair state; excludes counters and output-role bookkeeping metadata."""
        graph = self.graph.clone()
        if output is not None and output != graph.output_agent:
            graph.set_output(output)
        payload = graph.to_dict()
        payload.pop("version", None)
        layer_rank = {
            value: rank for rank, value in enumerate(
                sorted({node.layer for node in graph.nodes.values()})
            )
        }
        for node in payload.get("nodes", []):
            node["layer"] = layer_rank.get(int(node.get("layer", 0)), 0)
            metadata = node.get("metadata", {})
            metadata.pop("_runtime_is_output_agent", None)
        artifacts = {}
        for agent_id, artifact in sorted(self.runtime.artifacts.items()):
            artifacts[agent_id] = {
                "answer": artifact.answer,
                "summary": artifact.summary,
                "evidence": list(artifact.evidence),
                "environment_result": artifact.environment_result,
                "code_artifact_sha256": (
                    artifact.code_artifact_ref.artifact_sha256
                    if hasattr(artifact.code_artifact_ref, "artifact_sha256") else None
                ),
            }
        return hashlib.sha256(json.dumps(
            {"graph": payload, "artifacts": artifacts},
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")).hexdigest()

    def _output_switch_cycle(self, target: str) -> dict[str, Any]:
        current = set(self._unreachable_to_output())
        projected = set(self._unreachable_for_graph(self.graph, target))
        resolves = not projected
        improves = projected < current
        signature = self._connectivity_state_signature(output=target)
        repeated = signature in self._connectivity_repair_states
        blocked = bool(repeated and not improves and not resolves)
        return {
            "target": target,
            "unreachable_before": sorted(current),
            "unreachable_after": sorted(projected),
            "improves_reachability": improves,
            "resolves_reachability": resolves,
            "returns_to_visited_state": repeated,
            "blocked_as_cycle": blocked,
        }

    def topology_audit(self) -> dict[str, Any]:
        relation_count = len(self.graph.directed_edges) + len(self.graph.bidirectional_edges)
        unreachable = self._unreachable_to_output()
        return {
            "agent_count": len(self.graph.nodes),
            "configured_agent_count": sum(
                int(node.configured) for node in self.graph.nodes.values()
            ),
            "relation_count": relation_count,
            "weak_component_count": self._weak_component_count(),
            "weakly_disconnected": self._weak_component_count() > 1,
            "actual_relations": (
                [
                    {"source": source, "target": target, "relation": "directed"}
                    for source, target in sorted(self.graph.directed_edges)
                ]
                + [
                    {"source": source, "target": target, "relation": "bidirectional"}
                    for source, target in sorted(self.graph.bidirectional_edges)
                ]
            ),
            "output_agent": self.graph.output_agent,
            "unreachable_agents": list(unreachable),
            "all_agents_reach_output": (
                not unreachable if self.graph.output_agent is not None else None
            ),
            "disconnected_multi_agent": (len(self.graph.nodes) > 1 and relation_count == 0),
        }

    def _graph_state_snapshot(self) -> dict[str, Any]:
        # Snapshots also enter feedback/history sent to the Director. Keep the
        # sampled choice here, but retain probability/request audit details only
        # in the original relation_decision event and DirectorTurn records.
        relation_facts = {
            key: value for key, value in self._last_relation_decision.items()
            if key != "policy"
        }
        nodes = [
            {"id": agent_id, "layer": self.graph.nodes[agent_id].layer,
             "configured": self.graph.nodes[agent_id].configured}
            for agent_id in sorted(self.graph.nodes)
        ]
        actual_relations = self.topology_audit()["actual_relations"]
        if self.unified:
            targets = [key for key, node in self.graph.nodes.items()
                       if node.metadata.get("result_scope") == "task_result"]
            validation = {}
            for target in targets:
                candidate = self.graph.clone()
                candidate.output_agent = target
                validation[target] = {
                    "unreachable_nodes": list(self._unreachable_for_graph(candidate, target)),
                    "graph_blockers": list(candidate.validate(final=True)),
                }
            return {
                "nodes": [{**node, "result_scope": self.graph.nodes[node["id"]].metadata.get("result_scope")} for node in nodes],
                "actual_relations": actual_relations,
                "candidate_relations_are_not_edges": True,
                "submission_target": self.graph.output_agent,
                "task_result_agents": sorted(targets),
                "reachability_by_target": validation,
                "weak_components": self._weak_component_count(),
                "weakly_disconnected": self._weak_component_count() > 1,
                "last_relation_decision": relation_facts,
            }
        return {
            "nodes": nodes,
            "actual_relations": actual_relations,
            "candidate_relations_are_not_edges": True,
            "output_agent": self.graph.output_agent,
            "weak_components": self._weak_component_count(),
            "weakly_disconnected": self._weak_component_count() > 1,
            "unreachable_to_output": list(self._unreachable_to_output()),
            "last_relation_decision": relation_facts,
            "finish_validation_errors": list(self.graph.validate(final=True)),
        }

    def _topology_action_previews(
        self, legal_parameters: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if not (
            self._connectivity_repair_active
            or self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}
        ):
            return {"set_output": {}, "relation_choices": []}
        before = set(self._unreachable_to_output())
        output_previews = {
            agent_id: {
                "unreachable_after": list(self._unreachable_for_graph(self.graph, agent_id)),
                "improves_reachability": (
                    set(self._unreachable_for_graph(self.graph, agent_id)) < before
                ),
                "resolves_reachability": not self._unreachable_for_graph(self.graph, agent_id),
                "blocked_as_cycle": self._output_switch_cycle(agent_id)["blocked_as_cycle"]
                if self._connectivity_repair_active
                and self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}
                else False,
            }
            for agent_id in self._eligible_output_agents()
            if agent_id != self.graph.output_agent
        }
        relation_previews: list[dict[str, Any]] = []
        candidates = legal_parameters.get(ActionType.CONSIDER_RELATION.value, {}).get("relations", [])
        if not candidates:
            candidates = legal_parameters.get(ActionType.SET_RELATION.value, {}).get("relations", [])
        relevant_agents = set(self._unreachable_to_output())
        if self.graph.output_agent is not None:
            relevant_agents.add(self.graph.output_agent)
        for candidate in candidates:
            source = str(candidate["source"])
            target = str(candidate["target"])
            if (
                self.graph.output_agent is not None
                and not ({source, target} & relevant_agents)
            ):
                continue
            relation = RelationType(str(candidate["relation"]))
            present = (
                tuple(sorted((source, target))) in self.graph.bidirectional_edges
                if relation is RelationType.BIDIRECTIONAL
                else (source, target) in self.graph.directed_edges
            )
            choices: dict[str, Any] = {}
            for choice in ("off", "on"):
                simulated = self.graph.clone()
                desired = choice == "on"
                if desired != present:
                    if desired:
                        simulated.set_relation(source, target, relation)
                    else:
                        simulated.remove_relation(source, target, relation)
                unreachable = list(self._unreachable_for_graph(simulated, self.graph.output_agent))
                choices[choice] = {
                    "candidate_edge_present_after": desired,
                    "unreachable_to_output_after": unreachable,
                    "resolves_reachability": not unreachable,
                }
            relation_previews.append({
                "source": source, "target": target, "relation": relation.value,
                "actual_present_before": present, "choices": choices,
            })
        return {"set_output": output_previews, "relation_choices": relation_previews}

    def _enter_structural_repair(
        self,
        reason: str,
        *,
        agents: tuple[str, ...] = (),
        pair: tuple[str, str] | None = None,
        relation: str | None = None,
    ) -> None:
        if not self.config.structural_repair_enabled:
            return
        previous_reason = self.structural_repair_reason
        if previous_reason is None:
            self.structural_repair_entries += 1
            self._repair_transition = "entered"
        elif previous_reason != reason:
            self._repair_transition = "retargeted"
        if previous_reason != reason:
            self.repair_epoch += 1
            self.semantic_no_progress_streak = 0
            self._last_repair_signature = None
            self._repair_entry_pending = True
            self._recent_repair_actions = []
            self._consolidation_output_locked = False
            self._output_selection_budget_remaining = self.config.output_selection_budget
        self.structural_repair_reason = reason
        if self._is_consolidation_repair():
            self._connectivity_repair_active = False
            self._connectivity_repair_states.clear()
        self.structural_repair_agents = tuple(
            sorted(agent_id for agent_id in agents if agent_id in self.graph.nodes)
        )
        self.structural_repair_pair = pair
        self.structural_repair_relation = relation
        if reason in {"output_reachability", "disconnected_multi_agent"}:
            self._connectivity_repair_active = True
            self._connectivity_repair_states.add(self._connectivity_state_signature())
            if not self._is_consolidation_repair():
                # A generic invalid-action fuse must not strand a graph whose
                # known structural defect requires layer/relation edits.
                self.topology_edits_frozen = False
                self._invalid_repeat_count = 0
                self._last_invalid_signature = None
        if self._is_consolidation_repair() and self.graph.output_agent is not None:
            self._lock_consolidation_output()

    def _clear_structural_repair(self) -> None:
        if self.structural_repair_reason is None:
            return
        self.structural_repair_reason = None
        self.structural_repair_agents = ()
        self.structural_repair_pair = None
        self.structural_repair_relation = None
        self.structural_repair_resolutions += 1
        self._repair_transition = "resolved"
        self._consolidation_output_locked = False

    def _is_consolidation_repair(self) -> bool:
        return self.structural_repair_reason in {
            "token_budget_consolidation",
            "time_budget_consolidation",
        }

    def _is_output_lifecycle_repair(self) -> bool:
        return self._is_consolidation_repair() or self.structural_repair_reason in {
            "output_not_set",
            "output_artifact_unusable",
        }

    def _lock_consolidation_output(self) -> bool:
        output = self.graph.output_agent
        if not self._is_consolidation_repair() or output is None:
            return False
        if not self._artifact_is_usable_output(output):
            return False
        artifact = self.runtime.artifacts[output]
        self._consolidation_output_locked = True
        self._output_incumbent = {
            "agent_id": output,
            "artifact_id": artifact.artifact_id,
            "repair_epoch": self.repair_epoch,
        }
        return True

    def _structural_repair_gate_error(self, action: CanvasAction) -> str | None:
        reason = self.structural_repair_reason
        if not self.config.structural_repair_enabled or reason is None:
            return None
        if (
            action.action_type is ActionType.SET_PROMPT
            and str(action.target) in set(self.runtime.environment_commit_ready_agents())
            and self._webshop_staged_revision_allowed(str(action.target))
        ):
            # A staged transaction with Canvas-verifiable unresolved evidence
            # must remain revisable even when an unrelated topology audit is
            # also active. The normal prompt-revision evidence and budget gates
            # still run before the mutation is committed.
            return None
        allowed = set(self._structural_repair_allowed_action_types())
        if action.action_type not in allowed:
            return (
                "STRUCTURAL_REPAIR_REQUIRED "
                f"(reason={reason}; repair existing Agents before {action.action_type.value})"
            )
        if reason == "relation_layer_mismatch" and self.structural_repair_pair:
            pair = set(self.structural_repair_pair)
            if (
                action.action_type in {ActionType.CONSIDER_RELATION, ActionType.SET_RELATION}
                and {
                    str(action.source),
                    str(action.target),
                }
                != pair
            ):
                return "STRUCTURAL_REPAIR_REQUIRED (repair the rejected Agent pair first)"
            if action.action_type is ActionType.SET_LAYER and str(action.target) not in pair:
                return "STRUCTURAL_REPAIR_REQUIRED (change a layer in the rejected Agent pair)"
            if action.action_type is ActionType.DELETE_AGENT and str(action.target) not in pair:
                return "STRUCTURAL_REPAIR_REQUIRED (delete only an implicated Agent)"
            if (
                action.action_type is ActionType.SET_LAYER
                and action.target in self.graph.nodes
                and self.graph.nodes[str(action.target)].layer == action.layer
            ):
                return "STRUCTURAL_REPAIR_REQUIRED (SET_LAYER must change the current layer)"
        if (
            reason == "output_reachability"
            and action.action_type is ActionType.DELETE_AGENT
            and str(action.target) not in set(self.structural_repair_agents)
        ):
            return "STRUCTURAL_REPAIR_REQUIRED (delete only an unreachable Agent)"
        if (
            reason == "output_not_set"
            and action.action_type is ActionType.DELETE_AGENT
            and len(self.graph.nodes) <= 1
        ):
            return "STRUCTURAL_REPAIR_REQUIRED (the final Agent must be selected as output)"
        return None

    def _refresh_structural_repair(self, action: CanvasAction) -> None:
        if self.state is CanvasState.AWAITING_MODEL:
            return
        reason = self.structural_repair_reason
        if reason is None:
            # Construction remains free until an output is selected. Once one
            # exists, a newly configured disconnected Agent must be connected,
            # deleted, or made the output before the graph can grow again.
            # ADD_AGENT itself must still reach the mandatory SET_PROMPT
            # barrier; auditing it one turn early would deadlock that barrier.
            unreachable = (
                () if action.action_type is ActionType.ADD_AGENT else self._unreachable_to_output()
            )
            if unreachable:
                self._enter_structural_repair(
                    "output_reachability",
                    agents=unreachable,
                )
            elif (
                action.action_type is not ActionType.ADD_AGENT
                and self.graph.output_agent is None
                and self.topology_audit()["disconnected_multi_agent"]
            ):
                self._enter_structural_repair(
                    "disconnected_multi_agent",
                    agents=tuple(self.graph.nodes),
                )
            return
        if reason in {"token_budget_consolidation", "time_budget_consolidation"}:
            # Consolidation is a terminal budget lock, not a structural defect
            # repaired merely by selecting an output. Keep it latched until
            # _finish() validates and terminates the graph. DELETE_AGENT and
            # SET_OUTPUT may still make a non-final graph finishable.
            self.structural_repair_agents = tuple(sorted(self.graph.nodes))
            return
        if reason == "relation_layer_mismatch":
            pair = set(self.structural_repair_pair or ())
            repaired_pair = (
                action.action_type is ActionType.SET_RELATION
                and {
                    str(action.source),
                    str(action.target),
                }
                == pair
            )
            deleted_pair = (
                action.action_type is ActionType.DELETE_AGENT and str(action.target) in pair
            )
            if not (repaired_pair or deleted_pair):
                return
        if reason == "disconnected_multi_agent" and self.graph.output_agent is None:
            if self.topology_audit()["disconnected_multi_agent"]:
                return
            self._clear_structural_repair()
            return
        if self.graph.output_agent is None:
            if reason == "output_not_set":
                return
            self._clear_structural_repair()
            return
        unreachable = self._unreachable_to_output()
        if unreachable:
            self._enter_structural_repair(
                "output_reachability",
                agents=unreachable,
            )
        else:
            self._clear_structural_repair()

    def _structural_repair_allowed_action_types(self) -> tuple[ActionType, ...]:
        reason = self.structural_repair_reason
        if self._uses_staged_environment_commit():
            commit_ready = tuple(self.runtime.environment_commit_ready_agents())
            if commit_ready:
                return (ActionType.SET_OUTPUT, ActionType.DELETE_AGENT)
        if reason in {"output_not_set", "output_artifact_unusable"}:
            return (
                ActionType.ADD_AGENT,
                ActionType.SET_PROMPT,
                ActionType.SET_MODEL,
                ActionType.SET_LAYER,
                ActionType.CONSIDER_RELATION,
                ActionType.SET_RELATION,
                ActionType.REMOVE_RELATION,
                ActionType.DELETE_AGENT,
                ActionType.SET_OUTPUT,
            )
        if reason == "relation_layer_mismatch":
            return (
                ActionType.SET_LAYER,
                ActionType.CONSIDER_RELATION,
                ActionType.SET_RELATION,
                ActionType.DELETE_AGENT,
            )
        if reason in {"token_budget_consolidation", "time_budget_consolidation"}:
            if not self._consolidation_output_locked:
                return (
                    (ActionType.SET_OUTPUT,) if self._output_selection_budget_remaining > 0 else ()
                )
            if self.graph.validate(final=True):
                return (ActionType.DELETE_AGENT,)
            return (ActionType.FINISH,)
        if reason:
            return (
                ActionType.SET_LAYER,
                ActionType.CONSIDER_RELATION,
                ActionType.SET_RELATION,
                ActionType.REMOVE_RELATION,
                ActionType.DELETE_AGENT,
                ActionType.SET_OUTPUT,
            )
        return ()

    def _structural_repair_snapshot(self) -> dict[str, Any]:
        # Report the same state/protocol/parameter mask that admission uses.
        # The repair-type superset alone can advertise disabled relation actions.
        allowed = (
            self.control_snapshot()["allowed_actions"] if self.structural_repair_reason else []
        )
        guidance = ""
        if self.structural_repair_reason == "time_budget_consolidation":
            guidance = (
                "Time admission is active and no additional Worker execution is budgeted. "
                "The allowed actions and their valid targets are listed separately."
            )
        if self.structural_repair_reason == "token_budget_consolidation":
            guidance = (
                "The remaining Worker token budget does not admit another execution. The "
                "allowed actions and their valid targets are listed separately."
            )
        if self.structural_repair_reason == "relation_layer_mismatch":
            source, target = self.structural_repair_pair or ("", "")
            source_node = self.graph.nodes.get(source)
            target_node = self.graph.nodes.get(target)
            if source_node is not None and target_node is not None:
                if self.structural_repair_relation == "bidirectional":
                    guidance = (
                        f"Current layers are {source}={source_node.layer} and "
                        f"{target}={target_node.layer}; a bidirectional relation requires equal "
                        "layers."
                    )
                else:
                    guidance = (
                        f"Current layers are {source}={source_node.layer} and "
                        f"{target}={target_node.layer}; directed relations require the source "
                        "layer to be lower than the target layer."
                    )
        if self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}:
            guidance = (
                f"Actual edges are {self.topology_audit()['actual_relations']}; optional relation "
                "parameters are only candidates. Check topology_action_previews before switching "
                "output. A legal edge may still point away from the selected output. Keep layer "
                "and relation edits available until the graph validates."
            )
        return {
            "required": self.structural_repair_reason is not None,
            "reason": self.structural_repair_reason,
            "agents": list(self.structural_repair_agents),
            "pair": list(self.structural_repair_pair or ()),
            "relation": self.structural_repair_relation,
            "guidance": guidance,
            "allowed_actions": allowed,
            "transition": self._repair_transition,
            "entries_total": self.structural_repair_entries,
            "resolutions_total": self.structural_repair_resolutions,
            "blocked_actions_total": self.structural_repair_blocked_actions,
            "finish_reachability_rejections_total": self.finish_reachability_rejections,
            "relation_layer_rejections_total": self.relation_layer_rejections,
            "duplicate_agent_rejections_total": self.duplicate_agent_rejections,
            "time_budget_rejections_total": self.time_budget_rejections,
            "token_budget_admission_rejections_total": (self.token_budget_admission_rejections),
            "repair_epoch": self.repair_epoch,
            "progress_signature": (
                self._last_repair_signature.to_dict()
                if self._last_repair_signature is not None
                else None
            ),
            "semantic_no_progress_streak": self.semantic_no_progress_streak,
            "semantic_no_progress_limit": self.config.semantic_no_progress_limit,
            "semantic_no_progress_recoveries_total": (self.semantic_no_progress_recovery_count),
            "connectivity_repair_active": self._connectivity_repair_active,
            "connectivity_repair_visited_states": len(self._connectivity_repair_states),
            "connectivity_output_cycle_blocks": self._connectivity_cycle_blocks,
            "output_switches_without_progress_total": (self.output_switches_without_progress),
            "output_lifecycle_recoveries_total": self.output_lifecycle_recovery_count,
            "recent_actions": list(self._recent_repair_actions),
            "output_lifecycle": {
                "health": self._output_health(),
                "usable_candidates": list(self._usable_output_agents()),
                "locked": self._consolidation_output_locked,
                "selection_budget_remaining": self._output_selection_budget_remaining,
                "incumbent": dict(self._output_incumbent),
            },
        }

    def _audit_feedback(self, audit: dict[str, Any], repair: dict[str, Any]) -> str:
        if self.unified:
            return "Actual topology: " + json.dumps(self._graph_state_snapshot(), ensure_ascii=False, sort_keys=True) + ". FINISH(target) eligibility is recorded in result_assessments."
        unreachable = ", ".join(audit["unreachable_agents"]) or "none"
        actual_relations = json.dumps(audit["actual_relations"], ensure_ascii=False, sort_keys=True)
        facts = [
            "Topology audit: agents={agent_count}, configured={configured_agent_count}, "
            "relations={relation_count}, components={weak_component_count}, output={output}, "
            "unreachable=[{unreachable}], disconnected_multi_agent={disconnected}, "
            "actual_relations={actual_relations_text}. Candidate relation parameters are not graph edges.".format(
                **audit,
                output=audit["output_agent"] or "not set",
                unreachable=unreachable,
                disconnected=str(audit["disconnected_multi_agent"]).lower(),
                actual_relations_text=actual_relations,
            )
        ]
        if self._last_relation_decision:
            relation_fact = dict(self._last_relation_decision)
            if relation_fact.get("phase") == "choice":
                facts.append(
                    "Last relation choice: {source}->{target} {relation}, choice={choice}, "
                    "actual_present_after={present}.".format(
                        source=relation_fact.get("source"),
                        target=relation_fact.get("target"),
                        relation=relation_fact.get("relation_type"),
                        choice=relation_fact.get("choice"),
                        present=relation_fact.get("chosen_present"),
                    )
                )
            else:
                facts.append(
                    "Pending relation proposal only: {source}->{target} {relation}; "
                    "no edge is established until its on/off choice.".format(
                        source=relation_fact.get("source"),
                        target=relation_fact.get("target"),
                        relation=relation_fact.get("relation_type"),
                    )
                )
        if repair["required"]:
            facts.append(
                "STRUCTURAL_REPAIR_REQUIRED: reason={}; agents=[{}]; pair=[{}]; "
                "allowed_actions=[{}].".format(
                    repair["reason"],
                    ", ".join(repair["agents"]),
                    ", ".join(repair["pair"]),
                    ", ".join(repair["allowed_actions"]),
                )
            )
            if repair["guidance"]:
                facts.append("Structural repair guidance: " + repair["guidance"])
        facts.append(
            "Structural repair counters: entries={}, resolved={}, blocked_actions={}, "
            "finish_reachability_rejections={}, relation_layer_rejections={}, "
            "duplicate_agent_rejections={}, time_budget_rejections={}.".format(
                repair["entries_total"],
                repair["resolutions_total"],
                repair["blocked_actions_total"],
                repair["finish_reachability_rejections_total"],
                repair["relation_layer_rejections_total"],
                repair["duplicate_agent_rejections_total"],
                repair["time_budget_rejections_total"],
            )
        )
        return "\n".join(facts)

    @staticmethod
    def _one_line(value: object, limit: int) -> str:
        compact = " ".join(str(value or "").split())
        if limit <= 0 or len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 15)] + "... [truncated]"

    def _bound_feedback(self, feedback: str) -> str:
        limit = self.config.feedback_max_chars
        if limit <= 0 or len(feedback) <= limit:
            return feedback
        marker = "\n... [feedback truncated] ...\n"
        available = max(0, limit - len(marker))
        head = available * 2 // 3
        tail = available - head
        return feedback[:head] + marker + (feedback[-tail:] if tail else "")

    def _record(
        self,
        action: CanvasAction,
        *,
        accepted: bool,
        feedback: str,
        execution: ExecutionReport | None = None,
        rejection_code: str | None = None,
        protocol_recovery: bool = False,
        final_execution: bool = False,
        responsibility_issue: dict[str, Any] | None = None,
        delegation_field_repairs: list[dict[str, Any]] | None = None,
        rejection_details: dict[str, Any] | None = None,
        relation_decision: dict[str, Any] | None = None,
        invalid_repeat_count: int | None = None,
    ) -> CanvasStep:
        resolved_rejection_details = dict(rejection_details or {})
        if (
            accepted
            and action.action_type is ActionType.SET_OUTPUT
            and self._is_consolidation_repair()
            and self._lock_consolidation_output()
        ):
            self._output_selection_budget_remaining = max(
                0, self._output_selection_budget_remaining - 1
            )
        if not accepted:
            rejection_code = rejection_code or "rejected_action"
        if self.unified:
            self._record_unified_progress(accepted=accepted)
        self._record_repair_progress(
            action,
            accepted=accepted,
            rejection_code=rejection_code,
        )
        if not accepted:
            self._register_no_progress_rejection()
            common_details = self._rejection_details(rejection_code)
            common_details.update(resolved_rejection_details)
            common_details["repeat_count"] = self._invalid_repeat_count
            common_details["deterministic_recovery"] = (
                "refresh_and_choose_legal_action"
                if not self.topology_edits_frozen
                else "freeze_topology_edits_continue_existing_graph"
            )
            resolved_rejection_details = common_details
            recovery = json.dumps(
                resolved_rejection_details,
                ensure_ascii=False,
                sort_keys=True,
            )
            if "Recovery:" not in feedback:
                feedback = f"{feedback} Recovery: {recovery}"
        audit = self.topology_audit()
        repair = self._structural_repair_snapshot()
        decorated_feedback = self._bound_feedback(
            "\n".join(
                (
                    feedback,
                    self._audit_feedback(audit, repair),
                    f"Canvas state: {self.state.value}; "
                    + (f"round {self.round_index}/{self.director_round_limit}; "
                       if self.director_round_limit is not None else
                       f"Director decisions recorded={self.round_index} (statistics only); ")
                    + f"cumulative Worker tokens={self.total_tokens}/"
                    f"{self.config.max_total_tokens}.",
                )
            )
        )
        step = CanvasStep(
            event_id=f"{self.run_id}:canvas:{len(self.history)}",
            director_call_id=(self._director_call_context.call_id
                              if isinstance(self._director_call_context, DirectorCallContext)
                              and self._director_call_context.runtime_owned else ""),
            submission_receipt=(self.submission_receipt.to_dict()
                                if accepted and action.action_type is ActionType.FINISH
                                and self.submission_receipt is not None else None),
            round_index=self.round_index,
            action=action,
            accepted=accepted,
            active=self.active,
            feedback=decorated_feedback,
            graph=self.graph.to_dict(),
            dirty_agents=sorted(self.dirty_agents),
            invalidated_agents=sorted(self._step_invalidated_agents),
            scheduled_agents=sorted(self._step_scheduled_agents),
            executed_agents=(list(execution.executed_agents) if execution else []),
            reused_agents=(list(execution.reused_agents) if execution else []),
            remaining_dirty=sorted(self.dirty_agents),
            invalidation_reasons={
                agent_id: sorted(reasons)
                for agent_id, reasons in self._step_invalidation_reasons.items()
                if reasons
            },
            prompt_revision=dict(self._step_prompt_revision),
            execution=execution,
            rejection_code=rejection_code,
            protocol_recovery=protocol_recovery,
            final_execution=final_execution,
            topology_audit=audit,
            structural_repair=repair,
            responsibility_issue=dict(responsibility_issue or {}),
            responsibility_overlap_check=dict(self._step_responsibility_overlap_check),
            delegation_field_repairs=list(delegation_field_repairs or []),
            time_admission=dict(self._time_admission_event),
            token_admission=dict(self._token_admission_event),
            control_snapshot=self.control_snapshot(),
            rejection_details=resolved_rejection_details,
            relation_decision=dict(relation_decision or {}),
            invalid_repeat_count=(
                self._invalid_repeat_count
                if invalid_repeat_count is None
                else int(invalid_repeat_count)
            ),
            topology_edits_frozen=self.topology_edits_frozen,
        )
        self.history.append(step)
        if accepted:
            self._last_invalid_signature = None
            self._invalid_repeat_count = 0
        self._repair_transition = None
        return step

    def _register_no_progress_rejection(self) -> None:
        """Count every rejection made against the same authoritative state."""

        signature = json.dumps(
            {
                "version": self.graph.version,
                "state": self.state.value,
                "pending_agent_id": self.pending_agent_id,
                "pending_relation_decision": (
                    self.pending_relation_decision.to_dict()
                    if self.pending_relation_decision is not None
                    else None
                ),
                "dirty_agents": sorted(self.dirty_agents),
                "structural_repair_reason": self.structural_repair_reason,
                "structural_repair_agents": list(self.structural_repair_agents),
                "structural_repair_pair": list(self.structural_repair_pair or ()),
                "structural_repair_relation": self.structural_repair_relation,
                "structural_exploration_waived": self.structural_exploration_waived,
            },
            sort_keys=True,
        )
        if signature == self._last_invalid_signature:
            self._invalid_repeat_count += 1
        else:
            self._last_invalid_signature = signature
            self._invalid_repeat_count = 1
        if (
            self._invalid_repeat_count >= 3
            and not self.unified
            and self.state is CanvasState.BUILDING
            and self.graph.nodes
            and not self._connectivity_repair_active
            and self.structural_repair_reason not in {
                "output_reachability", "disconnected_multi_agent"
            }
        ):
            self.topology_edits_frozen = True

    def director_progress_signature(self) -> str:
        """Effective state, excluding rounds, feedback and rejection counters."""
        if self.unified:
            return self._unified_progress_signature()
        graph = self.graph.to_dict()
        graph.pop("version", None)
        return json.dumps(
            {
                "graph": graph,
                "state": self.state.value,
                "pending_agent": self.pending_agent_id,
                "pending_relation": (
                    self.pending_relation_decision.to_dict()
                    if self.pending_relation_decision
                    else None
                ),
                "dirty_agents": sorted(self.dirty_agents),
                "artifacts": {
                    key: value.artifact_id for key, value in self.runtime.artifacts.items()
                },
            },
            sort_keys=True,
        )

    def terminate_director_stall(self, code: str) -> CanvasStep:
        """Record runtime closure only; no model call or graph repair is invented."""
        if code not in {"director_no_legal_continuation", "director_no_progress_exhausted",
                        "director_action_protocol_exhausted"}:
            raise ValueError(f"unsupported Director stall: {code}")
        if (self.unified and code == "director_no_legal_continuation"
                and self._unified_transaction is None
                and self.director_edit_budget()["remaining"] == 0
                and not self._unified_control_snapshot()["allowed_actions"]):
            code = "director_edit_budget_dead_end"
        if (self.unified and code == "director_no_legal_continuation"
                and not self.graph.nodes and self.runtime_routes
                and self.director_round_limit is not None
                and not self.topology_edits_frozen and self._unified_transaction is None
                and 0 <= self.config.max_rounds - self.round_index < 4):
            # ADD_AGENT, SET_PROMPT, SET_MODEL and FINISH require four policy
            # actions. Deleting the last node can exhaust that completion path
            # before the numeric round counter reaches its absolute ceiling.
            code = "director_round_budget_dead_end"
        return self._fail_terminal_recovery(
            code=code,
            message="no legal continuation or bounded Director recovery exhausted",
        )

    def control_snapshot(self) -> dict[str, Any]:
        if self.unified:
            return self._unified_control_snapshot()
        """Bounded authoritative state supplied to the Director after every turn."""

        legal_ids = sorted(self.graph.nodes)
        legal_parameters = self._legal_action_parameters()
        relation_actions = (
            ["consider_relation"]
            if self.binary_relation_policy
            else ["set_relation", "remove_relation"]
        )
        allowed_actions = [
            "add_agent",
            "set_prompt",
            "set_model",
            "set_layer",
            *relation_actions,
            "delete_agent",
            "set_output",
            "finish",
        ]
        if (
            len(legal_ids) >= self.graph.max_agents
            or self.topology_edits_frozen
            or (self.runtime_routes and self.config.max_rounds - self.round_index < 5)
        ):
            allowed_actions.remove("add_agent")
        if self.topology_edits_frozen:
            frozen_actions = {ActionType.SET_OUTPUT, ActionType.FINISH}
            if ActionType.DELETE_AGENT in self._structural_repair_allowed_action_types():
                frozen_actions.add(ActionType.DELETE_AGENT)
            allowed_actions = [
                action_type.value
                for action_type in (
                    ActionType.SET_OUTPUT,
                    ActionType.DELETE_AGENT,
                    ActionType.FINISH,
                )
                if action_type in frozen_actions
            ]
        if self.state is CanvasState.AWAITING_PROMPT:
            allowed_actions = ["set_prompt"]
        if self.state is CanvasState.AWAITING_MODEL:
            allowed_actions = ["set_model"]
        if self.state is CanvasState.AWAITING_RELATION_CHOICE:
            allowed_actions = ["relation_choice"]
        exploration_satisfied = self._structural_exploration_satisfied()
        if (
            self.structural_exploration_required
            and not self.structural_exploration_waived
            and not exploration_satisfied
        ):
            allowed_actions = [
                value for value in allowed_actions if value not in {"set_output", "finish"}
            ]
        if self.structural_repair_reason is not None:
            repair_allowed = [
                action_type.value for action_type in self._structural_repair_allowed_action_types()
            ]
            if self.structural_repair_reason in {
                "token_budget_consolidation",
                "time_budget_consolidation",
            }:
                # SET_OUTPUT/FINISH explicitly waive the exploration stratum
                # when the budget lock is active, so expose what step() accepts.
                allowed_actions = repair_allowed
            else:
                allowed_actions = [value for value in allowed_actions if value in repair_allowed]
        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            allowed_actions = []
        commit_ready_agents = (
            list(self.runtime.environment_commit_ready_agents())
            if self._uses_staged_environment_commit()
            else []
        )
        if commit_ready_agents and self.state not in {CanvasState.FINISHED, CanvasState.FAILED}:
            # Preserve the staged transaction. The Director may select its
            # owner or explicitly remove a node proven unable to contribute to
            # any latched candidate; Canvas never prunes it as a side effect.
            allowed_actions = [ActionType.SET_OUTPUT.value]
            if legal_parameters[ActionType.DELETE_AGENT.value]["targets"]:
                allowed_actions.append(ActionType.DELETE_AGENT.value)
            prompt_parameters = legal_parameters[ActionType.SET_PROMPT.value]
            prompt_parameters["targets"] = []
            revision_evidence = prompt_parameters.get("revision_evidence_by_target", {})
            if isinstance(revision_evidence, dict):
                prompt_parameters["revision_evidence_by_target"] = {}
        if self.state is CanvasState.AWAITING_MODEL and not commit_ready_agents:
            allowed_actions = [ActionType.SET_MODEL.value]
            if (
                self.pending_agent_id in legal_parameters[ActionType.SET_OUTPUT.value]["targets"]
                and (
                    not self.structural_exploration_required
                    or self.structural_exploration_waived
                    or exploration_satisfied
                )
                and (
                    self.structural_repair_reason is None
                    or ActionType.SET_OUTPUT
                    in self._structural_repair_allowed_action_types()
                )
            ):
                allowed_actions.append(ActionType.SET_OUTPUT.value)
        parameterized_actions = {
            ActionType.SET_PROMPT.value: "targets",
            ActionType.SET_MODEL.value: "targets",
            ActionType.SET_LAYER.value: "targets",
            ActionType.CONSIDER_RELATION.value: "relations",
            ActionType.SET_RELATION.value: "relations",
            ActionType.REMOVE_RELATION.value: "relations",
            ActionType.DELETE_AGENT.value: "targets",
            ActionType.SET_OUTPUT.value: "targets",
        }
        allowed_actions = [
            action_name
            for action_name in allowed_actions
            if action_name not in parameterized_actions
            or bool(legal_parameters[action_name][parameterized_actions[action_name]])
        ]
        if not legal_parameters[ActionType.FINISH.value]["ready"]:
            allowed_actions = [
                action_name
                for action_name in allowed_actions
                if action_name != ActionType.FINISH.value
            ]
        # A pending binary decision can always be resolved (including a rejected
        # mutation). Structural edit masks must not hide this protocol action.
        if self.state is CanvasState.AWAITING_RELATION_CHOICE and self.pending_relation_decision:
            allowed_actions = ["relation_choice"]
        return {
            "canvas_version": self.graph.version,
            "director_action_protocol_version": DIRECTOR_ACTION_PROTOCOL_VERSION,
            "output_contract_version": OUTPUT_CONTRACT_VERSION,
            "submission_contract_version": SUBMISSION_CONTRACT_VERSION,
            "submission_status": "submitted" if self.submission_receipt else "candidate",
            "worker_protocol_status_version": WORKER_PROTOCOL_STATUS_VERSION,
            "worker_protocol_status": {
                agent_id: {
                    "artifact_id": self.runtime.artifacts[agent_id].artifact_id,
                    **summarize_worker_protocol(
                        answer=self.runtime.artifacts[agent_id].answer,
                        raw_response=self.runtime.artifacts[agent_id].raw_response,
                        diagnostics=self.runtime.artifacts[agent_id].protocol_diagnostics,
                    ),
                }
                for agent_id in legal_ids
                if agent_id in self.runtime.artifacts
            },
            "state": self.state.value,
            "legal_agent_ids": legal_ids,
            "pending_agent_id": self.pending_agent_id,
            "pending_relation_decision": (
                self.pending_relation_decision.to_dict()
                if self.pending_relation_decision is not None
                else None
            ),
            "agent_budget": {
                "used": len(legal_ids),
                "max": self.graph.max_agents,
                "remaining": max(0, self.graph.max_agents - len(legal_ids)),
            },
            "allowed_actions": allowed_actions,
            "action_field_requirements": {
                name: list(ACTION_FIELDS[name]) for name in allowed_actions if name in ACTION_FIELDS
            },
            "legal_action_parameters": legal_parameters,
            "graph_state": self._graph_state_snapshot(),
            "topology_action_previews": self._topology_action_previews(legal_parameters),
            "output_agent": self.graph.output_agent,
            "environment_commit_ready_agents": (
                list(self.runtime.environment_commit_ready_agents())
                if self.runtime.native_webshop and self.dataset == "webshop"
                else list(commit_ready_agents)
            ),
            "environment_owner_agents": list(self.runtime.environment_owner_agents()),
            **(
                {"environment_task_status": self._native_webshop_task_status()}
                if self.runtime.native_webshop and self.dataset == "webshop"
                else {}
            ),
            "environment_commit_resolution": (
                "SET_OUTPUT selects the output. FINISH validates the graph and commits its staged "
                "candidate. A clean selected output with no candidate receives at most one "
                "same-session output execution, charged to the shared revision allowance."
                if self.config.native_webshop_output_materialization
                and self.runtime.native_webshop
                and self.dataset == "webshop"
                else "SET_OUTPUT only selects an output. FINISH validates the graph, updates dirty "
                "nodes, then commits only the selected output's latest staged candidate. "
                "Candidates remain revisable before FINISH."
                if self.runtime.native_webshop and self.dataset == "webshop"
                else "SET_OUTPUT commits one staged candidate; commit confirmation cannot exist "
                "before SET_OUTPUT, so waiting for that confirmation is not an unresolved task "
                "constraint. The trusted staged Action is latched; later model prose cannot "
                "discard or revise it."
                if commit_ready_agents
                else None
            ),
            "topology_edits_frozen": self.topology_edits_frozen,
            "repair_progress": {
                "epoch": self.repair_epoch,
                "semantic_no_progress_streak": self.semantic_no_progress_streak,
                "recent_actions": list(self._recent_repair_actions),
                "connectivity_repair_active": self._connectivity_repair_active,
                "connectivity_repair_visited_states": len(self._connectivity_repair_states),
                "connectivity_output_cycle_blocks": self._connectivity_cycle_blocks,
            },
            "structural_exploration": {
                "required": self.structural_exploration_required,
                "satisfied": exploration_satisfied,
                "waived_for_budget": self.structural_exploration_waived,
                "minimum_connected_agents": 2,
            },
        }

    def _legal_action_parameters(self) -> dict[str, dict[str, Any]]:
        legal_ids = sorted(self.graph.nodes)
        revision_evidence = {
            agent_id: self._eligible_prompt_revision_evidence(agent_id)["public"]
            for agent_id in legal_ids
            if self.graph.nodes[agent_id].configured
        }
        if self.state is CanvasState.AWAITING_PROMPT and self.pending_agent_id:
            prompt_targets = [self.pending_agent_id]
        else:
            prompt_targets = [
                agent_id
                for agent_id in legal_ids
                if not self.graph.nodes[agent_id].configured
                or bool(revision_evidence.get(agent_id))
            ]
        layer_targets = list(legal_ids)
        delete_targets = list(legal_ids)
        reason = self.structural_repair_reason
        if reason == "relation_layer_mismatch" and self.structural_repair_pair:
            implicated = set(self.structural_repair_pair)
            layer_targets = [agent_id for agent_id in legal_ids if agent_id in implicated]
            delete_targets = [agent_id for agent_id in legal_ids if agent_id in implicated]
        elif reason == "output_reachability":
            implicated = set(self.structural_repair_agents)
            delete_targets = [agent_id for agent_id in legal_ids if agent_id in implicated]
        elif reason in {"output_not_set", "output_artifact_unusable"}:
            if reason == "output_not_set" and len(legal_ids) <= 1:
                delete_targets = []
        elif reason in {"token_budget_consolidation", "time_budget_consolidation"}:
            delete_targets = (
                list(self._unreachable_to_output()) if self._consolidation_output_locked else []
            )

        output_targets = [
            agent_id
            for agent_id in self._eligible_output_agents()
            if agent_id != self.graph.output_agent and not self._consolidation_output_locked
        ]
        if (
            self._connectivity_repair_active
            and reason in {"output_reachability", "disconnected_multi_agent"}
            and not (
                self._uses_staged_environment_commit()
                and self.runtime.environment_commit_ready_agents()
            )
        ):
            output_targets = [
                agent_id for agent_id in output_targets
                if not self._output_switch_cycle(agent_id)["blocked_as_cycle"]
            ]
        if self._uses_staged_environment_commit() and self.runtime.environment_commit_ready_agents():
            safe_delete_targets = set(self._staged_commit_safe_delete_targets())
            delete_targets = [agent_id for agent_id in delete_targets if agent_id in safe_delete_targets]
        directed_relations: list[dict[str, str]] = []
        bidirectional_relations: list[dict[str, str]] = []
        relation_candidates: list[dict[str, str]] = []
        for source in legal_ids:
            for target in legal_ids:
                if source == target:
                    continue
                pair = tuple(sorted((source, target)))
                if (
                    self.graph.nodes[source].layer < self.graph.nodes[target].layer
                    and (source, target) not in self.graph.directed_edges
                    and pair not in self.graph.bidirectional_edges
                ):
                    directed_relations.append(
                        {"source": source, "target": target, "relation": "directed"}
                    )
                if (
                    source < target
                    and self.graph.nodes[source].layer == self.graph.nodes[target].layer
                    and (source, target) not in self.graph.directed_edges
                    and (target, source) not in self.graph.directed_edges
                    and pair not in self.graph.bidirectional_edges
                ):
                    bidirectional_relations.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": "bidirectional",
                        }
                    )
                if source < target:
                    canonical_source, canonical_target, inferred = (
                        self._canonical_relation_candidate(source, target)
                    )
                    considered_key = (self.graph.version, (source, target))
                    if considered_key not in self._considered_relation_pairs:
                        relation_candidates.append(
                            {
                                "source": canonical_source,
                                "target": canonical_target,
                                "relation": inferred.value,
                            }
                        )
        if reason == "relation_layer_mismatch" and self.structural_repair_pair:
            pair = set(self.structural_repair_pair)
            directed_relations = [
                relation
                for relation in directed_relations
                if {relation["source"], relation["target"]} == pair
            ]
            bidirectional_relations = [
                relation
                for relation in bidirectional_relations
                if {relation["source"], relation["target"]} == pair
            ]

        removable_relations = [
            {"source": source, "target": target, "relation": "directed"}
            for source, target in sorted(self.graph.directed_edges)
        ] + [
            {"source": source, "target": target, "relation": "bidirectional"}
            for source, target in sorted(self.graph.bidirectional_edges)
        ]
        finish_ready = bool(
            self.state is CanvasState.BUILDING
            and not self.graph.validate(final=True)
            and not (
                self.structural_exploration_required
                and not self.structural_exploration_waived
                and not self._structural_exploration_satisfied()
            )
        )
        return {
            ActionType.ADD_AGENT.value: {
                "system_allocated_id": True,
                "remaining_slots": max(0, self.graph.max_agents - len(legal_ids)),
            },
            ActionType.SET_PROMPT.value: {
                "targets": prompt_targets,
                "revision_evidence_by_target": {
                    agent_id: revision_evidence[agent_id]
                    for agent_id in prompt_targets
                    if revision_evidence.get(agent_id)
                },
                "revision_counts": {
                    agent_id: self.prompt_revision_counts.get(agent_id, 0) for agent_id in legal_ids
                },
            },
            ActionType.SET_MODEL.value: {
                "targets": (
                    [self.pending_agent_id]
                    if self.state is CanvasState.AWAITING_MODEL
                    else [key for key in legal_ids if self.graph.nodes[key].prompt_configured]
                )
                if self.runtime_routes
                else [],
                "runtime_routes": list(self.runtime_routes),
            },
            ActionType.SET_LAYER.value: {
                "targets": layer_targets,
                "minimum_layer": 0,
                "current_layers": {
                    agent_id: self.graph.nodes[agent_id].layer for agent_id in layer_targets
                },
            },
            ActionType.CONSIDER_RELATION.value: {"relations": relation_candidates},
            ActionType.SET_RELATION.value: {
                "relations": directed_relations + bidirectional_relations
            },
            ActionType.REMOVE_RELATION.value: {"relations": removable_relations},
            ActionType.DELETE_AGENT.value: {"targets": delete_targets},
            ActionType.SET_OUTPUT.value: {"targets": output_targets},
            ActionType.FINISH.value: {"ready": finish_ready},
        }

    def _director_action_legality_error(
        self,
        action: CanvasAction,
    ) -> tuple[str, str] | None:
        snapshot = self.control_snapshot()
        action_name = action.action_type.value
        if action_name not in snapshot["allowed_actions"]:
            if (
                action.action_type is ActionType.SET_OUTPUT
                and str(action.target) == self.graph.output_agent
            ):
                return (
                    "output_already_selected",
                    f"{action.target} is already the output Agent",
                )
            return (
                "director_action_not_allowed",
                f"{action_name} is not allowed in the current Canvas state",
            )
        parameters = snapshot["legal_action_parameters"][action_name]
        target_actions = {
            ActionType.SET_PROMPT,
            ActionType.SET_MODEL,
            ActionType.SET_LAYER,
            ActionType.DELETE_AGENT,
            ActionType.SET_OUTPUT,
        }
        if self.unified:
            target_actions.update({ActionType.RUN_AGENT, ActionType.FINISH})
        if action.action_type in target_actions:
            target = str(action.target or action.agent_id or "")
            if target not in parameters["targets"]:
                code = (
                    "unknown_agent"
                    if target not in self.graph.nodes
                    else (
                        (
                            "output_already_selected"
                            if target == self.graph.output_agent
                            else "output_target_not_eligible"
                        )
                        if action.action_type is ActionType.SET_OUTPUT
                        else "director_parameter_not_allowed"
                    )
                )
                return (code, f"target {target or '<missing>'} is not legal for {action_name}")
        if action.action_type in {
            ActionType.CONSIDER_RELATION,
            ActionType.SET_RELATION,
            ActionType.REMOVE_RELATION,
        }:
            if action.action_type is ActionType.CONSIDER_RELATION:
                # This action proposes an unordered pair. Use exactly the same
                # layer-based inference as _begin_relation_decision; raw name
                # order must not reject an otherwise legal binary proposal.
                source, target, relation = self._canonical_relation_candidate(
                    str(action.source), str(action.target)
                )
                requested = {"source": source, "target": target, "relation": relation.value}
            else:
                requested = {
                    "source": str(action.source),
                    "target": str(action.target),
                    "relation": action.relation.value if action.relation else "",
                }
            if requested not in parameters["relations"]:
                return (
                    "director_parameter_not_allowed",
                    f"relation tuple is not legal for {action_name}",
                )
        return None

    def _structural_exploration_satisfied(self) -> bool:
        configured = [node for node in self.graph.nodes.values() if node.configured]
        if len(configured) < 2:
            return False
        audit = self.topology_audit()
        return bool(
            audit["relation_count"] > 0
            and audit["weak_component_count"] == 1
            and not audit["disconnected_multi_agent"]
        )

    @staticmethod
    def _classify_mutation_error(action: CanvasAction, message: str) -> str:
        lowered = message.casefold()
        if "unknown agent" in lowered:
            return "unknown_agent"
        if "agent budget exceeded" in lowered:
            return "agent_cap_exceeded"
        if "already exists" in lowered and action.action_type is ActionType.ADD_AGENT:
            return "duplicate_agent_id"
        if "relation" in lowered or "layer" in lowered:
            return "invalid_relation"
        if action.action_type is ActionType.SET_PROMPT:
            return "responsibility_violation"
        return "invalid_graph_edit"

    def _rejection_details(self, code: str) -> dict[str, Any]:
        snapshot = self.control_snapshot()
        recovery_actions: list[dict[str, Any]] = []
        if self.unified:
            return {"reason_code": code, "current_version": self.graph.version, "legal_agent_ids": list(snapshot["legal_agent_ids"]), "allowed_actions": list(snapshot["allowed_actions"]), "result_assessments": snapshot["result_assessments"]}
        if self.pending_agent_id:
            recovery_actions.append({"action": "set_prompt", "target": self.pending_agent_id})
        else:
            recovery_actions.extend(
                {"action": "set_output", "target": agent_id}
                for agent_id in snapshot["legal_action_parameters"]["set_output"]["targets"]
            )
            if self.structural_repair_reason is not None:
                recovery_actions.extend(
                    {"action": "delete_agent", "target": agent_id}
                    for agent_id in snapshot["legal_action_parameters"]["delete_agent"]["targets"]
                    if ActionType.DELETE_AGENT.value in snapshot["allowed_actions"]
                )
            if code == "output_answer_invalid" and "set_prompt" in snapshot["allowed_actions"]:
                target = self.graph.output_agent
                if target in snapshot["legal_action_parameters"]["set_prompt"]["targets"]:
                    recovery_actions.append(
                        {
                            "action": "set_prompt",
                            "target": target,
                            "revision_basis": "protocol_failure",
                            "evidence_agent_ids": [target],
                        }
                    )
            if (
                code != "output_answer_invalid"
                and ActionType.FINISH.value in snapshot["allowed_actions"]
            ):
                recovery_actions.append({"action": "finish"})
        return {
            "reason_code": code,
            "current_version": self.graph.version,
            "legal_agent_ids": list(snapshot["legal_agent_ids"]),
            "allowed_actions": list(snapshot["allowed_actions"]),
            "legal_recovery_actions": recovery_actions[:8],
        }

    def _reject_graph_action(
        self,
        action: CanvasAction,
        *,
        code: str,
        message: str,
        responsibility_issue: dict[str, Any] | None = None,
        rejection_details: dict[str, Any] | None = None,
        delegation_field_repairs: list[dict[str, Any]] | None = None,
    ) -> CanvasStep:
        return self._record(
            action,
            accepted=False,
            feedback=f"Rejected action: {message}.",
            rejection_code=code,
            responsibility_issue=responsibility_issue,
            rejection_details=rejection_details,
            delegation_field_repairs=delegation_field_repairs,
        )
