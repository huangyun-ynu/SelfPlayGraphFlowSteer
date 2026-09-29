from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .aime_submission import is_aime_dataset
from .alfworld import alfworld_lifecycles
from .answer_submission import AnswerFinalizer, AnswerSubmission, qa_token_f1
from .canvas import GraphCanvas
from .config import CanvasConfig, canonical_dataset_name
from .dataset_actions import DatasetActionRegistry
from .dataset_adapters import solver_task_text
from .deadline import RolloutDeadline
from .director import DirectorRun, GraphDirector
from .graph import FlowSteerStructureEvaluation
from .llm import ChatBackend
from .observability import (
    ExactMatchVerifier,
    ExecutionTrace,
    JSONLTraceStore,
    MultipleChoiceVerifier,
    NumericVerifier,
    TaskSpec,
    VerificationResult,
    Verifier,
    trace_from_canvas,
)
from .outcome_admission import terminal_policy_failure, trusted_environment_outcome
from .qa_metrics import hotpot_evidence_metrics, qa_official_metrics
from .runtime import (
    WORKER_BACKEND_FAILURE_SENTINEL,
    WORKER_PROTOCOL_FAILURE_SENTINEL,
    MultiAgentRuntime,
    artifact_backend_failure_records,
    artifact_integrity_failure_risks,
)
from .skills import SolverSkillBank
from .submission_contract import (
    SUBMISSION_CONTRACT_VERSION,
    OutcomeDecision,
    decide_text_outcome,
    is_text_submission_dataset,
    receipt_error,
    snapshot_hash,
)
from .swebench import public_swe_evaluation, swe_lifecycles
from .webshop import webshop_lifecycles
from .worker_usage_ledger import WorkerUsageLedger

_SWE_INFRASTRUCTURE_STATUSES = frozenset({"infrastructure_error", "timeout", "cancelled"})
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


def _deduplicate_backend_request_events(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        signature = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(dict(event))
    return unique


def _aggregate_output_agent_tool_evidence(
    canvas: GraphCanvas,
    output_artifact: Any,
) -> None:
    """Make final-output integrity reflect its whole Worker execution history.

    A prompt revision replaces the current Artifact, but it does not erase the
    Actions that produced the preceding Artifact.  Looking only at the final
    revision can therefore incorrectly label an Agent as having had *all* tool
    Actions fail, even when earlier initial Actions succeeded.  Preserve the
    current-Artifact facts for debugging while publishing an output-Agent-wide
    ledger for terminal eligibility and reporting.
    """

    if output_artifact is None or getattr(canvas, "unified", False):
        return
    output_agent = str(getattr(output_artifact, "agent_id", "")).strip()
    if not output_agent:
        return

    current_evidence = dict(getattr(output_artifact, "runtime_tool_evidence", {}) or {})
    current_evidence = dict(current_evidence.get("current_artifact_evidence", current_evidence))
    seen_artifacts: set[str] = set()
    successful_call_ids: list[str] = []
    failed_call_ids: list[str] = []
    failure_codes: list[str] = []
    for step in canvas.history:
        execution = step.execution
        if execution is None:
            continue
        for artifact in execution.artifacts.values():
            if str(getattr(artifact, "agent_id", "")) != output_agent:
                continue
            artifact_id = str(getattr(artifact, "artifact_id", "")).strip()
            # Execution reports contain snapshots of the Runtime cache.  The
            # same Artifact may appear in later zero-work synchronization
            # reports, so count each material Artifact only once.
            identity = artifact_id or f"history-{id(artifact)}"
            if identity in seen_artifacts:
                continue
            seen_artifacts.add(identity)
            evidence = dict(getattr(artifact, "runtime_tool_evidence", {}) or {})
            evidence = dict(evidence.get("current_artifact_evidence", evidence))
            successful_call_ids.extend(
                f"{identity}:{value}"
                for value in evidence.get("successful_call_ids", ())
                if str(value).strip()
            )
            failed_call_ids.extend(
                f"{identity}:{value}"
                for value in evidence.get("failed_call_ids", ())
                if str(value).strip()
            )
            failure_codes.extend(
                str(value) for value in evidence.get("failure_codes", ()) if str(value).strip()
            )

    # The selected Artifact can be produced by a recovery step after the last
    # normal Canvas execution.  Include it even if no history report retained
    # it yet.
    current_id = str(getattr(output_artifact, "artifact_id", "")).strip()
    current_identity = current_id or f"current-{id(output_artifact)}"
    if current_identity not in seen_artifacts:
        seen_artifacts.add(current_identity)
        successful_call_ids.extend(
            f"{current_identity}:{value}"
            for value in current_evidence.get("successful_call_ids", ())
            if str(value).strip()
        )
        failed_call_ids.extend(
            f"{current_identity}:{value}"
            for value in current_evidence.get("failed_call_ids", ())
            if str(value).strip()
        )
        failure_codes.extend(
            str(value) for value in current_evidence.get("failure_codes", ()) if str(value).strip()
        )

    attempted_count = len(successful_call_ids) + len(failed_call_ids)
    successful_count = len(successful_call_ids)
    failed_count = len(failed_call_ids)
    if attempted_count == 0:
        return

    all_actions_failed = successful_count == 0
    updated_evidence = {
        **current_evidence,
        "current_artifact_evidence": dict(current_evidence),
        "attempted_count": attempted_count,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "successful_call_ids": successful_call_ids,
        "failed_call_ids": failed_call_ids,
        "failure_codes": list(dict.fromkeys(failure_codes)),
        "all_actions_failed": all_actions_failed,
        "output_agent_execution_history": {
            "artifact_count": len(seen_artifacts),
            "current_artifact_attempted_count": int(
                current_evidence.get("attempted_count", 0) or 0
            ),
            "current_artifact_successful_count": int(
                current_evidence.get("successful_count", 0) or 0
            ),
            "current_artifact_failed_count": int(current_evidence.get("failed_count", 0) or 0),
        },
    }
    if not all_actions_failed:
        # These two risks are defined by the absence of any trusted tool
        # success.  They are false once the full output-Agent history contains
        # a successful Action; keep terminal failure itself intact.
        cleared = {
            "all_tool_actions_failed",
            "unsupported_tool_verification_claim",
        }
        output_artifact.integrity_risks = [
            risk for risk in output_artifact.integrity_risks if risk not in cleared
        ]
        output_artifact.unresolved_issues = [
            issue
            for issue in output_artifact.unresolved_issues
            if issue not in {f"runtime_integrity:{risk}" for risk in cleared}
        ]
        confidence_caps = dict(updated_evidence.get("confidence_caps", {}) or {})
        for risk in cleared:
            confidence_caps.pop(risk, None)
        updated_evidence["confidence_caps"] = confidence_caps
        claimed_confidence = float(
            getattr(output_artifact, "claimed_confidence", None)
            if getattr(output_artifact, "claimed_confidence", None) is not None
            else getattr(output_artifact, "confidence", 0.0)
        )
        output_artifact.confidence = max(
            0.0,
            min(1.0, claimed_confidence, *confidence_caps.values()),
        )
    output_artifact.runtime_tool_evidence = updated_evidence


def _is_model_attributed_terminal_tool_failure(artifact: Any) -> bool:
    """Whether a terminal tool failure is a completed model-policy outcome."""

    evidence = dict(getattr(artifact, "runtime_tool_evidence", {}) or {})
    failure_codes = {
        str(value).strip() for value in evidence.get("failure_codes", ()) if str(value).strip()
    }
    return bool(
        evidence.get("terminal_failure")
        and failure_codes
        and failure_codes <= _MODEL_ATTRIBUTED_ACTION_FAILURE_CODES
    )


def _swe_is_infrastructure_failure(evaluation: dict[str, Any]) -> bool:
    return str(evaluation.get("status", "")).strip().casefold() in (_SWE_INFRASTRUCTURE_STATUSES)


def _runtime_owned_swe_policy_evaluation(
    artifact: Any,
) -> dict[str, Any] | None:
    if artifact is None or not isinstance(artifact.swe_progress, dict):
        return None
    raw_policy_failure = artifact.swe_progress.get("policy_failure")
    if not isinstance(raw_policy_failure, dict):
        return None
    policy_failure = dict(raw_policy_failure)
    if not (
        policy_failure.get("status") == "typed_policy_failure"
        and policy_failure.get("attribution") == "model_policy"
        and policy_failure.get("normalized_diff_empty") is True
    ):
        return None
    return {
        "status": "typed_policy_failure",
        "environment_completed": True,
        "official": False,
        "synthetic": False,
        "patch_applied": False,
        "runtime_owned": True,
        "attribution": "model_policy",
        "failure_code": str(policy_failure.get("code", "read_only_policy_stall")),
        "detail": "trusted local Action ledger; remote harness not invoked",
    }


def _requires_structural_exploration(
    action_adapter: Any,
    seed: int,
    policy: str,
) -> bool:
    if policy == "off":
        return False
    if policy != "stratified":
        raise ValueError("structural_exploration_policy must be 'off' or 'stratified'")
    return bool(
        action_adapter is not None
        and action_adapter.environment_state.value == "stateless"
        and int(seed) % 5 in {1, 2, 3}
    )


@dataclass
class AdaptiveSolverResult:
    director_run: DirectorRun
    verification: VerificationResult | None
    trace: ExecutionTrace
    flowsteer_structure: FlowSteerStructureEvaluation
    skills_used: tuple[str, ...] = ()
    skill_context: dict[str, Any] = field(default_factory=dict)
    answer_submission: AnswerSubmission | None = None
    outcome_decision: OutcomeDecision | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "director_run": self.director_run.to_dict(),
            "submission_contract_version": (self.director_run.submission_receipt.version if self.director_run.submission_receipt else SUBMISSION_CONTRACT_VERSION),
            "outcome_decision": self.outcome_decision.to_dict() if self.outcome_decision else None,
            "submission_receipt": (self.director_run.submission_receipt.to_dict()
                                   if self.director_run.submission_receipt else None),
            "candidate_output": self.director_run.candidate_output,
            "verification": (
                {
                    "score": self.verification.score,
                    "passed": self.verification.passed,
                    "verifier": self.verification.verifier,
                    "detail": self.verification.detail,
                }
                if self.verification
                else None
            ),
            "trace": self.trace.to_dict(),
            "flowsteer_structure": self.flowsteer_structure.to_dict(),
            "skills_used": list(self.skills_used),
            "skill_context": self.skill_context,
            "answer_submission": (
                self.answer_submission.to_dict() if self.answer_submission else None
            ),
        }


class AdaptiveWorkflowSolver:
    """Stage-two inference integration; it never updates model parameters."""

    def __init__(
        self,
        *,
        director_backend: ChatBackend,
        runtime: MultiAgentRuntime,
        verifier: Verifier | None = None,
        skillbank: SolverSkillBank | None = None,
        trace_store: JSONLTraceStore | None = None,
        canvas_config: CanvasConfig | None = None,
        runtime_routes: tuple[str, ...] = (),
        model_router: None = None,
        action_registry: DatasetActionRegistry | None = None,
        answer_finalizer: AnswerFinalizer | None = None,
        rollout_deadline: RolloutDeadline | None = None,
        swe_duplicate_responsibility_policy: str = "record_only",
        director_prompt_variant: str = "v2.1",
        director_enable_thinking: bool | None = None,
        director_thinking_by_dataset: dict[str, bool] | None = None,
        required_nq_frozen_top_k: int = 0,
        nq_evidence_mode: str | None = None,
        nq_policy: Any | None = None,
        director_tokenizer: Any | None = None,
        post_director_hook: Callable[[TaskSpec, GraphCanvas, DirectorRun], None] | None = None,
    ) -> None:
        self.director_backend = director_backend
        self.runtime = runtime
        self.verifier = verifier
        self.skillbank = skillbank
        self.trace_store = trace_store
        self.canvas_config = canvas_config
        self.runtime_routes = tuple(runtime_routes)
        if model_router is not None:
            raise ValueError("MACE model routing is retired; use Director SET_MODEL")
        self.action_registry = action_registry or DatasetActionRegistry()
        self.answer_finalizer = answer_finalizer
        self.rollout_deadline = rollout_deadline
        self.swe_duplicate_responsibility_policy = (
            str(swe_duplicate_responsibility_policy).strip().casefold()
        )
        self.director_prompt_variant = str(director_prompt_variant).strip().casefold()
        self.director_enable_thinking = director_enable_thinking
        self.director_thinking_by_dataset = {
            canonical_dataset_name(dataset): enabled
            for dataset, enabled in (director_thinking_by_dataset or {}).items()
        }
        self.required_nq_frozen_top_k = required_nq_frozen_top_k
        self.nq_evidence_mode = nq_evidence_mode
        self.nq_policy = nq_policy
        self.director_tokenizer = director_tokenizer
        self.post_director_hook = post_director_hook
        self.active_canvas: GraphCanvas | None = None

    def finalize_answer(
        self,
        task: TaskSpec,
        raw_answer: str,
        *,
        raw_summary: str = "",
    ) -> AnswerSubmission:
        if self.answer_finalizer is None and is_aime_dataset(task.metadata.get("dataset", "")):
            return AnswerFinalizer().finalize(task, raw_answer)
        if self.answer_finalizer is None:
            raw = str(raw_answer or "").strip()
            return AnswerSubmission(
                raw_answer=raw,
                submitted_answer=raw,
                method="legacy_passthrough",
                changed=False,
                valid=bool(raw),
            )
        return self.answer_finalizer.finalize(
            task,
            raw_answer,
            raw_summary=raw_summary,
        )

    def _prepare_text_submission(self, task: TaskSpec, canvas: GraphCanvas) -> AnswerSubmission:
        """Normalize/recover public output before FINISH is accepted; never inspect gold."""
        artifact = self.runtime.artifacts.get(canvas.graph.output_agent)
        _aggregate_output_agent_tool_evidence(canvas, artifact)
        if (
            is_aime_dataset(canvas.dataset) and artifact is not None
            and "terminal_tool_failure" in artifact_integrity_failure_risks(artifact)
            and not _is_model_attributed_terminal_tool_failure(artifact)
            and not getattr(canvas, "_text_recovery_attempted", False)
        ):
            canvas._text_recovery_attempted = True
            before_id = artifact.artifact_id
            before_risks = artifact_integrity_failure_risks(artifact)
            recovery = canvas.recover_selected_output_agent(reason_code="aime_terminal_tool_failure")
            artifact = self.runtime.artifacts.get(canvas.graph.output_agent)
            _aggregate_output_agent_tool_evidence(canvas, artifact)
            after_risks = (artifact_integrity_failure_risks(artifact) if artifact else ["missing_output_artifact"])
            if artifact is not None and not after_risks:
                artifact.runtime_tool_evidence["recovered_failure"] = True
                artifact.runtime_tool_evidence.setdefault("current_artifact_evidence", {})["recovered_failure"] = True
            task.metadata["selected_output_recovery"] = {
                "attempted": True, "dataset": "aime", "scope": "selected_output_agent",
                "phase": "before_finish_acceptance", "reason": "terminal_tool_failure",
                "output_agent": canvas.graph.output_agent,
                "before_artifact_id": before_id,
                "after_artifact_id": artifact.artifact_id if artifact else None,
                "before_integrity_risks": before_risks, "after_integrity_risks": after_risks,
                "recovered": not after_risks,
                "worker_model_calls": recovery.execution.worker_model_calls_total if recovery.execution else 0,
            }
        submission = self.finalize_answer(
            task, artifact.answer if artifact else "",
            raw_summary=artifact.summary if artifact else "",
        )
        if self.nq_evidence_mode == "corpus_tool" and canonical_dataset_name(
            task.metadata.get("dataset")
        ) == "nq_open":
            check = self.runtime.validate_nq_submission(str(canvas.graph.output_agent or ""))
            task.metadata["nq_corpus_submission"] = dict(check)
            status = str(check.get("status", ""))
            if status == "insufficient_evidence" and check.get("valid"):
                return AnswerSubmission(
                    raw_answer=submission.raw_answer,
                    # FINISH receipts bind a nonempty immutable answer snapshot.
                    # Keep the explicit abstention sentinel; its answerability
                    # status is recorded above and receives zero below.
                    submitted_answer="insufficient_evidence",
                    method="nq_insufficient_evidence_v1",
                    changed=submission.raw_answer != "insufficient_evidence",
                    valid=True,
                    detail=str(check.get("reason", "")),
                )
            if not check.get("valid"):
                return replace(
                    submission,
                    valid=False,
                    detail=str(check.get("reason") or "invalid_evidence_submission"),
                )
        return submission

    def solve(self, task: TaskSpec, *, run_id: str) -> AdaptiveSolverResult:
        try:
            return self._solve(task, run_id=run_id)
        except BaseException:
            # Preparing an environment goal starts a real episode before the
            # Director. Configuration/skill/ledger failures must close it too.
            try:
                for lifecycle in alfworld_lifecycles(getattr(self.runtime.executor, "tools", {})):
                    lifecycle.close_all()
            finally:
                self.runtime.close_worker_usage_ledger()
            raise

    def _solve(self, task: TaskSpec, *, run_id: str) -> AdaptiveSolverResult:
        nq_task = canonical_dataset_name(task.metadata.get("dataset")) == "nq_open"
        if nq_task and self.nq_evidence_mode is not None:
            actual_mode = str(task.metadata.get("evidence_mode", "")).strip().casefold()
            if actual_mode != self.nq_evidence_mode:
                raise ValueError(
                    f"NQ task evidence_mode={actual_mode!r} conflicts with configured "
                    f"{self.nq_evidence_mode!r}"
                )
            if self.nq_evidence_mode == "corpus_tool":
                from .nq_corpus_tasks import validate_corpus_task

                validate_corpus_task(
                    {"id": task.task_id, "prompt": task.prompt, "metadata": task.metadata}
                )
                self.runtime.configure_nq_corpus(
                    task_id=task.task_id,
                    policy=self.nq_policy,
                    run_id=run_id,
                    trajectory_id=f"{run_id}:{self.runtime.seed}:{task.task_id}",
                )
        if (
            self.required_nq_frozen_top_k
            and nq_task
        ):
            from .nq_frozen_context import validate_frozen_context

            validate_frozen_context(
                {"id": task.task_id, "prompt": task.prompt, "metadata": task.metadata},
                top_k=self.required_nq_frozen_top_k,
            )
        task.metadata["judge_evaluation_scope"] = f"primary:{run_id}:{self.runtime.seed}"

        action_adapter = self.action_registry.resolve(task)
        lifecycle_tools = getattr(self.runtime.executor, "tools", {})
        active_webshop_lifecycles = webshop_lifecycles(lifecycle_tools)
        active_alfworld_lifecycles = alfworld_lifecycles(lifecycle_tools)
        active_swe_lifecycles = swe_lifecycles(lifecycle_tools)
        if action_adapter is not None and action_adapter.adapter_id == "webshop":
            if not active_webshop_lifecycles:
                raise ValueError("WebShop task requires configured WebShop Actions")
            for lifecycle in active_webshop_lifecycles:
                lifecycle.bind_task(task)
            self.runtime.environment_fingerprint = active_webshop_lifecycles[
                0
            ].environment_fingerprint
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            if not active_alfworld_lifecycles:
                raise ValueError("ALFWorld task requires configured ALFWorld Actions")
            for lifecycle in active_alfworld_lifecycles:
                lifecycle.bind_task(task)
            self.runtime.environment_fingerprint = active_alfworld_lifecycles[
                0
            ].environment_fingerprint
        if action_adapter is not None and action_adapter.adapter_id == "swe_bench":
            if not active_swe_lifecycles:
                raise ValueError("SWE-bench task requires configured SWE Actions")
            for lifecycle in active_swe_lifecycles:
                lifecycle.bind_task(task)
            self.runtime.environment_fingerprint = active_swe_lifecycles[0].environment_fingerprint
        effective_task = task.prompt
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            effective_task = active_alfworld_lifecycles[0].effective_task
        worker_task = (effective_task if action_adapter is not None and action_adapter.adapter_id == "alfworld"
                       else solver_task_text(task, include_submission_contract=self.answer_finalizer is not None))
        skill_manifest = {}
        if self.skillbank and hasattr(self.skillbank, "select_context"):
            scope_kwargs = (
                {"task_metadata": task.metadata}
                if getattr(self.skillbank, "supports_task_metadata", False)
                else {}
            )
            selected_skills, skill_context, skill_manifest = self.skillbank.select_context(
                effective_task,
                task_type=task.task_type,
                tokenizer=self.director_tokenizer,
                tools=getattr(self.runtime.executor, "tools", {}).keys(),
                **scope_kwargs,
            )
        else:
            selected_skills = (
                self.skillbank.retrieve(effective_task, task_type=task.task_type)
                if self.skillbank
                else []
            )
            skill_context = SolverSkillBank.format_prompt_context(selected_skills)
        base_canvas_config = self.canvas_config or CanvasConfig()
        dataset_key, selected_token_budget = base_canvas_config.token_budget_for_dataset(
            task.metadata.get("dataset", "")
        )
        task.metadata["canvas_token_budget"] = {
            "dataset": dataset_key,
            "max_total_tokens": selected_token_budget,
            "fallback_max_total_tokens": base_canvas_config.max_total_tokens,
        }
        canvas = GraphCanvas(
            task=worker_task,
            worker_task=worker_task,
            # HealthBench evaluates the next reply in a public conversation;
            # task.prompt may contain only a context-dependent follow-up.
            director_task=(
                solver_task_text(task)
                if action_adapter is not None
                and action_adapter.adapter_id == "healthbench_professional"
                else effective_task
            ),
            runtime=self.runtime,
            config=replace(base_canvas_config, max_total_tokens=selected_token_budget),
            runtime_routes=self.runtime_routes,
            task_type=task.task_type,
            action_adapter=action_adapter,
            dataset=str(task.metadata.get("dataset", "")),
            duplicate_responsibility_policy=(self.swe_duplicate_responsibility_policy),
            rollout_deadline=self.rollout_deadline,
            # The default policy is off so graph size remains task-conditioned.
            # The optional stratified ablation reproduces the historical behavior:
            # three of five stateless sibling seeds must try a connected multi-Agent
            # graph, while stateful environments retain dynamic topology choice.
            structural_exploration_required=_requires_structural_exploration(
                action_adapter,
                self.runtime.seed,
                base_canvas_config.structural_exploration_policy,
            ),
            binary_relation_policy=self.director_tokenizer is not None,
        )
        canvas.run_id = run_id
        usage_policy = base_canvas_config.worker_usage_policy(dataset_key)
        if usage_policy is not None:
            ledger_file = (
                Path(base_canvas_config.submission_journal_dir)
                / "worker_usage"
                / (hashlib.sha256(run_id.encode()).hexdigest() + ".sqlite3")
            )
            self.runtime.worker_usage_ledger = WorkerUsageLedger(
                ledger_file,
                question_attempt_id=run_id,
                threshold=int(usage_policy.get("start_threshold", selected_token_budget)),
                max_unsettled_attempts=int(usage_policy.get("max_unsettled_attempts", 2)),
            )
            canvas.total_tokens = self.runtime.worker_usage_ledger.status()["confirmed_used"]
        canvas.prepare_text_submission = lambda active: self._prepare_text_submission(task, active)
        self.active_canvas = canvas
        try:
            run = GraphDirector(
                backend=self.director_backend,
                canvas=canvas,
                solver_skill_context=skill_context,
                prompt_variant=self.director_prompt_variant,
                enable_thinking=self.director_thinking_by_dataset.get(
                    dataset_key, self.director_enable_thinking
                ),
                tokenizer=self.director_tokenizer,
                call_namespace=run_id,
            ).run()
        except Exception:
            for lifecycle in (
                *active_webshop_lifecycles,
                *active_alfworld_lifecycles,
                *active_swe_lifecycles,
            ):
                lifecycle.close_all()
            raise
        if self.runtime.worker_usage_ledger is not None:
            ledger = self.runtime.worker_usage_ledger
            canvas.total_tokens = ledger.status()["confirmed_used"]
            task.metadata["worker_usage"] = {
                **ledger.status(), "digest": ledger.digest(),
                "dispatch_policy_valid": ledger.dispatches_valid(),
                "question_attempt_id": run_id,
            }
        if active_swe_lifecycles:
            task.metadata["swe_recoverable_code_artifacts"] = [
                candidate
                for lifecycle in active_swe_lifecycles
                for candidate in lifecycle.recoverable_code_artifacts()
            ]
        receipt = canvas.submission_receipt
        if nq_task and self.nq_evidence_mode == "corpus_tool":
            task.metadata["nq_corpus_evidence_audit"] = self.runtime.nq_evidence_audit()
        frozen_run = copy.deepcopy(run) if receipt is not None else None
        if self.post_director_hook is not None:
            before_artifacts = snapshot_hash({key: value.to_dict() for key, value in self.runtime.artifacts.items()})
            before_binding = (self.runtime.artifact_input_binding(receipt.output_agent_id)
                              if receipt is not None else None)
            try:
                self.post_director_hook(task, canvas, run)
            except Exception as exc:
                if receipt is None:
                    raise
                task.metadata["post_submission_hook_error"] = type(exc).__name__
                task.metadata["output_contract_failure"] = {"reason": "post_submission_hook_error"}
            if receipt is not None:
                artifact = self.runtime.artifacts.get(receipt.output_agent_id)
                if (run.to_dict() != frozen_run.to_dict()
                        or canvas.state.value != "finished"
                        or snapshot_hash({key: value.to_dict() for key, value in self.runtime.artifacts.items()}) != before_artifacts
                        or snapshot_hash(canvas.graph.to_dict()) != receipt.graph_snapshot_hash
                        or canvas.submission_receipt is not receipt
                        or artifact is None or artifact.answer != receipt.raw_answer_snapshot
                        or artifact.artifact_id != receipt.artifact_id
                        or self.runtime.artifact_input_binding(receipt.output_agent_id) != before_binding):
                    task.metadata["output_contract_failure"] = {"reason": "post_submission_mutation"}
                run = frozen_run
                canvas.submission_receipt = receipt
        output_artifact = (
            self.runtime.artifacts.get(canvas.graph.output_agent)
            if canvas.graph.output_agent
            else None
        )
        flowsteer_structure = canvas.evaluate_flowsteer_structure()
        output_artifact = (
            self.runtime.artifacts.get(canvas.graph.output_agent)
            if canvas.graph.output_agent
            else None
        )
        _aggregate_output_agent_tool_evidence(canvas, output_artifact)
        artifact_integrity = {
            agent_id: {
                "claimed_confidence": artifact.claimed_confidence,
                "effective_confidence": artifact.confidence,
                "integrity_risks": list(artifact.integrity_risks),
                "runtime_tool_evidence": dict(artifact.runtime_tool_evidence),
                "swe_progress": dict(artifact.swe_progress),
                "alfworld_progress": dict(artifact.alfworld_progress),
                "webshop_progress": dict(artifact.webshop_progress),
            }
            for agent_id, artifact in self.runtime.artifacts.items()
        }
        historical_integrity = {}
        if canvas.unified:
            for step in canvas.history:
                if step.execution is None:
                    continue
                for artifact in [*step.execution.artifacts.values(),
                                 *getattr(step.execution, 'attempt_artifacts', ())]:
                    historical_integrity[artifact.artifact_id] = {
                        'agent_id': artifact.agent_id,
                        'backend_failure': artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL,
                        'runtime_tool_evidence': dict(artifact.runtime_tool_evidence),
                    }
            task.metadata['worker_artifact_history_integrity'] = historical_integrity
        output_integrity_failure_risks = (
            artifact_integrity_failure_risks(output_artifact) if output_artifact is not None else []
        )
        task.metadata["worker_artifact_integrity"] = artifact_integrity
        task.metadata["worker_output_integrity_risks"] = list(
            output_artifact.integrity_risks if output_artifact is not None else []
        )
        task.metadata["swe_output_progress"] = (
            dict(output_artifact.swe_progress)
            if output_artifact is not None and output_artifact.swe_progress
            else {}
        )
        task.metadata["alfworld_output_progress"] = (
            dict(output_artifact.alfworld_progress)
            if output_artifact is not None and output_artifact.alfworld_progress
            else {}
        )
        task.metadata["webshop_output_progress"] = (
            dict(output_artifact.webshop_progress)
            if output_artifact is not None and output_artifact.webshop_progress
            else {}
        )
        task.metadata["worker_artifact_integrity_failure"] = (
            {
                "output_agent": output_artifact.agent_id,
                "risks": output_integrity_failure_risks,
                "claimed_confidence": output_artifact.claimed_confidence,
                "effective_confidence": output_artifact.confidence,
            }
            if output_artifact is not None and output_integrity_failure_risks
            else None
        )
        unified_binding_error = None
        if canvas.unified:
            unified_binding_error = receipt_error(receipt, run=run, events=canvas.history,
                                                  run_id=run_id, dataset=dataset_key)
            if unified_binding_error:
                receipt = None
                task.metadata["submission_binding_error"] = unified_binding_error
            task.metadata["submission_status"] = "submitted" if receipt else "unsubmitted"
            task.metadata["submission_receipt"] = receipt.to_dict() if receipt else None
            task.metadata["submission_transaction"] = copy.deepcopy(canvas._unified_transaction)
        if action_adapter is not None and action_adapter.adapter_id == "webshop":
            output_agent = canvas.graph.output_agent if not canvas.unified or receipt else None
            environment_result = active_webshop_lifecycles[0].result_for(output_agent)
            task.metadata["webshop_environment_result"] = environment_result
            for lifecycle in active_webshop_lifecycles:
                lifecycle.close_all()
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            winning_artifacts = [
                artifact
                for artifact in self.runtime.artifacts.values()
                if artifact.environment_result.get("environment_completed") is True
                and artifact.environment_result.get("won") is True
            ]
            selected_output = canvas.graph.output_agent
            selected_result = (
                dict(output_artifact.environment_result)
                if output_artifact is not None and output_artifact.environment_result
                else active_alfworld_lifecycles[0].result_for(selected_output)
            )
            environment_result = dict(selected_result or {})
            task.metadata["alfworld_unselected_winning_episodes"] = [
                {
                    "agent_id": artifact.agent_id,
                    "environment_result": dict(artifact.environment_result),
                }
                for artifact in sorted(winning_artifacts, key=lambda item: item.agent_id)
                if artifact.agent_id != selected_output
            ]
            task.metadata["alfworld_environment_result"] = environment_result
            for lifecycle in active_alfworld_lifecycles:
                lifecycle.close_all()
        if action_adapter is not None and action_adapter.adapter_id == "swe_bench":
            output_agent = canvas.graph.output_agent if not canvas.unified or receipt else None
            environment_result = (
                dict(output_artifact.environment_result)
                if output_artifact and output_artifact.environment_result
                else active_swe_lifecycles[0].result_for(output_agent)
            )
            policy_evaluation = _runtime_owned_swe_policy_evaluation(output_artifact)
            if output_agent and policy_evaluation is not None:
                # This terminal state is proven by the trusted local Action
                # ledger, not by private SWE-bench tests.  Calling the remote
                # harness for an empty patch would add latency and could turn a
                # valid model-policy negative into an infrastructure failure.
                evaluation = policy_evaluation
            elif output_agent and active_swe_lifecycles[0].harness_backend is not None:
                private_evaluation = active_swe_lifecycles[0].evaluate_artifact(output_agent)
                evaluation = public_swe_evaluation(private_evaluation)
            elif not output_agent:
                evaluation = {
                    "status": "not_submitted",
                    "environment_completed": False,
                    "official": False,
                    "synthetic": False,
                    "detail": "SWE output agent is not set; harness was not invoked",
                }
            else:
                evaluation = {
                    "status": "infrastructure_error",
                    "environment_completed": False,
                    "official": False,
                    "synthetic": False,
                    "detail": "SWE harness backend is not configured",
                }
            task.metadata["swe_workspace_result"] = environment_result
            task.metadata["swe_environment_result"] = evaluation
            if _swe_is_infrastructure_failure(evaluation):
                task.metadata["swe_infrastructure_failure"] = {
                    "status": evaluation.get("status", "infrastructure_error"),
                    "detail": evaluation.get("detail", ""),
                }
            elif not bool(evaluation.get("environment_completed", False)):
                task.metadata["swe_execution_incomplete"] = {
                    "status": evaluation.get("status", "not_submitted"),
                    "detail": evaluation.get("detail", ""),
                }
            for lifecycle in active_swe_lifecycles:
                lifecycle.close_all()
        backend_failures = [
            artifact
            for artifact in self.runtime.artifacts.values()
            if artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
        ]
        backend_request_events = _deduplicate_backend_request_events(
            event
            for artifact in self.runtime.artifacts.values()
            for event in artifact.backend_request_events
        )
        task.metadata["backend_request_events"] = backend_request_events
        stale_output = bool(
            output_artifact is not None
            and output_artifact.answer not in {
                WORKER_BACKEND_FAILURE_SENTINEL, WORKER_PROTOCOL_FAILURE_SENTINEL
            }
            and not canvas.selected_output_is_current()
        )
        task.metadata["output_artifact_binding"] = self.runtime.artifact_input_binding(
            str(canvas.graph.output_agent or "")
        )
        if stale_output:
            task.metadata["output_contract_failure"] = {
                "reason": "stale_output_artifact", "output_agent": canvas.graph.output_agent,
                "artifact_id": output_artifact.artifact_id,
            }
        text_primary = is_text_submission_dataset(dataset_key)
        candidate_output = run.candidate_output
        task.metadata["submission_contract_version"] = "unified_submission_v1" if canvas.unified else SUBMISSION_CONTRACT_VERSION
        task.metadata["candidate_output"] = candidate_output
        candidate_submission = self.finalize_answer(
            task, candidate_output,
            raw_summary=output_artifact.summary if output_artifact else "",
        ) if text_primary else None
        task.metadata["candidate_answer_submission"] = (
            candidate_submission.to_dict() if candidate_submission else None
        )
        if text_primary:
            binding_error = unified_binding_error or receipt_error(
                receipt, run=run, events=canvas.history, run_id=run_id, dataset=dataset_key,
            )
            if binding_error:
                receipt = None
                run.output = ""
                answer_submission = AnswerSubmission(
                    raw_answer="", submitted_answer="", method="runtime_submission_gate",
                    changed=False, valid=False, detail=binding_error,
                )
            else:
                answer_submission = AnswerSubmission(
                    raw_answer=receipt.raw_answer_snapshot,
                    submitted_answer=receipt.submitted_answer_snapshot,
                    method=receipt.normalization_version,
                    changed=receipt.raw_answer_snapshot != receipt.submitted_answer_snapshot,
                    valid=True,
                )
            task.metadata["submission_status"] = "submitted" if receipt else "unsubmitted"
            task.metadata["submission_receipt"] = receipt.to_dict() if receipt else None
            task.metadata["diagnostic_qa_metrics"] = (
                qa_official_metrics(dataset_key, candidate_submission.submitted_answer, task.reference)
                if candidate_submission and candidate_submission.valid and not receipt else None
            )
            if dataset_key == "aime" and candidate_submission.valid and receipt is None:
                diagnostic = NumericVerifier().verify(task, candidate_submission.submitted_answer)
                task.metadata["diagnostic_qa_metrics"] = {
                    "schema": "aime_candidate_answer_em_v2", "em": diagnostic.score,
                }
        elif stale_output:
            run.output = ""
            task.metadata["output_contract_failure"] = {
                "reason": "stale_output_artifact", "output_agent": canvas.graph.output_agent,
                "artifact_id": output_artifact.artifact_id,
            }
            task.metadata["outcome_exclusion_reason"] = "stale_output_artifact"
            answer_submission = AnswerSubmission(
                raw_answer="", submitted_answer="", method="runtime_output_gate",
                changed=False, valid=False, detail="stale_output_artifact",
            )
        else:
            answer_submission = self.finalize_answer(
                task, run.output, raw_summary=output_artifact.summary if output_artifact else "",
            )
        if run.output.strip() in {WORKER_PROTOCOL_FAILURE_SENTINEL, WORKER_BACKEND_FAILURE_SENTINEL}:
            answer_submission = replace(answer_submission, submitted_answer="", valid=False,
                                        detail="runtime_failure_sentinel_not_an_answer")
        task.metadata["answer_submission"] = answer_submission.to_dict()
        scored_answer_available = bool(receipt) if text_primary else not stale_output
        task.metadata["qa_token_f1"] = (
            qa_token_f1(task, answer_submission.submitted_answer) if scored_answer_available else None
        )
        task.metadata["qa_official_metrics"] = (
            qa_official_metrics(task.metadata.get("dataset"), answer_submission.submitted_answer,
                                task.reference) if scored_answer_available else None
        )
        if (task.metadata["qa_official_metrics"] or {}).get("schema") == "hotpot_official_answer_v1":
            task.metadata["qa_official_metrics"]["evidence"] = hotpot_evidence_metrics(
                answer_submission.raw_answer, task.private_verifier_payload.get("supporting_facts"),
                task.metadata["qa_official_metrics"],
            )
        from .swe_failure_attribution import project_swe_step, swe_failure_attribution

        runtime_failure_evidence = (
            swe_failure_attribution(
                (project_swe_step(step) for step in canvas.history),
                worker_token_limit=canvas.config.max_total_tokens,
            ) if dataset_key == "swe_bench" else None
        )
        # Always overwrite incoming task metadata; only the runtime history can
        # establish these causes. They do not alter the Director's input/history.
        task.metadata["swe_failure_attribution"] = runtime_failure_evidence
        runtime_failure_reason = (
            runtime_failure_evidence["reason_codes"][0]
            if runtime_failure_evidence and runtime_failure_evidence["blocks_policy_failure"]
            else ""
        )
        terminal_failure = terminal_policy_failure(
            str(task.metadata.get("dataset", "")),
            terminal=not canvas.active,
            rejection_codes=[step.rejection_code for step in canvas.history if step.rejection_code],
            artifacts=artifact_integrity,
            historical_artifacts=historical_integrity,
            output_agent=canvas.graph.output_agent,
            rounds=canvas.round_index,
            max_rounds=canvas.director_round_limit or 0,
            director_edits=canvas.director_edits_used,
            director_edit_limit=canvas.config.director_edit_limit(canvas.dataset) if canvas.unified else None,
            worker_tokens=canvas.total_tokens,
            worker_token_limit=canvas.config.max_total_tokens,
            worker_budget_policy=(canvas.runtime.worker_usage_ledger.status()["policy"]
                                  if canvas.runtime.worker_usage_ledger else "strict_limit_v1"),
            worker_dispatch_valid=(canvas.runtime.worker_usage_ledger.dispatches_valid()
                                   if canvas.runtime.worker_usage_ledger else False),
            infrastructure_failure=bool(
                stale_output or backend_failures or task.metadata.get("swe_infrastructure_failure")
                or task.metadata.get("output_contract_failure")
            ),
            runtime_failure_evidence=runtime_failure_evidence,
        )
        task.metadata["runtime_terminal_policy_failure"] = terminal_failure
        trusted_environment = any(
            trusted_environment_outcome(dataset, task.metadata.get(key))
            for dataset, key in (
                ("alfworld", "alfworld_environment_result"),
                ("webshop", "webshop_environment_result"),
                ("swe_bench", "swe_environment_result"),
            )
        )
        outcome_decision = None
        if text_primary:
            verification = None
            scoring_error = ""
            if receipt is not None and receipt.normalization_version == "nq_insufficient_evidence_v1":
                verification = VerificationResult(
                    0.0, False, "nq_corpus_abstention_v1",
                    "answerability=insufficient_evidence; NQ-open abstentions receive zero",
                )
            elif receipt is not None and self.verifier is not None:
                # Retry only the immutable submitted answer, never the Worker graph.
                for attempt in range(2):
                    try:
                        verification = self.verifier.verify(task, receipt.submitted_answer_snapshot)
                        break
                    except Exception as exc:
                        scoring_error = "scoring_exception:" + type(exc).__name__
                        task.metadata.setdefault("submission_scoring_errors", []).append(
                            {"attempt": attempt + 1, "error_type": type(exc).__name__,
                             "receipt_ref": receipt.receipt_ref}
                        )
            outcome_decision = decide_text_outcome(
                receipt=receipt, verification=verification,
                terminal_failure=terminal_failure if receipt is None else None,
                reason=scoring_error or (
                    "execution_token_budget_overrun"
                    if receipt is None and canvas.total_tokens > canvas.config.max_total_tokens
                    else binding_error if receipt is None else ""
                ),
            )
            task.metadata["outcome_decision"] = outcome_decision.to_dict()
            if not outcome_decision.score_known:
                self.runtime.discard_peer_rewards(outcome_decision.status)
        elif canvas.unified and receipt is None:
            verification = None
            if terminal_failure is not None:
                verification = VerificationResult(0.0, False, "runtime_policy_terminal",
                    json.dumps(terminal_failure, ensure_ascii=False, sort_keys=True))
            else:
                task.metadata["outcome_exclusion_reason"] = runtime_failure_reason or unified_binding_error or "unsubmitted_unknown"
            self.runtime.discard_peer_rewards("unsubmitted")
        elif backend_failures:
            # Backend availability is not task correctness. Keep the trace for
            # diagnosis, but do not verify the sentinel or commit zero reward to
            # either MACE bandit. The rollout runner will leave this ID missing.
            self.runtime.discard_peer_rewards("worker_backend_failure")

            failure_details = [
                {**record, "agent_id": artifact.agent_id}
                for artifact in backend_failures
                for record in artifact_backend_failure_records(artifact)
            ]
            task.metadata["worker_backend_failure"] = {
                "count": len(backend_failures),
                "agents": sorted(artifact.agent_id for artifact in backend_failures),
                "routes": sorted(
                    {artifact.model_route or "unassigned" for artifact in backend_failures}
                ),
                "failure_types": sorted(
                    {str(record.get("kind", "unknown")) for record in failure_details}
                ),
                "failure_details": failure_details,
                "request_events": backend_request_events,
                "retryable": any(bool(record.get("retryable")) for record in failure_details),
                "counts_toward_route_circuit": any(
                    bool(record.get("counts_toward_route_circuit")) for record in failure_details
                ),
                "disable_route": any(
                    bool(record.get("disable_route")) for record in failure_details
                ),
            }
            # A later backend failure cannot erase an already committed
            # official environment result. Keep the incident as a separate
            # training exclusion; never ask a text Judge to score the sentinel.
            verification = (
                self.verifier.verify(task, answer_submission.submitted_answer)
                if trusted_environment and self.verifier
                else None
            )

        elif stale_output:
            verification = None
            self.runtime.discard_peer_rewards("stale_output_artifact")
        elif (
            terminal_failure
            and not trusted_environment
            and (
                not answer_submission.valid
                or canonical_dataset_name(task.metadata.get("dataset", ""))
                in {"alfworld", "webshop", "swe_bench"}
            )
        ):
            # A missing answer is not a low-quality HealthBench answer: only
            # this typed terminal path bypasses Judge. Valid answers still use
            # the complete official rubric and continuous reward adapter.
            verification = VerificationResult(
                0.0,
                False,
                "runtime_policy_terminal",
                json.dumps(terminal_failure, ensure_ascii=False, sort_keys=True),
            )
        elif task.metadata.get("swe_infrastructure_failure") or task.metadata.get(
            "swe_execution_incomplete"
        ):
            verification = None
            self.runtime.discard_peer_rewards(
                "swe_infrastructure_failure"
                if task.metadata.get("swe_infrastructure_failure")
                else "swe_execution_incomplete"
            )
        elif not answer_submission.valid and canonical_dataset_name(
            task.metadata.get("dataset", "")
        ) in {"aime", "nq_open", "hotpotqa", "healthbench_professional"}:
            # No legal answer and no explicit terminal evidence: unscored, not
            # an invented Judge failure/zero. A valid low-quality answer still
            # reaches the normal verifier below.
            verification = None
            task.metadata["outcome_exclusion_reason"] = "invalid_submission_attribution_unknown"
        else:
            verification = (
                VerificationResult(
                    0.0,
                    False,
                    self.verifier.name,
                    f"invalid_answer_submission:{answer_submission.detail}",
                )
                if self.verifier
                and not answer_submission.valid
                and is_aime_dataset(task.metadata.get("dataset", ""))
                else self.verifier.verify(task, answer_submission.submitted_answer)
                if self.verifier
                else None
            )
        if canvas.unified and not text_primary:
            outcome_decision = decide_text_outcome(
                receipt=receipt, verification=verification,
                terminal_failure=terminal_failure if receipt is None else None,
                reason=(runtime_failure_reason if receipt is None else "") or unified_binding_error or "",
            )
            task.metadata["outcome_decision"] = outcome_decision.to_dict()
        # Model selection is a Director action trained from final graph reward.
        # Never score a local responsibility against the original task answer.
        trace = trace_from_canvas(
            run_id=run_id,
            task=task,
            canvas=canvas,
            verification=verification,
        )
        if self.trace_store:
            self.trace_store.append(trace)
        return AdaptiveSolverResult(
            director_run=run,
            verification=verification,
            trace=trace,
            flowsteer_structure=flowsteer_structure,
            skills_used=tuple(skill.skill_id for skill in selected_skills),
            skill_context=skill_manifest,
            answer_submission=answer_submission,
            outcome_decision=outcome_decision,
        )


def _reference_reward_verifier(task: TaskSpec) -> Verifier | None:
    if task.reference is None or not str(task.reference).strip():
        return None
    task_type = task.task_type.casefold()
    if any(value in task_type for value in ("math", "numeric", "number")):
        return NumericVerifier()
    if any(value in task_type for value in ("multiple_choice", "choice", "mcq", "gpqa")):
        return MultipleChoiceVerifier()
    return ExactMatchVerifier()
