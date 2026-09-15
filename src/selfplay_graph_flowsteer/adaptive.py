from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
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
from .qa_metrics import qa_official_metrics
from .runtime import (
    WORKER_BACKEND_FAILURE_SENTINEL,
    WORKER_PROTOCOL_FAILURE_SENTINEL,
    MultiAgentRuntime,
    artifact_backend_failure_records,
    artifact_integrity_failure_risks,
)
from .skills import SolverSkillBank
from .swebench import public_swe_evaluation, swe_lifecycles
from .webshop import webshop_lifecycles

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

    if output_artifact is None:
        return
    output_agent = str(getattr(output_artifact, "agent_id", "")).strip()
    if not output_agent:
        return

    current_evidence = dict(getattr(output_artifact, "runtime_tool_evidence", {}) or {})
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

    def to_dict(self) -> dict[str, object]:
        return {
            "director_run": self.director_run.to_dict(),
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
        self.director_tokenizer = director_tokenizer
        self.post_director_hook = post_director_hook
        self.active_canvas: GraphCanvas | None = None

    def finalize_answer(
        self,
        task: TaskSpec,
        raw_answer: str,
        *,
        raw_summary: str = "",
        allow_model: bool = True,
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
            allow_model=allow_model,
        )

    def solve(self, task: TaskSpec, *, run_id: str) -> AdaptiveSolverResult:
        task.metadata["judge_evaluation_scope"] = f"primary:{run_id}:{self.runtime.seed}"

        skill_manifest = {}
        if self.skillbank and hasattr(self.skillbank, "select_context"):
            scope_kwargs = (
                {"task_metadata": task.metadata}
                if getattr(self.skillbank, "supports_task_metadata", False)
                else {}
            )
            selected_skills, skill_context, skill_manifest = self.skillbank.select_context(
                task.prompt,
                task_type=task.task_type,
                tokenizer=self.director_tokenizer,
                tools=getattr(self.runtime.executor, "tools", {}).keys(),
                **scope_kwargs,
            )
        else:
            selected_skills = (
                self.skillbank.retrieve(task.prompt, task_type=task.task_type)
                if self.skillbank
                else []
            )
            skill_context = SolverSkillBank.format_prompt_context(selected_skills)
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
            task=solver_task_text(
                task, include_submission_contract=self.answer_finalizer is not None
            ),
            director_task=task.prompt,
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
        self.active_canvas = canvas
        try:
            run = GraphDirector(
                backend=self.director_backend,
                canvas=canvas,
                solver_skill_context=skill_context,
                prompt_variant=self.director_prompt_variant,
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
        if self.post_director_hook is not None:
            self.post_director_hook(task, canvas, run)
        output_artifact = (
            self.runtime.artifacts.get(canvas.graph.output_agent)
            if canvas.graph.output_agent
            else None
        )
        if (
            run.finished
            and action_adapter is not None
            and action_adapter.adapter_id == "aime"
            and output_artifact is not None
            and "terminal_tool_failure" in artifact_integrity_failure_risks(output_artifact)
            and not _is_model_attributed_terminal_tool_failure(output_artifact)
        ):
            before_artifact_id = output_artifact.artifact_id
            before_risks = artifact_integrity_failure_risks(output_artifact)
            recovery_step = canvas.recover_selected_output_agent(
                reason_code="aime_terminal_tool_failure"
            )
            output_artifact = self.runtime.artifacts.get(canvas.graph.output_agent)
            after_risks = (
                artifact_integrity_failure_risks(output_artifact)
                if output_artifact is not None
                else ["missing_output_artifact"]
            )
            task.metadata["selected_output_recovery"] = {
                "attempted": True,
                "dataset": "aime",
                "scope": "selected_output_agent",
                "reason": "terminal_tool_failure",
                "output_agent": canvas.graph.output_agent,
                "before_artifact_id": before_artifact_id,
                "after_artifact_id": (
                    output_artifact.artifact_id if output_artifact is not None else None
                ),
                "before_integrity_risks": before_risks,
                "after_integrity_risks": after_risks,
                "recovered": not after_risks,
                "worker_model_calls": (
                    recovery_step.execution.worker_model_calls_total
                    if recovery_step.execution is not None
                    else 0
                ),
            }
            if output_artifact is not None:
                run.output = output_artifact.answer
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
        if action_adapter is not None and action_adapter.adapter_id == "webshop":
            output_agent = canvas.graph.output_agent
            environment_result = active_webshop_lifecycles[0].result_for(output_agent)
            task.metadata["webshop_environment_result"] = environment_result
            for lifecycle in active_webshop_lifecycles:
                lifecycle.close_all()
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            environment_result = (
                dict(output_artifact.environment_result)
                if output_artifact and output_artifact.environment_result
                else active_alfworld_lifecycles[0].result_for(canvas.graph.output_agent)
            )
            if not canvas.graph.output_agent:
                completed_artifacts = [
                    artifact
                    for artifact in self.runtime.artifacts.values()
                    if artifact.environment_result
                ]
                if len(completed_artifacts) == 1:
                    sole = completed_artifacts[0]
                    if (
                        sole.environment_result.get("environment_completed") is True
                        and sole.environment_result.get("won") is True
                    ):
                        # Preserve a unique already-won episode, without setting
                        # output or selecting the best of multiple Agent runs.
                        environment_result = dict(sole.environment_result)
                        task.metadata["environment_result_preservation"] = {
                            "source": "sole_completed_episode",
                            "agent_id": sole.agent_id,
                            "graph_mutated": False,
                        }
            task.metadata["alfworld_environment_result"] = environment_result
            for lifecycle in active_alfworld_lifecycles:
                lifecycle.close_all()
        if action_adapter is not None and action_adapter.adapter_id == "swe_bench":
            output_agent = canvas.graph.output_agent
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
        answer_submission = self.finalize_answer(
            task,
            run.output,
            raw_summary=output_artifact.summary if output_artifact else "",
            allow_model=not backend_failures,
        )
        if run.output.strip() in {
            WORKER_PROTOCOL_FAILURE_SENTINEL,
            WORKER_BACKEND_FAILURE_SENTINEL,
        }:
            answer_submission = replace(
                answer_submission,
                submitted_answer="",
                valid=False,
                detail="runtime_failure_sentinel_not_an_answer",
            )
        task.metadata["answer_submission"] = answer_submission.to_dict()
        task.metadata["qa_token_f1"] = qa_token_f1(task, answer_submission.submitted_answer)
        task.metadata["qa_official_metrics"] = qa_official_metrics(
            task.metadata.get("dataset"),
            answer_submission.submitted_answer,
            task.reference,
        )
        terminal_failure = terminal_policy_failure(
            str(task.metadata.get("dataset", "")),
            terminal=not canvas.active,
            rejection_codes=[step.rejection_code for step in canvas.history if step.rejection_code],
            artifacts=artifact_integrity,
            output_agent=canvas.graph.output_agent,
            rounds=canvas.round_index,
            max_rounds=canvas.config.max_rounds,
            worker_tokens=canvas.total_tokens,
            worker_token_limit=canvas.config.max_total_tokens,
            infrastructure_failure=bool(
                backend_failures or task.metadata.get("swe_infrastructure_failure")
            ),
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
        if backend_failures:
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
