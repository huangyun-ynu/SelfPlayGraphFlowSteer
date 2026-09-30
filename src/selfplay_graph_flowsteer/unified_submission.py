"""Shared result assessment and zero-Worker submission for every task adapter."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
from contextlib import contextmanager
import fcntl

from .actions import ActionType, UNIFIED_ACTION_FIELDS
from .submission_contract import DirectorCallContext, _issue_receipt, answer_hash, snapshot_hash
from .unified_contract import ACTION_PROTOCOL, PROTOCOL, SUBMISSION_VERSION, is_task_result


class SubmissionJournal:
    """Durable intent before any externally visible commit. Unknown is never retried blindly."""

    def __init__(self, directory: str, run_id: str):
        self.path = Path(directory) / (snapshot_hash(run_id) + ".json")

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else None

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class UnifiedSubmissionMixin:
    _DIRECTOR_EDIT_ACTIONS = frozenset({
        ActionType.ADD_AGENT, ActionType.SET_PROMPT, ActionType.SET_MODEL,
        ActionType.SET_LAYER, ActionType.SET_RELATION, ActionType.REMOVE_RELATION,
        ActionType.DELETE_AGENT,
    })

    @property
    def unified(self):
        return self.config.submission_protocol == PROTOCOL

    def _init_unified(self):
        self.director_edits_used = 0
        self._submission_lock = threading.RLock()
        self._unified_recovery_used = 0
        self._unified_transaction = None
        self._unified_observed_snapshots = {}
        self._unified_seen_states = set()
        self._unified_no_progress = 0
        self._unified_previous_submit_ready = False
        self._unified_submission_recoveries = set()

    def director_edit_budget(self):
        limit = self.config.director_edit_limit(self.dataset)
        return {"used": self.director_edits_used, "max": limit,
                "remaining": None if limit is None else max(0, limit - self.director_edits_used),
                "counting": "successful_graph_mutations_including_initial_construction",
                "reset_on_node_edit": False}

    def _webshop_schedule_control(self, assessments=None):
        if self.dataset != "webshop" or not self.runtime.webshop_scheduling_enabled:
            return None
        total = self.action_adapter.total_action_budget
        status = self.runtime.webshop_scheduling_status(total)
        assessments = assessments if assessments is not None else {
            key:self.submission_assessment(key) for key in self.graph.nodes}
        ready = [key for key,a in assessments.items() if a["submit_ready"]
                 and (a["payload_kind"] == "purchase" or status["total_remaining"] == 0)]
        result = {**status, "mode":"finish" if ready else "exhausted" if status["total_remaining"] == 0 else "working",
                  "finish_targets":ready,"promotion_targets":[],"cleanup_targets":[],
                  "run_blockers":{key:self.runtime.webshop_scheduling_blocker(node,self.graph)
                                  for key,node in self.graph.nodes.items() if node.configured}}
        if result["mode"] == "exhausted":
            known = []
            for key,node in self.graph.nodes.items():
                resource = self.runtime.environment_result_for(key)
                if (resource and resource.get("resource_status") != "unknown"
                        and resource.get("termination_reason") not in {"agent_never_executed","missing_output_agent","environment_step_failed"}):
                    known.append(key)
                    if node.configured and not is_task_result(node) and not resource.get("commit_pending"):
                        result["promotion_targets"].append(key)
            staged = set(self.runtime.environment_commit_ready_agents())
            result["cleanup_targets"] = [key for key in self.graph.nodes
                if key not in staged and any(other != key for other in known)]
        elif result["mode"] == "working" and not self.pending_relation_decision:
            # Completing a purchase/failure on one node must not trigger more
            # shopping merely because an unrelated research node is still present.
            terminal = [key for key,a in assessments.items()
                if a["blockers"] and all(b.startswith("graph:agents cannot influence output:") for b in a["blockers"])
                and (key in self.runtime.environment_commit_ready_agents() or a["payload_kind"] == "environment_failure")]
            if len(terminal) == 1:
                target = terminal[0]
                cleanup = [key for key in self.graph.nodes if key != target
                           and target not in self.graph.reachable_from(key)]
                if cleanup:
                    result.update(mode="cleanup",cleanup_targets=cleanup,completion_target=target)
            if (result["mode"] == "working" and status["research_remaining"] == 0
                    and not any(node.configured and is_task_result(node) for node in self.graph.nodes.values())):
                for key,node in self.graph.nodes.items():
                    resource = self.runtime.environment_result_for(key)
                    if (node.configured and resource and resource.get("resource_status") != "unknown"
                            and resource.get("termination_reason") not in {"agent_never_executed","missing_output_agent","environment_step_failed"}
                            and not resource.get("commit_pending")):
                        result["promotion_targets"].append(key)
                if result["promotion_targets"]:
                    result["mode"] = "handoff"
        return result

    def _director_edit_admission(self, action):
        scheduling = self._webshop_schedule_control()
        if scheduling:
            reason = None
            if scheduling["mode"] == "finish" and not (
                    action.action_type is ActionType.FINISH and action.target in scheduling["finish_targets"]):
                reason = "Submit an existing current result with FINISH; no further Worker execution is necessary."
            elif scheduling["mode"] == "handoff" and not (
                    action.action_type is ActionType.SET_PROMPT and action.target in scheduling["promotion_targets"]
                    and action.result_scope == "task_result"):
                reason = "The shared research allowance is spent. Explicitly promote one listed existing session owner to task_result; preserve its page and evidence instead of starting another session."
            elif scheduling["mode"] == "cleanup" and not (
                    action.action_type is ActionType.DELETE_AGENT and action.target in scheduling["cleanup_targets"]):
                reason = "A current completion is blocked only by unrelated graph nodes. Delete a listed unrelated node, then FINISH the existing result without rerunning its Worker."
            elif scheduling["mode"] == "exhausted" and not (
                    action.action_type is ActionType.FINISH
                    or (action.action_type is ActionType.DELETE_AGENT and action.target in scheduling["cleanup_targets"])
                    or (action.action_type is ActionType.SET_PROMPT and action.target in scheduling["promotion_targets"]
                        and action.result_scope == "task_result")):
                reason = "No environment actions remain. Delete listed unrelated nodes or promote an existing owner to task_result for a truthful zero-Worker failure receipt."
            elif action.action_type is ActionType.ADD_AGENT and scheduling["total_remaining"] < 3:
                reason = "A fresh session needs at least search, open, and Buy. Continue an existing session within the remaining budget."
            if reason:
                return self._reject_graph_action(action, code="webshop_scheduling_boundary",
                    message=reason,rejection_details={"webshop_scheduling":scheduling})
        if self.dataset == "webshop" and action.action_type is ActionType.DELETE_AGENT:
            node = self.graph.nodes.get(str(action.target))
            budget = self.runtime.shared_tool_budget_status(node.total_tool_budget if node else 0)
            reserve = budget.get("purchase_reservation", {})
            if reserve.get("owner") == action.target and reserve.get("reserved"):
                return self._reject_graph_action(action, code="purchase_budget_reserved",
                    message="Resume or explicitly abandon the protected owner before deleting its live purchase path.",
                    rejection_details={"purchase_reservation": reserve})
        remaining = self.director_edit_budget()["remaining"]
        if remaining is None:
            return None
        required = (3 if self.runtime_routes else 2) if action.action_type is ActionType.ADD_AGENT else 1
        if action.action_type not in self._DIRECTOR_EDIT_ACTIONS | {ActionType.CONSIDER_RELATION}:
            return None
        if remaining < required:
            return self._reject_graph_action(action, code="director_edit_budget_exhausted",
                message=(f"Only {remaining} Director edits remain; this action requires {required} "
                         "including required node configuration. Existing eligible results can still be submitted."),
                rejection_details={"director_edit_budget": self.director_edit_budget()})
        return None

    def _record_unified_progress(self, *, accepted):
        signature = self.director_progress_signature()
        repeated = signature in self._unified_seen_states
        ready = any(self.submission_assessment(target)["submit_ready"] for target in self.graph.nodes)
        # Deleting an isolated node can restore a previously seen, now valid
        # candidate. Give the Director a bounded chance to submit that repair.
        # The same semantic state receives this allowance only once, so repeated
        # add/delete cycles still exhaust the ordinary no-progress limit.
        if (accepted and repeated and ready and not self._unified_previous_submit_ready
                and signature not in self._unified_submission_recoveries):
            self._unified_submission_recoveries.add(signature)
            repeated = False
        self._unified_previous_submit_ready = ready
        self._unified_no_progress = self._unified_no_progress + 1 if repeated else 0
        self._unified_seen_states.add(signature)

    def _restore_submission_lock(self):
        if self.run_id and self._unified_transaction is None:
            existing = SubmissionJournal(self.config.submission_journal_dir, self.run_id).read()
            if existing:
                from .canvas import CanvasState
                self._unified_transaction = existing
                self.state = CanvasState.FAILED
                return True
        return False

    def _unified_progress_signature(self):
        graph = self.graph.to_dict()
        graph.pop("version", None)
        for node in graph.get("nodes", {}).values() if isinstance(graph.get("nodes"), dict) else graph.get("nodes", []):
            if isinstance(node, dict):
                node["metadata"] = {k: v for k, v in node.get("metadata", {}).items() if not k.startswith("_runtime_")}
        artifacts = {key: {"answer": value.answer, "summary": value.summary,
                    "issues": value.unresolved_issues, "evidence": value.evidence,
                    "code": value.code_artifact_ref.to_dict() if value.code_artifact_ref else None,
                    "resource": self.runtime.resource_signature(key)}
            for key, value in self.runtime.artifacts.items() if key in self.graph.nodes}
        return snapshot_hash({"graph": graph, "state": self.state.value,
                              "pending": self.pending_agent_id,
                              "dirty": sorted(self.dirty_agents), "results": artifacts})

    def observe_submission_candidates(self, call_id):
        self._unified_observed_snapshots[call_id] = {
            target: [self.graph.version, artifact.artifact_id,
                     self.runtime.artifact_input_binding(target).get("input_hash")]
            for target, artifact in self.runtime.artifacts.items()
            if target in self.graph.nodes
        }

    def _isolated_submission_draft(self, target):
        """An unexecuted, isolated pending node is not part of a result's dependencies."""
        pending = self.pending_agent_id
        node = self.graph.nodes.get(pending)
        return bool(pending and pending != target and node and not node.configured
            and pending not in self.runtime.artifacts
            and not any(pending in edge for edge in
                        self.graph.directed_edges | self.graph.bidirectional_edges))

    def _submission_graph(self, target):
        graph = self.graph.clone()
        if self._isolated_submission_draft(target):
            graph.delete_agent(self.pending_agent_id)
        graph.output_agent = target
        return graph

    def submission_assessment(self, target, *, include_payload=False):
        from .runtime import artifact_integrity_failure_risks

        if self.submission_receipt is not None and self.submission_receipt.output_agent_id == target:
            result = copy.deepcopy(self._unified_transaction["assessment"])
            result.update(submitted=True, submit_ready=False, blockers=[])
            if not include_payload:
                result["payload_hash"] = snapshot_hash(result.pop("payload"))
            return result

        node = self.graph.nodes.get(target)
        artifact = self.runtime.artifacts.get(target)
        blockers = []
        kind = "text"
        payload = {}
        if node is None:
            blockers.append("unknown_agent")
        elif not node.configured:
            blockers.append("configuration_incomplete")
        elif not is_task_result(node):
            blockers.append("local_result_only")
        if artifact is None:
            blockers.append("missing_artifact")
        else:
            if target in self.dirty_agents or not self.runtime.artifact_matches_current_input_signature(
                target, task=self.worker_task, graph=self.graph
            ):
                blockers.append("stale_artifact")
            risks = artifact_integrity_failure_risks(artifact)
            if self.dataset == "webshop":
                kind = "purchase"
                ready = target in self.runtime.environment_commit_ready_agents()
                payload = self.runtime.environment_result_for(target)
                if payload.get("resource_status") == "unknown":
                    blockers.append("environment_state_unknown")
                if not ready:
                    # Worker completion is not environment completion. Only runtime
                    # terminal evidence or the actual shared action ledger admits failure.
                    from .runtime import ActionBudgetLedger
                    exhausted = bool(node and ActionBudgetLedger.shared_total(node)
                        and node.allowed_tools and self.runtime.shared_tool_budget_status(
                            node.total_tool_budget)["remaining"] <= 0)
                    terminal = (payload.get("environment_completed") is True
                                or payload.get("done") is True)
                    policy_failed = (artifact.webshop_progress.get("trusted") is True
                        and artifact.webshop_progress.get("state") == "typed_policy_failure"
                        and artifact.webshop_progress.get("policy_failure", {}).get("runtime_terminal") is True)
                    if payload and (terminal or exhausted or policy_failed):
                        kind = "environment_failure"
                    else:
                        blockers.append("purchase_not_prepared")
            elif self.dataset == "alfworld":
                kind = "episode"
                payload = self.runtime.environment_result_for(target)
                if not payload or payload.get("environment_completed") is not True:
                    blockers.append("environment_result_unavailable")
            elif self.dataset == "swe_bench":
                kind = "code_patch"
                payload = artifact.code_artifact_ref.to_dict() if artifact.code_artifact_ref else {}
                progress = artifact.swe_progress
                if not (progress.get("trusted") is True and progress.get("commit_ready") is True
                        and ((payload and progress.get("test_after_latest_edit") is True)
                             or progress.get("grounded_failure") or progress.get("policy_failure"))):
                    blockers.append("patch_or_post_edit_test_required")
                elif not payload:
                    kind = "policy_failure"
                    payload = copy.deepcopy(artifact.swe_progress)
            else:
                if not str(artifact.answer).strip() or str(artifact.answer) in {"WORKER_BACKEND_FAILURE", "WORKER_PROTOCOL_FAILURE"}:
                    blockers.append("answer_missing")
                payload = {"answer": artifact.answer}
                if self.dataset == "nq_open" and self.runtime.nq_evidence_context is not None:
                    # Validate this target without changing the output or
                    # invoking a Worker. Other candidates cannot lend citations.
                    evidence = self.runtime.nq_evidence_context.validate(target, artifact.raw_response)
                    if not evidence.get("valid"):
                        blockers.append("nq_evidence:" + str(evidence.get("reason", "invalid_submission")))
                    payload["nq_corpus_submission"] = evidence
            trusted_environment = (
                (self.dataset == "webshop" and target in self.runtime.environment_commit_ready_agents())
                or (self.dataset == "alfworld" and payload.get("environment_completed") is True)
                or (self.dataset == "swe_bench" and "patch_or_post_edit_test_required" not in blockers)
            )
            if risks and not trusted_environment:
                blockers.extend("integrity:" + risk for risk in risks)
        if node is not None:
            candidate = self._submission_graph(target)
            blockers.extend("graph:" + error for error in candidate.validate(final=True))
        if ((self.pending_agent_id and not self._isolated_submission_draft(target))
                or self.pending_relation_decision):
            blockers.append("pending_configuration")
        if (self.structural_exploration_required and not self.structural_exploration_waived
                and not self._structural_exploration_satisfied()):
            blockers.append("structural_exploration_required")
        if self.runtime.worker_usage_ledger is None and self.total_tokens > self.config.max_total_tokens:
            blockers.append("worker_budget_exceeded")
        binding = self.runtime.artifact_input_binding(target) if artifact is not None else {}
        return {
            "target": target,
            "result_scope": node.metadata.get("result_scope") if node else None,
            "artifact_id": artifact.artifact_id if artifact else None,
            "input_signature": binding.get("input_hash"),
            "resource_signature": binding.get("resource_signature"),
            "agent_incarnation": node.metadata.get("incarnation_id") if node else None,
            "payload_kind": kind,
            **({"payload": payload} if include_payload else {"payload_hash": snapshot_hash(payload)}),
            "submit_ready": not blockers,
            "blockers": list(dict.fromkeys(blockers)),
        }

    def _webshop_shared_actions_exhausted(self):
        adapter = self.action_adapter
        return (self.dataset == "webshop" and adapter is not None
                and adapter.action_budget_policy == "shared_total_v1"
                and self.runtime.shared_tool_budget_status(adapter.total_action_budget)["remaining"] <= 0)

    def _unified_can_run(self, target, *, admitted_action=False):
        from .canvas import CanvasState
        if (self.state in {CanvasState.FINISHED, CanvasState.FAILED}
                or (not admitted_action and not self.active)):
            return False
        if (self.runtime.worker_usage_ledger is None
                and self.total_tokens >= self.config.max_total_tokens):
            return False
        node = self.graph.nodes.get(target)
        if node is None or not node.configured:
            return False
        if self.runtime.webshop_scheduling_blocker(node,self.graph) is not None:
            return False
        if (self.dataset == "webshop" and self.runtime.webshop_scheduling_enabled
                and target in self.runtime.environment_commit_ready_agents()):
            return False
        if self.runtime.webshop_reservation_blocker(node) is not None:
            return False
        if self.runtime.swe_execution_blocker(node) is not None:
            return False
        artifact = self.runtime.artifacts.get(target)
        if self.dataset in {"alfworld", "webshop"}:
            resource = self.runtime.environment_result_for(target)
            if resource.get("resource_status") == "unknown" or resource.get("termination_reason") == "environment_step_failed":
                return False
            if self.runtime.environment_continuation_status(target).get("can_continue") is False:
                return False
        if target in self.dirty_agents or artifact is None:
            return True
        from .runtime import artifact_integrity_failure_risks
        if (artifact_integrity_failure_risks(artifact)
                or artifact.webshop_progress.get("state") == "needs_recovery"):
            return self._unified_recovery_used < self.config.max_recovery_executions
        from .runtime import ActionBudgetLedger
        if (ActionBudgetLedger.shared_total(node)
                and node.allowed_tools
                and self.runtime.shared_tool_budget_status(node.total_tool_budget)["remaining"] <= 0):
            return False
        if self.dataset == "swe_bench":
            return is_task_result(node) and not (artifact.swe_progress.get("trusted") and artifact.swe_progress.get("commit_ready"))
        if self.dataset in {"alfworld", "webshop"}:
            result = self.runtime.environment_result_for(target)
            remaining = result.get("remaining_rollout_env_steps", result.get("remaining_env_steps", result.get("remaining_steps", 1)))
            budget = artifact.webshop_progress.get("action_budget", {})
            return (bool(result) and remaining > 0 and budget.get("total_remaining", 1) > 0
                    and result.get("termination_reason") != "environment_step_failed"
                    and not any(result.get(key) for key in ("done", "terminal", "purchased", "commit_ready")))
        return False

    def _unified_run(self, action):
        from .canvas import CanvasState
        # step() has admitted and charged this action's Director round already.
        # Keep every Worker/resource check, without demanding another round.
        if self.state is not CanvasState.BUILDING or not self._unified_can_run(str(action.target), admitted_action=True):
            return self._reject_graph_action(action, code="execution_not_admissible", message="No pending work or bounded recovery is available for this target.")
        target = str(action.target)
        from .runtime import artifact_integrity_failure_risks
        artifact = self.runtime.artifacts.get(target)
        if artifact and (artifact_integrity_failure_risks(artifact)
                or artifact.webshop_progress.get("state") == "needs_recovery"):
            self._unified_recovery_used += 1
        self.graph.nodes[target].metadata["_runtime_explicit_recovery"] = True
        self.dirty_agents.add(target)
        self.dirty_reasons.setdefault(target, set()).add("explicit_continuation")
        try:
            report = self._execute_dirty()
        except Exception as exc:
            from .canvas import RequiredWorkerBackendFailure, TokenBudgetExceeded
            if not isinstance(exc, (RequiredWorkerBackendFailure, TokenBudgetExceeded, ValueError)):
                raise
            self.state = CanvasState.FAILED
            return self._record(
                action, accepted=True, execution=getattr(exc, "report", None),
                feedback=f"Explicit execution stopped: {exc}",
                rejection_code=("worker_backend_unavailable" if isinstance(exc, RequiredWorkerBackendFailure)
                                else "execution_budget_exceeded" if isinstance(exc, TokenBudgetExceeded)
                                else "execution_failure"),
            )
        finally:
            self.graph.nodes[target].metadata.pop("_runtime_explicit_recovery", None)
        if report is not None and target in report.blocked_agents:
            return self._record(action, accepted=True, execution=report,
                feedback=self._feedback(
                    f"Continuation for {target} was blocked before its Worker request; pending work remains.",
                    report))
        return self._record(action, accepted=True, feedback="Executed the bounded continuation on current inputs and resources.", execution=report)

    def _unified_finish(self, action):
        from .answer_submission import AnswerFinalizer, AnswerSubmissionConfig
        from .canvas import CanvasState
        from .observability import TaskSpec

        journal = SubmissionJournal(self.config.submission_journal_dir, self.run_id)
        with self._submission_lock, journal.locked():
            context = self._director_call_context
            if not isinstance(context, DirectorCallContext) or not context.runtime_owned or context.run_id != self.run_id:
                return self._reject_graph_action(action, code="director_finish_source_required", message="A real Director FINISH call is required for every dataset.")
            assessment = self.submission_assessment(str(action.target), include_payload=True)
            if not assessment["submit_ready"]:
                return self._reject_graph_action(action, code="submission_not_ready", message="; ".join(assessment["blockers"]), rejection_details={"result_assessment": self.submission_assessment(str(action.target))})
            observed = self._unified_observed_snapshots.get(context.call_id)
            current_ref = [self.graph.version, assessment["artifact_id"], assessment["input_signature"]]
            if observed is not None and observed.get(str(action.target)) != current_ref:
                return self._reject_graph_action(action, code="stale_submission_observation", message="The candidate changed after the Director observed it.")
            artifact = self.runtime.artifacts[str(action.target)]
            binding = self.runtime.artifact_input_binding(str(action.target))
            if not binding.get("input_hash"):
                return self._reject_graph_action(action, code="submission_input_binding_missing", message="The candidate lacks runtime input provenance.")
            graph = self._submission_graph(str(action.target))
            transaction_id = snapshot_hash([self.run_id, action.target, artifact.artifact_id, binding["input_hash"]])
            existing = journal.read()
            if existing:
                self._unified_transaction = existing
                self.state = CanvasState.FAILED
                return self._reject_graph_action(action, code="submission_transaction_exists", message="A persisted submission exists; reconcile that transaction before any new operation.")
            transaction = {
                "version": SUBMISSION_VERSION, "transaction_id": transaction_id,
                "run_id": self.run_id, "director_call_id": context.call_id,
                "target": action.target, "assessment": assessment,
                "graph": graph.to_dict(), "state": "prepared",
            }
            journal.write(transaction)
            self._unified_transaction = transaction
            try:
                if assessment["payload_kind"] == "purchase":
                    transaction["state"] = "committing"
                    journal.write(transaction)
                    result = self.runtime.commit_environment_output(str(action.target))
                    if not result.get("purchased"):
                        raise RuntimeError("commit returned no confirmed purchase")
                    assessment["payload"] = copy.deepcopy(result)
                    artifact = self.runtime.artifacts[str(action.target)]
                raw_answer = str(artifact.answer or "")
                if assessment["payload_kind"] == "text":
                    submission = AnswerFinalizer(AnswerSubmissionConfig(enabled=True)).finalize(
                        TaskSpec(self.run_id, self.worker_task, metadata={"dataset": self.dataset}), raw_answer
                    )
                    submitted_answer = submission.submitted_answer
                    normalization = submission.method
                    nq_check = assessment["payload"].get("nq_corpus_submission", {})
                    if nq_check.get("status") == "insufficient_evidence":
                        submitted_answer = "insufficient_evidence"
                        normalization = "nq_insufficient_evidence_v1"
                else:
                    submitted_answer = json.dumps(assessment["payload"], ensure_ascii=False, sort_keys=True)
                    normalization = "runtime_result_payload_v1"
                self.graph = graph
                if self.pending_agent_id not in graph.nodes:
                    self.dirty_agents.discard(self.pending_agent_id)
                    self.dirty_reasons.pop(self.pending_agent_id, None)
                    self.pending_agent_id = None
                usage_ledger = self.runtime.worker_usage_ledger
                usage_status = usage_ledger.status() if usage_ledger is not None else None
                self.submission_receipt = _issue_receipt(
                    context=context, version=SUBMISSION_VERSION, dataset=self.dataset,
                    accepted_event_id=f"{self.run_id}:canvas:{len(self.history)}",
                    output_agent_id=str(action.target), graph_snapshot_hash=snapshot_hash(graph.to_dict()),
                    artifact_id=artifact.artifact_id, input_signature=binding["input_hash"],
                    execution_generation=int(binding.get("generation", 0)),
                    raw_answer_snapshot=raw_answer, submitted_answer_snapshot=submitted_answer,
                    answer_hash=answer_hash(submitted_answer), normalization_version=normalization,
                    worker_tokens_used=self.total_tokens, worker_token_limit=self.config.max_total_tokens,
                    worker_budget_policy=usage_status["policy"] if usage_status else "strict_limit_v1",
                    worker_usage_complete=usage_status["usage_complete"] if usage_status else True,
                    worker_usage_ledger_digest=usage_ledger.digest() if usage_ledger else "",
                    worker_dispatch_valid=usage_ledger.dispatches_valid() if usage_ledger else False,
                    worker_unsettled_attempts=usage_status["unsettled_attempt_count"] if usage_status else 0,
                    payload_kind=assessment["payload_kind"], payload=copy.deepcopy(assessment["payload"]),
                    transaction_id=transaction_id,
                )
                transaction.update(state="committed", receipt=self.submission_receipt.to_dict())
                journal.write(transaction)
                self.state = CanvasState.FINISHED
                return self._record(action, accepted=True, feedback="Submitted the selected current task result without a Worker call.", final_execution=True)
            except Exception as exc:
                transaction.update(state="commit_unknown", error_type=type(exc).__name__, error=str(exc)[:500])
                try:
                    journal.write(transaction)
                except OSError:
                    # The last durable PREPARED/COMMITTING record still prevents
                    # replay. Preserve factual confirmation in the in-memory audit.
                    transaction["persistence_failed"] = True
                self.submission_receipt = None
                self.state = CanvasState.FAILED
                return self._record(action, accepted=False, feedback="Submission status is uncertain; the persisted transaction is locked and must not be replayed with another target.", rejection_code="submission_commit_unknown", final_execution=True)

    def _protected_alfworld_candidates(self):
        if (self.dataset != "alfworld"
                or self.config.alfworld_terminal_candidate_policy != "finish_only_v1"):
            return []
        candidates = []
        for target in sorted(self.graph.nodes):
            assessment = self.submission_assessment(target, include_payload=True)
            payload = assessment.get("payload", {})
            if assessment["submit_ready"] and payload.get("done") is True and payload.get("won") is True:
                candidates.append(target)
        return candidates

    def _alfworld_budget_cleanup_targets(self):
        """Preview deletions that retain a current win without scheduling work.

        Only current artifacts are considered. A deletion must not invalidate
        any retained node, or leave pending execution behind. No old graph or
        candidate is restored, and the Director chooses each deletion.
        """
        ledger = self.runtime.worker_usage_ledger
        if (self.dataset != "alfworld" or ledger is None or not ledger.stop_reason()
                or ledger.status()["inflight_request_count"]
                or self._unified_transaction is not None or self.pending_relation_decision):
            return []
        winners = []
        for target in sorted(self.graph.nodes):
            assessment = self.submission_assessment(target, include_payload=True)
            payload = assessment.get("payload", {})
            if (payload.get("done") is True and payload.get("won") is True
                    and assessment["blockers"]
                    and all(reason.startswith("graph:agents cannot influence output:")
                            for reason in assessment["blockers"])):
                winners.append(target)
        cleanup = []
        for target in sorted(self.graph.nodes):
            # Pending configuration permits deleting only that pending node.
            if self.pending_agent_id and target != self.pending_agent_id:
                continue
            retained_winners = [winner for winner in winners if winner != target
                                and winner not in self.graph.reachable_from(target)]
            if not retained_winners:
                continue
            candidate = self.graph.clone()
            mutation = candidate.delete_agent(target)
            if candidate.validate(final=False) or mutation.dirty_agents:
                continue
            if any(not node.configured or agent in self.dirty_agents
                   or not self.runtime.artifact_matches_current_input_signature(
                       agent, task=self.worker_task, graph=candidate, allow_failed=True)
                   for agent, node in candidate.nodes.items()):
                continue
            cleanup.append(target)
        return cleanup

    def _unified_control_snapshot(self):
        from .canvas import CanvasState
        self._restore_submission_lock()
        parameters = self._legal_action_parameters()
        parameters.pop("set_output", None)
        assessments = {key: self.submission_assessment(key) for key in sorted(self.graph.nodes)}
        ready = [key for key, value in assessments.items() if value["submit_ready"]]
        executable = [key for key in sorted(self.graph.nodes) if self._unified_can_run(key)]
        ids = sorted(self.graph.nodes)
        parameters["finish"] = {"targets": ready, "ready": bool(ready),
            "discards_isolated_unconfigured_draft": self.pending_agent_id if any(
                self._isolated_submission_draft(key) for key in ready) else None}
        parameters["run_agent"] = {"targets": executable}
        parameters["delete_agent"]["targets"] = (
            [self.pending_agent_id] if self.pending_agent_id and self.state in
            {CanvasState.AWAITING_PROMPT, CanvasState.AWAITING_MODEL} else ids)
        parameters["set_layer"]["targets"] = ids
        parameters["set_prompt"]["targets"] = [self.pending_agent_id] if self.pending_agent_id else ids
        parameters["set_prompt"]["result_scopes"] = ["subtask", "task_result"]
        relations = ["consider_relation"] if self.binary_relation_policy else ["set_relation", "remove_relation"]
        allowed = ["add_agent", "set_prompt", "set_model", "set_layer", *relations, "delete_agent", "run_agent", "finish"]
        if len(ids) >= self.config.max_agents or self.topology_edits_frozen or (
                self.runtime_routes and self.director_round_limit is not None
                and self.director_round_limit - self.round_index < 4):
            allowed.remove("add_agent")
        if self._webshop_shared_actions_exhausted() and "add_agent" in allowed:
            allowed.remove("add_agent")
        if not self.active:
            allowed = []
        elif self.state is CanvasState.AWAITING_PROMPT:
            allowed = ["set_prompt", "delete_agent"] + (["finish"] if ready else [])
        elif self.state is CanvasState.AWAITING_MODEL:
            allowed = ["set_model", "delete_agent"] + (["finish"] if ready else [])
        elif self.state is CanvasState.AWAITING_RELATION_CHOICE:
            allowed = ["relation_choice"]
        allowed = [name for name in allowed if name in {"add_agent", "relation_choice"} or bool(parameters.get(name, {}).get("targets", parameters.get(name, {}).get("relations")))]
        remaining_edits = self.director_edit_budget()["remaining"]
        if remaining_edits is not None:
            if remaining_edits == 0:
                allowed = [name for name in allowed if name not in
                           {action.value for action in self._DIRECTOR_EDIT_ACTIONS} | {"consider_relation"}]
            if remaining_edits < (3 if self.runtime_routes else 2) and "add_agent" in allowed:
                allowed.remove("add_agent")
            # Do not advertise parameter choices for edits the controller will reject.
            for name in {action.value for action in self._DIRECTOR_EDIT_ACTIONS} | {"consider_relation"}:
                if name not in allowed:
                    parameters[name] = {}
        usage_ledger = self.runtime.worker_usage_ledger
        usage_status = usage_ledger.status() if usage_ledger else None
        if usage_status is not None:
            usage_status["digest"] = usage_ledger.digest()
            if usage_ledger.stop_reason() is not None:
                cleanup = self._alfworld_budget_cleanup_targets()
                parameters["delete_agent"]["targets"] = cleanup
                allowed = [name for name in allowed if name == "finish"
                           or (name == "delete_agent" and cleanup)]
        protected = self._protected_alfworld_candidates()
        if protected:
            allowed = ["finish"] if "finish" in allowed else []
            parameters["finish"]["targets"] = protected
        scheduling = self._webshop_schedule_control(assessments)
        if scheduling and self.active:
            if scheduling["mode"] == "finish":
                allowed = ["finish"]
                parameters["finish"]["targets"] = scheduling["finish_targets"]
            elif scheduling["mode"] == "handoff":
                allowed = [name for name in allowed if name == "set_prompt"]
                parameters["set_prompt"]["targets"] = scheduling["promotion_targets"]
                parameters["set_prompt"]["result_scopes"] = ["task_result"]
            elif scheduling["mode"] == "cleanup":
                allowed = [name for name in allowed if name == "delete_agent"]
                parameters["delete_agent"]["targets"] = scheduling["cleanup_targets"]
            elif scheduling["mode"] == "exhausted":
                allowed = [name for name in allowed if name in {"finish", "delete_agent", "set_prompt"}]
                parameters["delete_agent"]["targets"] = scheduling["cleanup_targets"]
                parameters["set_prompt"]["targets"] = scheduling["promotion_targets"]
                parameters["set_prompt"]["result_scopes"] = ["task_result"]
                allowed = [name for name in allowed if parameters.get(name,{}).get("targets")]
            elif scheduling["total_remaining"] < 3 and "add_agent" in allowed:
                allowed.remove("add_agent")
        return {
            **({"webshop_scheduling":scheduling} if scheduling else {}),
            "canvas_version": self.graph.version, "director_action_protocol_version": ACTION_PROTOCOL,
            "submission_contract_version": SUBMISSION_VERSION, "submission_protocol": PROTOCOL,
            "submission_status": self._unified_transaction["state"] if self._unified_transaction else "working",
            "state": self.state.value, "legal_agent_ids": ids, "pending_agent_id": self.pending_agent_id,
            "pending_relation_decision": self.pending_relation_decision.to_dict() if self.pending_relation_decision else None,
            "allowed_actions": allowed,
            "worker_usage": usage_status,
            "terminal_candidate_protection": {
                "policy": self.config.alfworld_terminal_candidate_policy,
                "targets": protected,
                "instruction": "Submit an existing trusted terminal candidate with FINISH(target); no more Worker calls or graph edits." if protected else "",
            } if self.dataset == "alfworld" else None,
            "action_field_requirements": {name: list(UNIFIED_ACTION_FIELDS[name]) for name in allowed if name in UNIFIED_ACTION_FIELDS},
            "legal_action_parameters": parameters, "graph_state": self._graph_state_snapshot(),
            "result_assessments": assessments,
            **({"recoverable_code_artifacts": self.runtime.recoverable_swe_candidates(),
                "candidate_recovery_instruction": "A Worker may apply a relevant archived patch with swe_apply_artifact, validate current dependencies, and run swe_test. Archived patches alone cannot be submitted."}
               if self.dataset == "swe_bench" else {}),
            "agent_budget": {"used": len(ids), "max": self.config.max_agents, "remaining": self.config.max_agents - len(ids)},
            "progress": {"no_progress_count": self._unified_no_progress,
                         "semantic_states_seen": len(self._unified_seen_states),
                         "submission_recovery_count": len(self._unified_submission_recoveries)},
            "recovery_budget": {"used": self._unified_recovery_used, "max": self.config.max_recovery_executions},
            "director_edit_budget": self.director_edit_budget(),
            "action_budget": (self.runtime.shared_tool_budget_status(self.action_adapter.total_action_budget)
                              if self.action_adapter is not None
                              and self.action_adapter.action_budget_policy == "shared_total_v1" else None),
            **({"round_budget": {"used": self.round_index, "max": self.director_round_limit,
                                 "remaining": max(0, self.director_round_limit - self.round_index)}}
               if self.director_round_limit is not None else
               {"decision_statistics": {"turns": self.round_index, "limited": False}}),
            "token_budget": {"used": self.total_tokens, "max": self.config.max_total_tokens, "remaining": max(0, self.config.max_total_tokens - self.total_tokens)},
            "output_agent": self.graph.output_agent,
            "environment_commit_resolution": "FINISH(target) commits only that current result, without executing Workers. RUN_AGENT continues unfinished work within the same resource limits.",
        }
