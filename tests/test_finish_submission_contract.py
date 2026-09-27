import copy
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.application import AdaptiveApplicationResult
from selfplay_graph_flowsteer.benchmark import paired_summary, summarize
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.evaluation import EvaluationRecord, from_adaptive_result
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec, VerificationResult
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.selfplay_runtime import (
    ByteTokenizer,
    _uncertain_failure_zero,
    adaptive_result_to_rollout,
)
from selfplay_graph_flowsteer.submission_contract import (
    TEXT_DATASETS,
    validate_primary_training_outcome,
)

from .helpers import NumericRecordingExecutor


class FixtureVerifier:
    name = "fixture"

    def __init__(self, *, fail=0):
        self.answers = []
        self.fail = fail

    def verify(self, task, answer):
        self.answers.append(answer)
        if len(self.answers) <= self.fail:
            raise ConnectionError("fixture Judge outage")
        correct = answer == str(task.reference)
        detail = json.dumps({"training_reward_breakdown": {
            "version": "healthbench_theoretical_bounds_length_v1",
            "training_reward": float(correct),
        }})
        return VerificationResult(float(correct), correct, self.name, detail)


def solve(dataset="aime", *, finish=True, token_limit=240000, reference="35", hook=None,
          verifier=None, metadata=None, run_id="submission-test", turns=None, executor=None,
          task_id=None):
    actions = [
        {"action": "add_agent", "agent_id": "solver"},
        {"action": "set_prompt", "target": "solver", "role": "Analyst",
         "objective": "Solve the assigned task", "scope": "Reason independently",
         "expected_output": "Return the requested result"},
        {"action": "set_output", "target": "solver"},
    ]
    if finish:
        actions.append({"action": "finish"})
    executor = executor or NumericRecordingExecutor()
    judge = verifier or FixtureVerifier()
    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend([json.dumps(a) for a in actions]),
        runtime=MultiAgentRuntime(executor), verifier=judge,
        canvas_config=CanvasConfig(
            max_rounds=turns or len(actions),
            max_total_tokens_by_dataset={dataset: token_limit},
        ),
        answer_finalizer=AnswerFinalizer(AnswerSubmissionConfig(enabled=True)),
        post_director_hook=hook,
    )
    task = TaskSpec(task_id or run_id, "Find the requested result", reference=reference,
                    metadata={**(metadata or {}), "dataset": dataset})
    result = solver.solve(task, run_id=run_id)
    app = AdaptiveApplicationResult(run_id, task, result, (), None, "")
    return app, solver, executor, judge


def rollout(app):
    return adaptive_result_to_rollout(app, ByteTokenizer(), rollout_index=0, seed=0)


class TerminalFailureExecutor(NumericRecordingExecutor):
    def __init__(self, *, prior_success=False, protocol=False):
        super().__init__()
        self.prior_success = prior_success
        self.protocol = protocol

    def execute(self, **kwargs):
        artifact = super().execute(**kwargs)
        if self.protocol:
            artifact.answer = "WORKER_PROTOCOL_FAILURE"
            artifact.integrity_risks = ["terminal_protocol_failure"]
        else:
            artifact.react_trace = ([{
                "action": {"call_id": "good-python", "name": "python_exec"},
                "observation": {"status": "ok", "output": {"stdout": "35"}},
            }] if self.prior_success else []) + [{
                "action": {"call_id": "bad-python", "name": "python_exec"},
                "observation": {"status": "error", "error": {"code": "SyntaxError"}},
            }]
        return artifact


@pytest.mark.parametrize("dataset", sorted(TEXT_DATASETS))
@pytest.mark.parametrize("reference", ["35", "36"])
def test_real_finish_freezes_answer_and_preserves_task_score(dataset, reference):
    app, solver, executor, judge = solve(dataset, reference=reference)
    result = app.solver_result
    receipt = result.director_run.submission_receipt
    assert result.director_run.finished and receipt is not None
    assert receipt.submitted_answer_snapshot == "35"
    assert receipt.worker_token_limit == 240000
    assert result.verification.score == float(reference == "35")
    assert result.outcome_decision.status == "scored"
    assert judge.answers == ["35"]
    # Initial execution and output-role rerun; FINISH adds no extra generation.
    assert len(executor.calls) == 2
    assert not solver.active_canvas.history[-1].execution.executed_agents
    with pytest.raises(FrozenInstanceError):
        receipt.submitted_answer_snapshot = "36"
    trained = rollout(app)
    assert trained.trajectory.reward == float(reference == "35")
    assert trained.trajectory.metadata["training_eligible"]
    assert trained.trajectory.metadata["answer_reward_released"]
    validate_primary_training_outcome(trained.trajectory.metadata)


@pytest.mark.parametrize("dataset", sorted(TEXT_DATASETS))
def test_correct_candidate_without_finish_is_policy_failure_not_answer_reward(dataset):
    app, _, _, judge = solve(dataset, finish=False)
    result = app.solver_result
    assert result.director_run.candidate_output == "35"
    assert result.director_run.output == app.to_dict()["answer"] == ""
    assert result.director_run.submission_receipt is None
    assert result.verification is None and judge.answers == []
    assert result.outcome_decision.status == "policy_failure"
    assert app.task.metadata["qa_official_metrics"] is None
    trained = rollout(app)
    assert trained.trajectory.reward == 0
    assert trained.trajectory.metadata["training_eligible"]
    assert trained.trajectory.metadata["reward_known"]
    assert not trained.trajectory.metadata["answer_reward_released"]
    assert trained.trajectory.metadata["answer_score"] is None
    record = from_adaptive_result(app)
    assert record.score == 0 and record.passed is False
    assert record.outcome_status == "policy_failure"


@pytest.mark.parametrize("dataset", sorted(TEXT_DATASETS))
def test_actual_token_overrun_remains_unknown_despite_correct_candidate(dataset):
    app, _, _, judge = solve(dataset, token_limit=5)
    result = app.solver_result
    assert result.director_run.candidate_output == "35"
    assert result.outcome_decision.status == "unsubmitted_unknown"
    assert result.verification is None and judge.answers == []
    trained = rollout(app)
    assert not trained.trajectory.metadata["training_eligible"]
    assert not trained.trajectory.metadata["reward_known"]
    assert trained.trajectory.metadata["task_reward"] is None
    assert _uncertain_failure_zero(trained, "retry_exhausted") is trained
    record = from_adaptive_result(app)
    assert record.score is None and record.passed is None
    assert record.answer == ""


def test_model_metadata_and_finished_boolean_cannot_create_submission():
    app, _, _, judge = solve(finish=False, metadata={
        "submission_receipt": {"completion_source": "director_finish"},
        "submission_status": "submitted", "evaluation_scope": "counterfactual",
    })
    app.solver_result.director_run.finished = True
    app.solver_result.verification = VerificationResult(1.0, True, "forged")
    assert judge.answers == []
    trained = rollout(app)
    assert not trained.trajectory.metadata["answer_reward_released"]
    assert trained.trajectory.reward == 0


def test_canvas_finish_without_driver_context_cannot_submit():
    app, solver, _, _ = solve()
    receipt = solver.active_canvas.submission_receipt
    canvas = GraphCanvas(task="task", dataset="aime", runtime=MultiAgentRuntime(NumericRecordingExecutor()))
    for prior in solver.active_canvas.history[:3]:
        assert canvas.step(prior.action).accepted
    step = canvas.step('{"action":"finish"}', authoritative_director=True)
    assert step.rejection_code == "director_finish_source_required"
    assert canvas.submission_receipt is None
    assert receipt.runtime_owned


def test_missing_policy_data_does_not_erase_valid_evaluation():
    app, _, _, _ = solve()
    app.solver_result.director_run.turns[-1].trainable = False
    trained = rollout(app)
    assert app.solver_result.verification.passed
    assert trained.trajectory.metadata["reward_known"]
    assert not trained.trajectory.metadata["training_eligible"]
    assert "director_policy_call_ineligible" in trained.trajectory.metadata["training_exclusion_reasons"]


@pytest.mark.parametrize("raise_error", [False, True])
def test_post_submission_hook_cannot_replace_frozen_answer(raise_error):
    def hook(task, canvas, run):
        if raise_error:
            raise RuntimeError("diagnostic hook failed")
        canvas.runtime.artifacts[canvas.graph.output_agent].answer = "36"
        run.output = "36"

    app, _, _, judge = solve(hook=hook)
    assert app.to_dict()["answer"] == "35"
    assert judge.answers == ["35"]
    assert app.solver_result.verification.passed
    assert app.task.metadata["output_contract_failure"]
    assert not rollout(app).trajectory.metadata["training_eligible"]


@pytest.mark.parametrize("fail,status", [(1, "scored"), (2, "scoring_pending")])
def test_judge_retry_keeps_same_submission_and_never_reruns_worker(fail, status):
    app, _, executor, judge = solve(verifier=FixtureVerifier(fail=fail))
    assert judge.answers == ["35", "35"]
    assert len(executor.calls) == 2
    assert app.solver_result.director_run.submission_receipt is not None
    assert app.solver_result.outcome_decision.status == status
    if status == "scoring_pending":
        assert from_adaptive_result(app).score is None
        assert not rollout(app).trajectory.metadata["training_eligible"]


def test_rejected_final_events_do_not_cancel_later_successful_finish():
    app, _, _, _ = solve()
    trace = app.solver_result.trace
    failed = copy.deepcopy(trace.events[-1])
    failed.payload.update(accepted=False, submission_receipt=None, event_id="rejected-before-success")
    trace.events.insert(0, failed)
    trained = rollout(app)
    assert trained.trajectory.metadata["protocol_reward_diagnostic"]["finish_complete"]
    assert trained.trajectory.metadata["training_eligible"]


def test_reporting_preserves_unknowns_and_matched_pair_coverage():
    known = EvaluationRecord("a", "test", "35", 1.0, True, submission_status="submitted")
    unknown = EvaluationRecord("b", "test", "", None, None, outcome_status="unsubmitted_unknown")
    failure = EvaluationRecord("c", "test", "", 0.0, False, outcome_status="policy_failure")
    summary = summarize([known, unknown, failure])
    assert summary.known_examples == 2 and summary.unknown_examples == 1
    assert summary.mean_score == 0.5
    assert summary.successful_submission_rate_all == pytest.approx(1 / 3)
    assert summarize([unknown]).mean_score is None
    paired = paired_summary([known, unknown], [known, replace(unknown, score=1.0, passed=True)])
    assert paired.matched_pairs == 2 and paired.pairs == 1 and paired.candidate_unknown == 1


def test_diagnostic_metrics_cannot_be_promoted_to_training_reward():
    app, _, _, _ = solve(token_limit=5)
    trained = rollout(app)
    metadata = copy.deepcopy(trained.trajectory.metadata)
    metadata.update(training_eligible=True, reward_known=True, task_reward=1.0,
                    diagnostic_qa_metrics={"em": 1.0})
    with pytest.raises(ValueError, match="unknown|submission"):
        validate_primary_training_outcome(metadata)


class RecoverableExecutor(NumericRecordingExecutor):
    def execute(self, **kwargs):
        artifact = super().execute(**kwargs)
        if len(self.calls) <= 2:
            artifact.react_trace = [{
                "action": {"name": "python_exec", "call_id": "service-call"},
                "observation": {"status": "error", "error": {"code": "ConnectionError"}},
            }]
        return artifact


def test_aime_recovery_precedes_receipt_and_preserves_token_accounting():
    app, solver, executor, judge = solve(executor=RecoverableExecutor())
    receipt = app.solver_result.director_run.submission_receipt
    assert receipt and len(executor.calls) == 3
    assert receipt.worker_tokens_used == 15
    assert app.to_dict()["token_in"] + app.to_dict()["token_out"] == 15
    assert app.task.metadata["selected_output_recovery"]["phase"] == "before_finish_acceptance"
    assert receipt.artifact_id == solver.runtime.artifacts[receipt.output_agent_id].artifact_id
    assert judge.answers == ["35"]
    assert rollout(app).trajectory.metadata["training_eligible"]
    with pytest.raises(ValueError, match="precede"):
        solver.active_canvas.recover_selected_output_agent(reason_code="late")


def test_tool_history_aggregation_is_idempotent():
    from selfplay_graph_flowsteer.adaptive import _aggregate_output_agent_tool_evidence

    app, solver, _, _ = solve(executor=TerminalFailureExecutor(prior_success=True))
    artifact = solver.runtime.artifacts["solver"]
    before = copy.deepcopy(artifact.runtime_tool_evidence)
    _aggregate_output_agent_tool_evidence(solver.active_canvas, artifact)
    assert artifact.runtime_tool_evidence == before


def test_old_or_different_admission_configuration_cannot_resume(tmp_path):
    from selfplay_graph_flowsteer.application import AdaptiveApplicationConfig
    from selfplay_graph_flowsteer.benchmark_tracking import BenchmarkRunTracker
    from selfplay_graph_flowsteer.execution_contract import require_same_semantics

    config = AdaptiveApplicationConfig()
    current = config.model_manifest()["execution_semantics"]
    changed = replace(config, canvas=replace(config.canvas, max_total_tokens=123))
    with pytest.raises(ValueError, match="execution semantics changed"):
        require_same_semantics(current, changed.model_manifest()["execution_semantics"])
    tracker = BenchmarkRunTracker(tmp_path)
    tracker.start_wandb({"execution_semantics": {"submission_contract_version": "legacy"}})
    with pytest.raises(ValueError, match="new run directory"):
        tracker.start_wandb({"execution_semantics": current})


def test_submission_cannot_be_replayed_under_a_different_run_id():
    first, _, _, _ = solve(run_id="first")
    second, _, _, _ = solve(run_id="second")
    second.solver_result.director_run.submission_receipt = first.solver_result.director_run.submission_receipt
    trained = rollout(second)
    assert not trained.trajectory.metadata["training_eligible"]
    assert not trained.trajectory.metadata["answer_reward_released"]


def test_five_slots_retain_policy_failures_with_independent_budgets():
    from selfplay_graph_flowsteer.selfplay import group_rollouts_by_task

    slots = []
    for index in range(5):
        app, _, _, _ = solve(run_id=f"slot-{index}", task_id="same-task", finish=index % 2 == 0)
        slots.append(adaptive_result_to_rollout(app, ByteTokenizer(), rollout_index=index, seed=index))
        receipt = app.solver_result.director_run.submission_receipt
        if receipt:
            assert receipt.worker_token_limit == 240000
            assert receipt.worker_tokens_used == 10
    assert len(group_rollouts_by_task(slots)["same-task"]) == 5
    assert len({slot.trajectory.rollout_id for slot in slots}) == 5
    assert [slot.trajectory.reward for slot in slots] == [1, 0, 1, 0, 1]
    assert all(slot.trajectory.metadata["reward_known"] for slot in slots)


def test_missing_benchmark_slots_remain_in_coverage_denominator():
    from selfplay_graph_flowsteer.benchmark_tracking import benchmark_aggregate

    app, _, _, _ = solve()
    record = from_adaptive_result(app)
    summary = benchmark_aggregate([record], {(record.task_id, 0): "aime"},
                                  planned_by_dataset={"aime": 30})["aime"]
    assert summary["planned_examples"] == 30 and summary["missing_examples"] == 29
    assert summary["unknown_examples"] == 29
    assert summary["submission_rate"] == summary["successful_submission_rate_all"] == 1 / 30
    assert summary["answer_em"] == 1 and summary["answer_metric_examples"] == 1
