"""Wrong model answers must stay gradable; incomplete executions must not."""

import pytest

from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from selfplay_graph_flowsteer.observability import (
    FlowSteerQAVerifier,
    MultiAnswerExactMatchVerifier,
    NumericVerifier,
    TaskSpec,
)

from .helpers import NumericRecordingExecutor
from .test_finish_submission_contract import rollout, solve


class RawAnswerExecutor(NumericRecordingExecutor):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def execute(self, **kwargs):
        artifact = super().execute(**kwargs)
        artifact.answer = self.answer
        artifact.summary = "An attractive alternative is 35 or Paris."
        return artifact


@pytest.mark.parametrize("raw", [
    "1681", "1105", "-1", "35.0", "[12,7,3]",
    'Final answer: 35\nFinal answer: 36',
])
def test_aime_wrong_format_is_frozen_scored_zero_and_kept_for_training(raw):
    app, _, executor, _ = solve(
        executor=RawAnswerExecutor(raw), verifier=NumericVerifier(), reference="35",
    )
    result = app.solver_result
    assert result.director_run.finished
    assert result.director_run.submission_receipt.submitted_answer_snapshot == raw
    assert result.answer_submission.submitted_answer == raw
    assert result.verification.score == 0 and not result.verification.passed
    assert result.outcome_decision.status == "scored"
    assert len(executor.calls) == 2
    trained = rollout(app)
    assert trained.trajectory.reward == 0
    assert trained.trajectory.metadata["reward_known"]
    assert trained.trajectory.metadata["training_eligible"]
    assert trained.trajectory.metadata["answer_reward_released"]


@pytest.mark.parametrize("dataset", ["nq_open", "hotpotqa"])
@pytest.mark.parametrize("verifier", [FlowSteerQAVerifier, MultiAnswerExactMatchVerifier])
@pytest.mark.parametrize("raw", [
    'Final answer: Paris\nFinal answer: London',
    '{"answer":"Paris","final_answer":"London"}',
    '{"answer":"Paris","answer":"London"}',
    '["Paris", "London"]',
])
def test_qa_ambiguous_answer_is_submitted_without_choosing_correct_candidate(dataset, verifier, raw):
    app, _, executor, _ = solve(
        dataset, executor=RawAnswerExecutor(raw), verifier=verifier(), reference="Paris",
    )
    result = app.solver_result
    assert result.director_run.finished
    assert result.answer_submission.submitted_answer == raw
    assert result.verification.score == 0 and not result.verification.passed
    assert result.outcome_decision.status == "scored"
    assert app.task.metadata["qa_official_metrics"]["answer_em"] == 0
    assert len(executor.calls) == 2
    trained = rollout(app)
    assert trained.trajectory.metadata["training_eligible"]
    assert trained.trajectory.metadata["reward_known"]
    assert trained.trajectory.reward == 0


@pytest.mark.parametrize("raw", ["I do not know.", "Brief incomplete advice. " * 150])
def test_healthbench_answer_quality_and_length_do_not_block_submission(raw):
    # No live medical Judge call: the recording rubric fixture verifies transport.
    app, _, executor, judge = solve(
        "healthbench_professional", executor=RawAnswerExecutor(raw), reference="35",
    )
    assert app.solver_result.director_run.finished
    assert judge.answers == [raw.strip()]
    assert app.solver_result.outcome_decision.status == "scored"
    assert len(executor.calls) == 2


@pytest.mark.parametrize("dataset", ["aime", "nq_open", "hotpotqa", "healthbench_professional"])
def test_empty_answer_remains_unsubmittable(dataset):
    app, _, _, judge = solve(dataset, executor=RawAnswerExecutor(""))
    assert not app.solver_result.director_run.finished
    assert app.solver_result.director_run.submission_receipt is None
    assert judge.answers == []


@pytest.mark.parametrize("dataset,raw,expected", [
    ("aime", r"\boxed{035}", "035"),
    ("nq_open", '{"answer":"Paris"}', "Paris"),
    ("hotpotqa", 'Final answer: yes', "yes"),
])
def test_valid_deterministic_normalization_is_unchanged(dataset, raw, expected):
    finalizer = AnswerFinalizer(AnswerSubmissionConfig(enabled=True))
    task = TaskSpec("normalization", "question", metadata={"dataset": dataset})
    assert finalizer.finalize(task, raw).submitted_answer == expected
