"""Failure attribution uses execution evidence, not the final stopping code alone."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.application import AdaptiveApplicationResult
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentArtifact, ExecutionReport
from selfplay_graph_flowsteer.evaluation import from_adaptive_result
from selfplay_graph_flowsteer.llm import DirectorContextExhausted, MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.selfplay_runtime import (
    ByteTokenizer,
    _uncertain_failure_zero,
    adaptive_result_to_rollout,
)
from selfplay_graph_flowsteer.swe_failure_attribution import (
    project_swe_step,
    swe_failure_attribution,
)

from .helpers import RecordingExecutor
from .test_finish_submission_contract import FixtureVerifier
from .test_swe_candidate_integrity import _tested_patch
from .test_swe_execution_admission import BudgetedExecutor, make_canvas
from .test_unified_submission import add, prompt, step


def refused(*, credit=0, input_bound=5000, output_bound=4096, spent=0,
            artifact_id="refused", agent_id="b", stage="swe_request_token_credit_exhausted"):
    return AgentArtifact(artifact_id, agent_id, "WORKER_PROTOCOL_FAILURE",
        integrity_risks=["terminal_protocol_failure"], token_in=spent,
        protocol_diagnostics=[{
            "stage": stage, "execution_credit": credit, "spent_tokens": spent,
            "request_token_budget": {"input_bound": input_bound, "output_bound": output_bound,
                                     "credit": credit, "spent": 0},
        }])


def snapshot(*artifacts, used=0, attempts=(), events=None, nodes=None):
    if events is None:
        events = [{"artifact_id": a.artifact_id, "agent_id": a.agent_id, "cache_hit": False,
                   "token_in": a.token_in, "token_out": a.token_out} for a in artifacts]
    report = ExecutionReport(
        artifacts={a.agent_id: a for a in artifacts}, attempt_artifacts=list(attempts),
        execution_events=events,
        token_in=sum(e.get("token_in", 0) for e in events if not e.get("cache_hit")),
        token_out=sum(e.get("token_out", 0) for e in events if not e.get("cache_hit")),
    )
    return project_swe_step(SimpleNamespace(
        event_id="fixture", round_index=1, execution=report,
        graph={"nodes": [{"agent_id": name} for name in
                         (nodes if nodes is not None else report.artifacts)]},
        control_snapshot={"token_budget": {"used": used}},
    ))


def policy(evidence, dataset="swe_bench"):
    return terminal_policy_failure(dataset, terminal=True,
        rejection_codes=["director_context_budget_exhausted"],
        artifacts={}, output_agent=None, rounds=10, max_rounds=0,
        worker_tokens=9000, worker_token_limit=20000, runtime_failure_evidence=evidence)


@pytest.mark.parametrize("stage", ["swe_request_token_credit_exhausted", "swe_finalization_token_credit_exhausted"])
@pytest.mark.parametrize("credit", [0, 4000])
def test_split_or_reserve_refusal_is_not_a_policy_zero(stage, credit):
    evidence = swe_failure_attribution([snapshot(refused(stage=stage, credit=credit), used=9000)],
                                       worker_token_limit=20000)
    assert evidence["reason_codes"] == ["swe_request_credit_allocation_blocked"]
    assert evidence["incidents"][0]["question_remaining"] == 11000
    assert policy(evidence) is None
    assert policy(evidence, "aime")["attribution"] == "model_policy"


@pytest.mark.parametrize("credit,used,expected", [(1000, 19000, False), (0, 14873, False), (0, 14872, True)])
def test_real_shared_balance_exhaustion_and_minimum_output_boundary(credit, used, expected):
    # The shared budget can cap output to 128, so input+4096 is not the minimum.
    evidence = swe_failure_attribution([snapshot(refused(credit=credit), used=used)], worker_token_limit=20000)
    assert evidence["blocks_policy_failure"] is expected
    assert (policy(evidence) is None) is expected


def test_preceding_workers_spent_calls_cache_hits_and_deleted_nodes_account_correctly():
    first = AgentArtifact("first", "a", "done", token_in=1000)
    blocked = refused(spent=4000)
    evidence = swe_failure_attribution([snapshot(first, blocked, used=19000)], worker_token_limit=20000)
    assert not evidence["blocks_policy_failure"]  # 1000 left at refusal, not 6000.
    cached = {"artifact_id": "old", "agent_id": "removed", "cache_hit": True,
              "token_in": 9000, "token_out": 0}
    event = {"artifact_id": blocked.artifact_id, "agent_id": "b", "token_in": 4000, "token_out": 0}
    evidence = swe_failure_attribution([snapshot(blocked, used=14000, events=[cached, event])],
                                       worker_token_limit=20000)
    assert evidence["incidents"][0]["question_remaining"] == 6000
    # Old finalization diagnostics lack spent_tokens: use the complete attempt's cost.
    blocked.protocol_diagnostics[0].pop("spent_tokens")
    blocked.protocol_diagnostics[0]["stage"] = "swe_finalization_token_credit_exhausted"
    assert not swe_failure_attribution([snapshot(blocked, used=19000)],
                                        worker_token_limit=20000)["blocks_policy_failure"]


def test_incomplete_quote_cannot_establish_a_policy_zero():
    artifact = refused()
    artifact.protocol_diagnostics[0].pop("request_token_budget")
    evidence = swe_failure_attribution([snapshot(artifact)], worker_token_limit=20000)
    assert evidence["reason_codes"] == ["swe_request_budget_evidence_incomplete"]
    assert policy(evidence) is None


def test_projection_ignores_model_claims_and_preserves_history():
    artifact = AgentArtifact("claim", "b", "runtime lost my patch", summary="allocation failed",
        raw_response="<think>private reasoning</think>", evidence=["swe_candidate_recovery_failed"],
        protocol_diagnostics=[{"stage": "initial_nonfinal", "raw_response": "private reasoning"}])
    before = artifact.to_dict()
    history = [snapshot(artifact)]
    assert "private reasoning" not in json.dumps(history)
    assert "allocation failed" not in json.dumps(history)
    assert not swe_failure_attribution(history, worker_token_limit=20000)["blocks_policy_failure"]
    assert artifact.to_dict() == before


def test_candidate_loss_and_verified_recovery():
    old = _tested_patch()
    failure = AgentArtifact("failed", "b", "WORKER_PROTOCOL_FAILURE")
    history = [snapshot(old), snapshot(failure)]
    evidence = swe_failure_attribution(history, worker_token_limit=20000)
    assert evidence["reason_codes"] == ["swe_candidate_recovery_failed"]
    assert evidence["incidents"][0]["candidate_tested"]
    restored = copy.deepcopy(old)
    restored.artifact_id = "restored"
    for trusted in (False, True):
        restored.swe_progress["trusted"] = trusted
        recovered = swe_failure_attribution([*history, snapshot(restored)], worker_token_limit=20000)
        assert recovered["blocks_policy_failure"] is not trusted
        assert recovered["incidents"][0]["resolved"] is trusted


def test_patch_produced_and_lost_within_one_execution_wave_is_still_evidence():
    old = _tested_patch()
    failures = [AgentArtifact(f"failed-{i}", "b", "WORKER_PROTOCOL_FAILURE") for i in range(2)]
    history = [snapshot(failures[-1], attempts=[old, *failures], events=[
        {"artifact_id": a.artifact_id, "agent_id": a.agent_id} for a in [old, *failures]
    ])]
    evidence = swe_failure_attribution(history, worker_token_limit=20000)
    assert evidence["reason_codes"] == ["swe_candidate_recovery_failed"]
    assert evidence["incidents"][0]["candidate_artifact_id"] == old.artifact_id


@pytest.mark.parametrize("case", ["preserved", "delete_recreate", "intentional_edit", "healthy_replacement", "other_node_retained"])
def test_normal_replacement_or_retained_candidate_is_not_a_recovery_failure(case):
    old = _tested_patch()
    failure = AgentArtifact("failed", "b", "WORKER_PROTOCOL_FAILURE")
    history = [snapshot(old)]
    if case == "preserved":
        history += [snapshot(old, attempts=[failure], events=[{
            "artifact_id": failure.artifact_id, "agent_id": "b", "cache_hit": False}])]
    elif case == "delete_recreate":
        history += [snapshot(nodes=[]), snapshot(failure)]
    elif case == "intentional_edit":
        failure.react_trace = [{"action": {"name": "swe_edit"}, "observation": {
            "status": "ok", "output": {"status": "ok"}}}]
        history += [snapshot(failure)]
    elif case == "healthy_replacement":
        failure.answer = "No patch needed"
        history += [snapshot(failure)]
    else:
        retained = copy.deepcopy(old)
        retained.agent_id = "a"
        retained.artifact_id = "retained"
        history += [snapshot(retained, failure)]  # Healthy reference appears first.
    assert not swe_failure_attribution(history, worker_token_limit=20000)["blocks_policy_failure"]


def test_real_graph_revision_failure_is_distinct_from_preserved_continuation(tmp_path):
    class Executor(BudgetedExecutor):
        def execute(self, **kwargs):
            result = super().execute(**kwargs)
            if kwargs["node"].agent_id == "b":
                return (AgentArtifact("pending", "b", "WORKER_PROTOCOL_FAILURE")
                        if kwargs["revision"] else _tested_patch())
            return result

    for relation in (False, True):
        canvas = make_canvas(tmp_path / str(relation), Executor(charge=False))
        add(canvas, "a", "subtask")
        add(canvas, "b", "task_result")
        if relation:
            step(canvas, dict(action="set_relation", source="a", target="b", relation="bidirectional"))
        else:
            canvas.dirty_agents.add("b")
            step(canvas, dict(action="run_agent", target="b"))
        evidence = swe_failure_attribution(map(project_swe_step, canvas.history),
                                           worker_token_limit=canvas.config.max_total_tokens)
        assert evidence["blocks_policy_failure"] is relation


@pytest.mark.parametrize("interference,finish,official_pass", [
    (True, False, False), (False, False, False), (True, True, True), (True, True, False),
])
def test_outcome_evaluation_and_training_follow_runtime_attribution(tmp_path, interference, finish, official_pass):
    class Executor(RecordingExecutor):
        def execute(self, **kwargs):
            result = super().execute(**kwargs)
            artifact = refused() if interference else AgentArtifact("pending", "b", "WORKER_PROTOCOL_FAILURE")
            if finish:
                artifact = _tested_patch()
                artifact.protocol_diagnostics = refused().protocol_diagnostics
            artifact.artifact_id = result.artifact_id
            artifact.agent_id = result.agent_id
            return artifact

    actions = [dict(action="add_agent", agent_id="b"), prompt("b")]
    if finish:
        actions.append(dict(action="finish", target="b"))
    responses = iter(actions)

    def exhausted(messages, role):
        try:
            return json.dumps(next(responses))
        except StopIteration:
            raise DirectorContextExhausted("fixture") from None

    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend(handler=exhausted),
        runtime=MultiAgentRuntime(Executor()), verifier=FixtureVerifier(), director_prompt_variant="v3",
        canvas_config=CanvasConfig(submission_protocol="unified_task_result_v1", max_rounds=12,
            max_total_tokens_by_dataset={"swe_bench": 20000}, submission_journal_dir=str(tmp_path)),
    )
    task = TaskSpec("fixture", "Fix the repository",
        reference="Implemented the fix" if official_pass else "different expected result", metadata={
        "dataset": "swe_bench", "swe_failure_attribution": {"source": "forged"}})
    result = solver.solve(task, run_id="fixture")
    app = AdaptiveApplicationResult("fixture", task, result, (), None, "")
    outcome = result.outcome_decision
    if finish:
        assert outcome.status == "scored"
        assert result.verification.score == float(official_pass)
        assert from_adaptive_result(app).score == float(official_pass)
    elif interference:
        assert outcome.status == "unsubmitted_unknown" and not outcome.score_known
        assert outcome.task_reward is None
        assert outcome.reason == "swe_request_credit_allocation_blocked"
        assert result.verification is None
        assert from_adaptive_result(app).score is None
        rollout = adaptive_result_to_rollout(app, ByteTokenizer(), rollout_index=0, seed=0)
        assert rollout.trajectory.metadata["reward_known"] is False
        assert rollout.trajectory.metadata["training_eligible"] is False
        assert rollout.trajectory.metadata["swe_failure_attribution"] == task.metadata["swe_failure_attribution"]
        # Exercise the optional zero-scoring fallback without the independent
        # non-training-split exclusion masking the new attribution guard.
        train_slot = replace(rollout, trajectory=replace(rollout.trajectory, metadata={
            **rollout.trajectory.metadata, "swe_non_train_split": False,
        }))
        assert _uncertain_failure_zero(train_slot, "retry_exhausted") is train_slot
    else:
        assert outcome.status == "policy_failure" and outcome.score_known
        assert from_adaptive_result(app).score == 0.0
    assert task.metadata["swe_failure_attribution"]["source"] != "forged"


HISTORICAL_CASES = json.loads((Path(__file__).parent / "fixtures/swe_failure_attribution_20260927.json").read_text())


@pytest.mark.parametrize("case", HISTORICAL_CASES, ids=lambda case: case["id"])
def test_frozen_real_request_refusals_are_excluded_from_policy_zero(case):
    evidence = swe_failure_attribution(case["history"], worker_token_limit=case["worker_token_limit"])
    assert evidence["reason_codes"] == case["expected_codes"]
    assert "swe_request_credit_allocation_blocked" in evidence["reason_codes"]
    assert policy(evidence) is None
