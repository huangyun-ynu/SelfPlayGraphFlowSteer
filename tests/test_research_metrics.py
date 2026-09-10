from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.counterfactual import RelationCredit
from selfplay_graph_flowsteer.outcome_metrics import collect_outcome_metrics
from selfplay_graph_flowsteer.research_metrics import (
    policy_signal,
    rollout_diagnostics,
    update_diagnostics,
)
from selfplay_graph_flowsteer.rollouts import TokenizedPolicyCall, TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.selfplay import FrontierScore
from selfplay_graph_flowsteer.wandb_tracking import cycle_payload

from .test_outcome_metrics import population


def sample():
    return TrainingSample("r", "q", (1, 2, 3), (0, 1, 1), 0.0, 0.0)


def credit():
    return RelationCredit(
        "r",
        0,
        "a",
        "b",
        "directed",
        0.0,
        1.0,
        -0.5,
        0.5,
        1,
        action_token_span=(2, 3),
        probability_present=0.5,
    )


def test_primary_dynamics_complete_groups_not_partial_or_health_binary(tmp_path):
    tasks, batch = population(tmp_path)
    report = collect_outcome_metrics(
        tmp_path, cycle=0, tasks=tasks, solver_batch=batch, k=5, snapshots={}
    )
    nq = report["research"]["datasets"]["nq_open"]
    hb = report["research"]["datasets"]["healthbench_professional"]
    assert nq["complete_reward_groups"] == 1
    assert nq["mixed_binary_groups"] == 1
    assert nq["mixed_binary_group_rate"] == 1
    assert nq["cost_by_outcome"]["unscored"]["slot_count"] == 2
    assert nq["orchestration"]["interactive_turns"]["count"] == 0
    assert nq["complexity_buckets"]["unknown"]["slot_count"] == 10
    assert hb["complete_binary_groups"] == 0
    assert hb["mixed_binary_group_rate"] is None
    assert hb["zero_reward_variance_groups"] == 1
    assert hb["cost_by_outcome"]["continuous_scored"]["slot_count"] == 5
    assert report["datasets"]["nq_open"]["share_of_admitted_solver_samples"] == 0.5


def test_context_utilization_and_length_stops_are_observed_not_imputed():
    missing = rollout_diagnostics({})
    assert missing["max_prompt_context_fraction"] is None
    assert missing["finished"] is None
    data = rollout_diagnostics(
        {
            "metadata": {
                "interactive_turns": 2,
                "rejected_turns": 1,
                "finished": False,
                "director_action_diagnostics": [
                    {
                        "finish_reason": "length",
                        "director_dynamic_budget": {"prompt_tokens": 90, "context_limit": 100},
                    },
                    {"finish_reason": "stop", "director_dynamic_budget": None},
                ],
            },
            "policy_calls": [{"token_ids": [1, 2, 3], "action_mask": [0, 1, 1]}],
        }
    )
    assert data["max_prompt_context_fraction"] == 0.9
    assert data["generation_length_stop_count"] == 1
    assert data["generation_finish_reason_count"] == 2
    assert data["policy_call_target_tokens"] == 2


def test_legacy_local_relation_signal_survives_zero_graph_advantage():
    batch = TrainingBatch("solver", (sample(),))
    result = policy_signal(batch, [credit()])
    assert result["target_tokens"] == 2
    assert result["positive_advantage_tokens"] == 1
    assert result["zero_advantage_tokens"] == 1
    assert result["zero_graph_advantage_local_signal_samples"] == 1
    assert result["nonzero_advantage_token_rate"] == 0.5
    assert (
        policy_signal(batch, [credit()], relation_weight=0)["nonzero_advantage_sample_count"] == 0
    )


def test_raw_binary_policy_counts_one_target_not_whole_trajectory():
    call = TokenizedPolicyCall(
        "r:0",
        (1, 2),
        (0, 1),
        relation_token_span=(1, 2),
        metadata={"relation_policy": {"token_ids": {"off": 1, "on": 2}}},
    )
    batch = TrainingBatch("solver", (replace(sample(), policy_calls=(call,)),))
    result = policy_signal(batch, [credit()])
    assert result["target_tokens"] == 1
    assert result["nonzero_advantage_token_rate"] == 1
    assert policy_signal(batch, ())["zero_advantage_tokens"] == 1


def test_update_schedule_is_whole_cycle_and_eval_is_no_update():
    solver = TrainingBatch(
        "solver",
        tuple(replace(sample(), rollout_id=f"r-{i}", task_id=f"q-{i // 5}") for i in range(70)),
    )
    proposer = TrainingBatch("proposer", solver.samples[:14])
    batches = {"solver": solver, "proposer": proposer}
    record = {"policies": {"solver": {"optimizer_steps": 1}, "proposer": {"optimizer_steps": 1}}}
    data = update_diagnostics(record, batches, (), epochs=1, mini_batch_size=70)
    assert data["roles"]["solver"]["eligible_tasks"] == 14
    assert data["roles"]["solver"]["expected_optimizer_steps"] == 1
    assert data["roles"]["solver"]["step_count_matches_plan"]
    assert data["roles"]["proposer"]["step_count_matches_plan"]
    eval_data = update_diagnostics(
        {"experiment": {"evaluation_only": True}}, batches, (), epochs=1, mini_batch_size=70
    )
    assert eval_data["roles"]["solver"]["expected_optimizer_steps"] == 0
    assert eval_data["roles"]["solver"]["actual_optimizer_steps"] == 0
    mismatch = update_diagnostics(record, batches, (), epochs=2, mini_batch_size=64)
    assert mismatch["roles"]["solver"]["expected_optimizer_steps"] == 4
    assert not mismatch["roles"]["solver"]["step_count_matches_plan"]


def test_cf_effect_frontier_stability_and_nan_are_diagnostic_only():
    frontier = FrontierScore(
        "q",
        1.0,
        0.5,
        0.25,
        (0.0, 1.0),
        provisional_graph_local=0.5,
        stable_graph_local=0.25,
        reverify_status="completed",
        pair_stability=(
            {"stability_gate_passed": False, "tie_softened": True},
            {"stability_gate_passed": True},
        ),
    )
    result = update_diagnostics(
        {},
        {"solver": TrainingBatch("solver", ())},
        (credit(), replace(credit(), q_present=float("nan"))),
        epochs=1,
        mini_batch_size=70,
        frontiers=[frontier],
    )
    assert result["relation_effect"]["completed_probe_count"] == 2
    assert result["relation_effect"]["finite_probe_count"] == 1
    assert result["relation_effect"]["absolute_effect"]["mean"] == 1
    assert result["frontier_stability"]["passed_pair_rate"] == 0.5
    assert result["frontier_stability"]["softened_tie_pair_count"] == 1
    assert result["frontier_stability"]["softened_tie_pair_rate"] == 0.5


def test_eval_dataset_curves_do_not_pollute_train_namespace():
    record = {
        "cycle": 2,
        "experiment": {"evaluation_only": True},
        "outcomes": {"datasets": {"nq_open": {"success_count": 3}}},
        "learning_dynamics": {"roles": {"solver": {"actual_optimizer_steps": 0}}},
    }
    data = cycle_payload(record)
    assert data["eval/datasets/nq_open/success_count"] == 3
    assert "cycle/datasets/nq_open/success_count" not in data
    assert data["cycle/is_evaluation"] == 1
    assert data["eval/learning_dynamics/roles/solver/actual_optimizer_steps"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_advantages_are_not_counted_as_useful_signal(value):
    batch = TrainingBatch("solver", (replace(sample(), advantage=value),))
    result = policy_signal(batch, ())
    assert result["nonfinite_advantage_tokens"] == 2
    assert result["nonzero_advantage_token_rate"] == 0
