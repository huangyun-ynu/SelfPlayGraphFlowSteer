"""Synthetic-only partial-group regression tests; no saved experiment inputs."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory, TokenizedPolicyCall
from selfplay_graph_flowsteer.selfplay import (
    AlternatingSnapshots,
    ProposedTask,
    SolverRollout,
    assemble_selfplay_result,
)
from selfplay_graph_flowsteer.training import (
    AlternatingGRPOTrainer,
    AlternatingTrainingConfig,
    MockPolicyTrainer,
    PolicyTrainingConfig,
    UnsafeTrainingBatchError,
    training_batch_from_dict,
)
from selfplay_graph_flowsteer.training_selection import build_training_selection


def fixture(counts):
    proposals, rollouts = [], {}
    for number, count in enumerate(counts, 1):
        task = TaskSpec(f"task-{number}", "synthetic arithmetic", reference="2")
        proposals.append(ProposedTask(task, "synthetic proposal", (1, 2), (0, 1)))
        for index in range(5):
            graph = MultiAgentGraph()
            graph.add_agent("a")
            graph.set_prompt("a", "synthetic solver")
            graph.set_output("a")
            rid = f"{task.task_id}-r{index}"
            trajectory = TokenizedDirectorTrajectory(
                rid,
                task.task_id,
                (1, 2, 3),
                (0, 1, 1),
                float(index % 2),
                graph.to_dict(),
                metadata={
                    "training_eligible": index < count,
                    "reward_known": True,
                    "training_exclusion_reasons": [] if index < count else ["missing_policy"],
                },
            )
            rollouts[rid] = SolverRollout(trajectory, graph)
    return proposals, rollouts


def assemble(counts):
    proposals, rollouts = fixture(counts)
    selection = build_training_selection(proposals, rollouts, 5)
    ids = set(selection["selected_rollout_ids"])
    proposals = [p for p in proposals if any(r.startswith(p.task.task_id + "-r") for r in ids)]
    snapshots = AlternatingSnapshots("synthetic-p", "synthetic-s")
    # Batch assembly expects collection to have reached the Solver phase.
    snapshots.advance()
    result = assemble_selfplay_result(
        proposals,
        [rollouts[rid] for rid in selection["selected_rollout_ids"]],
        snapshots,
        expected_rollouts_per_task=5,
        training_selection=selection,
    )
    return result, selection


@pytest.mark.parametrize("count", range(6))
def test_individual_admission_and_minimum_two(count):
    result, selection = assemble([count])
    assert len(result.solver_batch.samples) == (count if count >= 2 else 0)
    assert len(result.proposer_batch.samples) == int(count == 5)
    assert selection["groups"][0]["eligible_rollout_count"] == count
    assert all(row["reward"] is not None for row in selection["groups"][0]["rollouts"])


def test_scores_and_excluded_graphs_do_not_affect_selected_advantages():
    proposals, rollouts = fixture([4])
    selection = build_training_selection(proposals, rollouts, 5)

    def build():
        snapshots = AlternatingSnapshots("synthetic-p", "synthetic-s")
        snapshots.advance()
        return assemble_selfplay_result(
            proposals,
            [rollouts[rid] for rid in selection["selected_rollout_ids"]],
            snapshots,
            expected_rollouts_per_task=5,
            training_selection=selection,
        ).solver_batch

    before = build()
    bad = rollouts["task-1-r4"]
    rollouts["task-1-r4"] = replace(bad, trajectory=replace(bad.trajectory, reward=99999, graph={}))
    after = build()
    assert before == after
    assert [s.advantage for s in before.samples] == [-1, 1, -1, 1]
    assert all(s.metadata["group_reward_mean"] == 0.5 for s in before.samples)
    assert all(s.metadata["group_reward_std"] == 0.5 for s in before.samples)


def test_zero_score_does_not_admit_missing_policy_data():
    proposals, rollouts = fixture([5])
    r = rollouts["task-1-r0"]
    bad_call = TokenizedPolicyCall("call", (1, 2), (0, 1))
    rollouts["task-1-r0"] = replace(
        r, trajectory=replace(r.trajectory, reward=0, policy_calls=(bad_call,))
    )
    selection = build_training_selection(proposals, rollouts, 5)
    assert "task-1-r0" not in selection["selected_rollout_ids"]
    assert (
        "missing_behavior_log_probs" in selection["groups"][0]["rollouts"][0]["exclusion_reasons"]
    )
    assert selection["groups"][0]["selected_rollout_count"] == 4


def trainer(tmp_path, events, parallel=False, fail_solver=False):
    def factory(role, config, seed):
        events.append(role)
        if role == "solver" and fail_solver:
            raise RuntimeError("synthetic interruption")
        return MockPolicyTrainer(role, config, seed=seed)

    def policy(role):
        return PolicyTrainingConfig(
            Path("/unused-synthetic-model"),
            tmp_path / role,
            device="cpu" if role == "proposer" else "cpu:0",
            mini_batch_size=70,
        )

    return AlternatingGRPOTrainer(
        AlternatingTrainingConfig(
            policy("proposer"), policy("solver"), tmp_path / "state.json", parallel_roles=parallel
        ),
        trainer_factory=factory,
    )


@pytest.mark.parametrize(
    "counts,expected", [([4, 5], ["proposer", "solver"]), ([4], ["solver"]), ([1, 0], [])]
)
def test_role_updates_and_skip_counts(tmp_path, counts, expected):
    result, _ = assemble(counts)
    events = []
    t = trainer(tmp_path, events)
    updates = t.train_cycle(result.proposer_batch, result.solver_batch)
    assert events == expected
    assert t.state.global_step == len(expected)
    assert sum(u.optimizer_steps for u in updates) == len(expected)
    for update in updates:
        if update.role not in expected:
            assert update.status == "skipped"
            assert update.checkpoint == ""
    assert training_batch_from_dict(result.solver_batch.to_dict()) == result.solver_batch


def test_parallel_roles_falls_back_to_solver_only(tmp_path):
    result, _ = assemble([3])
    events = []
    t = trainer(tmp_path, events, parallel=True)
    t.train_cycle(result.proposer_batch, result.solver_batch)
    assert events == ["solver"]


def test_resume_after_proposer_skip_does_not_repeat_skip_or_update(tmp_path):
    result, _ = assemble([4])
    events = []
    t = trainer(tmp_path, events, fail_solver=True)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        t.train_cycle(result.proposer_batch, result.solver_batch)
    assert t.state.phase == "solver"
    t = trainer(tmp_path, events)
    p, s = t.train_cycle(result.proposer_batch, result.solver_batch)
    assert p.status == "skipped" and s.status == "updated"
    assert t.state.global_step == 1
    assert len(t.state.history) == 2


def test_inconsistent_membership_is_rejected_before_update(tmp_path):
    result, _ = assemble([4, 5])
    events = []
    bad = replace(result.solver_batch, samples=result.solver_batch.samples[:-1])
    with pytest.raises(UnsafeTrainingBatchError):
        trainer(tmp_path, events).train_cycle(result.proposer_batch, bad)
    assert events == []


def test_credit_filtering_at_trainer_boundary(tmp_path):
    result, _ = assemble([4])
    seen = []
    t = trainer(tmp_path, [])

    class Capture(MockPolicyTrainer):
        def update(self, batch, *, step, relation_credits=()):
            seen.extend(c.rollout_id for c in relation_credits)
            return super().update(batch, step=step)

    t.trainer_factory = lambda role, config, seed: Capture(role, config, seed=seed)
    t.train_cycle(
        result.proposer_batch,
        result.solver_batch,
        relation_credits=[
            SimpleNamespace(rollout_id="task-1-r0"),
            SimpleNamespace(rollout_id="task-1-r4"),
        ],
    )
    assert seen == ["task-1-r0"]


@pytest.mark.parametrize("all_failed", [False, True])
def test_mock_collection_partial_and_durable_resume(tmp_path, all_failed, monkeypatch):
    import selfplay_graph_flowsteer.selfplay_runtime as runtime_module
    from selfplay_graph_flowsteer.application import (
        create_adaptive_application,
        create_selfplay_snapshots,
        load_adaptive_config,
    )
    from selfplay_graph_flowsteer.cli import _MockProposer
    from selfplay_graph_flowsteer.selfplay_runtime import (
        ByteTokenizer,
        SelfPlayRolloutRunner,
        SelfPlayRunConfig,
    )
    from tests.test_selfplay_runtime import write_config

    original_adapter = runtime_module.adaptive_result_to_rollout

    def synthetic_policy_probabilities(*args, **kwargs):
        rollout = original_adapter(*args, **kwargs)
        calls = tuple(
            replace(c, behavior_log_probs=(-0.5,) * (len(c.token_ids) - 1))
            for c in rollout.trajectory.policy_calls
        )
        return replace(rollout, trajectory=replace(rollout.trajectory, policy_calls=calls))

    monkeypatch.setattr(
        runtime_module, "adaptive_result_to_rollout", synthetic_policy_probabilities
    )
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="numeric")
    calls = []

    class Application:
        def __init__(self, seed):
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            calls.append((kwargs["task_id"], self.seed))
            if all_failed or (kwargs["task_id"] == "task-1" and self.seed == 0):
                raise RuntimeError("synthetic isolated failure")
            return self.delegate.solve(*args, **kwargs)

        def close(self):
            self.delegate.close()

    run_config = SelfPlayRunConfig(
        3, workers=1, task_window=2, rollout_group_policy="eligible_subset"
    )

    def runner():
        return SelfPlayRolloutRunner(
            proposer=_MockProposer(),
            application_factory=Application,
            tokenizer=ByteTokenizer(),
            snapshots=create_selfplay_snapshots(config),
            output_dir=tmp_path / "synthetic-run",
            config=run_config,
        )

    result = runner().run(["synthetic first", "synthetic second"])
    assert len(result.solver_batch.samples) == (0 if all_failed else 5)
    assert len(result.proposer_batch.samples) == (0 if all_failed else 1)
    gate = json.loads((tmp_path / "synthetic-run/batch_gate.json").read_text())
    assert gate["status"] == ("skipped" if all_failed else "ready")
    count = len(calls)
    resumed = runner().run(["synthetic first", "synthetic second"], resume=True)
    assert len(calls) == count
    assert resumed.solver_batch == result.solver_batch
    assert resumed.proposer_batch == result.proposer_batch
    assert len(resumed.tasks) == 2  # Outcome reporting keeps all planned tasks.
    # Explicit repair reopens missing slots even without quarantine records.
    (tmp_path / "synthetic-run/training_selection.json").unlink()
    run_config = replace(run_config, allow_exact_rollout_resume=True)
    runner().run(["synthetic first", "synthetic second"], resume=True)
    assert len(calls) - count == (6 if all_failed else 1)
    from selfplay_graph_flowsteer.training_selection import verify_frozen_selection

    verify_frozen_selection(tmp_path / "synthetic-run")
    frozen = tmp_path / "synthetic-run/solver_batch.json"
    frozen.write_text(frozen.read_text() + " ")
    with pytest.raises(ValueError, match="frozen training artifact changed"):
        verify_frozen_selection(tmp_path / "synthetic-run")


@pytest.mark.parametrize("counts", [[4], [0, 1]])
def test_skipped_role_telemetry_recovers_current_cycle(tmp_path, counts):
    from selfplay_graph_flowsteer.cli import _pending_cycle_metrics_updates

    result, _ = assemble(counts)
    t = trainer(tmp_path, [])
    updates = t.train_cycle(result.proposer_batch, result.solver_batch)
    recovered = _pending_cycle_metrics_updates(t.state.to_dict(), tmp_path / "missing.json")
    assert recovered == updates
    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"cycle": 0}))
    assert _pending_cycle_metrics_updates(t.state.to_dict(), latest) is None


def test_interrupted_cycle_rejects_changed_advantages(tmp_path):
    result, _ = assemble([4])
    t = trainer(tmp_path, [], fail_solver=True)
    with pytest.raises(RuntimeError):
        t.train_cycle(result.proposer_batch, result.solver_batch)
    changed = replace(
        result.solver_batch,
        samples=(
            replace(result.solver_batch.samples[0], advantage=99),
            *result.solver_batch.samples[1:],
        ),
    )
    with pytest.raises(UnsafeTrainingBatchError, match="cannot change a batch"):
        trainer(tmp_path, []).train_cycle(result.proposer_batch, changed)


def test_unequal_rank_gradients_match_global_trajectory_mean(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    from selfplay_graph_flowsteer.solver_data_parallel import synchronize_mean_gradients

    # Nine synthetic trajectories; token counts differ, task A has four and B five.
    trajectories = [torch.arange(1, i + 3, dtype=torch.float32) for i in range(9)]

    def gradient(indices):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.25)
        losses = [(model(trajectories[i][:, None]).square()).mean() for i in indices]
        torch.stack(losses).mean().backward()
        return model

    expected = gradient(range(9)).weight.grad.clone()
    left, right = gradient(range(4)), gradient(range(4, 9))
    remote = right.weight.grad.clone().reshape(-1) * (5 / 9)
    call = 0

    def all_reduce(tensor, op=None):
        nonlocal call
        if call == 0:
            tensor.add_(5)
        elif call == 2:
            tensor.add_(remote)
        call += 1

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    synchronize_mean_gradients(left, 4, 9)
    assert torch.allclose(left.weight.grad, expected, rtol=1e-6)
    assert call == 3


def test_cli_defaults_to_subset_and_skips_unscored_mock_data(tmp_path, capsys):
    from selfplay_graph_flowsteer.cli import main
    from tests.test_training_ops import _write_config

    config = _write_config(tmp_path)
    seeds = tmp_path / "synthetic-seeds.jsonl"
    seeds.write_text('{"seed":"synthetic answer"}\n')
    output = tmp_path / "synthetic-experiment"
    assert (
        main(
            [
                "selfplay-experiment",
                "--config",
                str(config),
                "--seed-data",
                str(seeds),
                "--output",
                str(output),
                "--cycles",
                "1",
                "--rollouts",
                "2",
                "--verifier",
                "none",
                "--mock",
                "--mock-trainer",
            ]
        )
        == 0
    )
    capsys.readouterr()
    state = json.loads((output / "training_state.json").read_text())
    assert state["global_step"] == 0
    assert all(row["status"] == "skipped" for row in state["history"])
    assert state["proposer_checkpoint"] == state["solver_checkpoint"] == ""
    selections = list(output.glob("**/training_selection.json"))
    assert len(selections) == 1
    assert json.loads(selections[0].read_text())["rollout_group_policy"] == "eligible_subset"


def test_partial_solver_does_not_create_an_overweight_short_tail(tmp_path):
    from selfplay_graph_flowsteer.training import logical_mini_batch_size

    result, _ = assemble([4, 5])
    t = trainer(tmp_path, [])
    small = replace(t.config.solver, mini_batch_size=4)
    assert logical_mini_batch_size(result.solver_batch, small) == 9
    assert logical_mini_batch_size(result.proposer_batch, small) == 4
    update = MockPolicyTrainer("solver", small).update(result.solver_batch, step=1)
    assert update.samples == 9 and update.optimizer_steps == 1


def test_diagnostics_use_actual_global_mean_and_collection_only():
    from selfplay_graph_flowsteer.research_metrics import update_diagnostics

    result, _ = assemble([4, 5])
    record = {"policies": {"solver": {"optimizer_steps": 1}}}
    args = dict(batches={"solver": result.solver_batch}, credits=(), epochs=1, mini_batch_size=4)
    metrics = update_diagnostics(record, **args)["roles"]["solver"]
    assert metrics["logical_mini_batch_limit"] == 9
    assert metrics["expected_optimizer_steps"] == 1
    assert metrics["step_count_matches_plan"]
    metrics = update_diagnostics({"experiment": {"collection_only": True}}, **args)["roles"][
        "solver"
    ]
    assert metrics["expected_optimizer_steps"] == 0
    assert metrics["step_count_matches_plan"]


def test_probability_admission_precedes_group_freeze_and_resume(tmp_path, monkeypatch):
    import selfplay_graph_flowsteer.selfplay_runtime as runtime_module
    from selfplay_graph_flowsteer.application import (
        create_adaptive_application,
        create_selfplay_snapshots,
        load_adaptive_config,
    )
    from selfplay_graph_flowsteer.cli import _MockProposer
    from selfplay_graph_flowsteer.selfplay_runtime import (
        ByteTokenizer,
        SelfPlayRolloutRunner,
        SelfPlayRunConfig,
    )
    from tests.test_selfplay_runtime import write_config

    original = runtime_module.adaptive_result_to_rollout

    def adapt(*args, **kwargs):
        r = original(*args, **kwargs)
        return replace(
            r,
            trajectory=replace(
                r.trajectory,
                policy_calls=tuple(
                    replace(c, behavior_log_probs=(-0.5,) * (len(c.token_ids) - 1))
                    for c in r.trajectory.policy_calls
                ),
            ),
        )

    monkeypatch.setattr(runtime_module, "adaptive_result_to_rollout", adapt)
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="numeric")

    class Observer:
        def submit(self, trajectory):
            pass

        def finalize(self):
            return {"test": True}

        def identity_exclusions(self, trajectories):
            assert len(trajectories) == 6
            return {"task-1-r0": [{"call_id": "synthetic", "relation_call": True}]}

    def runner():
        return SelfPlayRolloutRunner(
            proposer=_MockProposer(),
            application_factory=lambda seed: create_adaptive_application(config, mock=True),
            tokenizer=ByteTokenizer(),
            snapshots=create_selfplay_snapshots(config),
            output_dir=tmp_path / "run",
            config=SelfPlayRunConfig(
                3, workers=1, task_window=2, rollout_group_policy="eligible_subset"
            ),
            primary_probability_observer=Observer(),
        )

    result = runner().run(["first", "second"])
    assert len(result.solver_batch.samples) == 5
    assert [s.task_id for s in result.proposer_batch.samples] == ["task-2"]
    assert "task-1-r0" not in [s.rollout_id for s in result.solver_batch.samples]
    selected = json.loads((tmp_path / "run/training_selection.json").read_text())
    assert selected["groups"][0]["selected_rollout_count"] == 2
    assert selected["groups"][0]["rollouts"][0]["exclusion_reasons"] == [
        "frozen_policy_probability_mismatch",
        "training_ineligible",
    ]
    resumed = runner().run(["first", "second"], resume=True)
    assert resumed.solver_batch == result.solver_batch
