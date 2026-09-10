"""Synthetic CPU-only checks for the independent Proposer learning protocol."""

import itertools
import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.features import GraphFeatures
from selfplay_graph_flowsteer.proposer_learning import (
    BASELINE_VERSION,
    NORMALIZATION,
    DatasetBaselineStore,
    frontier_evidence_exclusions,
    validate_independent_batches,
)
from selfplay_graph_flowsteer.rollouts import TokenizedPolicyCall
from selfplay_graph_flowsteer.selfplay import (
    AlternatingSnapshots,
    assemble_selfplay_result,
    graph_local_frontier,
    scalar_frontier,
    stable_graph_local_frontier,
)
from selfplay_graph_flowsteer.training import UnsafeTrainingBatchError, logical_mini_batch_size
from selfplay_graph_flowsteer.training_selection import INDEPENDENT_SCHEMA, build_training_selection
from tests.test_partial_training_groups import fixture, trainer


def baseline(values=None, cycle=0):
    return dict(
        version=BASELINE_VERSION,
        normalization=NORMALIZATION,
        values=values or {},
        cycle=cycle,
        decay=0.9,
    )


def records(counts):
    proposals, rollouts = fixture(counts)
    call = TokenizedPolicyCall("selection", (1, 2), (0, 1), behavior_log_probs=(-0.5,))
    return [replace(p, policy_calls=(call,)) for p in proposals], rollouts


def assemble(proposals, rollouts, frozen=None, reverify=None):
    selection = build_training_selection(proposals, rollouts, 5, schema=INDEPENDENT_SCHEMA)
    snapshots = AlternatingSnapshots("p", "s")
    snapshots.advance()
    result = assemble_selfplay_result(
        proposals,
        [rollouts[rid] for rid in selection["selected_rollout_ids"]],
        snapshots,
        training_selection=selection,
        frontier_evidence=[rollouts[rid] for rid in selection["frontier_rollout_ids"]],
        proposer_baseline=frozen or baseline(),
        frontier_reverification=reverify,
    )
    validate_independent_batches(result.proposer_batch, result.solver_batch)
    return result


def test_pair_average_has_same_expectation_for_all_uniform_subsets():
    rewards = [0.1, 0.9, 0.3, 0.0, 1.0]
    vectors = [GraphFeatures((1.0, float(i)), ("a", "b")) for i in range(5)]
    expected = graph_local_frontier(rewards, vectors)
    assert graph_local_frontier(rewards, vectors, normalization=NORMALIZATION) == expected
    for n in (2, 3, 4):
        values = [
            graph_local_frontier(
                [rewards[i] for i in ids], [vectors[i] for i in ids], normalization=NORMALIZATION
            )
            for ids in itertools.combinations(range(5), n)
        ]
        assert sum(values) / len(values) == pytest.approx(expected)
    assert scalar_frontier([0, 1], normalization=NORMALIZATION) == 1.6


def test_stable_normalization_counts_rejected_pairs_in_denominator():
    features = [GraphFeatures((1.0,), ("a",))] * 3
    score, pairs = stable_graph_local_frontier(
        [1, 0, 0.5], [0.8, 0.2, 0.9], features, normalization=NORMALIZATION
    )
    assert [p["stability_gate_passed"] for p in pairs] == [True, False, True]
    assert score == pytest.approx(1.6 / 3 * (0.64 + 0.36))


@pytest.mark.parametrize("count", range(6))
def test_frontier_and_proposer_do_not_require_solver_policy_admission(count):
    proposals, rollouts = records([count])
    result = assemble(proposals, rollouts)
    assert len(result.solver_batch.samples) == (count if count >= 2 else 0)
    assert len(result.proposer_batch.samples) == 1
    assert result.frontier_scores[0].metadata["actual_count"] == 5


def test_frontier_partial_group_uses_two_trusted_scores_and_preserves_solver_weights(tmp_path):
    proposals, rollouts = records([4, 5])
    for i in (2, 3, 4):
        r = rollouts[f"task-1-r{i}"]
        rollouts[r.trajectory.rollout_id] = replace(
            r,
            trajectory=replace(
                r.trajectory, metadata={**r.trajectory.metadata, "reward_known": False}
            ),
        )
    result = assemble(proposals, rollouts, baseline({"unknown": 0.2}))
    assert len(result.solver_batch.samples) == 7
    assert result.frontier_scores[0].graph_local == pytest.approx(1.6)
    assert result.proposer_batch.samples[0].advantage == pytest.approx(1.4)
    config = replace(trainer(tmp_path, []).config.solver, mini_batch_size=4)
    assert logical_mini_batch_size(result.solver_batch, config) == 7


def test_excluded_proposer_policy_does_not_discard_solver_or_trusted_baseline_evidence():
    proposals, rollouts = records([5])
    proposals[0] = replace(proposals[0], policy_calls=())
    result = assemble(proposals, rollouts)
    assert not result.proposer_batch.samples
    assert len(result.solver_batch.samples) == 5
    assert result.proposer_batch.metadata["frontier_dataset_means"]["unknown"] > 0


def test_existing_pool_attestation_and_audited_reward_gates_are_preserved():
    proposals, rollouts = records([5])
    proposals[0].metadata["pool_id"] = "synthetic-unvalidated"
    with pytest.raises(ValueError, match="validated task-pool"):
        assemble(proposals, rollouts)
    proposals[0].metadata["validated_pool_entry"] = True
    rollouts["task-1-r0"].trajectory.metadata["task_reward"] = 999
    with pytest.raises(ValueError, match="audited task_reward"):
        assemble(proposals, rollouts)


def test_unknown_zero_and_corrupt_graph_are_not_evidence():
    _, rollouts = records([5])
    r = rollouts["task-1-r0"]
    unknown = replace(
        r,
        trajectory=replace(
            r.trajectory, reward=0, metadata={**r.trajectory.metadata, "reward_known": False}
        ),
    )
    assert "untrusted_reward" in frontier_evidence_exclusions(unknown)
    corrupt = replace(r, trajectory=replace(r.trajectory, graph={}))
    assert "invalid_frontier_evidence" in frontier_evidence_exclusions(corrupt)
    counted_zero = replace(
        r,
        trajectory=replace(
            r.trajectory,
            reward=0,
            metadata={
                **r.trajectory.metadata,
                "reward_known": True,
                "uncertain_attribution_zero": {"attribution": "unknown"},
            },
        ),
    )
    assert "statistical_zero_without_trusted_outcome" in frontier_evidence_exclusions(counted_zero)


def test_reverification_requires_exact_primary_ids_and_order():
    proposals, rollouts = records([4])
    reverify = {
        "task-1": {
            "rewards": [0, 1, 0, 1, 0],
            "rollout_ids": [f"task-1-r{i}" for i in reversed(range(5))],
        }
    }
    with pytest.raises(ValueError, match="membership/order"):
        assemble(proposals, rollouts, reverify=reverify)


def test_nonreplayable_primary_does_not_abort_other_training(tmp_path):
    from selfplay_graph_flowsteer.selfplay_runtime import SelfPlayRolloutRunner, SelfPlayRunConfig

    proposals, rollouts = records([5])
    item = rollouts["task-1-r0"]
    item.graph.output_agent = None
    rollouts["task-1-r0"] = replace(
        item, trajectory=replace(item.trajectory, graph=item.graph.to_dict())
    )

    def unavailable(_seed):
        raise AssertionError("an unconfigured graph must never start an Executor")

    runner = SelfPlayRolloutRunner(
        proposer=None,
        application_factory=unavailable,
        tokenizer=None,
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(frontier_reverify_fraction=1),
    )
    runner._independent_frontier = True
    assert runner._collect_frontier_reverification(proposals, list(rollouts.values())) == {}
    result = assemble(proposals, rollouts)
    assert len(result.solver_batch.samples) == 5
    assert not result.proposer_batch.samples
    failures = json.loads((tmp_path / "frontier_reverification_failures.json").read_text())
    assert failures[0]["error_type"] == "NonReplayablePrimaryGraph"


def test_failed_reverify_excludes_only_proposer_and_does_not_supply_zero_baseline():
    proposals, rollouts = records([5])
    proposals[0].metadata["frontier_training_exclusion"] = (
        "frontier_reverify_infrastructure_failure"
    )
    result = assemble(proposals, rollouts)
    assert not result.proposer_batch.samples
    assert len(result.solver_batch.samples) == 5
    assert result.proposer_batch.metadata["frontier_dataset_means"] == {}


def test_zero_frontier_gets_negative_advantage_from_frozen_history():
    proposals, rollouts = records([5])
    rollouts = {
        rid: replace(r, trajectory=replace(r.trajectory, reward=0)) for rid, r in rollouts.items()
    }
    result = assemble(proposals, rollouts, baseline({"unknown": 0.2}))
    assert result.proposer_batch.samples[0].reward == 0
    assert result.proposer_batch.samples[0].advantage == -0.2


def test_baseline_freezes_before_collection_and_commits_once_in_cycle_order(tmp_path):
    store = DatasetBaselineStore(tmp_path / "state.json")
    first = store.freeze(tmp_path / "first.json", cycle=0)
    second = store.freeze(tmp_path / "second.json", cycle=1)
    store.commit(second, {"aime": 0})
    assert json.loads(store.path.read_text())["values"] == {}
    store.commit(first, {"aime": 1, "healthbench_professional": 0.2})
    state = json.loads(store.path.read_text())
    assert state["values"]["aime"] == pytest.approx(0.09)
    assert state["values"]["healthbench_professional"] == pytest.approx(0.02)
    assert second["values"] == {}
    assert store.freeze(tmp_path / "second.json", cycle=1) == second
    store.commit(first, {"aime": 1, "healthbench_professional": 0.2})
    assert json.loads(store.path.read_text()) == state
    with pytest.raises(ValueError, match="cannot change"):
        store.commit(first, {"aime": 0.5})


@pytest.mark.parametrize(
    "solver_count,broken_proposer,expected",
    [(0, False, ["proposer"]), (5, True, ["solver"]), (4, False, ["proposer", "solver"])],
)
def test_independent_role_updates(tmp_path, solver_count, broken_proposer, expected):
    proposals, rollouts = records([solver_count])
    if broken_proposer:
        proposals[0] = replace(proposals[0], policy_calls=())
    result = assemble(proposals, rollouts)
    events = []
    updates = trainer(tmp_path, events).train_cycle(result.proposer_batch, result.solver_batch)
    assert events == expected
    assert sum(u.optimizer_steps for u in updates) == len(expected)


def test_changed_frozen_advantage_is_rejected_before_update(tmp_path):
    result = assemble(*records([4]))
    bad = replace(
        result.proposer_batch, samples=(replace(result.proposer_batch.samples[0], advantage=7),)
    )
    events = []
    with pytest.raises(UnsafeTrainingBatchError):
        trainer(tmp_path, events).train_cycle(bad, result.solver_batch)
    assert not events


def test_healthbench_frontier_keeps_exact_solver_length_adjusted_rewards():
    from selfplay_graph_flowsteer.dataset_adapters import healthbench_training_reward

    proposals, rollouts = records([5])
    expected = []
    for index, (rid, rollout) in enumerate(rollouts.items()):
        reward = healthbench_training_reward(
            weighted_points=-8,
            positive_points=8,
            negative_point_magnitude=8,
            answer_length_chars=500 + 20 * index,
        ).training_reward
        expected.append(reward)
        rollouts[rid] = replace(rollout, trajectory=replace(rollout.trajectory, reward=reward))
    result = assemble(proposals, rollouts)
    assert list(result.frontier_scores[0].rewards) == expected
    assert [s.reward for s in result.solver_batch.samples] == expected
    assert result.proposer_batch.samples[0].reward > 0


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("baseline_mode", ["none", "ema"])
def test_runtime_partial_evidence_resume_and_baseline_commit(
    tmp_path, monkeypatch, pipeline, baseline_mode
):
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
    from selfplay_graph_flowsteer.training_selection import verify_frozen_selection
    from tests.test_selfplay_runtime import write_config

    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="numeric")
    original = runtime_module.adaptive_result_to_rollout

    def adapter(*args, **kwargs):
        r = original(*args, **kwargs)
        calls = tuple(
            replace(c, behavior_log_probs=(-0.5,) * (len(c.token_ids) - 1))
            for c in r.trajectory.policy_calls
        )
        reward = float(r.trajectory.seed % 2)
        return replace(
            r,
            trajectory=replace(
                r.trajectory,
                reward=reward,
                policy_calls=calls,
                metadata={
                    **r.trajectory.metadata,
                    "reward_known": True,
                    "training_eligible": True,
                    "training_exclusion_reasons": [],
                    "task_reward": reward,
                },
            ),
        )

    monkeypatch.setattr(runtime_module, "adaptive_result_to_rollout", adapter)
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
            if kwargs["task_id"] == "task-1" and self.seed == 0:
                raise RuntimeError("synthetic isolated missing trajectory")
            return self.delegate.solve(*args, **kwargs)

        def close(self):
            self.delegate.close()

    class Proposer(_MockProposer):
        def propose(self, *args, **kwargs):
            assert (tmp_path / "run/proposer_baseline_snapshot.json").exists()
            p = super().propose(*args, **kwargs)
            return replace(
                p,
                policy_calls=(
                    TokenizedPolicyCall("selection", (1, 2), (0, 1), behavior_log_probs=(-0.5,)),
                ),
            )

    seen_groups = []

    class Runner(SelfPlayRolloutRunner):
        def _collect_frontier_reverification(self, proposals, rollouts):
            result = {}
            for p in proposals:
                group = [r for r in rollouts if r.trajectory.task_id == p.task.task_id]
                seen_groups.append((p.task.task_id, len(group)))
                result[p.task.task_id] = dict(
                    task_id=p.task.task_id,
                    rollout_ids=[r.trajectory.rollout_id for r in group],
                    rewards=[r.trajectory.reward for r in group],
                )
            return result

    def runner():
        return Runner(
            proposer=Proposer(),
            application_factory=Application,
            tokenizer=ByteTokenizer(),
            snapshots=create_selfplay_snapshots(config),
            output_dir=tmp_path / "run",
            config=SelfPlayRunConfig(
                3,
                workers=1,
                task_window=2,
                counterfactuals_per_rollout=0,
                rollout_group_policy="eligible_subset",
                proposer_learning_mode=INDEPENDENT_SCHEMA,
                proposer_baseline_mode=baseline_mode,
                pipeline_frontier_by_dataset=pipeline,
                frontier_reverify_fraction=0.25,
            ),
        )

    result = runner().run(["synthetic first", "synthetic second"])
    assert ("task-1", 2) in seen_groups and ("task-2", 3) in seen_groups
    assert len(result.solver_batch.samples) == 5
    assert len(result.proposer_batch.samples) == 2
    validate_independent_batches(result.proposer_batch, result.solver_batch)
    verify_frozen_selection(tmp_path / "run")
    initial_calls = len(calls)
    baseline_path = (
        tmp_path
        / "run"
        / result.proposer_batch.metadata["proposer_baseline"].get(
            "ema_state_file", "proposer_baseline_state.json"
        )
    )
    before = baseline_path.read_text() if baseline_mode == "ema" else None
    if baseline_mode == "none":
        assert not baseline_path.exists()
        assert all(s.advantage == s.reward for s in result.proposer_batch.samples)
    # Simulate a crash after writing durable batches but before committing the EMA.
    if baseline_mode == "ema":
        baseline_path.unlink()
    restored = runner().run(["synthetic first", "synthetic second"], resume=True)
    assert restored.proposer_batch == result.proposer_batch
    assert restored.solver_batch == result.solver_batch
    assert restored.frontier_scores == result.frontier_scores
    assert len(calls) == initial_calls
    assert (baseline_path.read_text() if baseline_path.exists() else None) == before
    runner().run(["synthetic first", "synthetic second"], resume=True)
    assert (baseline_path.read_text() if baseline_path.exists() else None) == before


def test_disabled_baseline_preserves_history_and_freezes_zero(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import (
        freeze_proposer_baseline,
        NO_BASELINE_VERSION,
    )

    state = tmp_path / "state.json"
    state.write_text('{"historical": true}')
    before = state.read_bytes()
    snapshot, store = freeze_proposer_baseline(
        tmp_path / "snapshot.json", state_path=state, cycle=4, decay=0.9, mode="none"
    )
    assert snapshot["version"] == NO_BASELINE_VERSION
    assert snapshot["values"] == {} and store is None
    assert state.read_bytes() == before
    resumed, store = freeze_proposer_baseline(
        tmp_path / "snapshot.json", state_path=state, cycle=4, decay=0.9, mode="ema"
    )
    assert resumed == snapshot and store is None


def test_disabled_baseline_keeps_legacy_snapshot_on_resume(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    path = tmp_path / "snapshot.json"
    store = DatasetBaselineStore(tmp_path / "state.json", decay=0.9)
    frozen = store.freeze(path, cycle=0)
    resumed, resumed_store = freeze_proposer_baseline(
        path, state_path=tmp_path / "state.json", cycle=0, decay=0.9, mode="none"
    )
    assert resumed == frozen and resumed_store is not None


def test_disabled_baseline_rejects_nonzero_values():
    from selfplay_graph_flowsteer.proposer_learning import (
        validate_baseline_snapshot,
        NO_BASELINE_VERSION,
    )

    with pytest.raises(ValueError, match="empty values"):
        validate_baseline_snapshot(
            dict(version=NO_BASELINE_VERSION, normalization=NORMALIZATION, values={"aime": 0.2})
        )


def test_no_baseline_batch_uses_frontier_directly():
    from selfplay_graph_flowsteer.proposer_learning import NO_BASELINE_VERSION

    proposals, rollouts = records([4, 5])
    result = assemble(
        proposals,
        rollouts,
        dict(
            version=NO_BASELINE_VERSION,
            normalization=NORMALIZATION,
            values={},
            cycle=0,
            mode="none",
        ),
    )
    assert result.proposer_batch.samples
    assert all(s.advantage == s.reward for s in result.proposer_batch.samples)
    assert all(
        s.metadata["proposer_advantage_version"] == NO_BASELINE_VERSION
        for s in result.proposer_batch.samples
    )


def test_defaults_enable_ema_but_explicit_none_still_skips_history(tmp_path):
    from selfplay_graph_flowsteer.cli import build_parser
    from selfplay_graph_flowsteer.selfplay_runtime import SelfPlayRunConfig
    from selfplay_graph_flowsteer.proposer_learning import (
        freeze_proposer_baseline,
        frontier_stability_policy,
    )

    args = build_parser().parse_args(["selfplay-experiment", "--output", str(tmp_path)])
    assert args.proposer_baseline_mode == "ema"
    assert SelfPlayRunConfig().proposer_baseline_mode == "ema"
    state = tmp_path / "existing_ema.json"
    state.write_text("do not read or overwrite this historical state")
    snapshot, store = freeze_proposer_baseline(
        tmp_path / "new_snapshot.json", state_path=state, cycle=0, decay=0.9, mode="none"
    )
    assert store is None and snapshot["values"] == {}
    assert frontier_stability_policy(snapshot)["tie_weight"] == 0.1
    assert state.read_text() == "do not read or overwrite this historical state"


def test_soft_frontier_batch_preserves_solver_and_uses_direct_advantage(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    proposals, rollouts = records([4, 5])
    snapshot, _ = freeze_proposer_baseline(
        tmp_path / "snapshot.json",
        state_path=tmp_path / "ema.json",
        cycle=0,
        decay=0.9,
        mode="none",
    )
    reverify = {
        p.task.task_id: dict(
            rollout_ids=[f"{p.task.task_id}-r{i}" for i in range(5)],
            rewards=[0.0] * 5,
            reward_trusted=[True] * 5,
        )
        for p in proposals
    }
    result = assemble(proposals, rollouts, snapshot, reverify)
    original = assemble(proposals, rollouts)
    assert result.solver_batch.samples == original.solver_batch.samples
    assert result.solver_batch.objective == original.solver_batch.objective
    assert all(
        s.reward == pytest.approx(0.1 * f.provisional_graph_local)
        for s, f in zip(result.proposer_batch.samples, result.frontier_scores)
    )
    assert all(s.advantage == s.reward for s in result.proposer_batch.samples)
    assert any(s.reward > 0 for s in result.proposer_batch.samples)
    # Saved old batches retain the original hard gate even with complete evidence.
    old = assemble(proposals, rollouts, baseline(), reverify)
    assert all(s.reward == 0 for s in old.proposer_batch.samples)
    # Old or untrusted numeric zeros cannot silently become soft tie evidence.
    for value in reverify.values():
        value.pop("reward_trusted")
    unaudited = assemble(proposals, rollouts, snapshot, reverify)
    assert all(s.reward == 0 for s in unaudited.proposer_batch.samples)
    sample = result.proposer_batch.samples[0]
    sample.metadata["frontier"]["stability_policy"] = {"version": "wrong"}
    with pytest.raises(ValueError, match="stability policy"):
        validate_independent_batches(result.proposer_batch, result.solver_batch)


def test_resume_old_no_baseline_does_not_reinterpret_reward_rule(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import (
        NO_BASELINE_VERSION,
        freeze_proposer_baseline,
        frontier_stability_policy,
    )

    path = tmp_path / "snapshot.json"
    old = dict(
        version=NO_BASELINE_VERSION, normalization=NORMALIZATION, cycle=0, values={}, mode="none"
    )
    path.write_text(json.dumps(old))
    before = path.read_text()
    resumed, store = freeze_proposer_baseline(
        path, state_path=tmp_path / "ema.json", cycle=0, decay=0.9, mode="none"
    )
    assert resumed == old and store is None and path.read_text() == before
    assert frontier_stability_policy(resumed)["tie_weight"] == 0


def test_soft_ema_isolates_legacy_history_and_freezes_only_prior_cycles(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    original_path = tmp_path / "history.json"
    legacy = DatasetBaselineStore(original_path)
    old = legacy.freeze(tmp_path / "old.json", cycle=0)
    legacy.commit(old, {"aime": 1.0})
    old_text = original_path.read_text()
    first, store = freeze_proposer_baseline(
        tmp_path / "first.json", state_path=original_path, cycle=4, decay=0.9, mode="ema"
    )
    assert store.path != original_path
    assert first["values"] == {}
    assert first["ema_state_file"] == store.path.name
    assert json.loads(store.path.read_text())["frontier_stability"] == first["frontier_stability"]
    store.commit(first, {"aime": 0.3})
    second, second_store = freeze_proposer_baseline(
        tmp_path / "second.json", state_path=original_path, cycle=5, decay=0.9, mode="ema"
    )
    assert second_store.path == store.path
    assert second["values"]["aime"] == pytest.approx(0.03)
    assert first["values"] == {}  # No look-ahead into this batch's reward.
    second_store.commit(second, {"aime": 0.0})  # A trusted zero counts.
    third, third_store = freeze_proposer_baseline(
        tmp_path / "third.json", state_path=original_path, cycle=6, decay=0.9, mode="ema"
    )
    assert third["values"]["aime"] == pytest.approx(0.027)
    third_store.commit(third, {})  # Missing evidence does not become a zero.
    assert json.loads(store.path.read_text())["values"]["aime"] == pytest.approx(0.027)
    resumed, resumed_store = freeze_proposer_baseline(
        tmp_path / "first.json", state_path=original_path, cycle=4, decay=0.8, mode="none"
    )
    assert resumed == first and resumed_store.path == store.path
    assert resumed_store.decay == 0.9
    assert original_path.read_text() == old_text


@pytest.mark.parametrize("mismatch", ["weight", "version", "missing"])
def test_ema_rejects_incompatible_rule_on_freeze_resume_and_commit(tmp_path, mismatch):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    path = tmp_path / "history.json"
    snapshot_path = tmp_path / "first.json"
    first, store = freeze_proposer_baseline(
        snapshot_path, state_path=path, cycle=0, decay=0.9, mode="ema"
    )
    state = json.loads(store.path.read_text())
    if mismatch == "weight":
        state["frontier_stability"]["tie_weight"] = 0.25
    elif mismatch == "version":
        state["frontier_stability"]["version"] = "unknown_version"
    else:
        state.pop("frontier_stability")
    store.path.write_text(json.dumps(state))
    before = store.path.read_text()
    for candidate, cycle in [(snapshot_path, 0), (tmp_path / "next.json", 1)]:
        with pytest.raises(ValueError, match="stability policy"):
            freeze_proposer_baseline(candidate, state_path=path, cycle=cycle, decay=0.9, mode="ema")
    with pytest.raises(ValueError, match="stability policy"):
        store.commit(first, {"aime": 0.2})
    assert store.path.read_text() == before
    assert not (tmp_path / "next.json").exists()


def test_soft_ema_rejects_legacy_snapshot_even_for_duplicate_commit(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    first, store = freeze_proposer_baseline(
        tmp_path / "snapshot.json",
        state_path=tmp_path / "history.json",
        cycle=0,
        decay=0.9,
        mode="ema",
    )
    store.commit(first, {"aime": 0.3})
    before = store.path.read_text()
    legacy = dict(first)
    legacy.pop("frontier_stability")
    with pytest.raises(ValueError, match="stability policy"):
        store.commit(legacy, {"aime": 0.3})
    assert store.path.read_text() == before
    other_policy = {**first["frontier_stability"], "tie_weight": 0.25}
    with pytest.raises(ValueError, match="stability policy"):
        store.freeze(tmp_path / "snapshot.json", cycle=0, stability_policy=other_policy)


def test_soft_ema_rejects_alpha_changes_in_snapshot_before_training(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    snapshot_path = tmp_path / "snapshot.json"
    frozen, store = freeze_proposer_baseline(
        snapshot_path, state_path=tmp_path / "history.json", cycle=0, decay=0.9, mode="ema"
    )
    changed = {**frozen, "frontier_stability": {**frozen["frontier_stability"], "tie_weight": 0.25}}
    snapshot_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="stability policy"):
        freeze_proposer_baseline(
            snapshot_path, state_path=tmp_path / "history.json", cycle=0, decay=0.9, mode="ema"
        )


def test_soft_frontier_still_gets_negative_advantage_below_matching_ema(tmp_path):
    from selfplay_graph_flowsteer.proposer_learning import freeze_proposer_baseline

    first, store = freeze_proposer_baseline(
        tmp_path / "first.json",
        state_path=tmp_path / "history.json",
        cycle=0,
        decay=0.9,
        mode="ema",
    )
    store.commit(first, {"unknown": 1.0})
    second, _ = freeze_proposer_baseline(
        tmp_path / "second.json",
        state_path=tmp_path / "history.json",
        cycle=1,
        decay=0.9,
        mode="ema",
    )
    proposals, rollouts = records([5])
    task_id = proposals[0].task.task_id
    reverify = {
        task_id: dict(
            rollout_ids=[f"{task_id}-r{i}" for i in range(5)],
            rewards=[0.0] * 5,
            reward_trusted=[True] * 5,
        )
    }
    result = assemble(proposals, rollouts, second, reverify)
    sample = result.proposer_batch.samples[0]
    assert 0 < sample.reward < 0.1
    assert sample.advantage == pytest.approx(sample.reward - 0.1)
    assert sample.advantage < 0
    assert second["values"]["unknown"] == pytest.approx(0.1)
