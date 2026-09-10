from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory
from selfplay_graph_flowsteer.selfplay import (
    AlternatingSnapshots,
    DryRunSelfPlayCoordinator,
    ProposedTask,
    QwenTaskProposer,
    SelfPlayPhase,
    SelfPlaySeed,
    SolverRollout,
    assemble_selfplay_result,
    graph_local_frontier,
    load_selfplay_seed_jsonl,
    scalar_frontier,
)


class FixedProposer:
    def propose(self, seed: str, *, task_id: str) -> ProposedTask:
        return ProposedTask(
            TaskSpec(task_id, f"question {seed}", reference="answer"),
            response=seed,
            token_ids=(1,),
            action_mask=(1,),
        )


def solve(task: TaskSpec, index: int) -> SolverRollout:
    graph = MultiAgentGraph()
    graph.add_agent("solver")
    graph.set_prompt("solver", "solve")
    graph.set_output("solver")
    trajectory = TokenizedDirectorTrajectory(
        rollout_id=f"{task.task_id}-{index}",
        task_id=task.task_id,
        token_ids=(index + 1,),
        action_mask=(1,),
        reward=float(index % 2),
        graph=graph.to_dict(),
        seed=index,
    )
    return SolverRollout(trajectory, graph)


def test_frontier_reduces_to_four_p_one_minus_p() -> None:
    rewards = [0.0, 0.0, 1.0, 1.0]
    assert scalar_frontier(rewards) == pytest.approx(1.0)
    from selfplay_graph_flowsteer.features import structural_features

    graph = solve(TaskSpec("t", "q"), 0).graph
    features = [structural_features(graph)] * 4
    assert graph_local_frontier(rewards, features) == pytest.approx(1.0)


def test_answer_seed_without_hops_preserves_reference_and_metadata() -> None:
    seed = SelfPlaySeed(
        "zero",
        target_answer=0,
        metadata={"dataset": "aime", "task_type": "math", "verifier": "numeric"},
    )
    backend = MockBackend([json.dumps({"prompt": "What is one minus one?", "task_type": "math"})])

    proposal = QwenTaskProposer(backend).propose(seed, task_id="aime-zero")

    assert proposal.task.reference == 0
    assert proposal.task.metadata["dataset"] == "aime"
    assert proposal.task.metadata["verifier"] == "numeric"


def test_sesa_seed_preserves_hops_without_constraining_workflow_depth(tmp_path) -> None:
    path = tmp_path / "seeds.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "ssp-1",
                "ground_truth": "Betty Rubble",
                "search_turns": 3,
                "metadata": {"source": "ssp-50k"},
            }
        )
        + "\n"
    )
    seed = load_selfplay_seed_jsonl(path)[0]
    assert seed.target_answer == "Betty Rubble"
    assert seed.required_reasoning_hops == 3

    backend = MockBackend(
        [
            json.dumps(
                {
                    "prompt": "Which adoptive mother is identified by these linked clues?",
                    "task_type": "open_qa",
                    "required_reasoning_hops": 3,
                    "evidence_chain": ["studio to series", "series to family", "family to mother"],
                }
            )
        ]
    )
    proposal = QwenTaskProposer(backend).propose(seed, task_id="t-hop")
    assert proposal.task.reference == "Betty Rubble"
    assert proposal.task.metadata["required_reasoning_hops"] == 3
    assert "workflow" not in proposal.task.prompt.casefold()
    user_payload = json.loads(backend.calls[0]["messages"][1]["content"])
    assert user_payload["required_reasoning_hops"] == 3


@pytest.mark.parametrize(
    "payload,error",
    [
        (
            {
                "prompt": "A valid indirect question",
                "required_reasoning_hops": 2,
                "evidence_chain": ["only one"],
            },
            "evidence_chain",
        ),
        (
            {
                "prompt": "The answer is Betty Rubble",
                "required_reasoning_hops": 2,
                "evidence_chain": ["first", "second"],
            },
            "leaks",
        ),
    ],
)
def test_hop_conditioned_proposer_rejects_invalid_generation(payload, error) -> None:
    seed = SelfPlaySeed("Betty Rubble", target_answer="Betty Rubble", required_reasoning_hops=2)
    with pytest.raises(ValueError, match=error):
        QwenTaskProposer(MockBackend([json.dumps(payload)])).propose(seed, task_id="invalid")


def test_dry_run_builds_both_batches_without_training() -> None:
    coordinator = DryRunSelfPlayCoordinator(
        proposer=FixedProposer(), solve=solve, rollouts_per_task=4
    )
    result = coordinator.run(["a", "b"])
    assert len(result.tasks) == 2
    assert len(result.solver_batch.samples) == 8
    assert len(result.proposer_batch.samples) == 2
    assert result.optimizer_steps == 0
    assert result.solver_batch.optimizer_steps == 0
    assert coordinator.snapshots.phase is SelfPlayPhase.BATCH_READY


def test_batch_assembly_rejects_explicitly_ineligible_rollout() -> None:
    proposal = FixedProposer().propose("a", task_id="task-1")
    rollout = solve(proposal.task, 0)
    rollout.trajectory.metadata["training_eligible"] = False

    with pytest.raises(ValueError, match="training-ineligible"):
        assemble_selfplay_result(
            [proposal],
            [rollout],
            AlternatingSnapshots("proposer-0", "solver-0"),
        )


def test_evaluation_allows_fixed_pool_rows_without_training_attestation() -> None:
    proposal = FixedProposer().propose("a", task_id="task-1")
    proposal.metadata["pool_id"] = "heldout:1"
    rollouts = [solve(proposal.task, index) for index in range(2)]

    with pytest.raises(ValueError, match="task-pool attestation"):
        assemble_selfplay_result([proposal], rollouts, AlternatingSnapshots("p", "s"))

    result = assemble_selfplay_result(
        [proposal],
        rollouts,
        AlternatingSnapshots("p", "s"),
        evaluation_only=True,
    )
    assert len(result.solver_batch.samples) == 2


def test_canary_frontier_exclusion_does_not_remove_solver_or_fake_proposer_zero():
    proposals = [FixedProposer().propose("a", task_id=f"task-{i}") for i in range(2)]
    for proposal in proposals:
        proposal.metadata["seed_group"] = "shared"
    proposals[0].metadata["frontier_training_exclusion"] = "canary_executor_migration"
    rollouts = [solve(p.task, i) for p in proposals for i in range(2)]
    result = assemble_selfplay_result(proposals, rollouts, AlternatingSnapshots("p", "s"))
    assert len(result.solver_batch.samples) == 4
    assert len(result.proposer_batch.samples) == 1
    assert result.proposer_batch.samples[0].task_id == "task-1"
    assert result.proposer_batch.samples[0].advantage > 0
    assert result.frontier_scores[0].reverify_status == "excluded_executor_migration"


def test_snapshot_state_can_be_restored(tmp_path) -> None:
    snapshots = AlternatingSnapshots("p-3", "s-2")
    snapshots.advance()
    path = tmp_path / "snapshots.json"
    snapshots.save(path)
    restored = AlternatingSnapshots.load(path)
    assert restored.to_dict() == snapshots.to_dict()
    assert restored.frozen_role == "proposer"
