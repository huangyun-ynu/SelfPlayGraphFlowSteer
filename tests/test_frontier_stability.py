from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.application import GraphEvaluationBackendError
from selfplay_graph_flowsteer.features import GraphFeatures
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.observability import TaskSpec, VerificationResult
from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory
from selfplay_graph_flowsteer.selfplay import (
    AlternatingSnapshots,
    ProposedTask,
    SolverRollout,
    assemble_selfplay_result,
    select_frontier_reverification,
    stable_graph_local_frontier,
)
from selfplay_graph_flowsteer.selfplay_runtime import (
    ByteTokenizer,
    SelfPlayRolloutRunner,
    SelfPlayRunConfig,
    _executor_compatibility_signature,
)


def test_executor_compatibility_ignores_credential_scope_and_judge_wire_surface() -> None:
    old = {
        "healthbench_judge_audit": {"runtime_route": "gpt_judge"},
        "runtime_environment": {"served_model": "worker"},
        "runtime_environments": {
            "gpt": {"served_model": "gpt-5.5", "api_surface": "chat_completions"},
            "gpt_judge": {"served_model": "gpt-5.5", "api_surface": "responses"},
        },
    }
    repaired = {
        "healthbench_judge_audit": {"runtime_route": "gpt_judge"},
        "runtime_environment": {
            "served_model": "worker",
            "dataset_api_key_overrides": [],
        },
        "runtime_environments": {
            "gpt": {
                "served_model": "gpt-5.5",
                "api_surface": "chat_completions",
                "dataset_api_key_overrides": ["healthbench_professional"],
            },
            "gpt_judge": {
                "served_model": "gpt-5.5",
                "api_surface": "chat_completions",
                "dataset_api_key_overrides": [],
            },
        },
    }

    assert _executor_compatibility_signature(old) == _executor_compatibility_signature(repaired)


def test_scheme_d_selects_positive_quota_per_dataset_with_task_id_tiebreak() -> None:
    selected = select_frontier_reverification(
        [
            ("b", "binary", 1.0),
            ("a", "binary", 1.0),
            ("zero", "binary", 0.0),
            ("negative", "binary", -1.0),
            ("h2", "health", 0.2),
            ("h1", "health", 0.3),
        ]
    )
    assert selected == {"binary": ("a",), "health": ("h1",)}


def test_stable_frontier_gates_pairs_independently_without_winner_argmax() -> None:
    features = [GraphFeatures((float(index),), ("x",)) for index in range(3)]
    matrix = (
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    )
    score, records = stable_graph_local_frontier(
        (1.0, 0.0, 0.5),
        (0.8, 0.2, 0.9),
        features,
        kernel_matrix=matrix,
    )
    # 0:1 remains positive: mean delta=(1+.6)/2=.8 => .64.
    # 0:2 flips sign and 1:2 remains negative: mean=(-.5-.7)/2=-.6 => .36.
    assert score == pytest.approx((4.0 / 9.0) * (0.64 + 0.36))
    assert [item["stability_gate_passed"] for item in records] == [True, False, True]


def test_no_positive_frontier_selects_nothing_even_when_quota_is_one() -> None:
    assert select_frontier_reverification([("q", "aime", 0.0)]) == {"aime": ()}


def _graph(agent_count: int) -> MultiAgentGraph:
    graph = MultiAgentGraph()
    for index in range(agent_count):
        agent_id = f"a{index}"
        graph.add_agent(agent_id)
        graph.set_prompt(agent_id, f"role {index}")
    graph.set_output(f"a{agent_count - 1}")
    return graph


def test_runner_reverifies_the_whole_graph_group_with_one_executor_seed(tmp_path: Path) -> None:
    bundle = hashlib.sha256(b"{}").hexdigest()
    proposal = ProposedTask(
        task=TaskSpec(
            "q",
            "question",
            reference="answer",
            task_type="math",
            metadata={"dataset": "aime"},
        ),
        response='{"prompt":"question"}',
        token_ids=(1,),
        action_mask=(1,),
    )
    graphs = [_graph(2), _graph(1)]
    rollouts = [
        SolverRollout(
            TokenizedDirectorTrajectory(
                rollout_id=f"q-r{index}",
                task_id="q",
                token_ids=(1,),
                action_mask=(1,),
                reward=reward,
                graph=graph.to_dict(),
                seed=index,
                metadata={
                    "training_eligible": True,
                    "task_reward": reward,
                    "primary_executor_seed": 123,
                    "executor_bundle_signature": bundle,
                    "mace_window_audit": {
                        "peer_snapshot_hash": None,
                        "model_snapshot_hash": None,
                    },
                },
            ),
            graph,
        )
        for index, (graph, reward) in enumerate(zip(graphs, (1.0, 0.0), strict=True))
    ]
    observed_seeds: list[int] = []
    import threading

    overlap = threading.Barrier(2)

    class FakeApplication:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_manifest=lambda: {})
            self.runtime = SimpleNamespace(seed=None, peer_selector=None)
            self.solver = SimpleNamespace(model_router=None)
            self.mace_selector = None
            self.model_router = None

        def evaluate_graph(self, task, graph, *, seed, return_verification=False):
            del task
            assert return_verification
            observed_seeds.append(seed)
            if len(observed_seeds) <= 2:
                overlap.wait(timeout=3)
            score = 1.0 if len(graph.nodes) == 2 else 0.0
            return {
                "score": score,
                "prediction": "answer",
                "verification": VerificationResult(score, bool(score), "offline"),
            }

        def close(self) -> None:
            return None

    runner = SelfPlayRolloutRunner(
        proposer=SimpleNamespace(),
        application_factory=lambda _seed: FakeApplication(),
        tokenizer=ByteTokenizer(),
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(rollouts_per_task=2, frontier_reverify_fraction=0.25),
    )
    reverified = runner._collect_frontier_reverification([proposal], rollouts)
    result = assemble_selfplay_result(
        [proposal],
        rollouts,
        runner.snapshots,
        expected_rollouts_per_task=2,
        frontier_reverification=reverified,
    )

    assert len(observed_seeds) == 2
    assert len(set(observed_seeds)) == 1
    assert reverified["q"]["persistent_mace_updates"] is False
    assert result.frontier_scores[0].reverify_status == "completed"
    assert result.frontier_scores[0].stable_graph_local == pytest.approx(
        result.frontier_scores[0].provisional_graph_local
    )
    persisted = json.loads((tmp_path / "frontier_reverification.json").read_text())
    assert len(persisted[0]["graph_ids"]) == 2

    partial = [
        json.loads(line)
        for line in (tmp_path / "frontier_reverification_partial.jsonl").read_text().splitlines()
    ]
    assert partial == persisted

    # The numeric Frontier cutoff may be recomputed after durable metadata is
    # merged on resume.  Exact graph, executor, and seed identity is sufficient
    # to reuse a completed evaluation when task membership is unchanged.
    for path in (
        tmp_path / "frontier_reverification_partial.jsonl",
        tmp_path / "frontier_reverification_graph_partial.jsonl",
    ):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["selection_sha256"] = "stale-selection-manifest"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def unexpected_factory(_seed):
        raise AssertionError("a durable Frontier task must not be re-executed")

    resumed = SelfPlayRolloutRunner(
        proposer=SimpleNamespace(),
        application_factory=unexpected_factory,
        tokenizer=ByteTokenizer(),
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(rollouts_per_task=2, frontier_reverify_fraction=0.25),
    )._collect_frontier_reverification([proposal], rollouts)
    assert resumed == reverified

    (tmp_path / "frontier_reverification_partial.jsonl").unlink()
    (tmp_path / "frontier_reverification.json").unlink()
    graph_partial = [
        json.loads(line)
        for line in (tmp_path / "frontier_reverification_graph_partial.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sorted(row["graph_index"] for row in graph_partial) == [0, 1]
    resumed_from_graphs = SelfPlayRolloutRunner(
        proposer=SimpleNamespace(),
        application_factory=unexpected_factory,
        tokenizer=ByteTokenizer(),
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(rollouts_per_task=2, frontier_reverify_fraction=0.25),
    )._collect_frontier_reverification([proposal], rollouts)
    assert resumed_from_graphs == reverified


@pytest.mark.parametrize("failure_kind", ["backend", "token_credit", "missing_verification"])
def test_frontier_backend_failure_excludes_only_the_affected_proposer(
    tmp_path: Path, failure_kind
) -> None:
    bundle = hashlib.sha256(b"{}").hexdigest()
    proposals = [
        ProposedTask(
            task=TaskSpec(
                task_id,
                "question",
                reference="answer",
                task_type="math",
                metadata={"dataset": dataset},
            ),
            response='{"prompt":"question"}',
            token_ids=(1,),
            action_mask=(1,),
        )
        for task_id, dataset in (("bad", "aime"), ("good", "nq_open"))
    ]
    rollouts = []
    for proposal in proposals:
        for index, reward in enumerate((1.0, 0.0)):
            graph = _graph(index + 1)
            rollouts.append(
                SolverRollout(
                    TokenizedDirectorTrajectory(
                        rollout_id=f"{proposal.task.task_id}-r{index}",
                        task_id=proposal.task.task_id,
                        token_ids=(1,),
                        action_mask=(1,),
                        reward=reward,
                        graph=graph.to_dict(),
                        seed=index,
                        metadata={
                            "training_eligible": True,
                            "task_reward": reward,
                            "primary_executor_seed": 123,
                            "executor_bundle_signature": bundle,
                        },
                    ),
                    graph,
                )
            )

    class FakeApplication:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_manifest=lambda: {})
            self.runtime = SimpleNamespace(seed=None, peer_selector=None)
            self.solver = SimpleNamespace(model_router=None)
            self.mace_selector = None
            self.model_router = None

        def evaluate_graph(self, task, graph, *, seed, return_verification=False):
            del seed
            assert return_verification
            if task.task_id == "bad" and len(graph.nodes) == 1:
                if failure_kind == "missing_verification":
                    return {"score": 0.0, "prediction": "", "verification": None}
                if failure_kind == "token_credit":
                    from selfplay_graph_flowsteer.llm import (
                        RequestTokenCredit,
                        RequestTokenCreditExceeded,
                    )

                    raise RequestTokenCreditExceeded(200, RequestTokenCredit(100))
                raise GraphEvaluationBackendError({"routes": ["gpt"]})
            score = float(len(graph.nodes) == 2)
            return {
                "score": score,
                "prediction": "answer",
                "verification": VerificationResult(score, bool(score), "offline"),
            }

        def close(self) -> None:
            return None

    runner = SelfPlayRolloutRunner(
        proposer=SimpleNamespace(),
        application_factory=lambda _seed: FakeApplication(),
        tokenizer=ByteTokenizer(),
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(
            rollouts_per_task=2,
            frontier_reverify_workers=4,
            frontier_reverify_fraction=1.0,
        ),
    )

    reverified = runner._collect_frontier_reverification(proposals, rollouts)

    assert set(reverified) == {"good"}
    assert proposals[0].metadata["frontier_training_exclusion"] == (
        "frontier_reverify_infrastructure_failure"
    )
    failures = json.loads((tmp_path / "frontier_reverification_failures.json").read_text())
    assert [(row["task_id"], row["graph_index"]) for row in failures] == [("bad", 0)]
    assert reverified["good"]["reward_trusted"] == [True, True]


def test_runner_reverifies_selected_tasks_in_one_global_graph_pool(tmp_path: Path) -> None:
    import threading

    bundle = hashlib.sha256(b"{}").hexdigest()
    proposals = [
        ProposedTask(
            task=TaskSpec(
                task_id,
                "question",
                reference="answer",
                task_type="math",
                metadata={"dataset": dataset},
            ),
            response='{"prompt":"question"}',
            token_ids=(1,),
            action_mask=(1,),
        )
        for task_id, dataset in (("q1", "aime"), ("q2", "nq_open"))
    ]
    rollouts = []
    for proposal in proposals:
        for index, reward in enumerate((1.0, 0.0)):
            graph = _graph(index + 1)
            rollouts.append(
                SolverRollout(
                    TokenizedDirectorTrajectory(
                        rollout_id=f"{proposal.task.task_id}-r{index}",
                        task_id=proposal.task.task_id,
                        token_ids=(1,),
                        action_mask=(1,),
                        reward=reward,
                        graph=graph.to_dict(),
                        seed=index,
                        metadata={
                            "training_eligible": True,
                            "task_reward": reward,
                            "primary_executor_seed": 123,
                            "executor_bundle_signature": bundle,
                        },
                    ),
                    graph,
                )
            )

    all_graphs_started = threading.Barrier(4)

    class FakeApplication:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_manifest=lambda: {})
            self.runtime = SimpleNamespace(seed=None, peer_selector=None)
            self.solver = SimpleNamespace(model_router=None)
            self.mace_selector = None
            self.model_router = None

        def evaluate_graph(self, task, graph, *, seed, return_verification=False):
            del task, seed
            assert return_verification
            all_graphs_started.wait(timeout=3)
            score = float(len(graph.nodes) == 2)
            return {
                "score": score,
                "prediction": "answer",
                "verification": VerificationResult(score, bool(score), "offline"),
            }

        def close(self) -> None:
            return None

    runner = SelfPlayRolloutRunner(
        proposer=SimpleNamespace(),
        application_factory=lambda _seed: FakeApplication(),
        tokenizer=ByteTokenizer(),
        snapshots=AlternatingSnapshots("p", "s"),
        output_dir=tmp_path,
        config=SelfPlayRunConfig(
            rollouts_per_task=2,
            frontier_reverify_workers=4,
            frontier_reverify_fraction=1.0,
        ),
    )
    reverified = runner._collect_frontier_reverification(proposals, rollouts)

    assert set(reverified) == {"q1", "q2"}
    metrics = json.loads((tmp_path / "frontier_phase_metrics.json").read_text())
    assert metrics["scheduling"] == "global_graph_pool"
    assert metrics["workers"] == 4


@pytest.mark.parametrize(
    "primary,second,expected,softened",
    [
        ([1, 0], [1, 0], 1.0, False),
        ([1, 0], [0, 0], 0.1, True),
        ([1, 0], [1, 1], 0.1, True),
        ([0, 1], [0.5, 0.5], 0.1, True),
        ([1, 0], [0, 1], 0.0, False),
        ([0, 0], [1, 0], 0.0, False),
        ([0, 0], [0, 0], 0.0, False),
    ],
)
def test_soft_frontier_only_discounts_reverify_ties(primary, second, expected, softened):
    features = [GraphFeatures((1.0,), ("x",))] * 2
    score, records = stable_graph_local_frontier(
        primary,
        second,
        features,
        kernel_matrix=((1.0, 1.0), (1.0, 1.0)),
        reverify_trusted=[True, True],
    )
    assert score == pytest.approx(expected)
    assert records[0]["tie_softened"] is softened


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.25])
def test_tie_discount_preserves_all_pair_normalization(weight):
    features = [GraphFeatures((1.0,), ("x",))] * 3
    score, records = stable_graph_local_frontier(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        features,
        tie_weight=weight,
        normalization="pair_mean_ref5_v1",
    )
    assert score == pytest.approx(1.6 / 3 * (2 * weight))
    assert len(records) == 3
    assert records[2]["stable_pair_contribution"] == 0


@pytest.mark.parametrize("trusted", [[False, True], [True, False], [False, False]])
def test_unaudited_reverify_zero_does_not_receive_soft_credit(trusted):
    features = [GraphFeatures((1.0,), ("x",))] * 2
    score, records = stable_graph_local_frontier([1, 0], [0, 0], features, reverify_trusted=trusted)
    assert score == 0 and not records[0]["tie_softened"]


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_soft_frontier_rejects_invalid_discount(weight):
    with pytest.raises(ValueError, match="tie_weight"):
        stable_graph_local_frontier([], [], [], tie_weight=weight)
