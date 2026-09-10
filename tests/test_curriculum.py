from __future__ import annotations

import json
from pathlib import Path

from selfplay_graph_flowsteer.curriculum import (
    ADSBoundaryScheduler,
    CurriculumProfile,
    FixedTaskPool,
    TSDSRetriever,
)
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.selfplay import FixedPoolQwenProposer, SelfPlaySeed
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer


def _write_pool(tmp_path, rows):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return FixedTaskPool.from_jsonl([path])


def test_curriculum_profiles_keep_joint_and_formal_regimes() -> None:
    root = Path(__file__).resolve().parents[1]
    joint = CurriculumProfile.from_toml(root / "configs/curriculum/joint_1500.toml")
    formal = CurriculumProfile.from_toml(root / "configs/curriculum/formal_3500.toml")

    assert (joint.expected_datasets, joint.tasks_per_cycle, joint.task_window) == (3, 48, 4)
    assert (formal.expected_datasets, formal.tasks_per_cycle, formal.task_window) == (7, 14, 14)
    assert joint.mini_cluster_size == formal.mini_cluster_size == 8
    assert joint.rollouts_per_task == formal.rollouts_per_task == 5
    assert joint.rollout_workers == 20
    assert formal.rollout_workers == 64


def test_director_v21_paired_profile_matches_historical_collection_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    paired = CurriculumProfile.from_toml(
        root / "configs/curriculum/seven_dataset_v21_paired_10x5.toml"
    )

    assert paired.name == "seven_dataset_v21_paired_10x5"
    assert (paired.expected_datasets, paired.tasks_per_cycle) == (7, 35)
    assert (paired.task_window, paired.rollout_workers) == (8, 32)
    assert paired.rollouts_per_task == 5


def test_balanced_seeds_and_candidates_are_dataset_scoped(tmp_path) -> None:
    rows = [
        {
            "id": f"{dataset}:{index}",
            "dataset": dataset,
            "split": "train",
            "prompt": f"{dataset} question {index}",
            "answer": str(index),
            "cluster_id": f"{dataset}:cluster:{index % 2}",
            "difficulty_score": float(index),
            "embedding": [float(index), float(dataset == "b")],
        }
        for dataset in ("a", "b", "c")
        for index in range(4)
    ]
    pool = _write_pool(tmp_path, rows)
    seeds = pool.balanced_selection_seeds(6)
    assert [seed.metadata["dataset"] for seed in seeds] == ["a", "b", "c", "a", "b", "c"]
    second = pool.balanced_selection_seeds(6, offset_per_dataset=2)
    assert {seed.seed_id for seed in seeds}.isdisjoint(seed.seed_id for seed in second)

    scheduler = ADSBoundaryScheduler(pool, active_clusters=2, mini_cluster_size=2, cooldown=0)
    candidate_ids = scheduler.candidates(dataset="b")
    assert candidate_ids
    assert {pool.tasks[pool_id].dataset for pool_id in candidate_ids} == {"b"}


def test_balanced_seeds_do_not_count_duplicate_source_rows_as_distinct_tasks(tmp_path) -> None:
    rows = [
        {
            "id": "swe:instance-1",
            "dataset": "swe_bench",
            "split": "train",
            "prompt": f"replica {index}",
            "cluster_id": "swe:cluster:0",
            "difficulty_score": float(index),
            "embedding": [float(index), 1.0],
            "instance_id": "django__django-10000",
            "repo": "django/django",
            "base_commit": "0" * 40,
        }
        for index in range(2)
    ]
    rows.append(
        {
            "id": "swe:instance-2",
            "dataset": "swe_bench",
            "split": "train",
            "prompt": "different instance",
            "cluster_id": "swe:cluster:0",
            "difficulty_score": 2.0,
            "embedding": [2.0, 1.0],
            "instance_id": "django__django-10001",
            "repo": "django/django",
            "base_commit": "1" * 40,
        }
    )
    pool = _write_pool(tmp_path, rows)

    seeds = pool.balanced_selection_seeds(2)

    assert [seed.seed_id for seed in seeds] == ["swe:instance-1", "swe:instance-2"]


def test_ads_batch_update_aggregates_cluster_reward_and_preserves_dataset_probabilities(
    tmp_path,
) -> None:
    rows = [
        {
            "id": f"{dataset}:{cluster}:{index}",
            "dataset": dataset,
            "split": "train",
            "prompt": f"{dataset} {cluster} {index}",
            "answer": str(index),
            "cluster_id": f"{dataset}:{cluster}",
            "difficulty_score": float(index),
            "embedding": [float(cluster), float(index)],
        }
        for dataset in ("a", "b")
        for cluster in range(2)
        for index in range(3)
    ]
    pool = _write_pool(tmp_path, rows)
    scheduler = ADSBoundaryScheduler(pool, active_clusters=2, mini_cluster_size=1, cooldown=0)
    scheduler.reserve("a:0:0")
    assert "a:0:0" not in scheduler.candidates(dataset="a")

    scheduler.record_batch([("a:0:0", [1, 1]), ("a:0:1", [0, 0])])

    assert scheduler.cluster_states["a:0"].success_rate == 0.5
    assert (
        sum(scheduler.cluster_states[cid].prob for cid in scheduler.dataset_cluster_ids["a"]) == 1
    )
    assert (
        sum(scheduler.cluster_states[cid].prob for cid in scheduler.dataset_cluster_ids["b"]) == 1
    )
    assert "a:0:0" not in scheduler.reserved


def test_ads_cooldown_and_reservation_apply_to_repeated_source_rows(tmp_path) -> None:
    rows = [
        {
            "id": "swe:instance-1",
            "dataset": "replica_test",
            "split": "train",
            "prompt": f"replica {index}",
            "cluster_id": "swe:cluster:0",
            "difficulty_score": float(index),
            "embedding": [float(index), 1.0],
        }
        for index in range(2)
    ]
    rows.append(
        {
            "id": "swe:instance-2",
            "dataset": "replica_test",
            "split": "train",
            "prompt": "different instance",
            "cluster_id": "swe:cluster:0",
            "difficulty_score": 2.0,
            "embedding": [2.0, 1.0],
        }
    )
    pool = _write_pool(tmp_path, rows)
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=3, cooldown=10)

    scheduler.reserve("swe:instance-1")
    assert "swe:instance-1" not in scheduler.candidates(dataset="replica_test")
    assert "swe:instance-1#row-1" not in scheduler.candidates(dataset="replica_test")
    scheduler.record("swe:instance-1", [1.0])

    candidates = scheduler.candidates(dataset="replica_test")
    assert candidates == ["swe:instance-2"]


def test_fixed_pool_uses_official_prompt_and_removes_hop_condition(tmp_path) -> None:
    pool = _write_pool(
        tmp_path,
        [
            {
                "id": "nq:1",
                "dataset": "nq",
                "split": "train",
                "seed": "answer seed",
                "prompt": "the official question",
                "target_answers": ["answer"],
                "required_reasoning_hops": 3,
                "verifier": "multi_answer_exact_match",
                "embedding": [1.0, 0.0],
            },
            {
                "id": "nq:2",
                "dataset": "nq",
                "split": "train",
                "prompt": "another official question",
                "target_answers": ["second"],
                "verifier": "multi_answer_exact_match",
                "embedding": [0.9, 0.1],
            },
        ],
    )
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=2, cooldown=0)
    retriever = TSDSRetriever(pool, max_k=2, kde_k=2, sigma=0)
    backend = MockBackend([json.dumps({"candidate_id": "nq:2"})])
    proposer = FixedPoolQwenProposer(
        backend, pool, scheduler, retriever, ByteTokenizer(), candidate_count=2
    )

    proposal = proposer.propose(
        SelfPlaySeed("anchor", seed_id="nq:1", metadata={"dataset": "nq"}),
        task_id="task-1",
    )

    assert proposal.task.prompt == "another official question"
    assert proposal.task.reference == "second"
    assert proposal.task.metadata["required_reasoning_hops"] is None
    assert proposal.metadata["pool_id"] == "nq:2"
    assert backend.calls[0]["role"] == "proposer"


def test_fixed_pool_preserves_structured_conversation_metadata(tmp_path) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Initial medical question"},
            {"role": "assistant", "content": "Earlier clinical response"},
            {"role": "user", "content": "Clinical follow-up"},
        ]
    }
    pool = _write_pool(
        tmp_path,
        [
            {
                "id": "healthbench_professional:1",
                "dataset": "healthbench_professional",
                "split": "train",
                "prompt": "flattened compatibility prompt",
                "target_answers": [],
                "metadata": {"conversation": conversation},
                "embedding": [1.0, 0.0],
            }
        ],
    )

    task = pool.tasks[pool.ids[0]].task
    assert task.prompt == "flattened compatibility prompt"
    assert task.metadata["conversation"] == conversation


def test_fixed_pool_retries_an_invalid_candidate_with_the_allowed_ids(tmp_path) -> None:
    pool = _write_pool(
        tmp_path,
        [
            {
                "id": f"nq:{index}",
                "dataset": "nq",
                "split": "train",
                "prompt": f"question {index}",
                "target_answers": [str(index)],
                "embedding": [float(index), 1.0],
            }
            for index in range(3)
        ],
    )
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=2, cooldown=0)
    retriever = TSDSRetriever(pool, max_k=3, kde_k=2, sigma=0)

    def handler(messages, _role):
        if len(messages) == 2:
            return json.dumps({"candidate_id": "hallucinated:id"})
        recovery = json.loads(messages[-1]["content"])
        return json.dumps({"candidate_id": recovery["allowed_candidate_ids"][0]})

    backend = MockBackend(handler=handler)

    proposal = FixedPoolQwenProposer(
        backend, pool, scheduler, retriever, ByteTokenizer(), candidate_count=3
    ).propose("anchor", task_id="task-1")

    recovery = json.loads(backend.calls[1]["messages"][-1]["content"])
    assert proposal.metadata["pool_id"] in recovery["allowed_candidate_ids"]
    assert backend.calls[1]["temperature"] == 0.0
    assert backend.calls[1]["max_tokens"] == 256


def test_fixed_pool_can_select_anchor_when_it_is_the_only_boundary_item(tmp_path) -> None:
    pool = _write_pool(
        tmp_path,
        [
            {
                "id": f"health:{index}",
                "dataset": "health",
                "split": "train",
                "prompt": f"question {index}",
                "target_answers": [str(index)],
                "cluster_id": f"health:singleton:{index}",
                "cluster_size": 1,
                "rank_in_cluster": 0,
                "embedding": [float(index), 1.0],
            }
            for index in range(2)
        ],
    )
    scheduler = ADSBoundaryScheduler(
        pool,
        active_clusters=2,
        mini_cluster_size=1,
        cooldown=64,
    )
    scheduler.record("health:0", [1.0])
    retriever = TSDSRetriever(pool, max_k=2, kde_k=2, sigma=0)
    backend = MockBackend([json.dumps({"candidate_id": "health:1"})])

    proposal = FixedPoolQwenProposer(
        backend,
        pool,
        scheduler,
        retriever,
        ByteTokenizer(),
        candidate_count=2,
    ).propose(
        SelfPlaySeed("anchor", seed_id="health:1", metadata={"dataset": "health"}),
        task_id="task-2",
    )

    assert proposal.metadata["pool_id"] == "health:1"
    offered = json.loads(backend.calls[0]["messages"][-1]["content"])["candidates"]
    assert [item["candidate_id"] for item in offered] == ["health:1"]


def test_ads_boundary_moves_to_adjacent_difficulty(tmp_path) -> None:
    rows = [
        {
            "id": f"math:{index}",
            "dataset": "math",
            "split": "train",
            "prompt": f"problem {index}",
            "answer": index,
            "difficulty_score": float(index),
            "embedding": [float(index), 1.0],
        }
        for index in range(3)
    ]
    pool = _write_pool(tmp_path, rows)
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=1, cooldown=0)

    assert scheduler.candidates() == ["math:0"]
    scheduler.record("math:0", [1, 1, 1, 1, 1])
    assert scheduler.candidates() == ["math:1"]
    scheduler.record("math:1", [0, 0, 0, 0, 0])
    assert scheduler.candidates() == ["math:0"]


def test_tsds_returns_only_ads_candidates(tmp_path) -> None:
    pool = _write_pool(
        tmp_path,
        [
            {
                "id": f"task:{index}",
                "dataset": "qa",
                "split": "train",
                "prompt": str(index),
                "answer": str(index),
                "embedding": [float(index), 0.0],
            }
            for index in range(4)
        ],
    )
    retriever = TSDSRetriever(pool, max_k=3, kde_k=2, sigma=0)
    ranked = retriever.rank(
        ["task:0", "task:1", "task:2"],
        [[0.1, 0.0]],
        limit=2,
    )
    assert len(ranked) == 2
    assert set(ranked) <= {"task:0", "task:1", "task:2"}
    assert "task:0" in ranked


def test_proposer_receives_new_tsds_neighbor_beyond_boundary_mini_cluster(tmp_path) -> None:
    pool = _write_pool(
        tmp_path,
        [
            {
                "id": f"qa:{index}",
                "dataset": "qa",
                "split": "train",
                "prompt": f"question {index}",
                "answer": str(index),
                "difficulty_score": index,
                "embedding": [float(index), 0.0],
            }
            for index in range(4)
        ],
    )
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=1, cooldown=0)
    retriever = TSDSRetriever(pool, max_k=4, kde_k=2, sigma=0)

    def handler(messages, _role):
        candidates = json.loads(messages[-1]["content"])["candidates"]
        assert {item["candidate_id"] for item in candidates} > {"qa:0"}
        return json.dumps({"candidate_id": candidates[-1]["candidate_id"]})

    proposal = FixedPoolQwenProposer(
        MockBackend(handler=handler),
        pool,
        scheduler,
        retriever,
        candidate_count=3,
    ).propose("new anchor", task_id="selected")

    assert proposal.metadata["ads_boundary_ids"] == ["qa:0"]
    assert proposal.metadata["pool_id"] != "qa:0"


def test_pool_rejects_test_split(tmp_path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "held-out",
                "split": "test",
                "prompt": "must not train",
                "answer": "x",
                "embedding": [1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        FixedTaskPool.from_jsonl([path])
    except ValueError as exc:
        assert "train split" in str(exc)
    else:
        raise AssertionError("test split entered the training pool")
