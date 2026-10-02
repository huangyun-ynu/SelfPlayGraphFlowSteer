# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TSDS probability assignment: Copyright (c) 2024 Zifan Liu, MIT License.
# The full MIT text is retained at ``third_party/TSDS/LICENSE``.
"""Fixed-pool curriculum adapted from ADS and TSDS.

The ADS cluster state, boundary band, neighbour movement and smoothed cluster
distribution are migrated from ``third_party/ADS``.  The TSDS KDE-weighted
probability assignment is migrated from ``third_party/TSDS``.  FAISS is
replaced by exact NumPy distances because the local pools contain only hundreds
of tasks per dataset.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .observability import TaskSpec
from .swebench import sanitize_swe_pool_row


class TextEmbedder(Protocol):
    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]: ...


@dataclass(frozen=True)
class CurriculumProfile:
    """Runtime sizing for one fixed-pool ADS+TSDS experiment regime."""

    name: str
    expected_datasets: int
    tasks_per_cycle: int
    num_clusters_per_dataset: int
    active_clusters_per_dataset: int
    mini_cluster_size: int
    candidate_count: int
    task_window: int
    rollout_workers: int
    rollouts_per_task: int
    boundary_eps: float = 0.17
    alpha: float = 0.3
    cooldown_per_dataset: int = 64
    nominal_epoch_cycles: int = 0

    @classmethod
    def from_toml(cls, path: str | Path) -> CurriculumProfile:
        with Path(path).open("rb") as handle:
            payload = tomllib.load(handle)
        values = payload.get("curriculum", payload)
        if not isinstance(values, dict):
            raise ValueError("curriculum profile must contain a [curriculum] table")
        profile = cls(
            name=str(values.get("name", Path(path).stem)),
            expected_datasets=int(values["expected_datasets"]),
            tasks_per_cycle=int(values["tasks_per_cycle"]),
            num_clusters_per_dataset=int(values["num_clusters_per_dataset"]),
            active_clusters_per_dataset=int(values["active_clusters_per_dataset"]),
            mini_cluster_size=int(values["mini_cluster_size"]),
            candidate_count=int(values["candidate_count"]),
            task_window=int(values["task_window"]),
            rollout_workers=int(values["rollout_workers"]),
            rollouts_per_task=int(values["rollouts_per_task"]),
            boundary_eps=float(values.get("boundary_eps", 0.17)),
            alpha=float(values.get("alpha", 0.3)),
            cooldown_per_dataset=int(values.get("cooldown_per_dataset", 64)),
            nominal_epoch_cycles=int(values.get("nominal_epoch_cycles", 0)),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        positive = (
            self.expected_datasets,
            self.tasks_per_cycle,
            self.num_clusters_per_dataset,
            self.active_clusters_per_dataset,
            self.mini_cluster_size,
            self.candidate_count,
            self.task_window,
            self.rollout_workers,
            self.rollouts_per_task,
        )
        if min(positive) <= 0:
            raise ValueError("curriculum profile sizes must be positive")
        if not 0.0 < self.boundary_eps < 0.5:
            raise ValueError("curriculum boundary_eps must be in (0, 0.5)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("curriculum alpha must be in (0, 1]")
        if self.cooldown_per_dataset < 0 or self.nominal_epoch_cycles < 0:
            raise ValueError("curriculum cooldown/epoch values cannot be negative")


@dataclass(frozen=True)
class PoolTask:
    pool_id: str
    task: TaskSpec
    dataset: str
    cluster_id: str
    difficulty_score: float
    embedding: tuple[float, ...]


class FixedTaskPool:
    """Trusted train-only task pool; the Proposer can select but cannot edit it."""

    def __init__(self, tasks: Iterable[PoolTask]) -> None:
        items = list(tasks)
        if not items:
            raise ValueError("fixed task pool cannot be empty")
        ids = [item.pool_id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("fixed task pool ids must be unique")
        dimensions = {len(item.embedding) for item in items}
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError("all fixed-pool tasks need equal non-empty embeddings")
        self.tasks = {item.pool_id: item for item in items}
        self.ids = tuple(ids)

    @classmethod
    def from_jsonl(
        cls,
        paths: Iterable[str | Path],
        *,
        embedder: TextEmbedder | None = None,
        require_ads_metadata: bool = False,
        require_validation_manifest: bool = False,
    ) -> FixedTaskPool:
        rows: list[dict[str, Any]] = []
        for raw_path in paths:
            path = Path(raw_path)
            attestation: dict[str, Any] = {}
            if require_validation_manifest:
                manifest_path = path.with_name(f"{path.stem}_manifest.json")
                if not manifest_path.exists():
                    raise ValueError(f"validated task-pool manifest is missing: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("validator_version") not in {
                    "fixed_task_pool_validator_v1",
                    "fixed_task_pool_validator_v2",
                }:
                    raise ValueError("fixed task-pool validator version mismatch")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != manifest.get("validated_pool_sha256"):
                    raise ValueError("validated task-pool hash does not match its manifest")
                attestation = {
                    "validated_pool_entry": True,
                    "validated_pool_version": manifest["validator_version"],
                    "validated_pool_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "validated_pool_sha256": digest,
                    "verifier_contract_version": manifest.get("verifier_contract_version"),
                    "adapter_contract_version": manifest.get("adapter_contract_version"),
                }
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["__validation_attestation__"] = dict(attestation)
                    rows.append(row)
        if any(str(row.get("split", "train")).casefold() != "train" for row in rows):
            raise ValueError("fixed task pool accepts train split rows only")
        if require_ads_metadata:
            required = {
                "embedding",
                "cluster_id",
                "difficulty_score",
                "rank_in_cluster",
                "cluster_size",
            }
            for index, row in enumerate(rows):
                missing = sorted(required - row.keys())
                if missing:
                    raise ValueError(
                        "formal ADS pool row "
                        f"{row.get('id', index)!r} is missing {', '.join(missing)}; "
                        "run `spgfs prepare-ads-pool` first"
                    )
            grouped_ads_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped_ads_rows[str(row["cluster_id"])].append(row)
            for cluster_id, members in grouped_ads_rows.items():
                declared_sizes = {int(row["cluster_size"]) for row in members}
                ranks = sorted(int(row["rank_in_cluster"]) for row in members)
                expected_ranks = list(range(len(members)))
                if declared_sizes != {len(members)} or ranks != expected_ranks:
                    raise ValueError(
                        f"inconsistent ADS cluster metadata for {cluster_id!r}: "
                        "cluster_size must match the loaded cluster and ranks must be contiguous"
                    )

        prompts = [_task_prompt(row) for row in rows]
        missing_embeddings = ["embedding" not in row for row in rows]
        if any(missing_embeddings):
            if embedder is None:
                raise ValueError("task rows without embedding require a text embedder")
            encoded = embedder.encode(prompts, query=False)
        else:
            encoded = [tuple(float(value) for value in row["embedding"]) for row in rows]

        occurrences: Counter[str] = Counter()
        tasks: list[PoolTask] = []
        for index, (row, prompt, vector) in enumerate(zip(rows, prompts, encoded, strict=True)):
            source_id = str(row.get("id", row.get("source_id", index)))
            duplicate_index = occurrences[source_id]
            occurrences[source_id] += 1
            pool_id = source_id if duplicate_index == 0 else f"{source_id}#row-{duplicate_index}"
            metadata = {
                **dict(row.get("metadata") or {}),
                **dict(row.get("__validation_attestation__") or {}),
                **{
                    key: row[key]
                    for key in (
                        "dataset",
                        "split",
                        "mode",
                        "task_type",
                        "verifier",
                        "context_documents",
                        "cluster_id",
                        "difficulty_score",
                        "rank_in_cluster",
                        "cluster_size",
                        "ads_sample_id",
                    )
                    if key in row
                },
            }
            dataset = str(row.get("dataset", metadata.get("dataset", "unknown")))
            if dataset.strip().casefold() in {"swe_bench", "swe-bench", "swebench"}:
                swe_row = dict(row)
                swe_row["metadata"] = {
                    **dict(row.get("metadata") or {}),
                    **dict(row.get("__validation_attestation__") or {}),
                }
                metadata = sanitize_swe_pool_row(swe_row)
            cluster_id = str(row.get("cluster_id", metadata.get("cluster_id", dataset)))
            difficulty = float(
                row.get(
                    "difficulty_score",
                    metadata.get("difficulty_score", row.get("rank_in_cluster", index)),
                )
            )
            rank_in_cluster = int(row.get("rank_in_cluster", index))
            cluster_size = int(row.get("cluster_size", len(rows)))
            vector_tuple = tuple(float(value) for value in vector)
            if require_ads_metadata and (
                not np.isfinite(difficulty)
                or rank_in_cluster < 0
                or cluster_size <= rank_in_cluster
                or not np.isfinite(np.asarray(vector_tuple, dtype=np.float64)).all()
            ):
                raise ValueError(f"invalid ADS metadata for fixed-pool row {source_id!r}")
            reference = (
                None
                if dataset.strip().casefold() in {"swe_bench", "swe-bench", "swebench"}
                else _task_reference(row)
            )
            tasks.append(
                PoolTask(
                    pool_id=pool_id,
                    task=TaskSpec(
                        task_id=pool_id,
                        prompt=prompt,
                        reference=reference,
                        task_type=str(row.get("task_type", metadata.get("task_type", "general"))),
                        metadata={
                            **metadata,
                            "pool_id": pool_id,
                            "source_task_id": source_id,
                            "required_reasoning_hops": None,
                        },
                    ),
                    dataset=dataset,
                    cluster_id=cluster_id,
                    difficulty_score=difficulty,
                    embedding=vector_tuple,
                )
            )
        return cls(tasks)

    def selection_seeds(self, limit: int | None = None) -> list[Any]:
        from .selfplay import SelfPlaySeed

        ids = self.ids if limit is None else self.ids[: max(0, int(limit))]
        return [
            SelfPlaySeed(
                self.tasks[pool_id].task.prompt,
                seed_id=pool_id,
                metadata={
                    "pool_id": pool_id,
                    "dataset": self.tasks[pool_id].dataset,
                    "cluster_id": self.tasks[pool_id].cluster_id,
                },
            )
            for pool_id in ids
        ]

    def balanced_selection_seeds(self, limit: int, *, offset_per_dataset: int = 0) -> list[Any]:
        """Round-robin distinct source tasks across datasets.

        Some fixed pools intentionally contain several deterministic rows for
        one underlying source task (notably SWE ADS replicas).  A frozen
        evaluation/training anchor schedule must never count those replicas as
        different questions, even though the adaptive scheduler still needs to
        retain them as individual pool rows internally.
        """

        from .selfplay import SelfPlaySeed

        if limit <= 0:
            return []
        by_dataset: dict[str, list[str]] = defaultdict(list)
        seen_sources: dict[str, set[str]] = defaultdict(set)
        for pool_id in self.ids:
            task = self.tasks[pool_id]
            source_id = self._source_id(task)
            if source_id in seen_sources[task.dataset]:
                continue
            seen_sources[task.dataset].add(source_id)
            by_dataset[task.dataset].append(pool_id)
        datasets = sorted(by_dataset)
        selected: list[str] = []
        offset = max(0, int(offset_per_dataset))
        while len(selected) < limit:
            added = False
            for dataset in datasets:
                ids = by_dataset[dataset]
                if offset < len(ids):
                    selected.append(ids[offset])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            offset += 1
        return [
            SelfPlaySeed(
                self.tasks[pool_id].task.prompt,
                seed_id=pool_id,
                metadata={
                    "pool_id": pool_id,
                    "dataset": self.tasks[pool_id].dataset,
                    "cluster_id": self.tasks[pool_id].cluster_id,
                },
            )
            for pool_id in selected
        ]

    @staticmethod
    def _source_id(task: PoolTask) -> str:
        """Return the identity used to de-duplicate fixed schedule anchors."""

        return str(
            task.task.metadata.get("instance_id")
            or task.task.metadata.get("source_task_id")
            or task.pool_id
        )


@dataclass
class ADSClusterState:
    cluster_id: str
    cluster_size: int
    success_rate: float = 0.5
    prob: float = 0.0
    mini_positions: set[int] = field(default_factory=set)


class ADSBoundaryScheduler:
    """ADS boundary mini-clusters over a trusted fixed task pool."""

    def __init__(
        self,
        pool: FixedTaskPool,
        *,
        active_clusters: int = 4,
        mini_cluster_size: int = 32,
        boundary_eps: float = 0.17,
        alpha: float = 0.3,
        seed: int = 42,
        cooldown: int = 64,
    ) -> None:
        self.pool = pool
        self.active_clusters = min(
            int(active_clusters), len({t.cluster_id for t in pool.tasks.values()})
        )
        self.mini_cluster_size = int(mini_cluster_size)
        self.boundary_eps = float(boundary_eps)
        self.alpha = float(alpha)
        self.band_low = 0.5 - self.boundary_eps
        self.band_high = 0.5 + self.boundary_eps
        self.rng = np.random.default_rng(seed)
        self.cooldown = max(0, int(cooldown))
        self.recent: list[str] = []
        self.reserved: set[str] = set()
        self.selection_count: Counter[str] = Counter()
        self.task_success: dict[str, float] = {}
        self.source_to_pool_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
        for task in pool.tasks.values():
            source_id = self._source_id(task)
            self.source_to_pool_ids[(task.dataset, source_id)].add(task.pool_id)

        grouped: dict[str, list[PoolTask]] = defaultdict(list)
        for task in pool.tasks.values():
            grouped[task.cluster_id].append(task)
        self.cluster_ids = sorted(grouped)
        self.cluster_dataset: dict[str, str] = {}
        for cluster_id, tasks in grouped.items():
            datasets = {task.dataset for task in tasks}
            if len(datasets) != 1:
                raise ValueError(f"ADS cluster {cluster_id!r} mixes multiple datasets")
            self.cluster_dataset[cluster_id] = next(iter(datasets))
        self.dataset_cluster_ids = {
            dataset: tuple(
                cluster_id
                for cluster_id in self.cluster_ids
                if self.cluster_dataset[cluster_id] == dataset
            )
            for dataset in sorted(set(self.cluster_dataset.values()))
        }
        self.cluster_to_sorted_ids = {
            cluster_id: tuple(
                task.pool_id
                for task in sorted(tasks, key=lambda item: (item.difficulty_score, item.pool_id))
            )
            for cluster_id, tasks in grouped.items()
        }
        self.cluster_states = {
            cluster_id: ADSClusterState(
                cluster_id=cluster_id,
                cluster_size=len(task_ids),
                prob=1.0 / len(self.dataset_cluster_ids[self.cluster_dataset[cluster_id]]),
                mini_positions=set(range(min(self.mini_cluster_size, len(task_ids)))),
            )
            for cluster_id, task_ids in self.cluster_to_sorted_ids.items()
        }

    def candidates(self, *, dataset: str | None = None) -> list[str]:
        cluster_ids = (
            list(self.dataset_cluster_ids.get(dataset, ())) if dataset else self.cluster_ids
        )
        if not cluster_ids:
            raise ValueError(f"fixed task pool has no ADS clusters for dataset {dataset!r}")
        probs = np.asarray([self.cluster_states[cid].prob for cid in cluster_ids], dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(probs), 1 / len(probs))
        chosen = self.rng.choice(
            len(cluster_ids),
            size=min(self.active_clusters, len(cluster_ids)),
            replace=False,
            p=probs,
        )
        recent = self._recent_ids(dataset)
        buckets: list[list[str]] = []
        for cluster_index in chosen.tolist():
            cluster_id = cluster_ids[int(cluster_index)]
            task_ids = self.cluster_to_sorted_ids[cluster_id]
            positions = sorted(self.cluster_states[cluster_id].mini_positions)
            buckets.append(
                sorted(
                    (
                        task_ids[position]
                        for position in positions
                        if task_ids[position] not in recent
                        and task_ids[position] not in self.reserved
                    ),
                    key=lambda pool_id: (self.selection_count[pool_id], pool_id),
                )
            )
        selected = [
            bucket[index]
            for index in range(max((len(bucket) for bucket in buckets), default=0))
            for bucket in buckets
            if index < len(bucket)
        ]
        if not selected:
            selected = [
                pool_id
                for pool_id in self.pool.ids
                if pool_id not in recent
                and pool_id not in self.reserved
                and (dataset is None or self.pool.tasks[pool_id].dataset == dataset)
            ]
        return selected

    def repository(self, boundary_ids: Sequence[str]) -> list[str]:
        """Return the full candidate repository for the ADS-selected clusters."""

        cluster_ids = {
            self.pool.tasks[pool_id].cluster_id
            for pool_id in boundary_ids
            if pool_id in self.pool.tasks
        }
        datasets = {self.cluster_dataset[cluster_id] for cluster_id in cluster_ids}
        recent = set().union(*(self._recent_ids(dataset) for dataset in datasets))
        return [
            pool_id
            for pool_id in self.pool.ids
            if self.pool.tasks[pool_id].cluster_id in cluster_ids
            and pool_id not in recent
            and pool_id not in self.reserved
        ]

    def reserve(self, pool_id: str) -> None:
        if pool_id not in self.pool.tasks:
            raise KeyError(f"unknown fixed-pool task: {pool_id}")
        if pool_id not in self.task_success:
            self.reserved.update(self._source_equivalents(pool_id))

    def record(self, pool_id: str, rewards: Sequence[float]) -> None:
        self.record_batch([(pool_id, rewards)])

    def record_batch(self, observations: Sequence[tuple[str, Sequence[float]]]) -> None:
        """Apply one deterministic ADS update from a task execution window."""

        cluster_rhos: dict[str, list[float]] = defaultdict(list)
        movements: dict[str, dict[int, int]] = defaultdict(dict)
        affected_datasets: set[str] = set()
        for pool_id, rewards in observations:
            if pool_id not in self.pool.tasks:
                raise KeyError(f"unknown fixed-pool task: {pool_id}")
            if not rewards:
                continue
            rho = float(np.mean(np.clip(np.asarray(rewards, dtype=np.float64), 0.0, 1.0)))
            self.task_success[pool_id] = rho
            self.selection_count[pool_id] += 1
            self.reserved.difference_update(self._source_equivalents(pool_id))
            self.recent.append(pool_id)
            cluster_id = self.pool.tasks[pool_id].cluster_id
            affected_datasets.add(self.pool.tasks[pool_id].dataset)
            cluster_rhos[cluster_id].append(rho)
            task_ids = self.cluster_to_sorted_ids[cluster_id]
            position = task_ids.index(pool_id)
            if position in self.cluster_states[cluster_id].mini_positions:
                movements[cluster_id][position] = (
                    -1 if rho < self.band_low else 1 if rho > self.band_high else 0
                )

        retained = self.cooldown * len(self.dataset_cluster_ids)
        if retained and len(self.recent) > retained:
            self.recent = self.recent[-retained:]
        for cluster_id, values in cluster_rhos.items():
            self.cluster_states[cluster_id].success_rate = float(np.mean(values))
            self._update_mini_positions(cluster_id, movements[cluster_id])
        for dataset in sorted(affected_datasets):
            self._update_inter_cluster_distribution(dataset)

    def reset_epoch(self) -> None:
        """Reset transient ADS statistics at an explicit, reproducible epoch boundary."""

        self.task_success.clear()
        self.recent.clear()
        self.reserved.clear()
        for cluster_ids in self.dataset_cluster_ids.values():
            uniform = 1.0 / len(cluster_ids)
            for cluster_id in cluster_ids:
                state = self.cluster_states[cluster_id]
                state.success_rate = 0.5
                state.prob = uniform
                state.mini_positions = set(range(min(self.mini_cluster_size, state.cluster_size)))

    def _update_inter_cluster_distribution(self, dataset: str) -> None:
        cluster_ids = self.dataset_cluster_ids[dataset]
        rates = np.asarray(
            [self.cluster_states[cid].success_rate for cid in cluster_ids], dtype=np.float64
        )
        target = rates / rates.sum() if rates.sum() > 0 else np.full(len(rates), 1 / len(rates))
        for cluster_id, target_prob in zip(cluster_ids, target.tolist(), strict=True):
            state = self.cluster_states[cluster_id]
            state.prob = (1.0 - self.alpha) * state.prob + self.alpha * float(target_prob)
        total = sum(self.cluster_states[cluster_id].prob for cluster_id in cluster_ids)
        for cluster_id in cluster_ids:
            self.cluster_states[cluster_id].prob /= total

    def _recent_ids(self, dataset: str | None) -> set[str]:
        if not self.cooldown:
            return set()
        selected: list[str] = []
        for pool_id in reversed(self.recent):
            if dataset is None or self.pool.tasks[pool_id].dataset == dataset:
                selected.append(pool_id)
                if len(selected) == self.cooldown:
                    break
        return {
            equivalent for pool_id in selected for equivalent in self._source_equivalents(pool_id)
        }

    def _source_equivalents(self, pool_id: str) -> set[str]:
        task = self.pool.tasks[pool_id]
        source_id = self._source_id(task)
        return self.source_to_pool_ids[(task.dataset, source_id)]

    @staticmethod
    def _source_id(task: PoolTask) -> str:
        # SWE ADS pools may contain deterministic sampling replicas. The
        # official instance identity, rather than the row id, defines whether
        # two rows are the same underlying task.
        return str(
            task.task.metadata.get("instance_id")
            or task.task.metadata.get("source_task_id")
            or task.pool_id
        )

    def _update_mini_positions(self, cluster_id: str, directions: dict[int, int]) -> None:
        state = self.cluster_states[cluster_id]
        size = state.cluster_size
        old_positions = sorted(state.mini_positions)
        if size <= len(old_positions):
            return
        new_positions: set[int] = set()
        movers: list[tuple[int, int]] = []
        for position in old_positions:
            direction = directions.get(position, 0)
            (
                new_positions.add(position)
                if direction == 0
                else movers.append((position, direction))
            )
        easier = sorted(position for position, direction in movers if direction < 0)
        harder = sorted((position for position, direction in movers if direction > 0), reverse=True)
        for position in easier:
            new_positions.add(self._nearest_free(position - 1, -1, new_positions, size, position))
        for position in harder:
            new_positions.add(self._nearest_free(position + 1, 1, new_positions, size, position))
        state.mini_positions = new_positions

    @staticmethod
    def _nearest_free(
        desired: int, direction: int, occupied: set[int], size: int, fallback: int
    ) -> int:
        desired = min(max(desired, 0), size - 1)
        if desired not in occupied:
            return desired
        for radius in range(1, size):
            for candidate in (desired + direction * radius, desired - direction * radius):
                if 0 <= candidate < size and candidate not in occupied:
                    return candidate
        return fallback

    def state_dict(self) -> dict[str, Any]:
        return {
            "cluster_states": {
                cluster_id: {
                    "success_rate": state.success_rate,
                    "prob": state.prob,
                    "mini_positions": sorted(state.mini_positions),
                }
                for cluster_id, state in self.cluster_states.items()
            },
            "task_success": self.task_success,
            "selection_count": dict(self.selection_count),
            "recent": self.recent,
            "reserved": sorted(self.reserved),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        for cluster_id, values in payload.get("cluster_states", {}).items():
            if cluster_id not in self.cluster_states:
                continue
            state = self.cluster_states[cluster_id]
            state.success_rate = float(values["success_rate"])
            state.prob = float(values["prob"])
            state.mini_positions = {int(value) for value in values["mini_positions"]}
        self.task_success = {
            str(key): float(value) for key, value in payload.get("task_success", {}).items()
        }
        self.selection_count = Counter(
            {str(key): int(value) for key, value in payload.get("selection_count", {}).items()}
        )
        self.recent = [str(value) for value in payload.get("recent", [])]
        self.reserved = {str(value) for value in payload.get("reserved", [])}
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = payload["rng_state"]


class TSDSRetriever:
    """KNN-KDE candidate distribution migrated from the TSDS implementation."""

    def __init__(
        self,
        pool: FixedTaskPool,
        *,
        max_k: int = 32,
        kde_k: int = 16,
        sigma: float = 0.1,
        alpha: float = 0.5,
        c: float = 1.0,
    ) -> None:
        self.pool = pool
        self.max_k = int(max_k)
        self.kde_k = int(kde_k)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.c = float(c)

    def rank(
        self,
        candidate_ids: Sequence[str],
        query_vectors: Sequence[Sequence[float]],
        *,
        limit: int,
    ) -> list[str]:
        ids = list(dict.fromkeys(candidate_ids))
        if not ids:
            return []
        if not query_vectors:
            return ids[:limit]
        candidates = np.asarray(
            [self.pool.tasks[pool_id].embedding for pool_id in ids], dtype=np.float32
        )
        queries = np.asarray(query_vectors, dtype=np.float32)
        probabilities = _tsds_probabilities(
            queries,
            candidates,
            max_k=min(self.max_k, len(ids)),
            kde_k=min(self.kde_k, len(ids)),
            sigma=self.sigma,
            alpha=self.alpha,
            c=self.c,
        )
        ranked = np.argsort(-probabilities, kind="stable")
        return [ids[int(index)] for index in ranked[:limit] if probabilities[int(index)] > 0]


def _tsds_probabilities(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    max_k: int,
    kde_k: int,
    sigma: float,
    alpha: float,
    c: float,
) -> np.ndarray:
    """TSDS probability assignment; only nearest-neighbour search is NumPy based."""

    max_k = max(1, min(int(max_k), len(candidates)))
    kde_k = max(1, min(int(kde_k), len(candidates)))
    squared = ((query[:, None, :] - candidates[None, :, :]) ** 2).sum(axis=-1)
    top_indices = np.argsort(squared, axis=-1)[:, :max_k]
    top_dists = np.sqrt(np.take_along_axis(squared, top_indices, axis=1))

    if sigma == 0:
        top_kdes = np.ones_like(top_dists)
    else:
        unique_indices = np.unique(top_indices)
        features = candidates[unique_indices]
        kde_squared = ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.sort(kde_squared, axis=-1)[:, :kde_k]
        kernel = np.maximum(1.0 - nearest / (sigma**2), 0.0)
        kde = np.maximum(kernel.sum(axis=-1), 1e-12)
        kde_map = {int(index): float(kde[pos]) for pos, index in enumerate(unique_indices)}
        top_kdes = np.asarray(
            [[kde_map[int(index)] for index in row] for row in top_indices], dtype=np.float64
        )

    query_count, candidate_count = top_indices.shape[0], candidates.shape[0]
    if max_k == 1:
        probabilities = np.zeros(candidate_count)
        for row in top_indices:
            probabilities[int(row[0])] += 1.0 / query_count
        return probabilities / probabilities.sum()

    last_k = [0] * query_count
    heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(query_count)]
    heapq.heapify(heap)
    weighted_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(query_count)]
    mass = 1.0
    costs = np.zeros(query_count)
    total_cost = 0.0
    while heap:
        count, current_k, current_query = heapq.heappop(heap)
        mass = count
        total_cost -= costs[current_query]
        costs[current_query] = (
            top_dists[current_query][current_k + 1] * count - weighted_sum[current_query]
        )
        total_cost += costs[current_query]
        if alpha / c * total_cost >= (1 - alpha) * query_count:
            break
        last_k[current_query] = current_k
        if current_k < max_k - 2:
            count += 1.0 / top_kdes[current_query][current_k + 1]
            heapq.heappush(heap, (count, current_k + 1, current_query))
            weighted_sum[current_query] += (
                top_dists[current_query][current_k + 1] / top_kdes[current_query][current_k + 1]
            )

    probabilities = np.zeros(candidate_count)
    for query_index in range(query_count):
        assigned = 0.0
        for neighbor in range(last_k[query_index] + 1):
            value = 1.0 / query_count / mass / top_kdes[query_index][neighbor]
            probabilities[top_indices[query_index][neighbor]] += value
            assigned += value
        remainder = max(1.0 / query_count - assigned, 0.0)
        probabilities[top_indices[query_index][last_k[query_index] + 1]] += remainder
    total = probabilities.sum()
    return probabilities / total if total > 0 else np.full(candidate_count, 1 / candidate_count)


def _task_prompt(row: dict[str, Any]) -> str:
    prompt = str(
        row.get("prompt", row.get("task", row.get("problem", row.get("question", ""))))
    ).strip()
    if not prompt:
        raise ValueError("fixed-pool row has no prompt")
    return prompt


def _task_reference(row: dict[str, Any]) -> Any:
    answers = row.get("target_answers")
    if isinstance(answers, list) and answers:
        return answers[0] if len(answers) == 1 else answers
    return row.get("reference", row.get("answer"))
