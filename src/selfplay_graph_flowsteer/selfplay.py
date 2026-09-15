from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .features import GraphFeatures, execution_policy_features_many, graph_kernel_matrix
from .graph import MultiAgentGraph
from .graph_learning import build_graph_training_batch
from .learning import compute_group_advantages
from .llm import ChatBackend
from .observability import TaskSpec, task_to_public_dict
from .rollouts import (
    TokenizedDirectorTrajectory,
    TokenizedPolicyCall,
    Tokenizer,
    TrainingBatch,
    TrainingSample,
)

if TYPE_CHECKING:
    from .curriculum import ADSBoundaryScheduler, FixedTaskPool, TSDSRetriever


@dataclass(frozen=True)
class ProposedTask:
    task: TaskSpec
    response: str
    token_ids: tuple[int, ...] = ()
    action_mask: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_calls: tuple[TokenizedPolicyCall, ...] = ()


@dataclass(frozen=True)
class SelfPlaySeed:
    """A task-generation seed; reasoning hops never prescribe workflow depth."""

    content: str
    seed_id: str = ""
    target_answer: Any = None
    required_reasoning_hops: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("self-play seed content must not be empty")
        hops = self.required_reasoning_hops
        if hops is not None and hops < 1:
            raise ValueError("required_reasoning_hops must be at least one")
        if hops is not None and self.target_answer in (None, "", [], {}):
            raise ValueError("hop-conditioned seeds require a target_answer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.seed_id,
            "seed": self.content,
            "target_answer": self.target_answer,
            "required_reasoning_hops": self.required_reasoning_hops,
            "metadata": self.metadata,
        }


SeedInput = SelfPlaySeed | str


class TaskProposer(Protocol):
    def propose(self, seed: SeedInput, *, task_id: str) -> ProposedTask: ...


class QwenTaskProposer:
    """SESA-style answer-seed proposer isolated from Solver skills/state."""

    def __init__(self, backend: ChatBackend, tokenizer: Tokenizer | None = None) -> None:
        self.backend = backend
        self.tokenizer = tokenizer

    def propose(self, seed: SeedInput, *, task_id: str) -> ProposedTask:
        seed_spec = normalize_selfplay_seed(seed)
        has_target = seed_spec.target_answer not in (None, "", [], {})
        if not has_target:
            system_prompt = (
                "Create one verifiable, non-trivial task from the supplied seed. "
                "Return JSON with prompt, reference, and task_type."
            )
            user_content = seed_spec.content
        elif seed_spec.required_reasoning_hops is None:
            system_prompt = (
                "Create one concise, uniquely answerable task from the supplied target answer. "
                "Do not reveal any target-answer alias in the task. Return JSON with prompt "
                "and task_type. Workflow graph depth is chosen later by the Solver."
            )
            user_content = json.dumps(
                {
                    "target_answer": seed_spec.target_answer,
                    "seed_context": seed_spec.content,
                },
                ensure_ascii=False,
            )
        else:
            system_prompt = (
                "Create one concise, uniquely answerable multi-hop task from the supplied "
                "target answer. The task must require at least required_reasoning_hops "
                "distinct evidence/reasoning transitions. Do not reveal the target answer "
                "in the task. Return JSON with prompt, task_type, required_reasoning_hops, "
                "and evidence_chain. evidence_chain must be a JSON list with at least one "
                "non-empty entry per required hop. Workflow graph depth is chosen later by "
                "the Solver and must not be specified or inferred here."
            )
            user_content = json.dumps(
                {
                    "target_answer": seed_spec.target_answer,
                    "required_reasoning_hops": seed_spec.required_reasoning_hops,
                    "seed_context": seed_spec.content,
                },
                ensure_ascii=False,
            )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_content},
        ]
        generated = self.backend.generate(
            messages,
            role="proposer",
            temperature=0.8,
        )
        response = generated.text
        payload = _extract_json(response)
        prompt = str(payload.get("prompt", payload.get("task", ""))).strip()
        if not prompt:
            raise ValueError("proposer response has no task prompt")
        evidence_chain: list[Any] = []
        if seed_spec.required_reasoning_hops is not None:
            _validate_hop_conditioned_proposal(seed_spec, payload, prompt)
            evidence_chain = list(payload["evidence_chain"])
        elif has_target:
            _validate_target_not_leaked(seed_spec.target_answer, prompt)
        token_ids, action_mask = _tokenize_proposer_response(self.tokenizer, messages, response)
        policy_call = _tokenize_proposer_policy_call(
            self.tokenizer,
            messages,
            response,
            generated,
            call_id=f"{task_id}:proposal:0",
        )
        return ProposedTask(
            task=TaskSpec(
                task_id=task_id,
                prompt=prompt,
                reference=(
                    seed_spec.target_answer
                    if has_target
                    else payload.get("reference", payload.get("answer"))
                ),
                task_type=str(
                    payload.get("task_type", seed_spec.metadata.get("task_type", "general"))
                ),
                metadata={
                    **seed_spec.metadata,
                    "source_verifier": seed_spec.metadata.get("verifier"),
                    "verifier": (
                        seed_spec.metadata.get("verifier", "exact_match")
                        if has_target
                        else "exact_match"
                    ),
                    "selfplay_seed": seed_spec.to_dict(),
                    "required_reasoning_hops": seed_spec.required_reasoning_hops,
                    "evidence_chain": evidence_chain,
                },
            ),
            response=response,
            token_ids=token_ids,
            action_mask=action_mask,
            policy_calls=(policy_call,) if policy_call is not None else (),
            metadata={
                "seed_id": seed_spec.seed_id,
                "required_reasoning_hops": seed_spec.required_reasoning_hops,
            },
        )


class FixedPoolQwenProposer:
    """Trainable Qwen selector constrained to ADS+TSDS fixed-pool candidates."""

    def __init__(
        self,
        backend: ChatBackend,
        pool: FixedTaskPool,
        scheduler: ADSBoundaryScheduler,
        retriever: TSDSRetriever,
        tokenizer: Tokenizer | None = None,
        *,
        candidate_count: int = 8,
        prompt_preview_chars: int = 500,
        selection_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.pool = pool
        self.scheduler = scheduler
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.candidate_count = max(2, int(candidate_count))
        self.prompt_preview_chars = max(80, int(prompt_preview_chars))
        self.selection_attempts = max(1, int(selection_attempts))

    def propose(self, seed: SeedInput, *, task_id: str) -> ProposedTask:
        seed_spec = normalize_selfplay_seed(seed)
        anchor_id = seed_spec.seed_id if seed_spec.seed_id in self.pool.tasks else ""
        target_dataset = str(seed_spec.metadata.get("dataset", "")).strip()
        if anchor_id:
            target_dataset = self.pool.tasks[anchor_id].dataset
        if len(self.scheduler.dataset_cluster_ids) > 1 and not target_dataset:
            raise ValueError(
                "joint fixed-pool selection requires a dataset-scoped seed so TSDS never "
                "compares independently preprocessed PCA spaces"
            )
        boundary_ids = self.scheduler.candidates(dataset=target_dataset or None)
        active_clusters = {self.pool.tasks[pool_id].cluster_id for pool_id in boundary_ids}
        frontier_ids = [
            pool_id
            for pool_id, _success in sorted(
                self.scheduler.task_success.items(), key=lambda item: abs(item[1] - 0.5)
            )
            if self.pool.tasks[pool_id].cluster_id in active_clusters
        ][:4]
        query_ids = frontier_ids or ([anchor_id] if anchor_id else boundary_ids[:4])
        query_vectors = [self.pool.tasks[pool_id].embedding for pool_id in query_ids]
        repository_ids = self.scheduler.repository(boundary_ids)
        if anchor_id and len(repository_ids) > 1:
            repository_ids = [pool_id for pool_id in repository_ids if pool_id != anchor_id]
        retrieval_repository = [pool_id for pool_id in repository_ids if pool_id not in query_ids]
        retrieved = self.retriever.rank(
            retrieval_repository,
            query_vectors,
            limit=self.candidate_count,
        )
        boundary_slots = max(1, self.candidate_count // 2)
        ranked = list(
            dict.fromkeys(
                [
                    *[
                        pool_id
                        for pool_id in boundary_ids
                        if pool_id in repository_ids and pool_id != anchor_id
                    ][:boundary_slots],
                    *retrieved,
                ]
            )
        )[: self.candidate_count]
        if not ranked and anchor_id in repository_ids:
            # A sparse singleton-cluster pool can leave the scheduled anchor as
            # the only non-recent, non-reserved boundary item.  The anchor is a
            # real audited pool task, so selecting it is preferable to creating
            # a false proposal hole merely because it also served as the TSDS
            # query.  The usual non-anchor ADS/TSDS ranking remains unchanged.
            ranked = [anchor_id]
        if not ranked:
            raise ValueError("ADS+TSDS produced no selectable fixed-pool candidates")

        candidates = [
            {
                "candidate_id": pool_id,
                "dataset": self.pool.tasks[pool_id].dataset,
                "task_type": self.pool.tasks[pool_id].task.task_type,
                "prompt_preview": self.pool.tasks[pool_id].task.prompt[: self.prompt_preview_chars],
                "previous_success": self.scheduler.task_success.get(pool_id),
                "selection_count": self.scheduler.selection_count[pool_id],
            }
            for pool_id in ranked
        ]
        system_prompt = (
            "Select exactly one candidate task for the current Solver. The candidates were "
            "prepared by ADS boundary scheduling and TSDS similarity/diversity retrieval. "
            "Prefer a useful capability-boundary task, not a repeated mastered task. "
            'Return JSON only: {"candidate_id": "one listed id"}. You may select an id '
            "but must not rewrite any task, answer, verifier, or metadata."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "anchor": seed_spec.content[: self.prompt_preview_chars],
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = ""
        selected_id = ""
        policy_calls: list[TokenizedPolicyCall] = []
        for attempt in range(self.selection_attempts):
            generated = self.backend.generate(
                messages,
                role="proposer",
                temperature=0.8 if attempt == 0 else 0.0,
                max_tokens=None if attempt == 0 else 256,
            )
            response = generated.text
            policy_call = _tokenize_proposer_policy_call(
                self.tokenizer,
                messages,
                response,
                generated,
                call_id=f"{task_id}:selection:{attempt}",
            )
            if policy_call is not None:
                policy_calls.append(policy_call)
            payload = _extract_json(response)
            selected_id = str(payload.get("candidate_id", "")).strip()
            if selected_id in ranked:
                break
            if attempt + 1 < self.selection_attempts:
                messages = [
                    *messages,
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "error": "candidate_id must be one of the listed ids",
                                "allowed_candidate_ids": ranked,
                                "required_format": {"candidate_id": "one allowed id"},
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
        if selected_id not in ranked:
            raise ValueError(
                "proposer candidate_id is not in the ADS+TSDS candidate set after "
                f"{self.selection_attempts} attempts"
            )
        self.scheduler.reserve(selected_id)
        selected = self.pool.tasks[selected_id]
        validated_pool_entry = bool(selected.task.metadata.get("validated_pool_entry", False))
        token_ids, action_mask = _tokenize_proposer_response(self.tokenizer, messages, response)
        return ProposedTask(
            task=TaskSpec(
                task_id=task_id,
                prompt=selected.task.prompt,
                reference=selected.task.reference,
                task_type=selected.task.task_type,
                metadata={
                    **selected.task.metadata,
                    "pool_id": selected_id,
                    "proposer_mode": "fixed_pool_ads_tsds",
                    "candidate_ids": ranked,
                    "ads_boundary_ids": boundary_ids,
                    "tsds_query_ids": query_ids,
                    "anchor_pool_id": anchor_id or None,
                    "required_reasoning_hops": None,
                    "validated_pool_entry": validated_pool_entry,
                },
                private_verifier_payload=selected.task.private_verifier_payload,
            ),
            response=response,
            token_ids=token_ids,
            action_mask=action_mask,
            policy_calls=tuple(policy_calls),
            metadata={
                "pool_id": selected_id,
                "candidate_ids": ranked,
                "ads_boundary_ids": boundary_ids,
                "tsds_query_ids": query_ids,
                "selection_group": f"{selected.dataset}:{selected.cluster_id}",
                "validated_pool_entry": validated_pool_entry,
                "validated_pool_manifest_sha256": selected.task.metadata.get(
                    "validated_pool_manifest_sha256"
                ),
                "validated_pool_version": selected.task.metadata.get("validated_pool_version"),
            },
        )

    def observe(self, proposal: ProposedTask, rewards: Iterable[float]) -> None:
        self.scheduler.record(str(proposal.metadata["pool_id"]), list(rewards))

    def observe_many(self, observations: Iterable[tuple[ProposedTask, Iterable[float]]]) -> None:
        self.scheduler.record_batch(
            [
                (str(proposal.metadata["pool_id"]), list(rewards))
                for proposal, rewards in observations
            ]
        )

    def reserve(self, proposal: ProposedTask) -> None:
        self.scheduler.reserve(str(proposal.metadata["pool_id"]))

    def rehydrate_private_payload(self, proposal: ProposedTask) -> ProposedTask:
        """Restore verifier-only data from the trusted pool after public-log resume."""

        pool_id = str(proposal.metadata["pool_id"])
        source = self.pool.tasks[pool_id].task
        return replace(
            proposal,
            task=replace(
                proposal.task,
                private_verifier_payload=source.private_verifier_payload,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(payload)


def normalize_selfplay_seed(seed: SeedInput) -> SelfPlaySeed:
    if isinstance(seed, SelfPlaySeed):
        return seed
    return SelfPlaySeed(content=str(seed))


def selfplay_seed_from_mapping(item: dict[str, Any]) -> SelfPlaySeed:
    """Read both the local schema and SESA/SSP answer-seed aliases."""

    target_values = item.get("target_answers")
    if isinstance(target_values, list) and target_values:
        target = target_values[0] if len(target_values) == 1 else target_values
    else:
        target = next(
            (
                item[key]
                for key in ("target_answer", "ground_truth", "answer")
                if item.get(key) not in (None, "", [], {})
            ),
            None,
        )
    raw_content = next(
        (
            item[key]
            for key in (
                "seed",
                "target_answer",
                "ground_truth",
                "answer",
                "task",
                "problem",
                "question",
                "instruction",
                "problem_statement",
                "task_descriptions",
                "conversation",
            )
            if item.get(key) not in (None, "", [], {})
        ),
        None,
    )
    content = (
        json.dumps(raw_content, ensure_ascii=False)
        if isinstance(raw_content, (dict, list))
        else ""
        if raw_content is None
        else str(raw_content)
    )
    raw_hops = next(
        (
            item[key]
            for key in (
                "required_reasoning_hops",
                "required_hops",
                "search_turns",
                "hop",
                "hops",
            )
            if item.get(key) not in (None, "")
        ),
        None,
    )
    try:
        hops = int(raw_hops) if raw_hops is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid required reasoning hops: {raw_hops!r}") from exc
    metadata = dict(item.get("metadata") or {})
    for key in (
        "source",
        "sys_question_example",
        "dataset",
        "split",
        "mode",
        "task_type",
        "verifier",
        "prompt",
        "context_documents",
    ):
        if key in item and key not in metadata:
            metadata[key] = item[key]
    return SelfPlaySeed(
        content=content,
        seed_id=str(item.get("id", item.get("seed_id", item.get("index", "")))),
        target_answer=target,
        required_reasoning_hops=hops,
        metadata=metadata,
    )


def load_selfplay_seed_jsonl(path: str | Path) -> list[SelfPlaySeed]:
    seeds: list[SelfPlaySeed] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                seeds.append(selfplay_seed_from_mapping(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid self-play seed at {path}:{line_number}: {exc}") from exc
    return seeds


def _validate_hop_conditioned_proposal(
    seed: SelfPlaySeed, payload: dict[str, Any], prompt: str
) -> None:
    required = int(seed.required_reasoning_hops or 0)
    try:
        reported = int(payload.get("required_reasoning_hops"))
    except (TypeError, ValueError) as exc:
        raise ValueError("hop-conditioned proposal must report required_reasoning_hops") from exc
    if reported != required:
        raise ValueError(
            f"proposal hop mismatch: required {required}, proposer reported {reported}"
        )
    chain = payload.get("evidence_chain")
    if not isinstance(chain, list) or len(chain) < required:
        raise ValueError(f"proposal evidence_chain must contain at least {required} hops")
    if any(entry in (None, "", [], {}) for entry in chain[:required]):
        raise ValueError("proposal evidence_chain contains an empty hop")
    _validate_target_not_leaked(seed.target_answer, prompt)


def _validate_target_not_leaked(target: Any, prompt: str) -> None:
    values = target if isinstance(target, (list, tuple, set)) else [target]
    normalized_prompt = prompt.casefold()
    if any(
        str(value).strip().casefold() in normalized_prompt for value in values if str(value).strip()
    ):
        raise ValueError("proposer task prompt leaks the target answer")


@dataclass(frozen=True)
class SolverRollout:
    trajectory: TokenizedDirectorTrajectory
    graph: MultiAgentGraph


@dataclass(frozen=True)
class FrontierScore:
    task_id: str
    validity: float
    scalar: float
    graph_local: float
    rewards: tuple[float, ...]
    provisional_graph_local: float | None = None
    stable_graph_local: float | None = None
    reverify_status: str = "not_triggered"
    pair_stability: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def scalar_frontier(
    rewards: Iterable[float], *, validity: float = 1.0, normalization: str = "legacy"
) -> float:
    values = [float(value) for value in rewards]
    if len(values) < 2:
        return 0.0
    pair_sum = sum(
        (values[left] - values[right]) ** 2
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    from .proposer_learning import frontier_multiplier

    multiplier = frontier_multiplier(len(values), normalization)
    if normalization == "legacy" or len(values) == 5:
        return max(0.0, float(validity)) * 4.0 * pair_sum / (len(values) ** 2)
    return max(0.0, float(validity)) * multiplier * pair_sum


def graph_local_frontier(
    rewards: Iterable[float],
    features: Iterable[GraphFeatures],
    *,
    validity: float = 1.0,
    kernel_matrix: Iterable[Iterable[float]] | None = None,
    normalization: str = "legacy",
) -> float:
    values = [float(value) for value in rewards]
    vectors = list(features)
    if len(values) != len(vectors):
        raise ValueError("rewards and features must have equal length")
    if len(values) < 2:
        return 0.0
    matrix = (
        tuple(tuple(float(value) for value in row) for row in kernel_matrix)
        if kernel_matrix is not None
        else graph_kernel_matrix(vectors)
    )
    if len(matrix) != len(values) or any(len(row) != len(values) for row in matrix):
        raise ValueError("kernel matrix shape does not match rewards")
    weighted = sum(
        matrix[left][right] * (values[left] - values[right]) ** 2
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    from .proposer_learning import frontier_multiplier

    multiplier = frontier_multiplier(len(values), normalization)
    if normalization == "legacy" or len(values) == 5:
        return max(0.0, float(validity)) * 4.0 * weighted / (len(values) ** 2)
    return max(0.0, float(validity)) * multiplier * weighted


def select_frontier_reverification(
    candidates: Iterable[tuple[str, str, float]],
    *,
    fraction: float = 0.25,
    minimum_frontier: float = 0.0,
) -> dict[str, tuple[str, ...]]:
    """Select a stable top fraction of positive tasks independently per dataset."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("reverify fraction must be in [0, 1]")
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for task_id, dataset, score in candidates:
        grouped[str(dataset)].append((str(task_id), float(score)))
    selected: dict[str, tuple[str, ...]] = {}
    for dataset, items in sorted(grouped.items()):
        quota = math.ceil(fraction * len(items))
        positive = [item for item in items if item[1] > minimum_frontier]
        positive.sort(key=lambda item: (-item[1], item[0]))
        selected[dataset] = tuple(task_id for task_id, _score in positive[:quota])
    return selected


def stable_graph_local_frontier(
    primary_rewards: Iterable[float],
    reverify_rewards: Iterable[float],
    features: Iterable[GraphFeatures],
    *,
    validity: float = 1.0,
    kernel_matrix: Iterable[Iterable[float]] | None = None,
    normalization: str = "legacy",
    tie_weight: float = 0.1,
    reverify_trusted: Iterable[bool] | None = None,
) -> tuple[float, tuple[dict[str, Any], ...]]:
    """Retain consistent pairs and discount trusted reverify ties, never sign flips.

    Scalar callers must supply trusted rewards. Collection callers additionally
    pass per-graph trust flags; an unaudited cached zero cannot earn tie credit.
    Use tie_weight=0 for archived hard-gate batches.
    """

    first = tuple(float(value) for value in primary_rewards)
    second = tuple(float(value) for value in reverify_rewards)
    vectors = tuple(features)
    if not math.isfinite(tie_weight) or not 0.0 <= tie_weight <= 1.0:
        raise ValueError("Frontier tie_weight must be finite and in [0, 1]")
    if len(first) != len(second) or len(first) != len(vectors):
        raise ValueError("features and both reward vectors must have equal length")
    if any(not math.isfinite(value) for value in (*first, *second)):
        raise ValueError("Frontier rewards must be finite")
    trusted = tuple(reverify_trusted) if reverify_trusted is not None else (True,) * len(first)
    if len(trusted) != len(first) or any(type(flag) is not bool for flag in trusted):
        raise ValueError("reverify_trusted must contain one boolean per graph")
    if len(first) < 2:
        return 0.0, ()
    matrix = (
        tuple(tuple(float(value) for value in row) for row in kernel_matrix)
        if kernel_matrix is not None
        else graph_kernel_matrix(vectors)
    )
    if len(matrix) != len(first) or any(len(row) != len(first) for row in matrix):
        raise ValueError("kernel matrix shape does not match rewards")
    records: list[dict[str, Any]] = []
    weighted = 0.0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            delta_primary = first[left] - first[right]
            delta_reverify = second[left] - second[right]
            passed = delta_primary * delta_reverify > 0.0
            mean_delta = (delta_primary + delta_reverify) / 2.0
            softened = (
                tie_weight > 0.0
                and delta_primary != 0.0
                and delta_reverify == 0.0
                and trusted[left]
                and trusted[right]
            )
            contribution = (
                matrix[left][right] * mean_delta**2
                if passed
                else tie_weight * matrix[left][right] * delta_primary**2
                if softened
                else 0.0
            )
            weighted += contribution
            records.append(
                {
                    "graph_pair_id": f"{left}:{right}",
                    "left_index": left,
                    "right_index": right,
                    "pair_delta_primary": delta_primary,
                    "pair_delta_reverify": delta_reverify,
                    "kernel": matrix[left][right],
                    "stability_gate_passed": passed,
                    "tie_softened": softened,
                    "tie_weight": tie_weight,
                    "reverify_pair_trusted": trusted[left] and trusted[right],
                    "contribution_rule": (
                        "consistent"
                        if passed
                        else "trusted_tie_discount"
                        if softened
                        else "rejected"
                    ),
                    "stable_pair_contribution": contribution,
                    "rejection_reason": None if passed or softened else "zero_or_direction_flip",
                }
            )
    from .proposer_learning import frontier_multiplier

    multiplier = frontier_multiplier(len(first), normalization)
    score = (
        max(0.0, float(validity)) * 4.0 * weighted / (len(first) ** 2)
        if normalization == "legacy" or len(first) == 5
        else max(0.0, float(validity)) * multiplier * weighted
    )
    return score, tuple(records)


def group_rollouts_by_task(
    rollouts: Iterable[SolverRollout],
) -> dict[str, list[SolverRollout]]:
    groups: dict[str, list[SolverRollout]] = defaultdict(list)
    for rollout in rollouts:
        groups[rollout.trajectory.task_id].append(rollout)
    return dict(groups)


def reverify_rollouts(
    rollouts: Iterable[SolverRollout],
    verify: Callable[[SolverRollout], float],
    *,
    threshold: float = 0.8,
) -> list[SolverRollout]:
    del rollouts, verify, threshold
    raise RuntimeError(
        "single-rollout reward-threshold reverification is retired; use task-group "
        "provisional Frontier selection and stable_graph_local_frontier"
    )


class SelfPlayPhase(StrEnum):
    PROPOSER_COLLECTION = "proposer_collection"
    SOLVER_COLLECTION = "solver_collection"
    BATCH_READY = "batch_ready"


@dataclass
class AlternatingSnapshots:
    proposer_snapshot: str
    solver_snapshot: str
    phase: SelfPlayPhase = SelfPlayPhase.PROPOSER_COLLECTION
    cycle: int = 0

    @property
    def frozen_role(self) -> str:
        if self.phase is SelfPlayPhase.PROPOSER_COLLECTION:
            return "solver"
        if self.phase is SelfPlayPhase.SOLVER_COLLECTION:
            return "proposer"
        return "none"

    def advance(self) -> SelfPlayPhase:
        if self.phase is SelfPlayPhase.PROPOSER_COLLECTION:
            self.phase = SelfPlayPhase.SOLVER_COLLECTION
        elif self.phase is SelfPlayPhase.SOLVER_COLLECTION:
            self.phase = SelfPlayPhase.BATCH_READY
        else:
            self.cycle += 1
            self.phase = SelfPlayPhase.PROPOSER_COLLECTION
        return self.phase

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposer_snapshot": self.proposer_snapshot,
            "solver_snapshot": self.solver_snapshot,
            "phase": self.phase.value,
            "cycle": self.cycle,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AlternatingSnapshots:
        return cls(
            proposer_snapshot=str(payload["proposer_snapshot"]),
            solver_snapshot=str(payload["solver_snapshot"]),
            phase=SelfPlayPhase(str(payload.get("phase", SelfPlayPhase.PROPOSER_COLLECTION))),
            cycle=int(payload.get("cycle", 0)),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> AlternatingSnapshots:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DryRunSelfPlayResult:
    tasks: tuple[TaskSpec, ...]
    frontier_scores: tuple[FrontierScore, ...]
    proposer_batch: TrainingBatch
    solver_batch: TrainingBatch
    snapshots: dict[str, Any]
    optimizer_steps: int = 0
    proposal_extraction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task_to_public_dict(task) for task in self.tasks],
            "frontier_scores": [asdict(score) for score in self.frontier_scores],
            "proposer_batch": self.proposer_batch.to_dict(),
            "solver_batch": self.solver_batch.to_dict(),
            "snapshots": self.snapshots,
            "optimizer_steps": self.optimizer_steps,
            "proposal_extraction": self.proposal_extraction,
        }


class DryRunSelfPlayCoordinator:
    """SESA-style alternating collection that deliberately has no optimizer dependency."""

    def __init__(
        self,
        *,
        proposer: TaskProposer,
        solve: Callable[[TaskSpec, int], SolverRollout],
        rollouts_per_task: int = 5,
        snapshots: AlternatingSnapshots | None = None,
        reverify: Callable[[SolverRollout], float] | None = None,
    ) -> None:
        if rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two")
        self.proposer = proposer
        self.solve = solve
        self.rollouts_per_task = rollouts_per_task
        self.snapshots = snapshots or AlternatingSnapshots("proposer-0", "solver-0")
        self.reverify = reverify

    def run(self, seeds: Iterable[SeedInput]) -> DryRunSelfPlayResult:
        if self.snapshots.phase is not SelfPlayPhase.PROPOSER_COLLECTION:
            raise RuntimeError("cycle must start with proposer collection")
        proposals = [
            self.proposer.propose(seed, task_id=f"task-{index}")
            for index, seed in enumerate(seeds, start=1)
        ]
        self.snapshots.advance()
        all_rollouts = [
            self.solve(proposal.task, rollout_index)
            for proposal in proposals
            for rollout_index in range(self.rollouts_per_task)
        ]
        if self.reverify:
            all_rollouts = reverify_rollouts(all_rollouts, self.reverify)
        return assemble_selfplay_result(
            proposals,
            all_rollouts,
            self.snapshots,
            expected_rollouts_per_task=self.rollouts_per_task,
        )


def validate_pats_group_context(rollouts: Iterable[SolverRollout]) -> None:
    """Reject mixed scaffolds before attributing group reward differences to graphs."""
    for task_id, group in group_rollouts_by_task(rollouts).items():
        manifests = [item.trajectory.metadata.get("skill_context", {}) for item in group]
        if not any(manifest.get("pats_enabled") for manifest in manifests):
            continue
        identities = set()
        for manifest in manifests:
            context = manifest.get("context")
            if (
                manifest.get("pats_enabled") is not True
                or not manifest.get("snapshot_id")
                or not manifest.get("pats_scope")
                or not isinstance(context, str)
            ):
                raise ValueError(f"PATS task {task_id}: missing frozen skill context")
            digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
            if digest != manifest.get("context_sha256"):
                raise ValueError(f"PATS task {task_id}: skill context hash mismatch")
            identities.add((manifest["snapshot_id"], manifest["pats_scope"], digest))
        if len(identities) != 1:
            raise ValueError(f"PATS task {task_id}: mixed skill contexts within rollout group")


def assemble_selfplay_result(
    proposals: list[ProposedTask],
    all_rollouts: list[SolverRollout],
    snapshots: AlternatingSnapshots,
    *,
    expected_rollouts_per_task: int | None = None,
    graph_feature_extractor: Any | None = None,
    frontier_reverification: dict[str, dict[str, Any]] | None = None,
    evaluation_only: bool = False,
    training_selection: dict[str, Any] | None = None,
    frontier_evidence: list[SolverRollout] | None = None,
    proposer_baseline: dict[str, Any] | None = None,
) -> DryRunSelfPlayResult:
    """Assemble the two trainer-ready batches after SESA-style K rollouts."""

    validate_pats_group_context(all_rollouts)
    if frontier_evidence is not None:
        validate_pats_group_context([*all_rollouts, *frontier_evidence])
    if training_selection and training_selection.get("schema_version") == "independent_frontier_v2":
        from .proposer_learning import assemble_independent_result

        return assemble_independent_result(
            proposals,
            all_rollouts,
            frontier_evidence or [],
            snapshots,
            selection=training_selection,
            baseline=proposer_baseline or {},
            graph_feature_extractor=graph_feature_extractor,
            frontier_reverification=frontier_reverification,
        )

    explicitly_ineligible = [
        rollout.trajectory.rollout_id
        for rollout in all_rollouts
        if rollout.trajectory.metadata.get("training_eligible") is False
    ]
    if explicitly_ineligible:
        raise ValueError(
            "refusing to assemble training-ineligible rollouts: "
            + ", ".join(sorted(explicitly_ineligible))
        )
    groups = group_rollouts_by_task(all_rollouts)
    selection_groups = {
        group["task_id"]: group for group in (training_selection or {}).get("groups", ())
    }
    if training_selection is not None:
        selected_ids = training_selection["selected_rollout_ids"]
        if (
            len({r.trajectory.rollout_id for r in all_rollouts}) != len(all_rollouts)
            or [r.trajectory.rollout_id for r in all_rollouts] != selected_ids
        ):
            raise ValueError("rollouts do not match frozen training selection")
        if set(groups) != {p.task.task_id for p in proposals}:
            raise ValueError("selected proposals do not match Solver task groups")
        for proposal in proposals:
            group = selection_groups[proposal.task.task_id]
            if not 2 <= len(groups[proposal.task.task_id]) <= group["planned_rollout_count"]:
                raise ValueError("invalid partial training group size")
            if group["proposer_exclusion"]:
                proposal.metadata["frontier_training_exclusion"] = group["proposer_exclusion"]
    if expected_rollouts_per_task is not None and training_selection is None:
        expected = int(expected_rollouts_per_task)
        if expected < 1 or (expected < 2 and not evaluation_only):
            raise ValueError("expected_rollouts_per_task must be at least two")
        incomplete = {
            proposal.task.task_id: len(groups.get(proposal.task.task_id, ()))
            for proposal in proposals
            if len(groups.get(proposal.task.task_id, ())) != expected
        }
        if incomplete:
            detail = ", ".join(
                f"{task_id}={count}/{expected}" for task_id, count in sorted(incomplete.items())
            )
            raise ValueError("refusing to assemble incomplete rollout groups: " + detail)
    frontiers: list[FrontierScore] = []
    features_by_rollout: dict[str, GraphFeatures] = {}
    kernels_by_task: dict[str, tuple[tuple[float, ...], ...]] = {}
    for proposal in proposals:
        group = groups[proposal.task.task_id]
        if (
            not evaluation_only
            and proposal.metadata.get("pool_id") is not None
            and not proposal.metadata.get("validated_pool_entry", False)
        ):
            raise ValueError("fixed-pool proposal lacks a validated task-pool attestation")
        rewards = tuple(
            float(item.trajectory.metadata.get("task_reward", item.trajectory.reward))
            for item in group
        )
        if any(
            abs(reward - item.trajectory.reward) > 1e-12
            for reward, item in zip(rewards, group, strict=True)
        ):
            raise ValueError("trajectory reward differs from the audited task_reward field")
        features = (
            graph_feature_extractor.extract_many([item.graph for item in group])
            if graph_feature_extractor is not None
            else execution_policy_features_many([item.graph for item in group])
        )
        kernel = graph_kernel_matrix(features)
        kernels_by_task[proposal.task.task_id] = kernel
        features_by_rollout.update(
            {
                item.trajectory.rollout_id: feature
                for item, feature in zip(group, features, strict=True)
            }
        )
        valid = 1.0
        provisional = graph_local_frontier(rewards, features, validity=valid, kernel_matrix=kernel)
        reverify = (frontier_reverification or {}).get(proposal.task.task_id)
        if reverify is not None:
            reverify_rewards = tuple(float(value) for value in reverify["rewards"])
            stable, pair_records = stable_graph_local_frontier(
                rewards,
                reverify_rewards,
                features,
                validity=valid,
                kernel_matrix=kernel,
                tie_weight=0.0,  # Legacy collection schema keeps its archived reward rule.
            )
            graph_ids = tuple(str(value) for value in reverify.get("graph_ids", ()))
            if len(graph_ids) == len(rewards):
                pair_records = tuple(
                    {
                        **record,
                        "graph_pair_id": (
                            f"{graph_ids[int(record['left_index'])]}:"
                            f"{graph_ids[int(record['right_index'])]}"
                        ),
                        "left_graph_id": graph_ids[int(record["left_index"])],
                        "right_graph_id": graph_ids[int(record["right_index"])],
                    }
                    for record in pair_records
                )
            graph_local = stable
            reverify_status = "completed"
        else:
            stable = None
            pair_records = ()
            graph_local = provisional
            reverify_status = "not_triggered"
        if proposal.metadata.get("frontier_training_exclusion"):
            exclusion = proposal.metadata["frontier_training_exclusion"]
            reverify_status = {
                "partial_solver_group": "excluded_partial_solver_group",
                "canary_executor_migration": "excluded_executor_migration",
                "frontier_reverify_infrastructure_failure": (
                    "excluded_reverify_infrastructure_failure"
                ),
            }.get(exclusion, "excluded_unknown")
        frontiers.append(
            FrontierScore(
                task_id=proposal.task.task_id,
                validity=valid,
                scalar=scalar_frontier(rewards, validity=valid),
                graph_local=graph_local,
                rewards=rewards,
                provisional_graph_local=provisional,
                stable_graph_local=stable,
                reverify_status=reverify_status,
                pair_stability=pair_records,
            )
        )
    frontier_exclusions = {
        p.task.task_id: p.metadata["frontier_training_exclusion"]
        for p in proposals
        if p.metadata.get("frontier_training_exclusion")
    }
    solver_trajectories = [
        replace(
            rollout.trajectory,
            metadata={
                **rollout.trajectory.metadata,
                "frontier_training_exclusion": frontier_exclusions.get(rollout.trajectory.task_id),
                "rollout_group_complete": (
                    selection_groups[rollout.trajectory.task_id]["planned_group_complete"]
                    if training_selection is not None
                    else True
                ),
                **(
                    {
                        "training_selection_schema": training_selection["schema_version"],
                        "selected_group_rollout_ids": selection_groups[rollout.trajectory.task_id][
                            "selected_rollout_ids"
                        ],
                        "selected_group_size": selection_groups[rollout.trajectory.task_id][
                            "selected_rollout_count"
                        ],
                    }
                    if training_selection is not None
                    else {}
                ),
                "training_eligible": rollout.trajectory.metadata.get("training_eligible", True),
                "rollout_group_size_expected": (
                    expected_rollouts_per_task
                    if expected_rollouts_per_task is not None
                    else len(groups[rollout.trajectory.task_id])
                ),
            },
        )
        for rollout in all_rollouts
    ]
    solver_batch = build_graph_training_batch(
        solver_trajectories,
        features_by_rollout=features_by_rollout,
        kernels_by_task=kernels_by_task,
    )
    proposer_advantages: dict[str, float] = {}
    proposal_groups: dict[str, list[tuple[ProposedTask, FrontierScore]]] = defaultdict(list)
    for proposal, frontier in zip(proposals, frontiers, strict=True):
        if proposal.metadata.get("frontier_training_exclusion"):
            continue
        group_key = str(proposal.metadata.get("seed_group", proposal.task.task_id))
        proposal_groups[group_key].append((proposal, frontier))
    for group in proposal_groups.values():
        normalized = (
            compute_group_advantages(item[1].graph_local for item in group)
            if len(group) > 1
            else [group[0][1].graph_local]
        )
        for (proposal, _frontier), advantage in zip(group, normalized, strict=True):
            proposer_advantages[proposal.task.task_id] = advantage
    proposer_samples = tuple(
        TrainingSample(
            rollout_id=f"proposal-{proposal.task.task_id}",
            task_id=proposal.task.task_id,
            token_ids=proposal.token_ids,
            action_mask=proposal.action_mask,
            reward=frontier.graph_local,
            advantage=proposer_advantages[proposal.task.task_id],
            metadata={**proposal.metadata, "response": proposal.response},
            policy_calls=proposal.policy_calls,
        )
        for proposal, frontier in zip(proposals, frontiers, strict=True)
        if not proposal.metadata.get("frontier_training_exclusion")
    )
    proposer_batch = TrainingBatch(
        role="proposer",
        samples=proposer_samples,
        objective="graph_local_frontier",
    )
    if training_selection is not None:
        batch_metadata = {
            "training_selection_schema": training_selection["schema_version"],
            "rollout_group_policy": "eligible_subset",
            "selected_rollout_ids": training_selection["selected_rollout_ids"],
            "groups": list(selection_groups.values()),
        }
        solver_batch = replace(solver_batch, metadata=batch_metadata)
        proposer_batch = replace(proposer_batch, metadata=batch_metadata)
    snapshots.advance()
    return DryRunSelfPlayResult(
        tasks=tuple(proposal.task for proposal in proposals),
        frontier_scores=tuple(frontiers),
        proposer_batch=proposer_batch,
        solver_batch=solver_batch,
        snapshots={
            "proposer": snapshots.proposer_snapshot,
            "solver": snapshots.solver_snapshot,
            "phase": snapshots.phase.value,
            "cycle": snapshots.cycle,
        },
    )


def _tokenize_proposer_response(
    tokenizer: Tokenizer | None,
    messages: list[dict[str, str]],
    response: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if tokenizer is None:
        return (), ()
    chat_encoder = getattr(tokenizer, "encode_chat_trajectory", None)
    if callable(chat_encoder):
        token_ids, action_mask, _spans = chat_encoder(
            [*messages, {"role": "assistant", "content": response}]
        )
        return tuple(token_ids), tuple(action_mask)
    prompt = (
        "\n".join(f"{message['role'].title()}:\n{message['content']}" for message in messages)
        + "\nAssistant:\n"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    return (
        tuple(prompt_ids + response_ids),
        tuple([0] * len(prompt_ids) + [1] * len(response_ids)),
    )


def _tokenize_proposer_policy_call(
    tokenizer: Tokenizer | None,
    messages: list[dict[str, str]],
    response: str,
    generated: Any,
    *,
    call_id: str,
) -> TokenizedPolicyCall | None:
    if tokenizer is None:
        return None
    chat_encoder = getattr(tokenizer, "encode_chat_trajectory", None)
    if callable(chat_encoder):
        token_ids, _legacy_mask, assistant_spans = chat_encoder(
            [*messages, {"role": "assistant", "content": response}]
        )
        if not assistant_spans:
            raise ValueError("proposer tokenizer did not expose the completion span")
        reconstructed_span = assistant_spans[-1]
        recorded_prompt_ids = tuple(int(value) for value in generated.prompt_token_ids)
        recorded_ids = tuple(int(value) for value in generated.completion_token_ids)
        if recorded_prompt_ids and recorded_ids:
            token_ids = [*recorded_prompt_ids, *recorded_ids]
            completion_span = (len(recorded_prompt_ids), len(token_ids))
        elif recorded_ids:
            token_ids = [*token_ids[: reconstructed_span[0]], *recorded_ids]
            completion_span = (reconstructed_span[0], len(token_ids))
        else:
            completion_span = reconstructed_span
        action_mask = [0] * len(token_ids)
        action_mask[completion_span[0] : completion_span[1]] = [1] * (
            completion_span[1] - completion_span[0]
        )
    else:
        prompt_ids: list[int] = []
        for message in messages:
            prompt_ids.extend(tokenizer.encode(message["content"], add_special_tokens=False))
        completion_ids = tokenizer.encode(response, add_special_tokens=False)
        token_ids = [*prompt_ids, *completion_ids]
        completion_span = (len(prompt_ids), len(token_ids))
        action_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    sampled_log_probs = tuple(float(value) for value in generated.behavior_log_probs)
    completion_length = completion_span[1] - completion_span[0]
    is_mock = bool(generated.metadata.get("mock"))
    if not generated.training_eligible and not is_mock:
        raise ValueError("proposer response is not eligible for exact policy training")
    if sampled_log_probs and len(sampled_log_probs) != completion_length:
        raise ValueError("proposer behavior log-probs do not match completion tokens")
    if not sampled_log_probs and not is_mock:
        raise ValueError("proposer response lacks rollout-time behavior log-probs")
    behavior = [0.0] * max(0, len(token_ids) - 1)
    for offset, value in enumerate(sampled_log_probs):
        target_position = completion_span[0] + offset - 1
        if target_position < 0:
            raise ValueError("proposer completion lacks next-token context")
        behavior[target_position] = value
    return TokenizedPolicyCall(
        call_id=call_id,
        token_ids=tuple(int(value) for value in token_ids),
        action_mask=tuple(action_mask),
        behavior_log_probs=tuple(behavior) if sampled_log_probs else (),
        action_token_span=completion_span,
        metadata={
            "model_id": str(generated.model),
            "route_name": str(generated.metadata.get("route", "")),
            "token_provenance": str(generated.token_provenance),
            "trajectory_schema": "proposer_trajectory_v2_raw_policy_calls",
        },
    )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    first, last = stripped.find("{"), stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("proposer response must contain a JSON object")
