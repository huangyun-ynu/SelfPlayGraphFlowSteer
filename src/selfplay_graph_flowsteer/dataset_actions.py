from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observability import TaskSpec


class EnvironmentState(StrEnum):
    STATELESS = "stateless"
    STATEFUL = "stateful"


class SessionScope(StrEnum):
    NONE = "none"
    PER_AGENT = "per_agent"
    SHARED = "shared"


class ActionExecution(StrEnum):
    BATCH = "batch"
    SEQUENTIAL = "sequential"


class CommitPolicy(StrEnum):
    NONE = "none"
    SINGLE_COMMITTER = "single_committer"


class CommitActivation(StrEnum):
    NONE = "none"
    EXECUTE_ON_OUTPUT = "execute_on_output"
    COMMIT_PENDING_ON_OUTPUT = "commit_pending_on_output"
    EXPORT_LATEST_ARTIFACT = "export_latest_artifact"


@dataclass(frozen=True)
class DatasetActionAdapter:
    """Dataset-owned Worker action visibility and budget policy.

    The Director never sees or edits this policy. The Canvas applies it to every
    Worker node created for the task, and the runtime enforces the resulting
    action names before dispatching to an executor.
    """

    adapter_id: str
    datasets: tuple[str, ...]
    action_names: tuple[str, ...]
    initial_action_budget: int
    revision_action_budget: int
    total_action_budget: int
    environment_state: EnvironmentState | str = EnvironmentState.STATELESS
    session_scope: SessionScope | str = SessionScope.NONE
    action_execution: ActionExecution | str = ActionExecution.BATCH
    commit_policy: CommitPolicy | str = CommitPolicy.NONE
    commit_activation: CommitActivation | str = CommitActivation.NONE
    supports_parallel_reads: bool = True

    def __post_init__(self) -> None:
        adapter_id = self.adapter_id.strip().casefold()
        datasets = tuple(dict.fromkeys(value.strip().casefold() for value in self.datasets))
        actions = tuple(dict.fromkeys(value.strip() for value in self.action_names))
        environment_state = EnvironmentState(self.environment_state)
        session_scope = SessionScope(self.session_scope)
        action_execution = ActionExecution(self.action_execution)
        commit_policy = CommitPolicy(self.commit_policy)
        commit_activation = CommitActivation(self.commit_activation)
        if not adapter_id:
            raise ValueError("dataset action adapter_id cannot be empty")
        if not datasets or any(not value for value in datasets):
            raise ValueError("dataset action adapter requires dataset names")
        if any(not value for value in actions):
            raise ValueError("dataset action names cannot be empty")
        budgets = (
            int(self.initial_action_budget),
            int(self.revision_action_budget),
            int(self.total_action_budget),
        )
        if min(budgets) < 0:
            raise ValueError("dataset action budgets must be non-negative")
        if budgets[0] + budgets[1] > budgets[2]:
            raise ValueError("dataset phase action budgets exceed the total budget")
        if actions and budgets[2] <= 0:
            raise ValueError("visible dataset actions require a positive total budget")
        if not actions and any(budgets):
            raise ValueError("an empty dataset action set requires zero budgets")
        if environment_state is EnvironmentState.STATELESS and (
            session_scope is not SessionScope.NONE
            or action_execution is not ActionExecution.BATCH
            or commit_policy is not CommitPolicy.NONE
        ):
            raise ValueError("stateless adapters cannot declare sessions or commit policy")
        if commit_policy is CommitPolicy.SINGLE_COMMITTER and (
            environment_state is not EnvironmentState.STATEFUL
            or session_scope is not SessionScope.PER_AGENT
        ):
            raise ValueError("single_committer requires stateful per_agent sessions")
        if commit_policy is CommitPolicy.NONE and commit_activation is not CommitActivation.NONE:
            raise ValueError("commit activation requires a commit policy")
        if (
            commit_policy is CommitPolicy.SINGLE_COMMITTER
            and commit_activation is CommitActivation.NONE
        ):
            # Preserve the historical WebShop-compatible behavior for custom
            # adapters and old trace replays. New adapters declare this field.
            commit_activation = CommitActivation.EXECUTE_ON_OUTPUT
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "action_names", actions)
        object.__setattr__(self, "environment_state", environment_state)
        object.__setattr__(self, "session_scope", session_scope)
        object.__setattr__(self, "action_execution", action_execution)
        object.__setattr__(self, "commit_policy", commit_policy)
        object.__setattr__(self, "commit_activation", commit_activation)

    def capability_policy(self) -> dict[str, object]:
        return {
            "environment_state": self.environment_state.value,
            "session_scope": self.session_scope.value,
            "action_execution": self.action_execution.value,
            "commit_policy": self.commit_policy.value,
            "commit_activation": self.commit_activation.value,
            "supports_parallel_reads": bool(self.supports_parallel_reads),
        }


class DatasetActionRegistry:
    """Resolve exactly one Action Adapter from trusted task dataset metadata."""

    def __init__(
        self,
        adapters: Iterable[DatasetActionAdapter] = (),
        *,
        available_actions: Iterable[str] = (),
    ) -> None:
        self.available_actions = frozenset(str(value) for value in available_actions)
        self._by_dataset: dict[str, DatasetActionAdapter] = {}
        self._by_id: dict[str, DatasetActionAdapter] = {}
        for adapter in adapters:
            if adapter.adapter_id in self._by_id:
                raise ValueError(f"duplicate dataset action adapter: {adapter.adapter_id}")
            unavailable = set(adapter.action_names) - self.available_actions
            if unavailable:
                raise ValueError(
                    f"adapter {adapter.adapter_id} references unavailable actions: "
                    + ", ".join(sorted(unavailable))
                )
            self._by_id[adapter.adapter_id] = adapter
            for dataset in adapter.datasets:
                if dataset in self._by_dataset:
                    raise ValueError(f"multiple action adapters registered for dataset {dataset}")
                self._by_dataset[dataset] = adapter

    def resolve(self, task: TaskSpec) -> DatasetActionAdapter | None:
        dataset = str(task.metadata.get("dataset", "")).strip().casefold()
        # Frozen-context NQ keeps the canonical dataset name (so metrics and
        # budgets remain comparable) but deliberately has no runtime search.
        if dataset == "nq_open" and str(task.metadata.get("evidence_mode", "")).strip().casefold() in {
            "provided_context",
            "frozen_retrieval",
            "provided_context_inline",
        }:
            return self._by_id.get("nq_open_context")
        return self._by_dataset.get(dataset)

    def get(self, adapter_id: str) -> DatasetActionAdapter | None:
        return self._by_id.get(str(adapter_id).strip().casefold())

    def adapters(self) -> tuple[DatasetActionAdapter, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))


def default_dataset_action_registry(
    available_actions: Iterable[str],
    *,
    aime_budgets: tuple[int, int, int] = (3, 1, 4),
    retrieval_initial_budget: int = 3,
    webshop_budgets: tuple[int, int, int] = (12, 4, 16),
    webshop_staged_commit: bool = True,
    alfworld_budgets: tuple[int, int, int] = (50, 50, 100),
    swe_budgets: tuple[int, int, int] = (28, 12, 40),
) -> DatasetActionRegistry:
    """Build zero-action adapters and executable adapters whose actions exist."""

    available = tuple(dict.fromkeys(str(value) for value in available_actions))
    available_set = set(available)
    adapters: list[DatasetActionAdapter] = [
        DatasetActionAdapter(
            adapter_id="healthbench_professional",
            datasets=("healthbench_professional",),
            action_names=(),
            initial_action_budget=0,
            revision_action_budget=0,
            total_action_budget=0,
            environment_state=EnvironmentState.STATELESS,
            session_scope=SessionScope.NONE,
            action_execution=ActionExecution.BATCH,
            commit_policy=CommitPolicy.NONE,
            commit_activation=CommitActivation.NONE,
            supports_parallel_reads=True,
        ),
        DatasetActionAdapter(
            adapter_id="hotpotqa_context",
            datasets=("hotpotqa",),
            action_names=(),
            initial_action_budget=0,
            revision_action_budget=0,
            total_action_budget=0,
            environment_state=EnvironmentState.STATELESS,
            session_scope=SessionScope.NONE,
            action_execution=ActionExecution.BATCH,
            commit_policy=CommitPolicy.NONE,
            commit_activation=CommitActivation.NONE,
            supports_parallel_reads=True,
        ),
        DatasetActionAdapter(
            adapter_id="nq_open_context",
            datasets=("nq_open_context",),
            action_names=(),
            initial_action_budget=0,
            revision_action_budget=0,
            total_action_budget=0,
            environment_state=EnvironmentState.STATELESS,
            session_scope=SessionScope.NONE,
            action_execution=ActionExecution.BATCH,
            commit_policy=CommitPolicy.NONE,
            commit_activation=CommitActivation.NONE,
            supports_parallel_reads=True,
        ),
    ]
    aime_actions = ("symbolic_compute", "finite_search", "python_exec")
    if set(aime_actions) <= available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="aime",
                datasets=("aime",),
                action_names=aime_actions,
                initial_action_budget=aime_budgets[0],
                revision_action_budget=aime_budgets[1],
                total_action_budget=aime_budgets[2],
            )
        )
    if "search" in available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="retrieval_qa",
                datasets=("nq_open",),
                action_names=("search",),
                initial_action_budget=retrieval_initial_budget,
                revision_action_budget=1,
                total_action_budget=retrieval_initial_budget + 1,
            )
        )
    webshop_actions = ("webshop_search", "webshop_click")
    if set(webshop_actions) <= available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="webshop",
                datasets=("webshop",),
                action_names=webshop_actions,
                initial_action_budget=webshop_budgets[0],
                revision_action_budget=webshop_budgets[1],
                total_action_budget=webshop_budgets[2],
                environment_state=EnvironmentState.STATEFUL,
                session_scope=SessionScope.PER_AGENT,
                action_execution=ActionExecution.SEQUENTIAL,
                commit_policy=CommitPolicy.SINGLE_COMMITTER,
                commit_activation=(
                    CommitActivation.COMMIT_PENDING_ON_OUTPUT
                    if webshop_staged_commit
                    else CommitActivation.EXECUTE_ON_OUTPUT
                ),
                supports_parallel_reads=True,
            )
        )
    if "alfworld_step" in available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="alfworld",
                datasets=("alfworld",),
                action_names=("alfworld_step",),
                initial_action_budget=alfworld_budgets[0],
                revision_action_budget=alfworld_budgets[1],
                total_action_budget=alfworld_budgets[2],
                environment_state=EnvironmentState.STATEFUL,
                session_scope=SessionScope.PER_AGENT,
                action_execution=ActionExecution.SEQUENTIAL,
            )
        )
    swe_actions = (
        "swe_list",
        "swe_search",
        "swe_read",
        "swe_edit",
        "swe_apply_artifact",
        "swe_test",
        "swe_status",
    )
    if set(swe_actions) <= available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="swe_bench",
                datasets=("swe_bench", "swe-bench", "swebench"),
                action_names=swe_actions,
                initial_action_budget=swe_budgets[0],
                revision_action_budget=swe_budgets[1],
                total_action_budget=swe_budgets[2],
                environment_state=EnvironmentState.STATEFUL,
                session_scope=SessionScope.PER_AGENT,
                action_execution=ActionExecution.SEQUENTIAL,
                commit_policy=CommitPolicy.SINGLE_COMMITTER,
                commit_activation=CommitActivation.EXPORT_LATEST_ARTIFACT,
                supports_parallel_reads=False,
            )
        )
    return DatasetActionRegistry(adapters, available_actions=available)
