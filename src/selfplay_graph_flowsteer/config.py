from __future__ import annotations

from dataclasses import dataclass, field, replace

DEFAULT_DATASET_MAX_TOTAL_TOKENS = {
    "aime": 240_000,
    "nq_open": 240_000,
    "hotpotqa": 240_000,
    "webshop": 350_000,
    "alfworld": 350_000,
    "healthbench_professional": 240_000,
    "swe_bench": 350_000,
}

_DATASET_ALIASES = {
    "aime": "aime",
    "nq": "nq_open",
    "nq-open": "nq_open",
    "nq_open": "nq_open",
    "natural_questions": "nq_open",
    "hotpot": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "webshop": "webshop",
    "alfworld": "alfworld",
    "healthbench": "healthbench_professional",
    "healthbench-professional": "healthbench_professional",
    "healthbench_professional": "healthbench_professional",
    "swe-bench": "swe_bench",
    "swe_bench": "swe_bench",
    "swebench": "swe_bench",
}


def canonical_dataset_name(dataset: object) -> str:
    key = str(dataset or "").strip().casefold()
    return _DATASET_ALIASES.get(key, key)


@dataclass(frozen=True)
class ModelRoleConfig:
    model: str = "Qwen3.5-9B"
    system_prompt: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 2048
    enable_thinking: bool = False
    reasoning_effort: str | None = None
    api_surface: str = "chat_completions"


@dataclass
class ModelGatewayConfig:
    """Configuration for one model instance, never a cross-policy model bundle."""

    base_url: str = "http://127.0.0.1:8003/v1"
    api_key: str = "EMPTY"
    # Dataset-scoped credentials keep one logical route/model while selecting
    # the account required by a benchmark. Secrets are resolved before this
    # object is built and are never included in manifests.
    api_keys_by_dataset: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    request_profile: str = "qwen"
    # Empty for local policy models. Runtime routes set this so a rollout can
    # apply a narrowly scoped route/dataset timeout override.
    route_name: str = ""
    # Independent requests remain separate at the protocol layer.  This limit
    # only controls how many may be in flight to one endpoint so vLLM/SGLang can
    # continuously batch local generations without flooding remote providers.
    max_concurrency: int = 64
    # Dataset-scoped credentials may have a different provider capacity from
    # the route's default account.  Requests using the same endpoint,
    # credential, and effective limit share one process-wide gate.
    max_concurrency_by_dataset: dict[str, int] = field(default_factory=dict)
    # Local policy rollouts use an explicit provider seed so the recorded
    # rollout seed is the seed that actually governed sampling.  Fixed remote
    # runtime models leave this unset.
    sampling_seed: int | None = None
    network_path: str = "configured_proxy"
    stream: bool = False
    user_agent: str | None = None
    roles: dict[str, ModelRoleConfig] = field(
        default_factory=lambda: {
            "worker": ModelRoleConfig(model="Qwen3.5-9B", temperature=0.0),
        }
    )


@dataclass
class CanvasConfig:
    # Historical traces keep legacy semantics; new runs explicitly select v1.
    submission_protocol: str = "legacy"
    submission_protocol_by_dataset: dict[str, str] = field(default_factory=dict)
    submission_journal_dir: str = "state/submissions"
    alfworld_terminal_candidate_policy: str = "off"
    max_recovery_executions: int = 2
    action_budget_policy: str = "phase_split_v1"
    max_agents: int = 8
    max_rounds: int = 20
    director_budget_policy: str = "rounds_v1"
    # Successful graph mutations, including initial construction. None adds no
    # separate cap; invalid/no-op decisions still consume max_rounds.
    max_director_edits: int | None = None
    max_director_edits_by_dataset: dict[str, int] = field(default_factory=dict)
    max_total_tokens: int = 32_768
    max_total_tokens_by_dataset: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DATASET_MAX_TOTAL_TOKENS)
    )
    worker_token_budget_by_dataset: dict[str, dict[str, object]] = field(default_factory=dict)
    relay_max_chars: int = 4000
    feedback_max_chars: int = 6000
    artifact_summary_max_chars: int = 320
    structural_repair_enabled: bool = True
    graph_growth_token_reserve: int = 8192
    remaining_time_admission_enabled: bool = True
    worker_latency_quantile: float = 0.95
    worker_latency_window: int = 64
    worker_latency_min_samples: int = 3
    worker_latency_cold_start_s: float = 30.0
    finalization_time_reserve_s: float = 20.0
    finalization_time_reserve_by_dataset: dict[str, float] = field(
        default_factory=lambda: {
            "healthbench_professional": 180.0,
            "swe_bench": 120.0,
        }
    )
    # Includes WebShop serialized-request estimates and closure token reservations.
    # False retains post-execution actual-usage checks and environment action limits.
    remaining_token_admission_enabled: bool = True
    native_webshop_output_materialization: bool = False
    worker_token_quantile: float = 0.95
    worker_token_window: int = 64
    worker_token_min_samples: int = 3
    worker_token_cold_start: int = 4096
    finalization_token_reserve: int = 2048
    repair_recent_action_limit: int = 5
    semantic_no_progress_limit: int = 2
    output_selection_budget: int = 1
    structural_exploration_policy: str = "off"
    bidirectional_revision_policy: str = "always"
    bidirectional_revision_confidence_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.submission_protocol not in {"legacy", "unified_task_result_v1"}:
            raise ValueError("unsupported submission_protocol")
        protocols = {canonical_dataset_name(key): value
                     for key, value in self.submission_protocol_by_dataset.items()}
        for dataset, protocol in protocols.items():
            if dataset not in {"swe_bench", "alfworld"}:
                raise ValueError("dataset submission protocol overrides support SWE and ALFWorld only")
            if protocol not in {"legacy", "unified_task_result_v1"}:
                raise ValueError("unsupported dataset submission_protocol")
        if self.alfworld_terminal_candidate_policy not in {"off", "finish_only_v1"}:
            raise ValueError("unknown ALFWorld terminal candidate policy")
        if self.max_recovery_executions < 0:
            raise ValueError("max_recovery_executions must be non-negative")
        if self.action_budget_policy not in {"phase_split_v1", "shared_total_v1"}:
            raise ValueError("unknown canvas.action_budget_policy")
        for dataset, policy in self.worker_token_budget_by_dataset.items():
            if canonical_dataset_name(dataset) not in {"swe_bench", "alfworld"} or not isinstance(policy, dict):
                raise ValueError("reported Worker usage policy is supported for SWE and ALFWorld only")
            if (canonical_dataset_name(dataset) == "alfworld"
                    and protocols.get("alfworld", self.submission_protocol) != "unified_task_result_v1"):
                raise ValueError("ALFWorld reported usage requires unified_task_result_v1")
            if policy.get("policy") != "reported_usage_threshold_v1":
                raise ValueError("unknown SWE Worker usage policy")
            if policy.get("accounting_scope", "question_attempt") != "question_attempt":
                raise ValueError("SWE reported usage requires a question-attempt account")
            if type(policy.get("max_unsettled_attempts", 2)) is not int or policy.get("max_unsettled_attempts", 2) <= 0:
                raise ValueError("max_unsettled_attempts must be positive")
            if type(policy.get("max_inflight_requests", 1)) is not int or policy.get("max_inflight_requests", 1) != 1:
                raise ValueError("SWE reported usage requires one local in-flight request per question")
            if policy.get("unknown_usage_policy", "continue_bounded") != "continue_bounded":
                raise ValueError("unsupported unknown usage policy")
            if "start_threshold" in policy and (
                type(policy["start_threshold"]) is not int or policy["start_threshold"] <= 0
            ):
                raise ValueError("start_threshold must be a positive integer")
        limits = [*self.max_director_edits_by_dataset.values()]
        if self.max_director_edits is not None:
            limits.append(self.max_director_edits)
        if any(type(value) is not int or value < 0 for value in limits):
            raise ValueError("Director edit limits must be non-negative integers")
        if self.director_budget_policy not in {"rounds_v1", "edits_v1"}:
            raise ValueError("unknown Director budget policy")
        if self.director_budget_policy == "edits_v1" and (
                self.submission_protocol != "unified_task_result_v1" or self.max_director_edits is None):
            raise ValueError("edits_v1 requires unified submission and a finite default edit limit")

    def for_dataset(self, dataset: object) -> CanvasConfig:
        overrides = {canonical_dataset_name(key): value
                     for key, value in self.submission_protocol_by_dataset.items()}
        protocol = overrides.get(canonical_dataset_name(dataset))
        return (replace(self, submission_protocol=protocol)
                if protocol is not None else self)

    def director_edit_limit(self, dataset: object) -> int | None:
        overrides = {canonical_dataset_name(key): value
                     for key, value in self.max_director_edits_by_dataset.items()}
        return overrides.get(canonical_dataset_name(dataset), self.max_director_edits)

    def token_budget_for_dataset(self, dataset: object) -> tuple[str, int]:
        dataset_key = canonical_dataset_name(dataset)
        normalized = {
            canonical_dataset_name(name): int(limit)
            for name, limit in self.max_total_tokens_by_dataset.items()
        }
        return dataset_key, int(normalized.get(dataset_key, self.max_total_tokens))

    def worker_usage_policy(self, dataset: object) -> dict[str, object] | None:
        name = canonical_dataset_name(dataset)
        for key, value in self.worker_token_budget_by_dataset.items():
            if canonical_dataset_name(key) == name:
                return dict(value)
        return None
