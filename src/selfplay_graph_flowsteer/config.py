from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_DATASET_MAX_TOTAL_TOKENS = {
    "aime": 65_536,
    "nq_open": 65_536,
    "hotpotqa": 65_536,
    "webshop": 350_000,
    "alfworld": 350_000,
    "healthbench_professional": 65_536,
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
    max_agents: int = 8
    max_rounds: int = 20
    max_total_tokens: int = 32_768
    max_total_tokens_by_dataset: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DATASET_MAX_TOTAL_TOKENS)
    )
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
    remaining_token_admission_enabled: bool = True
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

    def token_budget_for_dataset(self, dataset: object) -> tuple[str, int]:
        dataset_key = canonical_dataset_name(dataset)
        normalized = {
            canonical_dataset_name(name): int(limit)
            for name, limit in self.max_total_tokens_by_dataset.items()
        }
        return dataset_key, int(normalized.get(dataset_key, self.max_total_tokens))
