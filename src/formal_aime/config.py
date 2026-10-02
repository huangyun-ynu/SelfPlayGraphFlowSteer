from __future__ import annotations

from dataclasses import dataclass, field, replace

DEFAULT_DATASET_MAX_TOTAL_TOKENS = {
    "aime": 240_000,
    "math_hard": 240_000,
    "nq_open": 240_000,
    "hotpotqa": 240_000,
    "webshop": 350_000,
    "alfworld": 350_000,
    "healthbench_professional": 240_000,
    "swe_bench": 350_000,
}

_DATASET_ALIASES = {
    "aime": "aime",
    "math_hard": "math_hard",
    "math-hard": "math_hard",
    "mathhard": "math_hard",
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
    generation_audit_dir: str | None = None
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
    webshop_action_budget_policy: str | None = None
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
    math_summary_head_chars: int = 256
    math_summary_tail_chars: int = 1024
    math_answer_feedback_chars: int = 2000
    structural_repair_enabled: bool = True
    native_webshop_output_materialization: bool = False
    repair_recent_action_limit: int = 5
    semantic_no_progress_limit: int = 2
    output_selection_budget: int = 1
    structural_exploration_policy: str = "off"
    bidirectional_revision_policy: str = "always"
    bidirectional_revision_confidence_threshold: float = 0.8

    def __post_init__(self) -> None:
        self.max_total_tokens_by_dataset = {
            **DEFAULT_DATASET_MAX_TOTAL_TOKENS,
            **{canonical_dataset_name(key): value for key, value in self.max_total_tokens_by_dataset.items()},
        }
        if self.submission_protocol not in {'legacy', 'unified_task_result_v1'}:
            raise ValueError('unsupported submission_protocol')
        protocols = {canonical_dataset_name(key): value for key, value in self.submission_protocol_by_dataset.items()}
        for dataset, protocol in protocols.items():
            if dataset not in DEFAULT_DATASET_MAX_TOTAL_TOKENS:
                raise ValueError('dataset submission protocol override requires a supported dataset')
            if protocol not in {'legacy', 'unified_task_result_v1'}:
                raise ValueError('unsupported dataset submission_protocol')
        if self.alfworld_terminal_candidate_policy not in {'off', 'finish_only_v1'}:
            raise ValueError('unknown ALFWorld terminal candidate policy')
        if self.max_recovery_executions < 0:
            raise ValueError('max_recovery_executions must be non-negative')
        if self.action_budget_policy not in {'phase_split_v1', 'shared_total_v1'}:
            raise ValueError('unknown canvas.action_budget_policy')
        if self.webshop_action_budget_policy not in {None, 'phase_split_v1', 'shared_total_v1'}:
            raise ValueError('unknown canvas.webshop_action_budget_policy')
        from .budget_policy import REPORTED_USAGE_DATASETS
        normalized_policies = {}
        for dataset, policy in self.worker_token_budget_by_dataset.items():
            name = canonical_dataset_name(dataset)
            if name not in REPORTED_USAGE_DATASETS or not isinstance(policy, dict):
                raise ValueError('unsupported dataset for reported Worker usage policy')
            if name in normalized_policies:
                raise ValueError('duplicate dataset Worker usage policy')
            normalized_policies[name] = dict(policy)
            if policy.get('policy') != 'reported_usage_threshold_v1':
                raise ValueError('unknown Worker usage policy')
            if policy.get('accounting_scope', 'question_attempt') != 'question_attempt':
                raise ValueError('reported usage requires a question-attempt account')
            if type(policy.get('max_unsettled_attempts', 2)) is not int or policy.get('max_unsettled_attempts', 2) <= 0:
                raise ValueError('max_unsettled_attempts must be positive')
            if type(policy.get('max_inflight_requests', 1)) is not int or policy.get('max_inflight_requests', 1) != 1:
                raise ValueError('reported usage requires one local in-flight request per question')
            if policy.get('unknown_usage_policy', 'continue_bounded') != 'continue_bounded':
                raise ValueError('unsupported unknown usage policy')
            if 'start_threshold' in policy and (type(policy['start_threshold']) is not int or policy['start_threshold'] <= 0):
                raise ValueError('start_threshold must be a positive integer')
            if policy.get('start_threshold', self.token_budget_for_dataset(name)[1]) != self.token_budget_for_dataset(name)[1]:
                raise ValueError('start_threshold must match max_total_tokens_by_dataset')
        from .budget_policy import default_usage_policy
        for name in REPORTED_USAGE_DATASETS:
            normalized_policies[name] = {
                **default_usage_policy(self.token_budget_for_dataset(name)[1]),
                **normalized_policies.get(name, {}),
            }
        self.worker_token_budget_by_dataset = normalized_policies
        limits = [*self.max_director_edits_by_dataset.values()]
        if self.max_director_edits is not None:
            limits.append(self.max_director_edits)
        if any((type(value) is not int or value < 0 for value in limits)):
            raise ValueError('Director edit limits must be non-negative integers')
        if self.director_budget_policy not in {'rounds_v1', 'edits_v1'}:
            raise ValueError('unknown Director budget policy')
        if self.director_budget_policy == 'edits_v1' and (self.submission_protocol != 'unified_task_result_v1' or self.max_director_edits is None):
            raise ValueError('edits_v1 requires unified submission and a finite default edit limit')

    def for_dataset(self, dataset: object) -> CanvasConfig:
        overrides = {canonical_dataset_name(key): value
                     for key, value in self.submission_protocol_by_dataset.items()}
        protocol = overrides.get(canonical_dataset_name(dataset))
        changes = {"submission_protocol": protocol} if protocol is not None else {}
        if canonical_dataset_name(dataset) == "webshop" and self.webshop_action_budget_policy:
            changes["action_budget_policy"] = self.webshop_action_budget_policy
        return replace(self, **changes) if changes else self

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
