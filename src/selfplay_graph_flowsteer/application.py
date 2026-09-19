from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shlex
import tomllib
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .adaptive import AdaptiveSolverResult, AdaptiveWorkflowSolver
from .agent_tools import (
    FiniteSearchTool,
    PythonExecutionTool,
    SearchServiceTool,
    SymbolicComputeTool,
)
from .alfworld import (
    ALFWorldEnvironmentVerifier,
    ALFWorldSessionLifecycle,
    ALFWorldStepTool,
    LocalALFWorldClient,
    alfworld_lifecycles,
)
from .answer_submission import AnswerFinalizer, AnswerSubmissionConfig
from .config import (
    DEFAULT_DATASET_MAX_TOTAL_TOKENS,
    CanvasConfig,
    ModelGatewayConfig,
    ModelRoleConfig,
    canonical_dataset_name,
)
from .dataset_actions import default_dataset_action_registry
from .dataset_adapters import (
    HEALTHBENCH_PROFESSIONAL_JUDGE_MODEL,
    HealthBenchOfficialRubricVerifier,
    HealthBenchRubricVerifier,
    solver_task_text,
)
from .deadline import RolloutDeadline, WorkerWallClockLimitExceeded
from .delegation import DUPLICATE_RESPONSIBILITY_POLICIES
from .healthbench_audit import HealthBenchJudgeAuditStore
from .latency import RouteLatencyTracker, RouteTokenTracker
from .llm import (
    BinaryChoiceResponse,
    ChatBackend,
    GeminiNativeBackend,
    MockBackend,
    OpenAICompatibleBackend,
    request_dataset,
)
from .observability import (
    AutoVerifier,
    ExactMatchVerifier,
    FlowSteerQAVerifier,
    JSONLTraceStore,
    MultiAnswerExactMatchVerifier,
    MultipleChoiceVerifier,
    NumericVerifier,
    TokenF1Verifier,
    TaskSpec,
    VerificationResult,
    Verifier,
    task_requires_reference,
    task_to_public_dict,
)
from .pats import PatsConfig
from .protocol_reward import LEGACY_REWARD_VERSION, DirectorRewardConfig
from .rollouts import Tokenizer
from .route_health import PersistentRouteCircuitOpenError, RouteHealthStore
from .runtime import (
    WORKER_BACKEND_FAILURE_SENTINEL,
    ModelAgentExecutor,
    MultiAgentRuntime,
    RoutedModelAgentExecutor,
    artifact_backend_failure_records,
)
from .selfplay import AlternatingSnapshots, FixedPoolQwenProposer, QwenTaskProposer
from .skills import (
    E5SkillEmbedder,
    SESASolverSkillDistiller,
    SolverFailureCase,
    SolverSkillBank,
    SolverSkillLifecycle,
)
from .swebench import (
    CodeArtifactStore,
    SSHSWEHarnessBackend,
    SWEOutcomeVerifier,
    SWEWorkspaceLifecycle,
    TrustedSWEVerifierRegistry,
    public_swe_evaluation,
    shared_tencent_cvm_lease,
    swe_lifecycles,
    swe_tools,
)
from .webshop import (
    WebShopClickTool,
    WebShopEnvironmentVerifier,
    WebShopHTTPClient,
    WebShopSearchTool,
    WebShopSessionLifecycle,
    webshop_lifecycles,
)
from .webshop_budget import budget_partition


class GraphEvaluationIncompleteError(RuntimeError):
    """A replay did not complete required graph execution; no trusted score."""


class GraphEvaluationBackendError(RuntimeError):
    """A graph re-execution is invalid because a required Worker failed."""

    def __init__(self, failure: dict[str, Any]) -> None:
        self.failure = dict(failure)
        routes = ",".join(str(value) for value in failure.get("routes", ()))
        super().__init__("graph evaluation Worker backend failed; routes=" + (routes or "unknown"))


REMOTE_RUNTIME_MAX_CONCURRENCY = 16
_REMOTE_RUNTIME_20_CONCURRENCY_MODELS = frozenset({"gpt-6-astra"})
_REMOTE_RUNTIME_30_CONCURRENCY_PREFIXES = ("deepseek", "minimax")


def _allowed_physical_gpu_ids() -> set[int]:
    """Return the explicit per-run physical GPU allowlist (default: GPU 0)."""

    raw = os.environ.get("SPGFS_ALLOWED_PHYSICAL_GPUS", "0")
    try:
        allowed = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError("SPGFS_ALLOWED_PHYSICAL_GPUS must contain integer GPU ids") from exc
    if not allowed or any(gpu < 0 for gpu in allowed):
        raise ValueError("SPGFS_ALLOWED_PHYSICAL_GPUS must contain non-negative GPU ids")
    return allowed


@dataclass(frozen=True)
class RoleModelConfig:
    """One independently served and checkpointed model-role boundary."""

    base_url: str
    served_model: str
    base_model_path: Path
    checkpoint_path: Path
    api_key: str = "EMPTY"
    timeout_s: float = 120.0
    trainable: bool = True

    def validate(self, name: str) -> None:
        if not self.base_url.strip():
            raise ValueError(f"models.{name}.base_url cannot be empty")
        if not self.served_model.strip():
            raise ValueError(f"models.{name}.served_model cannot be empty")
        if self.timeout_s <= 0:
            raise ValueError(f"models.{name}.timeout_s must be positive")

    def identity(self) -> tuple[str, str]:
        return self.base_url.rstrip("/"), self.served_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "served_model": self.served_model,
            "base_model_path": str(self.base_model_path),
            "checkpoint_path": str(self.checkpoint_path),
            "trainable": self.trainable,
        }


@dataclass(frozen=True)
class FixedRuntimeConfig:
    """Fixed inference environment used by workers, MANTA and skill distillation."""

    base_url: str
    served_model: str
    model_path: Path | None
    api_key: str = "EMPTY"
    api_keys_by_dataset: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    request_profile: str = "qwen"
    healthbench_grader_reasoning_effort: str | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool = False
    max_tokens: int = 2048
    api_surface: str = "chat_completions"
    user_agent: str | None = None
    network_path: str = "configured_proxy"
    stream: bool = False
    max_concurrency: int = 4
    max_concurrency_by_dataset: dict[str, int] = field(default_factory=dict)
    managed_locally: bool = True
    frozen: bool = True

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("runtime.max_tokens must be positive")
        if not self.base_url.strip() or not self.served_model.strip():
            raise ValueError("runtime endpoint and served_model cannot be empty")
        if self.user_agent is not None and (
            not self.user_agent or any(ord(c) < 32 or ord(c) > 126 for c in self.user_agent)
        ):
            raise ValueError("runtime.user_agent must be nonempty printable ASCII")
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")
        if any(
            not canonical_dataset_name(dataset) or not str(api_key).strip()
            for dataset, api_key in self.api_keys_by_dataset.items()
        ):
            raise ValueError("runtime dataset API-key overrides must be nonempty")
        if self.max_concurrency <= 0:
            raise ValueError("runtime.max_concurrency must be positive")
        if any(
            not canonical_dataset_name(dataset) or int(limit) <= 0
            for dataset, limit in self.max_concurrency_by_dataset.items()
        ):
            raise ValueError("runtime dataset concurrency overrides must be positive")
        model_name = self.served_model.casefold()
        remote_limit = (
            30
            if model_name.startswith(_REMOTE_RUNTIME_30_CONCURRENCY_PREFIXES)
            else (
                20
                if model_name in _REMOTE_RUNTIME_20_CONCURRENCY_MODELS
                else REMOTE_RUNTIME_MAX_CONCURRENCY
            )
        )
        if not self.managed_locally and self.max_concurrency > remote_limit:
            raise ValueError(
                f"externally managed runtime.max_concurrency must not exceed {remote_limit}"
            )
        if not self.managed_locally and any(
            int(limit) > remote_limit for limit in self.max_concurrency_by_dataset.values()
        ):
            raise ValueError(
                f"externally managed runtime dataset concurrency must not exceed {remote_limit}"
            )
        if self.network_path not in {"direct", "configured_proxy"}:
            raise ValueError("runtime.network_path must be direct or configured_proxy")
        if self.stream and (
            self.api_surface != "chat_completions" or self.request_profile != "generic"
        ):
            raise ValueError("runtime.stream requires generic Chat Completions")
        if self.request_profile not in {"qwen", "generic", "gemini", "responses_text"}:
            raise ValueError("runtime.request_profile must be qwen, generic, gemini or responses_text")
        if self.api_surface not in {"chat_completions", "responses"}:
            raise ValueError("runtime.api_surface must be chat_completions or responses")
        if self.request_profile == "responses_text" and self.api_surface != "responses":
            raise ValueError("responses_text requires the Responses API")
        if self.api_surface == "responses" and self.request_profile not in {"generic", "responses_text"}:
            raise ValueError("Responses runtimes require generic or responses_text")
        if self.healthbench_grader_reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError(
                "runtime.healthbench_grader_reasoning_effort must be low, medium, or high"
            )
        if self.reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError("runtime.reasoning_effort must be low, medium, or high")
        if self.managed_locally and self.model_path is None:
            raise ValueError("locally managed runtime requires model_path")
        if not self.frozen:
            raise ValueError("runtime must remain fixed during Proposer/Solver collection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "served_model": self.served_model,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "max_concurrency": self.max_concurrency,
            "dataset_concurrency_overrides": {
                canonical_dataset_name(dataset): int(limit)
                for dataset, limit in sorted(self.max_concurrency_by_dataset.items())
            },
            "managed_locally": self.managed_locally,
            "frozen": self.frozen,
            "healthbench_grader_reasoning_effort": (self.healthbench_grader_reasoning_effort),
            "reasoning_effort": self.reasoning_effort,
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "api_surface": self.api_surface,
            "dataset_api_key_overrides": sorted(self.api_keys_by_dataset),
            "user_agent": self.user_agent,
            "network_path": self.network_path,
            "stream": self.stream,
        }


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = False
    service_url: str = "http://127.0.0.1:8010/retrieve"
    top_k: int = 3
    timeout_s: float = 120.0
    max_tool_rounds: int = 3

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.service_url.startswith(("http://", "https://")):
            raise ValueError("retrieval.service_url must be an HTTP(S) URL")
        if min(self.top_k, self.max_tool_rounds) <= 0 or self.timeout_s <= 0:
            raise ValueError("retrieval limits and timeout must be positive")


@dataclass(frozen=True)
class AIMEActionConfig:
    enabled: bool = False
    timeout_s: float = 5.0
    max_output_chars: int = 8000
    max_code_chars: int = 12000
    memory_limit_mb: int = 1024
    max_initial_calls: int = 3
    max_revision_calls: int = 1
    max_total_calls: int = 4

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.timeout_s <= 0:
            raise ValueError("aime_actions.timeout_s must be positive")
        if min(self.max_output_chars, self.max_code_chars, self.memory_limit_mb) <= 0:
            raise ValueError("aime_actions limits must be positive")
        if min(self.max_initial_calls, self.max_revision_calls, self.max_total_calls) < 0:
            raise ValueError("aime_actions call budgets must be non-negative")
        if self.max_initial_calls + self.max_revision_calls > self.max_total_calls:
            raise ValueError("aime_actions phase call budgets exceed max_total_calls")


@dataclass(frozen=True)
class WebShopConfig:
    enabled: bool = False
    service_url: str = "http://127.0.0.1:8020"
    timeout_s: float = 10.0
    max_observation_chars: int = 4_000
    max_query_chars: int = 500
    max_initial_calls: int = 12
    max_revision_calls: int = 4
    max_total_calls: int = 16
    staged_commit_enabled: bool = True
    max_pending_sessions: int = 8
    pending_ttl_s: float = 900.0
    search_observation_mode: str = "structured_only"

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.service_url.startswith(("http://", "https://")):
            raise ValueError("webshop.service_url must be an HTTP(S) URL")
        if (
            min(
                self.timeout_s,
                self.max_query_chars,
                self.max_pending_sessions,
                self.pending_ttl_s,
            )
            <= 0
        ):
            raise ValueError("webshop limits and timeout must be positive")
        if self.max_observation_chars < 0:
            raise ValueError("webshop.max_observation_chars must be non-negative")
        budgets = (
            self.max_initial_calls,
            self.max_revision_calls,
            self.max_total_calls,
        )
        if min(budgets) < 0:
            raise ValueError("webshop call budgets must be non-negative")
        if budgets[0] + budgets[1] > budgets[2]:
            raise ValueError("webshop phase call budgets exceed max_total_calls")
        if self.search_observation_mode not in {
            "legacy",
            "retain_page_text",
            "structured_only",
        }:
            raise ValueError(
                "webshop.search_observation_mode must be legacy, retain_page_text, "
                "or structured_only"
            )


@dataclass(frozen=True)
class ALFWorldConfig:
    enabled: bool = False
    data_root: Path = Path("datasets/alfworld/data_assets/json_2.1.1")
    max_episode_steps: int = 50
    max_rollout_steps: int = 400
    max_observation_chars: int = 12_000
    max_initial_calls: int = 50
    max_revision_calls: int = 50
    max_total_calls: int = 100
    worker_guidance_policy: str = "factual_memory_v1"

    def validate(self) -> None:
        if self.worker_guidance_policy not in {
            "factual_memory_v1",
            "raw_state_v1",
            "legacy_full_v1",
        }:
            raise ValueError(
                "alfworld.worker_guidance_policy must be 'factual_memory_v1', "
                "'raw_state_v1', or 'legacy_full_v1'"
            )
        if not self.enabled:
            return
        if not self.data_root.is_dir():
            raise ValueError("alfworld.data_root must be an existing directory")
        if (
            min(
                self.max_episode_steps,
                self.max_rollout_steps,
                self.max_observation_chars,
            )
            <= 0
        ):
            raise ValueError("alfworld limits must be positive")
        budgets = (
            self.max_initial_calls,
            self.max_revision_calls,
            self.max_total_calls,
        )
        if min(budgets) < 0:
            raise ValueError("alfworld call budgets must be non-negative")
        if budgets[0] + budgets[1] > budgets[2]:
            raise ValueError("alfworld phase call budgets exceed max_total_calls")


@dataclass(frozen=True)
class SWEConfig:
    enabled: bool = False
    repo_cache_root: Path = Path("state/swe/repo-cache")
    workspace_root: Path = Path("state/swe/workspaces")
    artifact_store_root: Path = Path("state/swe/artifacts")
    verifier_registry_path: Path = Path("state/private/swe/verifier-registry.json")
    lifecycle_log_path: Path = Path("state/private/swe/lifecycle.jsonl")
    verifier_log_path: Path = Path("state/private/swe/verifier-client.jsonl")
    verifier_host: str = ""
    verifier_user: str = "sweeval"
    verifier_identity_file: Path = Path("config/private/swe_identity")
    verifier_known_hosts_file: Path = Path(
        "state/deployments/20260824-swe-verifier-tencent/known_hosts"
    )
    cvm_auto_start: bool = False
    cvm_auto_stop: bool = False
    cvm_region: str = "ap-singapore"
    cvm_instance_id: str = "ins-5n1zolfw"
    cvm_secret_id_env: str = "TENCENTCLOUD_SECRET_ID"
    cvm_secret_key_env: str = "TENCENTCLOUD_SECRET_KEY"
    cvm_endpoint: str = "cvm.tencentcloudapi.com"
    cvm_timeout_s: float = 600.0
    cvm_poll_s: float = 5.0
    dataset_revision: str = ""
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 720.0
    local_test_timeout_s: float = 60.0
    max_output_chars: int = 12_000
    max_file_chars: int = 200_000
    max_patch_bytes: int = 2_000_000
    max_initial_calls: int = 20
    max_revision_calls: int = 12
    max_total_calls: int = 32
    duplicate_responsibility_policy: str = "record_only"
    test_profiles: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"python_syntax": ("python", "-m", "py_compile")}
    )

    def validate(self) -> None:
        if self.duplicate_responsibility_policy not in DUPLICATE_RESPONSIBILITY_POLICIES:
            raise ValueError(
                "swe.duplicate_responsibility_policy must be one of: "
                + ", ".join(sorted(DUPLICATE_RESPONSIBILITY_POLICIES))
            )
        if not self.enabled:
            return
        required_directories = {
            "repo_cache_root": self.repo_cache_root,
        }
        for name, path in required_directories.items():
            if not path.is_dir():
                raise ValueError(f"swe.{name} must be an existing directory")
        required_files = {
            "verifier_registry_path": self.verifier_registry_path,
            "verifier_identity_file": self.verifier_identity_file,
            "verifier_known_hosts_file": self.verifier_known_hosts_file,
        }
        for name, path in required_files.items():
            if not path.is_file():
                raise ValueError(f"swe.{name} must be an existing file")
        if self.verifier_identity_file.stat().st_mode & 0o077:
            raise ValueError("swe.verifier_identity_file must not be group/world accessible")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.dataset_revision):
            raise ValueError("swe.dataset_revision must be a pinned hexadecimal revision")
        if not self.verifier_host.strip() or not self.verifier_user.strip():
            raise ValueError("swe verifier host and user cannot be empty")
        if self.cvm_auto_start or self.cvm_auto_stop:
            if self.cvm_auto_stop and not self.cvm_auto_start:
                raise ValueError("swe.cvm_auto_stop requires swe.cvm_auto_start")
            if not self.cvm_region.strip() or not self.cvm_instance_id.strip():
                raise ValueError("SWE CVM region and instance_id are required when enabled")
            if not self.cvm_secret_id_env.strip() or not self.cvm_secret_key_env.strip():
                raise ValueError("SWE CVM credential environment variable names are required")
            if min(self.cvm_timeout_s, self.cvm_poll_s) <= 0:
                raise ValueError("SWE CVM timeout and poll interval must be positive")
        if (
            min(
                self.connect_timeout_s,
                self.request_timeout_s,
                self.local_test_timeout_s,
                self.max_output_chars,
                self.max_file_chars,
                self.max_patch_bytes,
            )
            <= 0
        ):
            raise ValueError("swe timeouts and size limits must be positive")
        budgets = (
            self.max_initial_calls,
            self.max_revision_calls,
            self.max_total_calls,
        )
        if min(budgets) < 0 or budgets[0] + budgets[1] > budgets[2]:
            raise ValueError("swe phase call budgets are invalid")
        if not self.test_profiles or any(
            not name.strip() or not command or any(not str(part) for part in command)
            for name, command in self.test_profiles.items()
        ):
            raise ValueError("swe.test_profiles must contain non-empty fixed argv lists")
        for path in (self.lifecycle_log_path, self.verifier_log_path):
            if "private" not in {part.casefold() for part in path.parts}:
                raise ValueError("SWE audit logs must be stored under a private path")


def _default_role(name: str, port: int, *, trainable: bool = True) -> RoleModelConfig:
    return RoleModelConfig(
        base_url=f"http://127.0.0.1:{port}/v1",
        served_model="Qwen3.5-9B",
        base_model_path=Path("models/Qwen3.5-9B"),
        checkpoint_path=Path(f"state/checkpoints/{name}"),
        trainable=trainable,
    )


@dataclass(frozen=True)
class AdaptiveApplicationConfig:
    proposer_model: RoleModelConfig = field(default_factory=lambda: _default_role("proposer", 8001))
    solver_model: RoleModelConfig = field(default_factory=lambda: _default_role("solver", 8002))
    runtime: FixedRuntimeConfig = field(
        default_factory=lambda: FixedRuntimeConfig(
            base_url="http://127.0.0.1:8003/v1",
            served_model="Qwen3.5-9B",
            model_path=Path("models/Qwen3.5-9B"),
        )
    )
    runtime_name: str = "default"
    additional_runtimes: dict[str, FixedRuntimeConfig] = field(default_factory=dict)
    worker_runtime_routes: tuple[str, ...] = ("default",)
    runtime_endpoint_pools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    endpoint_pool_retry_attempts: int = 2
    endpoint_pool_retry_backoff_s: float = 1.0
    endpoint_pool_member_queue_wait_s: float = 0.5
    skill_distiller_runtime: str = "default"
    route_health_path: Path = Path("state/route_health.json")
    route_health_cooldown_s: float = 600.0
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    verifier: str = "none"
    mace_enabled: bool = False  # Retired compatibility field; True is rejected.
    mace_alpha: float = 1.0
    mace_regularization: float = 1.0
    seed: int = 0
    # Legacy input only: peer statistics are no longer loaded or written.
    mace_statistics_path: Path = Path("state/mace_statistics.json")
    mace_model_statistics_path: Path = Path("state/mace_model_statistics.json")
    skillbank_enabled: bool = True
    skillbank_mode: str = "legacy"
    skillbank_usage: str = "always"
    # Runtime context set by collection entry points, never read from TOML.
    skillbank_training: bool = False
    skillbank_snapshot_frozen: bool = False
    pats: PatsConfig = field(default_factory=PatsConfig)
    skillbank_prompt_token_budget: int = 1024
    skillbank_retrieval_min_score: float | None = None
    skillbank_path: Path = Path("state/solver_skillbank.json")
    skill_cases_path: Path = Path("state/solver_skill_cases.json")
    skillbank_max_skills: int = 800
    skillbank_dedup_threshold: float = 0.93
    skillbank_dedup_review_threshold: float = 0.90
    skillbank_retrieve_top_k: int = 3
    skillbank_min_retrieved_for_evict: int = 3
    skillbank_update_freq: int = 10
    skillbank_pending_queue_max: int = 300
    skillbank_min_pending: int = 20
    skillbank_generate_per_update: int = 50
    skillbank_distill_concurrency: int = 2
    skillbank_activation_policy: str = "paired"
    skillbank_embedding_model_path: str | Path | None = "intfloat/e5-base-v2"
    trace_path: Path = Path("state/traces.jsonl")
    healthbench_judge_audit_path: Path = Path("state/private/healthbench_judge_audit")
    healthbench_judge_input_cost_per_million: float | None = None
    healthbench_judge_output_cost_per_million: float | None = None
    healthbench_auto_grader_mode: str = "official"
    # Provider attestation must report this stable model identity.
    healthbench_judge_expected_model: str = HEALTHBENCH_PROFESSIONAL_JUDGE_MODEL
    healthbench_judge_runtime_route: str | None = None
    persist_runtime_updates: bool = True
    allocated_gpu_ids: tuple[int, ...] = (0,)
    proposer_gpu_id: int = 0
    solver_gpu_id: int = 0
    runtime_gpu_id: int = 0
    proposer_service_gpu_memory_utilization: float = 0.30
    solver_service_gpu_memory_utilization: float = 0.35
    allow_policy_gpu_colocation: bool = True
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    aime_actions: AIMEActionConfig = field(default_factory=AIMEActionConfig)
    webshop: WebShopConfig = field(default_factory=WebShopConfig)
    alfworld: ALFWorldConfig = field(default_factory=ALFWorldConfig)
    swe: SWEConfig = field(default_factory=SWEConfig)
    answer_submission: AnswerSubmissionConfig = field(default_factory=AnswerSubmissionConfig)
    director_reward: DirectorRewardConfig = field(default_factory=DirectorRewardConfig)
    director_prompt_variant: str = "v2.1"

    @property
    def skillbank_context_enabled(self) -> bool:
        return (
            self.skillbank_enabled
            and self.skillbank_usage != "off"
            and (self.skillbank_usage == "always" or self.skillbank_training)
        )

    def validate(self) -> None:
        self.proposer_model.validate("proposer")
        self.solver_model.validate("solver")
        cost_rates = (
            self.healthbench_judge_input_cost_per_million,
            self.healthbench_judge_output_cost_per_million,
        )
        if (cost_rates[0] is None) != (cost_rates[1] is None):
            raise ValueError(
                "HealthBench Judge input/output cost rates must be configured together"
            )
        if any(value is not None and value < 0 for value in cost_rates):
            raise ValueError("HealthBench Judge cost rates must be non-negative")
        if "private" not in {part.casefold() for part in self.healthbench_judge_audit_path.parts}:
            raise ValueError("HealthBench Judge audit path must be private")
        if self.healthbench_auto_grader_mode not in {"official", "low_cost"}:
            raise ValueError("HealthBench auto grader mode must be 'official' or 'low_cost'")
        if not self.healthbench_judge_expected_model.strip():
            raise ValueError("HealthBench Judge expected model cannot be empty")
        runtime_pool = self.runtime_pool()
        for runtime in runtime_pool.values():
            runtime.validate()
        self.retrieval.validate()
        self.aime_actions.validate()
        self.webshop.validate()
        self.alfworld.validate()
        self.swe.validate()
        self.answer_submission.validate()
        self.director_reward.validate()
        if self.director_prompt_variant not in {"v2", "v2.1"}:
            raise ValueError("director.prompt_variant must be 'v2' or 'v2.1'")
        if not self.worker_runtime_routes:
            raise ValueError("runtime_routing.worker_routes cannot be empty")
        if self.route_health_cooldown_s <= 0:
            raise ValueError("runtime_routing.health_cooldown_s must be positive")
        unknown_routes = set(self.worker_runtime_routes) - set(runtime_pool)
        if unknown_routes:
            raise ValueError("unknown worker runtime routes: " + ", ".join(sorted(unknown_routes)))
        for logical, members in self.runtime_endpoint_pools.items():
            if (
                logical not in runtime_pool
                or len(members) < 2
                or len(set(members)) != len(members)
                or set(members) - set(runtime_pool)
            ):
                raise ValueError("invalid endpoint pool: " + logical)
            configs = [runtime_pool[member] for member in (logical, *members)]
            if len({(item.served_model, item.reasoning_effort) for item in configs}) != 1:
                raise ValueError("endpoint pool must use the same model and effort")
        if (
            self.endpoint_pool_retry_attempts < 0
            or self.endpoint_pool_retry_backoff_s < 0
            or self.endpoint_pool_member_queue_wait_s < 0
        ):
            raise ValueError("endpoint pool retry settings must be non-negative")
        if (
            self.healthbench_judge_runtime_route is not None
            and self.healthbench_judge_runtime_route not in runtime_pool
        ):
            raise ValueError(
                f"unknown HealthBench judge runtime route: {self.healthbench_judge_runtime_route}"
            )
        if self.skill_distiller_runtime not in runtime_pool:
            raise ValueError(
                f"unknown skill distiller runtime route: {self.skill_distiller_runtime}"
            )
        if (
            self.answer_submission.enabled
            and self.answer_submission.qa_model_enabled
            and self.answer_submission.runtime_route not in runtime_pool
        ):
            raise ValueError(
                f"unknown answer submission runtime route: {self.answer_submission.runtime_route}"
            )
        if any(runtime.managed_locally for runtime in self.additional_runtimes.values()):
            raise ValueError("additional runtimes must be externally managed")
        if self.proposer_model.identity() == self.solver_model.identity():
            raise ValueError("Proposer and Solver must use independent served model instances")
        if self.proposer_model.checkpoint_path == self.solver_model.checkpoint_path:
            raise ValueError("Proposer and Solver checkpoint paths must be different")
        if not self.proposer_model.trainable or not self.solver_model.trainable:
            raise ValueError("Proposer and Solver model roles must be trainable")
        if self.canvas.max_agents <= 0 or self.canvas.max_rounds <= 0:
            raise ValueError("canvas agent/round budgets must be positive")
        if (
            self.canvas.max_total_tokens <= 0
            or self.canvas.relay_max_chars <= 0
            or self.canvas.feedback_max_chars <= 0
            or self.canvas.artifact_summary_max_chars <= 0
        ):
            raise ValueError("canvas token and context budgets must be positive")
        dataset_token_budgets = {
            canonical_dataset_name(dataset): int(limit)
            for dataset, limit in self.canvas.max_total_tokens_by_dataset.items()
        }
        if any(not dataset or limit <= 0 for dataset, limit in dataset_token_budgets.items()):
            raise ValueError(
                "canvas.max_total_tokens_by_dataset requires non-empty dataset keys "
                "and positive limits"
            )
        minimum_canvas_token_budget = min(
            (self.canvas.max_total_tokens, *dataset_token_budgets.values())
        )
        if not 0 <= self.canvas.graph_growth_token_reserve < minimum_canvas_token_budget:
            raise ValueError(
                "canvas graph_growth_token_reserve must be non-negative and below "
                "every configured max_total_tokens budget"
            )
        if not 0 < self.canvas.worker_latency_quantile <= 1:
            raise ValueError("canvas.worker_latency_quantile must be in (0, 1]")
        if (
            min(
                self.canvas.worker_latency_window,
                self.canvas.worker_latency_min_samples,
            )
            <= 0
        ):
            raise ValueError("canvas Worker latency sample limits must be positive")
        if self.canvas.worker_latency_min_samples > self.canvas.worker_latency_window:
            raise ValueError(
                "canvas.worker_latency_min_samples cannot exceed worker_latency_window"
            )
        if (
            min(
                self.canvas.worker_latency_cold_start_s,
                self.canvas.finalization_time_reserve_s,
            )
            <= 0
        ):
            raise ValueError("canvas Worker latency and finalization reserves must be positive")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.canvas.finalization_time_reserve_by_dataset.values()
        ):
            raise ValueError("dataset finalization reserves must be finite and positive")
        if not 0 < self.canvas.worker_token_quantile <= 1:
            raise ValueError("canvas.worker_token_quantile must be in (0, 1]")
        if (
            min(
                self.canvas.worker_token_window,
                self.canvas.worker_token_min_samples,
                self.canvas.worker_token_cold_start,
                self.canvas.finalization_token_reserve,
            )
            <= 0
        ):
            raise ValueError("canvas Worker token sample limits and reserves must be positive")
        if self.canvas.worker_token_min_samples > self.canvas.worker_token_window:
            raise ValueError("canvas.worker_token_min_samples cannot exceed worker_token_window")
        if self.canvas.finalization_token_reserve >= minimum_canvas_token_budget:
            raise ValueError(
                "canvas.finalization_token_reserve must be below every configured "
                "max_total_tokens budget"
            )
        if (
            min(
                self.canvas.repair_recent_action_limit,
                self.canvas.semantic_no_progress_limit,
                self.canvas.output_selection_budget,
            )
            <= 0
        ):
            raise ValueError(
                "canvas repair history, progress, and selection limits must be positive"
            )
        if self.canvas.structural_exploration_policy not in {"off", "stratified"}:
            raise ValueError("canvas.structural_exploration_policy must be 'off' or 'stratified'")
        if self.canvas.bidirectional_revision_policy not in {
            "always",
            "evidence_gated",
        }:
            raise ValueError(
                "canvas.bidirectional_revision_policy must be 'always' or 'evidence_gated'"
            )
        if not 0.0 <= self.canvas.bidirectional_revision_confidence_threshold <= 1.0:
            raise ValueError("canvas.bidirectional_revision_confidence_threshold must be in [0, 1]")
        if self.verifier not in {
            "none",
            "auto",
            "exact_match",
            "multi_answer_exact_match",
            "flowsteer_qa",
            "token_f1",
            "numeric",
            "multiple_choice",
            "healthbench_rubric",
            "healthbench_rubric_low_cost",
            "swe_outcome",
        }:
            raise ValueError("unsupported verifier.mode")
        if self.swe.enabled and self.verifier not in {"auto", "swe_outcome"}:
            raise ValueError("enabled SWE execution requires strict outcome verifier mode")
        if self.mace_enabled:
            raise ValueError("MACE is retired; remove [mace] and use Director SET_MODEL")
        if self.skillbank_mode not in {"legacy", "director_skill_v2"}:
            raise ValueError("unknown skillbank mode")
        if self.skillbank_usage not in {"always", "training_only", "off"}:
            raise ValueError("skillbank usage must be always, training_only or off")
        self.pats.validate()
        if self.pats.enabled and self.skillbank_mode != "director_skill_v2":
            raise ValueError("PATS requires solver_skillbank.mode='director_skill_v2'")
        if self.skillbank_prompt_token_budget <= 0:
            raise ValueError("skill prompt token budget must be positive")
        if self.skillbank_activation_policy not in {"paired", "checked"}:
            raise ValueError("skill activation policy must be paired or checked")
        if self.skillbank_distill_concurrency <= 0:
            raise ValueError("skill distillation concurrency must be positive")
        if (
            self.skillbank_retrieval_min_score is not None
            and not -1 <= self.skillbank_retrieval_min_score <= 1
        ):
            raise ValueError("skill retrieval cutoff must be in [-1,1]")
        if self.skillbank_max_skills <= 0 or self.skillbank_retrieve_top_k <= 0:
            raise ValueError("solver_skillbank size and retrieval limits must be positive")
        if not 0.0 <= self.skillbank_dedup_threshold <= 1.0:
            raise ValueError("solver_skillbank.dedup_threshold must be in [0, 1]")
        if not 0.0 <= self.skillbank_dedup_review_threshold <= self.skillbank_dedup_threshold:
            raise ValueError("solver_skillbank.dedup_review_threshold must be <= dedup_threshold")
        if (
            min(
                self.skillbank_min_retrieved_for_evict,
                self.skillbank_update_freq,
                self.skillbank_pending_queue_max,
                self.skillbank_min_pending,
                self.skillbank_generate_per_update,
            )
            <= 0
        ):
            raise ValueError("solver_skillbank lifecycle limits must be positive")
        service_memory = (
            self.proposer_service_gpu_memory_utilization,
            self.solver_service_gpu_memory_utilization,
        )
        allowed_physical_gpus = _allowed_physical_gpu_ids()
        if not self.allocated_gpu_ids or any(
            gpu not in allowed_physical_gpus for gpu in self.allocated_gpu_ids
        ):
            raise ValueError(
                "resources.allocated_gpu_ids must be a subset of "
                "SPGFS_ALLOWED_PHYSICAL_GPUS="
                + ",".join(str(gpu) for gpu in sorted(allowed_physical_gpus))
            )
        if any(not 0.0 < value <= 1.0 for value in service_memory):
            raise ValueError("service GPU memory utilization must be in (0, 1]")
        policy_gpus = (self.proposer_gpu_id, self.solver_gpu_id)
        if self.proposer_gpu_id == self.solver_gpu_id:
            if not self.allow_policy_gpu_colocation:
                raise ValueError("locally managed policy GPU assignments must be distinct")
            if sum(service_memory) > 0.70:
                raise ValueError(
                    "co-located policy service GPU memory utilization must sum to <= 0.70"
                )
        assigned = policy_gpus
        if self.runtime.managed_locally:
            if self.runtime_gpu_id in policy_gpus and not self.allow_policy_gpu_colocation:
                raise ValueError(
                    "locally managed runtime GPU overlap requires explicit GPU colocation"
                )
            assigned = (*assigned, self.runtime_gpu_id)
        if any(gpu not in self.allocated_gpu_ids for gpu in assigned):
            raise ValueError("all service GPUs must be included in resources.allocated_gpu_ids")

    def model_manifest(self) -> dict[str, Any]:
        runtime_pool = self.runtime_pool()
        return {
            "policies": {
                "proposer": self.proposer_model.to_dict(),
                "solver": self.solver_model.to_dict(),
            },
            "runtime_environment": self.runtime.to_dict(),
            "runtime_environments": {
                name: runtime.to_dict() for name, runtime in runtime_pool.items()
            },
            "runtime_routing": {
                "peer_selection_policy": "removed_direct_neighbors_v1",
                "model_selection_policy": "director_set_model_v1",
                "action_protocol": "director_model_v1",
                "counterfactual_execution": "full_graph_v1",
                "worker_routes": list(self.worker_runtime_routes),
                "endpoint_pools": {
                    key: list(value) for key, value in self.runtime_endpoint_pools.items()
                },
                "endpoint_pool_retry_attempts": self.endpoint_pool_retry_attempts,
                "endpoint_pool_retry_backoff_s": self.endpoint_pool_retry_backoff_s,
                "endpoint_pool_member_queue_wait_s": (
                    self.endpoint_pool_member_queue_wait_s
                ),
                "skill_distiller": self.skill_distiller_runtime,
                "health_state_path": str(self.route_health_path),
                "health_cooldown_s": self.route_health_cooldown_s,
            },
            "canvas_execution": {
                "worker_protocol_status_version": "worker_protocol_status_v1",
                "fallback_max_total_tokens": self.canvas.max_total_tokens,
                "max_total_tokens_by_dataset": {
                    canonical_dataset_name(dataset): int(limit)
                    for dataset, limit in sorted(self.canvas.max_total_tokens_by_dataset.items())
                },
                "structural_exploration_policy": (self.canvas.structural_exploration_policy),
                "bidirectional_revision_policy": (self.canvas.bidirectional_revision_policy),
                "bidirectional_revision_confidence_threshold": (
                    self.canvas.bidirectional_revision_confidence_threshold
                ),
                "bidirectional_revision_wave_budget": 1,
            },
            "retrieval": asdict(self.retrieval),
            "aime_actions": asdict(self.aime_actions),
            "webshop": asdict(self.webshop),
            "alfworld": {
                **asdict(self.alfworld),
                "data_root": str(self.alfworld.data_root),
            },
            "swe_bench": {
                "enabled": self.swe.enabled,
                "environment_state": "stateful",
                "session_scope": "per_agent",
                "commit_policy": "single_committer",
                "dataset_revision": self.swe.dataset_revision,
                "max_initial_calls": self.swe.max_initial_calls,
                "max_revision_calls": self.swe.max_revision_calls,
                "max_total_calls": self.swe.max_total_calls,
                "duplicate_responsibility_policy": (self.swe.duplicate_responsibility_policy),
                "test_profiles": sorted(self.swe.test_profiles),
            },
            "answer_submission": asdict(self.answer_submission),
            "director_reward": asdict(self.director_reward),
            "director_prompt": {"variant": self.director_prompt_variant},
            "healthbench_judge_audit": {
                "runtime_route": self.healthbench_judge_runtime_route,
                "path": str(self.healthbench_judge_audit_path),
                "auto_grader_mode": self.healthbench_auto_grader_mode,
                "expected_model": self.healthbench_judge_expected_model,
                "input_cost_per_million": (self.healthbench_judge_input_cost_per_million),
                "output_cost_per_million": (self.healthbench_judge_output_cost_per_million),
            },
        }

    def runtime_pool(self) -> dict[str, FixedRuntimeConfig]:
        name = self.runtime_name.strip()
        if not name:
            raise ValueError("runtime.name cannot be empty")
        if name in self.additional_runtimes:
            raise ValueError(f"duplicate runtime name: {name}")
        return {name: self.runtime, **self.additional_runtimes}


def load_adaptive_config(path: str | Path, *, validate: bool = True) -> AdaptiveApplicationConfig:
    source = Path(path).resolve()
    _load_project_env(source)
    with source.open("rb") as handle:
        payload = tomllib.load(handle)
    root = source.parent.parent if source.parent.name == "configs" else source.parent
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("adaptive config requires [models.proposer] and [models.solver]")
    missing = [name for name in ("proposer", "solver") if name not in models]
    if missing:
        raise ValueError("missing model sections: " + ", ".join(missing))
    canvas = payload.get("canvas", {})
    if not isinstance(canvas, dict):
        raise ValueError("canvas must be a TOML table")
    raw_dataset_token_budgets = canvas.get(
        "max_total_tokens_by_dataset",
        DEFAULT_DATASET_MAX_TOTAL_TOKENS,
    )
    if not isinstance(raw_dataset_token_budgets, dict):
        raise ValueError("canvas.max_total_tokens_by_dataset must be a TOML table")
    if bool(canvas.get("enforce_flowsteer_structure", False)):
        raise ValueError(
            "canvas.enforce_flowsteer_structure is deprecated: fixed role/topology gates "
            "are incompatible with the generic-Agent protocol"
        )
    obsolete_execution_keys = {
        "initial_build_rounds",
        "max_repair_edits",
        "execute_each_step",
    } & set(canvas)
    if obsolete_execution_keys:
        raise ValueError(
            "incremental dirty-subgraph execution is mandatory and is no longer "
            "configurable with: " + ", ".join(sorted(obsolete_execution_keys))
        )
    mace = payload.get("mace", {})
    skills = payload.get("solver_skillbank", {})
    trace = payload.get("trace", {})
    healthbench_audit = payload.get("healthbench_judge_audit", {})
    verifier = payload.get("verifier", {})
    resources = payload.get("resources", {})
    retrieval = payload.get("retrieval", {})
    aime_actions = payload.get("aime_actions", payload.get("python_tool", {}))
    webshop = payload.get("webshop", {})
    alfworld = payload.get("alfworld", {})
    swe = payload.get("swe", {})
    if not isinstance(swe, dict):
        raise ValueError("swe must be a TOML table")
    raw_swe_test_profiles = swe.get(
        "test_profiles", {"python_syntax": ["python", "-m", "py_compile"]}
    )
    if not isinstance(raw_swe_test_profiles, dict):
        raise ValueError("swe.test_profiles must be a TOML table")
    if any(not isinstance(command, list) for command in raw_swe_test_profiles.values()):
        raise ValueError("each swe.test_profiles entry must be an argv array")
    answer_submission = payload.get("answer_submission", {})
    if not isinstance(answer_submission, dict):
        raise ValueError("answer_submission must be a TOML table")
    director_reward = payload.get("director_reward")
    if director_reward is not None and not isinstance(director_reward, dict):
        raise ValueError("director_reward must be a TOML table")
    director = payload.get("director", {})
    if not isinstance(director, dict):
        raise ValueError("director must be a TOML table")
    runtime_payload = payload.get("runtime", {})
    if not isinstance(runtime_payload, dict):
        raise ValueError("runtime must be a TOML table")
    runtime_name = str(runtime_payload.get("name", "default")).strip()
    additional_runtime_payload = payload.get("runtimes", {})
    if not isinstance(additional_runtime_payload, dict):
        raise ValueError("runtimes must be a TOML table")
    runtime_routing = payload.get("runtime_routing", {})
    if not isinstance(runtime_routing, dict):
        raise ValueError("runtime_routing must be a TOML table")
    config = AdaptiveApplicationConfig(
        proposer_model=_role_model(models["proposer"], root, "proposer", trainable=True),
        solver_model=_role_model(models["solver"], root, "solver", trainable=True),
        runtime=_fixed_runtime(runtime_payload, root),
        runtime_name=runtime_name,
        additional_runtimes={
            str(name): _fixed_runtime(value, root)
            for name, value in additional_runtime_payload.items()
        },
        worker_runtime_routes=tuple(
            str(value) for value in runtime_routing.get("worker_routes", [runtime_name])
        ),
        runtime_endpoint_pools={
            str(key): tuple(str(member) for member in value)
            for key, value in runtime_routing.get("endpoint_pools", {}).items()
        },
        endpoint_pool_retry_attempts=int(runtime_routing.get("pool_retry_attempts", 2)),
        endpoint_pool_retry_backoff_s=float(runtime_routing.get("pool_retry_backoff_s", 1.0)),
        endpoint_pool_member_queue_wait_s=float(
            runtime_routing.get("pool_member_queue_wait_s", 0.5)
        ),
        skill_distiller_runtime=str(runtime_routing.get("skill_distiller", runtime_name)),
        route_health_path=_path(
            runtime_routing.get("health_state_path"), root, "state/route_health.json"
        ),
        route_health_cooldown_s=float(runtime_routing.get("health_cooldown_s", 600.0)),
        canvas=CanvasConfig(
            max_agents=int(canvas.get("max_agents", 8)),
            max_rounds=int(canvas.get("max_rounds", 20)),
            max_total_tokens=int(canvas.get("max_total_tokens", 32_768)),
            max_total_tokens_by_dataset={
                canonical_dataset_name(dataset): int(limit)
                for dataset, limit in raw_dataset_token_budgets.items()
            },
            relay_max_chars=int(canvas.get("relay_max_chars", 4000)),
            feedback_max_chars=int(canvas.get("feedback_max_chars", 6000)),
            artifact_summary_max_chars=int(canvas.get("artifact_summary_max_chars", 320)),
            structural_repair_enabled=bool(canvas.get("structural_repair_enabled", True)),
            graph_growth_token_reserve=int(canvas.get("graph_growth_token_reserve", 8192)),
            remaining_time_admission_enabled=bool(
                canvas.get("remaining_time_admission_enabled", True)
            ),
            worker_latency_quantile=float(canvas.get("worker_latency_quantile", 0.95)),
            worker_latency_window=int(canvas.get("worker_latency_window", 64)),
            worker_latency_min_samples=int(canvas.get("worker_latency_min_samples", 3)),
            worker_latency_cold_start_s=float(canvas.get("worker_latency_cold_start_s", 30.0)),
            finalization_time_reserve_s=float(canvas.get("finalization_time_reserve_s", 20.0)),
            finalization_time_reserve_by_dataset={
                str(key): float(value)
                for key, value in canvas.get(
                    "finalization_time_reserve_by_dataset",
                    {
                        "healthbench_professional": 180.0,
                        "swe_bench": 120.0,
                    },
                ).items()
            },
            remaining_token_admission_enabled=bool(
                canvas.get("remaining_token_admission_enabled", True)
            ),
            worker_token_quantile=float(canvas.get("worker_token_quantile", 0.95)),
            worker_token_window=int(canvas.get("worker_token_window", 64)),
            worker_token_min_samples=int(canvas.get("worker_token_min_samples", 3)),
            worker_token_cold_start=int(canvas.get("worker_token_cold_start", 4096)),
            finalization_token_reserve=int(canvas.get("finalization_token_reserve", 2048)),
            repair_recent_action_limit=int(canvas.get("repair_recent_action_limit", 5)),
            semantic_no_progress_limit=int(canvas.get("semantic_no_progress_limit", 2)),
            output_selection_budget=int(canvas.get("output_selection_budget", 1)),
            structural_exploration_policy=str(canvas.get("structural_exploration_policy", "off"))
            .strip()
            .casefold(),
            bidirectional_revision_policy=str(
                canvas.get("bidirectional_revision_policy", "always")
            ).strip(),
            bidirectional_revision_confidence_threshold=float(
                canvas.get("bidirectional_revision_confidence_threshold", 0.8)
            ),
        ),
        verifier=str(verifier.get("mode", "none")),
        mace_enabled=bool(mace.get("enabled", False)),
        mace_alpha=float(mace.get("alpha", 1.0)),
        mace_regularization=float(mace.get("regularization", 1.0)),
        seed=int(runtime_routing.get("seed", mace.get("seed", 0))),
        mace_statistics_path=_path(mace.get("statistics_path"), root, "state/mace_statistics.json"),
        mace_model_statistics_path=_path(
            mace.get("model_statistics_path"), root, "state/mace_model_statistics.json"
        ),
        skillbank_enabled=bool(skills.get("enabled", True)),
        skillbank_mode=str(skills.get("mode", "legacy")),
        skillbank_usage=str(skills.get("usage", "always")),
        pats=PatsConfig(**skills.get("pats", {})),
        skillbank_prompt_token_budget=int(skills.get("prompt_token_budget", 1024)),
        skillbank_retrieval_min_score=(
            float(skills["retrieval_min_score"]) if "retrieval_min_score" in skills else None
        ),
        skillbank_path=_path(skills.get("path"), root, "state/solver_skillbank.json"),
        skill_cases_path=_path(skills.get("cases_path"), root, "state/solver_skill_cases.json"),
        skillbank_max_skills=int(skills.get("max_skills", 800)),
        skillbank_dedup_threshold=float(skills.get("dedup_threshold", 0.93)),
        skillbank_dedup_review_threshold=float(skills.get("dedup_review_threshold", 0.90)),
        skillbank_retrieve_top_k=int(skills.get("retrieve_top_k", 3)),
        skillbank_min_retrieved_for_evict=int(skills.get("min_retrieved_for_evict", 3)),
        skillbank_update_freq=int(skills.get("update_freq", 10)),
        skillbank_pending_queue_max=int(skills.get("pending_queue_max", 300)),
        skillbank_min_pending=int(skills.get("min_pending", 20)),
        skillbank_generate_per_update=int(skills.get("generate_per_update", 50)),
        skillbank_distill_concurrency=int(skills.get("distill_concurrency", 2)),
        skillbank_activation_policy=str(skills.get("activation_policy", "paired")),
        skillbank_embedding_model_path=_model_reference(
            skills.get("embedding_model_path", "intfloat/e5-base-v2"), root
        ),
        trace_path=_path(trace.get("path"), root, "state/traces.jsonl"),
        healthbench_judge_audit_path=_path(
            healthbench_audit.get("path"),
            root,
            "state/private/healthbench_judge_audit",
        ),
        healthbench_judge_input_cost_per_million=(
            float(healthbench_audit["input_cost_per_million"])
            if healthbench_audit.get("input_cost_per_million") is not None
            else None
        ),
        healthbench_judge_output_cost_per_million=(
            float(healthbench_audit["output_cost_per_million"])
            if healthbench_audit.get("output_cost_per_million") is not None
            else None
        ),
        healthbench_auto_grader_mode=str(healthbench_audit.get("auto_grader_mode", "official"))
        .strip()
        .casefold(),
        healthbench_judge_runtime_route=(
            str(healthbench_audit["runtime_route"]).strip()
            if healthbench_audit.get("runtime_route") is not None
            else None
        ),
        healthbench_judge_expected_model=str(
            healthbench_audit.get("expected_model", HEALTHBENCH_PROFESSIONAL_JUDGE_MODEL)
        ).strip(),
        allocated_gpu_ids=tuple(int(value) for value in resources.get("allocated_gpu_ids", [0])),
        proposer_gpu_id=int(resources.get("proposer_gpu_id", 0)),
        solver_gpu_id=int(resources.get("solver_gpu_id", 0)),
        runtime_gpu_id=int(resources.get("runtime_gpu_id", 0)),
        proposer_service_gpu_memory_utilization=float(
            resources.get("proposer_service_gpu_memory_utilization", 0.30)
        ),
        solver_service_gpu_memory_utilization=float(
            resources.get("solver_service_gpu_memory_utilization", 0.35)
        ),
        allow_policy_gpu_colocation=bool(resources.get("allow_policy_gpu_colocation", True)),
        retrieval=RetrievalConfig(
            enabled=bool(retrieval.get("enabled", False)),
            service_url=str(retrieval.get("service_url", "http://127.0.0.1:8010/retrieve")),
            top_k=int(retrieval.get("top_k", 3)),
            timeout_s=float(retrieval.get("timeout_s", 120.0)),
            max_tool_rounds=int(retrieval.get("max_tool_rounds", 3)),
        ),
        aime_actions=AIMEActionConfig(
            enabled=bool(aime_actions.get("enabled", False)),
            timeout_s=float(aime_actions.get("timeout_s", 5.0)),
            max_output_chars=int(aime_actions.get("max_output_chars", 8000)),
            max_code_chars=int(aime_actions.get("max_code_chars", 12000)),
            memory_limit_mb=int(aime_actions.get("memory_limit_mb", 1024)),
            max_initial_calls=int(aime_actions.get("max_initial_calls", 3)),
            max_revision_calls=int(aime_actions.get("max_revision_calls", 1)),
            max_total_calls=int(aime_actions.get("max_total_calls", 4)),
        ),
        webshop=WebShopConfig(
            enabled=bool(webshop.get("enabled", False)),
            service_url=str(webshop.get("service_url", "http://127.0.0.1:8020")),
            timeout_s=float(webshop.get("timeout_s", 10.0)),
            max_observation_chars=int(webshop.get("max_observation_chars", 12_000)),
            max_query_chars=int(webshop.get("max_query_chars", 500)),
            max_initial_calls=int(webshop.get("max_initial_calls", 12)),
            max_revision_calls=int(webshop.get("max_revision_calls", 4)),
            max_total_calls=int(webshop.get("max_total_calls", 16)),
            staged_commit_enabled=bool(webshop.get("staged_commit_enabled", True)),
            max_pending_sessions=int(webshop.get("max_pending_sessions", 8)),
            pending_ttl_s=float(webshop.get("pending_ttl_s", 900.0)),
            search_observation_mode=str(webshop.get("search_observation_mode", "structured_only")),
        ),
        alfworld=ALFWorldConfig(
            enabled=bool(alfworld.get("enabled", False)),
            data_root=_path(
                alfworld.get("data_root"),
                root,
                "datasets/alfworld/data_assets/json_2.1.1",
            ),
            max_episode_steps=int(alfworld.get("max_episode_steps", 50)),
            max_rollout_steps=int(alfworld.get("max_rollout_steps", 400)),
            max_observation_chars=int(alfworld.get("max_observation_chars", 12_000)),
            max_initial_calls=int(alfworld.get("max_initial_calls", 50)),
            max_revision_calls=int(alfworld.get("max_revision_calls", 50)),
            max_total_calls=int(alfworld.get("max_total_calls", 100)),
            worker_guidance_policy=str(alfworld.get("worker_guidance_policy", "factual_memory_v1"))
            .strip()
            .casefold(),
        ),
        swe=SWEConfig(
            enabled=bool(swe.get("enabled", False)),
            repo_cache_root=_path(swe.get("repo_cache_root"), root, "state/swe/repo-cache"),
            workspace_root=_path(swe.get("workspace_root"), root, "state/swe/workspaces"),
            artifact_store_root=_path(swe.get("artifact_store_root"), root, "state/swe/artifacts"),
            verifier_registry_path=_path(
                swe.get("verifier_registry_path"),
                root,
                "state/private/swe/verifier-registry.json",
            ),
            lifecycle_log_path=_path(
                swe.get("lifecycle_log_path"),
                root,
                "state/private/swe/lifecycle.jsonl",
            ),
            verifier_log_path=_path(
                swe.get("verifier_log_path"),
                root,
                "state/private/swe/verifier-client.jsonl",
            ),
            verifier_host=str(swe.get("verifier_host", "")),
            verifier_user=str(swe.get("verifier_user", "sweeval")),
            verifier_identity_file=_path(
                swe.get("verifier_identity_file"),
                root,
                "config/private/swe_identity",
            ),
            verifier_known_hosts_file=_path(
                swe.get("verifier_known_hosts_file"),
                root,
                "state/deployments/20260824-swe-verifier-tencent/known_hosts",
            ),
            cvm_auto_start=bool(swe.get("cvm_auto_start", False))
            or os.environ.get("SPGFS_SWE_CVM_AUTO_START", "").strip() == "1",
            cvm_auto_stop=bool(swe.get("cvm_auto_stop", False))
            or os.environ.get("SPGFS_SWE_CVM_AUTO_STOP", "").strip() == "1",
            cvm_region=str(swe.get("cvm_region", "ap-singapore")),
            cvm_instance_id=str(swe.get("cvm_instance_id", "ins-5n1zolfw")),
            cvm_secret_id_env=str(
                swe.get("cvm_secret_id_env", "TENCENTCLOUD_SECRET_ID")
            ),
            cvm_secret_key_env=str(
                swe.get("cvm_secret_key_env", "TENCENTCLOUD_SECRET_KEY")
            ),
            cvm_endpoint=str(swe.get("cvm_endpoint", "cvm.tencentcloudapi.com")),
            cvm_timeout_s=float(swe.get("cvm_timeout_s", 600.0)),
            cvm_poll_s=float(swe.get("cvm_poll_s", 5.0)),
            dataset_revision=str(swe.get("dataset_revision", "")),
            connect_timeout_s=float(swe.get("connect_timeout_s", 10.0)),
            request_timeout_s=float(swe.get("request_timeout_s", 720.0)),
            local_test_timeout_s=float(swe.get("local_test_timeout_s", 60.0)),
            max_output_chars=int(swe.get("max_output_chars", 12_000)),
            max_file_chars=int(swe.get("max_file_chars", 200_000)),
            max_patch_bytes=int(swe.get("max_patch_bytes", 2_000_000)),
            max_initial_calls=int(swe.get("max_initial_calls", 20)),
            max_revision_calls=int(swe.get("max_revision_calls", 12)),
            max_total_calls=int(swe.get("max_total_calls", 32)),
            duplicate_responsibility_policy=str(
                swe.get("duplicate_responsibility_policy", "record_only")
            )
            .strip()
            .casefold(),
            test_profiles={
                str(name): tuple(str(part) for part in command)
                for name, command in raw_swe_test_profiles.items()
            },
        ),
        answer_submission=AnswerSubmissionConfig(
            enabled=bool(answer_submission.get("enabled", False)),
            qa_model_enabled=bool(answer_submission.get("qa_model_enabled", False)),
            runtime_route=str(answer_submission.get("runtime_route", "")),
            max_tokens=int(answer_submission.get("max_tokens", 128)),
            require_source_span=bool(answer_submission.get("require_source_span", True)),
        ),
        # Config files created before protocol_gate_v1 remain replayable. New
        # experiment configs must opt in explicitly so a resumed rollout group
        # can never change reward semantics silently.
        director_reward=DirectorRewardConfig(
            version=str((director_reward or {}).get("version", LEGACY_REWARD_VERSION))
        ),
        director_prompt_variant=str(director.get("prompt_variant", "v2.1")).strip().casefold(),
    )
    if validate:
        config.validate()
    return config


def _role_model(
    payload: object,
    root: Path,
    name: str,
    *,
    trainable: bool,
) -> RoleModelConfig:
    if not isinstance(payload, dict):
        raise ValueError(f"models.{name} must be a TOML table")
    default_port = {"proposer": 8001, "solver": 8002}[name]
    return RoleModelConfig(
        base_url=str(payload.get("base_url", f"http://127.0.0.1:{default_port}/v1")),
        api_key=str(payload.get("api_key", "EMPTY")),
        served_model=str(payload.get("served_model", "Qwen3.5-9B")),
        base_model_path=_path(payload.get("base_model_path"), root, "models/Qwen3.5-9B"),
        checkpoint_path=_path(payload.get("checkpoint_path"), root, f"state/checkpoints/{name}"),
        timeout_s=float(payload.get("timeout_s", 120.0)),
        trainable=bool(payload.get("trainable", trainable)),
    )


def _fixed_runtime(payload: object, root: Path) -> FixedRuntimeConfig:
    if not isinstance(payload, dict):
        raise ValueError("runtime must be a TOML table")
    managed_locally = bool(payload.get("managed_locally", True))
    raw_model_path = payload.get("model_path")
    model_path = (
        _path(raw_model_path, root, "models/Qwen3.5-9B")
        if raw_model_path is not None or managed_locally
        else None
    )
    default_max_concurrency = 4 if managed_locally else REMOTE_RUNTIME_MAX_CONCURRENCY
    return FixedRuntimeConfig(
        base_url=str(payload.get("base_url", "http://127.0.0.1:8003/v1")),
        api_key=_api_key(payload),
        api_keys_by_dataset=_api_keys_by_dataset(payload),
        served_model=str(payload.get("served_model", "Qwen3.5-9B")),
        model_path=model_path,
        timeout_s=float(payload.get("timeout_s", 120.0)),
        request_profile=str(payload.get("request_profile", "qwen")).strip().lower(),
        api_surface=str(payload.get("api_surface", "chat_completions")).strip().lower(),
        user_agent=payload.get("user_agent"),
        network_path=str(payload.get("network_path", "configured_proxy")),
        stream=bool(payload.get("stream", False)),
        enable_thinking=bool(payload.get("enable_thinking", False)),
        max_tokens=int(payload.get("max_tokens", 2048)),
        reasoning_effort=(
            str(payload["reasoning_effort"]).strip().lower()
            if payload.get("reasoning_effort") is not None
            else None
        ),
        healthbench_grader_reasoning_effort=(
            str(payload["healthbench_grader_reasoning_effort"]).strip().lower()
            if payload.get("healthbench_grader_reasoning_effort") is not None
            else None
        ),
        max_concurrency=int(payload.get("max_concurrency", default_max_concurrency)),
        max_concurrency_by_dataset={
            canonical_dataset_name(dataset): int(limit)
            for dataset, limit in payload.get("max_concurrency_by_dataset", {}).items()
        },
        managed_locally=managed_locally,
        frozen=bool(payload.get("frozen", True)),
    )


def _api_key(payload: dict[str, Any]) -> str:
    """Resolve API credentials without requiring secrets in TOML files."""

    key_file = str(payload.get("api_key_file", "")).strip()
    if key_file:
        value = Path(key_file).expanduser().read_text().strip()
        if not value:
            raise ValueError("API key file is empty")
        return value
    env_name = str(payload.get("api_key_env", "")).strip()
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ValueError(f"environment variable {env_name!r} is required for the API key")
        return value
    return str(payload.get("api_key", "EMPTY"))


def _api_keys_by_dataset(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("api_key_env_by_dataset", {})
    if not isinstance(raw, dict):
        raise ValueError("runtime.api_key_env_by_dataset must be a TOML table")
    resolved: dict[str, str] = {}
    for dataset, env_value in raw.items():
        name = str(env_value).strip()
        value = os.environ.get(name, "").strip()
        if not name or not value:
            raise ValueError(
                f"environment variable {name!r} is required for dataset {dataset!r} API key"
            )
        resolved[canonical_dataset_name(dataset)] = value
    return resolved


def _load_project_env(config_path: Path) -> None:
    """Load the nearest project .env without overriding the caller environment."""

    env_path = next(
        (parent / ".env" for parent in config_path.parents if (parent / ".env").is_file()),
        None,
    )
    if env_path is None:
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env entry at {env_path}:{line_number}")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid .env variable name at {env_path}:{line_number}")
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid .env value at {env_path}:{line_number}") from exc
        if len(parsed) > 1:
            raise ValueError(f"unquoted whitespace in .env value at {env_path}:{line_number}")
        value = parsed[0] if parsed else ""
        os.environ.setdefault(name, value)


def _path(value: object, root: Path, default: str) -> Path:
    path = Path(str(value if value is not None else default)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _model_reference(value: object, root: Path) -> str | Path | None:
    source = str(value or "").strip()
    if not source:
        return None
    path = Path(source).expanduser()
    if path.is_absolute() or source.startswith(".") or (root / path).exists():
        return path if path.is_absolute() else (root / path).resolve()
    return source


@dataclass(frozen=True)
class AdaptiveApplicationResult:
    run_id: str
    task: TaskSpec
    solver_result: AdaptiveSolverResult
    skills_used: tuple[str, ...]
    mace_statistics_path: str | None
    trace_store_path: str

    def to_dict(self) -> dict[str, Any]:
        result = self.solver_result.to_dict()
        submission = self.solver_result.answer_submission
        worker_token_in, worker_token_out = _unique_worker_token_totals(
            self.solver_result.trace.events,
            run_id=self.run_id,
        )
        return {
            "run_id": self.run_id,
            "task": task_to_public_dict(self.task),
            "answer": (
                submission.submitted_answer
                if submission
                else self.solver_result.director_run.output
            ),
            "raw_answer": self.solver_result.director_run.output,
            "answer_submission": submission.to_dict() if submission else None,
            "finished": self.solver_result.director_run.finished,
            "verification": result["verification"],
            "flowsteer_structure": result["flowsteer_structure"],
            "final_graph": self.solver_result.director_run.graph,
            "skills_used": list(self.skills_used),
            "trace_run_id": self.solver_result.trace.run_id,
            "trace_store_path": self.trace_store_path,
            "mace_statistics_path": self.mace_statistics_path,
            "token_in": worker_token_in + (submission.formatter_token_in if submission else 0),
            "token_out": worker_token_out + (submission.formatter_token_out if submission else 0),
            "model_roles": self.task.metadata.get("model_roles", {}),
        }


def _unique_worker_token_totals(trace_events: list[Any], *, run_id: str) -> tuple[int, int]:
    """Sum each executed Worker artifact once, even if a final Canvas event reuses it.

    Canvas emits the latest ExecutionReport with a final event so downstream
    audit can see the selected artifact. That event may contain no new Worker
    execution. Artifact identity, scoped by run, is the accounting boundary;
    report-level token fields are intentionally not used because they include
    the reused artifact again.
    """

    seen_artifacts: set[str] = set()
    token_in = token_out = 0
    for event_index, event in enumerate(trace_events):
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            continue
        execution = payload.get("execution")
        if not isinstance(execution, dict):
            continue
        artifacts = execution.get("artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            # Error-only reports normally have zero token usage. Preserve any
            # non-zero legacy report without pretending that it is deduplicable.
            token_in += int(execution.get("token_in", 0) or 0)
            token_out += int(execution.get("token_out", 0) or 0)
            continue
        for agent_id, artifact in artifacts.items():
            if not isinstance(artifact, dict):
                continue
            artifact_id = str(artifact.get("artifact_id", "")).strip()
            identity = (
                f"{run_id}:{artifact_id}"
                if artifact_id
                else f"{run_id}:event-{event_index}:{agent_id}"
            )
            if identity in seen_artifacts:
                continue
            seen_artifacts.add(identity)
            token_in += int(artifact.get("token_in", 0) or 0)
            token_out += int(artifact.get("token_out", 0) or 0)
    return token_in, token_out


class AdaptiveSolverApplication:
    """Runnable composition root for the complete inference-only Adaptive Solver."""

    def __init__(
        self,
        *,
        config: AdaptiveApplicationConfig,
        solver: AdaptiveWorkflowSolver,
        runtime: MultiAgentRuntime,
        skillbank: SolverSkillBank | None,
        skill_lifecycle: SolverSkillLifecycle | None,
        mace_selector: None = None,
        model_router: None = None,
        owned_backends: tuple[ChatBackend, ...] = (),
    ) -> None:
        self.config = config
        self.solver = solver
        self.runtime = runtime
        self.skillbank = skillbank
        self.skill_lifecycle = skill_lifecycle
        if mace_selector is not None:
            raise ValueError("peer selector has been removed")
        self.mace_selector = None
        if model_router is not None:
            raise ValueError("MACE is retired; use Director SET_MODEL")
        self.owned_backends = owned_backends

    def close(self) -> None:
        closed: set[int] = set()
        for backend in self.owned_backends:
            if id(backend) in closed:
                continue
            closed.add(id(backend))
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def set_rollout_deadline(self, deadline: RolloutDeadline | None) -> None:
        """Install one deadline across Director, Canvas, Worker, and API gates."""

        self.solver.rollout_deadline = deadline
        set_executor_deadline = getattr(self.runtime.executor, "set_deadline_context", None)
        if callable(set_executor_deadline):
            set_executor_deadline(deadline)
        for backend in self.owned_backends:
            setter = getattr(backend, "set_deadline_context", None)
            if callable(setter):
                setter(deadline)

    def configure_scoped_worker_routes(
        self, *, dataset: str, request_role: str = "primary"
    ) -> dict[str, Any]:
        """Apply an early route circuit for one new plain-text primary rollout."""

        dataset_key = canonical_dataset_name(dataset)
        store = RouteHealthStore(
            self.config.route_health_path,
            cooldown_s=self.config.route_health_cooldown_s,
        )
        available, blocked = store.available_scoped_routes(
            self.config.worker_runtime_routes,
            dataset=dataset_key,
            request_role=request_role,
        )
        diagnostics = {
            "dataset": dataset_key,
            "request_role": request_role,
            "available_routes": list(available),
            "blocked_routes": blocked,
        }
        if not available:
            blocked_summary = ", ".join(
                f"{route}({state['block_reason']})" for route, state in sorted(blocked.items())
            )
            raise PersistentRouteCircuitOpenError(
                "all Worker routes are blocked for scoped primary admission "
                f"{dataset_key}/{request_role}: {blocked_summary}"
            )
        self.solver.runtime_routes = available
        return diagnostics

    def solve(
        self,
        prompt: str,
        *,
        task_id: str = "task",
        task_type: str = "general",
        reference: Any = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        private_verifier_payload: dict[str, Any] | None = None,
    ) -> AdaptiveApplicationResult:
        if self.solver.rollout_deadline is not None:
            self.solver.rollout_deadline.check("application_solve_start")
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("task prompt cannot be empty")
        prospective_task = TaskSpec(
            task_id,
            prompt,
            reference=reference,
            task_type=task_type,
            metadata=dict(metadata or {}),
            private_verifier_payload=dict(private_verifier_payload or {}),
        )
        needs_reference = self.config.verifier in {
            "exact_match",
            "multi_answer_exact_match",
            "flowsteer_qa",
            "token_f1",
            "numeric",
            "multiple_choice",
        } or (self.config.verifier == "auto" and task_requires_reference(prospective_task))
        if needs_reference and reference is None:
            raise ValueError(f"{self.config.verifier} verifier requires a reference answer")
        self.runtime.reset()

        task = TaskSpec(
            task_id,
            prompt,
            reference=reference,
            task_type=task_type,
            metadata={
                **dict(metadata or {}),
                "model_roles": self.config.model_manifest(),
                "skills_enabled": self.config.skillbank_enabled,
                "action_protocol": "director_model_v1",
            },
            private_verifier_payload=dict(private_verifier_payload or {}),
        )
        resolved_run_id = run_id or f"adaptive-{uuid.uuid4().hex}"
        with request_dataset(task.metadata.get("dataset", task.task_type)):
            result = self.solver.solve(task, run_id=resolved_run_id)
        if self.solver.rollout_deadline is not None:
            try:
                self.solver.rollout_deadline.check("application_solve_complete")
            except WorkerWallClockLimitExceeded as exc:
                # The solver has already returned and persisted a verifier result.
                # Preserve that evidence and the policy record; downstream reward
                # and training admission still decide whether either is usable.
                verification = result.trace.verification
                if verification is None or not math.isfinite(verification.score):
                    raise
                task.metadata["postsolve_deadline_exceeded"] = exc.to_dict()
                result.trace.task.metadata["postsolve_deadline_exceeded"] = exc.to_dict()
        result.trace.task.metadata["executor_bundle_signature"] = hashlib.sha256(
            json.dumps(self.config.model_manifest(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return AdaptiveApplicationResult(
            run_id=resolved_run_id,
            task=task,
            solver_result=result,
            skills_used=result.skills_used,
            mace_statistics_path=None,
            trace_store_path=str(self.config.trace_path),
        )

    def supports_graph_counterfactual(self, task: TaskSpec) -> dict[str, Any]:
        """Final-score/lifecycle capability, independent of intermediate scoring."""
        if self.solver.verifier is None:
            return {"supported": False, "reason": "final_verifier_missing"}
        dataset = canonical_dataset_name(task.metadata.get("dataset", ""))
        registry = getattr(self.solver, "action_registry", None)
        adapter = registry.resolve(task) if registry is not None else None
        tools = getattr(self.runtime.executor, "tools", {})
        required = {
            "webshop": webshop_lifecycles,
            "alfworld": alfworld_lifecycles,
            "swe_bench": swe_lifecycles,
        }
        if dataset in required:
            lifecycles = required[dataset](tools)
            if adapter is None or not lifecycles:
                return {"supported": False, "reason": f"{dataset}_isolated_lifecycle_missing"}
            if dataset == "swe_bench" and any(item.harness_backend is None for item in lifecycles):
                return {"supported": False, "reason": "swe_final_harness_missing"}
        return {"supported": True, "reason": "final_verifier_and_isolated_lifecycle_configured"}

    def evaluate_graph(
        self,
        task: TaskSpec,
        graph: Any,
        *,
        seed: int,
        initial_artifacts: dict[str, dict[str, Any]] | None = None,
        dirty_agents: set[str] | None = None,
        return_verification: bool = False,
    ) -> float | dict[str, Any]:
        """Complete intervention evaluation on an exclusively owned branch application."""
        if initial_artifacts is not None or dirty_agents is not None:
            raise ValueError("local/prefix evaluation retired; full graph execution required")
        if self.solver.verifier is None:
            raise ValueError("full graph evaluation requires a final task verifier")
        task = copy.deepcopy(task)
        for key in list(task.metadata):
            if (
                key.endswith(("_environment_result", "_workspace_result"))
                or key
                in {
                    "answer_submission",
                    "qa_token_f1",
                    "qa_official_metrics",
                    "worker_backend_failure",
                    "worker_artifact_integrity_failure",
                    "swe_infrastructure_failure",
                    "swe_execution_incomplete",
                    "training_eligible",
                    "mace_rewards",
                    "mace_decisions",
                    "mace_window_audit",
                }
                or key.startswith("_runtime_")
            ):
                task.metadata.pop(key, None)
        task.metadata["judge_evaluation_scope"] = (
            "graph:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "seed": seed,
                        "graph": graph.to_dict(),
                        "phase": task.metadata.get("graph_evaluation_phase", "counterfactual"),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        graph = graph.clone()
        if not graph.output_agent or any(not node.configured for node in graph.nodes.values()):
            raise ValueError(
                "full graph evaluation requires output and complete node configuration"
            )
        for node in graph.nodes.values():
            for key in list(node.metadata):
                if key.startswith("_runtime_"):
                    node.metadata.pop(key)
        previous_full_replay = self.runtime.full_graph_replay
        self.runtime.full_graph_replay = True
        active_webshop_lifecycles: tuple[WebShopSessionLifecycle, ...] = ()
        active_alfworld_lifecycles: tuple[ALFWorldSessionLifecycle, ...] = ()
        active_swe_lifecycles: tuple[SWEWorkspaceLifecycle, ...] = ()
        try:
            self.runtime.reset()
            self.runtime.seed = int(seed)
            action_registry = getattr(self.solver, "action_registry", None)
            action_adapter = action_registry.resolve(task) if action_registry is not None else None
            lifecycle_tools = getattr(self.runtime.executor, "tools", {})
            if action_adapter is not None and action_adapter.adapter_id == "webshop":
                active_webshop_lifecycles = webshop_lifecycles(lifecycle_tools)
                if not active_webshop_lifecycles:
                    raise ValueError("WebShop task requires configured WebShop Actions")
                for lifecycle in active_webshop_lifecycles:
                    lifecycle.bind_task(task)
                    lifecycle.reserve_full_graph_owner(str(graph.output_agent))
                self.runtime.environment_fingerprint = active_webshop_lifecycles[
                    0
                ].environment_fingerprint
            if action_adapter is not None and action_adapter.adapter_id == "alfworld":
                active_alfworld_lifecycles = alfworld_lifecycles(lifecycle_tools)
                if not active_alfworld_lifecycles:
                    raise ValueError("ALFWorld task requires configured ALFWorld Actions")
                for lifecycle in active_alfworld_lifecycles:
                    lifecycle.bind_task(task)
                self.runtime.environment_fingerprint = active_alfworld_lifecycles[
                    0
                ].environment_fingerprint
            if action_adapter is not None and action_adapter.adapter_id == "swe_bench":
                active_swe_lifecycles = swe_lifecycles(lifecycle_tools)
                if not active_swe_lifecycles:
                    raise ValueError("SWE-bench task requires configured SWE Actions")
                for lifecycle in active_swe_lifecycles:
                    lifecycle.bind_task(task)
                self.runtime.environment_fingerprint = active_swe_lifecycles[
                    0
                ].environment_fingerprint
            calls = max(
                1,
                int(
                    self.runtime.estimate_execution_tokens(
                        graph,
                        set(graph.nodes),
                        quantile=self.config.canvas.worker_token_quantile,
                        minimum_samples=self.config.canvas.worker_token_min_samples,
                        cold_start_tokens=self.config.canvas.worker_token_cold_start,
                    )["call_count"]
                ),
            )
            dataset = canonical_dataset_name(task.metadata.get("dataset", ""))
            token_budget = self.config.canvas.max_total_tokens_by_dataset.get(
                dataset, self.config.canvas.max_total_tokens
            )
            webshop_partition = (
                budget_partition(
                    total_limit=int(token_budget),
                    spent=0,
                    configured_minimum=self.config.canvas.finalization_token_reserve,
                    call_count=calls,
                    closure=False,
                )
                if dataset == "webshop"
                else None
            )
            for node in graph.nodes.values():
                node.metadata["_runtime_token_credit"] = int(token_budget) // calls
                if webshop_partition is not None:
                    node.metadata.update(
                        _runtime_token_credit=webshop_partition["per_execution_credit"],
                        _runtime_reserved_closure_tokens=webshop_partition[
                            "reserved_closure_tokens"
                        ],
                        _runtime_budget_phase="exploration",
                    )
                if dataset in {"nq_open", "hotpotqa", "webshop"}:
                    if dataset != "webshop":
                        node.metadata["_runtime_budget_kind"] = "short_qa_request_credit_v1"
                        node.metadata["_runtime_finalization_output_reserve"] = (
                            self.config.canvas.finalization_token_reserve
                        )
                else:
                    # Environment branches must obey admission before a request,
                    # not just reject an already over-budget graph afterwards.
                    node.metadata["_runtime_budget_kind"] = "full_graph_request_credit_v1"
            with request_dataset(task.metadata.get("dataset", task.task_type)):
                report = self.runtime.execute(
                    task=solver_task_text(
                        task,
                        include_submission_contract=(
                            getattr(self.solver, "answer_finalizer", None) is not None
                        ),
                    ),
                    graph=graph,
                    dirty_agents=None,
                )
            if report.cache_hits:
                raise RuntimeError("full graph branch reused execution cache")
            if set(report.artifacts) != set(graph.nodes):
                raise RuntimeError("full graph branch did not execute every node")
            if active_webshop_lifecycles:
                self.runtime.complete_full_graph_webshop_output(
                    task=solver_task_text(task),
                    graph=graph,
                    report=report,
                    remaining_token_credit=max(
                        0, int(token_budget) - report.token_in - report.token_out
                    ),
                )
            if graph.output_agent in self.runtime.environment_commit_ready_agents():
                self.runtime.commit_environment_output(graph.output_agent)
            self.last_graph_evaluation = {
                "execution_mode": "full_graph_v1",
                "seed": int(seed),
                "nodes": sorted(graph.nodes),
                "execution": report.to_dict(),
                "token_budget": int(token_budget),
            }
            if report.incomplete_bidirectional_components:
                raise GraphEvaluationIncompleteError(
                    "full graph branch has incomplete bidirectional execution; no counterfactual credit"
                )
            if any(
                artifact.model == "runtime-budget-boundary"
                or (
                    artifact.token_in + artifact.token_out == 0
                    and any(
                        item.get("no_request_dispatched")
                        and "credit_exhausted" in str(item.get("stage", ""))
                        for item in artifact.protocol_diagnostics
                    )
                )
                for artifact in report.artifacts.values()
            ):
                raise RuntimeError(
                    "full graph branch could not admit Worker execution; no counterfactual credit"
                )
            output = ""
            if graph.output_agent and graph.output_agent in self.runtime.artifacts:
                output = self.runtime.artifacts[graph.output_agent].answer
                summary = self.runtime.artifacts[graph.output_agent].summary
            else:
                summary = ""
            output_artifact = (
                self.runtime.artifacts.get(graph.output_agent) if graph.output_agent else None
            )
            if action_adapter is not None and action_adapter.adapter_id == "webshop":
                task.metadata["webshop_environment_result"] = (
                    dict(output_artifact.environment_result)
                    if output_artifact and output_artifact.environment_result
                    else active_webshop_lifecycles[0].result_for(graph.output_agent)
                )
            if action_adapter is not None and action_adapter.adapter_id == "alfworld":
                task.metadata["alfworld_environment_result"] = (
                    dict(output_artifact.environment_result)
                    if output_artifact and output_artifact.environment_result
                    else active_alfworld_lifecycles[0].result_for(graph.output_agent)
                )
            if action_adapter is not None and action_adapter.adapter_id == "swe_bench":
                output_agent = graph.output_agent
                environment_result = (
                    dict(output_artifact.environment_result)
                    if output_artifact and output_artifact.environment_result
                    else active_swe_lifecycles[0].result_for(output_agent)
                )
                if output_agent and active_swe_lifecycles[0].harness_backend is not None:
                    private_evaluation = active_swe_lifecycles[0].evaluate_artifact(output_agent)
                    evaluation = public_swe_evaluation(private_evaluation)
                else:
                    raise RuntimeError("SWE counterfactual has no output artifact verifier")
                task.metadata["swe_workspace_result"] = environment_result
                task.metadata["swe_environment_result"] = evaluation
                if not bool(evaluation.get("environment_completed", False)):
                    raise RuntimeError("SWE counterfactual official harness did not complete")
            if report.token_in + report.token_out > int(token_budget):
                raise RuntimeError("full graph branch exceeded its complete execution token budget")
            for key in ("webshop_environment_result", "alfworld_environment_result"):
                outcome = task.metadata.get(key)
                if isinstance(outcome, dict) and (
                    outcome.get("termination_reason")
                    in {"agent_never_executed", "missing_output_agent", "environment_error"}
                    or outcome.get("error")
                    or outcome.get("infrastructure_failure")
                    or ("environment_completed" in outcome and not outcome["environment_completed"])
                ):
                    raise RuntimeError(
                        f"{key}: incomplete/infrastructure outcome, no counterfactual credit"
                    )
            backend_failures = [
                artifact
                for artifact in report.artifacts.values()
                if artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
            ]
            if backend_failures:
                failure_details = [
                    {**record, "agent_id": artifact.agent_id}
                    for artifact in backend_failures
                    for record in artifact_backend_failure_records(artifact)
                ]
                request_events = [
                    dict(event)
                    for artifact in backend_failures
                    for event in artifact.backend_request_events
                    if isinstance(event, dict)
                ]
                raise GraphEvaluationBackendError(
                    {
                        "count": len(backend_failures),
                        "agents": sorted(artifact.agent_id for artifact in backend_failures),
                        "routes": sorted(
                            {artifact.model_route or "unassigned" for artifact in backend_failures}
                        ),
                        "failure_types": sorted(
                            {str(record.get("kind", "unknown")) for record in failure_details}
                        ),
                        "failure_details": failure_details,
                        "request_events": request_events,
                        "retryable": any(
                            bool(record.get("retryable")) for record in failure_details
                        ),
                        "counts_toward_route_circuit": any(
                            bool(record.get("counts_toward_route_circuit"))
                            for record in failure_details
                        ),
                        "disable_route": any(
                            bool(record.get("disable_route")) for record in failure_details
                        ),
                    }
                )
            finalize_answer = getattr(self.solver, "finalize_answer", None)
            submission = (
                finalize_answer(task, output, raw_summary=summary)
                if callable(finalize_answer)
                else None
            )
            prediction = submission.submitted_answer if submission else output
            verification = (
                VerificationResult(
                    0.0,
                    False,
                    self.solver.verifier.name,
                    f"invalid_answer_submission:{submission.detail}",
                )
                if submission is not None
                and not submission.valid
                and submission.method == "aime_strict_submission_v1"
                else self.solver.verifier.verify(task, prediction)
            )
            task_score = float(verification.score)
            if (
                canonical_dataset_name(task.metadata.get("dataset", ""))
                == "healthbench_professional"
            ):
                try:
                    detail = json.loads(str(verification.detail or "{}"))
                    task_score = float(detail["training_reward_breakdown"]["training_reward"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "HealthBench graph evaluation lacks normalized training reward"
                    ) from exc
            if return_verification:
                return {
                    "score": task_score,
                    "prediction": prediction,
                    "verification": verification,
                }
            return task_score
        finally:
            for lifecycle in (
                *active_webshop_lifecycles,
                *active_alfworld_lifecycles,
                *active_swe_lifecycles,
            ):
                lifecycle.close_all()
            self.runtime.discard_peer_rewards()
            self.runtime.full_graph_replay = previous_full_replay


def create_adaptive_application(
    config: AdaptiveApplicationConfig,
    *,
    mock: bool = False,
    director_backend: ChatBackend | None = None,
    worker_backend: ChatBackend | None = None,
    distiller_backend: ChatBackend | None = None,
    verifier: Verifier | None = None,
    route_latency_tracker: RouteLatencyTracker | None = None,
    route_token_tracker: RouteTokenTracker | None = None,
    swe_lifecycle: SWEWorkspaceLifecycle | None = None,
    director_tokenizer: Tokenizer | None = None,
    director_enable_thinking: bool | None = None,
) -> AdaptiveSolverApplication:
    config.validate()
    route_health = RouteHealthStore(
        config.route_health_path, cooldown_s=config.route_health_cooldown_s
    )
    active_routes, blocked_routes = route_health.available_routes(config.worker_runtime_routes)
    if not active_routes:
        blocked_summary = ", ".join(
            f"{route}({state['block_reason']})" for route, state in sorted(blocked_routes.items())
        )
        raise PersistentRouteCircuitOpenError(
            "all configured Worker routes are blocked by persisted health state: " + blocked_summary
        )
    if active_routes != config.worker_runtime_routes:
        config = replace(config, worker_runtime_routes=active_routes)
    if config.healthbench_judge_runtime_route is not None:
        judge_routes, blocked_judge_routes = route_health.available_routes(
            (config.healthbench_judge_runtime_route,)
        )
        if not judge_routes:
            state = blocked_judge_routes[config.healthbench_judge_runtime_route]
            raise PersistentRouteCircuitOpenError(
                "HealthBench Judge route is blocked by persisted health state: "
                f"{config.healthbench_judge_runtime_route}({state['block_reason']})"
            )
    owned_backends: list[ChatBackend] = []
    tools: dict[str, Any] = {}
    if config.retrieval.enabled:
        tools["search"] = SearchServiceTool(
            config.retrieval.service_url,
            top_k=config.retrieval.top_k,
            timeout_s=config.retrieval.timeout_s,
        )
    if config.aime_actions.enabled:
        tools["symbolic_compute"] = SymbolicComputeTool(
            timeout_s=min(config.aime_actions.timeout_s, 3.0),
            max_output_chars=config.aime_actions.max_output_chars,
            memory_limit_mb=config.aime_actions.memory_limit_mb,
        )
        tools["finite_search"] = FiniteSearchTool(
            timeout_s=min(config.aime_actions.timeout_s, 3.0),
            max_output_chars=config.aime_actions.max_output_chars,
            memory_limit_mb=config.aime_actions.memory_limit_mb,
        )
        tools["python_exec"] = PythonExecutionTool(
            timeout_s=config.aime_actions.timeout_s,
            max_output_chars=config.aime_actions.max_output_chars,
            max_code_chars=config.aime_actions.max_code_chars,
            memory_limit_mb=config.aime_actions.memory_limit_mb,
        )
    webshop_lifecycle = None
    if config.webshop.enabled:
        webshop_lifecycle = WebShopSessionLifecycle(
            WebShopHTTPClient(
                config.webshop.service_url,
                timeout_s=config.webshop.timeout_s,
            ),
            max_observation_chars=config.webshop.max_observation_chars,
            max_pending_sessions=config.webshop.max_pending_sessions,
            pending_ttl_s=config.webshop.pending_ttl_s,
            stage_purchases=config.webshop.staged_commit_enabled,
            search_observation_mode=config.webshop.search_observation_mode,
        )
        tools["webshop_search"] = WebShopSearchTool(
            webshop_lifecycle,
            max_query_chars=config.webshop.max_query_chars,
        )
        tools["webshop_click"] = WebShopClickTool(webshop_lifecycle)
    alfworld_lifecycle = None
    if config.alfworld.enabled:
        alfworld_lifecycle = ALFWorldSessionLifecycle(
            LocalALFWorldClient(),
            data_root=config.alfworld.data_root,
            max_episode_steps=config.alfworld.max_episode_steps,
            max_rollout_steps=config.alfworld.max_rollout_steps,
            max_observation_chars=config.alfworld.max_observation_chars,
        )
        tools["alfworld_step"] = ALFWorldStepTool(alfworld_lifecycle)
    if swe_lifecycle is None and config.swe.enabled:
        verifier_registry = TrustedSWEVerifierRegistry(config.swe.verifier_registry_path)
        if verifier_registry.dataset_revision != config.swe.dataset_revision:
            raise ValueError("SWE config and trusted registry dataset revisions differ")
        server_lease = None
        if config.swe.cvm_auto_start or config.swe.cvm_auto_stop:
            secret_id = os.environ.get(config.swe.cvm_secret_id_env, "").strip()
            secret_key = os.environ.get(config.swe.cvm_secret_key_env, "").strip()
            if not secret_id or not secret_key:
                raise ValueError(
                    "SWE CVM auto lifecycle requires credentials in "
                    f"{config.swe.cvm_secret_id_env} and {config.swe.cvm_secret_key_env}"
                )
            server_lease = shared_tencent_cvm_lease(
                secret_id=secret_id,
                secret_key=secret_key,
                region=config.swe.cvm_region,
                instance_id=config.swe.cvm_instance_id,
                endpoint=config.swe.cvm_endpoint,
                timeout_s=config.swe.cvm_timeout_s,
                poll_s=config.swe.cvm_poll_s,
                stop_when_idle=config.swe.cvm_auto_stop,
            )
        swe_lifecycle = SWEWorkspaceLifecycle(
            repo_cache_root=config.swe.repo_cache_root,
            workspace_root=config.swe.workspace_root,
            artifact_store=CodeArtifactStore(config.swe.artifact_store_root),
            log_path=config.swe.lifecycle_log_path,
            test_profiles=config.swe.test_profiles,
            test_timeout_s=config.swe.local_test_timeout_s,
            max_output_chars=config.swe.max_output_chars,
            max_file_chars=config.swe.max_file_chars,
            harness_backend=SSHSWEHarnessBackend(
                host=config.swe.verifier_host,
                user=config.swe.verifier_user,
                identity_file=config.swe.verifier_identity_file,
                known_hosts_file=config.swe.verifier_known_hosts_file,
                dataset_revision=config.swe.dataset_revision,
                connect_timeout_s=config.swe.connect_timeout_s,
                request_timeout_s=config.swe.request_timeout_s,
                max_patch_bytes=config.swe.max_patch_bytes,
                log_path=config.swe.verifier_log_path,
                server_lease=server_lease,
            ),
            verifier_registry=verifier_registry,
        )
    if swe_lifecycle is not None:
        tools.update(swe_tools(swe_lifecycle))
    action_registry = default_dataset_action_registry(
        tools,
        aime_budgets=(
            config.aime_actions.max_initial_calls,
            config.aime_actions.max_revision_calls,
            config.aime_actions.max_total_calls,
        ),
        retrieval_initial_budget=config.retrieval.max_tool_rounds,
        webshop_budgets=(
            config.webshop.max_initial_calls,
            config.webshop.max_revision_calls,
            config.webshop.max_total_calls,
        ),
        webshop_staged_commit=config.webshop.staged_commit_enabled,
        alfworld_budgets=(
            config.alfworld.max_initial_calls,
            config.alfworld.max_revision_calls,
            config.alfworld.max_total_calls,
        ),
        swe_budgets=(
            config.swe.max_initial_calls,
            config.swe.max_revision_calls,
            config.swe.max_total_calls,
        ),
    )
    worker_executor: ModelAgentExecutor | RoutedModelAgentExecutor
    runtime_backends: dict[str, ChatBackend] = {}
    if mock:
        defaults = _mock_adaptive_backends(
            binary_relations=director_tokenizer is not None,
            runtime_route=config.worker_runtime_routes[0],
        )
        director_backend = director_backend or defaults[0]
        worker_backend = worker_backend or defaults[1]
        distiller_backend = distiller_backend or defaults[2]
        worker_executor = ModelAgentExecutor(
            worker_backend,
            tools=tools,
            action_registry=action_registry,
            max_tool_rounds=config.retrieval.max_tool_rounds,
            alfworld_worker_guidance_policy=(config.alfworld.worker_guidance_policy),
        )
    else:
        if director_backend is None:
            director_backend = OpenAICompatibleBackend(
                _gateway_config(
                    config.solver_model,
                    {"graph-director": 0.6},
                    sampling_seed=config.seed,
                    director_enable_thinking=director_enable_thinking,
                )
            )
            owned_backends.append(director_backend)
        if worker_backend is None or distiller_backend is None:
            runtime_backends = {
                name: _create_runtime_backend(runtime, route_name=name)
                for name, runtime in config.runtime_pool().items()
            }
            owned_backends.extend(runtime_backends.values())
            from .endpoint_pool import EndpointPoolBackend

            physical_backends = dict(runtime_backends)
            for logical, members in config.runtime_endpoint_pools.items():
                runtime_backends[logical] = EndpointPoolBackend(
                    logical,
                    {name: physical_backends[name] for name in members},
                    config.route_health_path.parent / "endpoint_pools",
                    pool_retry_attempts=config.endpoint_pool_retry_attempts,
                    retry_backoff_s=config.endpoint_pool_retry_backoff_s,
                    member_queue_wait_s=config.endpoint_pool_member_queue_wait_s,
                )
                # Install the shared rollout clock on the pool wrapper too,
                # not only on its physical clients.
                owned_backends.append(runtime_backends[logical])
        if worker_backend is None:
            worker_executor = RoutedModelAgentExecutor(
                runtime_backends,
                config.worker_runtime_routes,
                tools=tools,
                action_registry=action_registry,
                max_tool_rounds=config.retrieval.max_tool_rounds,
                alfworld_worker_guidance_policy=(config.alfworld.worker_guidance_policy),
            )
        else:
            worker_executor = ModelAgentExecutor(
                worker_backend,
                tools=tools,
                action_registry=action_registry,
                max_tool_rounds=config.retrieval.max_tool_rounds,
                alfworld_worker_guidance_policy=(config.alfworld.worker_guidance_policy),
            )
        if distiller_backend is None:
            distiller_backend = runtime_backends[config.skill_distiller_runtime]
    assert director_backend and distiller_backend
    fixed_backends = [distiller_backend]
    if worker_backend is not None:
        fixed_backends.append(worker_backend)
    if any(director_backend is fixed_backend for fixed_backend in fixed_backends):
        raise ValueError("Solver policy backend cannot be shared with fixed runtime")

    mace_selector = None  # Legacy constructor rejects any active selector.
    model_router = None
    runtime = MultiAgentRuntime(
        worker_executor,
        relay_max_chars=config.canvas.relay_max_chars,
        seed=config.seed,
        peer_selector=mace_selector,
        exploration_horizon=config.canvas.max_rounds,
        bidirectional_revision_policy=(config.canvas.bidirectional_revision_policy),
        bidirectional_revision_confidence_threshold=(
            config.canvas.bidirectional_revision_confidence_threshold
        ),
        route_latency_tracker=(
            route_latency_tracker
            or RouteLatencyTracker(window_size=config.canvas.worker_latency_window)
        ),
        route_token_tracker=(
            route_token_tracker or RouteTokenTracker(window_size=config.canvas.worker_token_window)
        ),
    )
    skillbank = (
        SolverSkillBank(
            config.skillbank_path,
            max_skills=config.skillbank_max_skills,
            dedup_threshold=config.skillbank_dedup_threshold,
            retrieve_top_k=config.skillbank_retrieve_top_k,
            min_retrieved_for_evict=config.skillbank_min_retrieved_for_evict,
            embedder=(
                E5SkillEmbedder(config.skillbank_embedding_model_path)
                if config.skillbank_embedding_model_path and not mock
                else None
            ),
        )
        if config.skillbank_context_enabled and config.skillbank_mode == "legacy"
        else None
    )
    if config.skillbank_context_enabled and config.skillbank_mode == "director_skill_v2":
        from .skill_evolution_v2 import load_bank

        skillbank = load_bank(
            config,
            embedder=(
                E5SkillEmbedder(config.skillbank_embedding_model_path)
                if config.skillbank_embedding_model_path and not mock
                else None
            ),
        )
    lifecycle = (
        SolverSkillLifecycle(
            skillbank,
            SESASolverSkillDistiller(distiller_backend),
            update_freq=config.skillbank_update_freq,
            pending_queue_max=config.skillbank_pending_queue_max,
            min_pending=config.skillbank_min_pending,
            generate_per_update=config.skillbank_generate_per_update,
        )
        if skillbank and config.skillbank_mode == "legacy"
        else None
    )
    if lifecycle:
        lifecycle.load_pending(config.skill_cases_path)
    grader_backend = (
        (
            runtime_backends[config.healthbench_judge_runtime_route]
            if config.healthbench_judge_runtime_route is not None
            else runtime_backends.get("gpt", distiller_backend)
        )
        if not mock
        else distiller_backend
    )
    selected_verifier = verifier or _verifier(
        config.verifier,
        adapters={
            "healthbench_rubric": HealthBenchOfficialRubricVerifier(
                grader_backend,
                parallel_rubrics=not mock,
                require_judge_attestation=not mock,
                expected_judge_model=config.healthbench_judge_expected_model,
                audit_store=HealthBenchJudgeAuditStore(config.healthbench_judge_audit_path),
                input_cost_per_million=(config.healthbench_judge_input_cost_per_million),
                output_cost_per_million=(config.healthbench_judge_output_cost_per_million),
            ),
            "healthbench_rubric_low_cost": HealthBenchRubricVerifier(
                grader_backend,
                audit_store=HealthBenchJudgeAuditStore(config.healthbench_judge_audit_path),
                input_cost_per_million=(config.healthbench_judge_input_cost_per_million),
                output_cost_per_million=(config.healthbench_judge_output_cost_per_million),
            ),
            "webshop_environment": WebShopEnvironmentVerifier(),
            "alfworld_environment": ALFWorldEnvironmentVerifier(),
            "swe_outcome": SWEOutcomeVerifier(),
        },
        aliases=(
            {"healthbench_rubric": "healthbench_rubric_low_cost"}
            if config.healthbench_auto_grader_mode == "low_cost"
            else None
        ),
    )
    answer_finalizer = None
    if config.answer_submission.enabled:
        formatter_backend = None
        if config.answer_submission.qa_model_enabled:
            formatter_backend = (
                distiller_backend
                if mock
                else runtime_backends.get(config.answer_submission.runtime_route)
            )
            if (
                formatter_backend is None
                and config.answer_submission.runtime_route == config.skill_distiller_runtime
            ):
                formatter_backend = distiller_backend
        answer_finalizer = AnswerFinalizer(
            config.answer_submission,
            qa_backend=formatter_backend,
        )
    solver = AdaptiveWorkflowSolver(
        director_backend=director_backend,
        runtime=runtime,
        verifier=selected_verifier,
        skillbank=skillbank,
        trace_store=JSONLTraceStore(config.trace_path),
        canvas_config=config.canvas,
        runtime_routes=config.worker_runtime_routes,
        model_router=model_router,
        action_registry=action_registry,
        answer_finalizer=answer_finalizer,
        swe_duplicate_responsibility_policy=(config.swe.duplicate_responsibility_policy),
        director_prompt_variant=config.director_prompt_variant,
        director_tokenizer=director_tokenizer,
    )
    return AdaptiveSolverApplication(
        config=config,
        solver=solver,
        runtime=runtime,
        skillbank=skillbank,
        skill_lifecycle=lifecycle,
        mace_selector=mace_selector,
        model_router=model_router,
        owned_backends=tuple(owned_backends),
    )


def _gateway_config(
    model: RoleModelConfig,
    temperatures: dict[str, float],
    *,
    sampling_seed: int | None = None,
    director_enable_thinking: bool | None = None,
) -> ModelGatewayConfig:
    return ModelGatewayConfig(
        base_url=model.base_url,
        api_key=model.api_key,
        timeout_s=model.timeout_s,
        sampling_seed=sampling_seed,
        roles={
            role: ModelRoleConfig(
                model=model.served_model,
                temperature=temperature,
                top_p=0.95 if role == "graph-director" else 1.0,
                top_k=20 if role == "graph-director" else None,
                enable_thinking=(
                    role == "graph-director"
                    if director_enable_thinking is None
                    else role == "graph-director" and director_enable_thinking
                ),
            )
            for role, temperature in temperatures.items()
        },
    )


def _runtime_gateway_config(
    runtime: FixedRuntimeConfig,
    temperatures: dict[str, float],
    *,
    route_name: str = "",
) -> ModelGatewayConfig:
    return ModelGatewayConfig(
        base_url=runtime.base_url,
        api_key=runtime.api_key,
        api_keys_by_dataset=dict(runtime.api_keys_by_dataset),
        timeout_s=runtime.timeout_s,
        request_profile=runtime.request_profile,
        route_name=route_name,
        max_concurrency=runtime.max_concurrency,
        max_concurrency_by_dataset=dict(runtime.max_concurrency_by_dataset),
        user_agent=runtime.user_agent,
        network_path=runtime.network_path,
        stream=runtime.stream,
        roles={
            role: ModelRoleConfig(
                model=runtime.served_model,
                temperature=temperature,
                enable_thinking=runtime.enable_thinking,
                max_tokens=runtime.max_tokens,
                reasoning_effort=(
                    runtime.healthbench_grader_reasoning_effort
                    if role in {"healthbench-grader", "healthbench-grader-chat"}
                    and runtime.healthbench_grader_reasoning_effort is not None
                    else runtime.reasoning_effort
                ),
                api_surface=("responses" if role == "healthbench-grader" else runtime.api_surface),
            )
            for role, temperature in temperatures.items()
        },
    )


def _create_runtime_backend(runtime: FixedRuntimeConfig, *, route_name: str = "") -> ChatBackend:
    gateway = _runtime_gateway_config(
        runtime,
        {
            "skill-distiller": 0.0,
            "worker": 0.0,
            "healthbench-grader": 0.0,
            "healthbench-grader-chat": 0.0,
            "answer-formatter": 0.0,
        },
        route_name=route_name,
    )
    if runtime.request_profile == "gemini":
        return GeminiNativeBackend(gateway)
    return OpenAICompatibleBackend(gateway)


def create_qwen_task_proposer(
    config: AdaptiveApplicationConfig,
    *,
    backend: ChatBackend | None = None,
    tokenizer: Tokenizer | None = None,
) -> QwenTaskProposer:
    """Create the Proposer only from its own independently served model role."""

    config.validate()
    proposer_backend = backend or OpenAICompatibleBackend(
        _gateway_config(
            config.proposer_model,
            {"proposer": 0.8},
            sampling_seed=config.seed,
        )
    )
    return QwenTaskProposer(proposer_backend, tokenizer)


def create_fixed_pool_proposer(
    config: AdaptiveApplicationConfig,
    pool: Any,
    scheduler: Any,
    retriever: Any,
    *,
    backend: ChatBackend | None = None,
    tokenizer: Tokenizer | None = None,
    candidate_count: int = 8,
) -> FixedPoolQwenProposer:
    """Create the trainable selector without exposing Solver or SkillBank state."""

    config.validate()
    proposer_backend = backend or OpenAICompatibleBackend(
        _gateway_config(
            config.proposer_model,
            {"proposer": 0.8},
            sampling_seed=config.seed,
        )
    )
    return FixedPoolQwenProposer(
        proposer_backend,
        pool,
        scheduler,
        retriever,
        tokenizer,
        candidate_count=candidate_count,
    )


def create_selfplay_snapshots(config: AdaptiveApplicationConfig) -> AlternatingSnapshots:
    """Bind self-play state to distinct Proposer and Solver checkpoint roots."""

    config.validate()
    return AlternatingSnapshots(
        proposer_snapshot=_current_policy_snapshot(
            config.proposer_model.checkpoint_path, config.proposer_model.base_model_path
        ),
        solver_snapshot=_current_policy_snapshot(
            config.solver_model.checkpoint_path, config.solver_model.base_model_path
        ),
    )


def consolidate_selfplay_skills(
    config: AdaptiveApplicationConfig,
    result: Any,
    *,
    mock: bool,
    step: int,
    cycle_dir: Path | None = None,
    tokenizer: Any | None = None,
) -> tuple[tuple[str, str], ...]:
    """Apply SESA's frontier-failure gate to Solver Director skill evolution."""

    if not config.skillbank_context_enabled:
        return ()
    if config.skillbank_mode == "director_skill_v2":
        from .skill_evolution_v2 import atomic_json, consolidate

        try:
            kwargs = {"tokenizer": tokenizer} if config.pats.enabled else {}
            return consolidate(config, result, step=step, mock=mock, cycle_dir=cycle_dir, **kwargs)
        except Exception as exc:
            # Raw rollout evidence remains authoritative and can be re-ingested.
            # Skill maintenance must not cancel an otherwise valid PPO update.
            import warnings

            warnings.warn(
                f"SkillBank v2 evidence maintenance failed: {type(exc).__name__}",
                RuntimeWarning,
                stacklevel=2,
            )
            if cycle_dir is not None:
                with contextlib.suppress(OSError):
                    atomic_json(
                        cycle_dir / "skillbank_v2_maintenance_error.json",
                        {
                            "step": step,
                            "error_type": type(exc).__name__,
                            "raw_rollouts_preserved": True,
                        },
                    )
            return (("skill_v2", f"maintenance_failed:{type(exc).__name__}"),)
    application = create_adaptive_application(config, mock=mock)
    lifecycle = application.skill_lifecycle
    if lifecycle is None:
        return ()
    tasks = {task.task_id: task for task in result.tasks}
    frontiers = {frontier.task_id: frontier for frontier in result.frontier_scores}
    for sample in result.solver_batch.samples:
        if sample.metadata.get("failure_mode") == "director_protocol_failure":
            continue
        for skill_id in sample.metadata.get("skills_used", ()):
            if skill_id in lifecycle.bank.skills:
                lifecycle.bank.record_outcome(
                    skill_id,
                    step=step,
                    helpful=sample.reward > 0.0,
                    hurt=sample.reward <= 0.0,
                )
        frontier = frontiers[sample.task_id]
        success_rate = sum(float(value) > 0.0 for value in frontier.rewards) / max(
            1, len(frontier.rewards)
        )
        if sample.reward > 0.0 or not 0.0 < success_rate < 1.0:
            continue
        task = tasks[sample.task_id]
        lifecycle.collect_failure(
            SolverFailureCase(
                task=task.prompt,
                task_type=task.task_type,
                failure_trace=json.dumps(
                    sample.metadata.get("solver_trace", {}), ensure_ascii=False
                ),
                failure_mode=str(sample.metadata.get("failure_mode", "task_verification_failure")),
                reference=str(task.reference or ""),
                solver_answer=str(sample.metadata.get("solver_answer", "")),
                used_skill_ids=tuple(sample.metadata.get("skills_used", ())),
                frontier_score=success_rate,
                uid=sample.rollout_id,
            )
        )
    changes = tuple(lifecycle.evolve(step=step))
    lifecycle.save_pending(config.skill_cases_path)
    lifecycle.bank.save()
    return changes


def _current_policy_snapshot(checkpoint_root: Path, base_model_path: Path) -> str:
    latest = checkpoint_root / "latest.json"
    if not latest.exists():
        return str(base_model_path.resolve())
    payload = json.loads(latest.read_text(encoding="utf-8"))
    checkpoint = Path(str(payload["path"]))
    return str(checkpoint.resolve()) if checkpoint.exists() else str(base_model_path.resolve())


def _verifier(
    name: str,
    *,
    adapters: dict[str, Verifier] | None = None,
    aliases: dict[str, str] | None = None,
) -> Verifier | None:
    builtins: dict[str, Verifier | None] = {
        "none": None,
        "auto": AutoVerifier(adapters, aliases=aliases),
        "exact_match": ExactMatchVerifier(),
        "multi_answer_exact_match": MultiAnswerExactMatchVerifier(),
        "flowsteer_qa": FlowSteerQAVerifier(),
        "token_f1": TokenF1Verifier(),
        "numeric": NumericVerifier(),
        "multiple_choice": MultipleChoiceVerifier(),
    }
    if name in builtins:
        return builtins[name]
    selected = (adapters or {}).get(name)
    if selected is None:
        raise ValueError(f"verifier {name!r} requires a registered dataset adapter")
    return selected


def _mock_adaptive_backends(
    *, binary_relations: bool = False, runtime_route: str = "default"
) -> tuple[MockBackend, MockBackend, MockBackend]:
    relation_action = "consider_relation" if binary_relations else "set_relation"
    director = MockBackend(
        [
            '{"action":"add_agent","agent_id":"analyst"}',
            '{"action":"set_prompt","target":"analyst",'
            '"role":"Independent analyst",'
            '"objective":"Solve the assigned task independently.",'
            '"scope":"Develop and verify a candidate result.",'
            '"expected_output":"Return a concise answer with essential justification."}',
            json.dumps(
                {"action": "set_model", "target": "analyst", "runtime_route": runtime_route}
            ),
            '{"action":"add_agent","agent_id":"verifier"}',
            '{"action":"set_prompt","target":"verifier",'
            '"role":"Independent verifier",'
            '"objective":"Check the assigned task and produce the best answer.",'
            '"scope":"Verify correctness independently.",'
            '"expected_output":"Return the corrected direct final answer."}',
            json.dumps(
                {"action": "set_model", "target": "verifier", "runtime_route": runtime_route}
            ),
            json.dumps(
                {
                    "action": relation_action,
                    "source": "analyst",
                    "target": "verifier",
                    **({} if binary_relations else {"relation": "bidirectional"}),
                }
            ),
            '{"action":"add_agent","agent_id":"formatter"}',
            '{"action":"set_prompt","target":"formatter",'
            '"role":"Final formatter",'
            '"objective":"Synthesize visible artifacts into the requested final form.",'
            '"scope":"Resolve disagreement using only visible artifacts.",'
            '"expected_output":"Return one concise direct final answer."}',
            json.dumps(
                {"action": "set_model", "target": "formatter", "runtime_route": runtime_route}
            ),
            '{"action":"set_layer","target":"formatter","layer":1}',
            json.dumps(
                {
                    "action": relation_action,
                    "source": "analyst",
                    "target": "formatter",
                    **({} if binary_relations else {"relation": "directed"}),
                }
            ),
            json.dumps(
                {
                    "action": relation_action,
                    "source": "verifier",
                    "target": "formatter",
                    **({} if binary_relations else {"relation": "directed"}),
                }
            ),
            '{"action":"set_output","target":"formatter"}',
            '{"action":"finish"}',
        ],
        binary_responses=[
            BinaryChoiceResponse(
                choice="on",
                model="mock:graph-director",
                probabilities={"off": 0.5, "on": 0.5},
                log_probabilities={"off": -0.6931471805599453, "on": -0.6931471805599453},
                token_ids={"off": 256, "on": 257},
                metadata={"mock": True},
            )
            for _ in range(3)
            if binary_relations
        ],
    )
    worker = MockBackend(
        handler=lambda _messages, _role: json.dumps(
            {
                "answer": "mock adaptive output",
                "summary": "mock adaptive output",
                "confidence": 1.0,
                "evidence": [],
                "unresolved_issues": [],
                "tool_summary": [],
            }
        )
    )
    distiller = MockBackend(
        handler=lambda _messages, _role: json.dumps(
            {
                "name": "Verify before finalizing",
                "description": "Add an explicit verification pass after a failed workflow.",
                "trigger": "A prior workflow of the same task type failed verification.",
                "plan": "Compare the failed trace with a successful trace and verify the output.",
                "pitfall": "Do not copy an answer without checking it.",
                "constraint": "Workflow Solver only.",
                "kind": "verification",
            }
        )
    )
    return director, worker, distiller
