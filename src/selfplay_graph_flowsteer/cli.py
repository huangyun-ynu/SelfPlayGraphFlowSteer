from __future__ import annotations
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from .ads_preprocessing import ADSPreprocessingConfig, prepare_ads_pool
from .application import (
    create_adaptive_application,
    create_fixed_pool_proposer,
    create_qwen_task_proposer,
    create_selfplay_snapshots,
    load_adaptive_config,
)
from .benchmark import BenchmarkRunner, load_flowsteer_records, write_benchmark
from .curriculum import ADSBoundaryScheduler, CurriculumProfile, FixedTaskPool, TSDSRetriever
from .features import E5DelegationEncoder, SemanticGraphFeatureExtractor
from .graph import MultiAgentGraph
from .latency import RouteLatencyTracker, RouteTokenTracker
from .learning import load_fixed_jsonl
from .llm import MockBackend
from .observability import JSONLTraceStore, replay_trace
from .probability_cache import AsyncSolverProbabilityCache, probability_cache_binding
from .qwen_compat import qwen_vllm_server_args, qwen_vllm_server_env
from .rollouts import TokenizedDirectorTrajectory
from .runtime import ModelAgentExecutor, MultiAgentRuntime
from .selfplay import (
    AlternatingSnapshots,
    DryRunSelfPlayCoordinator,
    FrontierScore,
    ProposedTask,
    SeedInput,
    SelfPlaySeed,
    SolverRollout,
    load_selfplay_seed_jsonl,
    normalize_selfplay_seed,
)
from .selfplay_runtime import (
    PRIMARY_JOB_ORDER_CHOICES,
    ByteTokenizer,
    HuggingFaceTokenizer,
    SelfPlayRolloutRunner,
    SelfPlayRunConfig,
    _read_jsonl,
)
from .services import ModelServiceSpec, VLLMServiceManager
from .training import (
    AlternatingGRPOTrainer,
    AlternatingTrainingConfig,
    MockPolicyTrainer,
    PolicyTrainingConfig,
    PolicyUpdateResult,
    load_relation_credits,
    load_training_batch,
    training_device_for_gpu,
)
from .training_metrics import TrainingMetricsStore, collect_cycle_metrics

VERIFIERS = (
    "none",
    "auto",
    "exact_match",
    "multi_answer_exact_match",
    "numeric",
    "multiple_choice",
    "healthbench_rubric",
    "healthbench_rubric_low_cost",
)
DEFAULT_CURRICULUM_PROFILE = (
    Path(__file__).resolve().parents[2] / "configs" / "curriculum" / "joint_1500.toml"
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class _CollectedExperimentCycle:
    cycle: int
    cycle_dir: Path
    seeds: list[Any]
    result: Any
    collection_config: Any
    training_config: AlternatingTrainingConfig
    evaluation_only: bool
    resumed_partial_cycle: bool
    started_monotonic: float
    collection_elapsed_s: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelfPlayGraphFlowSteer runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config", help="parse and validate a TOML config")
    validate.add_argument("path", type=Path)
    adaptive = commands.add_parser(
        "adaptive-solve", help="run Director model selection + MANTA communication + Verifier"
    )
    adaptive.add_argument("--task", required=True)
    adaptive.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "adaptive.toml",
    )
    adaptive.add_argument("--mock", action="store_true")
    adaptive.add_argument("--task-id", default="task")
    adaptive.add_argument("--task-type", default="general")
    adaptive.add_argument("--reference")
    adaptive.add_argument("--run-id")
    adaptive.add_argument("--verifier", choices=VERIFIERS)
    for role in ("proposer", "solver"):
        adaptive.add_argument(f"--{role}-base-url")
        adaptive.add_argument(f"--{role}-api-key")
        adaptive.add_argument(f"--{role}-model")
        adaptive.add_argument(f"--{role}-checkpoint", type=Path)
    adaptive.add_argument("--runtime-base-url")
    adaptive.add_argument("--runtime-api-key")
    adaptive.add_argument("--runtime-model")
    selfplay = commands.add_parser(
        "selfplay-rollout", help="collect real Proposer/Solver rollouts and training batches"
    )
    selfplay.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "adaptive.toml",
    )
    selfplay.add_argument("--seed-data", type=Path)
    selfplay.add_argument(
        "--task-pool",
        type=Path,
        action="append",
        default=[],
        help="train JSONL pool(s) used by ADS+TSDS fixed-pool selection",
    )
    selfplay.add_argument("--seed", action="append", default=[])
    selfplay.add_argument("--num-tasks", type=int)
    selfplay.add_argument("--rollouts", type=int, default=5)
    selfplay.add_argument("--base-seed", type=int, default=0)
    selfplay.add_argument("--max-tokens", type=int, default=4096)
    selfplay.add_argument("--counterfactuals-per-rollout", type=int, default=1)
    selfplay.add_argument("--workers", type=int, default=1)
    selfplay.add_argument("--task-window", type=int, default=1)
    selfplay.add_argument(
        "--task-scheduling-policy",
        choices=("logical_windows", "frozen_manifest_dynamic"),
        default="logical_windows",
    )
    selfplay.add_argument(
        "--primary-job-order",
        choices=tuple(sorted(PRIMARY_JOB_ORDER_CHOICES)),
        default="round_robin",
        help="order primary rollout admission; long_tail_first uses versioned duration estimates",
    )
    selfplay.add_argument("--max-active-task-groups", type=int, default=8)
    selfplay.add_argument("--pipeline-counterfactuals", action="store_true")
    selfplay.add_argument("--counterfactual-workers", type=int, default=2)
    selfplay.add_argument("--counterfactual-pair-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--rollout-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--stateful-rollout-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--swe-rollout-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--rollout-slot-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--stateful-slot-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--swe-slot-wall-time-s", type=float, default=900.0)
    selfplay.add_argument("--swe-non-trainable-recovery-attempts", type=int, default=1)
    selfplay.add_argument("--non-swe-recovery-attempts", type=int, default=1)
    selfplay.add_argument("--rollout-no-progress-time-s", type=float, default=120.0)
    selfplay.add_argument("--request-wall-time-s", type=float, default=180.0)
    selfplay.add_argument("--replacement-rollouts-per-task", type=int, default=2)
    selfplay.add_argument("--proposals-per-seed", type=int, default=1)
    selfplay.add_argument("--output", type=Path, required=True)
    selfplay.add_argument(
        "--resume",
        action="store_true",
        help="reload a collection only after all primary rollouts are durable; exact missing-ID recollection is disabled",
    )
    selfplay.add_argument("--mock", action="store_true")
    selfplay.add_argument("--verifier", choices=VERIFIERS)
    train = commands.add_parser(
        "train-cycle", help="update Proposer then Solver from one collected rollout directory"
    )
    train.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "adaptive.toml",
    )
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--proposer-learning-rate", type=float, default=1e-05)
    train.add_argument("--solver-learning-rate", type=float, default=1e-05)
    train.add_argument("--epochs", type=int, default=1)
    _add_grpo_arguments(train)
    train.add_argument(
        "--device", help="explicit device override; default/cuda uses the role GPU from [resources]"
    )
    train.add_argument("--full-finetune", action="store_true")
    train.add_argument(
        "--allow-stale-rollouts",
        action="store_true",
        help="allow a batch collected by checkpoints other than the currently loaded policies",
    )
    train.add_argument("--mock-trainer", action="store_true")
    train.add_argument("--manage-services", action="store_true")
    train.add_argument("--service-state-dir", type=Path, default=Path("state/services"))
    benchmark = commands.add_parser(
        "benchmark", help="run a fixed JSONL dataset and optionally compare FlowSteer"
    )
    benchmark.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "adaptive.toml",
    )
    benchmark.add_argument("--dataset", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--verifier", choices=VERIFIERS, default="auto")
    benchmark.add_argument("--seed", type=int, action="append", default=[])
    benchmark.add_argument("--flowsteer-baseline", type=Path)
    benchmark.add_argument("--mock", action="store_true")
    services = commands.add_parser("model-services", help="manage owned vLLM model services")
    services.add_argument("action", choices=("status", "start", "stop", "refresh"))
    services.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "adaptive.toml",
    )
    services.add_argument("--state-dir", type=Path, default=Path("state/services"))
    services.add_argument("--role", choices=("proposer", "solver", "runtime", "all"), default="all")
    services.add_argument("--checkpoint", type=Path)
    services.add_argument("--wait-s", type=float, default=120.0)
    ads = commands.add_parser(
        "prepare-ads-pool", help="extract base-policy embeddings/NLL and add ADS K-Means metadata"
    )
    ads.add_argument("--input", type=Path, action="append", required=True)
    ads.add_argument("--output", type=Path, required=True)
    ads.add_argument("--artifacts-dir", type=Path, required=True)
    ads.add_argument("--model-path", type=Path, default=Path("models/Qwen3.5-9B"))
    ads.add_argument("--num-clusters", type=int, required=True)
    ads.add_argument("--device", default="cuda:0")
    ads.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ads.add_argument("--batch-size", type=int, default=1)
    ads.add_argument("--max-length", type=int, default=4096)
    ads.add_argument("--pca-dim", type=int, default=128)
    ads.add_argument("--seed", type=int, default=42)
    ads.add_argument("--overwrite", action="store_true")
    experiment = commands.add_parser(
        "selfplay-experiment",
        help="run repeated rollout -> Proposer update -> Solver update cycles",
    )
    experiment.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "adaptive.toml",
    )
    experiment.add_argument("--seed-data", type=Path)
    experiment.add_argument(
        "--frozen-pool-selection",
        type=Path,
        help="replay the exact fixed-pool selections recorded in a tasks.jsonl file; requires --task-pool and --mock-trainer and is intended for paired evaluation",
    )
    experiment.add_argument(
        "--task-pool",
        type=Path,
        action="append",
        default=[],
        help="train JSONL pool(s) used by ADS+TSDS fixed-pool selection",
    )
    experiment.add_argument(
        "--curriculum-profile",
        type=Path,
        default=DEFAULT_CURRICULUM_PROFILE,
        help="joint-pool curriculum sizing profile; defaults to the 3-dataset/1500-task regime",
    )
    experiment.add_argument("--tasks-per-cycle", type=int)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument(
        "--route-report",
        type=Path,
        help="fresh probe report covering configured Worker routes and enabled endpoint-pool members",
    )
    experiment.add_argument(
        "--route-subset",
        default="",
        help="optional comma-separated subset of routes qualified by --route-report",
    )
    experiment.add_argument(
        "--minimum-selected-routes",
        type=int,
        choices=(1, 4, 5),
        default=5,
        help="minimum qualified Worker routes selected from a six-route report; four enables degraded-route experiments; one requires frozen, single-cycle, no-MACE evaluation",
    )
    experiment.add_argument("--max-route-report-age-s", type=float, default=1800.0)
    experiment.add_argument(
        "--canary-exclude-migrated-frontier",
        action="store_true",
        help="exclude incomparable Proposer samples after explicit canary Executor migration",
    )
    experiment.add_argument(
        "--checkpoint-root",
        type=Path,
        help="experiment-owned checkpoint directory; defaults to OUTPUT/checkpoints so debug and formal runs cannot overwrite the configured policy lineage",
    )
    experiment.add_argument(
        "--runtime-state-root",
        type=Path,
        help="experiment-owned trace directory; defaults to OUTPUT/runtime_state",
    )
    experiment.add_argument("--cycles", type=int, default=1)
    experiment.add_argument(
        "--collection-only",
        action="store_true",
        help="Preserve training collection, CF and Frontier, but skip both policy updates .",
    )
    experiment.add_argument(
        "--final-cycle-collection-only",
        action="store_true",
        help="Keep final-cycle training collection and async overlap, save batches, then skip its updates.",
    )
    experiment.add_argument(
        "--final-cycle-evaluation-only",
        action="store_true",
        help="collect and score the final cycle without updating either policy; the final cycle is recorded as evaluation_only and contributes no optimizer steps",
    )
    experiment.add_argument(
        "--resume",
        action="store_true",
        help="skip completed cycles and resume training from complete rollout batches; incomplete rollout collection is preserved but not recollected",
    )
    experiment.add_argument(
        "--resume-planned-interruption",
        action="store_true",
        help="Resume exact missing slots after an intentional configuration reload; preserve saved outcomes.",
    )
    experiment.add_argument(
        "--resume-transient-backend-circuit",
        action="store_true",
        help="with --resume, reopen only groups quarantined by a preserved transient Worker backend circuit and collect their exact missing rollout IDs",
    )
    experiment.add_argument(
        "--resume-after-attribution-classifier-repair",
        action="store_true",
        help="with --resume, continue an interrupted collection only after a verified attribution-classifier repair attestation",
    )
    experiment.add_argument(
        "--resume-after-infrastructure-repair",
        action="store_true",
        help="with --resume, recollect a preserved immutable task manifest after an attested environment or tool repair",
    )
    experiment.add_argument(
        "--resume-after-uncertain-group-recollection",
        action="store_true",
        help="with --resume, discard and recollect every sibling of an attested task group whose attribution-uncertain recovery was exhausted",
    )
    experiment.add_argument(
        "--resume-after-proposal-selection-repair",
        action="store_true",
        help="with --resume, fill attested fixed-pool proposal holes while retaining durable unaffected proposals and primary rollouts",
    )
    experiment.add_argument("--rollouts", type=int)
    experiment.add_argument("--proposals-per-seed", type=int, default=1)
    experiment.add_argument("--workers", type=int)
    experiment.add_argument("--task-window", type=int)
    experiment.add_argument(
        "--task-scheduling-policy",
        choices=("logical_windows", "frozen_manifest_dynamic"),
        default="frozen_manifest_dynamic",
    )
    experiment.add_argument(
        "--primary-job-order",
        choices=tuple(sorted(PRIMARY_JOB_ORDER_CHOICES)),
        default="round_robin",
        help="order primary rollout admission; long_tail_first uses versioned duration estimates",
    )
    experiment.add_argument("--max-active-task-groups", type=int, default=8)
    experiment.add_argument(
        "--uncertain-attribution-zero-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score attribution-uncertain failures as zero after bounded recovery; preserve policy-data gates",
    )
    experiment.add_argument(
        "--backend-failure-retries",
        type=int,
        default=0,
        help="positive values preserve failed backend slots and continue independent collection; no whole-rollout replay in normal runs (retry count is legacy-only); 0 aborts on a backend incident",
    )
    experiment.add_argument(
        "--proposer-learning-mode",
        choices=("independent_frontier_v2", "legacy"),
        default="independent_frontier_v2",
        help="new collections use partial Frontier evidence and frozen dataset EMA advantages",
    )
    experiment.add_argument("--proposer-baseline-mode", choices=("none", "ema"), default="ema")
    experiment.add_argument("--proposer-baseline-decay", type=float, default=0.9)
    experiment.add_argument(
        "--rollout-group-policy",
        choices=("eligible_subset", "complete"),
        default="eligible_subset",
        help="eligible_subset keeps 2+ admitted siblings; complete preserves the legacy K gate",
    )
    experiment.add_argument(
        "--allow-skipped-task-groups",
        action="store_true",
        help="legacy complete mode: allow remaining complete groups; eligible_subset already admits groups with at least two qualified trajectories",
    )
    experiment.add_argument(
        "--continue-on-uncertain-attribution-exhausted",
        action="store_true",
        help="diagnostic-only: quarantine a task group after its one attribution-uncertain same-slot recovery is exhausted, instead of aborting the whole cycle; requires --allow-skipped-task-groups in complete mode",
    )
    experiment.add_argument(
        "--enable-swe",
        action="store_true",
        help="enable the configured official SWE adapter for this experiment",
    )
    experiment.add_argument(
        "--director-prompt-variant",
        choices=("v2", "v2.1"),
        help="override the configured Director prompt for an explicitly recorded A/B arm",
    )
    experiment.add_argument(
        "--policy-gpu-id",
        type=int,
        help="place experiment-owned Proposer/Solver services on one explicit physical GPU",
    )
    experiment.add_argument(
        "--proposer-gpu-id",
        type=int,
        help="place the experiment-owned Proposer service/trainer on this physical GPU",
    )
    experiment.add_argument(
        "--solver-gpu-id",
        type=int,
        help="place the experiment-owned Solver service/trainer on this physical GPU",
    )
    experiment.add_argument("--pipeline-counterfactuals", action="store_true")
    experiment.add_argument(
        "--pipeline-frontier-by-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="submit a dataset's Frontier reverify as soon as its primary rows are durable",
    )
    experiment.add_argument(
        "--pipeline-next-cycle-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="prepare the next fixed-pool selection and metadata while the current update runs",
    )
    experiment.add_argument(
        "--async-next-cycle-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="collect cycle N+1 from the frozen cycle-N policy snapshots while cycle N updates; requires a distinct inference GPU and permits at most one stale update",
    )
    experiment.add_argument(
        "--async-rollout-gpu-id",
        type=int,
        help="dedicated inference GPU for bounded-stale next-cycle collection",
    )
    experiment.add_argument("--async-rollout-port", type=int, default=18013)
    experiment.add_argument("--async-rollout-gpu-memory-utilization", type=float, default=0.75)
    experiment.add_argument(
        "--cycle-target-time-s",
        type=float,
        default=0.0,
        help="performance target only; never truncate a cycle or optimizer step",
    )
    experiment.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="numeric-only telemetry; local records always retained",
    )
    experiment.add_argument("--counterfactual-workers", type=int, default=2)
    experiment.add_argument("--counterfactual-pair-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--rollout-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--stateful-rollout-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--swe-rollout-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--rollout-slot-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--stateful-slot-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--swe-slot-wall-time-s", type=float, default=900.0)
    experiment.add_argument("--swe-non-trainable-recovery-attempts", type=int, default=1)
    experiment.add_argument("--non-swe-recovery-attempts", type=int, default=1)
    experiment.add_argument("--rollout-no-progress-time-s", type=float, default=120.0)
    experiment.add_argument("--request-wall-time-s", type=float, default=180.0)
    experiment.add_argument("--replacement-rollouts-per-task", type=int, default=2)
    experiment.add_argument("--counterfactuals-per-rollout", type=int, default=1)
    experiment.add_argument("--verifier", choices=VERIFIERS, default="auto")
    experiment.add_argument("--proposer-learning-rate", type=float, default=1e-05)
    experiment.add_argument("--solver-learning-rate", type=float, default=1e-05)
    experiment.add_argument("--epochs", type=int, default=1)
    _add_grpo_arguments(experiment)
    experiment.add_argument(
        "--device", help="explicit device override; default/cuda uses the role GPU from [resources]"
    )
    experiment.add_argument("--full-finetune", action="store_true")
    experiment.add_argument("--mock", action="store_true")
    experiment.add_argument("--mock-trainer", action="store_true")
    experiment.add_argument(
        "--freeze-runtime-state", action="store_true", help="disable persistent runtime updates"
    )
    experiment.add_argument("--manage-services", action="store_true")
    experiment.add_argument("--service-state-dir", type=Path, default=Path("state/services"))
    build = commands.add_parser("build-graph", help="validate a graph JSON document")
    build.add_argument("path", type=Path)
    inspect_trace = commands.add_parser("inspect-trace", help="inspect stored trace JSONL")
    inspect_trace.add_argument("path", type=Path)
    inspect_trace.add_argument("--run-id")
    replay = commands.add_parser("replay", help="replay accepted Canvas actions")
    replay.add_argument("path", type=Path)
    replay.add_argument("--run-id", required=True)
    dry = commands.add_parser("dry-run-selfplay", help="build self-play batches without training")
    dry.add_argument("--mock", action="store_true", required=True)
    dry.add_argument("--num-tasks", type=int, default=2)
    dry.add_argument("--rollouts", type=int, default=5)
    dry.add_argument("--output", type=Path)
    return parser


def _nominal_optimizer_steps(
    cycles: int, samples_per_cycle: int, epochs: int, mini_batch_size: int
) -> int:
    """Each cycle/epoch flushes its own short tail; samples never carry over."""
    return max(1, cycles * epochs * math.ceil(samples_per_cycle / mini_batch_size))


def _add_grpo_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the FlowSteer optimizer recipe without coupling the two policy LRs."""
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coefficient", type=float, default=0.005)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--total-optimizer-steps",
        type=int,
        help="legacy shared scheduler horizon; experiment mode computes role horizons when omitted",
    )
    parser.add_argument(
        "--proposer-total-optimizer-steps",
        type=int,
        help="Proposer scheduler horizon; does not change shared batch/accumulation settings",
    )
    parser.add_argument(
        "--solver-total-optimizer-steps",
        type=int,
        help="Solver scheduler horizon; does not change shared batch/accumulation settings",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=64,
        help="logical PPO/GRPO mini-batch size (SESA recipe: 64)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="maximum per-device dynamic micro-batch size",
    )
    parser.add_argument(
        "--max-micro-batch-tokens",
        type=int,
        default=16384,
        help="maximum padded tokens in one dynamic device micro-batch",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="nominal value for manifests; dynamic packing accumulates actual samples to the mini-batch",
    )
    parser.add_argument(
        "--parallel-role-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train Proposer and Solver concurrently on distinct configured GPUs",
    )
    parser.add_argument(
        "--solver-data-parallel-gpus",
        type=int,
        nargs=2,
        default=None,
        help="two physical GPUs for one global Solver update; roles run sequentially",
    )
    parser.add_argument(
        "--raw-policy-backward-mode", choices=("micro", "call", "timeline"), default="micro"
    )
    parser.add_argument(
        "--solver-training-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Solver learner precision; leaves Proposer and serving precision unchanged",
    )
    parser.add_argument(
        "--async-solver-probability-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="precompute frozen Solver/reference probabilities on the Proposer GPU during collection",
    )
    parser.add_argument(
        "--short-call-batching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental short-call batching inside probability precomputation; keep disabled until numerical acceptance",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--activation-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stage saved autograd tensors on CPU to reduce training-only GPU peak memory",
    )
    parser.add_argument(
        "--activation-cpu-offload-min-tokens",
        type=int,
        default=0,
        help="when activation CPU offload is enabled, stage only micro-batches containing a sequence at least this long; zero stages every call",
    )


def adaptive_solve(args: argparse.Namespace) -> int:
    config = load_adaptive_config(args.config)
    if args.verifier is not None:
        config = replace(config, verifier=args.verifier)
    for role in ("proposer", "solver"):
        current = getattr(config, f"{role}_model")
        role_overrides = {
            field_name: value
            for (field_name, value) in (
                ("base_url", getattr(args, f"{role}_base_url")),
                ("api_key", getattr(args, f"{role}_api_key")),
                ("served_model", getattr(args, f"{role}_model")),
                ("checkpoint_path", getattr(args, f"{role}_checkpoint")),
            )
            if value is not None
        }
        if role_overrides:
            config = replace(config, **{f"{role}_model": replace(current, **role_overrides)})
    runtime_overrides = {
        key: value
        for (key, value) in (
            ("base_url", args.runtime_base_url),
            ("api_key", args.runtime_api_key),
            ("served_model", args.runtime_model),
        )
        if value is not None
    }
    if runtime_overrides:
        config = replace(config, runtime=replace(config.runtime, **runtime_overrides))
    application = create_adaptive_application(config, mock=args.mock)
    result = application.solve(
        args.task,
        task_id=args.task_id,
        task_type=args.task_type,
        reference=args.reference,
        run_id=args.run_id,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.solver_result.director_run.finished else 2


def validate_config(path: Path) -> int:
    config = load_adaptive_config(path)
    print(
        json.dumps(
            {
                "valid": True,
                "proposer": config.proposer_model.served_model,
                "solver": config.solver_model.served_model,
                "runtime": config.runtime.served_model,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_graph(path: Path) -> int:
    graph = MultiAgentGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    graph.assert_valid(final=True)
    print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
    return 0


def inspect_trace(path: Path, run_id: str | None) -> int:
    store = JSONLTraceStore(path)
    payload: object = store.load(run_id).to_dict() if run_id else store.list_run_ids()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def replay(path: Path, run_id: str) -> int:
    backend = MockBackend()
    runtime = MultiAgentRuntime(ModelAgentExecutor(backend))
    canvas = replay_trace(JSONLTraceStore(path).load(run_id), runtime=runtime)
    print(json.dumps(canvas.graph.to_dict(), ensure_ascii=False, indent=2))
    return 0


class _MockProposer:
    def propose(self, seed: SeedInput, *, task_id: str) -> ProposedTask:
        from .observability import TaskSpec

        seed_spec = normalize_selfplay_seed(seed)
        response = json.dumps({"prompt": f"Resolve {seed_spec.content}", "reference": "ok"})
        return ProposedTask(
            task=TaskSpec(
                task_id,
                f"Resolve {seed_spec.content}",
                reference=seed_spec.target_answer
                if seed_spec.target_answer not in (None, "", [], {})
                else "ok",
                metadata={
                    "selfplay_seed": seed_spec.to_dict(),
                    "required_reasoning_hops": seed_spec.required_reasoning_hops,
                },
            ),
            response=response,
            token_ids=tuple(response.encode()),
            action_mask=tuple((1 for _ in response.encode())),
        )


def _mock_solver(task: object, rollout_index: int) -> SolverRollout:
    task_id = str(task.task_id)
    graph = MultiAgentGraph()
    graph.add_agent("solver")
    graph.set_prompt("solver", "Solve and verify the task.")
    graph.set_output("solver")
    response = f'{{"action":"finish","sample":{rollout_index}}}'
    trajectory = TokenizedDirectorTrajectory(
        rollout_id=f"{task_id}-r{rollout_index}",
        task_id=task_id,
        token_ids=tuple(response.encode()),
        action_mask=tuple((1 for _ in response.encode())),
        reward=float(rollout_index % 2 == 0),
        graph=graph.to_dict(),
        seed=rollout_index,
        executor_version="mock-v1",
    )
    return SolverRollout(trajectory, graph)


def dry_run_selfplay(num_tasks: int, rollouts: int, output: Path | None) -> int:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    result = DryRunSelfPlayCoordinator(
        proposer=_MockProposer(), solve=_mock_solver, rollouts_per_task=rollouts
    ).run((f"seed-{index}" for index in range(num_tasks)))
    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def selfplay_rollout(args: argparse.Namespace) -> int:
    config = replace(load_adaptive_config(args.config), persist_runtime_updates=False)
    if args.verifier is not None:
        config = replace(config, verifier=args.verifier)
    if not args.mock and config.verifier == "none":
        raise ValueError(
            "real self-play requires a task verifier; choose exact_match or another mode"
        )
    seeds = list(args.seed)
    if args.seed_data:
        seeds.extend(_load_seeds(args.seed_data))
    fixed_components = None
    if args.task_pool:
        fixed_components = _fixed_pool_components(
            args.task_pool, config=config, mock=args.mock, seed=args.base_seed
        )
        seeds = fixed_components[0].selection_seeds(args.num_tasks or 128)
    seeds = [seed for seed in seeds if not isinstance(seed, str) or seed.strip()]
    if args.num_tasks is not None:
        seeds = seeds[: args.num_tasks]
    if not seeds:
        raise ValueError("provide --task-pool, --seed, or --seed-data")
    tokenizer = (
        ByteTokenizer()
        if args.mock
        else HuggingFaceTokenizer(config.proposer_model.base_model_path)
    )
    graph_feature_extractor = (
        None
        if args.mock
        else SemanticGraphFeatureExtractor(
            E5DelegationEncoder(config.graph_embedding_model_path or "intfloat/e5-base-v2")
        )
    )
    if fixed_components is not None:
        (pool, scheduler, retriever, backend) = fixed_components
        proposer = create_fixed_pool_proposer(
            config, pool, scheduler, retriever, backend=backend, tokenizer=tokenizer
        )
    elif args.mock:
        proposer = _MockProposer()
    else:
        proposer = create_qwen_task_proposer(config, tokenizer=tokenizer)
    route_latency_tracker = RouteLatencyTracker(window_size=config.canvas.worker_latency_window)
    route_token_tracker = RouteTokenTracker(
        window_size=config.canvas.worker_token_window, path=args.output / "route_token_usage.json"
    )

    def application_factory(seed: int):
        seeded = replace(config, seed=seed)
        return create_adaptive_application(
            seeded,
            mock=args.mock,
            route_latency_tracker=route_latency_tracker,
            route_token_tracker=route_token_tracker,
            director_tokenizer=tokenizer,
        )

    result = SelfPlayRolloutRunner(
        proposer=proposer,
        application_factory=application_factory,
        tokenizer=tokenizer,
        snapshots=create_selfplay_snapshots(config),
        output_dir=args.output,
        config=SelfPlayRunConfig(
            args.rollouts,
            args.base_seed,
            args.max_tokens,
            args.counterfactuals_per_rollout,
            args.workers,
            args.proposals_per_seed,
            args.task_window,
            require_all_proposals=args.task_scheduling_policy == "frozen_manifest_dynamic",
            task_scheduling_policy=args.task_scheduling_policy,
            primary_job_order=args.primary_job_order,
            max_active_task_groups=args.max_active_task_groups,
            structural_exploration_policy=config.canvas.structural_exploration_policy,
            rollout_wall_time_s=args.rollout_wall_time_s,
            stateful_rollout_wall_time_s=args.stateful_rollout_wall_time_s,
            swe_rollout_wall_time_s=args.swe_rollout_wall_time_s,
            rollout_slot_wall_time_s=args.rollout_slot_wall_time_s,
            stateful_slot_wall_time_s=args.stateful_slot_wall_time_s,
            swe_slot_wall_time_s=args.swe_slot_wall_time_s,
            swe_non_trainable_recovery_attempts=args.swe_non_trainable_recovery_attempts,
            non_swe_recovery_attempts=args.non_swe_recovery_attempts,
            rollout_no_progress_time_s=args.rollout_no_progress_time_s,
            request_wall_time_s=args.request_wall_time_s,
            replacement_rollouts_per_task=args.replacement_rollouts_per_task,
            route_health_path=config.route_health_path,
            route_health_cooldown_s=config.route_health_cooldown_s,
            worker_runtime_routes=config.worker_runtime_routes,
            pipeline_counterfactuals=args.pipeline_counterfactuals,
            counterfactual_workers=args.counterfactual_workers,
            counterfactual_pair_wall_time_s=args.counterfactual_pair_wall_time_s,
            task_execution_window=max(args.task_window, args.max_active_task_groups),
        ),
        graph_feature_extractor=graph_feature_extractor,
    ).run(seeds, resume=args.resume)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def train_cycle(args: argparse.Namespace) -> int:
    from .training_selection import verify_frozen_selection

    frozen_selection = verify_frozen_selection(args.run_dir)
    proposer_batch = load_training_batch(args.run_dir / "proposer_batch.json")
    solver_batch = load_training_batch(args.run_dir / "solver_batch.json")
    if solver_batch.metadata.get("training_selection_schema") and frozen_selection is None:
        raise ValueError("partial-group training requires its frozen selection manifest")
    adaptive = load_adaptive_config(args.config)
    durable_training_state = _load_optional_json(args.run_dir / "training_state.json") or {}
    pending_metrics_updates = _pending_cycle_metrics_updates(
        durable_training_state, args.run_dir / "training_metrics_latest.json"
    )
    validation_roles = (
        ()
        if pending_metrics_updates is not None
        else ("solver",)
        if durable_training_state.get("phase") == "solver"
        else ("proposer", "solver")
    )
    active_roles = {b.role for b in (proposer_batch, solver_batch) if b.samples}
    validation_roles = tuple((role for role in validation_roles if role in active_roles))
    if validation_roles:
        _validate_rollout_policy_snapshots(
            args.run_dir, adaptive, allow_stale=args.allow_stale_rollouts, roles=validation_roles
        )
    (proposer_device, solver_device) = _training_devices(adaptive, args.device)
    if not set(args.solver_data_parallel_gpus or ()) <= set(adaptive.allocated_gpu_ids):
        raise ValueError("Solver data-parallel GPUs must be explicitly allocated in config")
    proposer = _policy_training_config(
        adaptive.proposer_model.base_model_path,
        adaptive.proposer_model.checkpoint_path,
        args.proposer_learning_rate,
        proposer_device,
        args,
        total_optimizer_steps=args.proposer_total_optimizer_steps
        or args.total_optimizer_steps
        or 300,
    )
    solver = _policy_training_config(
        adaptive.solver_model.base_model_path,
        adaptive.solver_model.checkpoint_path,
        args.solver_learning_rate,
        solver_device,
        args,
        total_optimizer_steps=args.solver_total_optimizer_steps
        or args.total_optimizer_steps
        or 300,
    )
    config = AlternatingTrainingConfig(
        proposer,
        replace(
            solver,
            dtype=getattr(args, "solver_training_dtype", "bfloat16"),
            data_parallel_gpu_ids=tuple(args.solver_data_parallel_gpus or ()),
        ),
        args.run_dir / "training_state.json",
        parallel_roles=args.parallel_role_training,
    )
    factory = (
        (lambda role, policy, seed: MockPolicyTrainer(role, policy, seed=seed))
        if args.mock_trainer
        else None
    )
    manager = VLLMServiceManager(args.service_state_dir) if args.manage_services else None
    before_role = checkpoint_callback = None
    initially_running: dict[str, bool] = {}
    if manager:
        specs = _service_specs(adaptive)
        initially_running = {
            role: manager.status(specs[role]).running for role in ("proposer", "solver")
        }

        def before_role(role: str) -> None:
            roles = ("proposer", "solver") if config.solver.data_parallel_gpu_ids else (role,)
            for target in roles:
                if initially_running[target]:
                    manager.stop(target)

        def checkpoint_callback(role: str, checkpoint: str) -> None:
            if config.solver.data_parallel_gpu_ids:
                return
            if initially_running[role]:
                manager.refresh(_service_specs(adaptive)[role], checkpoint)

    trainer = AlternatingGRPOTrainer(
        config,
        trainer_factory=factory,
        before_role_callback=before_role,
        checkpoint_callback=checkpoint_callback,
    )
    relation_credits = load_relation_credits(
        args.run_dir
        / (
            "training_relation_credits.jsonl"
            if (args.run_dir / "training_selection.json").exists()
            else "relation_counterfactuals.jsonl"
        )
    )
    if pending_metrics_updates is not None:
        results = pending_metrics_updates
    else:
        try:
            results = trainer.train_cycle(
                proposer_batch, solver_batch, relation_credits=relation_credits
            )
        finally:
            if manager:
                _restore_managed_policy_services(manager, adaptive, initially_running)
    TrainingMetricsStore(args.run_dir).append(
        collect_cycle_metrics(
            cycle=max(0, trainer.state.cycle - 1),
            proposer_batch=proposer_batch,
            solver_batch=solver_batch,
            updates=results,
            frontier_scores=_load_frontier_scores(args.run_dir / "frontier_scores.json"),
            relation_credits=relation_credits,
            mace_path=None,
            context={
                "config": str(args.config.resolve()),
                "run_dir": str(args.run_dir.resolve()),
                "verifier": adaptive.verifier,
                "model_selection_policy": "director_set_model_v1",
                "proposer_device": proposer_device,
                "solver_device": solver_device,
            },
            proposal_extraction=_load_optional_json(args.run_dir / "proposal_extraction.json"),
        )
    )
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


def _pending_cycle_metrics_updates(
    state: dict[str, Any], latest_metrics_path: Path
) -> tuple[PolicyUpdateResult, PolicyUpdateResult] | None:
    """Recover telemetry after a committed update without replaying the batch."""
    current_cycle = int(state.get("cycle", 0)) - 1
    tagged = {
        row["role"]: row
        for row in state.get("history", ())
        if row.get("cycle") == current_cycle and row.get("role") in {"proposer", "solver"}
    }
    if state.get("phase") == "proposer" and set(tagged) == {"proposer", "solver"}:
        latest = _load_optional_json(latest_metrics_path) or {}
        if int(latest.get("cycle", -1)) >= current_cycle:
            return None
        return tuple(
            (
                PolicyUpdateResult(
                    **{**tagged[role], "step_metrics": tuple(tagged[role].get("step_metrics", ()))}
                )
                for role in ("proposer", "solver")
            )
        )
    if (
        state.get("phase") != "proposer"
        or int(state.get("cycle", 0)) <= 0
        or int(state.get("solver_step", 0)) <= 0
    ):
        return None
    latest_metrics = _load_optional_json(latest_metrics_path) or {}
    recorded_solver_step = int(latest_metrics.get("policies", {}).get("solver", {}).get("step", 0))
    if recorded_solver_step >= int(state["solver_step"]):
        return None
    by_role: dict[str, PolicyUpdateResult] = {}
    expected_steps = {
        "proposer": int(state.get("proposer_step", 0)),
        "solver": int(state.get("solver_step", 0)),
    }
    for payload in reversed(list(state.get("history", []))):
        role = str(payload.get("role", ""))
        if role not in expected_steps or role in by_role:
            continue
        if payload.get("cycle") is not None:
            if int(payload["cycle"]) != int(state["cycle"]) - 1:
                continue
        elif int(payload.get("step", -1)) != expected_steps[role]:
            continue
        normalized = dict(payload)
        normalized["step_metrics"] = tuple(normalized.get("step_metrics", ()))
        by_role[role] = PolicyUpdateResult(**normalized)
    if set(by_role) != {"proposer", "solver"}:
        raise RuntimeError(
            "training state contains a committed cycle without a complete proposer/solver history; refusing both telemetry recovery and duplicate optimization"
        )
    return (by_role["proposer"], by_role["solver"])


def benchmark(args: argparse.Namespace) -> int:
    config = replace(
        load_adaptive_config(args.config), verifier=args.verifier, persist_runtime_updates=False
    )
    route_latency_tracker = RouteLatencyTracker(window_size=config.canvas.worker_latency_window)

    def application_factory(seed: int):
        return create_adaptive_application(
            replace(config, seed=seed), mock=args.mock, route_latency_tracker=route_latency_tracker
        )

    records = BenchmarkRunner(
        application_factory, checkpoint=str(config.solver_model.checkpoint_path)
    ).run(load_fixed_jsonl(args.dataset), seeds=args.seed or [0])
    baseline = load_flowsteer_records(args.flowsteer_baseline) if args.flowsteer_baseline else None
    payload = write_benchmark(args.output, records, baseline=baseline)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def model_services(args: argparse.Namespace) -> int:
    config = load_adaptive_config(args.config)
    manager = VLLMServiceManager(args.state_dir)
    specs = _service_specs(config)
    if args.role != "all" and args.role not in specs:
        raise ValueError(
            f"{args.role} is externally managed; model-services only controls local vLLM services"
        )
    roles = list(specs) if args.role == "all" else [args.role]
    payload: list[dict[str, object]] = []
    for role in roles:
        spec = specs[role]
        if args.action == "start":
            status = manager.start(spec, wait_s=args.wait_s)
        elif args.action == "stop":
            manager.stop(role, wait_s=args.wait_s)
            status = manager.status(spec)
        elif args.action == "refresh":
            if args.checkpoint is None or len(roles) != 1:
                raise ValueError("refresh requires one --role and --checkpoint")
            status = manager.refresh(spec, args.checkpoint, wait_s=args.wait_s)
        else:
            status = manager.status(spec)
        payload.append(asdict(status))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def prepare_ads_pool_command(args: argparse.Namespace) -> int:
    manifest = prepare_ads_pool(
        args.input,
        ADSPreprocessingConfig(
            model_path=args.model_path,
            output_path=args.output,
            artifacts_dir=args.artifacts_dir,
            num_clusters=args.num_clusters,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pca_dim=args.pca_dim,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _service_specs(config) -> dict[str, ModelServiceSpec]:
    specs = {
        "proposer": ModelServiceSpec(
            "proposer",
            urlsplit(config.proposer_model.base_url).port or 8001,
            config.proposer_model.served_model,
            config.proposer_model.base_model_path,
            _latest_adapter(config.proposer_model.checkpoint_path),
            gpu_memory_utilization=config.proposer_service_gpu_memory_utilization,
            extra_args=qwen_vllm_server_args(
                config.proposer_model.served_model, native_tools=False
            ),
            extra_env=qwen_vllm_server_env(config.proposer_model.served_model),
            gpu_ids=(config.proposer_gpu_id,),
        ),
        "solver": ModelServiceSpec(
            "solver",
            urlsplit(config.solver_model.base_url).port or 8002,
            config.solver_model.served_model,
            config.solver_model.base_model_path,
            _latest_adapter(config.solver_model.checkpoint_path),
            gpu_memory_utilization=config.solver_service_gpu_memory_utilization,
            extra_args=qwen_vllm_server_args(config.solver_model.served_model, native_tools=False),
            extra_env=qwen_vllm_server_env(config.solver_model.served_model),
            gpu_ids=(config.solver_gpu_id,),
        ),
    }
    if config.runtime.managed_locally:
        if config.runtime.model_path is None:
            raise ValueError("locally managed runtime requires model_path")
        specs["runtime"] = ModelServiceSpec(
            "runtime",
            8003,
            config.runtime.served_model,
            config.runtime.model_path,
            extra_args=qwen_vllm_server_args(config.runtime.served_model),
            extra_env=qwen_vllm_server_env(config.runtime.served_model),
            gpu_ids=(config.runtime_gpu_id,),
        )
    return specs


def _async_rollout_service(
    config,
    *,
    gpu_id: int,
    port: int,
    gpu_memory_utilization: float,
    proposer_snapshot: str | None = None,
    solver_snapshot: str | None = None,
) -> tuple[ModelServiceSpec, Any]:
    """Serve the two frozen behavior adapters from one independent base."""
    if (
        config.proposer_model.base_model_path.resolve()
        != config.solver_model.base_model_path.resolve()
    ):
        raise ValueError("async shared rollout service requires one common policy base model")
    proposer_alias = "spgfs-proposer-behavior"
    solver_alias = "spgfs-solver-behavior"

    def adapter_for(snapshot: str | None, checkpoint_root: Path, base: Path) -> Path | None:
        if snapshot is None:
            return _latest_adapter(checkpoint_root)
        path = Path(snapshot).resolve()
        if path == base.resolve():
            return None
        if not path.is_dir():
            raise ValueError(f"async behavior checkpoint does not exist: {path}")
        return path

    proposer_adapter = adapter_for(
        proposer_snapshot,
        config.proposer_model.checkpoint_path,
        config.proposer_model.base_model_path,
    )
    solver_adapter = adapter_for(
        solver_snapshot, config.solver_model.checkpoint_path, config.solver_model.base_model_path
    )
    modules = tuple(
        (
            (alias, adapter)
            for (alias, adapter) in (
                (proposer_alias, proposer_adapter),
                (solver_alias, solver_adapter),
            )
            if adapter is not None
        )
    )
    base_aliases = tuple(
        (
            alias
            for (alias, adapter) in (
                (proposer_alias, proposer_adapter),
                (solver_alias, solver_adapter),
            )
            if adapter is None
        )
    ) or ("spgfs-async-policy-base",)
    base_url = f"http://127.0.0.1:{port}/v1"
    rollout_config = replace(
        config,
        proposer_model=replace(
            config.proposer_model, base_url=base_url, served_model=proposer_alias
        ),
        solver_model=replace(config.solver_model, base_url=base_url, served_model=solver_alias),
        allocated_gpu_ids=tuple(dict.fromkeys((*config.allocated_gpu_ids, gpu_id))),
    )
    rollout_config.validate()
    spec = ModelServiceSpec(
        role="async_rollout",
        port=port,
        served_model="spgfs-async-policy-base",
        base_model_path=config.solver_model.base_model_path,
        served_model_aliases=base_aliases,
        lora_modules=modules,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_args=qwen_vllm_server_args(config.solver_model.served_model, native_tools=False),
        extra_env=qwen_vllm_server_env(config.solver_model.served_model),
        gpu_ids=(gpu_id,),
    )
    return (spec, rollout_config)


def _latest_adapter(root: Path) -> Path | None:
    latest = root / "latest.json"
    if not latest.exists():
        return None
    path = Path(json.loads(latest.read_text(encoding="utf-8"))["path"])
    return path if path.exists() else None


def _training_devices(config, override: str | None) -> tuple[str, str]:
    if override and override != "cuda":
        return (override, override)
    return (
        training_device_for_gpu(config.proposer_gpu_id),
        training_device_for_gpu(config.solver_gpu_id),
    )


def _restore_managed_policy_services(
    manager: VLLMServiceManager, config, enabled: dict[str, bool]
) -> None:
    specs = _service_specs(config)
    for role in ("proposer", "solver"):
        if enabled.get(role) and (not manager.status(specs[role]).running):
            manager.start(specs[role])


def _validate_post_update_state(
    cycle_dir: Path,
    updates: tuple[PolicyUpdateResult, ...],
    training_config: AlternatingTrainingConfig,
    *,
    manager: VLLMServiceManager | None,
    adaptive_config: Any,
) -> dict[str, Any]:
    """Cheap committed-state validation; performs no model forward pass."""
    started = time.monotonic()
    policies = {"proposer": training_config.proposer, "solver": training_config.solver}
    rows: list[dict[str, Any]] = []
    for update in updates:
        numeric = (update.loss, update.policy_loss, update.kl, update.grad_norm)
        if update.status == "skipped":
            if (
                update.optimizer_steps != 0
                or update.checkpoint
                or update.masked_tokens != 0
                or (not update.skip_reason)
                or (not all((math.isfinite(float(x)) for x in numeric)))
            ):
                raise RuntimeError(f"invalid skipped {update.role} update record")
            rows.append(
                {
                    "role": update.role,
                    "status": "skipped",
                    "skip_reason": update.skip_reason,
                    "optimizer_steps": 0,
                    "checkpoint": "",
                    "latest_pointer": None,
                    "parameter_update_l2": 0.0,
                }
            )
            continue
        if update.optimizer_steps <= 0 or not all((math.isfinite(float(x)) for x in numeric)):
            raise RuntimeError(f"invalid committed {update.role} update metrics")
        checkpoint = Path(update.checkpoint)
        if not checkpoint.exists():
            raise RuntimeError(f"committed {update.role} checkpoint is missing")
        latest_path = policies[update.role].checkpoint_root / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if Path(latest["path"]).resolve() != checkpoint.resolve():
            raise RuntimeError(f"{update.role} latest pointer does not match committed checkpoint")
        deltas = [
            float(row["parameter_update_l2"])
            for row in update.step_metrics
            if row.get("parameter_update_l2") is not None
        ]
        if deltas and (not any((value > 0 and math.isfinite(value) for value in deltas))):
            raise RuntimeError(f"{update.role} optimizer reported no finite parameter change")
        rows.append(
            {
                "role": update.role,
                "checkpoint": str(checkpoint),
                "step": update.step,
                "optimizer_steps": update.optimizer_steps,
                "masked_tokens": update.masked_tokens,
                "parameter_update_l2": deltas,
            }
        )
    services: dict[str, Any] = {}
    if manager is not None:
        specs = _service_specs(adaptive_config)
        for role in ("proposer", "solver"):
            status = manager.status(specs[role])
            services[role] = asdict(status)
            if not status.running:
                raise RuntimeError(f"{role} service was not restored after update")
    validation_scope = [
        "finite_update_metrics",
        "checkpoint_exists",
        "latest_pointer_matches",
        "parameter_delta_when_reported",
    ]
    if manager is not None:
        validation_scope.append("managed_service_running")
    payload = {
        "schema_version": "post_update_validation_v1",
        "validation_scope": validation_scope,
        "model_forward_executed": False,
        "updates": rows,
        "services": services,
        "duration_s": time.monotonic() - started,
    }
    (cycle_dir / "post_update_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


_DEFAULT_MINIMUM_QUALIFIED_ROUTES = 5


def _can_resume_isolated_judge_failure(cycle_dir: Path) -> bool:
    """Reopen only missing slots after the former Judge-wide circuit bug."""
    from .selfplay_runtime import _is_healthbench_judge_backend_failure

    errors = _read_jsonl(cycle_dir / "rollout_errors.jsonl")
    judge_failures = [
        row
        for row in errors
        if _is_healthbench_judge_backend_failure(row.get("backend_failure", {}))
    ]
    return bool(judge_failures) and all(
        (
            row.get("error_type") == "BackendCircuitOpenError"
            or _is_healthbench_judge_backend_failure(row.get("backend_failure", {}))
            for row in errors
        )
    )


def _apply_fresh_route_report(config, args: argparse.Namespace):
    """Restrict a real experiment to a fresh qualified route subset."""
    if args.route_report is None:
        if not args.mock:
            raise ValueError("real selfplay-experiment requires --route-report")
        return (config, None)
    path = args.route_report.resolve()
    age_s = time.time() - path.stat().st_mtime
    if age_s < 0 or age_s > args.max_route_report_age_s:
        raise ValueError(f"route qualification is stale: age_s={age_s:.1f}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = frozenset((str(value) for value in payload.get("routes_requested", [])))
    required = set(config.worker_runtime_routes) | {
        member for members in config.runtime_endpoint_pools.values() for member in members
    }
    if config.healthbench_judge_runtime_route:
        required.add(config.healthbench_judge_runtime_route)
    if not required <= requested:
        raise ValueError("route report must probe all configured route candidates and pool members")
    minimum_selected_routes = int(getattr(args, "minimum_selected_routes", 5))
    if minimum_selected_routes not in {1, 4, 5}:
        raise ValueError("minimum selected routes must be one, four or five")
    usable = {str(value) for value in payload.get("usable_routes", [])}
    dedicated_judge = config.healthbench_judge_runtime_route
    if dedicated_judge and dedicated_judge not in usable:
        raise ValueError("dedicated HealthBench Judge route must be freshly qualified")
    qualified = tuple(
        (
            str(value)
            for value in payload.get("usable_routes", [])
            if str(value) in config.worker_runtime_routes
        )
    )
    if len(set(qualified)) < minimum_selected_routes or not set(qualified) <= requested:
        raise ValueError("route report does not contain the requested minimum usable routes")
    selected = tuple(
        dict.fromkeys((value.strip() for value in args.route_subset.split(",") if value.strip()))
    )
    routes = selected or qualified
    if len(set(routes)) < minimum_selected_routes or not set(routes) <= set(qualified):
        raise ValueError("selected routes must be a subset of the freshly qualified routes")
    pool = config.runtime_pool()
    missing = set(routes) - set(pool)
    if missing:
        raise ValueError(f"qualified routes absent from config: {sorted(missing)}")
    primary = routes[0]
    judge_route = dedicated_judge or ("gpt" if "gpt" in qualified else None)
    support_routes = tuple(dict.fromkeys((*routes, *((judge_route,) if judge_route else ()))))
    active_pools = {
        key: members
        for (key, members) in config.runtime_endpoint_pools.items()
        if key in support_routes
    }
    pool_members = {member for members in active_pools.values() for member in members}
    if not pool_members <= usable:
        raise ValueError("all active endpoint pool members must be freshly qualified")
    support_routes = tuple(dict.fromkeys((*support_routes, *sorted(pool_members))))
    restricted = replace(
        config,
        runtime=pool[primary],
        runtime_name=primary,
        additional_runtimes={route: pool[route] for route in support_routes if route != primary},
        worker_runtime_routes=routes,
        runtime_endpoint_pools=active_pools,
        support_runtime="gpt" if "gpt" in support_routes else primary,
    )
    restricted.validate()
    return (
        restricted,
        {
            "path": str(path),
            "age_s_at_start": age_s,
            "qualified_routes": list(qualified),
            "selected_routes": list(routes),
            "minimum_selected_routes": minimum_selected_routes,
            "degraded_route_mode": minimum_selected_routes < _DEFAULT_MINIMUM_QUALIFIED_ROUTES,
            "support_routes": list(support_routes),
            "healthbench_judge_route": judge_route,
            "deepseek_excluded": "deepseek" not in routes,
        },
    )


def _validate_rate_limit_repair_resume(cycle_dir: Path, rollouts_per_task: int) -> bool:
    """Reopen only a documented set of missing 429 slots after a route repair."""
    marker = cycle_dir / "rate_limit_repair.json"
    if not marker.exists():
        return False
    repair = json.loads(marker.read_text())
    affected = set(repair.get("affected_rollout_ids", []))
    persisted = {
        str(r.get("rollout_id", "")) for r in _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
    }
    tasks = {
        str((r.get("task") or {}).get("task_id", ""))
        for r in _read_jsonl(cycle_dir / "tasks.jsonl")
    } - {""}
    expected = {f"{task}-r{i}" for task in tasks for i in range(rollouts_per_task)}
    limited = {
        str(r.get("rollout_id", ""))
        for r in _read_jsonl(cycle_dir / "backend_api_events.jsonl")
        if r.get("event") == "backend_request_failure" and r.get("status_code") == 429
    }
    if (
        repair.get("status") != "prepared"
        or not affected
        or (not tasks)
        or (not affected <= limited)
        or (not affected <= expected)
        or (not expected - persisted <= affected)
        or (not set(repair.get("preserved_rollout_ids", [])) <= persisted)
    ):
        raise ValueError("rate-limit repair does not match preserved failed/successful slots")
    return True


def _validate_transient_backend_circuit_resume(cycle_dir: Path) -> None:
    """Fail closed before exposing the internal exact-ID resume switch."""
    quarantine_path = cycle_dir / "quarantined_groups.jsonl"
    errors_path = cycle_dir / "rollout_errors.jsonl"
    if not quarantine_path.is_file() or not errors_path.is_file():
        raise ValueError("transient backend-circuit resume requires preserved quarantine/errors")
    quarantines = _read_jsonl(quarantine_path)
    latest: dict[str, dict[str, Any]] = {}
    for row in quarantines:
        task_id = str(row.get("task_id", ""))
        if task_id:
            latest[task_id] = row
    rollout_ids = {
        str(row.get("rollout_id", "")) for row in _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
    } - {""}
    pending_backend = []
    invalid_quarantines = []
    for task_id, row in latest.items():
        status = row.get("status")
        reason = row.get("reason")
        if (
            status == "quarantined"
            and reason == "backend_circuit_open"
            and (not row.get("non_trainable_rollout_ids"))
        ):
            pending_backend.append(task_id)
            continue
        if (
            status == "quarantined"
            and reason == "non_trainable_rollout_group"
            and row.get("non_trainable_rollout_ids")
        ):
            continue
        if (
            status == "reopened_for_exact_resume"
            and row.get("prior_quarantine_reason") == "backend_circuit_open"
        ):
            missing = {str(value) for value in row.get("missing_rollout_ids", [])}
            if missing and missing <= rollout_ids:
                continue
        invalid_quarantines.append(task_id)
    if not pending_backend or invalid_quarantines:
        raise ValueError(
            f"resume gate requires at least one pending backend-circuit quarantine and permits only preserved non-trainable groups or fully recovered historical backend groups; invalid={sorted(invalid_quarantines)}"
        )
    allowed_backend_errors = {
        "BackendCircuitOpenError",
        "WorkerBackendUnavailableError",
        "BackendRetryExhaustedError",
    }
    errors = _read_jsonl(errors_path)
    pending_task_ids = set(pending_backend)
    invalid_errors = [
        row
        for row in errors
        if str(row.get("error_type", "")) not in allowed_backend_errors
        and str(row.get("task_id", "")) not in pending_task_ids
    ]
    error_types = {str(row.get("error_type", "")) for row in errors}
    if not errors or "BackendCircuitOpenError" not in error_types or invalid_errors:
        raise ValueError(
            "resume gate accepts backend errors plus typed errors belonging to groups quarantined by the same backend circuit"
        )


def _validate_infrastructure_repair_resume(cycle_dir: Path) -> None:
    """Permit exact IDs only for an explicitly attested fresh recollection."""
    repair_path = cycle_dir / "collection_repair_attestation.json"
    if not repair_path.is_file():
        raise ValueError("infrastructure-repair resume requires collection_repair_attestation.json")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    recollection_mode = repair.get("recollection_mode")
    valid_full_cycle = bool(
        recollection_mode == "full_cycle" and repair.get("source_cycle_archive")
    )
    valid_exact_missing = bool(
        recollection_mode == "exact_missing_rollout_ids"
        and repair.get("affected_rollout_ids")
        and repair.get("repaired_instance_ids")
        and repair.get("repair_evidence")
        and (repair.get("persisted_rollouts_verified_unaffected") is True)
    )
    if (
        repair.get("status") != "approved"
        or repair.get("incident_class") != "infrastructure"
        or (not repair.get("source_tasks_sha256"))
        or (not (valid_full_cycle or valid_exact_missing))
    ):
        raise ValueError("invalid infrastructure-repair recollection attestation")


def _validate_attribution_classifier_repair_resume(cycle_dir: Path) -> None:
    """Permit one missing slot only after a concrete post-terminal classifier fix."""
    repair_path = cycle_dir / "collection_repair_attestation.json"
    if not repair_path.is_file():
        raise ValueError("classifier-repair resume requires collection_repair_attestation.json")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    affected = {str(value) for value in repair.get("reclassified_rollout_ids", ())}
    persisted = {
        str(row.get("rollout_id", "")) for row in _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
    } - {""}
    task_ids = {
        str((row.get("task") or {}).get("task_id", ""))
        for row in _read_jsonl(cycle_dir / "tasks.jsonl")
    } - {""}
    affected_tasks = {rollout_id.rsplit("-r", 1)[0] for rollout_id in affected}
    if (
        repair.get("status") != "approved"
        or repair.get("incident_class") != "attribution_uncertain_recovery_exhausted"
        or repair.get("recollection_mode") != "exact_reclassified_rollout_ids"
        or (not repair.get("source_tasks_sha256"))
        or (not repair.get("repair_evidence"))
        or (repair.get("persisted_rollouts_verified_unaffected") is not True)
        or (not affected)
        or (len(affected) != 1)
        or (affected & persisted)
        or (affected_tasks - task_ids)
    ):
        raise ValueError("invalid attribution-classifier repair attestation")


def _validate_uncertain_group_recollection_resume(
    cycle_dir: Path, *, rollouts_per_task: int
) -> dict[str, int]:
    """Require a fresh group, or verified partial progress from its reopened epoch."""
    repair_path = cycle_dir / "collection_repair_attestation.json"
    if not repair_path.is_file():
        raise ValueError("uncertain-group resume requires collection_repair_attestation.json")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    affected_tasks = {str(value) for value in repair.get("affected_task_ids", ())}
    affected_rollouts = {str(value) for value in repair.get("affected_rollout_ids", ())}
    expected_rollouts = {
        f"{task_id}-r{rollout_index}"
        for task_id in affected_tasks
        for rollout_index in range(rollouts_per_task)
    }
    persisted_rollouts = {
        str(row.get("rollout_id", "")) for row in _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
    } - {""}
    task_ids = {
        str((row.get("task") or {}).get("task_id", ""))
        for row in _read_jsonl(cycle_dir / "tasks.jsonl")
    } - {""}
    raw_offsets = repair.get("policy_sampling_attempt_offsets", {})
    policy_sampling_attempt_offsets = (
        {str(task_id): offset for (task_id, offset) in raw_offsets.items()}
        if isinstance(raw_offsets, dict)
        else {}
    )
    valid_offsets = bool(
        set(policy_sampling_attempt_offsets) == affected_tasks
        and all(
            (
                isinstance(offset, int) and (not isinstance(offset, bool)) and (offset > 0)
                for offset in policy_sampling_attempt_offsets.values()
            )
        )
    )
    incidents = _read_jsonl(cycle_dir / "collection_incidents.jsonl")
    reopened_epoch = bool(
        valid_offsets
        and incidents
        and (incidents[-1].get("event") == "collection_abort_reopened_for_full_group_recollection")
        and (incidents[-1].get("task_id") in affected_tasks)
    )
    current_epoch_rows = all(
        (
            row.get("task_id") in affected_tasks
            and row.get("metadata", {}).get("policy_sampling_attempt_offset")
            == policy_sampling_attempt_offsets.get(row.get("task_id"))
            for row in _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
            if str(row.get("rollout_id", "")) in affected_rollouts
        )
    )
    if (
        repair.get("status") != "approved"
        or repair.get("incident_class") != "attribution_uncertain_recovery_exhausted"
        or repair.get("recollection_mode") != "exact_affected_task_groups"
        or (repair.get("group_recollection_scope") != "all_siblings")
        or (not repair.get("source_tasks_sha256"))
        or (not repair.get("repair_evidence"))
        or (repair.get("persisted_rollouts_verified_unaffected") is not True)
        or (not affected_tasks)
        or (affected_tasks - task_ids)
        or (affected_rollouts != expected_rollouts)
        or (
            affected_rollouts & persisted_rollouts and (not (reopened_epoch and current_epoch_rows))
        )
        or (not valid_offsets)
    ):
        raise ValueError("invalid uncertain-group recollection attestation")
    return policy_sampling_attempt_offsets


def _validate_proposal_selection_repair_resume(cycle_dir: Path) -> None:
    """Allow exact resume only for fixed-pool task IDs missing after a typed proposal bug."""
    repair_path = cycle_dir / "collection_repair_attestation.json"
    if not repair_path.is_file():
        raise ValueError("proposal-selection repair requires collection_repair_attestation.json")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    affected_tasks = {str(value) for value in repair.get("affected_task_ids", ())}
    persisted_tasks = {
        str((row.get("task") or {}).get("task_id", ""))
        for row in _read_jsonl(cycle_dir / "tasks.jsonl")
    } - {""}
    persisted_rollout_tasks = {
        str(row.get("task_id", "")) for row in _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
    } - {""}
    failed_proposals = {
        str(row.get("task_id", ""))
        for row in _read_jsonl(cycle_dir / "proposal_attempts.jsonl")
        if row.get("success") is False and row.get("failure_kind") == "format_or_validation_failure"
    }
    if (
        repair.get("status") != "approved"
        or repair.get("incident_class") != "proposal_selection_failure"
        or repair.get("recollection_mode") != "exact_missing_proposal_and_rollouts"
        or (not repair.get("source_tasks_sha256"))
        or (not repair.get("repair_evidence"))
        or (repair.get("persisted_rollouts_verified_unaffected") is not True)
        or (not affected_tasks)
        or (not affected_tasks <= failed_proposals)
        or (affected_tasks & persisted_tasks)
        or (affected_tasks & persisted_rollout_tasks)
    ):
        raise ValueError("invalid proposal-selection repair attestation")


def selfplay_experiment(args: argparse.Namespace) -> int:
    from .wandb_tracking import WandbTracker

    tracker = WandbTracker(args.output, getattr(args, "wandb_mode", "disabled"))
    failed = True
    try:
        result = _selfplay_experiment(args, tracker)
        failed = False
        return result
    except BaseException:
        try:
            from .outcome_metrics import record_interrupted_collection

            partial = record_interrupted_collection(
                args.output, getattr(args, "rollouts", None) or 5
            )
            tracker.log_interruption(partial)
        except Exception as telemetry_error:
            import warnings

            warnings.warn(
                f"Interruption telemetry failed ({type(telemetry_error).__name__}); original error preserved",
                stacklevel=2,
            )
        raise
    finally:
        tracker.finish(failed=failed)


def _cycle_collection_only(args: argparse.Namespace, cycle: int) -> bool:
    return bool(
        getattr(args, "collection_only", False)
        or (getattr(args, "final_cycle_collection_only", False) and cycle == args.cycles - 1)
    )


def _selfplay_experiment(args: argparse.Namespace, tracker) -> int:
    if getattr(args, "final_cycle_collection_only", False) and args.final_cycle_evaluation_only:
        raise ValueError("choose final collection-only or frozen evaluation-only, not both")
    if not math.isfinite(args.cycle_target_time_s) or args.cycle_target_time_s < 0:
        raise ValueError("cycle target time must be finite and non-negative")
    if args.cycles <= 0:
        raise ValueError("cycles must be positive")
    if args.async_next_cycle_rollouts:
        if args.cycles < 2:
            raise ValueError("async next-cycle rollouts require at least two cycles")
        if not args.manage_services:
            raise ValueError("async next-cycle rollouts require --manage-services")
        if args.async_rollout_gpu_id is None or args.async_rollout_gpu_id < 0:
            raise ValueError("async next-cycle rollouts require a non-negative rollout GPU")
        if not 0.0 < args.async_rollout_gpu_memory_utilization <= 1.0:
            raise ValueError("async rollout GPU memory utilization must be in (0, 1]")
        if not 1 <= args.async_rollout_port <= 65535:
            raise ValueError("async rollout port must be in [1, 65535]")
        if args.collection_only:
            raise ValueError("async next-cycle rollouts require actual policy updates")
    if (
        args.continue_on_uncertain_attribution_exhausted
        and (not args.allow_skipped_task_groups)
        and (args.rollout_group_policy == "complete")
    ):
        raise ValueError(
            "--continue-on-uncertain-attribution-exhausted requires --allow-skipped-task-groups"
        )
    if args.resume_planned_interruption and (not args.resume):
        raise ValueError("--resume-planned-interruption requires --resume")
    if args.resume_transient_backend_circuit and (not args.resume):
        raise ValueError("--resume-transient-backend-circuit requires --resume")
    if args.resume_after_attribution_classifier_repair and (not args.resume):
        raise ValueError("--resume-after-attribution-classifier-repair requires --resume")
    if args.resume_after_infrastructure_repair and (not args.resume):
        raise ValueError("--resume-after-infrastructure-repair requires --resume")
    if args.resume_after_uncertain_group_recollection and (not args.resume):
        raise ValueError("--resume-after-uncertain-group-recollection requires --resume")
    if args.resume_after_proposal_selection_repair and (not args.resume):
        raise ValueError("--resume-after-proposal-selection-repair requires --resume")
    if args.resume_transient_backend_circuit and (
        args.resume_after_attribution_classifier_repair or args.resume_after_infrastructure_repair
    ):
        raise ValueError("choose only one exact-resume incident policy per self-play experiment")
    selected_repair_policies = sum(
        (
            int(value)
            for value in (
                args.resume_transient_backend_circuit,
                args.resume_after_attribution_classifier_repair,
                args.resume_after_infrastructure_repair,
                args.resume_after_uncertain_group_recollection,
                args.resume_after_proposal_selection_repair,
            )
        )
    )
    if selected_repair_policies > 1:
        raise ValueError("choose only one exact-resume repair policy per self-play experiment")
    if args.frozen_pool_selection and (not args.task_pool):
        raise ValueError("--frozen-pool-selection requires --task-pool")
    if args.frozen_pool_selection and (not args.mock_trainer):
        raise ValueError("--frozen-pool-selection is evaluation-only and requires --mock-trainer")
    curriculum_profile = (
        CurriculumProfile.from_toml(args.curriculum_profile) if args.task_pool else None
    )
    tasks_per_cycle = args.tasks_per_cycle or (
        curriculum_profile.tasks_per_cycle if curriculum_profile else 128
    )
    rollouts_per_task = args.rollouts or (
        curriculum_profile.rollouts_per_task if curriculum_profile else 5
    )
    rollout_workers = args.workers or (
        curriculum_profile.rollout_workers if curriculum_profile else 1
    )
    task_window = args.task_window or (curriculum_profile.task_window if curriculum_profile else 1)
    config = replace(
        load_adaptive_config(args.config), verifier=args.verifier, persist_runtime_updates=False
    )
    if args.enable_swe:
        config = replace(config, swe=replace(config.swe, enabled=True))
    if args.policy_gpu_id is not None and (
        args.proposer_gpu_id is not None or args.solver_gpu_id is not None
    ):
        raise ValueError(
            "--policy-gpu-id cannot be combined with --proposer-gpu-id/--solver-gpu-id"
        )
    if args.policy_gpu_id is not None:
        config = replace(
            config,
            allocated_gpu_ids=(args.policy_gpu_id,),
            proposer_gpu_id=args.policy_gpu_id,
            solver_gpu_id=args.policy_gpu_id,
            runtime_gpu_id=args.policy_gpu_id,
        )
    elif args.proposer_gpu_id is not None or args.solver_gpu_id is not None:
        proposer_gpu_id = (
            args.proposer_gpu_id if args.proposer_gpu_id is not None else config.proposer_gpu_id
        )
        solver_gpu_id = (
            args.solver_gpu_id if args.solver_gpu_id is not None else config.solver_gpu_id
        )
        assigned = [proposer_gpu_id, solver_gpu_id]
        if config.runtime.managed_locally:
            assigned.append(config.runtime_gpu_id)
        config = replace(
            config,
            allocated_gpu_ids=tuple(dict.fromkeys(assigned)),
            proposer_gpu_id=proposer_gpu_id,
            solver_gpu_id=solver_gpu_id,
        )
    if args.async_next_cycle_rollouts:
        occupied_policy_gpus = {config.proposer_gpu_id, config.solver_gpu_id}
        if config.runtime.managed_locally:
            occupied_policy_gpus.add(config.runtime_gpu_id)
        if args.async_rollout_gpu_id in occupied_policy_gpus:
            raise ValueError(
                "async rollout GPU must be distinct from both training GPUs and local runtime GPU"
            )
    if args.director_prompt_variant is not None:
        config = replace(config, director_prompt_variant=args.director_prompt_variant)
    config.validate()
    if not set(args.solver_data_parallel_gpus or ()) <= set(config.allocated_gpu_ids):
        raise ValueError("Solver data-parallel GPUs must be explicitly allocated in config")
    (config, route_qualification) = _apply_fresh_route_report(config, args)
    checkpoint_root = (args.checkpoint_root or args.output / "checkpoints").resolve()
    runtime_state_root = (args.runtime_state_root or args.output / "runtime_state").resolve()
    config = replace(
        config,
        persist_runtime_updates=False,
        proposer_model=replace(config.proposer_model, checkpoint_path=checkpoint_root / "proposer"),
        solver_model=replace(config.solver_model, checkpoint_path=checkpoint_root / "solver"),
        trace_path=runtime_state_root / "traces.jsonl",
        route_health_path=runtime_state_root / "route_health.json",
        healthbench_judge_audit_path=args.output.resolve() / "private" / "healthbench_judge",
        swe=replace(
            config.swe,
            workspace_root=args.output.resolve() / "workspaces",
            artifact_store_root=args.output.resolve() / "private" / "artifacts",
            lifecycle_log_path=args.output.resolve() / "private" / "lifecycle.jsonl",
            verifier_log_path=args.output.resolve() / "private" / "verifier-client.jsonl",
        ),
    )
    if not args.mock and config.verifier == "none":
        raise ValueError(
            "real self-play requires a task verifier; choose exact_match or another mode"
        )
    fixed_components = None
    frozen_proposer = None
    if args.task_pool:
        fixed_components = _fixed_pool_components(
            args.task_pool,
            config=config,
            mock=args.mock,
            seed=config.seed,
            profile=curriculum_profile,
        )
        if args.frozen_pool_selection:
            (seeds, frozen_proposer) = _load_frozen_pool_selection(
                args.frozen_pool_selection, fixed_components[0]
            )
            if len(seeds) != tasks_per_cycle:
                raise ValueError(
                    f"frozen selection row count must equal tasks_per_cycle: {len(seeds)} != {tasks_per_cycle}"
                )
        else:
            seeds = fixed_components[0].balanced_selection_seeds(tasks_per_cycle)
    elif args.seed_data:
        seeds = _load_seeds(args.seed_data)
    else:
        seeds = []
    if not seeds:
        raise ValueError("seed dataset is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    (proposer_device, solver_device) = _training_devices(config, args.device)
    effective_gradient_accumulation = math.ceil(args.mini_batch_size / args.micro_batch_size)
    if (
        args.gradient_accumulation_steps is not None
        and args.gradient_accumulation_steps != effective_gradient_accumulation
    ):
        raise ValueError(
            "gradient_accumulation_steps must equal ceil(mini_batch_size / micro_batch_size) for SESA-style mini-batch training"
        )
    args.gradient_accumulation_steps = effective_gradient_accumulation
    collection_only = getattr(args, "collection_only", False)
    training_cycles = (
        0
        if collection_only
        else args.cycles
        - int(
            args.final_cycle_evaluation_only or getattr(args, "final_cycle_collection_only", False)
        )
    )
    nominal_proposer_steps = _nominal_optimizer_steps(
        training_cycles, tasks_per_cycle, args.epochs, args.mini_batch_size
    )
    nominal_solver_steps = _nominal_optimizer_steps(
        training_cycles,
        tasks_per_cycle * rollouts_per_task,
        args.epochs,
        max(1, tasks_per_cycle * rollouts_per_task)
        if args.rollout_group_policy == "eligible_subset"
        else args.mini_batch_size,
    )
    proposer_total_optimizer_steps = (
        args.proposer_total_optimizer_steps or args.total_optimizer_steps or nominal_proposer_steps
    )
    solver_total_optimizer_steps = (
        args.solver_total_optimizer_steps or args.total_optimizer_steps or nominal_solver_steps
    )
    training_config = AlternatingTrainingConfig(
        _policy_training_config(
            config.proposer_model.base_model_path,
            config.proposer_model.checkpoint_path,
            args.proposer_learning_rate,
            proposer_device,
            args,
            total_optimizer_steps=proposer_total_optimizer_steps,
        ),
        _policy_training_config(
            config.solver_model.base_model_path,
            config.solver_model.checkpoint_path,
            args.solver_learning_rate,
            solver_device,
            args,
            total_optimizer_steps=solver_total_optimizer_steps,
        ),
        args.output / "training_state.json",
        parallel_roles=args.parallel_role_training,
    )
    training_config = replace(
        training_config,
        solver=replace(
            training_config.solver,
            dtype=getattr(args, "solver_training_dtype", "bfloat16"),
            data_parallel_gpu_ids=tuple(args.solver_data_parallel_gpus or ()),
        ),
    )
    training_config.validate()
    factory = (
        (lambda role, policy, seed: MockPolicyTrainer(role, policy, seed=seed))
        if args.mock_trainer
        else None
    )
    manager = VLLMServiceManager(args.service_state_dir) if args.manage_services else None
    if manager:
        for spec in _service_specs(config).values():
            manager.start(spec)
    progress_path = args.output / "experiment_progress.json"
    previous_progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if args.resume and progress_path.exists()
        else {}
    )
    summaries: list[dict[str, object]] = list(previous_progress.get("cycles", []))
    completed_cycles = int(previous_progress.get("completed_cycles", 0))
    if completed_cycles > args.cycles:
        raise ValueError("recorded completed cycles exceed requested --cycles")
    if fixed_components is not None and args.resume and completed_cycles:
        previous_state = args.output / f"cycle-{completed_cycles - 1:04d}" / "curriculum_state.json"
        if previous_state.exists():
            fixed_components[1].load_state_dict(
                json.loads(previous_state.read_text(encoding="utf-8"))
            )
    from .outcome_metrics import record_provenance

    provenance_id = record_provenance(args.output, args.config, args.task_pool)
    tracker.start(
        {
            "planned_cycles": args.cycles,
            "tasks_per_cycle": tasks_per_cycle,
            "provenance_id": provenance_id,
            "rollout_workers": rollout_workers,
            "counterfactual_workers": args.counterfactual_workers,
            "max_micro_batch_tokens": args.max_micro_batch_tokens,
            "max_sequence_length": training_config.solver.max_sequence_length,
            "proposer_learning_rate": training_config.proposer.learning_rate,
            "solver_learning_rate": training_config.solver.learning_rate,
            "allocated_gpu_ids": list(
                dict.fromkeys(
                    (
                        *config.allocated_gpu_ids,
                        *((args.async_rollout_gpu_id,) if args.async_next_cycle_rollouts else ()),
                    )
                )
            ),
            "rollouts_per_task": rollouts_per_task,
            "epochs": args.epochs,
            "mini_batch_size": args.mini_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "parallel_role_training": args.parallel_role_training,
            "clip_range": args.clip_range,
            "kl_coefficient": args.kl_coefficient,
            "entropy_coefficient": args.entropy_coefficient,
            "metric_schema": "primary_outcomes_v1",
            "mock": args.mock,
            "mock_trainer": args.mock_trainer,
        }
    )
    if completed_cycles == args.cycles:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0
    prewarmed_cycle_seeds: dict[int, list[Any]] = {}

    def cycle_seeds(cycle: int) -> list[Any]:
        nonlocal seeds
        if fixed_components is not None and frozen_proposer is None:
            seeds = prewarmed_cycle_seeds.pop(cycle, None) or fixed_components[
                0
            ].balanced_selection_seeds(
                tasks_per_cycle,
                offset_per_dataset=cycle
                * math.ceil(tasks_per_cycle / len(fixed_components[1].dataset_cluster_ids)),
            )
        if (
            fixed_components is not None
            and curriculum_profile is not None
            and curriculum_profile.nominal_epoch_cycles
            and (cycle > 0)
            and (cycle % curriculum_profile.nominal_epoch_cycles == 0)
        ):
            fixed_components[1].reset_epoch()
        return list(seeds)

    def collect_cycle(
        cycle: int,
        selected_seeds: list[Any],
        *,
        collection_config: Any,
        behavior_update_index: int,
        collection_mode: str,
        snapshots: Any,
        enable_probability_cache: bool,
    ) -> _CollectedExperimentCycle:
        cycle_started_monotonic = time.monotonic()
        evaluation_only = args.final_cycle_evaluation_only and cycle == args.cycles - 1
        cycle_collection_only = _cycle_collection_only(args, cycle)
        cycle_dir = args.output / f"cycle-{cycle:04d}"
        resumed_partial_cycle = bool(args.resume and (cycle_dir / "solver_rollouts.jsonl").exists())
        policy_sampling_attempt_offsets: dict[str, int] = {}
        rate_limit_repair_resume = bool(
            args.resume and _validate_rate_limit_repair_resume(cycle_dir, rollouts_per_task)
        )
        if args.resume_transient_backend_circuit and cycle_dir.exists():
            _validate_transient_backend_circuit_resume(cycle_dir)
        if (
            args.resume_after_infrastructure_repair
            and cycle_dir.exists()
            and (
                not (
                    (cycle_dir / "solver_batch.json").exists()
                    and (cycle_dir / "proposer_batch.json").exists()
                )
            )
        ):
            _validate_infrastructure_repair_resume(cycle_dir)
        if args.resume_after_attribution_classifier_repair and cycle_dir.exists():
            _validate_attribution_classifier_repair_resume(cycle_dir)
        if args.resume_after_uncertain_group_recollection and cycle_dir.exists():
            policy_sampling_attempt_offsets = _validate_uncertain_group_recollection_resume(
                cycle_dir, rollouts_per_task=rollouts_per_task
            )
        if args.resume_after_proposal_selection_repair and cycle_dir.exists():
            _validate_proposal_selection_repair_resume(cycle_dir)
        previous_collapse_streak = 0
        if cycle > 0:
            previous_collapse_events = _read_jsonl(
                args.output / f"cycle-{cycle - 1:04d}" / "collapse_monitor.jsonl"
            )
            if previous_collapse_events and previous_collapse_events[-1].get("alert"):
                previous_collapse_streak = int(
                    previous_collapse_events[-1].get("consecutive_alert_windows", 0)
                )
        tokenizer = (
            ByteTokenizer()
            if args.mock
            else HuggingFaceTokenizer(collection_config.proposer_model.base_model_path)
        )
        graph_feature_extractor = (
            None
            if args.mock
            else SemanticGraphFeatureExtractor(
                E5DelegationEncoder(
                    collection_config.graph_embedding_model_path or "intfloat/e5-base-v2"
                )
            )
        )
        if frozen_proposer is not None:
            proposer = frozen_proposer
        elif fixed_components is not None:
            (pool, scheduler, retriever, backend) = fixed_components
            proposer = create_fixed_pool_proposer(
                collection_config,
                pool,
                scheduler,
                retriever,
                backend=backend,
                tokenizer=tokenizer,
                candidate_count=curriculum_profile.candidate_count,
            )
        else:
            proposer = (
                _MockProposer()
                if args.mock
                else create_qwen_task_proposer(collection_config, tokenizer=tokenizer)
            )
        route_latency_tracker = RouteLatencyTracker(
            window_size=collection_config.canvas.worker_latency_window
        )
        route_token_tracker = RouteTokenTracker(
            window_size=collection_config.canvas.worker_token_window,
            path=args.output / "route_token_usage.json",
        )

        def application_factory(
            seed: int,
            route_latency_tracker: RouteLatencyTracker = route_latency_tracker,
            route_token_tracker: RouteTokenTracker = route_token_tracker,
            tokenizer: Any = tokenizer,
        ):
            return create_adaptive_application(
                replace(collection_config, seed=seed),
                mock=args.mock,
                route_latency_tracker=route_latency_tracker,
                route_token_tracker=route_token_tracker,
                director_tokenizer=tokenizer,
            )

        cycle_training_config = training_config
        probability_observer = None
        if enable_probability_cache and (not (evaluation_only or cycle_collection_only)):
            if manager is None:
                raise ValueError(
                    "async Solver probability cache requires --manage-services so GPU ownership can be transferred safely"
                )
            binding = probability_cache_binding(training_config.solver)
            cache_path = cycle_dir / "solver_probability_cache.jsonl"
            cache_policy = replace(
                training_config.solver,
                device=proposer_device,
                data_parallel_gpu_ids=(),
                probability_cache_path=cache_path,
                probability_cache_binding=binding,
            )
            probability_observer = AsyncSolverProbabilityCache(
                output_path=cache_path,
                config=cache_policy,
                prepare_callback=lambda: manager.stop("proposer"),
            )
            cycle_training_config = replace(
                training_config,
                solver=replace(
                    training_config.solver,
                    probability_cache_path=cache_path,
                    probability_cache_binding=binding,
                ),
            )
        rollout_result = SelfPlayRolloutRunner(
            proposer=proposer,
            application_factory=application_factory,
            tokenizer=tokenizer,
            snapshots=snapshots,
            output_dir=cycle_dir,
            config=SelfPlayRunConfig(
                rollouts_per_task,
                cycle * 100000,
                args.max_sequence_length,
                args.counterfactuals_per_rollout,
                rollout_workers,
                args.proposals_per_seed,
                task_window,
                True,
                task_scheduling_policy=args.task_scheduling_policy,
                primary_job_order=args.primary_job_order,
                max_active_task_groups=args.max_active_task_groups,
                structural_exploration_policy=collection_config.canvas.structural_exploration_policy,
                initial_collapse_alert_streak=previous_collapse_streak,
                rollout_wall_time_s=args.rollout_wall_time_s,
                stateful_rollout_wall_time_s=args.stateful_rollout_wall_time_s,
                swe_rollout_wall_time_s=args.swe_rollout_wall_time_s,
                rollout_slot_wall_time_s=args.rollout_slot_wall_time_s,
                stateful_slot_wall_time_s=args.stateful_slot_wall_time_s,
                swe_slot_wall_time_s=args.swe_slot_wall_time_s,
                swe_non_trainable_recovery_attempts=args.swe_non_trainable_recovery_attempts,
                non_swe_recovery_attempts=args.non_swe_recovery_attempts,
                rollout_no_progress_time_s=args.rollout_no_progress_time_s,
                request_wall_time_s=args.request_wall_time_s,
                replacement_rollouts_per_task=args.replacement_rollouts_per_task,
                allow_exact_rollout_resume=rate_limit_repair_resume
                or (
                    args.resume_planned_interruption
                    and (not (cycle_dir / "primary_outcomes.json").exists())
                )
                or args.resume_transient_backend_circuit
                or (args.resume and _can_resume_isolated_judge_failure(cycle_dir))
                or (
                    args.resume
                    and (cycle_dir / "tasks.jsonl").exists()
                    and (not (cycle_dir / "solver_rollouts.jsonl").exists())
                    and (not (cycle_dir / "rollout_errors.jsonl").exists())
                )
                or args.resume_after_attribution_classifier_repair
                or args.resume_after_infrastructure_repair
                or args.resume_after_uncertain_group_recollection
                or args.resume_after_proposal_selection_repair
                or args.continue_on_uncertain_attribution_exhausted,
                allow_attribution_classifier_repair_resume=args.resume_after_attribution_classifier_repair,
                allow_infrastructure_repair_resume=args.resume_after_infrastructure_repair,
                allow_uncertain_group_recollection_resume=args.resume_after_uncertain_group_recollection,
                policy_sampling_attempt_offsets=policy_sampling_attempt_offsets,
                route_health_path=collection_config.route_health_path,
                route_health_cooldown_s=collection_config.route_health_cooldown_s,
                worker_runtime_routes=collection_config.worker_runtime_routes,
                rollout_group_policy="complete" if evaluation_only else args.rollout_group_policy,
                proposer_learning_mode=getattr(
                    args, "proposer_learning_mode", "independent_frontier_v2"
                ),
                proposer_baseline_mode=getattr(args, "proposer_baseline_mode", "ema"),
                proposer_baseline_decay=getattr(args, "proposer_baseline_decay", 0.9),
                proposer_baseline_path=args.output / "proposer_baseline_state.json",
                proposer_baseline_cycle=cycle,
                require_all_planned_task_groups_for_training=(
                    evaluation_only or args.rollout_group_policy == "complete"
                )
                and (not args.allow_skipped_task_groups),
                continue_on_uncertain_attribution_exhausted=args.continue_on_uncertain_attribution_exhausted,
                backend_failure_retry_attempts=args.backend_failure_retries,
                uncertain_attribution_zero_reward=args.uncertain_attribution_zero_reward,
                pipeline_counterfactuals=args.pipeline_counterfactuals,
                counterfactual_workers=args.counterfactual_workers,
                counterfactual_pair_wall_time_s=args.counterfactual_pair_wall_time_s,
                evaluation_only=evaluation_only,
                frontier_reverify_fraction=0.0 if evaluation_only else 0.25,
                pipeline_frontier_by_dataset=args.pipeline_frontier_by_dataset,
                canary_exclude_migrated_frontier=args.canary_exclude_migrated_frontier,
                task_execution_window=None
                if evaluation_only
                and args.freeze_runtime_state
                and (args.task_scheduling_policy == "frozen_manifest_dynamic")
                else max(task_window, args.max_active_task_groups),
            ),
            graph_feature_extractor=graph_feature_extractor,
            primary_probability_observer=probability_observer,
        ).run(selected_seeds, resume=args.resume and cycle_dir.exists())
        from .async_cycle import bind_rollout_result_lineage

        rollout_result = bind_rollout_result_lineage(
            rollout_result,
            output_dir=cycle_dir,
            target_cycle=cycle,
            behavior_update_index=behavior_update_index,
            collection_mode=collection_mode,
        )
        collection_elapsed_s = time.monotonic() - cycle_started_monotonic
        return _CollectedExperimentCycle(
            cycle=cycle,
            cycle_dir=cycle_dir,
            seeds=selected_seeds,
            result=rollout_result,
            collection_config=collection_config,
            training_config=cycle_training_config,
            evaluation_only=evaluation_only,
            resumed_partial_cycle=resumed_partial_cycle,
            started_monotonic=cycle_started_monotonic,
            collection_elapsed_s=collection_elapsed_s,
        )

    first_snapshots = create_selfplay_snapshots(config)
    first_update_committed = bool(
        args.resume
        and training_config.state_path.exists()
        and (
            int(json.loads(training_config.state_path.read_text(encoding="utf-8")).get("cycle", 0))
            > completed_cycles
        )
    )
    first_lineage_path = args.output / f"cycle-{completed_cycles:04d}" / "policy_lineage.json"
    first_saved_stale = bool(
        args.resume
        and first_lineage_path.exists()
        and (int(json.loads(first_lineage_path.read_text()).get("staleness_updates", 0)) > 0)
    )
    collected_cycle = collect_cycle(
        completed_cycles,
        cycle_seeds(completed_cycles),
        collection_config=config,
        behavior_update_index=completed_cycles,
        collection_mode="synchronous",
        snapshots=first_snapshots,
        enable_probability_cache=args.async_solver_probability_cache
        and (not first_update_committed)
        and (not first_saved_stale),
    )
    pipeline_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-next-cycle")
        if args.async_next_cycle_rollouts
        else None
    )
    pipeline_state_path = args.output / "async_cycle_pipeline.json"
    try:
        for cycle in range(completed_cycles, args.cycles):
            if collected_cycle.cycle != cycle:
                raise RuntimeError("async cycle queue returned an out-of-order batch")
            cycle_started_monotonic = collected_cycle.started_monotonic
            evaluation_only = collected_cycle.evaluation_only
            cycle_collection_only = _cycle_collection_only(args, cycle)
            cycle_dir = collected_cycle.cycle_dir
            seeds = collected_cycle.seeds
            rollout_result = collected_cycle.result
            cycle_training_config = collected_cycle.training_config
            resumed_partial_cycle = collected_cycle.resumed_partial_cycle
            collection_elapsed_s = collected_cycle.collection_elapsed_s
            next_cycle_future = None
            overlap_started_monotonic = None
            next_cycle_is_frozen_evaluation = bool(
                args.final_cycle_evaluation_only and cycle + 1 == args.cycles - 1
            )
            if (
                args.async_next_cycle_rollouts
                and cycle + 1 < args.cycles
                and (not next_cycle_is_frozen_evaluation)
            ):
                assert manager is not None and pipeline_executor is not None
                saved_pipeline = (
                    json.loads(pipeline_state_path.read_text(encoding="utf-8"))
                    if args.resume and pipeline_state_path.exists()
                    else {}
                )
                resume_queued_cycle = bool(
                    int(saved_pipeline.get("collection_cycle", -1)) == cycle + 1
                    and saved_pipeline.get("state") in {"queued", "collecting", "ready", "failed"}
                )
                behavior_index = (
                    int(saved_pipeline["behavior_update_index"]) if resume_queued_cycle else cycle
                )
                behavior_snapshots = (
                    AlternatingSnapshots(
                        str(saved_pipeline["proposer_snapshot"]),
                        str(saved_pipeline["solver_snapshot"]),
                    )
                    if resume_queued_cycle
                    else create_selfplay_snapshots(config)
                )
                (rollout_spec, rollout_config) = _async_rollout_service(
                    config,
                    gpu_id=args.async_rollout_gpu_id,
                    port=args.async_rollout_port,
                    gpu_memory_utilization=args.async_rollout_gpu_memory_utilization,
                    proposer_snapshot=behavior_snapshots.proposer_snapshot,
                    solver_snapshot=behavior_snapshots.solver_snapshot,
                )
                next_seeds = cycle_seeds(cycle + 1)
                overlap_started_monotonic = time.monotonic()
                pipeline_payload = {
                    "schema_version": "async_cycle_pipeline_v1",
                    "queue_depth": 1,
                    "state": "queued",
                    "training_cycle": cycle,
                    "collection_cycle": cycle + 1,
                    "behavior_update_index": behavior_index,
                    "proposer_snapshot": behavior_snapshots.proposer_snapshot,
                    "solver_snapshot": behavior_snapshots.solver_snapshot,
                }
                _atomic_write_json(pipeline_state_path, pipeline_payload)

                def collect_next_cycle(
                    *,
                    next_cycle: int = cycle + 1,
                    selected: list[Any] = next_seeds,
                    service_spec: ModelServiceSpec = rollout_spec,
                    next_config: Any = rollout_config,
                    frozen_snapshots: Any = behavior_snapshots,
                    behavior_index: int = behavior_index,
                    state: dict[str, Any] = pipeline_payload,
                ) -> _CollectedExperimentCycle:
                    completed_artifacts = all(
                        (
                            (args.output / f"cycle-{next_cycle:04d}" / name).is_file()
                            for name in (
                                "proposer_batch.json",
                                "solver_batch.json",
                                "snapshots.json",
                            )
                        )
                    )
                    if not completed_artifacts:
                        manager.start(service_spec)
                    _atomic_write_json(pipeline_state_path, {**state, "state": "collecting"})
                    try:
                        ready = collect_cycle(
                            next_cycle,
                            selected,
                            collection_config=next_config,
                            behavior_update_index=behavior_index,
                            collection_mode="async_one_step_stale",
                            snapshots=frozen_snapshots,
                            enable_probability_cache=False,
                        )
                    except BaseException as exc:
                        _atomic_write_json(
                            pipeline_state_path,
                            {
                                **state,
                                "state": "failed",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            },
                        )
                        raise
                    _atomic_write_json(
                        pipeline_state_path,
                        {
                            **state,
                            "state": "ready",
                            "collection_elapsed_s": ready.collection_elapsed_s,
                        },
                    )
                    return ready

                next_cycle_future = pipeline_executor.submit(collect_next_cycle)
            warmup_executor = None
            warmup_future = None
            warmup_started = None
            if (
                args.pipeline_next_cycle_warmup
                and (not args.async_next_cycle_rollouts)
                and (cycle + 1 < args.cycles)
                and (fixed_components is not None)
                and (frozen_proposer is None)
            ):
                warmup_started = time.monotonic()
                warmup_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="cycle-warmup"
                )
                next_offset = (cycle + 1) * math.ceil(
                    tasks_per_cycle / len(fixed_components[1].dataset_cluster_ids)
                )
                warmup_future = warmup_executor.submit(
                    fixed_components[0].balanced_selection_seeds,
                    tasks_per_cycle,
                    offset_per_dataset=next_offset,
                )
            from .outcome_metrics import collect_outcome_metrics

            outcomes = collect_outcome_metrics(
                cycle_dir,
                cycle=cycle,
                tasks=rollout_result.tasks,
                solver_batch=rollout_result.solver_batch,
                k=rollouts_per_task,
                snapshots=rollout_result.snapshots,
            )
            update_started_monotonic = time.monotonic()
            before_role = checkpoint_callback = None
            if manager:

                def before_role(role: str) -> None:
                    del role
                    for service_role in ("proposer", "solver"):
                        manager.stop(service_role)
                    from .services import _release_project_gpu_reservations

                    _release_project_gpu_reservations(
                        (config.proposer_gpu_id, config.solver_gpu_id)
                    )

                def checkpoint_callback(role: str, checkpoint: str) -> None:
                    if training_config.solver.data_parallel_gpu_ids:
                        return
                    manager.refresh(_service_specs(config)[role], checkpoint)

            relation_credits = load_relation_credits(
                cycle_dir
                / (
                    "training_relation_credits.jsonl"
                    if (cycle_dir / "training_selection.json").exists()
                    else "relation_counterfactuals.jsonl"
                )
            )
            from .async_cycle import validate_batch_lineage

            training_state_payload = (
                json.loads(training_config.state_path.read_text(encoding="utf-8"))
                if args.resume and training_config.state_path.exists()
                else {}
            )
            update_already_committed = int(training_state_payload.get("cycle", 0)) > cycle
            learner_snapshots = (
                AlternatingSnapshots(
                    str(rollout_result.snapshots["proposer"]),
                    str(rollout_result.snapshots["solver"]),
                )
                if update_already_committed
                else create_selfplay_snapshots(config)
            )
            learner_binding = validate_batch_lineage(
                rollout_result.proposer_batch,
                rollout_result.solver_batch,
                learner_update_index=cycle,
                learner_proposer_snapshot=learner_snapshots.proposer_snapshot,
                learner_solver_snapshot=learner_snapshots.solver_snapshot,
            )
            if update_already_committed:
                learner_binding["mode"] = "committed_update_recovery"
            _atomic_write_json(cycle_dir / "learner_policy_binding.json", learner_binding)
            updates: tuple[PolicyUpdateResult, ...]
            training_compute_elapsed_s = 0.0
            service_restore_elapsed_s = 0.0
            post_update_validation_elapsed_s = 0.0
            if evaluation_only or cycle_collection_only:
                updates = ()
            else:
                trainer = AlternatingGRPOTrainer(
                    cycle_training_config,
                    trainer_factory=factory,
                    before_role_callback=before_role,
                    checkpoint_callback=None
                    if cycle_training_config.parallel_roles
                    else checkpoint_callback,
                )
                training_succeeded = False
                try:
                    committed = trainer.committed_cycle_results(cycle) if args.resume else None
                    if committed is not None:
                        updates = committed
                        durations = [float(update.duration_seconds) for update in updates]
                        training_compute_elapsed_s = (
                            max(durations)
                            if cycle_training_config.parallel_roles
                            else sum(durations)
                        )
                    else:
                        training_compute_started = time.monotonic()
                        updates = trainer.train_cycle(
                            rollout_result.proposer_batch,
                            rollout_result.solver_batch,
                            relation_credits=relation_credits,
                        )
                        training_compute_elapsed_s = time.monotonic() - training_compute_started
                    training_succeeded = True
                finally:
                    if (
                        manager
                        and training_succeeded
                        and (cycle + 1 < args.cycles)
                        and (not args.async_next_cycle_rollouts)
                    ):
                        restore_started = time.monotonic()
                        _restore_managed_policy_services(
                            manager, config, {"proposer": True, "solver": True}
                        )
                        service_restore_elapsed_s = time.monotonic() - restore_started
                validation = _validate_post_update_state(
                    cycle_dir,
                    updates,
                    cycle_training_config,
                    manager=manager
                    if cycle + 1 < args.cycles and (not args.async_next_cycle_rollouts)
                    else None,
                    adaptive_config=config,
                )
                post_update_validation_elapsed_s = float(validation["duration_s"])
            next_cycle_warmup_elapsed_s = 0.0
            if warmup_future is not None and warmup_executor is not None:
                next_seeds = warmup_future.result()
                warmup_executor.shutdown(wait=True)
                prewarmed_cycle_seeds[cycle + 1] = next_seeds
                next_cycle_warmup_elapsed_s = time.monotonic() - float(warmup_started)
                (cycle_dir / "next_cycle_warmup.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "next_cycle_warmup_v1",
                            "next_cycle": cycle + 1,
                            "seed_count": len(next_seeds),
                            "seed_ids": [seed.seed_id for seed in next_seeds],
                            "operations": ["fixed_pool_selection", "selection_metadata_validation"],
                            "solver_rollouts_started": False,
                            "duration_s": next_cycle_warmup_elapsed_s,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            async_collection_wait_s = 0.0
            async_overlap_window_s = 0.0
            if next_cycle_future is not None:
                wait_started = time.monotonic()
                collected_cycle = next_cycle_future.result()
                async_collection_wait_s = time.monotonic() - wait_started
                async_overlap_window_s = time.monotonic() - float(overlap_started_monotonic)
            cycle_metrics = collect_cycle_metrics(
                cycle=cycle,
                proposer_batch=rollout_result.proposer_batch,
                solver_batch=rollout_result.solver_batch,
                updates=updates,
                frontier_scores=rollout_result.frontier_scores,
                relation_credits=relation_credits,
                mace_path=None,
                context={
                    "config": str(args.config.resolve()),
                    "provenance_id": provenance_id,
                    "seed_data": str(args.seed_data.resolve()) if args.seed_data else None,
                    "frozen_pool_selection": str(args.frozen_pool_selection.resolve())
                    if args.frozen_pool_selection
                    else None,
                    "task_pools": [str(path.resolve()) for path in args.task_pool],
                    "seed_count": len(seeds),
                    "rollouts_per_task": rollouts_per_task,
                    "proposals_per_seed": args.proposals_per_seed,
                    "task_window": task_window,
                    "rollout_workers": rollout_workers,
                    "cycle_elapsed_s": time.monotonic() - cycle_started_monotonic,
                    "resumed_partial_cycle": resumed_partial_cycle,
                    "timing_scope": "current_process_attempt",
                    "collection_elapsed_s": collection_elapsed_s,
                    "training_and_restore_elapsed_s": time.monotonic() - update_started_monotonic,
                    "training_compute_elapsed_s": training_compute_elapsed_s,
                    "service_restore_elapsed_s": service_restore_elapsed_s,
                    "post_update_validation_elapsed_s": post_update_validation_elapsed_s,
                    "next_cycle_warmup_elapsed_s": next_cycle_warmup_elapsed_s,
                    "async_next_cycle_rollouts": args.async_next_cycle_rollouts,
                    "async_collection_cycle": cycle + 1 if next_cycle_future is not None else None,
                    "async_collection_wait_s": async_collection_wait_s,
                    "async_overlap_window_s": async_overlap_window_s,
                    "cycle_target_s": args.cycle_target_time_s,
                    "cycle_target_exceeded": bool(
                        args.cycle_target_time_s > 0
                        and time.monotonic() - cycle_started_monotonic > args.cycle_target_time_s
                    ),
                    "curriculum_profile": asdict(curriculum_profile)
                    if curriculum_profile is not None
                    else None,
                    "verifier": config.verifier,
                    "model_selection_policy": "director_set_model_v1",
                    "structural_exploration_policy": config.canvas.structural_exploration_policy,
                    "proposer_device": proposer_device,
                    "solver_device": solver_device,
                    "checkpoint_root": str(checkpoint_root),
                    "runtime_state_root": str(runtime_state_root),
                    "route_qualification": route_qualification,
                    "evaluation_only": evaluation_only,
                    "collection_only": cycle_collection_only,
                    "mock_trainer": args.mock_trainer,
                    "mock_rollouts": args.mock,
                    "optimizer_schedule": {
                        "epochs": args.epochs,
                        "mini_batch_size": args.mini_batch_size,
                        "micro_batch_size": args.micro_batch_size,
                        "max_micro_batch_tokens": args.max_micro_batch_tokens,
                        "activation_cpu_offload": args.activation_cpu_offload,
                        "activation_cpu_offload_min_tokens": args.activation_cpu_offload_min_tokens,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "parallel_role_training": args.parallel_role_training,
                        "proposer_total_optimizer_steps": proposer_total_optimizer_steps,
                        "solver_total_optimizer_steps": solver_total_optimizer_steps,
                        "formula": "cycles * epochs * ceil(role_samples_per_cycle / logical_mini_batch_size)",
                        "solver_global_trajectory_mean": args.rollout_group_policy
                        == "eligible_subset",
                    },
                },
                proposal_extraction=rollout_result.proposal_extraction,
            )
            cycle_metrics["outcomes"] = outcomes
            from .research_metrics import update_diagnostics

            cycle_metrics["learning_dynamics"] = update_diagnostics(
                cycle_metrics,
                {"proposer": rollout_result.proposer_batch, "solver": rollout_result.solver_batch},
                relation_credits,
                epochs=args.epochs,
                mini_batch_size=args.mini_batch_size,
                relation_weight=training_config.solver.relation_credit_weight,
                frontiers=rollout_result.frontier_scores,
            )
            TrainingMetricsStore(args.output).append(cycle_metrics)
            tracker.log_cycle(cycle_metrics)
            summaries.append(
                {
                    "cycle": cycle,
                    "tasks": len(rollout_result.tasks),
                    "mode": "collection_only"
                    if cycle_collection_only
                    else "evaluation_only"
                    if evaluation_only
                    else "train",
                    "updates": [asdict(update) for update in updates],
                    "metrics": cycle_metrics,
                }
            )
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {"completed_cycles": cycle + 1, "cycles": summaries},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(progress_path)
            if fixed_components is not None and (not args.async_next_cycle_rollouts):
                curriculum_state_path = cycle_dir / "curriculum_state.json"
                if curriculum_state_path.exists():
                    fixed_components[1].load_state_dict(
                        json.loads(curriculum_state_path.read_text(encoding="utf-8"))
                    )
            if cycle + 1 < args.cycles:
                if next_cycle_future is not None:
                    state = json.loads(pipeline_state_path.read_text(encoding="utf-8"))
                    _atomic_write_json(
                        pipeline_state_path,
                        {**state, "state": "consumed", "completed_training_cycle": cycle},
                    )
                else:
                    collected_cycle = collect_cycle(
                        cycle + 1,
                        cycle_seeds(cycle + 1),
                        collection_config=config,
                        behavior_update_index=cycle + 1,
                        collection_mode="synchronous",
                        snapshots=create_selfplay_snapshots(config),
                        enable_probability_cache=args.async_solver_probability_cache,
                    )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if pipeline_executor is not None:
            pipeline_executor.shutdown(wait=True, cancel_futures=True)
        if manager is not None and args.async_next_cycle_rollouts:
            manager.stop("async_rollout")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


def _policy_training_config(
    base_model_path: Path,
    checkpoint_root: Path,
    learning_rate: float,
    device: str,
    args: argparse.Namespace,
    *,
    total_optimizer_steps: int,
) -> PolicyTrainingConfig:
    effective_gradient_accumulation = math.ceil(args.mini_batch_size / args.micro_batch_size)
    if (
        args.gradient_accumulation_steps is not None
        and args.gradient_accumulation_steps != effective_gradient_accumulation
    ):
        raise ValueError(
            "gradient_accumulation_steps must equal ceil(mini_batch_size / micro_batch_size) for SESA-style mini-batch training"
        )
    return PolicyTrainingConfig(
        base_model_path,
        checkpoint_root,
        learning_rate=learning_rate,
        epochs=args.epochs,
        clip_range=args.clip_range,
        kl_coefficient=args.kl_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        mini_batch_size=args.mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=effective_gradient_accumulation,
        max_micro_batch_tokens=args.max_micro_batch_tokens,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_optimizer_steps=total_optimizer_steps,
        max_sequence_length=args.max_sequence_length,
        device=device,
        use_lora=not args.full_finetune,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        activation_cpu_offload=args.activation_cpu_offload,
        activation_cpu_offload_min_tokens=args.activation_cpu_offload_min_tokens,
        raw_policy_backward_mode=getattr(args, "raw_policy_backward_mode", "micro"),
        short_call_batching=getattr(args, "short_call_batching", False),
    )


def _validate_rollout_policy_snapshots(
    run_dir: Path,
    adaptive: Any,
    *,
    allow_stale: bool,
    roles: tuple[str, ...] = ("proposer", "solver"),
) -> None:
    path = run_dir / "snapshots.json"
    if allow_stale or not path.exists():
        return
    collected = json.loads(path.read_text(encoding="utf-8"))
    current = create_selfplay_snapshots(adaptive)
    mismatches = [
        role
        for (role, expected) in (
            ("proposer", current.proposer_snapshot),
            ("solver", current.solver_snapshot),
        )
        if role in roles
        if str(collected.get(role, "")) != expected
    ]
    if mismatches:
        raise ValueError(
            "rollout batch is stale for "
            + ", ".join(mismatches)
            + "; recollect it or explicitly pass --allow-stale-rollouts"
        )


def _load_seeds(path: Path) -> list[SeedInput]:
    return list(load_selfplay_seed_jsonl(path))


class _FrozenPoolProposer:
    """Replay exact pool selections while retaining private verifier payloads."""

    def __init__(self, proposals: dict[str, ProposedTask]) -> None:
        self.proposals = proposals

    def propose(self, seed: SeedInput, *, task_id: str) -> ProposedTask:
        seed_spec = normalize_selfplay_seed(seed)
        try:
            template = self.proposals[seed_spec.seed_id]
        except KeyError as exc:
            raise ValueError(f"frozen pool selection is missing {seed_spec.seed_id!r}") from exc
        return replace(template, task=replace(template.task, task_id=task_id))

    def rehydrate_private_payload(self, proposal: ProposedTask) -> ProposedTask:
        """Restore private data and pool attestation after public-log resume."""
        pool_id = str(proposal.metadata.get("pool_id", ""))
        try:
            template = self.proposals[pool_id]
        except KeyError as exc:
            raise ValueError(f"resumed frozen proposal has unknown pool_id {pool_id!r}") from exc
        attestation_keys = (
            "validated_pool_entry",
            "validated_pool_manifest_sha256",
            "validated_pool_version",
            "validated_pool_sha256",
        )
        attestation = {key: template.metadata.get(key) for key in attestation_keys}
        return replace(
            proposal,
            task=replace(
                proposal.task,
                metadata={**proposal.task.metadata, **attestation},
                private_verifier_payload=template.task.private_verifier_payload,
            ),
            metadata={**proposal.metadata, **attestation},
        )


def _load_frozen_pool_selection(
    path: Path, pool: FixedTaskPool
) -> tuple[list[SeedInput], _FrozenPoolProposer]:
    rows = _read_jsonl(path)
    seeds: list[SeedInput] = []
    proposals: dict[str, ProposedTask] = {}
    for line_number, row in enumerate(rows, start=1):
        recorded_task = row.get("task")
        if not isinstance(recorded_task, dict):
            raise ValueError(f"frozen selection row {line_number} has no task object")
        recorded_metadata = dict(recorded_task.get("metadata") or {})
        proposal_metadata = dict(row.get("metadata") or {})
        pool_id = str(recorded_metadata.get("pool_id") or proposal_metadata.get("pool_id") or "")
        if not pool_id or pool_id not in pool.tasks:
            raise ValueError(
                f"frozen selection row {line_number} references unknown pool_id {pool_id!r}"
            )
        if pool_id in proposals:
            raise ValueError(f"frozen selection repeats pool_id {pool_id!r}")
        selected = pool.tasks[pool_id]
        if (
            str(recorded_task.get("prompt", "")) != selected.task.prompt
            or str(recorded_task.get("task_type", "")) != selected.task.task_type
        ):
            raise ValueError(
                f"frozen selection row {line_number} does not match current pool task {pool_id!r}"
            )
        pool_attestation = {
            "validated_pool_entry": bool(selected.task.metadata.get("validated_pool_entry", False)),
            "validated_pool_manifest_sha256": selected.task.metadata.get(
                "validated_pool_manifest_sha256"
            ),
            "validated_pool_version": selected.task.metadata.get("validated_pool_version"),
            "validated_pool_sha256": selected.task.metadata.get("validated_pool_sha256"),
        }
        proposal = ProposedTask(
            task=replace(
                selected.task,
                task_id=str(recorded_task.get("task_id", f"task-{line_number}")),
                metadata={
                    **selected.task.metadata,
                    **recorded_metadata,
                    "pool_id": pool_id,
                    "proposer_mode": "frozen_pool_replay",
                    "frozen_selection_source": str(path.resolve()),
                },
            ),
            response=str(row.get("response", "")),
            token_ids=tuple((int(value) for value in row.get("token_ids", []))),
            action_mask=tuple((int(value) for value in row.get("action_mask", []))),
            metadata={
                **proposal_metadata,
                **pool_attestation,
                "pool_id": pool_id,
                "proposer_mode": "frozen_pool_replay",
                "frozen_selection_source": str(path.resolve()),
            },
        )
        if len(proposal.token_ids) != len(proposal.action_mask):
            raise ValueError(f"frozen selection row {line_number} has mismatched token arrays")
        proposals[pool_id] = proposal
        seeds.append(
            SelfPlaySeed(
                content=selected.task.prompt,
                seed_id=pool_id,
                metadata={
                    "pool_id": pool_id,
                    "dataset": selected.dataset,
                    "cluster_id": selected.cluster_id,
                    "frozen_selection_position": line_number,
                },
            )
        )
    if not seeds:
        raise ValueError("frozen pool selection is empty")
    return (seeds, _FrozenPoolProposer(proposals))


class _MockTaskEmbedder:
    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]:
        del query
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            values = [0.0] * 16
            for index, byte in enumerate(text.encode("utf-8")):
                values[index % len(values)] += float(byte) / 255.0
            norm = sum((value * value for value in values)) ** 0.5 or 1.0
            vectors.append(tuple((value / norm for value in values)))
        return vectors


def _mock_fixed_pool_backend() -> MockBackend:

    def handler(messages, _role):
        payload = json.loads(messages[-1]["content"])
        return json.dumps(
            {"candidate_id": payload["candidates"][0]["candidate_id"]}, ensure_ascii=False
        )

    return MockBackend(handler=handler)


def _fixed_pool_components(
    paths: list[Path],
    *,
    config: Any,
    mock: bool,
    seed: int,
    profile: CurriculumProfile | None = None,
) -> tuple[FixedTaskPool, ADSBoundaryScheduler, TSDSRetriever, MockBackend | None]:
    embedder = _MockTaskEmbedder() if mock else None
    pool = FixedTaskPool.from_jsonl(
        paths, embedder=embedder, require_ads_metadata=not mock, require_validation_manifest=True
    )
    scheduler = ADSBoundaryScheduler(
        pool,
        active_clusters=profile.active_clusters_per_dataset if profile else 4,
        mini_cluster_size=profile.mini_cluster_size if profile else 32,
        boundary_eps=profile.boundary_eps if profile else 0.17,
        alpha=profile.alpha if profile else 0.3,
        cooldown=profile.cooldown_per_dataset if profile else 64,
        seed=seed,
    )
    if profile is not None and (not mock):
        if len(scheduler.dataset_cluster_ids) != profile.expected_datasets:
            raise ValueError(
                f"curriculum profile {profile.name!r} expects {profile.expected_datasets} datasets, loaded {len(scheduler.dataset_cluster_ids)}"
            )
        mismatched = {
            dataset: len(cluster_ids)
            for (dataset, cluster_ids) in scheduler.dataset_cluster_ids.items()
            if len(cluster_ids) != profile.num_clusters_per_dataset
        }
        if mismatched:
            raise ValueError(
                f"curriculum profile {profile.name!r} expects {profile.num_clusters_per_dataset} ADS clusters per dataset; got {mismatched}"
            )
    retriever = TSDSRetriever(pool)
    return (pool, scheduler, retriever, _mock_fixed_pool_backend() if mock else None)


def _load_frontier_scores(path: Path) -> list[FrontierScore]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [FrontierScore(**item) for item in payload]


def _load_optional_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "validate-config": lambda: validate_config(args.path),
        "adaptive-solve": lambda: adaptive_solve(args),
        "build-graph": lambda: build_graph(args.path),
        "inspect-trace": lambda: inspect_trace(args.path, args.run_id),
        "replay": lambda: replay(args.path, args.run_id),
        "dry-run-selfplay": lambda: dry_run_selfplay(args.num_tasks, args.rollouts, args.output),
        "selfplay-rollout": lambda: selfplay_rollout(args),
        "train-cycle": lambda: train_cycle(args),
        "benchmark": lambda: benchmark(args),
        "model-services": lambda: model_services(args),
        "prepare-ads-pool": lambda: prepare_ads_pool_command(args),
        "selfplay-experiment": lambda: selfplay_experiment(args),
    }
    return handlers[args.command]()
