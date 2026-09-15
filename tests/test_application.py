from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.application import (
    REMOTE_RUNTIME_MAX_CONCURRENCY,
    AdaptiveSolverApplication,
    FixedRuntimeConfig,
    _unique_worker_token_totals,
    _verifier,
    create_adaptive_application,
    create_qwen_task_proposer,
    create_selfplay_snapshots,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.cli import _service_specs, main
from selfplay_graph_flowsteer.dataset_adapters import HealthBenchRubricVerifier
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.selfplay import FrontierScore


def write_config(
    tmp_path, *, verifier: str = "none", healthbench_auto_grader_mode: str = "official"
):
    config = tmp_path / "adaptive.toml"
    config.write_text(
        f"""
[models.proposer]
base_url = "http://127.0.0.1:8001/v1"
served_model = "Qwen3.5-9B"
base_model_path = "models/Qwen3.5-9B"
checkpoint_path = "state/checkpoints/proposer"
trainable = true

[models.solver]
base_url = "http://127.0.0.1:8002/v1"
served_model = "Qwen3.5-9B"
base_model_path = "models/Qwen3.5-9B"
checkpoint_path = "state/checkpoints/solver"
trainable = true

[runtime]
base_url = "http://127.0.0.1:8003/v1"
served_model = "Qwen3.5-9B"
model_path = "models/Qwen3.5-9B"
frozen = true

[canvas]
max_agents = 4
max_rounds = 24
relay_max_chars = 1000

[mace]
enabled = false
statistics_path = "state/mace.json"


[trace]
path = "state/traces.jsonl"

[verifier]
mode = "{verifier}"

[healthbench_judge_audit]
path = "state/private/healthbench_judge"
auto_grader_mode = "{healthbench_auto_grader_mode}"
""".strip(),
        encoding="utf-8",
    )
    return config


def test_load_config_accepts_archived_director_prompt_variant(tmp_path) -> None:
    path = write_config(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('\n\n[director]\nprompt_variant = "v2"\n')

    config = load_adaptive_config(path)
    config.validate()

    assert config.director_prompt_variant == "v2"
    assert create_adaptive_application(config, mock=True).solver.director_prompt_variant == "v2"


def test_worker_token_totals_deduplicate_reused_final_execution_artifact() -> None:
    artifact_1 = {"artifact_id": "artifact_1", "token_in": 100, "token_out": 10}
    artifact_2 = {"artifact_id": "artifact_2", "token_in": 200, "token_out": 20}
    events = [
        SimpleNamespace(payload={"execution": {"artifacts": {"a": artifact_1}}}),
        SimpleNamespace(payload={"execution": {"artifacts": {"a": artifact_2}}}),
        # SET_OUTPUT / FINISH carries artifact_2 for audit but does not execute it again.
        SimpleNamespace(payload={"execution": {"artifacts": {"a": artifact_2}}}),
    ]

    assert _unique_worker_token_totals(events, run_id="task-7-r3") == (300, 30)


def test_dataset_specific_canvas_token_budget_is_selected_per_task(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application = create_adaptive_application(config, mock=True)

    result = application.solve(
        "answer the retrieval question",
        task_id="nq-budget",
        metadata={"dataset": "nq"},
    )

    assert result.task.metadata["canvas_token_budget"] == {
        "dataset": "nq_open",
        "max_total_tokens": 65536,
        "fallback_max_total_tokens": 32768,
    }
    assert application.solver.active_canvas is not None
    assert application.solver.active_canvas.config.max_total_tokens == 65536
    assert config.canvas.max_total_tokens == 32768


def test_dataset_token_budgets_validate_against_canvas_reserves(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    with pytest.raises(ValueError, match="every configured max_total_tokens budget"):
        replace(
            config,
            canvas=replace(
                config.canvas,
                max_total_tokens_by_dataset={"nq_open": 4096},
            ),
        ).validate()


def test_bidirectional_revision_config_rejects_unknown_policy_and_threshold(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    with pytest.raises(ValueError, match="bidirectional_revision_policy"):
        replace(
            config,
            canvas=replace(
                config.canvas,
                bidirectional_revision_policy="unbounded",
            ),
        ).validate()
    with pytest.raises(ValueError, match="bidirectional_revision_confidence_threshold"):
        replace(
            config,
            canvas=replace(
                config.canvas,
                bidirectional_revision_confidence_threshold=1.1,
            ),
        ).validate()


def test_structural_exploration_config_supports_off_and_stratified(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    assert config.canvas.structural_exploration_policy == "off"

    replace(
        config,
        canvas=replace(config.canvas, structural_exploration_policy="stratified"),
    ).validate()
    with pytest.raises(ValueError, match="structural_exploration_policy"):
        replace(
            config,
            canvas=replace(config.canvas, structural_exploration_policy="forced"),
        ).validate()


def test_alfworld_worker_guidance_policy_defaults_validates_and_enters_manifest(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    assert config.alfworld.worker_guidance_policy == "factual_memory_v1"
    assert config.model_manifest()["alfworld"]["worker_guidance_policy"] == "factual_memory_v1"
    for policy in ("factual_memory_v1", "raw_state_v1", "legacy_full_v1"):
        replace(
            config,
            alfworld=replace(config.alfworld, worker_guidance_policy=policy),
        ).validate()
    with pytest.raises(ValueError, match="worker_guidance_policy"):
        replace(
            config,
            alfworld=replace(config.alfworld, worker_guidance_policy="goal_planner"),
        ).validate()


def test_swe_duplicate_responsibility_policy_defaults_to_record_only_and_validates(
    tmp_path,
) -> None:
    source = write_config(tmp_path)
    config = load_adaptive_config(source)

    assert config.swe.duplicate_responsibility_policy == "record_only"
    assert config.model_manifest()["swe_bench"]["duplicate_responsibility_policy"] == "record_only"
    for policy in ("off", "record_only", "warn_once", "reject"):
        replace(
            config,
            swe=replace(config.swe, duplicate_responsibility_policy=policy),
        ).validate()
    with pytest.raises(ValueError, match="duplicate_responsibility_policy"):
        replace(
            config,
            swe=replace(config.swe, duplicate_responsibility_policy="mandatory"),
        ).validate()

    with source.open("a", encoding="utf-8") as handle:
        handle.write('\n\n[swe]\nduplicate_responsibility_policy = "warn_once"\n')
    assert load_adaptive_config(source).swe.duplicate_responsibility_policy == "warn_once"


def test_explicit_healthbench_low_cost_verifier_is_cli_selectable(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path, verifier="healthbench_rubric_low_cost"))
    adapter = HealthBenchRubricVerifier(MockBackend([]))

    assert config.verifier == "healthbench_rubric_low_cost"
    assert (
        _verifier(
            config.verifier,
            adapters={"healthbench_rubric_low_cost": adapter},
        )
        is adapter
    )


def test_healthbench_judge_route_is_independent_of_worker_routes(tmp_path) -> None:
    path = write_config(tmp_path)
    path.write_text(
        path.read_text().replace(
            "[healthbench_judge_audit]",
            '[healthbench_judge_audit]\nruntime_route = "judge_only"',
        )
        + """
[runtimes.judge_only]
api_surface = "responses"
request_profile = "generic"
base_url = "https://judge.invalid/v1"
served_model = "gpt-5.5"
managed_locally = false
max_concurrency = 16
reasoning_effort = "low"
healthbench_grader_reasoning_effort = "low"
"""
    )
    config = load_adaptive_config(path)
    assert config.healthbench_judge_runtime_route == "judge_only"
    route = config.runtime_pool()["judge_only"]
    assert route.api_surface == "responses"
    assert route.to_dict()["api_surface"] == "responses"
    assert route.reasoning_effort == route.to_dict()["reasoning_effort"] == "low"
    assert "judge_only" not in config.worker_runtime_routes
    assert config.model_manifest()["healthbench_judge_audit"]["runtime_route"] == "judge_only"
    with pytest.raises(ValueError, match="unknown HealthBench judge runtime route"):
        replace(config, healthbench_judge_runtime_route="missing").validate()


def test_healthbench_auto_low_cost_mode_only_aliases_auto_selection(tmp_path) -> None:
    config = load_adaptive_config(
        write_config(
            tmp_path,
            verifier="auto",
            healthbench_auto_grader_mode="low_cost",
        )
    )
    official = HealthBenchRubricVerifier(MockBackend([]))
    low_cost = HealthBenchRubricVerifier(MockBackend([]))
    adapters = {
        "healthbench_rubric": official,
        "healthbench_rubric_low_cost": low_cost,
    }

    selected = _verifier(
        config.verifier,
        adapters=adapters,
        aliases={"healthbench_rubric": "healthbench_rubric_low_cost"},
    )
    assert selected is not None
    assert selected.adapters[selected.aliases["healthbench_rubric"]] is low_cost
    assert _verifier("healthbench_rubric", adapters=adapters) is official
    assert config.model_manifest()["healthbench_judge_audit"]["auto_grader_mode"] == "low_cost"


def test_swe_config_builds_actions_without_exposing_ssh_credentials(tmp_path) -> None:
    repo_cache = tmp_path / "repo-cache"
    repo_cache.mkdir()
    private_root = tmp_path / "state" / "private" / "swe"
    private_root.mkdir(parents=True)
    registry = private_root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_revision": "b" * 40,
                "instances": [
                    {
                        "instance_id": "org__repo-1",
                        "repo": "org/repo",
                        "base_commit": "a" * 40,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    identity = tmp_path / "identity"
    identity.write_text("private-key-placeholder", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("example.invalid key", encoding="utf-8")
    source = write_config(tmp_path, verifier="auto")
    with source.open("a", encoding="utf-8") as handle:
        handle.write(
            f"""

[swe]
enabled = true
repo_cache_root = "{repo_cache}"
workspace_root = "{tmp_path / "workspaces"}"
artifact_store_root = "{tmp_path / "artifacts"}"
verifier_registry_path = "{registry}"
lifecycle_log_path = "{private_root / "lifecycle.jsonl"}"
verifier_log_path = "{private_root / "verifier.jsonl"}"
verifier_host = "example.invalid"
verifier_user = "sweeval"
verifier_identity_file = "{identity}"
verifier_known_hosts_file = "{known_hosts}"
dataset_revision = "{"b" * 40}"

[swe.test_profiles]
python_syntax = ["python", "-m", "py_compile"]
"""
        )
    config = load_adaptive_config(source)
    application = create_adaptive_application(config, mock=True)

    tools = application.runtime.executor.tools
    assert "swe_edit" in tools
    manifest = config.model_manifest()["swe_bench"]
    assert manifest["commit_policy"] == "single_committer"
    assert manifest["duplicate_responsibility_policy"] == "record_only"
    assert application.solver.swe_duplicate_responsibility_policy == "record_only"
    serialized = json.dumps(manifest)
    assert "example.invalid" not in serialized
    assert str(identity) not in serialized


def test_resources_require_explicit_physical_gpu_allowlist(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "0")
    config = load_adaptive_config(write_config(tmp_path))
    assert config.allocated_gpu_ids == (0,)
    assert config.proposer_gpu_id == 0
    assert config.solver_gpu_id == 0
    assert config.runtime_gpu_id == 0

    with pytest.raises(ValueError, match="SPGFS_ALLOWED_PHYSICAL_GPUS=0"):
        replace(config, allocated_gpu_ids=(0, 3)).validate()

    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "3,4")
    replace(
        config,
        allocated_gpu_ids=(4,),
        proposer_gpu_id=4,
        solver_gpu_id=4,
        runtime_gpu_id=4,
    ).validate()


def test_answer_submission_configuration_is_loaded_and_wired(tmp_path) -> None:
    config_path = write_config(tmp_path)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[answer_submission]
enabled = true
qa_model_enabled = true
runtime_route = "default"
max_tokens = 96
require_source_span = true
"""
        )

    config = load_adaptive_config(config_path)
    application = create_adaptive_application(config, mock=True)

    assert config.answer_submission.enabled
    assert config.answer_submission.max_tokens == 96
    assert application.solver.answer_finalizer is not None


def test_director_reward_configuration_is_versioned_and_backward_compatible(
    tmp_path,
) -> None:
    legacy = load_adaptive_config(write_config(tmp_path))
    assert legacy.director_reward.version == "legacy_v1"

    config_path = write_config(tmp_path)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[director_reward]
version = "protocol_gate_v1"
"""
        )
    current = load_adaptive_config(config_path)

    assert current.director_reward.version == "protocol_gate_v1"
    assert current.model_manifest()["director_reward"] == {"version": "protocol_gate_v1"}

    v2_path = write_config(tmp_path)
    with v2_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[director_reward]
version = "protocol_gate_v2"
"""
        )
    v2 = load_adaptive_config(v2_path)
    assert v2.director_reward.version == "protocol_gate_v2"


def test_exact_match_requires_reference_and_verifies_output(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path, verifier="exact_match"))
    application = create_adaptive_application(config, mock=True)
    with pytest.raises(ValueError, match="requires a reference"):
        application.solve("task")
    result = application.solve("task", reference="mock adaptive output")
    assert result.solver_result.verification
    assert result.solver_result.verification.passed


def test_legacy_peer_statistics_are_ignored_and_preserved(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    config.mace_statistics_path.parent.mkdir(parents=True, exist_ok=True)
    config.mace_statistics_path.write_text("legacy file: must not load or overwrite")
    first = create_adaptive_application(config, mock=True)
    first.solve("first")
    second = create_adaptive_application(config, mock=True)
    assert first.mace_selector is second.mace_selector is None
    assert config.mace_statistics_path.read_text() == "legacy file: must not load or overwrite"


def test_reference_answer_supplies_grounded_mace_reward_without_output_verifier(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path, verifier="none"))
    application = create_adaptive_application(config, mock=True)

    result = application.solve("solve this", task_type="qa", reference="mock adaptive output")

    assert result.solver_result.verification is None
    assert "mace_rewards" not in result.task.metadata
    assert application.mace_selector is None


def test_adaptive_solve_cli_runs_complete_mock_path(tmp_path, capsys) -> None:
    config = write_config(tmp_path)
    assert (
        main(
            [
                "adaptive-solve",
                "--config",
                str(config),
                "--task",
                "demo",
                "--task-id",
                "cli-task",
                "--run-id",
                "cli-run",
                "--mock",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "cli-run"
    assert payload["answer"] == "mock adaptive output"
    assert payload["final_graph"]["output_agent"] == "formatter"


def test_config_rejects_obsolete_staged_execution_keys(tmp_path) -> None:
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[canvas]\n",
        "[canvas]\ninitial_build_rounds = 15\nmax_repair_edits = 3\nexecute_each_step = false\n",
        1,
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="incremental dirty-subgraph execution is mandatory"):
        load_adaptive_config(config_path)


def test_runtime_api_key_can_be_resolved_from_environment(tmp_path, monkeypatch) -> None:
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        'served_model = "Qwen3.5-9B"\nmodel_path = "models/Qwen3.5-9B"\nfrozen = true',
        'api_key_env = "NEXUS_API_KEY"\nserved_model = "gemini-test"\n'
        'model_path = "models/Qwen3.5-9B"\n'
        'request_profile = "generic"\nfrozen = true',
        1,
    )
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("NEXUS_API_KEY", "secret-from-env")

    config = load_adaptive_config(config_path)

    assert config.runtime.api_key == "secret-from-env"
    assert config.runtime.request_profile == "generic"


def test_runtime_api_key_environment_variable_is_required(tmp_path) -> None:
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[runtime]\n", '[runtime]\napi_key_env = "MISSING_NEXUS_API_KEY"\n', 1
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="MISSING_NEXUS_API_KEY"):
        load_adaptive_config(config_path)


def test_project_env_file_is_loaded_without_overriding_shell(tmp_path, monkeypatch) -> None:
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[runtime]\n", '[runtime]\napi_key_env = "NEXUS_API_KEY"\n', 1
    )
    config_path.write_text(text, encoding="utf-8")
    (tmp_path / ".env").write_text("NEXUS_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.delenv("NEXUS_API_KEY", raising=False)

    config = load_adaptive_config(config_path)
    assert config.runtime.api_key == "file-secret"

    monkeypatch.setenv("NEXUS_API_KEY", "shell-secret")
    config = load_adaptive_config(config_path)
    assert config.runtime.api_key == "shell-secret"


def test_remote_runtime_is_not_started_as_a_local_vllm_service(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            managed_locally=False,
            max_concurrency=REMOTE_RUNTIME_MAX_CONCURRENCY,
        ),
        runtime_gpu_id=config.solver_gpu_id,
    )

    config.validate()

    assert set(_service_specs(config)) == {"proposer", "solver"}


def test_role_service_gpu_memory_utilization_is_loaded(tmp_path) -> None:
    config_path = write_config(tmp_path)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n[resources]\n"
            "proposer_service_gpu_memory_utilization = 0.30\n"
            "solver_service_gpu_memory_utilization = 0.35\n"
        )

    config = load_adaptive_config(config_path)
    specs = _service_specs(config)

    assert specs["proposer"].gpu_memory_utilization == 0.30
    assert specs["solver"].gpu_memory_utilization == 0.35


def test_policy_gpu_colocation_requires_explicit_bounded_memory_budget(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    colocated = replace(
        config,
        solver_gpu_id=config.proposer_gpu_id,
        proposer_service_gpu_memory_utilization=0.30,
        solver_service_gpu_memory_utilization=0.30,
        allow_policy_gpu_colocation=True,
    )
    colocated.validate()

    with pytest.raises(ValueError, match="must sum to <= 0.70"):
        replace(
            colocated,
            proposer_service_gpu_memory_utilization=0.40,
            solver_service_gpu_memory_utilization=0.40,
        ).validate()

    with pytest.raises(ValueError, match="must be distinct"):
        replace(colocated, allow_policy_gpu_colocation=False).validate()


def test_multiple_remote_runtimes_and_worker_routes_are_loaded(tmp_path, monkeypatch) -> None:
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[runtime]\n", '[runtime]\nname = "minimax"\n', 1
    )
    text += """

[runtimes.grok]
base_url = "https://example.test/grok/v1"
api_key_env = "NEXUS_API_KEY"
served_model = "grok-4.5"
request_profile = "generic"
managed_locally = false
frozen = true

[runtime_routing]
worker_routes = ["minimax", "grok"]
skill_distiller = "minimax"
"""
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("NEXUS_API_KEY", "test-key")

    config = load_adaptive_config(config_path)
    manifest = config.model_manifest()

    assert set(config.runtime_pool()) == {"minimax", "grok"}
    assert config.worker_runtime_routes == ("minimax", "grok")
    assert config.skill_distiller_runtime == "minimax"
    assert config.additional_runtimes["grok"].model_path is None
    assert config.additional_runtimes["grok"].max_concurrency == REMOTE_RUNTIME_MAX_CONCURRENCY
    assert manifest["runtime_routing"]["worker_routes"] == ["minimax", "grok"]

    learned = create_adaptive_application(replace(config, verifier="exact_match"), mock=True)
    result = learned.solve("task", reference="mock adaptive output")
    selected_routes = {
        node["metadata"]["runtime_route"]
        for node in result.solver_result.director_run.graph["nodes"]
    }
    assert selected_routes <= {"minimax", "grok"}
    assert not hasattr(learned, "model_router")
    assert "mace_rewards" not in result.task.metadata
    assert "mace_decisions" not in result.task.metadata
    assert (
        sum(
            json.loads(turn.model_action).get("action") == "set_model"
            for turn in result.solver_result.director_run.turns
        )
        == 3
    )


def test_remote_runtime_rejects_concurrency_drift(tmp_path, monkeypatch) -> None:
    config_path = write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text += """

[runtimes.grok]
base_url = "https://example.test/grok/v1"
api_key_env = "NEXUS_API_KEY"
served_model = "grok-4.5"
request_profile = "generic"
max_concurrency = 17
managed_locally = false
frozen = true
"""
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("NEXUS_API_KEY", "test-key")

    with pytest.raises(
        ValueError,
        match="externally managed runtime.max_concurrency must not exceed 16",
    ):
        load_adaptive_config(config_path)


def test_gpt_6_astra_remote_runtime_allows_twenty_concurrent_requests() -> None:
    runtime = FixedRuntimeConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        served_model="gpt-6-astra",
        model_path=None,
        request_profile="generic",
        network_path="direct",
        max_concurrency=20,
        managed_locally=False,
    )

    runtime.validate()
    with pytest.raises(
        ValueError,
        match="externally managed runtime.max_concurrency must not exceed 20",
    ):
        replace(runtime, max_concurrency=21).validate()


@pytest.mark.parametrize("model", ["deepseek-flash", "MiniMax-M2.7"])
def test_high_capacity_remote_runtime_allows_thirty_concurrent_requests(model) -> None:
    runtime = FixedRuntimeConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        served_model=model,
        model_path=None,
        request_profile="generic",
        network_path="direct",
        max_concurrency=30,
        managed_locally=False,
    )

    runtime.validate()
    with pytest.raises(
        ValueError,
        match="externally managed runtime.max_concurrency must not exceed 30",
    ):
        replace(runtime, max_concurrency=31).validate()


def test_proposer_solver_are_separate_but_share_base_initialization(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    assert config.proposer_model.base_model_path == config.solver_model.base_model_path
    assert config.proposer_model.identity() != config.solver_model.identity()
    assert config.proposer_model.checkpoint_path != config.solver_model.checkpoint_path
    assert config.proposer_model.trainable and config.solver_model.trainable
    assert config.runtime.frozen
    snapshots = create_selfplay_snapshots(config)
    assert snapshots.proposer_snapshot == str(config.proposer_model.base_model_path.resolve())
    assert snapshots.solver_snapshot == str(config.solver_model.base_model_path.resolve())


def test_shared_proposer_solver_instance_is_rejected(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    shared = replace(
        config,
        solver_model=replace(
            config.solver_model,
            base_url=config.proposer_model.base_url,
            served_model=config.proposer_model.served_model,
        ),
    )
    with pytest.raises(ValueError, match="independent served model"):
        shared.validate()


def test_qwen_proposer_uses_only_injected_proposer_backend(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    backend = MockBackend([json.dumps({"prompt": "question", "reference": "answer"})])
    proposer = create_qwen_task_proposer(config, backend=backend)
    proposer.propose("seed", task_id="p1")
    assert backend.calls[0]["role"] == "proposer"


def test_solver_backend_cannot_be_reused_as_fixed_runtime(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    shared = MockBackend()
    with pytest.raises(ValueError, match="cannot be shared"):
        create_adaptive_application(
            config,
            director_backend=shared,
            worker_backend=shared,
            distiller_backend=shared,
        )
