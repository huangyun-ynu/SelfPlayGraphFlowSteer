from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.adaptive import _aggregate_output_agent_tool_evidence
from selfplay_graph_flowsteer.application import (
    AdaptiveSolverApplication,
    create_adaptive_application,
    create_selfplay_snapshots,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.backend_failures import (
    BackendFailureClassification,
    BackendRequestError,
)
from selfplay_graph_flowsteer.cli import _MockProposer, main
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.counterfactual import (
    RelationDecision,
    evaluate_relation_decision,
    schedule_relation_decisions,
)
from selfplay_graph_flowsteer.distributed import ThreadRolloutPool
from selfplay_graph_flowsteer.evaluation import from_adaptive_result, from_flowsteer_trajectory
from selfplay_graph_flowsteer.execution_audit import audit_cross_agent_read_overlap
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import (
    EnvironmentResultIncompleteError,
    ExecutionTrace,
    TaskSpec,
    TraceEvent,
    VerificationResult,
)
from selfplay_graph_flowsteer.rollouts import TokenizedDirectorTrajectory
from selfplay_graph_flowsteer.runtime import (
    WorkerWallClockLimitExceeded,
)
from selfplay_graph_flowsteer.selfplay import ProposedTask, SolverRollout
from selfplay_graph_flowsteer.selfplay_runtime import (
    BackendRetryExhaustedError,
    ByteTokenizer,
    CollectionInfrastructureIncidentError,
    InsufficientCompleteRolloutGroupsError,
    SelfPlayRolloutRunner,
    SelfPlayRunConfig,
    WorkerBackendUnavailableError,
    _freeze_primary_job_schedule,
    _order_primary_jobs,
    _primary_job_rollout_id,
    _recovery_decision,
    _rollout_sampling_seed,
    _runtime_owned_model_policy_failure,
    adaptive_result_to_rollout,
)


def _duplicate_swe_read_fixture(*, relation: bool = False, edit: bool = False):
    fields = {
        "role": "Repository implementation owner",
        "objective": "Implement fixes for affected distribution CDF methods",
        "scope": "SymPy probability distributions Arcsin Benini Beta BetaPrime",
        "expected_output": "Return a repository code patch with corrected CDF methods",
    }
    graph = {
        "nodes": [
            {"agent_id": "a", "metadata": {"director_delegation": fields}},
            {"agent_id": "b", "metadata": {"director_delegation": dict(fields)}},
        ],
        "relations": ([{"source": "a", "target": "b", "relation": "directed"}] if relation else []),
    }
    reads = [
        ("swe_list", {"path": ".", "workspace_version": 0}),
        (
            "swe_search",
            {"query": "_cdf", "path": "sympy/stats", "workspace_version": 0},
        ),
        (
            "swe_read",
            {
                "path": "sympy/stats/crv_types.py",
                "start_line": 1,
                "end_line": 200,
                "workspace_version": 0,
            },
        ),
    ]

    def artifact(agent_id: str) -> dict[str, object]:
        react_trace = [
            {
                "action": {
                    "call_id": f"{agent_id}-read-{index}",
                    "name": name,
                    "arguments": arguments,
                },
                "observation": {"status": "ok", "output": {"workspace_version": 0}},
            }
            for index, (name, arguments) in enumerate(reads)
        ]
        if edit and agent_id == "a":
            react_trace.append(
                {
                    "action": {
                        "call_id": "a-edit",
                        "name": "swe_edit",
                        "arguments": {
                            "operation": "replace",
                            "path": "sympy/stats/crv_types.py",
                            "workspace_version": 0,
                        },
                    },
                    "observation": {
                        "status": "ok",
                        "output": {"workspace_version": 1, "changed_files": ["x.py"]},
                    },
                }
            )
        return {
            "artifact_id": f"artifact-{agent_id}",
            "agent_id": agent_id,
            "source_artifact_ids": [],
            "react_trace": react_trace,
        }

    events = [
        {
            "payload": {
                "execution": {
                    "artifacts": {"artifact-a": artifact("a"), "artifact-b": artifact("b")}
                }
            }
        }
    ]
    return events, graph


def test_cross_agent_exact_read_overlap_flags_record_only_duplicate_exploration() -> None:
    events, graph = _duplicate_swe_read_fixture()

    audit = audit_cross_agent_read_overlap(events, graph)

    assert audit["policy"] == "record_only"
    assert audit["duplicate_read_only_exploration"] is True
    assert audit["flagged_pair_count"] == 1
    pair = audit["pairs"][0]
    assert pair["shared_exact_read_count"] == 3
    assert pair["exact_read_overlap_ratio"] == 1.0
    assert pair["same_workspace_version"] is True
    assert pair["successful_edit_count"] == 0


def test_cross_agent_read_overlap_does_not_flag_related_or_editing_agents() -> None:
    related_events, related_graph = _duplicate_swe_read_fixture(relation=True)
    edit_events, edit_graph = _duplicate_swe_read_fixture(edit=True)

    related = audit_cross_agent_read_overlap(related_events, related_graph)
    edited = audit_cross_agent_read_overlap(edit_events, edit_graph)

    assert related["duplicate_read_only_exploration"] is False
    assert related["pairs"][0]["relation_present"] is True
    assert edited["duplicate_read_only_exploration"] is False
    assert edited["pairs"][0]["successful_edit_count"] == 1


def write_config(tmp_path):
    config = tmp_path / "adaptive.toml"
    config.write_text(
        """
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

[mace]
enabled = false
statistics_path = "state/mace.json"


[trace]
path = "state/traces.jsonl"

[verifier]
mode = "none"

[director_reward]
version = "protocol_gate_v1"
""".strip(),
        encoding="utf-8",
    )
    return config


def test_adaptive_result_becomes_masked_solver_trajectory(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve("task", task_id="q1")
    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=7)
    assert rollout.trajectory.action_mask
    assert 1 in rollout.trajectory.action_mask
    assert rollout.trajectory.policy_calls
    assert all(0 in call.action_mask for call in rollout.trajectory.policy_calls)
    assert all(1 in call.action_mask for call in rollout.trajectory.policy_calls)
    assert rollout.trajectory.seed == 7
    assert "director_action_diagnostics" in rollout.trajectory.metadata
    assert rollout.trajectory.metadata["director_action_repairs"] >= 0
    assert rollout.trajectory.metadata["director_action_repair_successes"] >= 0
    assert rollout.trajectory.metadata["raw_solver_answer"] == "mock adaptive output"
    assert rollout.trajectory.metadata["submitted_answer"] == "mock adaptive output"
    assert rollout.trajectory.metadata["answer_submission"]["method"] == "legacy_passthrough"
    assert rollout.trajectory.metadata["failure_mode"] == "task_verification_failure"
    assert rollout.trajectory.metadata["incremental_execution_count"] >= 1
    assert rollout.trajectory.metadata["worker_executed_agent_count"] >= 1
    assert rollout.trajectory.metadata["worker_reused_agent_count"] >= 1
    assert rollout.trajectory.metadata["worker_scheduled_agent_count"] >= 1
    assert rollout.trajectory.metadata["worker_model_call_count"] >= 1
    assert rollout.trajectory.metadata["worker_initial_model_call_count"] >= 1
    assert rollout.trajectory.metadata["worker_revision_model_call_count"] >= 0
    assert rollout.trajectory.metadata["worker_cache_hit_count"] >= 0
    assert rollout.trajectory.metadata["worker_component_execution_count"] >= 1
    assert rollout.trajectory.metadata["worker_bidirectional_revision_gate_count"] >= 1
    assert rollout.trajectory.metadata["worker_bidirectional_revision_required_count"] >= 1
    assert (
        rollout.trajectory.metadata["worker_bidirectional_revision_wave_count"]
        == rollout.trajectory.metadata["worker_bidirectional_revision_required_count"]
    )
    assert isinstance(
        rollout.trajectory.metadata["worker_bidirectional_revision_decisions"],
        list,
    )
    assert 0.0 <= rollout.trajectory.metadata["worker_attempt_cache_hit_rate"] <= 1.0
    assert rollout.trajectory.metadata["prompt_revision_count"] >= 0
    assert rollout.trajectory.metadata["prompt_revision_target_model_call_count"] >= 0
    assert rollout.trajectory.metadata["prompt_revision_worker_model_call_count"] >= 0
    assert rollout.trajectory.metadata["prompt_revision_evidence_rejection_count"] >= 0
    assert isinstance(rollout.trajectory.metadata["prompt_revision_events"], list)
    assert 0.0 <= rollout.trajectory.metadata["worker_cache_reuse_rate"] <= 1.0
    assert rollout.trajectory.metadata["feedback_truncation_count"] == 0
    assert rollout.trajectory.metadata["structural_repair_entry_count"] >= 0
    assert (
        rollout.trajectory.metadata["structural_repair_resolution_count"]
        == rollout.trajectory.metadata["structural_repair_entry_count"]
    )
    assert rollout.trajectory.metadata["structural_repair_blocked_action_count"] == 0
    assert rollout.trajectory.metadata["semantic_no_progress_recovery_count"] == 0
    assert rollout.trajectory.metadata["output_switch_without_progress_count"] == 0
    assert rollout.trajectory.metadata["output_lifecycle_recovery_count"] == 0
    assert rollout.trajectory.metadata["disconnected_multi_agent_step_count"] >= 0
    assert rollout.trajectory.metadata["training_eligible"] is True
    assert rollout.trajectory.metadata["terminal_graph_status"] == "valid_finished"
    assert isinstance(rollout.trajectory.metadata["worker_artifact_integrity"], dict)
    assert rollout.trajectory.metadata["worker_artifact_integrity_failure"] is None

    result.task.metadata["worker_backend_failure"] = {
        "count": 1,
        "routes": ["minimax"],
        "failure_types": ["TimeoutError"],
    }
    invalid = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=9, seed=9)
    assert invalid.trajectory.reward == 0.0
    assert invalid.trajectory.metadata["answer_reward_released"] is False
    assert invalid.trajectory.metadata["failure_mode"] == "worker_backend_failure"
    assert invalid.trajectory.metadata["training_eligible"] is False
    assert "worker_backend_failure" in invalid.trajectory.metadata["training_exclusion_reasons"]
    result.task.metadata.pop("worker_backend_failure")

    result.task.metadata["worker_artifact_integrity"] = {
        "solver": {
            "claimed_confidence": 1.0,
            "effective_confidence": 0.25,
            "integrity_risks": ["unsupported_tool_verification_claim"],
            "runtime_tool_evidence": {
                "trusted": True,
                "attempted_count": 1,
                "successful_count": 0,
            },
        }
    }
    result.task.metadata["worker_output_integrity_risks"] = ["unsupported_tool_verification_claim"]
    result.task.metadata["worker_artifact_integrity_failure"] = {
        "output_agent": "solver",
        "risks": ["unsupported_tool_verification_claim"],
        "claimed_confidence": 1.0,
        "effective_confidence": 0.25,
    }
    unsafe_integrity = adaptive_result_to_rollout(
        result, ByteTokenizer(), rollout_index=14, seed=14
    )
    integrity_metadata = unsafe_integrity.trajectory.metadata
    assert integrity_metadata["training_eligible"] is False
    assert integrity_metadata["failure_mode"] == "worker_artifact_integrity_failure"
    assert "worker_artifact_integrity_failure" in integrity_metadata["training_exclusion_reasons"]
    assert integrity_metadata["worker_output_integrity_risks"] == [
        "unsupported_tool_verification_claim"
    ]
    result.task.metadata.pop("worker_artifact_integrity")
    result.task.metadata.pop("worker_output_integrity_risks")
    result.task.metadata.pop("worker_artifact_integrity_failure")

    original_dataset = result.task.metadata.get("dataset")
    original_source_split = result.task.metadata.get("source_split")
    result.task.metadata["dataset"] = "swe_bench"
    result.task.metadata["source_split"] = "train"
    result.task.metadata["swe_output_progress"] = {
        "trusted": True,
        "commit_required": True,
        "commit_ready": False,
        "state": "edit_required",
    }
    incomplete_commit = adaptive_result_to_rollout(
        result, ByteTokenizer(), rollout_index=15, seed=15
    )
    assert incomplete_commit.trajectory.metadata["training_eligible"] is False
    assert (
        "swe_output_commit_incomplete"
        in incomplete_commit.trajectory.metadata["training_exclusion_reasons"]
    )
    result.task.metadata["swe_output_progress"]["commit_ready"] = True
    result.task.metadata["swe_output_progress"].update(
        {
            "state": "typed_policy_failure",
            "policy_failure": {
                "status": "typed_policy_failure",
                "code": "read_only_policy_stall",
                "official_reward": 0.0,
            },
        }
    )
    typed_policy_failure = adaptive_result_to_rollout(
        result, ByteTokenizer(), rollout_index=16, seed=16
    )
    assert typed_policy_failure.trajectory.reward == 0.0
    # A progress flag without a verifier result or terminal attribution ledger
    # must not silently manufacture a trainable zero.
    assert typed_policy_failure.trajectory.metadata["training_eligible"] is False
    assert (
        "missing_trusted_verification"
        in (typed_policy_failure.trajectory.metadata["training_exclusion_reasons"])
    )
    assert typed_policy_failure.trajectory.metadata["reward_semantics"] == "outcome_only"
    assert (
        "swe_output_commit_incomplete"
        not in typed_policy_failure.trajectory.metadata["training_exclusion_reasons"]
    )
    result.task.metadata.pop("swe_output_progress")
    if original_dataset is None:
        result.task.metadata.pop("dataset")
    else:
        result.task.metadata["dataset"] = original_dataset
    if original_source_split is None:
        result.task.metadata.pop("source_split")
    else:
        result.task.metadata["source_split"] = original_source_split

    result.task.metadata["swe_infrastructure_failure"] = {
        "status": "timeout",
        "detail": "synthetic harness timeout",
    }
    unsafe_swe = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=10, seed=10)
    assert unsafe_swe.trajectory.metadata["training_eligible"] is False
    assert (
        "swe_infrastructure_failure" in unsafe_swe.trajectory.metadata["training_exclusion_reasons"]
    )
    result.task.metadata.pop("swe_infrastructure_failure")

    result.task.metadata["swe_environment_result"] = {
        "status": "resolved",
        "environment_completed": True,
        "official": False,
        "synthetic": True,
    }
    synthetic_swe = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=11, seed=11)
    assert synthetic_swe.trajectory.metadata["training_eligible"] is False
    assert (
        "swe_synthetic_evaluation"
        in synthetic_swe.trajectory.metadata["training_exclusion_reasons"]
    )
    result.task.metadata.pop("swe_environment_result")

    result.task.metadata.update({"dataset": "swe_bench", "source_split": "verified"})
    evaluation_split = adaptive_result_to_rollout(
        result, ByteTokenizer(), rollout_index=12, seed=12
    )
    assert evaluation_split.trajectory.metadata["training_eligible"] is False
    assert (
        "swe_non_train_split" in evaluation_split.trajectory.metadata["training_exclusion_reasons"]
    )
    result.task.metadata.update(
        {
            "experiment_split": "train",
            "split_manifest_id": "spgfs-swe-verified-iid-v1",
        }
    )
    authorized_iid_train = adaptive_result_to_rollout(
        result, ByteTokenizer(), rollout_index=13, seed=13
    )
    assert (
        "swe_non_train_split"
        not in authorized_iid_train.trajectory.metadata["training_exclusion_reasons"]
    )
    result.task.metadata.pop("dataset")
    result.task.metadata.pop("source_split")
    result.task.metadata.pop("experiment_split")
    result.task.metadata.pop("split_manifest_id")

    result.solver_result.verification = VerificationResult(1.0, True, "test")
    successful = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=1, seed=8)
    assert successful.trajectory.metadata["failure_mode"] is None

    result.solver_result.director_run.finished = False
    unfinished = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=2, seed=9)
    assert unfinished.trajectory.metadata["answer_score"] == 1.0
    assert unfinished.trajectory.metadata["answer_reward_released"] is False
    assert unfinished.trajectory.reward == 0.0
    assert unfinished.trajectory.metadata["director_reward_version"] == "outcome_only_v1"
    assert unfinished.trajectory.metadata["protocol_audit_version"] == "protocol_gate_v1"
    assert unfinished.trajectory.metadata["protocol_score"] == pytest.approx(0.75)
    assert unfinished.trajectory.metadata["protocol_qualified"] is False
    assert unfinished.trajectory.metadata["failure_mode"] == "director_protocol_failure"
    assert unfinished.trajectory.metadata["training_eligible"] is False
    assert unfinished.trajectory.metadata["terminal_graph_status"] == "unsafe_partial"


def test_aime_model_tool_policy_failure_is_trainable_negative(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Return the integer 484", task_id="aime-policy-failure"
    )
    result.task.metadata["dataset"] = "aime"
    result.task.metadata["worker_artifact_integrity"] = {
        "solver": {
            "runtime_tool_evidence": {
                "attempted_count": 1,
                "successful_count": 0,
                "failure_codes": ["SyntaxError"],
            }
        }
    }
    result.task.metadata["worker_artifact_integrity_failure"] = {
        "output_agent": "solver",
        "risks": [
            "all_tool_actions_failed",
            "terminal_tool_failure",
            "unsupported_tool_verification_claim",
        ],
    }
    result.solver_result.verification = VerificationResult(0.0, False, "numeric")

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=42)
    metadata = rollout.trajectory.metadata

    assert rollout.trajectory.reward == 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["terminal_status"] == "typed_policy_failure"
    assert metadata["terminal_graph_status"] == "typed_policy_failure"
    assert metadata["failure_mode"] == "typed_policy_failure"
    assert metadata["typed_policy_failure"]["code"] == ("aime_model_tool_policy_failure")


def test_aime_model_tool_failure_with_prior_success_is_not_retried(tmp_path) -> None:
    """A failed final Action does not erase earlier successful Worker Actions."""

    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Return the integer 484", task_id="aime-terminal-tool-policy-failure"
    )
    result.task.metadata.update(
        {
            "dataset": "aime",
            "worker_artifact_integrity": {
                "solver": {
                    "runtime_tool_evidence": {
                        "attempted_count": 3,
                        "successful_count": 2,
                        "failed_count": 1,
                        "failure_codes": ["ImportError", "SecurityError"],
                        "terminal_failure": True,
                    }
                }
            },
            "worker_artifact_integrity_failure": {
                "output_agent": "solver",
                "risks": ["terminal_tool_failure"],
            },
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "numeric")

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=42)

    assert rollout.trajectory.reward == 0.0
    assert rollout.trajectory.metadata["training_eligible"] is True
    assert rollout.trajectory.metadata["typed_policy_failure"]["code"] == (
        "aime_model_tool_policy_failure"
    )
    decision = _recovery_decision("aime", rollout=rollout)
    assert decision.scope.value == "none"
    assert decision.reason == "dataset_terminal_contract_failed"


def test_output_tool_evidence_aggregates_across_revisions() -> None:
    initial = SimpleNamespace(
        agent_id="solver",
        artifact_id="artifact-initial",
        runtime_tool_evidence={
            "successful_call_ids": ["initial-python"],
            "failed_call_ids": [],
            "failure_codes": [],
        },
    )
    final = SimpleNamespace(
        agent_id="solver",
        artifact_id="artifact-final",
        runtime_tool_evidence={
            "attempted_count": 1,
            "successful_count": 0,
            "failed_count": 1,
            "successful_call_ids": [],
            "failed_call_ids": ["revision-python"],
            "failure_codes": ["SecurityError"],
            "terminal_failure": True,
            "all_actions_failed": True,
            "unsupported_tool_verification_claim": True,
            "confidence_caps": {
                "all_tool_actions_failed": 0.35,
                "terminal_tool_failure": 0.5,
                "unsupported_tool_verification_claim": 0.25,
            },
        },
        integrity_risks=[
            "all_tool_actions_failed",
            "terminal_tool_failure",
            "unsupported_tool_verification_claim",
        ],
        unresolved_issues=[
            "runtime_integrity:all_tool_actions_failed",
            "runtime_integrity:terminal_tool_failure",
            "runtime_integrity:unsupported_tool_verification_claim",
        ],
        claimed_confidence=0.95,
        confidence=0.25,
    )
    canvas = SimpleNamespace(
        history=[
            SimpleNamespace(execution=SimpleNamespace(artifacts={"solver": initial})),
            SimpleNamespace(execution=SimpleNamespace(artifacts={"solver": final})),
        ]
    )

    _aggregate_output_agent_tool_evidence(canvas, final)

    evidence = final.runtime_tool_evidence
    assert evidence["attempted_count"] == 2
    assert evidence["successful_count"] == 1
    assert evidence["failed_count"] == 1
    assert evidence["all_actions_failed"] is False
    assert evidence["output_agent_execution_history"]["artifact_count"] == 2
    assert final.integrity_risks == ["terminal_tool_failure"]
    assert final.unresolved_issues == ["runtime_integrity:terminal_tool_failure"]
    assert final.confidence == pytest.approx(0.5)


def test_aime_terminal_worker_protocol_failure_is_trainable_negative(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Return the integer 484", task_id="aime-terminal-protocol-policy-failure"
    )
    result.task.metadata.update(
        {
            "dataset": "aime",
            "worker_artifact_integrity": {
                "solver": {
                    "runtime_tool_evidence": {"attempted_count": 1},
                    "integrity_risks": ["terminal_protocol_failure"],
                }
            },
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "numeric")
    result.solver_result.director_run.finished = False
    # This is the live failure shape: Worker exhausted finalization without a
    # valid artifact, so Director cannot legally select an output agent.
    result.solver_result.director_run.graph["output_agent"] = None

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=42)

    assert rollout.trajectory.reward == 0.0
    assert rollout.trajectory.metadata["training_eligible"] is True
    assert rollout.trajectory.metadata["typed_policy_failure"]["code"] == (
        "aime_worker_final_protocol_policy_failure"
    )
    assert _recovery_decision("aime", rollout=rollout).scope.value == "none"


def test_swe_post_commit_protocol_failure_preserves_official_result(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Fix the repository", task_id="swe-protocol-failure"
    )
    result.task.metadata.update(
        {
            "dataset": "swe_bench",
            "source_split": "train",
            "worker_artifact_integrity_failure": {
                "output_agent": "solver",
                "risks": ["terminal_protocol_failure"],
            },
            "swe_output_progress": {
                "trusted": True,
                "state": "tested",
                "selected_as_output": True,
                "commit_required": True,
                "commit_ready": True,
                "workspace_changed": True,
                "test_after_latest_edit": True,
            },
            "swe_environment_result": {
                "status": "unresolved",
                "official": True,
                "synthetic": False,
                "environment_completed": True,
            },
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "swe_outcome")
    result.solver_result.director_run.finished = False

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=43)
    metadata = rollout.trajectory.metadata

    assert rollout.trajectory.reward == 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["terminal_status"] == "completed"
    assert metadata["typed_policy_failure"] is None
    assert metadata["reward_admission_reason"] == "trusted_task_result"
    result.solver_result.verification = VerificationResult(1.0, True, "swe_outcome")
    success = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=2, seed=43)
    assert success.trajectory.reward == 1.0

    result.task.metadata["swe_infrastructure_failure"] = {"status": "timeout"}
    excluded = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=1, seed=43)
    assert excluded.trajectory.metadata["training_eligible"] is False
    assert excluded.trajectory.metadata["typed_policy_failure"] is None


@pytest.mark.parametrize("dataset", ["webshop", "alfworld"])
def test_stateful_missing_output_is_admitted_only_after_recovery_exhaustion(
    tmp_path, dataset
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the stateful task", task_id=f"{dataset}-missing-output"
    )
    result.task.metadata.update(
        {
            "dataset": dataset,
            f"{dataset}_environment_result": {
                "environment_completed": False,
                "done": False,
                "steps": 0,
                "termination_reason": "missing_output_agent",
                "budget_truncated": False,
            },
            "admit_stateful_policy_failure_terminal": False,
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, f"{dataset}_environment")
    result.solver_result.director_run.finished = False
    result.solver_result.director_run.graph["output_agent"] = None

    recoverable = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=45)
    assert recoverable.trajectory.metadata["training_eligible"] is False
    assert recoverable.trajectory.metadata["typed_policy_failure"] is None

    result.task.metadata["admit_stateful_policy_failure_terminal"] = True
    if dataset == "webshop":
        # A missing output alone cannot manufacture trusted environment evidence.
        untrusted = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=45)
        assert untrusted.trajectory.metadata["training_eligible"] is False
        result.task.metadata["worker_artifact_integrity"] = {
            "shopper": {
                "webshop_progress": {
                    "trusted": True,
                    "state": "completed",
                    "commit_ready": False,
                    "environment_owner": "shopper",
                    "environment_access": "mutable_owner",
                }
            }
        }
    terminal = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=45)
    metadata = terminal.trajectory.metadata
    assert terminal.trajectory.metadata["answer_score"] == 0.0
    assert terminal.trajectory.reward <= 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["terminal_status"] == "typed_policy_failure"
    assert metadata["typed_policy_failure"]["code"] == (
        "webshop_worker_no_staged_purchase"
        if dataset == "webshop"
        else "alfworld_director_missing_output_policy_failure"
    )


def test_alfworld_semantic_stall_is_an_immediate_trainable_negative_sample(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the stateful task", task_id="alfworld-semantic-stall"
    )
    result.task.metadata.update(
        {
            "dataset": "alfworld",
            "alfworld_environment_result": {
                "environment_completed": True,
                "done": False,
                "won": False,
                "steps": 5,
                "termination_reason": "worker_final_before_environment_done",
                "budget_truncated": False,
            },
            "alfworld_output_progress": {
                "trusted": True,
                "state": "typed_policy_failure",
                "policy_failure": {
                    "status": "typed_policy_failure",
                    "code": "alfworld_semantic_no_progress",
                    "attribution": "model_policy",
                    "fuse_threshold": 4,
                    "semantic_no_progress_count": 4,
                    "semantic_no_progress_streak": 4,
                    "repeated_transition_count": 3,
                    "last_commands": ["inventory", "examine fridge 1"],
                },
            },
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "alfworld_environment")

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=45)
    metadata = rollout.trajectory.metadata
    assert rollout.trajectory.reward == 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["terminal_status"] == "typed_policy_failure"
    assert metadata["typed_policy_failure"]["code"] == "alfworld_semantic_no_progress"


def test_webshop_semantic_stall_is_an_immediate_trainable_negative_sample(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the shopping task", task_id="webshop-semantic-stall"
    )
    result.task.metadata.update(
        {
            "dataset": "webshop",
            "webshop_environment_result": {
                "environment_completed": True,
                "done": False,
                "steps": 5,
                "termination_reason": "missing_output_agent",
                "budget_truncated": False,
            },
            "worker_artifact_integrity": {
                "shopper": {
                    "integrity_risks": [],
                    "webshop_progress": {
                        "trusted": True,
                        "state": "typed_policy_failure",
                        "policy_failure": {
                            "status": "typed_policy_failure",
                            "code": "webshop_semantic_no_progress",
                            "attribution": "model_policy",
                            "fuse_threshold": 4,
                            "semantic_no_progress_count": 4,
                            "semantic_no_progress_streak": 4,
                            "duplicate_state_action_count": 3,
                            "unique_public_evidence_count": 2,
                        },
                    },
                }
            },
            "worker_artifact_integrity_failure": None,
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "webshop_environment")
    result.solver_result.director_run.finished = False
    result.solver_result.director_run.graph["output_agent"] = None

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=45)
    metadata = rollout.trajectory.metadata
    assert rollout.trajectory.reward == 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["terminal_status"] == "typed_policy_failure"
    assert metadata["typed_policy_failure"]["code"] == "webshop_semantic_no_progress"
    assert metadata["typed_policy_failure"]["worker_agent"] == "shopper"


def test_adaptive_result_aggregates_prompt_revision_trace_audit(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    initial_prompt = {
        "action": "set_prompt",
        "target": "solver",
        "role": "Analyst",
        "objective": "Solve the task.",
        "scope": "Reason independently.",
        "expected_output": "Return a finding.",
    }
    revision_prompt = {
        "action": "set_prompt",
        "target": "solver",
        "role": "Verifier",
        "objective": "Recheck the task after the role changed.",
        "scope": "Verify independently.",
        "expected_output": "Return a checked finding.",
        "revision_basis": "structural_role_change",
        "evidence_agent_ids": ["solver"],
    }
    repeated_revision = {
        **revision_prompt,
        "role": "Reviewer",
        "objective": "Repeat the check without new evidence.",
    }
    director = MockBackend(
        [
            '{"action":"add_agent","agent_id":"solver"}',
            json.dumps(initial_prompt),
            '{"action":"set_model","target":"solver","runtime_route":"default"}',
            '{"action":"set_layer","target":"solver","layer":1}',
            json.dumps(revision_prompt),
            json.dumps(repeated_revision),
            '{"action":"set_output","target":"solver"}',
            '{"action":"finish"}',
        ]
    )
    result = create_adaptive_application(config, mock=True, director_backend=director).solve(
        "task", task_id="prompt-revision"
    )

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=7)
    metadata = rollout.trajectory.metadata

    assert metadata["prompt_revision_count"] == 1
    assert metadata["prompt_revision_basis_counts"] == {"structural_role_change": 1}
    assert metadata["prompt_revision_target_model_call_count"] == 1
    assert metadata["prompt_revision_worker_model_call_count"] == 1
    assert metadata["prompt_revision_evidence_rejection_count"] == 1
    assert metadata["prompt_revision_events"][0]["target"] == "solver"


def test_adaptive_result_records_mandatory_bidirectional_revision(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    worker = MockBackend(
        handler=lambda _messages, _role: json.dumps(
            {
                "answer": "shared checked answer",
                "summary": "independent derivation reached the shared answer",
                "confidence": 0.95,
                "evidence": ["checked derivation"],
                "unresolved_issues": [],
                "tool_summary": [],
            }
        )
    )
    result = create_adaptive_application(config, mock=True, worker_backend=worker).solve(
        "task", task_id="revision-skip"
    )

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=7)
    metadata = rollout.trajectory.metadata

    assert metadata["worker_bidirectional_revision_gate_count"] == 1
    assert metadata["worker_bidirectional_revision_required_count"] == 1
    assert metadata["worker_bidirectional_revision_skipped_agent_count"] == 0
    assert metadata["worker_bidirectional_revision_wave_count"] == 1
    assert metadata["worker_revision_model_call_count"] == 2
    assert metadata["worker_bidirectional_revision_reason_counts"] == {"policy_always": 1}
    assert metadata["worker_bidirectional_revision_decisions"][0]["revision_required"] is True


def test_adaptive_result_records_outcome_reward_and_separate_protocol_audit(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Find the requested answer.", task_id="q-v2"
    )

    rollout = adaptive_result_to_rollout(
        result,
        ByteTokenizer(),
        rollout_index=0,
        seed=7,
        reward_version="protocol_gate_v2",
    )

    metadata = rollout.trajectory.metadata
    assert metadata["director_reward_version"] == "outcome_only_v1"
    assert metadata["protocol_audit_version"] == "protocol_gate_v2"
    assert metadata["task_reward"] == rollout.trajectory.reward
    assert metadata["delegation_fidelity"] is True
    assert metadata["delegation_issues"] == []
    assert metadata["protocol_reward"]["delegation_fidelity"] is True


def test_swe_rollout_reward_is_strict_outcome_only(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Fix the repository issue.", task_id="swe-task"
    )
    result.task.metadata.update(
        {
            "dataset": "swe_bench",
            "source_split": "verified",
            "experiment_split": "train",
            "split_manifest_id": "iid-v1",
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "swe_outcome")
    # A generic unfinished protocol would have a fractional negative reward.
    # SWE keeps only the official binary outcome and is excluded separately.
    result.solver_result.director_run.finished = False

    rollout = adaptive_result_to_rollout(
        result,
        ByteTokenizer(),
        rollout_index=0,
        seed=7,
        reward_version="protocol_gate_v2",
    )

    assert rollout.trajectory.reward == 0.0
    assert rollout.trajectory.metadata["base_director_reward"] == 0.0
    assert rollout.trajectory.metadata["reward_semantics"] == "outcome_only"
    assert rollout.trajectory.metadata["training_eligible"] is False


def test_selfplay_cli_persists_two_policy_batches_and_resumes(tmp_path, capsys) -> None:
    config_path = write_config(tmp_path)
    output = tmp_path / "run"
    arguments = [
        "selfplay-rollout",
        "--config",
        str(config_path),
        "--seed",
        "seed answer",
        "--rollouts",
        "2",
        "--output",
        str(output),
        "--mock",
    ]
    assert main(arguments) == 0
    capsys.readouterr()
    assert main([*arguments, "--resume"]) == 0
    capsys.readouterr()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["trainable_roles"] == ["proposer", "solver"]
    assert manifest["optimizer_steps"] == 0
    assert manifest["director_reward_versions"] == ["outcome_only_v1"]
    assert manifest["structural_exploration_policies"] == ["off"]
    assert manifest["rollout_batching"] == {
        "mode": "independent_requests_continuous_server_batch",
        "task_scheduling_policy": "logical_windows",
        "primary_job_order": "round_robin",
        "primary_duration_estimate_version": "20260909-cycle1-14x5-v1",
        "max_active_task_groups": 1,
        "curriculum_observation_order": "logical_manifest_windows",
        "logical_window_size": 2,
        "max_active_rollouts": 1,
        "incomplete_group_policy": "quarantine_without_exact_resume",
        "training_eligibility_policy": "valid_finished_execution_only",
    }
    assert len((output / "solver_rollouts.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "proposer_batch.json").read_text())["role"] == "proposer"
    assert json.loads((output / "solver_batch.json").read_text())["role"] == "solver"
    assert not (output / "mace_snapshots").exists()
    assert (
        json.loads((output / "action_protocol.json").read_text())["action_protocol"]
        == "director_model_v1"
    )
    rollout_rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert all("mace_window_audit" not in row["metadata"] for row in rollout_rows)


def test_single_rollout_requires_evaluation_mode(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    def runner(evaluation_only):
        return SelfPlayRolloutRunner(
            proposer=_MockProposer(),
            application_factory=lambda seed: create_adaptive_application(config, mock=True),
            tokenizer=ByteTokenizer(),
            snapshots=create_selfplay_snapshots(config),
            output_dir=tmp_path / "single_evaluation",
            config=SelfPlayRunConfig(
                1, evaluation_only=evaluation_only, frontier_reverify_fraction=0.0
            ),
        )

    with pytest.raises(ValueError, match="at least two"):
        runner(False)
    runner(True).run(["seed"])
    rows = (tmp_path / "single_evaluation/solver_rollouts.jsonl").read_text().splitlines()
    assert len(rows) == 1


def test_runner_rejects_exact_resume_of_an_incomplete_task_by_default(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "partial"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2),
    )
    runner.run(["seed"])
    lines = (output / "solver_rollouts.jsonl").read_text().splitlines()
    (output / "solver_rollouts.jsonl").write_text(lines[0] + "\n")
    factory_calls: list[int] = []
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: (
            factory_calls.append(seed),
            create_adaptive_application(config, mock=True),
        )[1],
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2),
    )
    with pytest.raises(RuntimeError, match="exact rollout resume is disabled"):
        runner.run(["seed"], resume=True)
    assert factory_calls == []
    assert len((output / "solver_rollouts.jsonl").read_text().splitlines()) == 1


def test_task_window_proposes_before_execution_and_batches_curriculum_updates(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    events: list[str] = []
    observation_windows: list[list[str]] = []

    class WindowedProposer:
        def propose(self, seed, *, task_id: str):
            events.append(f"propose:{task_id}")
            return _MockProposer().propose(seed, task_id=task_id)

        def observe_many(self, observations):
            observation_windows.append(
                [proposal.task.task_id for proposal, _rewards in observations]
            )

    def application_factory(seed):
        events.append(f"execute:{seed}")
        return create_adaptive_application(config, mock=True)

    SelfPlayRolloutRunner(
        proposer=WindowedProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "windowed",
        config=SelfPlayRunConfig(2, workers=1, task_window=2),
    ).run(["one", "two", "three"])

    assert events[:2] == ["propose:task-1", "propose:task-2"]
    assert observation_windows == [["task-1", "task-2"], ["task-3"]]


def test_task_window_submits_rollouts_round_robin_across_tasks(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class RecordingPool:
        def __init__(self) -> None:
            self.job_ids: list[tuple[str, int]] = []

        def iter_map(self, function, jobs):
            jobs = list(jobs)
            self.job_ids.extend(
                (proposal.task.task_id, rollout_index) for proposal, rollout_index in jobs
            )
            return iter(function(job) for job in jobs)

    pool = RecordingPool()
    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "round-robin",
        config=SelfPlayRunConfig(5, task_window=4, workers=4),
        rollout_pool=pool,
    ).run(["one", "two", "three", "four"])

    assert pool.job_ids[:4] == [
        ("task-1", 0),
        ("task-2", 0),
        ("task-3", 0),
        ("task-4", 0),
    ]


def test_long_tail_order_starts_every_task_then_prioritizes_slow_datasets() -> None:
    def proposal(task_id: str, dataset: str) -> ProposedTask:
        return ProposedTask(
            TaskSpec(task_id, f"solve {dataset}", metadata={"dataset": dataset}),
            response="{}",
        )

    alfworld = proposal("alf", "alfworld")
    hotpot = proposal("hot", "hotpotqa")
    swe = proposal("swe", "swe_bench")
    jobs = [
        (alfworld, 0),
        (hotpot, 0),
        (swe, 0),
        (alfworld, 1),
        (hotpot, 1),
        (swe, 1),
    ]

    ordered = _order_primary_jobs(jobs, "long_tail_first")

    assert [_primary_job_rollout_id(job) for job in ordered] == [
        "alf-r0",
        "hot-r0",
        "swe-r0",
        "alf-r1",
        "swe-r1",
        "hot-r1",
    ]


def test_frozen_long_tail_schedule_is_replayed_exactly_on_resume(tmp_path) -> None:
    def proposal(task_id: str, dataset: str) -> ProposedTask:
        return ProposedTask(
            TaskSpec(task_id, f"solve {dataset}", metadata={"dataset": dataset}),
            response="{}",
        )

    jobs = [
        (proposal("hot", "hotpotqa"), 0),
        (proposal("swe", "swe_bench"), 0),
        (proposal("hot", "hotpotqa"), 1),
        (proposal("swe", "swe_bench"), 1),
    ]
    first, first_window = _freeze_primary_job_schedule(
        tmp_path,
        window_start=0,
        jobs=jobs,
        requested_order="long_tail_first",
    )
    resumed, resumed_window = _freeze_primary_job_schedule(
        tmp_path,
        window_start=0,
        jobs=reversed(jobs),
        requested_order="round_robin",
    )

    first_ids = [_primary_job_rollout_id(job) for job in first]
    assert [_primary_job_rollout_id(job) for job in resumed] == first_ids
    assert resumed_window == first_window
    assert first_window["order"] == "long_tail_first"
    payload = json.loads((tmp_path / "rollout_job_schedule.json").read_text())
    assert payload["duration_estimate_version"] == "20260909-cycle1-14x5-v1"
    assert payload["windows"][0]["ordered_rollout_ids"] == first_ids


def test_frozen_manifest_dynamic_refills_groups_but_observes_logical_windows(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    events: list[str] = []
    observation_windows: list[list[str]] = []

    class FrozenProposer:
        def propose(self, seed, *, task_id: str):
            events.append(f"propose:{task_id}")
            return _MockProposer().propose(seed, task_id=task_id)

        def observe_many(self, observations):
            observation_windows.append(
                [proposal.task.task_id for proposal, _rewards in observations]
            )

    class RecordingGroupedPool:
        def __init__(self) -> None:
            self.calls: list[tuple[list[tuple[str, int]], int]] = []

        def iter_map_grouped(self, function, jobs, *, group_key, max_active_groups):
            jobs = list(jobs)
            self.calls.append(
                (
                    [
                        (str(group_key(job)), rollout_index)
                        for job in jobs
                        for _proposal, rollout_index in (job,)
                    ],
                    max_active_groups,
                )
            )
            return iter(function(job) for job in jobs)

    def application_factory(seed):
        events.append(f"execute:{seed}")
        return create_adaptive_application(config, mock=True)

    pool = RecordingGroupedPool()
    SelfPlayRolloutRunner(
        proposer=FrozenProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "dynamic",
        config=SelfPlayRunConfig(
            2,
            workers=2,
            task_window=2,
            require_all_proposals=True,
            task_scheduling_policy="frozen_manifest_dynamic",
            max_active_task_groups=2,
        ),
        rollout_pool=pool,
    ).run(["one", "two", "three"])

    first_execute = next(
        index for index, event in enumerate(events) if event.startswith("execute:")
    )
    assert events[:first_execute] == [
        "propose:task-1",
        "propose:task-2",
        "propose:task-3",
    ]
    assert len(pool.calls) == 1
    assert pool.calls[0][1] == 2
    assert pool.calls[0][0][:3] == [
        ("task-1", 0),
        ("task-2", 0),
        ("task-3", 0),
    ]
    assert observation_windows == [["task-1", "task-2"], ["task-3"]]
    scheduler_events = [
        json.loads(line)
        for line in (tmp_path / "dynamic" / "rollout_scheduler_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    event_names = [row["event"] for row in scheduler_events]
    assert event_names.count("job_submitted") == 6
    assert event_names.count("job_started") == 6
    assert event_names.count("attempt_started") == 6
    assert event_names.count("attempt_finished") == 6
    assert event_names.count("job_finished") == 6
    assert event_names.count("group_admitted") == 3
    assert event_names.count("group_released") == 3
    for task_id in ("task-1", "task-2", "task-3"):
        admitted = next(
            index
            for index, row in enumerate(scheduler_events)
            if row["event"] == "group_admitted" and row["task_id"] == task_id
        )
        released = next(
            index
            for index, row in enumerate(scheduler_events)
            if row["event"] == "group_released" and row["task_id"] == task_id
        )
        assert admitted < released


def test_frozen_manifest_dynamic_requires_all_proposals(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    with pytest.raises(ValueError, match="requires require_all_proposals=true"):
        SelfPlayRolloutRunner(
            proposer=_MockProposer(),
            application_factory=lambda seed: create_adaptive_application(config, mock=True),
            tokenizer=ByteTokenizer(),
            snapshots=create_selfplay_snapshots(config),
            output_dir=tmp_path / "invalid-dynamic",
            config=SelfPlayRunConfig(
                2,
                task_scheduling_policy="frozen_manifest_dynamic",
            ),
        )


def test_resume_cannot_bypass_a_persisted_structural_collapse_stop(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "collapsed"
    output.mkdir()
    (output / "collapse_monitor.jsonl").write_text(
        json.dumps(
            {
                "single_agent_rate": 1.0,
                "within_task_unique_graph_ratio": 0.2,
                "alert": True,
                "consecutive_alert_windows": 2,
            }
        )
        + "\n"
    )
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            structural_exploration_policy="stratified",
            collapse_patience_windows=2,
        ),
    )

    with pytest.raises(RuntimeError, match="collapse stop remains active on resume"):
        runner.run(["seed"], resume=True)


def test_policy_off_reclassifies_persisted_legacy_only_collapse_stop(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "legacy-collapse-off"
    output.mkdir()
    (output / "collapse_monitor.jsonl").write_text(
        json.dumps(
            {
                "window_key": "old-window",
                "single_agent_rate": 1.0,
                "within_task_unique_graph_ratio": 0.2,
                "alert": True,
                "alert_reasons": ["legacy_low_diversity"],
                "consecutive_alert_windows": 2,
            }
        )
        + "\n"
    )
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            structural_exploration_policy="off",
            collapse_patience_windows=2,
        ),
    )

    result = runner.run(["seed"], resume=True)

    assert len(result.solver_batch.samples) == 2
    policy_event = json.loads(
        (output / "collapse_policy_events.jsonl").read_text().splitlines()[-1]
    )
    assert policy_event["event"] == "resume_legacy_stop_reclassified"


def test_policy_off_keeps_legacy_low_diversity_observable_without_alert(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "legacy-observable-off",
        config=SelfPlayRunConfig(
            2,
            structural_exploration_policy="off",
            collapse_single_agent_threshold=0.0,
            collapse_unique_graph_threshold=1.0,
            collapse_task_relation_threshold=1.0,
        ),
    )

    result = runner.run(["seed"])
    event = json.loads(
        (tmp_path / "legacy-observable-off" / "collapse_monitor.jsonl").read_text().splitlines()[-1]
    )

    assert len(result.solver_batch.samples) == 2
    assert "single_agent_rate" in event
    assert "within_task_unique_graph_ratio" in event
    assert event["alert"] is False
    assert event["alert_reasons"] == []
    assert event["thresholds"]["structural_exploration_policy"] == "off"


def test_collapse_streak_can_be_inherited_across_cycle_boundary(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "next-cycle",
        config=SelfPlayRunConfig(
            2,
            structural_exploration_policy="stratified",
            collapse_single_agent_threshold=0.0,
            collapse_unique_graph_threshold=1.0,
            collapse_task_relation_threshold=1.0,
            collapse_patience_windows=2,
            initial_collapse_alert_streak=1,
        ),
    )

    runner.run(["seed"])
    event = json.loads(
        (tmp_path / "next-cycle" / "collapse_monitor.jsonl").read_text().splitlines()[-1]
    )
    assert event["alert"] is True
    assert event["consecutive_alert_windows"] == 2


def test_task_conditioned_relation_use_is_not_global_structural_collapse(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "task-conditioned",
        config=SelfPlayRunConfig(5, task_window=4),
    )

    def rollout(task_id: str, index: int, *, relation: bool) -> SolverRollout:
        graph = MultiAgentGraph()
        graph.add_agent("a")
        graph.set_prompt("a", "solve")
        output = "a"
        if relation:
            graph.add_agent("b")
            graph.set_prompt("b", "check")
            graph.set_relation("a", "b", "bidirectional")
            output = "b"
        graph.set_output(output)
        trajectory = TokenizedDirectorTrajectory(
            f"{task_id}-r{index}",
            task_id,
            (1,),
            (1,),
            1.0,
            graph.to_dict(),
            metadata={"relation_counterfactual_candidate_count": int(relation)},
        )
        return SolverRollout(trajectory, graph)

    tasks = [f"task-{index}" for index in range(1, 5)]
    rollouts = {
        f"{task_id}-r{index}": rollout(
            task_id,
            index,
            relation=(task_id == "task-2" and index < 3),
        )
        for task_id in tasks
        for index in range(5)
    }
    event = runner._record_collapse_window(
        window_start=0,
        window_entries=[
            (task_id, _MockProposer().propose("seed", task_id=task_id)) for task_id in tasks
        ],
        rollouts_by_id=rollouts,
        prior_events=[],
    )

    assert event is not None
    assert event["single_agent_rate"] == pytest.approx(0.85)
    assert event["within_task_unique_graph_ratio"] <= 0.4
    assert event["max_task_relation_rate"] == pytest.approx(0.6)
    assert event["disconnected_multi_agent_rate"] == 0.0
    assert event["alert"] is False

    disconnected = MultiAgentGraph()
    for agent_id in ("a", "b"):
        disconnected.add_agent(agent_id)
        disconnected.set_prompt(agent_id, "solve")
    disconnected.set_output("a")
    bad = TokenizedDirectorTrajectory(
        "task-4-r0",
        "task-4",
        (1,),
        (1,),
        0.0,
        disconnected.to_dict(),
        metadata={"relation_counterfactual_candidate_count": 0},
    )
    rollouts["task-4-r0"] = SolverRollout(bad, disconnected)
    disconnected_event = runner._record_collapse_window(
        window_start=0,
        window_entries=[
            (task_id, _MockProposer().propose("seed", task_id=task_id)) for task_id in tasks
        ],
        rollouts_by_id=rollouts,
        prior_events=[],
    )
    assert disconnected_event is not None
    assert disconnected_event["disconnected_multi_agent_rate"] == pytest.approx(0.05)
    assert disconnected_event["alert"] is True
    assert "disconnected_multi_agent" in disconnected_event["alert_reasons"]


def test_parallel_rollout_failure_preserves_other_successes_for_resume(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class OneFailureApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            if self.seed == 0:
                raise RuntimeError("one rollout failed")
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "partial-success"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=OneFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=2),
    )

    with pytest.raises(RuntimeError, match="preserving independent successes"):
        runner.run(["seed"])

    saved = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert saved == ["task-1-r1"]
    assert errors == [
        {
            "task_id": "task-1",
            "rollout_id": "task-1-r0",
            "rollout_index": 0,
            "error_type": "RuntimeError",
            "message": "one rollout failed",
            "attempt_count": 1,
        }
    ]
    quarantine = [
        json.loads(line) for line in (output / "quarantined_groups.jsonl").read_text().splitlines()
    ]
    assert quarantine[0]["task_id"] == "task-1"
    assert quarantine[0]["valid_rollout_ids"] == ["task-1-r1"]
    assert quarantine[0]["missing_rollout_ids"] == ["task-1-r0"]
    assert quarantine[0]["training_excluded"] is True
    assert quarantine[0]["exact_resume_allowed"] is False
    assert json.loads((output / "batch_gate.json").read_text())["status"] == "blocked"


def test_rollouts_are_persisted_in_completion_order_without_group_barrier(tmp_path) -> None:
    import time

    config = load_adaptive_config(write_config(tmp_path))

    class DelayedApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            time.sleep(0.15 if self.seed == 0 else 0.01)
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "rolling-completion"
    result = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=DelayedApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=2),
    ).run(["seed"])

    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert persisted == ["task-1-r1", "task-1-r0"]
    assert len(result.solver_batch.samples) == 2


def test_incomplete_task_group_is_quarantined_while_complete_sibling_trains(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class OneTaskFailureApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            if kwargs["task_id"] == "task-1" and self.seed == 0:
                raise RuntimeError("isolated task-1 failure")
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "group-quarantine"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=OneTaskFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=2, task_window=2),
    )

    result = runner.run(["first", "second"])

    assert [task.task_id for task in result.tasks] == ["task-2"]
    assert {sample.task_id for sample in result.proposer_batch.samples} == {"task-2"}
    assert {sample.task_id for sample in result.solver_batch.samples} == {"task-2"}
    assert len(result.solver_batch.samples) == 2
    assert all(
        sample.metadata["rollout_group_complete"] is True
        and sample.metadata["rollout_group_size_expected"] == 2
        for sample in result.solver_batch.samples
    )
    quarantine = [
        json.loads(line) for line in (output / "quarantined_groups.jsonl").read_text().splitlines()
    ]
    assert len(quarantine) == 1
    assert quarantine[0]["task_id"] == "task-1"
    assert quarantine[0]["valid_rollout_count"] == 1
    gate = json.loads((output / "batch_gate.json").read_text())
    assert gate["status"] == "ready"
    assert gate["complete_task_ids"] == ["task-2"]
    assert gate["quarantined_task_ids"] == ["task-1"]

    resumed = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: (_ for _ in ()).throw(
            AssertionError("sealed quarantine and complete group must not recollect")
        ),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=2, task_window=2),
    ).run(["first", "second"], resume=True)
    assert [task.task_id for task in resumed.tasks] == ["task-2"]


def test_strict_batch_gate_blocks_training_until_every_planned_group_is_complete(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class OneTaskFailureApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            if kwargs["task_id"] == "task-1" and self.seed == 0:
                raise RuntimeError("isolated task-1 failure")
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "strict-group-gate"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=OneTaskFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=2,
            task_window=2,
            require_all_planned_task_groups_for_training=True,
        ),
    )

    with pytest.raises(
        InsufficientCompleteRolloutGroupsError,
        match="all planned task groups are required before training",
    ):
        runner.run(["first", "second"])

    gate = json.loads((output / "batch_gate.json").read_text())
    assert gate["status"] == "blocked"
    assert gate["planned_task_group_count"] == 2
    assert gate["complete_task_ids"] == ["task-2"]
    assert gate["incomplete_task_ids"] == ["task-1"]
    assert gate["all_planned_task_groups_complete"] is False
    assert gate["require_all_planned_task_groups_for_training"] is True
    assert not (output / "proposer_batch.json").exists()
    assert not (output / "solver_batch.json").exists()


def test_non_trainable_attempts_are_audited_but_never_persisted_as_primary(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class OneUnsafeApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            result = self.delegate.solve(*args, **kwargs)
            if kwargs["task_id"] == "task-1":
                result.solver_result.director_run.finished = False
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "unsafe-group-quarantine"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=OneUnsafeApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=2, task_window=2),
    )
    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["first", "second"])

    assert 0 <= len((output / "solver_rollouts.jsonl").read_text().splitlines()) <= 3
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 1
    assert {item["rollout_id"] for item in attempts} == {"task-1-r0"}
    assert all(item["accepted_as_primary"] is False for item in attempts)
    assert all(item["same_slot_seed_preserved"] is True for item in attempts)
    abort = json.loads((output / "collection_abort.json").read_text())
    assert abort["incident_class"] == "attribution_uncertain_recovery_exhausted"
    assert not (output / "solver_batch.json").exists()


def test_evaluation_mode_skips_uncertain_exhausted_group_without_aborting(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class OneUnsafeApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            result = self.delegate.solve(*args, **kwargs)
            if kwargs["task_id"] == "task-1":
                result.solver_result.director_run.finished = False
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "evaluation-skip-uncertain"
    result = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=OneUnsafeApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=2,
            task_window=2,
            require_all_planned_task_groups_for_training=False,
            continue_on_uncertain_attribution_exhausted=True,
        ),
    ).run(["first", "second"])

    assert [task.task_id for task in result.tasks] == ["task-2"]
    assert not (output / "collection_abort.json").exists()
    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert any(item["error_type"] == "UncertainAttributionExhaustedError" for item in errors)
    quarantine = [
        json.loads(line) for line in (output / "quarantined_groups.jsonl").read_text().splitlines()
    ]
    assert quarantine[-1]["task_id"] == "task-1"


def test_timeout_gets_one_same_slot_recovery_and_preserves_rollout_id(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    factory_seeds: list[int] = []
    attempts_by_seed: dict[int, int] = {}

    class TimeoutThenSuccessApplication:
        def __init__(self, seed: int) -> None:
            factory_seeds.append(seed)
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        @property
        def runtime(self):
            return self.delegate.runtime

        def solve(self, *args, **kwargs):
            attempts_by_seed[self.seed] = attempts_by_seed.get(self.seed, 0) + 1
            if self.seed == 0 and attempts_by_seed[self.seed] == 1:
                raise WorkerWallClockLimitExceeded(
                    "test wall-clock budget",
                    reason="no_progress",
                    stage="director_turn_start",
                    elapsed_s=90.0,
                    idle_s=90.0,
                )
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "timeout-replacement"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=TimeoutThenSuccessApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            allow_legacy_whole_rollout_recovery=True,
            workers=2,
            replacement_rollouts_per_task=2,
        ),
    )

    result = runner.run(["seed"])

    assert len(result.solver_batch.samples) == 2
    assert 0 in factory_seeds and 1 in factory_seeds
    recovery_seed = next(seed for seed in factory_seeds if seed not in {0, 1})
    rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    replacement = next(row for row in rows if row["rollout_id"] == "task-1-r0")
    assert replacement["seed"] == recovery_seed
    assert replacement["metadata"]["replacement_attempt"] == 1
    assert replacement["metadata"]["same_slot_seed_preserved"] is True
    assert replacement["metadata"]["same_slot_executor_seed_preserved"] is True
    assert replacement["metadata"]["policy_resampled_on_replacement"] is True
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert [event["success"] for event in attempts] == [False, True]
    assert attempts[0]["timeout"]["reason"] == "no_progress"
    assert (output / "rollout_timeout_snapshots.jsonl").exists()


def test_transient_backend_failure_aborts_without_same_slot_recovery_for_non_swe(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    factory_seeds: list[int] = []
    attempts_by_seed: dict[int, int] = {}

    class BackendThenSuccessApplication:
        def __init__(self, seed: int) -> None:
            factory_seeds.append(seed)
            attempts_by_seed[seed] = attempts_by_seed.get(seed, 0) + 1
            self.seed = seed
            self.attempt = attempts_by_seed[seed]
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            if self.seed == 0 and self.attempt == 1:
                return SimpleNamespace(
                    task=TaskSpec(
                        kwargs["task_id"],
                        "task",
                        metadata={
                            **kwargs["metadata"],
                            "worker_backend_failure": {
                                "count": 1,
                                "agents": ["solver"],
                                "routes": ["grok"],
                                "failure_types": ["APITimeoutError"],
                                "retryable": True,
                            },
                        },
                    ),
                    run_id=kwargs["run_id"],
                )
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "backend-same-slot-recovery"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=BackendThenSuccessApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2),
    )
    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["seed"])

    assert factory_seeds == [0]
    assert not (output / "rollout_attempts.jsonl").exists()
    incident = json.loads((output / "collection_abort.json").read_text())
    assert incident["incident_class"] == "infrastructure"


@pytest.mark.parametrize("legacy", [False, True])
def test_backend_failure_never_replays_primary_unless_explicit_legacy(tmp_path, legacy) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    attempts_by_seed: dict[int, int] = {}

    class BackendThenSuccessApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            attempts_by_seed[seed] = attempts_by_seed.get(seed, 0) + 1
            self.attempt = attempts_by_seed[seed]
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            if self.seed == 0 and self.attempt == 1:
                return SimpleNamespace(
                    task=TaskSpec(
                        kwargs["task_id"],
                        "task",
                        metadata={
                            **kwargs["metadata"],
                            "worker_backend_failure": {
                                "count": 1,
                                "agents": ["solver"],
                                "routes": ["grok"],
                                "failure_types": ["APITimeoutError"],
                                "retryable": True,
                            },
                        },
                    ),
                    run_id=kwargs["run_id"],
                )
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "opt-in-backend-retry"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=BackendThenSuccessApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2, backend_failure_retry_attempts=1, allow_legacy_whole_rollout_recovery=legacy
        ),
    )
    if not legacy:
        with pytest.raises(RuntimeError):
            runner.run(["seed"])
        assert sum(attempts_by_seed.values()) == 2
        assert all(count == 1 for count in attempts_by_seed.values())
        events = [
            json.loads(x)
            for x in (output / "rollout_scheduler_events.jsonl").read_text().splitlines()
        ]
        assert not any(x["event"] == "retry_queued" for x in events)
        attempts = [
            json.loads(x) for x in (output / "rollout_attempts.jsonl").read_text().splitlines()
        ]
        assert attempts[0]["recovery_reason"] == "backend_request_failed_no_rollout_replay"
        return
    result = runner.run(["seed"])

    assert len(result.solver_batch.samples) == 2
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert [event["success"] for event in attempts] == [False, True]
    assert attempts[0]["recovery_reason"] == "retryable_backend_failure"
    assert attempts[1]["accepted_as_primary"] is True
    assert not (output / "collection_abort.json").exists()
    # A sleeping failed slot must not hold the only worker. Its successful
    # sibling is committed first and is not executed again during recovery.
    persisted = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert persisted[0]["seed"] == 1
    assert persisted[0]["rollout_id"] != persisted[1]["rollout_id"]
    assert attempts_by_seed[1] == 1
    assert sum(attempts_by_seed.values()) == 3


def test_stateful_integrity_failure_recreates_application_for_same_slot(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    factory_seeds: list[int] = []
    attempts_by_seed: dict[int, int] = {}

    class WebShopProposer:
        def propose(self, _seed, *, task_id):
            return ProposedTask(
                TaskSpec(
                    task_id,
                    "complete the shopping task",
                    task_type="environment",
                    metadata={"dataset": "webshop"},
                ),
                response='{"prompt":"complete the shopping task"}',
            )

    class InvalidThenFreshApplication:
        def __init__(self, seed: int) -> None:
            factory_seeds.append(seed)
            attempts_by_seed[seed] = attempts_by_seed.get(seed, 0) + 1
            self.seed = seed
            self.attempt = attempts_by_seed[seed]
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            result = self.delegate.solve(*args, **kwargs)
            result.solver_result.verification = VerificationResult(0.0, False, "mock_environment")
            if self.seed == 0 and self.attempt == 1:
                result.task.metadata["worker_artifact_integrity_failure"] = {"agents": ["solver"]}
                result.task.metadata["worker_output_integrity_risks"] = ["terminal_tool_failure"]
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "stateful-fresh-session-recovery"
    result = SelfPlayRolloutRunner(
        proposer=WebShopProposer(),
        application_factory=InvalidThenFreshApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, allow_legacy_whole_rollout_recovery=True),
    ).run(["seed"])

    assert len(result.solver_batch.samples) == 2
    assert sorted(factory_seeds)[:2] == [0, 1]
    assert len(factory_seeds) == 3
    assert max(factory_seeds) not in {0, 1}
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert [item["success"] for item in attempts] == [False, True]
    assert attempts[0]["recovery_scope"] == "fresh_stateful_session"
    assert attempts[0]["recovery_reason"] == ("stateful_environment_or_action_integrity_failure")
    assert attempts[0]["seed"] == 0
    assert attempts[1]["seed"] != 0
    assert attempts[1]["policy_resampled_on_replacement"] is True
    assert all(item["same_slot_executor_seed_preserved"] is True for item in attempts)


def test_stateful_missing_terminal_result_is_an_infrastructure_incident() -> None:
    decision = _recovery_decision(
        "alfworld",
        error=EnvironmentResultIncompleteError("missing_output_agent"),
    )

    assert decision.scope.value == "none"
    assert decision.reason == "stateful_environment_terminal_result_missing"
    assert decision.infrastructure_incident is True


def test_aime_terminal_tool_failure_gets_one_full_same_slot_fallback(tmp_path, monkeypatch) -> None:
    from .helpers import install_numeric_mock_worker

    install_numeric_mock_worker(monkeypatch)
    config = load_adaptive_config(write_config(tmp_path))
    factory_seeds: list[int] = []
    attempts_by_seed: dict[int, int] = {}

    class AIMEProposer:
        def propose(self, _seed, *, task_id):
            return ProposedTask(
                TaskSpec(
                    task_id,
                    "Return the integer 20",
                    reference="20",
                    task_type="math",
                    metadata={"dataset": "aime"},
                ),
                response='{"prompt":"Return the integer 20","reference":"20"}',
            )

    class InvalidThenFreshApplication:
        def __init__(self, seed: int) -> None:
            factory_seeds.append(seed)
            attempts_by_seed[seed] = attempts_by_seed.get(seed, 0) + 1
            self.seed = seed
            self.attempt = attempts_by_seed[seed]
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            result = self.delegate.solve(*args, **kwargs)
            result.solver_result.verification = VerificationResult(1.0, True, "mock_numeric")
            if self.seed == 0 and self.attempt == 1:
                result.task.metadata["worker_artifact_integrity_failure"] = {"agents": ["solver"]}
                result.task.metadata["worker_output_integrity_risks"] = ["terminal_tool_failure"]
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "aime-full-primary-fallback"
    result = SelfPlayRolloutRunner(
        proposer=AIMEProposer(),
        application_factory=InvalidThenFreshApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, allow_legacy_whole_rollout_recovery=True),
    ).run(["seed"])

    assert len(result.solver_batch.samples) == 2
    assert sorted(factory_seeds)[:2] == [0, 1]
    assert len(factory_seeds) == 3
    assert max(factory_seeds) not in {0, 1}
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert [item["success"] for item in attempts] == [False, True]
    assert attempts[0]["recovery_scope"] == "full_primary"
    assert attempts[0]["recovery_reason"] == ("aime_selected_output_recovery_exhausted")
    assert attempts[0]["seed"] == 0
    assert attempts[1]["seed"] != 0
    assert attempts[1]["policy_resampled_on_replacement"] is True
    assert all(item["same_slot_executor_seed_preserved"] is True for item in attempts)


def test_swe_non_trainable_attempt_recovers_same_slot_seed_before_primary(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    factory_seeds: list[int] = []
    attempts_by_seed: dict[int, int] = {}
    installed_deadlines: list[float] = []

    class SWEProposer:
        def propose(self, seed, *, task_id):
            del seed
            response = json.dumps({"prompt": "Fix the repository", "reference": "runtime verifier"})
            return ProposedTask(
                task=TaskSpec(
                    task_id,
                    "Fix the repository",
                    reference="runtime verifier",
                    task_type="code",
                    metadata={"dataset": "swe_bench", "source_split": "train"},
                ),
                response=response,
                token_ids=tuple(response.encode()),
                action_mask=tuple(1 for _ in response.encode()),
            )

    class IncompleteThenGroundedApplication:
        def __init__(self, seed: int) -> None:
            factory_seeds.append(seed)
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        @property
        def runtime(self):
            return self.delegate.runtime

        def set_rollout_deadline(self, deadline) -> None:
            installed_deadlines.append(deadline.total_timeout_s)
            self.delegate.set_rollout_deadline(deadline)

        def solve(self, *args, **kwargs):
            result = self.delegate.solve(*args, **kwargs)
            result.solver_result.verification = VerificationResult(0.0, False, "mock_swe")
            attempts_by_seed[self.seed] = attempts_by_seed.get(self.seed, 0) + 1
            progress = {
                "trusted": True,
                "commit_required": True,
                "commit_ready": True,
                "state": "grounded_failure",
                "grounded_failure": {
                    "status": "grounded_failure",
                    "code": "repository_contract_mismatch",
                },
            }
            if self.seed == 0 and attempts_by_seed[self.seed] == 1:
                time.sleep(0.05)
                progress.update(
                    {"commit_ready": False, "state": "edit_required", "grounded_failure": {}}
                )
            result.task.metadata.update(
                {
                    "dataset": "swe_bench",
                    "source_split": "train",
                    "swe_output_progress": progress,
                }
            )
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "swe-same-slot-recovery"
    result = SelfPlayRolloutRunner(
        proposer=SWEProposer(),
        application_factory=IncompleteThenGroundedApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            allow_legacy_whole_rollout_recovery=True,
            workers=1,
            swe_rollout_wall_time_s=600.0,
            swe_slot_wall_time_s=600.0,
            swe_non_trainable_recovery_attempts=1,
        ),
    ).run(["seed"])

    assert len(result.solver_batch.samples) == 2
    assert factory_seeds[:2] == [0, 1]
    recovery_seed = factory_seeds[2]
    assert recovery_seed not in {0, 1}
    assert 599.0 < installed_deadlines[0] <= 600.0
    assert 599.0 < installed_deadlines[1] <= 600.0
    assert 0.0 < installed_deadlines[2] < installed_deadlines[0]
    rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    recovered = next(row for row in rows if row["rollout_id"] == "task-1-r0")
    assert recovered["seed"] == recovery_seed
    assert recovered["metadata"]["replacement_attempt"] == 1
    assert recovered["metadata"]["same_slot_seed_preserved"] is True
    assert recovered["metadata"]["same_slot_executor_seed_preserved"] is True
    assert recovered["metadata"]["policy_resampled_on_replacement"] is True
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert [event["success"] for event in attempts] == [False, True]
    assert {event["seed"] for event in attempts} == {0, recovery_seed}
    assert {event["original_rollout_seed"] for event in attempts} == {0}
    assert len({event["primary_executor_seed"] for event in attempts}) == 1
    failed_attempts = [
        json.loads(line)
        for line in (output / "rollout_attempt_trajectories.jsonl").read_text().splitlines()
    ]
    assert len(failed_attempts) == 1
    assert failed_attempts[0]["trajectory"]["metadata"]["training_eligible"] is False


def test_swe_backend_failure_aborts_before_same_slot_recovery(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    factory_seeds: list[int] = []
    attempts_by_seed: dict[int, int] = {}

    class SWEProposer:
        def propose(self, seed, *, task_id):
            del seed
            response = json.dumps({"prompt": "Fix the repository", "reference": "runtime verifier"})
            return ProposedTask(
                task=TaskSpec(
                    task_id,
                    "Fix the repository",
                    reference="runtime verifier",
                    task_type="code",
                    metadata={"dataset": "swe_bench", "source_split": "train"},
                ),
                response=response,
                token_ids=tuple(response.encode()),
                action_mask=tuple(1 for _ in response.encode()),
            )

    class BackendFailureThenGroundedApplication:
        def __init__(self, seed: int) -> None:
            factory_seeds.append(seed)
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        @property
        def runtime(self):
            return self.delegate.runtime

        def solve(self, *args, **kwargs):
            attempts_by_seed[self.seed] = attempts_by_seed.get(self.seed, 0) + 1
            if self.seed == 0 and attempts_by_seed[self.seed] == 1:
                raise WorkerBackendUnavailableError(
                    {
                        "routes": ["kiro"],
                        "failure_types": ["request_timeout"],
                        "retryable": True,
                        "counts_toward_route_circuit": True,
                    }
                )
            result = self.delegate.solve(*args, **kwargs)
            result.task.metadata.update(
                {
                    "dataset": "swe_bench",
                    "source_split": "train",
                    "swe_output_progress": {
                        "trusted": True,
                        "commit_required": True,
                        "commit_ready": True,
                        "state": "grounded_failure",
                        "grounded_failure": {
                            "status": "grounded_failure",
                            "code": "repository_contract_mismatch",
                        },
                    },
                }
            )
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "swe-backend-same-slot-recovery"
    runner = SelfPlayRolloutRunner(
        proposer=SWEProposer(),
        application_factory=BackendFailureThenGroundedApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            swe_non_trainable_recovery_attempts=1,
        ),
    )

    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["seed"])

    assert factory_seeds == [0]
    assert not (output / "solver_rollouts.jsonl").exists()
    assert not (output / "rollout_attempts.jsonl").exists()
    incidents = [
        json.loads(line)
        for line in (output / "collection_incidents.jsonl").read_text().splitlines()
    ]
    assert incidents[0]["error_type"] == WorkerBackendUnavailableError.__name__
    assert incidents[0]["counted_as_recovery_attempt"] is False


def test_worker_backend_failure_aborts_cycle_without_counting_a_recovery_attempt(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class BackendFailureApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        @property
        def solver(self):
            return SimpleNamespace(verifier=None)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "routes": ["minimax"],
                        "failure_types": ["TimeoutError"],
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    output = tmp_path / "worker-backend-hole"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=BackendFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=2, counterfactuals_per_rollout=1),
    )

    with pytest.raises(
        CollectionInfrastructureIncidentError,
        match="last committed checkpoint",
    ):
        runner.run(["seed"])

    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert len(errors) == 1
    assert {item["error_type"] for item in errors} == {WorkerBackendUnavailableError.__name__}
    assert not (output / "solver_rollouts.jsonl").exists()
    assert not (output / "relation_counterfactuals.jsonl").exists()
    assert not (output / "curriculum_observations.jsonl").exists()
    assert not (output / "proposer_batch.json").exists()
    assert not (output / "solver_batch.json").exists()
    assert not (output / "rollout_attempts.jsonl").exists()
    incidents = [
        json.loads(line)
        for line in (output / "collection_incidents.jsonl").read_text().splitlines()
    ]
    assert {item["event"] for item in incidents} == {
        "infrastructure_incident",
        "collection_aborted",
    }
    abort = json.loads((output / "collection_abort.json").read_text())
    assert abort["incident_class"] == "infrastructure"
    assert abort["counted_as_recovery_attempt"] is False
    assert abort["checkpoint_resume_policy"] == "last_committed_checkpoint_only"
    with pytest.raises(CollectionInfrastructureIncidentError, match="exact rollout resume"):
        runner.run(["seed"], resume=True)


def test_backend_failure_circuit_stops_submitting_remaining_rollouts(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class BackendFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["APIConnectionError"],
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    class LazyPool:
        def iter_map(self, function, jobs):
            return (function(job) for job in jobs)

    output = tmp_path / "backend-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=BackendFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=3,
        ),
        rollout_pool=LazyPool(),
    )

    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["seed"])

    assert len(application_calls) == 1
    events = [
        json.loads(line) for line in (output / "backend_api_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["terminal_backend_failure"]
    assert not (output / "rollout_attempts.jsonl").exists()
    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert len(errors) == 1
    assert errors[0]["error_type"] == WorkerBackendUnavailableError.__name__


def test_evaluation_backend_failure_circuit_preserves_remaining_slots(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class BackendFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["APIConnectionError"],
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    class LazyPool:
        def iter_map(self, function, jobs):
            return (function(job) for job in jobs)

    output = tmp_path / "evaluation-backend-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=BackendFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=3,
            continue_after_backend_failure_circuit=True,
        ),
        rollout_pool=LazyPool(),
    )

    with pytest.raises(InsufficientCompleteRolloutGroupsError):
        runner.run(["seed"])

    assert len(application_calls) == 5
    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert len(errors) == 5


def test_local_queue_failures_do_not_open_provider_route_circuit(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class QueueFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            detail = {
                "event": "backend_request_failure",
                "backend_failure": True,
                "origin": "local_queue",
                "kind": "queue_timeout",
                "route": "grok",
                "retryable": True,
                "counts_toward_route_circuit": False,
                "disable_route": False,
            }
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["queue_timeout"],
                        "failure_details": [detail],
                        "request_events": [detail],
                        "retryable": True,
                        "counts_toward_route_circuit": False,
                        "disable_route": False,
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    class LazyPool:
        def iter_map(self, function, jobs):
            return (function(job) for job in jobs)

    output = tmp_path / "queue-failure-no-provider-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=QueueFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            3,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=10,
        ),
        rollout_pool=LazyPool(),
    )

    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["seed"])

    assert len(application_calls) == 1
    events = [
        json.loads(line) for line in (output / "backend_api_events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "terminal_backend_failure" for event in events) == 1
    assert all(event["event"] != "backend_circuit_open" for event in events)
    assert not (output / "rollout_attempts.jsonl").exists()


def test_permanent_route_configuration_failure_opens_route_immediately(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class AuthFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            detail = {
                "event": "backend_request_failure",
                "backend_failure": True,
                "origin": "route_configuration",
                "kind": "auth_failure",
                "route": "grok",
                "retryable": False,
                "counts_toward_route_circuit": True,
                "disable_route": True,
            }
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["auth_failure"],
                        "failure_details": [detail],
                        "request_events": [detail],
                        "retryable": False,
                        "counts_toward_route_circuit": True,
                        "disable_route": True,
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    class LazyPool:
        def iter_map(self, function, jobs):
            return (function(job) for job in jobs)

    output = tmp_path / "permanent-route-failure"
    health_path = tmp_path / "route-health.json"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=AuthFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            backend_failure_route_threshold=3,
            backend_failure_total_threshold=3,
            route_health_path=health_path,
            worker_runtime_routes=("grok",),
        ),
        rollout_pool=LazyPool(),
    )

    with pytest.raises(RuntimeError):
        runner.run(["seed"])

    assert len(application_calls) == 1
    health = json.loads(health_path.read_text())
    assert health["routes"]["grok"]["status"] == "open"
    assert health["routes"]["grok"]["last_failure_type"] == "auth_failure"


def test_backend_retry_exhaustion_preserves_permanent_route_circuit_policy() -> None:
    detail = {
        "event": "backend_request_failure",
        "origin": "route_configuration",
        "kind": "model_not_found",
        "route": "gemini",
        "retryable": False,
        "counts_toward_route_circuit": True,
        "disable_route": True,
    }
    error = BackendRetryExhaustedError(
        {
            "routes": ["gemini"],
            "failure_types": ["model_not_found"],
            "failure_details": [detail],
            "request_events": [detail],
            "retryable": False,
            "counts_toward_route_circuit": True,
            "disable_route": True,
        }
    )

    assert error.route_policy("gemini") == (True, True)
    assert error.disable_route is True
    assert error.request_events == (detail,)


def test_backend_circuit_cancels_running_rollouts_without_starting_replacements(
    tmp_path,
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []
    first_window_started = threading.Barrier(4)

    class ConcurrentFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)
            self.seed = seed
            self.deadline = None
            self.solver = SimpleNamespace(active_canvas=None, verifier=None)

        def set_rollout_deadline(self, deadline) -> None:
            self.deadline = deadline

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            first_window_started.wait(timeout=2)
            if self.seed < 2:
                task = TaskSpec(
                    task_id,
                    "task",
                    metadata={
                        **metadata,
                        "worker_backend_failure": {
                            "count": 1,
                            "agents": ["solver"],
                            "routes": ["grok"],
                            "failure_types": ["APIConnectionError"],
                        },
                    },
                )
                return SimpleNamespace(task=task, run_id=run_id)
            while True:
                assert self.deadline is not None
                self.deadline.check("concurrent_test")
                time.sleep(0.005)

        def close(self) -> None:
            return None

    output = tmp_path / "concurrent-backend-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=ConcurrentFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            workers=4,
            replacement_rollouts_per_task=2,
            rollout_wall_time_s=10.0,
            rollout_no_progress_time_s=10.0,
            request_wall_time_s=10.0,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=3,
            non_swe_recovery_attempts=0,
        ),
        rollout_pool=ThreadRolloutPool(workers=4),
    )

    started = time.monotonic()
    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["seed"])

    assert time.monotonic() - started < 2.0
    assert set(application_calls) == {0, 1, 2, 3}
    assert all(seed < 1_000_003 for seed in application_calls)


def test_success_resets_window_consecutive_backend_failure_count(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class FailureSuccessFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)
            self.solver = SimpleNamespace(active_canvas=None, verifier=None)

        def set_rollout_deadline(self, deadline) -> None:
            setter = getattr(self.delegate, "set_rollout_deadline", None)
            if callable(setter):
                setter(deadline)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            if self.seed == 1:
                return self.delegate.solve(
                    prompt,
                    task_id=task_id,
                    task_type=task_type,
                    reference=reference,
                    run_id=run_id,
                    metadata=metadata,
                )
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["default"],
                        "failure_types": ["APIConnectionError"],
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "consecutive-circuit-reset"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=FailureSuccessFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            3,
            replacement_rollouts_per_task=0,
            non_swe_recovery_attempts=0,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=2,
        ),
    )

    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["seed"])

    assert application_calls == [0]
    events = [
        json.loads(line) for line in (output / "backend_api_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["terminal_backend_failure"]


def test_deadline_profile_uses_stateless_stateful_and_aime_overrides() -> None:
    runner = object.__new__(SelfPlayRolloutRunner)
    runner.config = SelfPlayRunConfig()

    def proposal(dataset: str) -> ProposedTask:
        return ProposedTask(
            TaskSpec("task", "prompt", metadata={"dataset": dataset}),
            response="{}",
        )

    assert runner._deadline_profile(proposal("nq_open")) == (900.0, 180.0, {})
    assert runner._deadline_profile(proposal("webshop")) == (900.0, 180.0, {})
    assert runner._deadline_profile(proposal("alfworld")) == (900.0, 180.0, {})
    assert runner._slot_wall_time_s(proposal("webshop")) == 900.0
    assert runner._slot_wall_time_s(proposal("alfworld")) == 900.0
    assert runner._deadline_profile(proposal("swe_bench")) == (900.0, 180.0, {})
    assert runner._deadline_profile(proposal("swebench")) == (900.0, 180.0, {})
    assert runner._deadline_profile(proposal("aime")) == (
        900.0,
        240.0,
        {"grok": 300.0},
    )
    assert runner._deadline_profile_payload(proposal("swe_bench")) == {
        "dataset": "swe_bench",
        "total_timeout_s": 900.0,
        "control_idle_timeout_s": 120.0,
        "request_timeout_s": 180.0,
        "request_timeout_overrides_s": {},
    }


def test_backend_circuit_cannot_train_a_truncated_later_window(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class MixedApplication:
        def __init__(self, seed: int) -> None:
            self.delegate = create_adaptive_application(config, mock=True)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            if task_id == "task-1":
                return self.delegate.solve(
                    prompt,
                    task_id=task_id,
                    task_type=task_type,
                    reference=reference,
                    run_id=run_id,
                    metadata=metadata,
                )
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["APIConnectionError"],
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "later-window-backend-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=MixedApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            task_window=1,
            backend_failure_route_threshold=1,
            backend_failure_total_threshold=1,
        ),
    )

    with pytest.raises(CollectionInfrastructureIncidentError):
        runner.run(["safe first window", "backend failure second window"])

    assert (output / "collection_abort.json").exists()
    assert not (output / "proposer_batch.json").exists()
    assert not (output / "solver_batch.json").exists()


@pytest.mark.parametrize(
    ("evaluation_only", "non_trainable_rollout_ids"),
    [(False, []), (True, ["task-1-r0"])],
)
def test_runner_resumes_only_the_exact_missing_rollout_id(
    tmp_path, evaluation_only, non_trainable_rollout_ids
) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "hole"
    factory_calls: list[int] = []

    def application_factory(seed):
        factory_calls.append(seed)
        return create_adaptive_application(config, mock=True)

    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            allow_exact_rollout_resume=True,
            evaluation_only=evaluation_only,
            frontier_reverify_fraction=0.0,
        ),
    )
    runner.run(["seed"])
    rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    kept = [row for row in rows if row["rollout_id"] != "task-1-r2"]
    (output / "solver_rollouts.jsonl").write_text(
        "\n".join(json.dumps(row) for row in kept) + "\n", encoding="utf-8"
    )
    (output / "quarantined_groups.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "status": "quarantined",
                "reason": "incomplete_rollout_group",
                "missing_rollout_ids": ["task-1-r2"],
                "non_trainable_rollout_ids": non_trainable_rollout_ids,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "rollout_errors.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "rollout_id": "task-1-r2",
                "error_type": "BackendRetryExhaustedError",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    factory_calls.clear()

    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            5,
            allow_exact_rollout_resume=True,
            evaluation_only=evaluation_only,
            frontier_reverify_fraction=0.0,
        ),
    )
    runner.run(["seed"], resume=True)

    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert factory_calls == [2]
    assert sorted(persisted) == [f"task-1-r{index}" for index in range(5)]
    assert len(persisted) == len(set(persisted))
    quarantine_events = [
        json.loads(line) for line in (output / "quarantined_groups.jsonl").read_text().splitlines()
    ]
    assert quarantine_events[-1]["status"] == "reopened_for_exact_resume"
    assert quarantine_events[-1]["reason"] == (
        "evaluation_missing_slot_recovery" if evaluation_only else "transient_backend_recovery"
    )


def test_runner_allows_attested_attribution_classifier_repair_resume(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "attested-classifier-repair"

    def application_factory(seed):
        return create_adaptive_application(config, mock=True)

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1),
    ).run(["seed"])
    rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    (output / "solver_rollouts.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    (output / "collection_abort.json").write_text(
        json.dumps({"incident_class": "attribution_uncertain_recovery_exhausted"}) + "\n",
        encoding="utf-8",
    )
    (output / "collection_repair_attestation.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "incident_class": "attribution_uncertain_recovery_exhausted",
                "reclassified_rollout_ids": ["task-1-r0"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            allow_exact_rollout_resume=True,
            allow_attribution_classifier_repair_resume=True,
        ),
    ).run(["seed"], resume=True)

    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert sorted(persisted) == ["task-1-r0", "task-1-r1"]


def test_runner_allows_training_only_resume_after_repaired_collection_abort(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "completed-training-resume"
    factory_calls: list[int] = []

    def application_factory(seed):
        factory_calls.append(seed)
        return create_adaptive_application(config, mock=True)

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1),
    ).run(["seed"])
    durable_names = (
        "solver_rollouts.jsonl",
        "frontier_reverification.json",
        "frontier_scores.json",
        "proposer_batch.json",
        "solver_batch.json",
        "snapshots.json",
    )
    durable_before = {
        name: (output / name).read_bytes() for name in durable_names if (output / name).is_file()
    }
    (output / "collection_abort.json").write_text(
        json.dumps(
            {
                "incident_class": "attribution_uncertain_recovery_exhausted",
                "task_id": "task-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    factory_calls.clear()

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1),
    ).run(["seed"], resume=True)

    assert factory_calls == []
    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert sorted(persisted) == ["task-1-r0", "task-1-r1"]
    assert {name: (output / name).read_bytes() for name in durable_before} == durable_before
    incidents = [
        json.loads(line)
        for line in (output / "collection_incidents.jsonl").read_text().splitlines()
    ]
    assert incidents[-1]["event"] == "completed_collection_reopened_for_training_only_resume"
    pipeline = [
        json.loads(line) for line in (output / "pipeline_events.jsonl").read_text().splitlines()
    ]
    assert pipeline[-1]["event"] == "completed_postcollection_artifacts_reused"
    assert pipeline[-1]["frontier_reverification_reexecuted"] is False


def test_runner_allows_postcollection_resume_after_complete_primary_collection(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "completed-primary-postcollection-resume"
    factory_calls: list[int] = []

    def application_factory(seed):
        factory_calls.append(seed)
        return create_adaptive_application(config, mock=True)

    run_config = SelfPlayRunConfig(2, workers=1, frontier_reverify_fraction=0.0)
    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=run_config,
    ).run(["seed"])
    for name in (
        "frontier_scores.json",
        "proposer_batch.json",
        "solver_batch.json",
        "snapshots.json",
    ):
        (output / name).unlink()
    (output / "collection_abort.json").write_text(
        json.dumps({"incident_class": "infrastructure", "task_id": "task-1"}) + "\n",
        encoding="utf-8",
    )
    factory_calls.clear()

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=run_config,
    ).run(["seed"], resume=True)

    assert factory_calls == []
    assert len((output / "solver_rollouts.jsonl").read_text().splitlines()) == 2
    incidents = [
        json.loads(line)
        for line in (output / "collection_incidents.jsonl").read_text().splitlines()
    ]
    assert (
        incidents[-1]["event"] == "completed_primary_collection_reopened_for_postcollection_resume"
    )


def test_runner_allows_attested_infrastructure_exact_missing_resume(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "attested-infrastructure-repair"
    factory_calls: list[int] = []

    def application_factory(seed):
        factory_calls.append(seed)
        return create_adaptive_application(config, mock=True)

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1),
    ).run(["seed"])
    rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    (output / "solver_rollouts.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    (output / "collection_abort.json").write_text(
        json.dumps({"incident_class": "infrastructure", "rollout_id": "task-1-r1"}) + "\n",
        encoding="utf-8",
    )
    factory_calls.clear()

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            allow_exact_rollout_resume=True,
            allow_infrastructure_repair_resume=True,
        ),
    ).run(["seed"], resume=True)

    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert factory_calls == [1]
    assert sorted(persisted) == ["task-1-r0", "task-1-r1"]


def test_runner_recollects_every_missing_sibling_after_uncertain_group_abort(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "uncertain-full-group-recollection"
    factory_calls: list[int] = []

    def application_factory(seed):
        factory_calls.append(seed)
        return create_adaptive_application(config, mock=True)

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1),
    ).run(["seed"])
    (output / "solver_rollouts.jsonl").write_text("", encoding="utf-8")
    (output / "collection_abort.json").write_text(
        json.dumps(
            {
                "incident_class": "attribution_uncertain_recovery_exhausted",
                "task_id": "task-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "quarantined_groups.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "status": "quarantined",
                "reason": "incomplete_rollout_group",
                "missing_rollout_ids": ["task-1-r0", "task-1-r1"],
                "non_trainable_rollout_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    factory_calls.clear()

    SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=application_factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            allow_exact_rollout_resume=True,
            allow_uncertain_group_recollection_resume=True,
            policy_sampling_attempt_offsets={"task-1": 2},
        ),
    ).run(["seed"], resume=True)

    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert factory_calls == [
        _rollout_sampling_seed(0, "task-1", 0, 2),
        _rollout_sampling_seed(0, "task-1", 1, 2),
    ]
    assert sorted(persisted) == ["task-1-r0", "task-1-r1"]


def test_runner_deduplicates_identical_persisted_rollout_ids(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    output = tmp_path / "duplicate"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2),
    )
    runner.run(["seed"])
    path = output / "solver_rollouts.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")

    resumed = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: (_ for _ in ()).throw(
            AssertionError("complete rollout must not run again")
        ),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2),
    )
    result = resumed.run(["seed"], resume=True)
    assert len(result.solver_batch.samples) == 2


def test_runner_closes_application_when_rollout_raises(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    closed: list[bool] = []

    class FailingApplication:
        def solve(self, *_args, **_kwargs):
            raise RuntimeError("failed rollout")

        def close(self) -> None:
            closed.append(True)

    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: FailingApplication(),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "failed",
        config=SelfPlayRunConfig(2),
    )
    proposal = _MockProposer().propose("seed", task_id="task-1")

    with pytest.raises(RuntimeError, match="failed rollout"):
        runner._collect((proposal, 0))

    assert closed == [True]


def test_runner_records_problem_extraction_success_and_failure(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class PartlyInvalidProposer:
        def propose(self, seed: str, *, task_id: str) -> ProposedTask:
            if seed == "bad":
                raise ValueError("proposer response must contain a JSON object")
            return _MockProposer().propose(seed, task_id=task_id)

    output = tmp_path / "extraction"
    result = SelfPlayRolloutRunner(
        proposer=PartlyInvalidProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2),
    ).run(["bad", "good"])
    assert result.proposal_extraction == {
        "attempts": 2,
        "successes": 1,
        "failures": 1,
        "success_rate": 0.5,
        "failure_kinds": {"json_parse_failure": 1},
    }
    attempts = [
        json.loads(line) for line in (output / "proposal_attempts.jsonl").read_text().splitlines()
    ]
    assert [item["success"] for item in attempts] == [False, True]
    assert json.loads((output / "proposal_extraction.json").read_text())["success_rate"] == 0.5


def _binary_relation_decision(
    source: str,
    target: str,
    choice: str,
    probability_on: float,
) -> dict[str, object]:
    return {
        "phase": "choice",
        "source": source,
        "target": target,
        "relation_type": "bidirectional",
        "choice": choice,
        "chosen_present": choice == "on",
        "policy": {
            "choice": choice,
            "probabilities": {"off": 1.0 - probability_on, "on": probability_on},
            "log_probabilities": {"off": -0.7, "on": -0.68},
            "token_ids": {"off": 101, "on": 102},
        },
    }


def test_relation_scheduler_uses_same_seed_counterfactual_prefix() -> None:
    graph = MultiAgentGraph()
    graph.add_agent("a")
    graph.set_prompt("a", "a")
    graph.add_agent("b")
    graph.set_prompt("b", "b")
    prefix = graph.clone()
    graph.set_relation("a", "b", "bidirectional")
    trace = ExecutionTrace(
        "run",
        TaskSpec("task", "prompt"),
        [
            TraceEvent(
                3,
                "canvas_step",
                {
                    "accepted": True,
                    "raw_action": "on",
                    "graph_before": prefix.to_dict(),
                    "graph": graph.to_dict(),
                    "relation_decision": _binary_relation_decision("a", "b", "on", 0.6),
                },
            )
        ],
        graph.to_dict(),
    )
    decision = schedule_relation_decisions(trace)[0]
    seen: list[int] = []
    credit = evaluate_relation_decision(
        decision,
        rollout_id="r1",
        seed=11,
        evaluate=lambda candidate, seed: (
            seen.append(seed) or float(bool(candidate.bidirectional_edges))
        ),
    )
    assert credit and credit.q_present == 1.0 and credit.q_absent == 0.0
    assert seen == [11, 11]


def test_relation_probe_is_selected_after_rollout_then_replayed_from_prefix() -> None:
    prefix = MultiAgentGraph()
    for agent_id in ("a", "b"):
        prefix.add_agent(agent_id)
        prefix.set_prompt(agent_id, agent_id)
    prefix.set_output("a")

    with_relation = prefix.clone()
    with_relation.set_relation("a", "b", "bidirectional")
    final = with_relation.clone()
    final.set_output("b")

    artifact_a = AgentArtifact("artifact-a", "a", "answer-a").to_dict()
    artifact_b = AgentArtifact("artifact-b", "b", "answer-b").to_dict()
    trace = ExecutionTrace(
        "prefix-replay",
        TaskSpec("task", "prompt"),
        [
            TraceEvent(
                0,
                "canvas_step",
                {
                    "accepted": True,
                    "raw_action": '{"action":"set_output","target":"a"}',
                    "graph_before": prefix.to_dict(),
                    "graph": prefix.to_dict(),
                    "execution": {"artifacts": {"a": artifact_a, "b": artifact_b}},
                    "protocol_recovery": False,
                    "director_turn_index": 0,
                },
            ),
            TraceEvent(
                1,
                "canvas_step",
                {
                    "accepted": True,
                    "raw_action": "on",
                    "graph_before": prefix.to_dict(),
                    "graph": with_relation.to_dict(),
                    "execution": {"artifacts": {"a": artifact_a, "b": artifact_b}},
                    "protocol_recovery": False,
                    "director_turn_index": 1,
                    "relation_decision": _binary_relation_decision("a", "b", "on", 0.6),
                },
            ),
            TraceEvent(
                2,
                "canvas_step",
                {
                    "accepted": True,
                    "raw_action": '{"action":"set_output","target":"b"}',
                    "graph_before": with_relation.to_dict(),
                    "graph": final.to_dict(),
                    "execution": {"artifacts": {"a": artifact_a, "b": artifact_b}},
                    "protocol_recovery": False,
                    "director_turn_index": 2,
                },
            ),
        ],
        final.to_dict(),
    )

    decision = schedule_relation_decisions(
        trace,
        limit=1,
        seed=9,
        action_token_spans=((0, 1), (1, 2), (2, 3)),
    )[0]
    assert not MultiAgentGraph.from_dict(decision.graph_prefix).bidirectional_edges
    assert len(decision.suffix_events) == 1
    assert decision.prefix_artifacts["a"]["answer"] == "answer-a"

    branches = []

    def evaluate_full(graph, _seed):
        branches.append(graph.clone())
        return float(bool(graph.bidirectional_edges))

    credit = evaluate_relation_decision(
        decision,
        rollout_id="full-replay-r0",
        seed=23,
        evaluate=evaluate_full,
        action_token_span=(1, 2),
    )
    assert credit.q_absent == 0 and credit.q_present == 1
    assert all(branch.output_agent == "b" for branch in branches)
    assert not branches[0].bidirectional_edges
    assert branches[1].bidirectional_edges == {("a", "b")}
    assert decision.prefix_artifacts["a"]["answer"] == "answer-a"


def test_relation_scheduler_excludes_choice_whose_endpoint_is_later_deleted() -> None:
    prefix = MultiAgentGraph()
    for agent_id in ("a", "b"):
        prefix.add_agent(agent_id)
        prefix.set_prompt(agent_id, agent_id)
    with_relation = prefix.clone()
    with_relation.set_relation("a", "b", "bidirectional")
    final = with_relation.clone()
    final.delete_agent("b")
    final.set_output("a")
    trace = ExecutionTrace(
        "deleted-relation-endpoint",
        TaskSpec("task", "prompt"),
        [
            TraceEvent(
                0,
                "canvas_step",
                {
                    "accepted": True,
                    "raw_action": "on",
                    "graph_before": prefix.to_dict(),
                    "graph": with_relation.to_dict(),
                    "protocol_recovery": False,
                    "director_turn_index": 0,
                    "relation_decision": _binary_relation_decision("a", "b", "on", 0.6),
                },
            ),
            TraceEvent(
                1,
                "canvas_step",
                {
                    "accepted": True,
                    "raw_action": '{"action":"delete_agent","target":"b"}',
                    "graph_before": with_relation.to_dict(),
                    "graph": final.to_dict(),
                    "protocol_recovery": False,
                    "director_turn_index": 1,
                },
            ),
        ],
        final.to_dict(),
    )

    assert (
        schedule_relation_decisions(
            trace,
            limit=1,
            action_token_spans=((0, 1), (1, 2)),
        )
        == []
    )


def test_full_evaluator_rejects_prefix_artifact_reuse() -> None:
    application = object.__new__(AdaptiveSolverApplication)
    with pytest.raises(ValueError, match="local/prefix evaluation retired"):
        application.evaluate_graph(
            TaskSpec("task", "public"),
            MultiAgentGraph(),
            seed=31,
            initial_artifacts={"a": {"answer": "POISON"}},
            dirty_agents={"b"},
        )


def test_relation_scheduler_selects_closest_half_trainable_binary_choice() -> None:
    graph = MultiAgentGraph()
    for agent_id in ("a", "b", "c"):
        graph.add_agent(agent_id)
        graph.set_prompt(agent_id, agent_id)
    graph.set_relation("a", "b", "bidirectional")
    graph.set_relation("a", "c", "bidirectional")
    events = [
        TraceEvent(
            0,
            "canvas_step",
            {
                "accepted": True,
                "raw_action": '{"action":"set_output","target":"a"}',
                "protocol_recovery": False,
                "director_turn_index": 0,
            },
        ),
        TraceEvent(
            10,
            "canvas_step",
            {
                "accepted": True,
                "raw_action": (
                    '{"action":"set_relation","source":"b","target":"c","relation":"bidirectional"}'
                ),
                "protocol_recovery": True,
                "director_turn_index": None,
            },
        ),
        TraceEvent(
            11,
            "canvas_step",
            {
                "accepted": True,
                "raw_action": "on",
                "protocol_recovery": False,
                "director_turn_index": 1,
                "graph_before": graph.to_dict(),
                "graph": graph.to_dict(),
                "relation_decision": _binary_relation_decision("a", "b", "on", 0.51),
            },
        ),
        TraceEvent(
            12,
            "canvas_step",
            {
                "accepted": True,
                "raw_action": "on",
                "protocol_recovery": False,
                "director_turn_index": 2,
                "graph_before": graph.to_dict(),
                "graph": graph.to_dict(),
                "relation_decision": _binary_relation_decision("a", "c", "on", 0.52),
            },
        ),
    ]
    trace = ExecutionTrace("random-selection", TaskSpec("task", "prompt"), events, graph.to_dict())

    first = schedule_relation_decisions(
        trace,
        limit=1,
        seed=17,
        action_token_spans=((0, 1), None, (2, 3)),
    )
    second = schedule_relation_decisions(
        trace,
        limit=1,
        seed=17,
        action_token_spans=((0, 1), None, (2, 3)),
    )

    assert first == second
    assert len(first) == 1
    assert first[0].action_index == 2
    assert (first[0].source, first[0].target) == ("a", "c")


def test_counterfactual_failure_keeps_completed_primary_rollout(tmp_path, monkeypatch) -> None:
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="exact_match")
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "counterfactual-failure",
        config=SelfPlayRunConfig(2, counterfactuals_per_rollout=1),
    )
    proposal = _MockProposer().propose("seed", task_id="task-1")

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.schedule_relation_decisions",
        lambda *_args, **_kwargs: [RelationDecision(0, "a", "b", 0.5, {}, True)],
    )

    def fail_probe(*_args, **_kwargs):
        raise RuntimeError("optional probe failed")

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.evaluate_relation_decision",
        fail_probe,
    )

    rollout, credits = runner._collect((proposal, 0))

    assert not credits
    assert rollout.trajectory.rollout_id == "task-1-r0"
    errors = rollout.trajectory.metadata["relation_counterfactual_errors"]
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["message"] == "optional probe failed"


def test_primary_rollouts_are_durable_before_counterfactual_interruption(
    tmp_path, monkeypatch
) -> None:
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="exact_match")
    output = tmp_path / "primary-before-counterfactual"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1, counterfactuals_per_rollout=1),
    )

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.schedule_relation_decisions",
        lambda *_args, **_kwargs: [RelationDecision(0, "a", "b", 0.5, {}, True)],
    )

    def interrupt_probe(*_args, rollout_id, **_kwargs):
        persisted = {
            json.loads(line)["rollout_id"]
            for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
        }
        assert rollout_id in persisted
        raise KeyboardInterrupt("interrupt during optional probe")

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.evaluate_relation_decision",
        interrupt_probe,
    )

    with pytest.raises(KeyboardInterrupt, match="optional probe"):
        runner.run(["seed"])

    persisted = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert sorted(persisted) == ["task-1-r0", "task-1-r1"]
    assert json.loads((output / "progress.json").read_text())["rollouts"] == 2
    assert not (output / "proposer_batch.json").exists()
    assert not (output / "solver_batch.json").exists()

    resumed = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: (_ for _ in ()).throw(
            AssertionError("durable primary must not be recollected")
        ),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(2, workers=1, counterfactuals_per_rollout=1),
    )
    resumed.run(["seed"], resume=True)

    persisted_after_resume = [
        json.loads(line)["rollout_id"]
        for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert persisted_after_resume == persisted
    assert (output / "proposer_batch.json").exists()
    assert (output / "solver_batch.json").exists()


def test_counterfactual_pipeline_overlaps_next_primary_and_preserves_results(
    tmp_path, monkeypatch
) -> None:
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="exact_match")
    pipeline_output = tmp_path / "pipeline"
    serial_output = tmp_path / "serial"
    next_primary_started = threading.Event()

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.schedule_relation_decisions",
        lambda *_args, **_kwargs: [RelationDecision(0, "a", "b", 0.5, {}, True)],
    )
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.evaluate_relation_decision",
        lambda *_args, **_kwargs: None,
    )

    pipeline_runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=pipeline_output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            task_window=1,
            counterfactuals_per_rollout=1,
            pipeline_counterfactuals=True,
            counterfactual_workers=1,
        ),
    )
    original_primary = pipeline_runner._collect_primary
    original_counterfactual = pipeline_runner._collect_counterfactual

    def collect_primary(job, **kwargs):
        if job[0].task.task_id == "task-2":
            next_primary_started.set()
        return original_primary(job, **kwargs)

    def collect_counterfactual(primary):
        if primary.proposal.task.task_id == "task-1":
            assert next_primary_started.wait(timeout=2)
        return original_counterfactual(primary)

    monkeypatch.setattr(pipeline_runner, "_collect_primary", collect_primary)
    monkeypatch.setattr(pipeline_runner, "_collect_counterfactual", collect_counterfactual)
    pipeline_result = pipeline_runner.run(["one", "two", "three"])

    serial_result = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=serial_output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            task_window=1,
            counterfactuals_per_rollout=1,
        ),
    ).run(["one", "two", "three"])

    events = [
        json.loads(line)
        for line in (pipeline_output / "pipeline_events.jsonl").read_text().splitlines()
    ]
    by_event = {
        (item["event"], item["window_index"]): item["timestamp"]
        for item in events
        if "window_index" in item
    }
    assert (
        by_event[("primary_window_started", 1)] < by_event[("counterfactual_rollout_completed", 0)]
    )
    assert sum(item["event"] == "window_finalized" for item in events) == 3
    assert [item.rollout_id for item in pipeline_result.solver_batch.samples] == [
        item.rollout_id for item in serial_result.solver_batch.samples
    ]
    assert [item.reward for item in pipeline_result.solver_batch.samples] == [
        item.reward for item in serial_result.solver_batch.samples
    ]
    manifest = json.loads((pipeline_output / "run_manifest.json").read_text())
    assert manifest["rollout_batching"]["counterfactual_pipeline"] == (
        "global_queue_as_each_primary_rollout_completes"
    )
    assert manifest["rollout_batching"]["final_training_barrier"] == (
        "wait_for_bounded_counterfactual_completion"
    )


def test_dynamic_counterfactual_pipeline_overlaps_remaining_primary_rollouts(
    tmp_path, monkeypatch
) -> None:
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="exact_match")
    next_primary_started = threading.Event()

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.schedule_relation_decisions",
        lambda *_args, **_kwargs: [RelationDecision(0, "a", "b", 0.5, {}, True)],
    )
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.evaluate_relation_decision",
        lambda *_args, **_kwargs: None,
    )

    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda _seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "dynamic",
        config=SelfPlayRunConfig(
            2,
            workers=1,
            task_window=3,
            task_execution_window=1,
            task_scheduling_policy="frozen_manifest_dynamic",
            max_active_task_groups=3,
            require_all_proposals=True,
            counterfactuals_per_rollout=1,
            pipeline_counterfactuals=True,
            counterfactual_workers=1,
        ),
    )
    original_primary = runner._collect_primary
    original_counterfactual = runner._collect_counterfactual

    def collect_primary(job, **kwargs):
        if job[0].task.task_id == "task-2":
            next_primary_started.set()
        return original_primary(job, **kwargs)

    def collect_counterfactual(primary):
        if primary.proposal.task.task_id == "task-1":
            assert next_primary_started.wait(timeout=2)
        return original_counterfactual(primary)

    monkeypatch.setattr(runner, "_collect_primary", collect_primary)
    monkeypatch.setattr(runner, "_collect_counterfactual", collect_counterfactual)
    runner.run(["one", "two", "three"])

    events = [
        json.loads(line)
        for line in (tmp_path / "dynamic" / "pipeline_events.jsonl").read_text().splitlines()
    ]
    streamed = [event for event in events if event["event"] == "counterfactual_rollout_started"]
    assert streamed
    assert all(event["scheduling"] == "after_primary_rollout" for event in streamed)
    boundaries = [event for event in events if event["event"] == "counterfactual_update_boundary"]
    assert len(boundaries) == 1
    assert boundaries[0]["submitted"] == 6
    manifest = json.loads((tmp_path / "dynamic" / "run_manifest.json").read_text())
    assert manifest["rollout_batching"]["counterfactual_pipeline"] == (
        "global_queue_as_each_primary_rollout_completes"
    )


def test_counterfactual_pair_uses_one_independent_shared_deadline(tmp_path, monkeypatch) -> None:
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="exact_match")
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=tmp_path / "independent-pair-budget",
        config=SelfPlayRunConfig(
            2,
            rollout_wall_time_s=900,
            stateful_rollout_wall_time_s=900,
            swe_rollout_wall_time_s=900,
            counterfactual_pair_wall_time_s=17,
        ),
    )
    proposal = _MockProposer().propose("seed", task_id="task-1")
    primary = runner._collect_primary((proposal, 0))
    primary.primary_duration_s = 899.0
    primary.rollout.trajectory.metadata["action_token_spans"] = ((0, 1),)
    primary.decisions = (RelationDecision(0, "a", "b", 0.5, {}, True),)
    deadlines = []

    def branch_factory(seed):
        branch = create_adaptive_application(config, mock=True)
        install = branch.set_rollout_deadline

        def capture(deadline):
            deadlines.append(deadline)
            install(deadline)

        branch.set_rollout_deadline = capture
        branch.evaluate_graph = lambda *_args, **_kwargs: 1.0
        return branch

    runner.application_factory = branch_factory

    def evaluate_pair(_decision, *, seed, evaluate, **_kwargs):
        evaluate(primary.rollout.graph, seed)
        evaluate(primary.rollout.graph, seed)
        return None

    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.evaluate_relation_decision",
        evaluate_pair,
    )
    rollout, credits = runner._collect_counterfactual(primary)

    assert credits == []
    assert len(deadlines) == 2
    assert deadlines[0] is deadlines[1]
    assert deadlines[0].total_timeout_s == 17
    assert deadlines[0].absolute_wall_timeout_s == 17
    assert rollout.trajectory.metadata["relation_counterfactual_budget"] == {
        "scope": "independent_off_on_pair",
        "pair_wall_budget_s": 17,
        "primary_duration_deducted": False,
    }


def test_update_boundary_waits_for_bounded_running_credit_before_training(
    tmp_path, monkeypatch
) -> None:
    config = replace(load_adaptive_config(write_config(tmp_path)), verifier="exact_match")
    output = tmp_path / "bounded-counterfactual-boundary"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: create_adaptive_application(config, mock=True),
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            task_window=1,
            counterfactuals_per_rollout=1,
            pipeline_counterfactuals=True,
            counterfactual_workers=1,
            rollout_wall_time_s=900,
            stateful_rollout_wall_time_s=900,
            swe_rollout_wall_time_s=900,
            counterfactual_pair_wall_time_s=900,
            frontier_reverify_fraction=0,
        ),
    )
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.selfplay_runtime.schedule_relation_decisions",
        lambda *_args, **_kwargs: [RelationDecision(0, "a", "b", 0.5, {}, True)],
    )
    running_started = threading.Event()
    running_finished = threading.Event()
    release_running_worker = threading.Event()

    def collect_counterfactual(primary):
        rollout_id = primary.rollout.trajectory.rollout_id
        if rollout_id.endswith("r0"):
            return primary.rollout, [
                {
                    "rollout_id": rollout_id,
                    "action_index": 0,
                    "source": "a",
                    "target": "b",
                    "relation": "delegation",
                    "q_absent": 0.0,
                    "q_present": 1.0,
                    "advantage_absent": -0.5,
                    "advantage_present": 0.5,
                    "seed": primary.executor_seed,
                }
            ]
        running_started.set()
        assert release_running_worker.wait(timeout=10)
        running_finished.set()
        return primary.rollout, [
            {
                "rollout_id": rollout_id,
                "action_index": 0,
                "source": "a",
                "target": "b",
                "relation": "delegation",
                "q_absent": 0.0,
                "q_present": 1.0,
                "advantage_absent": -0.5,
                "advantage_present": 0.5,
                "seed": primary.executor_seed,
            }
        ]

    monkeypatch.setattr(runner, "_collect_counterfactual", collect_counterfactual)
    original_finalize = runner._finalize_window

    def finalize_after_running_starts(*args, **kwargs):
        assert running_started.wait(timeout=2)
        threading.Timer(0.25, release_running_worker.set).start()
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(runner, "_finalize_window", finalize_after_running_starts)
    result = runner.run(["seed"])
    assert running_finished.is_set()
    assert len(result.solver_batch.samples) == 2
    credits = [
        json.loads(line)
        for line in (output / "relation_counterfactuals.jsonl").read_text().splitlines()
    ]
    assert [row["rollout_id"] for row in credits] == ["task-1-r0", "task-1-r1"]
    boundary = json.loads((output / "counterfactual_update_boundary.json").read_text())
    assert boundary["submitted"] == 2
    assert boundary["completed_before_boundary"] == 1
    assert boundary["unfinished_at_boundary"] == 1
    assert boundary["completed_after_wait"] == 2
    assert boundary["cancelled_or_discarded"] == 0
    assert boundary["admitted_credit_count"] == 2
    assert boundary["waited_for_incomplete"] is True
    assert not (output / "relation_counterfactual_cancellations.jsonl").exists()
    primary_rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert len(primary_rows) == 2
    assert all(
        row["metadata"]["deadline_profile"]["total_timeout_s"] == 900 for row in primary_rows
    )


def test_evaluation_adapter_normalizes_both_systems(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve("task", task_id="q1")
    adaptive = from_adaptive_result(result)
    flowsteer = from_flowsteer_trajectory(
        {"task_id": "q1", "output": "answer", "reward": 1, "seed": 2}
    )
    assert [item.system for item in (adaptive, flowsteer)] == [
        "selfplay_graph_flowsteer",
        "flowsteer",
    ]


def test_exhausted_webshop_missing_output_is_attributed_to_worker_without_stage() -> None:
    failure = _runtime_owned_model_policy_failure(
        "webshop",
        training_exclusion_reasons=[
            "not_finished",
            "invalid_final_graph",
            "execution_incomplete",
            "execution_budget_exceeded",
        ],
        verification=VerificationResult(0.0, False, "webshop_environment"),
        worker_backend_failure=None,
        worker_artifact_integrity={
            "shopper": {
                "webshop_progress": {
                    "trusted": True,
                    "state": "completed",
                    "commit_ready": False,
                    "environment_owner": "shopper",
                    "environment_access": "mutable_owner",
                }
            }
        },
        worker_artifact_integrity_failure=None,
        swe_output_progress={},
        alfworld_output_progress={},
        webshop_output_progress={},
        swe_environment_result={},
        swe_infrastructure_failure=None,
        swe_synthetic_evaluation=False,
        swe_non_train_split=False,
        stateful_environment_result={
            "done": False,
            "steps": 0,
            "termination_reason": "missing_output_agent",
            "budget_truncated": False,
        },
        admit_stateful_policy_failure_terminal=True,
    )

    assert failure is not None
    assert failure["code"] == "webshop_worker_no_staged_purchase"
    assert "execution_budget_exceeded" in failure["original_training_exclusion_reasons"]

    assert (
        _runtime_owned_model_policy_failure(
            "webshop",
            training_exclusion_reasons=["execution_budget_exceeded"],
            verification=VerificationResult(0.0, False, "webshop_environment"),
            worker_backend_failure=None,
            worker_artifact_integrity={},
            worker_artifact_integrity_failure=None,
            swe_output_progress={},
            alfworld_output_progress={},
            webshop_output_progress={},
            swe_environment_result={},
            swe_infrastructure_failure=None,
            swe_synthetic_evaluation=False,
            swe_non_train_split=False,
            stateful_environment_result={
                "done": False,
                "steps": 0,
                "termination_reason": "missing_output_agent",
                "budget_truncated": False,
            },
            admit_stateful_policy_failure_terminal=False,
        )
        is None
    )


def test_exhausted_webshop_commit_ready_without_output_is_attributed_to_director() -> None:
    failure = _runtime_owned_model_policy_failure(
        "webshop",
        training_exclusion_reasons=[
            "not_finished",
            "invalid_final_graph",
            "execution_incomplete",
            "execution_budget_exceeded",
        ],
        verification=VerificationResult(0.0, False, "webshop_environment"),
        worker_backend_failure=None,
        worker_artifact_integrity={
            "shopper": {
                "webshop_progress": {
                    "trusted": True,
                    "state": "purchase_staged",
                    "commit_ready": True,
                }
            }
        },
        worker_artifact_integrity_failure=None,
        swe_output_progress={},
        alfworld_output_progress={},
        webshop_output_progress={},
        swe_environment_result={},
        swe_infrastructure_failure=None,
        swe_synthetic_evaluation=False,
        swe_non_train_split=False,
        stateful_environment_result={
            "done": False,
            "steps": 0,
            "termination_reason": "missing_output_agent",
            "budget_truncated": False,
        },
        admit_stateful_policy_failure_terminal=True,
    )

    assert failure is not None
    assert failure["code"] == "webshop_director_commit_ready_not_selected"
    assert failure["attribution"] == "director_policy"
    assert failure["commit_ready_agents"] == ["shopper"]


def test_webshop_output_closure_failure_is_a_trainable_zero_in_same_group(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the shopping task", task_id="webshop-output-closure"
    )
    progress = {
        "trusted": True,
        "state": "completed",
        "environment_owner": "solver",
        "environment_access": "mutable_owner",
        "commit_ready": False,
    }
    result.task.metadata.update(
        {
            "dataset": "webshop",
            "webshop_environment_result": {
                "reward": 0.0,
                "purchased": False,
                "done": False,
                "steps": 13,
                "termination_reason": "active",
                "purchase_committed": False,
            },
            "webshop_output_progress": progress,
            "worker_artifact_integrity": {
                "solver": {
                    "integrity_risks": [],
                    "webshop_progress": progress,
                }
            },
            "worker_artifact_integrity_failure": None,
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "webshop_environment")
    result.solver_result.director_run.finished = False
    result.solver_result.trace.events.append(
        TraceEvent(
            len(result.solver_result.trace.events),
            "canvas_step",
            {
                "accepted": True,
                "final_execution": True,
                "rejection_code": "webshop_output_closure_incomplete",
            },
        )
    )

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=2, seed=47)
    metadata = rollout.trajectory.metadata

    assert rollout.trajectory.reward == 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["terminal_status"] == "typed_policy_failure"
    assert metadata["failure_mode"] == "typed_policy_failure"
    assert metadata["typed_policy_failure"] == {
        "status": "typed_policy_failure",
        "code": "webshop_output_closure_incomplete",
        "dataset": "webshop",
        "attribution": "model_policy",
        "official_environment_terminal": False,
        "runtime_terminal": True,
        "official_score": 0.0,
        "environment_owner": "solver",
        "environment_steps": 13,
        "termination_reason": "active",
        "original_training_exclusion_reasons": [
            "execution_incomplete",
            "not_finished",
            "webshop_output_closure_incomplete",
        ],
        "official_step_limit": False,
    }
    # A trainable negative is final for the slot: the scheduler must not grant
    # it a selective retry that successful sibling trajectories never receive.
    assert _recovery_decision("webshop", rollout=rollout).scope.value == "none"


def test_webshop_output_closure_does_not_hide_execution_failure(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the shopping task", task_id="webshop-unsafe-output-closure"
    )
    progress = {
        "trusted": True,
        "state": "completed",
        "environment_owner": "solver",
        "environment_access": "mutable_owner",
    }
    result.task.metadata.update(
        {
            "dataset": "webshop",
            "webshop_environment_result": {
                "reward": 0.0,
                "purchased": False,
                "done": False,
                "steps": 3,
                "termination_reason": "active",
            },
            "webshop_output_progress": progress,
            "worker_artifact_integrity": {},
            "worker_artifact_integrity_failure": None,
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "webshop_environment")
    result.solver_result.director_run.finished = False
    for rejection_code in (
        "webshop_output_closure_incomplete",
        "execution_failure",
    ):
        result.solver_result.trace.events.append(
            TraceEvent(
                len(result.solver_result.trace.events),
                "canvas_step",
                {"accepted": True, "rejection_code": rejection_code},
            )
        )

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=1, seed=48)
    metadata = rollout.trajectory.metadata

    assert metadata["training_eligible"] is False
    assert metadata["typed_policy_failure"] is None
    assert "execution_failure" in metadata["training_exclusion_reasons"]


def test_webshop_official_step_limit_closure_is_a_trainable_zero(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the shopping task", task_id="webshop-step-limit-closure"
    )
    progress = {
        "trusted": True,
        "state": "completed",
        "environment_owner": "solver",
        "environment_access": "mutable_owner",
    }
    result.task.metadata.update(
        {
            "dataset": "webshop",
            "webshop_environment_result": {
                "reward": 0.0,
                "purchased": False,
                "done": True,
                "steps": 20,
                "termination_reason": "step_limit",
            },
            "webshop_output_progress": progress,
            "worker_artifact_integrity": {
                "solver": {
                    "integrity_risks": ["terminal_tool_failure"],
                    "webshop_progress": progress,
                }
            },
            "worker_artifact_integrity_failure": {
                "output_agent": "solver",
                "risks": ["terminal_tool_failure"],
            },
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "webshop_environment")
    result.solver_result.director_run.finished = False
    result.solver_result.trace.events.append(
        TraceEvent(
            len(result.solver_result.trace.events),
            "canvas_step",
            {
                "accepted": True,
                "final_execution": True,
                "rejection_code": "webshop_output_closure_incomplete",
            },
        )
    )

    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=49)
    metadata = rollout.trajectory.metadata

    assert rollout.trajectory.reward == 0.0
    assert metadata["training_eligible"] is True
    assert metadata["training_exclusion_reasons"] == []
    assert metadata["typed_policy_failure"]["official_environment_terminal"] is True
    assert metadata["typed_policy_failure"]["official_step_limit"] is True


def test_five_sibling_group_keeps_webshop_closure_zero_for_training(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))

    class OneClosureZeroApplication:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            result = self.delegate.solve(*args, **kwargs)
            if self.seed != 0:
                return result
            owner = str(result.solver_result.director_run.graph["output_agent"])
            progress = {
                "trusted": True,
                "state": "completed",
                "environment_owner": owner,
                "environment_access": "mutable_owner",
            }
            result.task.metadata.update(
                {
                    "dataset": "webshop",
                    "webshop_environment_result": {
                        "reward": 0.0,
                        "purchased": False,
                        "done": False,
                        "steps": 9,
                        "termination_reason": "active",
                    },
                    "webshop_output_progress": progress,
                    "worker_artifact_integrity": {
                        owner: {
                            "integrity_risks": [],
                            "webshop_progress": progress,
                        }
                    },
                    "worker_artifact_integrity_failure": None,
                }
            )
            result.solver_result.verification = VerificationResult(
                0.0, False, "webshop_environment"
            )
            result.solver_result.director_run.finished = False
            result.solver_result.trace.events.append(
                TraceEvent(
                    len(result.solver_result.trace.events),
                    "canvas_step",
                    {
                        "accepted": True,
                        "final_execution": True,
                        "rejection_code": "webshop_output_closure_incomplete",
                    },
                )
            )
            return result

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "five-sibling-with-closure-zero"
    result = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=OneClosureZeroApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(rollouts_per_task=5, workers=1),
    ).run(["shopping task"])

    assert len(result.solver_batch.samples) == 5
    assert all(
        sample.metadata["rollout_group_complete"] is True
        and sample.metadata["rollout_group_size_expected"] == 5
        for sample in result.solver_batch.samples
    )
    closure_samples = [
        sample
        for sample in result.solver_batch.samples
        if (sample.metadata.get("typed_policy_failure") or {}).get("code")
        == "webshop_output_closure_incomplete"
    ]
    assert len(closure_samples) == 1
    assert closure_samples[0].reward == 0.0
    assert closure_samples[0].metadata["training_eligible"] is True
    assert not (output / "quarantined_groups.jsonl").exists()


@pytest.mark.parametrize("exception_only", [False, True])
def test_uncertain_zero_persists_score_without_aborting_or_fabricating_policy(
    tmp_path, exception_only
):
    from selfplay_graph_flowsteer.outcome_metrics import collect_outcome_metrics

    config = load_adaptive_config(write_config(tmp_path))
    attempts = []

    class UnsafeApplication:
        def __init__(self, seed):
            self.delegate = create_adaptive_application(config, mock=True)

        @property
        def solver(self):
            return self.delegate.solver

        def solve(self, *args, **kwargs):
            if kwargs["task_id"] == "task-1":
                attempts.append(kwargs["task_id"])
                if exception_only:
                    raise WorkerWallClockLimitExceeded(
                        "test uncertain timeout",
                        reason="no_progress",
                        stage="director_turn_start",
                        elapsed_s=90.0,
                        idle_s=90.0,
                    )
            result = self.delegate.solve(*args, **kwargs)
            if kwargs["task_id"] == "task-1":
                result.solver_result.director_run.finished = False
            return result

        def close(self):
            self.delegate.close()

    output = tmp_path / "uncertain-zero"
    result = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=UnsafeApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2, workers=2, task_window=2, uncertain_attribution_zero_reward=True
        ),
    ).run(["first", "second"])
    assert len(attempts) == 2  # two slots; local recovery does not restart the whole trajectory
    assert not (output / "collection_abort.json").exists()
    raw = [json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()]
    failed = [row for row in raw if row["task_id"] == "task-1"]
    if exception_only:
        assert failed == []
        scores = [
            json.loads(line)
            for line in (output / "uncertain_failure_outcomes.jsonl").read_text().splitlines()
        ]
        assert len(scores) == 2
        assert all("token_ids" not in row for row in scores)
    else:
        assert len(failed) == 2
        assert all(row["reward"] == 0 and row["metadata"]["reward_known"] for row in failed)
        assert all(
            row["metadata"]["uncertain_attribution_zero"]["attribution"] == "unknown"
            for row in failed
        )
        assert all(not row["metadata"]["training_eligible"] for row in failed)
    assert all(sample.task_id == "task-2" for sample in result.solver_batch.samples)
    report = collect_outcome_metrics(
        output, cycle=0, tasks=result.tasks, solver_batch=result.solver_batch, k=2, snapshots={}
    )
    assert report["overall"]["uncertain_attribution_zero_count"] == 2
    assert report["overall"]["scored_count"] >= 2
    assert report["overall"]["policy_failure_zero_count"] == 0


@pytest.mark.parametrize("no_legal_actions", [False, True])
@pytest.mark.parametrize("zero_unknown", [False, True])
def test_bounded_director_terminal_is_scored_without_outer_recollection(
    tmp_path,
    monkeypatch,
    no_legal_actions,
    zero_unknown,
):
    from selfplay_graph_flowsteer.canvas import GraphCanvas
    from selfplay_graph_flowsteer.outcome_metrics import collect_outcome_metrics
    from selfplay_graph_flowsteer.rollouts import TrainingBatch

    if no_legal_actions:
        original_snapshot = GraphCanvas.control_snapshot
        monkeypatch.setattr(
            GraphCanvas,
            "control_snapshot",
            lambda self: {
                **original_snapshot(self),
                "allowed_actions": [],
            },
        )
    terminal_code = (
        "director_no_legal_continuation" if no_legal_actions else "director_no_progress_exhausted"
    )
    config = load_adaptive_config(write_config(tmp_path))
    backends = []

    def factory(seed):
        application = create_adaptive_application(config, mock=True)
        backend = MockBackend(["not JSON"] * 4)
        application.solver.director_backend = backend
        backends.append(backend)
        return application

    output = tmp_path / "bounded-director"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=factory,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            uncertain_attribution_zero_reward=zero_unknown,
            require_all_planned_task_groups_for_training=True,
            allow_exact_rollout_resume=True,
        ),
    )
    with pytest.raises(InsufficientCompleteRolloutGroupsError):
        runner.run(["task"])
    assert len(backends) == 2 and all(
        len(b.calls) == (0 if no_legal_actions else 4) for b in backends
    )
    rows = [
        json.loads(line) for line in (output / "solver_rollouts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    for row in rows:
        metadata = row["metadata"]
        assert metadata["reward_known"] is zero_unknown
        assert metadata["bounded_director_terminal"] == terminal_code
        if zero_unknown:
            assert row["reward"] == 0
            assert metadata["uncertain_attribution_zero"]["attribution"] == "unknown"
        else:
            assert "uncertain_attribution_zero" not in metadata
        assert terminal_code in metadata["training_exclusion_reasons"]
        assert not metadata["training_eligible"]
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 2
    assert all(a["replacement_attempt"] == 0 and a["accepted_as_primary"] for a in attempts)
    assert not (output / "collection_abort.json").exists()
    report = collect_outcome_metrics(
        output, cycle=0, tasks=(), solver_batch=TrainingBatch("solver", ()), k=2, snapshots={}
    )
    assert report["overall"]["scored_count"] == (2 if zero_unknown else 0)
    assert report["overall"]["uncertain_attribution_zero_count"] == (2 if zero_unknown else 0)
    assert report["overall"]["training_eligible_count"] == 0
    persisted = (output / "solver_rollouts.jsonl").read_bytes()
    with pytest.raises(InsufficientCompleteRolloutGroupsError):
        runner.run(["task"], resume=True)
    assert len(backends) == 2
    assert (output / "solver_rollouts.jsonl").read_bytes() == persisted


def test_uncertain_zero_preserves_trusted_scores_and_policy_records():
    from selfplay_graph_flowsteer.selfplay_runtime import _uncertain_failure_zero

    trajectory = TokenizedDirectorTrajectory(
        rollout_id="x",
        task_id="t",
        token_ids=(1, 2),
        action_mask=(0, 1),
        reward=0.7,
        graph={},
        metadata={
            "training_eligible": False,
            "training_exclusion_reasons": ["director_policy_call_ineligible"],
        },
    )
    rollout = SolverRollout(trajectory, MultiAgentGraph())
    scored = _uncertain_failure_zero(rollout, "unresolved")
    assert scored.trajectory.reward == 0
    assert scored.trajectory.token_ids == trajectory.token_ids
    assert scored.trajectory.action_mask == trajectory.action_mask
    assert scored.trajectory.metadata["training_eligible"] is False
    assert scored.trajectory.metadata["training_exclusion_reasons"] == [
        "director_policy_call_ineligible"
    ]
    assert trajectory.reward == 0.7
    for extra in (
        {"reward_known": True},
        {"worker_backend_failure": True},
        {"swe_infrastructure_failure": True},
        {"swe_non_train_split": True},
    ):
        protected = replace(
            rollout, trajectory=replace(trajectory, metadata={**trajectory.metadata, **extra})
        )
        assert _uncertain_failure_zero(protected, "unresolved") is protected


def test_uncertain_zero_cli_default_and_opt_out():
    from selfplay_graph_flowsteer.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(
        ["selfplay-experiment", "--output", "unused"]
    ).uncertain_attribution_zero_reward
    assert not parser.parse_args(
        ["selfplay-experiment", "--output", "unused", "--no-uncertain-attribution-zero-reward"]
    ).uncertain_attribution_zero_reward


@pytest.mark.parametrize("score", [0.0, 1.0, 0.37])
@pytest.mark.parametrize("zero_unknown", [False, True])
@pytest.mark.parametrize("bounded_terminal", [None, "director_no_progress_exhausted"])
def test_trusted_score_is_persisted_before_training_admission(
    tmp_path,
    monkeypatch,
    score,
    zero_unknown,
    bounded_terminal,
):
    from selfplay_graph_flowsteer.outcome_metrics import collect_outcome_metrics
    from selfplay_graph_flowsteer.rollouts import TokenizedPolicyCall, TrainingBatch
    from selfplay_graph_flowsteer.selfplay_runtime import _PrimaryCollection

    config = load_adaptive_config(write_config(tmp_path))
    collected, closed, originals = [], [], []

    def collect(self, job, **kwargs):
        proposal, index = job
        collected.append((proposal.task.task_id, index))
        trajectory = TokenizedDirectorTrajectory(
            rollout_id=f"{proposal.task.task_id}-r{index}",
            task_id=proposal.task.task_id,
            token_ids=(1, 2),
            action_mask=(0, 1),
            reward=score,
            graph={},
            policy_calls=(TokenizedPolicyCall("actual-test-call", (1, 2), (0, 1), (-0.5,)),),
            metadata={
                "reward_known": True,
                "reward_admission_reason": "trusted_task_result",
                "bounded_director_terminal": bounded_terminal,
                "training_eligible": False,
                "task_reward": score,
                "task_outcome_passed": score == 1.0,
                "verification": {"score": score, "passed": score == 1.0},
                "training_exclusion_reasons": ["tool_failure_attribution_unresolved"],
                "worker_artifact_integrity": {
                    "a": {
                        "runtime_tool_evidence": {"failure_codes": ["repeated_no_progress_action"]}
                    }
                },
            },
        )
        originals.append(json.loads(json.dumps(trajectory.to_dict())))
        return _PrimaryCollection(
            proposal,
            index,
            index,
            index,
            SimpleNamespace(close=lambda: closed.append(index)),
            None,
            SolverRollout(trajectory, MultiAgentGraph()),
            685.0,
        )

    monkeypatch.setattr(SelfPlayRolloutRunner, "_collect_primary", collect)
    output = tmp_path / "trusted-score"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=lambda seed: None,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            2,
            workers=1,
            require_all_planned_task_groups_for_training=True,
            uncertain_attribution_zero_reward=zero_unknown,
            allow_exact_rollout_resume=True,
        ),
    )
    with pytest.raises(InsufficientCompleteRolloutGroupsError):
        runner.run(["one task"])
    assert len(collected) == len(closed) == 2
    assert not (output / "collection_abort.json").exists()
    raw_path = output / "solver_rollouts.jsonl"
    original_bytes = raw_path.read_bytes()
    assert [json.loads(s) for s in raw_path.read_text().splitlines()] == originals
    assert not (output / "solver_batch.json").exists()
    attempts = [json.loads(s) for s in (output / "rollout_attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 2
    assert all(a["replacement_attempt"] == 0 and a["accepted_as_primary"] for a in attempts)
    assert all(
        a["recovery_reason"] == "trusted_score_preserved_training_excluded" for a in attempts
    )
    report = collect_outcome_metrics(
        output, cycle=0, tasks=(), solver_batch=TrainingBatch("solver", ()), k=2, snapshots={}
    )
    assert report["overall"]["scored_count"] == 2
    assert report["overall"]["training_eligible_count"] == 0
    assert report["overall"]["uncertain_attribution_zero_count"] == 0
    with pytest.raises(InsufficientCompleteRolloutGroupsError):
        runner.run(["one task"], resume=True)
    assert len(collected) == 2  # explicit resume must not recollect an already scored slot
    assert raw_path.read_bytes() == original_bytes


def test_capacity_exhaustion_does_not_cancel_remaining_slots(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class QueueFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            detail = {
                "event": "backend_request_failure",
                "backend_failure": True,
                "origin": "provider_capacity",
                "kind": "account_pool_exhausted",
                "route": "grok",
                "retryable": True,
                "counts_toward_route_circuit": False,
                "disable_route": False,
            }
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["account_pool_exhausted"],
                        "failure_details": [detail],
                        "request_events": [detail],
                        "retryable": True,
                        "counts_toward_route_circuit": False,
                        "disable_route": False,
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    class LazyPool:
        def iter_map(self, function, jobs):
            return (function(job) for job in jobs)

    output = tmp_path / "queue-failure-no-provider-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=QueueFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            3,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=1,
            backend_failure_retry_attempts=1,
        ),
        rollout_pool=LazyPool(),
    )

    with pytest.raises(RuntimeError):
        runner.run(["seed"])

    assert len(application_calls) == 3
    events = [
        json.loads(line) for line in (output / "backend_api_events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "terminal_backend_failure" for event in events) == 3
    assert all(event["event"] != "backend_circuit_open" for event in events)
    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert len(errors) == 3
    assert all(e["error_type"] == "BackendRetryExhaustedError" for e in errors)
    assert all("persistent_route_health" not in e for e in errors)


def test_transient_timeouts_do_not_cancel_remaining_slots(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    application_calls: list[int] = []

    class QueueFailureApplication:
        def __init__(self, seed: int) -> None:
            application_calls.append(seed)

        def solve(self, prompt, *, task_id, task_type, reference, run_id, metadata):
            del prompt, task_type, reference
            detail = {
                "event": "backend_request_failure",
                "backend_failure": True,
                "origin": "client_network",
                "kind": "request_timeout",
                "route": "grok",
                "retryable": True,
                "counts_toward_route_circuit": True,
                "disable_route": False,
            }
            task = TaskSpec(
                task_id,
                "task",
                metadata={
                    **metadata,
                    "worker_backend_failure": {
                        "count": 1,
                        "agents": ["solver"],
                        "routes": ["grok"],
                        "failure_types": ["request_timeout"],
                        "failure_details": [detail],
                        "request_events": [detail],
                        "retryable": True,
                        "counts_toward_route_circuit": True,
                        "disable_route": False,
                    },
                },
            )
            return SimpleNamespace(task=task, run_id=run_id)

        def close(self) -> None:
            return None

    class LazyPool:
        def iter_map(self, function, jobs):
            return (function(job) for job in jobs)

    output = tmp_path / "queue-failure-no-provider-circuit"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=QueueFailureApplication,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            3,
            backend_failure_route_threshold=2,
            backend_failure_total_threshold=1,
            backend_failure_retry_attempts=1,
            route_health_path=tmp_path / "health.json",
        ),
        rollout_pool=LazyPool(),
    )

    with pytest.raises(RuntimeError):
        runner.run(["seed"])

    assert len(application_calls) == 3
    events = [
        json.loads(line) for line in (output / "backend_api_events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "terminal_backend_failure" for event in events) == 3
    assert all(event["event"] != "backend_circuit_open" for event in events)
    errors = [
        json.loads(line) for line in (output / "rollout_errors.jsonl").read_text().splitlines()
    ]
    assert len(errors) == 3
    assert all(e["error_type"] == "BackendRetryExhaustedError" for e in errors)
    assert all("persistent_route_health" not in e for e in errors)


def test_direct_retryable_backend_request_retries_the_same_slot(tmp_path) -> None:
    config = load_adaptive_config(write_config(tmp_path))
    solve_calls = 0

    class DirectTimeoutThenSuccess:
        def __init__(self, seed: int) -> None:
            self.delegate = create_adaptive_application(config, mock=True)

        def solve(self, *args, **kwargs):
            nonlocal solve_calls
            solve_calls += 1
            if solve_calls == 1:
                raise BackendRequestError(
                    BackendFailureClassification(
                        backend_failure=True,
                        origin="client_network",
                        kind="request_timeout",
                        retryable=True,
                        counts_toward_route_circuit=True,
                    )
                )
            return self.delegate.solve(*args, **kwargs)

        def close(self) -> None:
            self.delegate.close()

    output = tmp_path / "direct-timeout-retry"
    runner = SelfPlayRolloutRunner(
        proposer=_MockProposer(),
        application_factory=DirectTimeoutThenSuccess,
        tokenizer=ByteTokenizer(),
        snapshots=create_selfplay_snapshots(config),
        output_dir=output,
        config=SelfPlayRunConfig(
            1,
            evaluation_only=True,
            frontier_reverify_fraction=0.0,
            allow_legacy_whole_rollout_recovery=True,
            backend_failure_retry_attempts=1,
        ),
    )

    result = runner.run(["seed"])

    assert solve_calls == 2
    assert len(result.solver_batch.samples) == 1
    attempts = [
        json.loads(line) for line in (output / "rollout_attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[0]["recovery_reason"] == "retryable_backend_failure"


@pytest.mark.parametrize("tool_failed", [False, True])
def test_staged_webshop_purchase_is_trainable_zero_without_recollection(tmp_path, tool_failed):
    config = load_adaptive_config(write_config(tmp_path))
    result = create_adaptive_application(config, mock=True).solve(
        "Complete the shopping task", task_id="staged-purchase"
    )
    progress = {
        "trusted": True,
        "state": "purchase_staged",
        "commit_ready": True,
        "commit_protocol_status": "awaiting_canvas_output_selection",
    }
    result.task.metadata.update(
        {
            "dataset": "webshop",
            "webshop_output_progress": progress,
            "worker_artifact_integrity_failure": None,
            "worker_artifact_integrity": {
                "solver": {
                    "webshop_progress": progress,
                    "runtime_tool_evidence": {
                        "trusted": True,
                        "successful_count": 6,
                        "failed_count": int(tool_failed),
                        "failure_codes": ["unknown_tool_error"] if tool_failed else [],
                    },
                }
            },
            "admit_stateful_policy_failure_terminal": False,
        }
    )
    result.solver_result.verification = VerificationResult(0.0, False, "webshop_environment")
    result.solver_result.director_run.finished = False
    rollout = adaptive_result_to_rollout(result, ByteTokenizer(), rollout_index=0, seed=0)
    m = rollout.trajectory.metadata
    assert m["training_eligible"] is (not tool_failed)
    if not tool_failed:
        assert rollout.trajectory.reward == 0.0
        assert m["reward_known"] is True
        assert m["finished"] is False
        assert m["typed_policy_failure"]["code"] == "webshop_director_staged_purchase_not_committed"
        assert m["training_exclusion_reasons"] == []
        assert rollout.trajectory.policy_calls


def test_relation_replay_preserves_terminal_runtime_credit_but_rejects_policy_drift():
    from dataclasses import replace
    from selfplay_graph_flowsteer.counterfactual import RelationDecision, _replay_relation_branch

    graph = MultiAgentGraph()
    for name in ["a", "b"]:
        graph.add_agent(name)
        graph.set_prompt(name, name)
        graph.nodes[name].metadata["_runtime_token_credit"] = 100
    expected = graph.clone()
    expected.set_relation("a", "b", "bidirectional")
    for node in expected.nodes.values():
        node.metadata["_runtime_token_credit"] = 75
    d = RelationDecision(
        0, "a", "b", 0.5, graph.to_dict(), True, expected_final_graph=expected.to_dict()
    )
    for present in [False, True]:
        branch, _, _ = _replay_relation_branch(d, present=present)
        assert all(n.metadata["_runtime_token_credit"] == 75 for n in branch.nodes.values())
        assert bool(branch.bidirectional_edges) == present
    expected.nodes["a"].prompt = "unrecorded policy change"
    with pytest.raises(ValueError, match="diverged"):
        _replay_relation_branch(replace(d, expected_final_graph=expected.to_dict()), present=True)


def test_relation_replay_preserves_recorded_output_pruning_on_both_branches():
    from dataclasses import replace
    from selfplay_graph_flowsteer.counterfactual import RelationDecision, _replay_relation_branch

    prefix = MultiAgentGraph()
    for name in ("a", "b", "c"):
        prefix.add_agent(name)
        prefix.set_prompt(name, name, runtime_route="gpt")
    before = prefix.clone()
    before.set_relation("a", "b", "bidirectional")
    after = before.clone()
    after.set_output("a")
    after.assign_exclusive_capability("a", "environment_commit")
    after.nodes["a"].metadata.update(
        _runtime_webshop_output_closure=True,
        _runtime_budget_phase="closure",
        _runtime_reserved_closure_tokens=0,
    )
    after.delete_agent("c")
    event = {
        "raw_action": '{"action":"set_output","target":"a"}',
        "graph_before": before.to_dict(),
        "graph": after.to_dict(),
    }
    decision = RelationDecision(
        0,
        "a",
        "b",
        0.5,
        prefix.to_dict(),
        True,
        suffix_events=(event,),
        expected_final_graph=after.to_dict(),
    )
    for present in (False, True):
        branch, _, _ = _replay_relation_branch(decision, present=present)
        assert set(branch.nodes) == {"a", "b"}
        assert bool(branch.bidirectional_edges) == present
        assert branch.output_agent == "a"
        assert branch.nodes["a"].metadata["exclusive_capabilities"] == ["environment_commit"]
        assert branch.nodes["a"].metadata["_runtime_budget_phase"] == "closure"
    forged = after.clone()
    forged.delete_agent("b")
    with pytest.raises(ValueError, match="deterministic cleanup"):
        _replay_relation_branch(
            replace(decision, suffix_events=({**event, "graph": forged.to_dict()},)), present=True
        )
