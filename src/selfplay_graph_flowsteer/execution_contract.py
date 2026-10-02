"""Bind prompt, state-machine and skill semantics across collection and learning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def execution_semantics(prompt_variant: str = "v2.1", *, admission_config: dict | None = None) -> dict:
    from . import runtime
    from .aime_formal import VERSION as AIME_VERSION, engine_contract
    from .actions import DIRECTOR_ACTION_PROTOCOL_VERSION
    from .director import director_prompt_components
    from .output_contract import OUTPUT_CONTRACT_VERSION, WORKER_OUTPUT_ROLE_VERSION
    from .hotpot_answer_contract import HOTPOT_ANSWER_CONTRACT_VERSION
    from .pats_semantics import SEMANTIC_REVISION, contract_hash
    from .submission_contract import SUBMISSION_CONTRACT_VERSION

    base, hints = director_prompt_components(prompt_variant)
    from .config import canonical_dataset_name
    scoped_protocols = (admission_config or {}).get("canvas", {}).get("submission_protocol_by_dataset", {})
    variants_by_dataset = {
        canonical_dataset_name(dataset): "v3" if protocol == "unified_task_result_v1" else prompt_variant
        for dataset, protocol in scoped_protocols.items()
    }
    canvas_config = (admission_config or {}).get("canvas", {})
    dataset_thresholds = {
        canonical_dataset_name(dataset): threshold
        for dataset, threshold in canvas_config.get("max_total_tokens_by_dataset", {}).items()
    }
    usage_contracts = {
        canonical_dataset_name(dataset): {
            "budget_policy": policy["policy"],
            "budget_accounting_scope": policy.get("accounting_scope", "question_attempt"),
            "budget_threshold": policy.get("start_threshold", dataset_thresholds.get(
                canonical_dataset_name(dataset), canvas_config.get("max_total_tokens"))),
            "counted_roles": ["worker"],
            "counted_usage": ["input_tokens", "output_tokens"],
            "max_inflight_requests": policy.get("max_inflight_requests", 1),
            "max_unsettled_attempts": policy.get("max_unsettled_attempts", 2),
            "unknown_usage_policy": policy.get("unknown_usage_policy", "continue_bounded"),
            "predicted_admission": False,
            "per_execution_allocations": False,
            "closure_reserves": False,
            "legal_final_request_overshoot": True,
            "actual_timeouts_changed": False,
        }
        for dataset, policy in canvas_config.get("worker_token_budget_by_dataset", {}).items()
    }
    scoped_prompt_hashes = {}
    for dataset, variant in variants_by_dataset.items():
        scoped_base, scoped_hints = director_prompt_components(variant)
        scoped_prompt_hashes[dataset] = {
            key: _digest(scoped_base.rstrip() + "\n\n" + hint.strip() + "\n")
            for key, hint in sorted(scoped_hints.items())
        }
    rendered = {}
    for dataset, adapter in (
        ("hotpotqa", "hotpotqa_context"),
        ("nq_open", "retrieval_qa"),
        ("aime", "aime"),
        ("healthbench_professional", "healthbench_professional"),
        ("alfworld", "alfworld"),
        ("webshop", "webshop"),
        ("swe_bench", "swe_bench"),
        ("", ""),
    ):
        for selected in (False, True):
            args = dict(
                dataset=dataset,
                short_answer_qa=dataset in {"hotpotqa", "nq_open"},
                is_output_agent=selected,
            )
            instruction = runtime._worker_output_instruction([], action_adapter=adapter, **args)
            recovery = runtime._finalization_recovery_messages(
                instruction=instruction,
                react_trace=[],
                previous_attempt_issue="invalid_json",
                preserve_healthbench_response=(dataset == "healthbench_professional"
                    and usage_contracts.get(dataset, {}).get("budget_policy") == "reported_usage_threshold_v1"),
                visible_context={
                    "public_task_context": "PUBLIC TASK",
                    "assigned_task": "DELEGATION",
                    "action_environment": {"adapter": adapter},
                },
                **args,
            )
            responsibility = ""
            if variants_by_dataset.get(dataset, prompt_variant) == "v3":
                from .contracts import AgentNode
                from .unified_contract import PROTOCOL, result_instruction

                node = AgentNode(agent_id="manifest", metadata={
                    "submission_protocol": PROTOCOL,
                    "result_scope": "task_result" if selected else "subtask",
                })
                responsibility = result_instruction(node, dataset)
            rendered[f"{dataset}:{selected}"] = _digest([instruction, responsibility, recovery])
    # Conservative code provenance covers all branches, including tool-dependent
    # prompts and delegation templates not represented by the renders above.
    root = Path(__file__).parent
    source_hashes = {
        name: hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest()
        for name in (
            "execution_contract",
            "actions",
            "output_contract",
            "hotpot_answer_contract",
            "qa_public_task",
            "qa_result_contract",
            "qa_schema_repair",
            "qa_worker_feedback",
            "director_observation",
            "director_timeline",
            "dataset_adapters",
            "learning",
            "runtime",
            "llm",
            "artifact_protocol",
            "healthbench_artifact",
            "contracts",
            "endpoint_pool",
            "director",
            "canvas",
            "graph",
            "delegation",
            "counterfactual",
            "adaptive",
            "answer_submission",
            "submission_contract",
            "unified_contract",
            "unified_submission",
            "webshop_native",
            "webshop",
            "webshop_sidecar",
            "webshop_profiles",
            "webshop_action_reserve",
            "webshop_scheduling",
            "webshop_native_executor",
            "alfworld",
            "swebench",
            "worker_usage_ledger",
            "budget_policy",
            "student_action_protocol",
            "swe_public_tests",
            "swe_public_recipes",
            "_swe_public_probe",
            "outcome_admission",
            "swe_failure_attribution",
            "selfplay_runtime",
            "protocol_reward",
            "application",
            "evaluation",
            "observability",
            "benchmark",
            "benchmark_tracking",
            "benchmark_reporting",
            "qa_metrics",
            "qa_submission",
            "nq_evidence",
            "nq_corpus_tasks",
            "aime_submission",
            "aime_formal",
            "outcome_metrics",
            "rollouts",
            "training",
            "selfplay",
            "async_cycle",
            "config",
            "pats_semantics",
            "pats_refiner",
            "pats",
            "skill_evolution_v2",
        )
    }
    return {
        "director_action_protocol_version": "director_action_json_v3" if prompt_variant == "v3" else DIRECTOR_ACTION_PROTOCOL_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "submission_contract_version": "unified_submission_v1" if prompt_variant == "v3" else SUBMISSION_CONTRACT_VERSION,
        "submission_admission_config": json.loads(json.dumps(admission_config or {})),
        "worker_usage_accounting_contract": usage_contracts,
        "healthbench_artifact_repair_version": (
            "healthbench_artifact_repair_v1"
            if usage_contracts.get("healthbench_professional", {}).get("budget_policy")
            == "reported_usage_threshold_v1" else None
        ),
        "qa_implementation_contract": {
            "version": "qa-b1-formal-20261002",
            "public_task_version": runtime.QA_PUBLIC_TASK_VERSION,
            "result_contract_version": runtime.QA_RESULT_CONTRACT_VERSION,
            "hotpot_answer_contract_version": HOTPOT_ANSWER_CONTRACT_VERSION,
            "director_observation": json.loads(json.dumps(
                (admission_config or {}).get("director_observation", {})
            )),
        },
        "worker_output_role_version": WORKER_OUTPUT_ROLE_VERSION,
        "director_prompt_variant": prompt_variant,
        "director_prompt_variant_by_dataset": variants_by_dataset,
        "director_prompt_template_sha256_by_dataset": scoped_prompt_hashes,
        "director_action_protocol_version_by_dataset": {
            dataset: "director_action_json_v3" if variant == "v3" else DIRECTOR_ACTION_PROTOCOL_VERSION
            for dataset, variant in variants_by_dataset.items()
        },
        "submission_contract_version_by_dataset": {
            dataset: "unified_submission_v1" if variant == "v3" else SUBMISSION_CONTRACT_VERSION
            for dataset, variant in variants_by_dataset.items()
        },
        "director_prompt_template_sha256_by_problem_type": {
            key: _digest(base.rstrip() + "\n\n" + hint.strip() + "\n")
            for key, hint in sorted(hints.items())
        },
        "worker_and_recovery_sha256": rendered,
        "aime_implementation_contract": (
            engine_contract()
            if (admission_config or {}).get("aime_implementation") == AIME_VERSION else None
        ),
        "contract_source_sha256": source_hashes,
        "director_seed_sha256": hashlib.sha256(
            (root / "director_seed_v2.json").read_bytes()
        ).hexdigest(),
        "pats_semantic_revision": SEMANTIC_REVISION,
        "pats_runtime_contract_sha256": contract_hash(prompt_variant),
        "pats_runtime_contract_sha256_by_dataset": {
            dataset: contract_hash(variant) for dataset, variant in variants_by_dataset.items()
        },
    }


def manifest_semantics(manifest: dict) -> dict | None:
    return manifest.get("execution_semantics")


def require_same_semantics(expected, actual) -> None:
    # Transport/policy-lag overrides must never waive this equality.
    if expected != actual:
        raise ValueError("execution semantics changed; start a fresh collection")


def bind_rollout_contract(batches, rollouts):
    from dataclasses import replace

    values = [manifest_semantics(r.trajectory.metadata.get("model_roles", {})) for r in rollouts]
    versions = {r.trajectory.metadata.get("submission_contract_version", "legacy") for r in rollouts}
    scoped = bool(values and values[0] and values[0].get("submission_contract_version_by_dataset"))
    if not scoped and len(versions) > 1:
        raise ValueError("submission contract changed within rollout group")
    if not values or not any(value is not None for value in values):
        return batches  # Offline/legacy fixtures; the live learner requires a binding.
    for rollout, value in zip(rollouts, values):
        require_same_semantics(values[0], value)
        _validate_dataset_submission_contract(rollout.trajectory.metadata, value)
    return tuple(
        replace(batch, metadata={**batch.metadata, "execution_semantics": values[0]})
        for batch in batches
    )


def _validate_dataset_submission_contract(metadata: dict, semantics: dict | None) -> None:
    """A mixed batch may use different protocols only as declared by its manifest."""
    from .config import canonical_dataset_name

    scoped = (semantics or {}).get("submission_contract_version_by_dataset", {})
    if not scoped:
        return
    dataset = canonical_dataset_name(metadata.get("dataset", ""))
    if not dataset:
        raise ValueError("dataset required for mixed submission contracts")
    expected = scoped.get(dataset, semantics["submission_contract_version"])
    if metadata.get("submission_contract_version", "legacy") != expected:
        raise ValueError(f"submission contract does not match dataset: {dataset}")


def validate_training_contract(proposer_batch, solver_batch, *, expected=None):
    recorded = solver_batch.metadata.get("execution_semantics")
    require_same_semantics(recorded, proposer_batch.metadata.get("execution_semantics"))
    if expected is not None:
        require_same_semantics(expected, recorded)
    if recorded is not None:
        require_same_semantics(recorded, execution_semantics(
            recorded["director_prompt_variant"],
            admission_config=recorded.get("submission_admission_config"),
        ))
    for sample in solver_batch.samples:
        from .submission_contract import validate_primary_training_outcome

        validate_primary_training_outcome(sample.metadata)
        observed = manifest_semantics(sample.metadata.get("model_roles", {}))
        if recorded is not None or observed is not None:
            require_same_semantics(recorded, observed)
        _validate_dataset_submission_contract(sample.metadata, recorded)
