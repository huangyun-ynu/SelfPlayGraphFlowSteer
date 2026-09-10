"""Read-only training-dynamics diagnostics inspired by FlowSteer and SESA.

These are observations, never reward terms, acceptance gates or policy inputs.
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from .outcome_metrics import describe, number


def ratio(n, d):
    return n / d if d else None


def rollout_diagnostics(record):
    meta = record.get("metadata", {})
    diagnostics = meta.get("director_action_diagnostics") or []
    finish = [d["finish_reason"] for d in diagnostics if d.get("finish_reason") is not None]
    budgets = [
        d["director_dynamic_budget"]
        for d in diagnostics
        if isinstance(d.get("director_dynamic_budget"), dict)
    ]
    utilization = [
        b["prompt_tokens"] / b["context_limit"]
        for b in budgets
        if number(b.get("prompt_tokens")) is not None
        and number(b.get("context_limit"))
        and (b["context_limit"] > 0)
    ]
    calls = record.get("policy_calls") or []
    lengths = [len(call["token_ids"]) for call in calls if "token_ids" in call]
    masked = [sum(call["action_mask"][1:]) for call in calls if "action_mask" in call]
    return {
        "interactive_turns": number(meta.get("interactive_turns")),
        "rejected_turns": number(meta.get("rejected_turns")),
        "finished": meta.get("finished") if isinstance(meta.get("finished"), bool) else None,
        "generation_finish_reason_count": len(finish),
        "generation_length_stop_count": sum((value == "length" for value in finish)),
        "context_observed_call_count": len(utilization),
        "max_prompt_context_fraction": max(utilization) if utilization else None,
        "policy_call_count": len(calls) if "policy_calls" in record else None,
        "policy_call_sequence_tokens": sum(lengths) if lengths else None,
        "policy_call_target_tokens": sum(masked) if masked else None,
    }


def primary_dynamics(rows, k):
    """All planned primary slots, not only the optimizer's admitted subset."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    complete = [g for g in groups.values() if len(g) == k and all((r["reward_known"] for r in g))]
    binary = [g for g in complete if all((r["task_outcome_passed"] is not None for r in g))]
    spreads = [
        max((r["training_reward"] for r in g)) - min((r["training_reward"] for r in g))
        for g in complete
    ]
    zero = sum((value <= 1e-12 for value in spreads))
    all_wrong = sum((all((r["task_outcome_passed"] is False for r in g)) for g in binary))
    all_right = sum((all((r["task_outcome_passed"] is True for r in g)) for g in binary))
    conditions = {
        "success": [r for r in rows if r["task_outcome_passed"] is True],
        "failure": [r for r in rows if r["task_outcome_passed"] is False],
        "continuous_scored": [
            r for r in rows if r["reward_known"] and r["task_outcome_passed"] is None
        ],
        "unscored": [r for r in rows if not r["reward_known"]],
    }
    cost_fields = (
        "duration_s",
        "recorded_total_token_in",
        "recorded_total_token_out",
        "environment_actions",
        "graph_agents",
        "graph_relations",
    )
    buckets = defaultdict(list)
    for row in rows:
        agents = row["graph_agents"]
        bucket = (
            "unknown"
            if agents is None
            else "one"
            if agents == 1
            else "two_three"
            if 2 <= agents <= 3
            else "four_plus"
            if agents >= 4
            else "empty"
        )
        buckets[bucket].append(row)
    return {
        "complete_reward_groups": len(complete),
        "reward_group_range": describe(spreads),
        "zero_reward_variance_groups": zero,
        "zero_reward_variance_group_rate": ratio(zero, len(complete)),
        "variable_reward_groups": len(complete) - zero,
        "complete_binary_groups": len(binary),
        "all_failure_groups": all_wrong,
        "all_success_groups": all_right,
        "mixed_binary_groups": len(binary) - all_wrong - all_right,
        "mixed_binary_group_rate": ratio(len(binary) - all_wrong - all_right, len(binary)),
        "split_counts": dict(Counter((str(r.get("split") or "unknown") for r in rows))),
        "orchestration": {
            **{
                field: describe((r.get("diagnostics", {}).get(field) for r in rows))
                for field in (
                    "interactive_turns",
                    "rejected_turns",
                    "finished",
                    "max_prompt_context_fraction",
                    "policy_call_count",
                    "policy_call_sequence_tokens",
                    "policy_call_target_tokens",
                )
            },
            "rejected_turn_rate": ratio(
                sum(
                    (
                        r["diagnostics"]["rejected_turns"]
                        for r in rows
                        if r.get("diagnostics", {}).get("rejected_turns") is not None
                        and r["diagnostics"].get("interactive_turns") is not None
                    )
                ),
                sum(
                    (
                        r["diagnostics"]["interactive_turns"]
                        for r in rows
                        if r.get("diagnostics", {}).get("rejected_turns") is not None
                        and r["diagnostics"].get("interactive_turns") is not None
                    )
                ),
            ),
            "finish_reason_observed_calls": sum(
                (r.get("diagnostics", {}).get("generation_finish_reason_count", 0) for r in rows)
            ),
            "length_stop_calls": sum(
                (r.get("diagnostics", {}).get("generation_length_stop_count", 0) for r in rows)
            ),
        },
        "cost_by_outcome": {
            name: {
                "slot_count": len(group),
                **{field: describe((r.get(field) for r in group)) for field in cost_fields},
                "interactive_turns": describe(
                    (r.get("diagnostics", {}).get("interactive_turns") for r in group)
                ),
            }
            for (name, group) in conditions.items()
        },
        "complexity_buckets": {
            name: {
                "slot_count": len(group),
                "scored_count": sum((r["reward_known"] for r in group)),
                "binary_scored_count": sum((r["task_outcome_passed"] is not None for r in group)),
                "success_count": sum((r["task_outcome_passed"] is True for r in group)),
                "success_rate": ratio(
                    sum((r["task_outcome_passed"] is True for r in group)),
                    sum((r["task_outcome_passed"] is not None for r in group)),
                ),
                "reward": describe((r["training_reward"] for r in group if r["reward_known"])),
            }
            for (name, group) in sorted(buckets.items())
        },
    }


def policy_signal(batch, credits, relation_weight=1.0):
    """Count actual next-token policy targets, including binary credit replacement.

    Nonzero advantages are potential policy-loss signal, not proof of nonzero
    gradients. KL may update a batch even when every policy advantage is zero.
    """
    from .training import _relation_call_advantage, token_advantages

    credits = tuple(credits)
    total = positive = negative = nonfinite = active_samples = relation_rescued = 0
    for sample in batch.samples:
        values = []
        for index, call in enumerate(sample.policy_calls):
            if call.relation_token_span is not None and call.metadata.get("relation_policy"):
                values.append(_relation_call_advantage(sample, credits, index, relation_weight))
            else:
                values.extend([sample.advantage] * sum(call.action_mask[1:]))
        if not sample.policy_calls:
            values = [
                adv
                for (adv, mask) in zip(
                    token_advantages(sample, credits, relation_weight=relation_weight)[1:],
                    sample.action_mask[1:],
                    strict=True,
                )
                if mask
            ]
        active = any((math.isfinite(v) and abs(v) > 1e-12 for v in values))
        active_samples += int(active)
        relation_rescued += int(abs(sample.advantage) <= 1e-12 and active)
        total += len(values)
        positive += sum((math.isfinite(v) and v > 1e-12 for v in values))
        negative += sum((math.isfinite(v) and v < -1e-12 for v in values))
        nonfinite += sum((not math.isfinite(v) for v in values))
    return {
        "sample_count": len(batch.samples),
        "target_tokens": total,
        "positive_advantage_tokens": positive,
        "negative_advantage_tokens": negative,
        "nonfinite_advantage_tokens": nonfinite,
        "zero_advantage_tokens": total - positive - negative - nonfinite,
        "nonzero_advantage_token_rate": ratio(positive + negative, total),
        "nonzero_advantage_sample_count": active_samples,
        "nonzero_advantage_sample_rate": ratio(active_samples, len(batch.samples)),
        "zero_graph_advantage_local_signal_samples": relation_rescued,
    }


def update_diagnostics(
    record, batches, credits, *, epochs, mini_batch_size, relation_weight=1.0, frontiers=()
):
    (credits, frontiers) = (tuple(credits), tuple(frontiers))
    evaluation = bool(record.get("experiment", {}).get("evaluation_only"))
    output = {
        "evaluation_only": evaluation,
        "mock_trainer": bool(record.get("experiment", {}).get("mock_trainer")),
        "roles": {},
    }
    for role, batch in batches.items():
        update = record.get("policies", {}).get(role, {})
        n = len(batch.samples)
        full_group_mean = role == "solver" and getattr(batch, "metadata", {}).get(
            "training_selection_schema"
        ) in {"eligible_subset_v1", "independent_frontier_v2"}
        logical_size = max(1, n) if full_group_mean else mini_batch_size
        disabled = evaluation or bool(record.get("experiment", {}).get("collection_only"))
        nominal = 0 if disabled else epochs * math.ceil(n / logical_size)
        actual = int(update.get("optimizer_steps", 0))
        output["roles"][role] = {
            "eligible_samples": n,
            "eligible_tasks": len({s.task_id for s in batch.samples}),
            "epochs": epochs,
            "logical_mini_batch_limit": logical_size,
            "expected_optimizer_steps": nominal,
            "actual_optimizer_steps": actual,
            "step_count_matches_plan": nominal == actual,
            "update_result_present": bool(update),
            "signal": policy_signal(batch, credits if role == "solver" else (), relation_weight),
        }
    valid = [
        (c.q_absent, c.q_present, c.probability_present)
        for c in credits
        if all((number(v) is not None for v in (c.q_absent, c.q_present, c.probability_present)))
    ]
    output["relation_effect"] = {
        "completed_probe_count": len(credits),
        "finite_probe_count": len(valid),
        "signed_on_minus_off": describe((on - off for (off, on, _) in valid)),
        "absolute_effect": describe((abs(on - off) for (off, on, _) in valid)),
        "zero_effect_count": sum((abs(on - off) <= 1e-12 for (off, on, _) in valid)),
        "zero_effect_rate": ratio(
            sum((abs(on - off) <= 1e-12 for (off, on, _) in valid)), len(valid)
        ),
        "distance_from_half": describe((abs(p - 0.5) for (_, _, p) in valid)),
    }
    pairs = [pair for f in frontiers for pair in f.pair_stability]
    observed_pairs = [p for p in pairs if isinstance(p.get("stability_gate_passed"), bool)]
    output["frontier_stability"] = {
        "task_count": len(frontiers),
        "reverified_task_count": sum((f.reverify_status == "completed" for f in frontiers)),
        "observed_pair_count": len(observed_pairs),
        "passed_pair_count": sum((p["stability_gate_passed"] for p in observed_pairs)),
        "passed_pair_rate": ratio(
            sum((p["stability_gate_passed"] for p in observed_pairs)), len(observed_pairs)
        ),
        "softened_tie_pair_count": sum(p.get("tie_softened") is True for p in observed_pairs),
        "softened_tie_pair_rate": ratio(
            sum(p.get("tie_softened") is True for p in observed_pairs), len(observed_pairs)
        ),
        "provisional_score": describe((f.provisional_graph_local for f in frontiers)),
        "stable_score": describe((f.stable_graph_local for f in frontiers)),
    }
    return output
