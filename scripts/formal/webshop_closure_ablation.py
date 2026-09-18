"""Process-local prompt ablation using the ordinary benchmark entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from selfplay_graph_flowsteer import cli, delegation, runtime
from selfplay_graph_flowsteer.cli import main


OLD_POLICY = (
    "Unknown required constraints are not verified. You decide whether to purchase "
    "given the public support, conflicts, remaining uncertainty and inspection budget. "
)
NEW_POLICY = (
    "Your objective is to complete a purchase that best satisfies the user's request "
    "within the available action budget. Manage the remaining budget so you can select "
    "the requested options and execute Buy Now. Include navigation back to the product "
    "page when needed. When the remaining budget is only enough to finish purchasing "
    "the best candidate you have observed, stop further exploration. Use public "
    "evidence to compare candidates, select the closest available requested options, "
    "and execute Buy Now. A candidate need not satisfy every requirement to be worth "
    "purchasing. Record unsupported or conflicting requirements honestly in "
    "purchase_evidence.unresolved_constraints. Do not claim they are verified. "
    "Return without staging a purchase only when no executable purchase path remains "
    "or no observed product is relevant to the request. "
)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "removed", "best_so_far", "budget_aware"), required=True)
    parser.add_argument("--worker-route", choices=("deepseek",))
    args, benchmark_args = parser.parse_known_args()
    output = Path(benchmark_args[benchmark_args.index("--output") + 1])
    if output.exists():
        raise ValueError("Use a fresh output directory for each experiment arm")
    if args.worker_route:
        load_config = cli.load_adaptive_config

        def fixed_route_config(*positional, **kwargs):
            config = load_config(*positional, **kwargs)
            if config.runtime_pool()[args.worker_route].enable_thinking:
                raise ValueError("The experiment requires Worker thinking disabled")
            return replace(config, worker_runtime_routes=(args.worker_route,))

        cli.load_adaptive_config = fixed_route_config
    original_instruction = runtime._worker_output_instruction
    before = original_instruction([{"name": "webshop_click", "stateful": True}], action_adapter="webshop")
    budget_policy = dict(delegation._DATASET_OUTPUT_CONTRACTS["webshop"])["agent_purchase_authority"] + " "
    if args.variant == "budget_aware" and "Purchase early only" not in budget_policy:
        raise ValueError("Budget-aware production policy is not installed")
    current_policy = budget_policy if budget_policy in before else NEW_POLICY if NEW_POLICY in before else OLD_POLICY
    if before.count(current_policy) != 1:
        raise ValueError("WebShop prompt changed; review the ablation before running")
    replacement = {"original": OLD_POLICY, "removed": "", "best_so_far": NEW_POLICY, "budget_aware": budget_policy}[args.variant]

    def instruction(available_actions: object, *, action_adapter: str = "") -> str:
        text = original_instruction(available_actions, action_adapter=action_adapter)
        return text.replace(current_policy, replacement) if action_adapter == "webshop" else text

    runtime._worker_output_instruction = instruction
    if args.variant == "original":
        delegation._WEBSHOP_EXPECTED_OUTPUT = (
            "Choose a product and requested options using public evidence; decide whether to call "
            "latest Buy Now. Record uncertainty honestly; never claim an unresolved constraint is verified."
        )
        authority = (
            "The Agent decides whether to inspect, compare, purchase or report a blocker. "
            "Purchase evidence records that decision and its unresolved constraints; it is not "
            "an oracle correctness check. The official environment determines the score."
        )
    else:
        delegation._WEBSHOP_EXPECTED_OUTPUT = (
            "Report the shopping task result and any unresolved constraints."
            if args.variant == "removed"
            else "Stage purchase of the best observed relevant product within the action budget; "
            "select requested options and honestly report unresolved constraints."
        )
        authority = budget_policy.strip() if args.variant == "budget_aware" else NEW_POLICY.strip()
    delegation._DATASET_OUTPUT_CONTRACTS["webshop"] = tuple(
        (key, authority if key == "agent_purchase_authority" else text)
        for key, text in delegation._DATASET_OUTPUT_CONTRACTS["webshop"]
        if args.variant != "removed" or key != "agent_purchase_authority"
    )
    after = runtime._worker_output_instruction(
        [{"name": "webshop_click", "stateful": True}], action_adapter="webshop"
    )
    audit = {
        "variant": args.variant,
        "worker_route": args.worker_route,
        "worker_instruction": after,
        "expected_output": delegation._WEBSHOP_EXPECTED_OUTPUT,
        "delegation_contract": delegation._DATASET_OUTPUT_CONTRACTS["webshop"],
        "benchmark_args": benchmark_args,
        "protocol_preserved": True,
        "fresh_sessions": True,
    }
    audit["prompt_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True).encode()
    ).hexdigest()
    output.mkdir(parents=True)
    (output / "prompt_ablation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return main(["benchmark", *benchmark_args])


if __name__ == "__main__":
    raise SystemExit(run())
