"""SkillFlow-style per-episode observation/action history for WebShop Workers.

Reference: local SkillFlow 74be52bb6bd9f0e9e68dacb72636b75649197983,
training/environment.py (_react_step and _build_react_prompt).
Only public observations and attempted Actions are retained, never model reasoning.
"""

from __future__ import annotations

import copy
from typing import Any

WEBSHOP_WORKER_MEMORY_POLICIES = frozenset({"factual_memory_v1", "skillflow_history_v1"})

HISTORY_INSTRUCTION = (
    "The action_environment.react_history contains this episode's chronological public "
    "observations and attempted Actions, including earlier executions by the same owner. "
    "Each observation is the page BEFORE that Action; its result is the next observation "
    "or the latest state. Error entries report failed attempts, not successful changes. "
    "Use the history to compare previously seen products, details and searches. Historical "
    "target IDs and state versions are records only: choose executable Actions from the "
    "latest state's valid_subactions and use its state_version. The latest selected_options "
    "are authoritative; historical option clicks may no longer be selected. "
)


def append_webshop_history(
    history: list[dict[str, Any]],
    *,
    before: dict[str, Any],
    action: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Record one selected attempt, without imposing a length/product-count limit."""
    entry = {
        "step": len(history) + 1,
        "observation": str(before.get("page_text", "")),
        "page_type": str(before.get("page_type", "")),
        "action": {
            "name": str(action.get("name", "")),
            "arguments": copy.deepcopy(action.get("arguments", {})),
        },
        "status": str(result.get("status", "")),
    }
    if result.get("status") != "ok":
        error = result.get("error")
        entry["error"] = copy.deepcopy(error)
    history.append(entry)


def project_webshop_history_environment(environment: dict[str, Any]) -> None:
    """Replace model-facing derived memory with history; keep live tool/commit state.

    The caller owns a deep copy. Internal ledgers and audit observations are unchanged.
    Full page text/history bypass the factual journal's 6000 and page's 8000 limits.
    """
    state = environment.get("state")
    if isinstance(state, dict):
        for key in ("candidate_coverage", "search_decision_state", "action_decision_support"):
            state.pop(key, None)
        for action in state.get("valid_subactions", []):
            if not isinstance(action, dict):
                continue
            for key in (
                "inspection_status",
                "visit_count",
                "candidate_evidence",
                "observed_option_groups",
                "evidence_status",
                "action_semantics",
                "navigation_effect",
            ):
                action.pop(key, None)
    progress = environment.get("webshop_progress")
    if isinstance(progress, dict):
        # Required by this project's staged purchase protocol, not shopping memory.
        environment["webshop_progress"] = {
            key: progress[key]
            for key in ("policy_contract", "purchase_evidence_checkpoint")
            if key in progress
        }
    environment.pop("public_constraint_matrix", None)
