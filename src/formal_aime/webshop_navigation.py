"""Navigation semantics from the current public page, shared by UI and budgets."""
from __future__ import annotations


def navigation_effect(state: dict, action: dict) -> str:
    target = str(action.get("target_id", ""))
    page = state.get("page_type")
    if target.startswith("previous_page:"):
        return {
            "product_section": "return_to_current_product_page",
            "product": "return_to_search_results",
            "search_results": "previous_results_page",
        }.get(page, "")
    if target.startswith("back_to_search:"):
        return "return_to_search"
    if target.startswith("next_page:") and page == "search_results":
        return "next_results_page"
    return str(action.get("navigation_effect", ""))


def annotate_navigation(state: dict) -> None:
    for action in state.get("valid_subactions", []):
        if not isinstance(action, dict):
            continue
        effect = navigation_effect(state, action)
        if effect:
            action["navigation_effect"] = effect
        elif str(action.get("target_id", "")).startswith("previous_page:"):
            action.pop("navigation_effect", None)
