import json

import pytest

from selfplay_graph_flowsteer.runtime import (
    _annotate_webshop_product_state,
    _annotate_webshop_search_state,
    _update_webshop_product_inspections,
    _webshop_action_decision_support,
    _webshop_progress_prompt,
)


def project(state, inspections):
    return _webshop_progress_prompt(
        queries=[], visited_products=list(inspections), product_inspections=inspections,
        candidate_ledger={}, strategy_variant="factual_state_only", recent_actions=[],
        duplicate_action_count=0, semantic_no_progress_count=0, semantic_no_progress_streak=0,
        searches_since_last_product_open=0, current_state=state,
    )


def test_compaction_cannot_reverse_public_visit_or_section_facts():
    asin = "b012345678"
    inspections = {asin: {"asin": asin, "sections_viewed": ["description"],
                          "section_evidence": {"description": "x" * 9000}}}
    search = {"page_type": "search_results", "valid_subactions": [
        {"kind": "open_product", "target_id": f"open_product:1:{asin}", "inspection_status": "inspected"}
    ]}
    progress = project(search, inspections)
    assert progress["product_inspections"] == []
    assert _webshop_action_decision_support(search, progress=progress)[0]["already_inspected"] is True
    section = {"page_type": "product_section", "product": {"asin": asin}, "valid_subactions": [
        {"kind": "view_section", "target_id": f"view_description:{asin}", "label": "Description"}
    ]}
    _annotate_webshop_product_state(section, product_inspections=inspections)
    hints = _webshop_action_decision_support(section, progress=project(section, inspections))
    assert hints[0]["already_observed"] is True
    assert hints[0]["evidence_retained"] is True


def test_evicted_details_do_not_erase_visits_after_serialization_and_revisit():
    records = {}
    for i in range(8):
        asin = f"b{i:09d}"
        _update_webshop_product_inspections(
            action_name="webshop_click", arguments={"target_id": f"view_description:{asin}"},
            output={"page_type": "product_section", "product": {"asin": asin}, "page_text": "Not machine washable."},
            product_inspections=records,
        )
    records = json.loads(json.dumps(records))
    first = "b000000000"
    assert "section_evidence" not in records[first]
    search = {"page_type": "search_results", "valid_subactions": [
        {"kind": "open_product", "target_id": f"open_product:1:{first}"}
    ]}
    _annotate_webshop_search_state(search, product_inspections=records, repeated_public_evidence=True)
    assert search["valid_subactions"][0]["inspection_status"] == "inspected"
    assert not search["valid_subactions"][0]["candidate_evidence"]["product_page_evidence_available"]
    page = {"page_type": "product", "product": {"asin": first}, "valid_subactions": [
        {"kind": "view_section", "target_id": f"view_description:{first}", "label": "Description"}
    ]}
    _annotate_webshop_product_state(page, product_inspections=records)
    assert page["valid_subactions"][0]["evidence_status"] == "observed_not_retained"
    hint = _webshop_action_decision_support(page, progress=project(page, records))[0]
    assert hint["already_observed"] and hint["may_restore_section_context"]
    assert not hint["evidence_retained"]


@pytest.mark.parametrize("page,effect", [
    ("product", "return_to_search_results"),
    ("product_section", "return_to_current_product_page"),
])
def test_previous_page_hint_depends_on_actual_page(page, effect):
    state = {"page_type": page, "valid_subactions": [{"target_id": "previous_page:4"}]}
    _annotate_webshop_product_state(state, product_inspections={})
    assert state["valid_subactions"][0]["navigation_effect"] == effect


def test_missing_visit_information_is_unknown_not_negative():
    hint = _webshop_action_decision_support({"valid_subactions": [
        {"kind": "open_product", "target_id": "open_product:1:B012345678"}
    ]}, progress={})[0]
    assert hint["already_inspected"] is None
