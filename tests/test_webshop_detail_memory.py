from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.runtime import (
    _update_webshop_product_inspections,
    _webshop_progress_prompt,
)


@pytest.mark.parametrize("length", [1401, 3012, 9000])
def test_detail_tail_survives_storage_and_decision_checkpoint(length):
    asin = "b012345678"
    tail = " Waterproof: no."
    description = "D" * (length - len(tail)) + tail
    inspections = {}
    state = {"page_type": "product_section", "product": {"asin": asin}}
    _update_webshop_product_inspections(
        action_name="webshop_click",
        arguments={"target_id": f"view_description:{asin}"},
        output={
            **state,
            "page_text": f"Instruction:\nBuy shoes\n[button]Back[button_]\n{description}\n",
        },
        product_inspections=inspections,
    )
    # A later section update must not truncate the previously retained text.
    inspections = json.loads(json.dumps(inspections))
    _update_webshop_product_inspections(
        action_name="webshop_click",
        arguments={"target_id": f"view_features:{asin}"},
        output={**state, "page_text": "Cotton upper."},
        product_inspections=inspections,
    )
    evidence = inspections[asin]["section_evidence"]
    assert evidence == {"description": description, "features": "Cotton upper."}
    prompt = _webshop_progress_prompt(
        queries=[], visited_products=[asin], product_inspections=inspections,
        candidate_ledger={}, strategy_variant="factual_state_only", recent_actions=[],
        duplicate_action_count=0, semantic_no_progress_count=0,
        semantic_no_progress_streak=0, searches_since_last_product_open=0,
        current_state=state,
    )
    # Even when the overall journal evicts older records, the live product's
    # decision checkpoint must not silently drop the evidence's decisive tail.
    assert prompt["decision_checkpoint"]["retained_section_evidence"] == evidence
