"""Public candidate names survive the real search -> product -> section path."""
import copy
import json
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.runtime import (
    _update_webshop_candidate_ledger,
    _update_webshop_product_inspections,
    _webshop_progress_prompt,
    _webshop_restore_keyed_records,
)
from selfplay_graph_flowsteer.webshop_identity import visible_product_asin


CAPTURE = json.loads((Path(__file__).parent / "fixtures/webshop_candidate_title_real_trajectory.json").read_text())


def test_real_00060_names_survive_navigation_journal_restore_and_prompt_projection():
    ledger, inspections = {}, {}
    traces = copy.deepcopy(CAPTURE["react_trace"])
    expected = {
        visible_product_asin(p): p["label"].strip()[:240]
        for p in traces[0]["observation"]["output"]["valid_subactions"]
        if p["kind"] == "open_product"
    }
    assert len(expected) == 10
    for trace in traces:
        state, action = trace["observation"]["output"], trace["action"]
        before = copy.deepcopy(state)
        _update_webshop_product_inspections(
            action_name=action["name"], arguments=action["arguments"],
            output=state, product_inspections=inspections, section_max_chars=1400,
        )
        _update_webshop_candidate_ledger(state, product_inspections=inspections, candidate_ledger=ledger)
        assert state == before  # Retaining a label must not enrich the public page.
        assert {k: v.get("preview_title") for k, v in ledger.items()} == expected
        ledger = _webshop_restore_keyed_records(json.loads(json.dumps(list(ledger.values()))), key="asin")
        projected = _webshop_progress_prompt(
            queries=[traces[0]["action"]["arguments"]["query"]],
            visited_products=list(inspections), product_inspections=inspections,
            candidate_ledger=ledger, strategy_variant="factual_state_only", recent_actions=[],
            duplicate_action_count=0, semantic_no_progress_count=0,
            semantic_no_progress_streak=0, searches_since_last_product_open=0,
            current_state=state, section_max_chars=1400,
        )
        retained = projected["candidate_ledger"]
        if state["page_type"] != "product_section":
            assert any(p["inspection_status"] == "not_inspected" for p in retained)
        # The existing prompt cap may evict previews on detail-heavy sections;
        # the persistent journal above must still retain all ten public names.
        assert all(p["preview_title"] == expected[p["asin"]] for p in retained)
        assert all("preview_price" not in p for p in retained)


@pytest.mark.parametrize("title, label, expected", [
    ("Structured title", "Visible label", "Structured title"),
    (None, " Visible label ", "Visible label"),
    ("   ", "Visible label", "Visible label"),
    (None, "N" * 300, "N" * 240),
    (None, None, None),
])
def test_search_formats_preserve_title_precedence_and_bounded_public_fallback(title, label, expected):
    page = {"page_type": "search_results", "valid_subactions": [
        {"kind": "open_product", "target_id": "open_product:1:B012345678", "title": title, "label": label},
    ]}
    ledger = {}
    _update_webshop_candidate_ledger(page, product_inspections={}, candidate_ledger=ledger)
    assert ledger["b012345678"].get("preview_title") == expected
