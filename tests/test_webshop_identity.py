from __future__ import annotations

import copy
import json

import pytest

from selfplay_graph_flowsteer.runtime import (
    _annotate_webshop_search_state,
    _update_webshop_candidate_ledger,
    _update_webshop_product_inspections,
    _webshop_action_decision_support,
    _webshop_semantic_action,
)
from selfplay_graph_flowsteer.webshop import WebShopSessionLifecycle, _resolve_visible_target_id
from selfplay_graph_flowsteer.webshop_identity import visible_product_asin

ASIN = "B012345678"
OTHER = "B087654321"


def search_page(*, ordinal=1):
    return WebShopSessionLifecycle(None, search_observation_mode="legacy")._bounded(
        {
            "page_type": "search_results",
            "page_text": "Original rendered search results",
            "valid_subactions": [
                {"kind": "open_product", "target_id": f"open_product:{ordinal}:{asin}",
                 "asin": asin, "title": "Same title", "label": "Same title", "price": 12.0}
                for asin in (ASIN, OTHER)
            ],
        }
    )


@pytest.mark.parametrize(
    "action, expected",
    [
        ({"kind": "open_product", "asin": f" {ASIN} "}, ASIN.lower()),
        ({"kind": "open_product", "target_id": f"open_product:0:{ASIN}"}, ASIN.lower()),
        ({"kind": "open_product", "asin": None, "target_id": f"open_product:12:{ASIN.lower()}"}, ASIN.lower()),
        ({"kind": "open_product", "asin": OTHER, "target_id": f"open_product:1:{ASIN}"}, OTHER.lower()),
        ({"kind": "navigate", "target_id": f"open_product:1:{ASIN}"}, ""),
        ({"kind": "open_product", "label": ASIN}, ""),
        ({"kind": "open_product", "target_id": f"open_product:x:{ASIN}"}, ""),
        ({"kind": "open_product", "target_id": f"open_product:1:{ASIN}:extra"}, ""),
        ({"kind": "open_product", "target_id": "open_product:1:short"}, ""),
        (None, ""),
    ],
)
def test_identity_uses_only_explicit_or_renderer_action_identity(action, expected):
    before = copy.deepcopy(action)
    assert visible_product_asin(action) == expected
    assert action == before


def test_legacy_revisit_updates_memory_and_support_without_restoring_search_fields():
    inspections, ledger = {}, {}
    first = search_page()
    _update_webshop_candidate_ledger(first, product_inspections=inspections, candidate_ledger=ledger)
    assert set(ledger) == {ASIN.lower(), OTHER.lower()}
    assert all(record["inspection_status"] == "not_inspected" for record in ledger.values())
    assert all("preview_price" not in record for record in ledger.values())
    assert all(record["preview_title"] == "Same title" for record in ledger.values())

    _update_webshop_product_inspections(
        action_name="webshop_click", arguments={"target_id": f"open_product:1:{ASIN}"},
        output={"page_type": "product", "product": {"asin": ASIN, "title": "Observed title", "price": 12}},
        product_inspections=inspections,
    )
    # Match a revision's persisted memory after page-local result order changes.
    inspections = json.loads(json.dumps(inspections))
    page = search_page(ordinal=7)
    original_ids = [a["target_id"] for a in page["valid_subactions"]]
    _annotate_webshop_search_state(page, product_inspections=inspections, repeated_public_evidence=True)
    _update_webshop_candidate_ledger(page, product_inspections=inspections, candidate_ledger=ledger)
    support = _webshop_action_decision_support(page, progress={"product_inspections": list(inspections.values())})

    visited, fresh = page["valid_subactions"]
    assert visited["inspection_status"] == "inspected" and visited["visit_count"] == 1
    assert fresh["inspection_status"] == "not_inspected" and fresh["visit_count"] == 0
    assert page["candidate_coverage"]["inspected_products"] == 1
    assert support[0]["already_inspected"] is True
    assert support[0]["may_add_product_page_evidence"] is False
    assert support[1]["already_inspected"] is False
    assert support[1]["may_add_product_page_evidence"] is True
    assert ledger[ASIN.lower()]["appearance_count"] == 2
    assert ledger[ASIN.lower()]["product_page_evidence"]["product_title"] == "Observed title"
    assert [a["target_id"] for a in page["valid_subactions"]] == original_ids
    assert page["page_text"] == "Original rendered search results"
    assert all(not {"asin", "title", "price"} & a.keys() for a in page["valid_subactions"])
    semantic = _webshop_semantic_action(page, "webshop_click", {"target_id": original_ids[0]})
    assert semantic["asin"] == ASIN.lower()
    assert "target_id" not in semantic
    assert semantic["executable_target_retained"] is False


def test_legacy_target_repair_requires_one_currently_visible_matching_product():
    targets = search_page(ordinal=7)["valid_subactions"]
    canonical = targets[0]["target_id"]
    assert _resolve_visible_target_id(f"open_product:99:{ASIN}", targets) == (
        canonical, "visible_open_product_asin_canonicalization"
    )
    assert _resolve_visible_target_id(canonical, targets) == (canonical, None)
    missing = "open_product:99:B999999999"
    assert _resolve_visible_target_id(missing, targets) == (missing, None)
    ambiguous = targets + [{"kind": "open_product", "target_id": f"open_product:8:{ASIN}"}]
    requested = f"open_product:99:{ASIN}"
    assert _resolve_visible_target_id(requested, ambiguous) == (requested, None)
