"""Offline reproductions of audited WebShop issues; no model or live service calls."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from selfplay_graph_flowsteer.delegation import delegation_task_alignment_issue
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import (
    _annotate_webshop_product_state,
    _annotate_webshop_search_state,
    _webshop_action_decision_support,
    _webshop_section_evidence,
)
from selfplay_graph_flowsteer.webshop import WebShopSessionLifecycle
from selfplay_graph_flowsteer.webshop_sidecar import WebShopSession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new audit file; historical results are preserved")
    args = parser.parse_args()
    results = {}
    asin = "B012345678"
    lifecycle = WebShopSessionLifecycle(None, search_observation_mode="legacy")
    search = lifecycle._bounded({
        "page_type": "search_results", "page_text": "Visible product",
        "valid_subactions": [{"target_id": f"open_product:1:{asin}", "kind": "open_product", "asin": asin, "label": "Product", "title": "Product"}],
    })
    inspections = {asin.lower(): {"asin": asin.lower(), "visit_count": 1}}
    _annotate_webshop_search_state(search, product_inspections=inspections, repeated_public_evidence=False)
    support = _webshop_action_decision_support(search, progress={"product_inspections": list(inspections.values())})
    assert search["valid_subactions"][0]["inspection_status"] == "inspected"
    assert support[0]["already_inspected"] is True
    assert "asin" not in search["valid_subactions"][0]
    results["legacy_identity_loss"] = {
        "status": "fixed",
        "known_inspected_asin": asin, "actual_coverage": search["candidate_coverage"],
        "action_support": support[0], "expected_inspected_products": 1,
    }

    products = SimpleNamespace(product=lambda value: {"name": "Product", "options": {}})
    worker = SimpleNamespace(request=lambda operation, payload: {
        "available_actions": ["click[back to search]", f"click[{asin}]"],
        "observation_text": "Search results", "terminal": False,
    })
    session = WebShopSession("offline-repro", worker, products, "goal-1", current_asin=asin)
    public = session._project({
        "available_actions": ["click[buy now]", "click[< prev]"],
        "observation_text": "Product page", "terminal": False,
    }, action_kind="reset")
    _annotate_webshop_product_state(public, product_inspections={})
    back = next(a for a in public["valid_subactions"] if a["target_id"].startswith("previous_page:"))
    next_page = session.click(back["target_id"])
    assert back["navigation_effect"] == "return_to_current_product_page"
    assert next_page["page_type"] == "search_results"
    results["incorrect_product_back_semantics"] = {
        "before": public["page_type"], "declared_effect": back["navigation_effect"],
        "actual_next_page": next_page["page_type"],
    }

    client = SimpleNamespace(
        create_session=lambda goal_id, seed: {"session_id": "offline-repro", **public},
        close_session=lambda session_id: None,
    )
    lifecycle = WebShopSessionLifecycle(client)
    lifecycle.bind_task(TaskSpec("repro", "Buy a product", metadata={"goal_id": "goal-1"}))
    state = lifecycle.begin_execution(agent_id="owner", seed=0, revision=False)
    lifecycle.end_execution()
    lifecycle.set_committer("owner")
    authority = lifecycle.closure_budget_context("owner")
    assert "remaining_steps" not in state
    assert authority == {"eligible": False, "reason": "official_remaining_steps_unknown"}
    results["closure_contract_mismatch"] = {"sidecar_has_remaining_steps": False, "closure_authority": authority}

    rendered = "Instruction: [SEP] Buy waterproof shoes [SEP] Back to Search [SEP] < Prev [SEP] Description: cotton shoes"
    retained = _webshop_section_evidence(rendered)
    assert "Buy waterproof shoes" in retained
    results["request_mixed_into_section_evidence"] = {
        "rendered_section": rendered, "retained_as_product_evidence": retained,
        "expected_section_body": "Description: cotton shoes",
    }

    path = Path("state/formal-eval/webshop-env-feedback-c24-20260923/trajectories/f437c430f9eb772f16d19807.json")
    trajectory = json.loads(path.read_text())
    event = next(e["payload"] for e in trajectory["events"] if e["payload"].get("rejection_code") == "responsibility_violation")
    fields = json.loads(event["raw_action"])
    raw_issue = delegation_task_alignment_issue(fields, public_task=trajectory["task"]["prompt"], dataset="webshop")
    # Canvas passes its decorated task (including Submission contract), not just
    # the original shopping request, into the lexical alignment check.
    issue = delegation_task_alignment_issue(fields, public_task=trajectory["director_run"]["task"], dataset="webshop")
    assert raw_issue is None
    assert issue is not None
    assert set(issue.details["novel_terms"]) == {"meet", "requirements"}
    results["delegation_false_positive"] = {
        "task": trajectory["task"]["task_id"], "objective": fields["objective"],
        "issue_with_original_task": None, "issue_with_submission_contract": asdict(issue),
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
