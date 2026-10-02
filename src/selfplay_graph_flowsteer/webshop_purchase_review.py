"""Public evidence for a Director's purchase decision, never a semantic scorer."""
from __future__ import annotations

import copy
import hashlib
import json

POLICY = "director_purchase_review_v1"
GUIDANCE = (
    "Buy Now prepares a purchase for Director review; the environment purchase has not run. "
    "In purchase_evidence.review, report requirement assessments as verified, unknown or "
    "contradicted, citing evidence_refs from candidate_comparison. Include the candidates you "
    "compared and a concise selection_reason. These are your judgments, not runtime verdicts. "
    "The Director can FINISH this proposal or continue the same session. Missing observations "
    "and default search variants do not establish that another selectable variant is absent."
)

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {"type": "array", "maxItems": 20, "items": {
            "type": "object", "properties": {
                "requirement": {"type": "string", "minLength": 1},
                "status": {"type": "string", "enum": ["verified", "unknown", "contradicted"]},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            }, "required": ["requirement", "status", "evidence_refs"], "additionalProperties": False}},
        "candidate_comparison": {"type": "array", "maxItems": 20, "items": {
            "type": "object", "properties": {
                "asin": {"type": "string", "minLength": 1},
                "decision": {"type": "string", "enum": ["selected", "retained", "not_inspected", "rejected"]},
                "reason": {"type": "string"},
            }, "required": ["asin", "decision", "reason"], "additionalProperties": False}},
        "selection_reason": {"type": "string"},
    }, "required": ["requirements", "candidate_comparison", "selection_reason"],
    "additionalProperties": False,
}


def comparison_view(memory, state):
    """Join only this owner's observations; retain discovery order and all candidates."""
    live_asin = str((state.get("product") or {}).get("asin", "")).casefold()
    cards, sources = [], {}
    for asin, product in memory.data["products"].items():
        prefix = asin + "/"
        title = memory.title(asin)
        sources[prefix + "title"] = {"asin": asin, "source": "title", "text": title,
                                      "scope": "observed_title"}
        snapshot = product.get("last_product_observation") or {}
        prices = {k: v for k, v in (snapshot.get("product") or {}).items()
                  if k in {"price", "price_min", "price_max", "price_text"}}
        if not prices and product.get("previews"):
            prices = copy.deepcopy(product["previews"][-1]["values"])
        refs = [prefix + "title"]
        if product.get("opened"):
            options = memory._get(product["options"][-1]["ref"]) if product.get("options") else {}
            for name, value in [("price", prices), ("options", options)]:
                ref = prefix + name
                origin = product["options"][-1] if name == "options" and product.get("options") else snapshot
                sources[ref] = {"asin": asin, "source": name, "value": value,
                                "state_version": origin.get("state_version"),
                                "observation_source": origin.get("source"),
                                "coverage": "partial" if origin.get("partial") else "full",
                                "scope": "historical_observation"}
                refs.append(ref)
        for name, entries in product.get("sections", {}).items():
            if not entries:
                continue
            ref = prefix + name
            latest = entries[-1]
            sources[ref] = {"asin": asin, "source": name,
                           "text": memory._get(latest["cleaned"]),
                           "observation_source": latest["source"],
                           "coverage": "partial" if latest.get("partial") else "full"}
            refs.append(ref)
        # A live selection is separate from historical option availability.
        cards.append({"asin": asin, "title": title, "opened": bool(product.get("opened")),
            "visit_count": product["visit_count"], "preview_is_not_selected_variant": True,
            "options_observation": "observed" if product.get("options") else "unobserved",
            "sections_observed": list(product.get("sections", {})), "evidence_refs": refs,
            "price_at_observation": prices,
            "current_selected_options": copy.deepcopy(state.get("selected_options", {}))
                if asin == live_asin else None})
    if live_asin in memory.data["products"]:
        ref = live_asin + "/price"
        sources[ref] = {"asin": live_asin, "source": "price", "scope": "current_state",
            "state_version": state.get("state_version"),
            "value": {k: v for k, v in (state.get("product") or {}).items()
                      if k in {"price", "price_min", "price_max", "price_text"}}}
    batches = memory.data["batches"]
    changes = []
    previous_order, previous_previews, seen = [], {}, set()
    for batch in batches:
        order = [c["asin"] for c in batch["candidates"]]
        previews = {c["asin"]: c.get("preview") for c in batch["candidates"]}
        current = set(order)
        changes.append({"query_id": batch.get("query_id"),
            "event_sequence": batch.get("event_sequence"), "new_candidate_count": len(current - seen),
            "previously_seen_count": len(current & seen), "same_set_as_previous_batch": current == set(previous_order),
            "order_changed": order != previous_order,
            "preview_changed_count": sum(item is not None and previous_previews.get(asin) is not None and previous_previews[asin] != item
                                         for asin, item in previews.items()),
            "currently_opened_count": sum(bool(memory.data["products"][a].get("opened")) for a in current),
            "currently_unopened_count": sum(not memory.data["products"][a].get("opened") for a in current)})
        seen.update(current)
        previous_order, previous_previews = order, previews
    return {"source": "owner_observed_public_facts_only", "semantic_ranking": False,
            "candidates": cards, "sources": sources, "result_changes": changes,
            "judgments": "The Agent chooses relevance, assessments and the next Action."}


def review_packet(*, task, state, evidence, comparison):
    """Expand genuine citations for the Director and label all model judgments."""
    review = evidence.get("review") or {}
    sources = comparison.get("sources", {})
    assessments, referenced = [], set()
    for item in review.get("requirements", []):
        row = copy.deepcopy(item)
        refs = row.get("evidence_refs", [])
        missing = [ref for ref in refs if ref not in sources]
        row["reference_status"] = "missing" if missing else "present" if refs else "not_supplied"
        row["missing_refs"] = missing
        row["status_is_worker_claim"] = True
        referenced.update(ref for ref in refs if ref in sources)
        assessments.append(row)
    current = str((state.get("product") or {}).get("asin", "")).casefold()
    referenced.update(ref for ref in sources if ref.startswith(current + "/"))
    packet = {"policy": POLICY, "phase": "review_pending", "public_task": task,
        "product": copy.deepcopy(state.get("product", {})),
        "selected_options": copy.deepcopy(state.get("selected_options", {})),
        "state_version": state.get("state_version"), "assessments": assessments,
        "price_precision": "range" if any(k in (state.get("product") or {}) for k in ("price_min", "price_max")) else "observed_fields_only",
        "public_completion_path": {"buy_action_cost": 1, "finish_environment_action_cost": 0,
            "additional_investigation_cost": "Depends on Actions the Director and Worker choose; no path is invented."},
        "assessment_status": "provided" if assessments else "not_supplied",
        "worker_verified_claims": copy.deepcopy(evidence["verified_requirements"]),
        "worker_unresolved_claims": copy.deepcopy(evidence["unresolved_constraints"]),
        "selection_reason": review.get("selection_reason", ""),
        "worker_candidate_comparison": copy.deepcopy(review.get("candidate_comparison", [])),
        "observed_candidate_facts": copy.deepcopy(comparison.get("candidates", [])),
        "public_evidence": {ref: copy.deepcopy(sources[ref]) for ref in sorted(referenced)},
        "result_changes": copy.deepcopy(comparison.get("result_changes", [])),
        "evidence_delivery": "complete_for_referenced_and_current_product_sources",
        "semantic_verdict": "Director decision required; runtime checks provenance, not satisfaction",
        "choices": "FINISH commits this exact proposal; RUN_AGENT resumes this session for work you assign."}
    binding = [current, packet["selected_options"], packet["state_version"], packet["public_evidence"]]
    packet["proposal_fingerprint"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return packet
