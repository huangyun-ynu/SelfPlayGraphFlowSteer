"""Render all observed public facts automatically, without per-block quotas."""
from __future__ import annotations

import copy
from .webshop_memory import WebShopMemory, POLICY, PROJECTION_REVISION, encoded


def _unique_latest(entries, key):
    """Deduplicate equal content, preserving the latest source for each value."""
    result = {}
    for entry in entries:
        value = key(entry)
        result.pop(value, None)
        result[value] = entry
    return list(result.values())


def project_memory(store: WebShopMemory, *, state: dict, remaining: dict | None = None,
                   recent_actions=(), no_progress: int = 0, planned_asin: str = "") -> dict:
    products = store.data["products"]
    asin = str((state.get("product") or {}).get("asin", "")).casefold()
    batch = store.data["batches"][-1] if store.data["batches"] else {}
    ordered = list(dict.fromkeys([k for k in (asin, planned_asin.casefold()) if k in products]
                                + list(products)))
    result = {
        "schema": POLICY, "projection_revision": PROJECTION_REVISION,
        "current": {"asin": asin, "live_choices_source": "action_environment.state",
                    "remaining": remaining or {}, "no_progress_streak": no_progress},
        "candidate_ledger": [], "observed_products": [], "evidence": [],
        "history": {"queries": [], "actions": [], "result_batches": []},
        "memory": {"coverage": store.data["coverage"], "candidate_count": len(products),
                   "recent_batch_count": len(batch.get("candidates", [])),
                   "omitted_candidates": 0, "omitted_section_records": 0,
                   "recent_batch_partial": False, "history_partial": (
                       store.data["coverage"] == "legacy_partial" or any(
                           q["search_count"] is None for q in store.data["queries"])),
                   "delivery": "automatic", "projection_chars": 0,
                   "history_is_not_current_state": True,
                   "observed_products_scope": "historical_public_snapshots_not_current_choices",
                   "full_evidence_retained_in_run_store": True,
                   "fixed_character_quotas": False},
    }
    if asin:
        result["current"]["fact"] = store.fact(asin)
        result["current"]["sections_observed"] = list(products.get(asin, {}).get("sections", {}))

    for index, query in enumerate(store.data["queries"]):
        result["history"]["queries"].append({"query_id": index,
            **{k: copy.deepcopy(v) for k, v in query.items() if k != "key"}})
    for event in store.data["events"]:
        result["history"]["actions"].append({k: copy.deepcopy(v) for k, v in event.items()
                                               if k != "source"})
    result["history"]["result_batches"] = [
        {"query_id": batch.get("query_id"), "event_sequence": batch.get("event_sequence"),
         "candidate_order": [c["asin"] for c in batch["candidates"]]}
        for batch in store.data["batches"]]
    if not store.data["events"] and recent_actions:
        result["history"]["legacy_recent_actions"] = copy.deepcopy(list(recent_actions))
        result["history"]["legacy_action_history_partial"] = True

    for key in ordered:
        p = products[key]
        titles = _unique_latest(p.get("titles", []), lambda e: e["ref"])
        title = store.title(key)
        latest_title_ref = p["titles"][-1]["ref"] if p.get("titles") else None
        card = {"asin": key, "title": title,
                "product_page": store.fact(key)["observation_status"],
                "visit_count": p["visit_count"],
                "query_ids": list(dict.fromkeys(b["query_id"] for b in store.data["batches"]
                    if b.get("query_id") is not None and any(c["asin"] == key for c in b["candidates"])))}
        prior_titles = [store._get(e["ref"]) for e in titles if e["ref"] != latest_title_ref]
        if prior_titles:
            card["other_observed_titles"] = prior_titles
        if p.get("previews"):
            card["search_preview_prices"] = [e["values"] for e in
                _unique_latest(p["previews"], lambda e: encoded(e["values"]))]
            card["preview_is_not_selected_variant"] = True
        result["candidate_ledger"].append(card)
        if p.get("opened"):
            observations = p.get("product_observations") or ([p["last_product_observation"]]
                if p.get("last_product_observation") else [])
            prices = []
            for snapshot in observations:
                value = {k: copy.deepcopy(v) for k, v in snapshot.get("product", {}).items()
                         if k in {"price", "price_min", "price_max", "price_text"}}
                prices.append({"value": value, "state_version": snapshot.get("state_version")})
            prices = _unique_latest(prices, lambda e: encoded(e["value"]))
            choices = []
            for entry in _unique_latest(p.get("options", []), lambda e: e["ref"]):
                choices.append({"values": store._get(entry["ref"]),
                    "state_version": entry.get("state_version"),
                    "evidence_in_store": "partial" if entry.get("partial") else "full"})
            item = {"asin": key, "price_at_observation": prices[-1]["value"] if prices else {},
                    "state_version": p.get("last_product_observation", {}).get("state_version"),
                    "option_values": choices[-1]["values"] if choices else {},
                    "evidence_in_prompt": "full"}
            if len(prices) > 1:
                item["earlier_observed_prices"] = prices[:-1]
            if len(choices) > 1:
                item["earlier_observed_options"] = choices[:-1]
            selections = _unique_latest(p.get("options", []),
                lambda e: encoded(e.get("selected_options_at_observation", {})))
            item["historical_selected_options"] = [
                {"values": copy.deepcopy(e.get("selected_options_at_observation", {})),
                 "state_version": e.get("state_version"), "scope": "historical_not_current"}
                for e in selections]
            result["observed_products"].append(item)

        for name, versions in p.get("sections", {}).items():
            for entry in _unique_latest(versions, lambda e: e["cleaned"]):
                text = store._get(entry["cleaned"])
                record = {"asin": key, "section": name,
                    "observation_status": "observed",
                    "evidence_in_store": "partial" if entry.get("partial") else "full",
                    "evidence_in_prompt": "full", "source": entry["source"],
                    "full_chars": len(text), "excerpts": []}
                effect = state.get("action_effect") or {}
                on_page = (key == asin and state.get("page_type") == "product_section"
                           and effect.get("kind") == "view_" + name
                           and str(state.get("page_text", "")) == store._get(entry["raw"]))
                if on_page:
                    record["current_page_ref"] = "action_environment.state.page_text"
                else:
                    record["excerpts"] = [{"range": [0, len(text)], "text": text}]
                result["evidence"].append(record)

    # Reuse live price/options only when they fully represent the latest snapshot.
    if state.get("page_type") == "product":
        live_options = {}
        for action in state.get("valid_subactions", []):
            if action.get("option_name") and "option_value" in action:
                values = live_options.setdefault(action["option_name"], [])
                if action["option_value"] not in values:
                    values.append(action["option_value"])
        for item in result["observed_products"]:
            if item["asin"] != asin:
                continue
            live_price = {k: v for k, v in (state.get("product") or {}).items()
                          if k in {"price", "price_min", "price_max", "price_text"}}
            if item["option_values"] == live_options:
                item.pop("option_values")
                item["current_options_ref"] = "action_environment.state.valid_subactions"
            if item["price_at_observation"] == live_price:
                item.pop("price_at_observation")
                item["current_price_ref"] = "action_environment.state.product"

    for _ in range(3):
        result["memory"]["projection_chars"] = len(encoded(result))
    return result


def action_decision_support(state: dict) -> list[dict]:
    """Read annotations from the authoritative fact snapshot, never prompt lists."""
    result = []
    for action in state.get("valid_subactions", []):
        if not isinstance(action, dict):
            continue
        entry = {"target_id": action.get("target_id"), "kind": action.get("kind"), "agent_selectable": True}
        if "memory_fact" in action:
            entry.update(copy.deepcopy(action["memory_fact"]))
        if action.get("navigation_effect"):
            entry["navigation_effect"] = action["navigation_effect"]
        result.append(entry)
    return result
