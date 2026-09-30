"""Automatic bounded projection; memory never adds a Worker Action or quota."""
from __future__ import annotations

import copy
import re
from typing import Any
from .webshop_memory import WebShopMemory, POLICY, encoded

MAX_CHARS = 6000
_STOP_WORDS = frozenset('i am looking for want would like find buy a an the it should be and or with that this is are to of in on at have has need must product item please suitable'.split())


def _terms(task: str) -> set[str]:
    return {word for word in re.findall(r"[\w.-]+", task.casefold())
            if word not in _STOP_WORDS and (len(word) >= 3 or word.isdigit())}


def _excerpts(text: str, terms: set[str]) -> list[tuple[int, int]]:
    """Public literal matches select excerpts, never product/requirement judgments.

    Fixed windows preserve exact original text. Include the opening context and
    windows containing task terms; the caller marks all omitted ranges explicitly.
    """
    windows = [(i, min(len(text), i + 420)) for i in range(0, len(text), 420)]
    if not windows:
        return []
    ranked = sorted(windows[1:], key=lambda span: (
        -sum(term in text[span[0]:span[1]].casefold() for term in terms), span[0]))
    return [windows[0], *ranked]


def project_memory(store: WebShopMemory, *, state: dict, remaining: dict | None = None,
                   recent_actions=(), no_progress: int = 0,
                   planned_asin: str = "", max_chars: int = MAX_CHARS) -> dict:
    if max_chars < 900:
        raise ValueError("WebShop memory projection budget must be at least 900 characters")
    products = store.data["products"]
    asin = str((state.get("product") or {}).get("asin", "")).casefold()
    latest = store.data["batches"][-1] if store.data["batches"] else {}
    batch = [c["asin"] for c in latest.get("candidates", [])]
    pinned = [k for k in (asin, planned_asin.casefold()) if k in products]
    ordered = list(dict.fromkeys(pinned + batch))
    result: dict[str, Any] = {
        "schema": POLICY,
        "current": {"asin": asin[:100], "live_choices_source": "action_environment.state",
                    "remaining": remaining or {}, "no_progress_streak": no_progress},
        "candidate_ledger": [], "evidence": [],
        "history": {"recent_queries": [b["query"] for b in store.data["batches"][-3:]]},
        "memory": {"coverage": store.data["coverage"], "candidate_count": len(products),
                   "recent_batch_count": len(batch), "omitted_candidates": len(products),
                   "delivery": "automatic", "history_is_not_current_state": True,
                   "observed_products_scope": "historical_public_snapshots_not_current_choices",
                   "excerpt_selection": "public_order_and_literal_task_terms",
                   "full_evidence_retained_in_run_store": True, "projection_chars": 0,
                   "omitted_section_records": 0, "recent_batch_partial": False},
    }
    result["history"]["recent_actions"] = [
        {"action": a.get("action"), "new_evidence": a.get("new_evidence"),
         **({"rejection": a["rejection_code"]} if a.get("rejection_code") else {})}
        for a in recent_actions[-2:]]
    while len(encoded(result["history"])) > 350:
        result["memory"]["history_partial"] = True
        if result["history"]["recent_queries"]:
            result["history"]["recent_queries"].pop(0)
        elif result["history"]["recent_actions"]:
            result["history"]["recent_actions"].pop(0)
        else:
            break
    if asin:
        result["current"]["fact"] = store.fact(asin)
        result["current"]["sections_observed"] = list(products.get(asin, {}).get("sections", {}))
        while len(encoded(result["current"])) > 800 and result["current"]["sections_observed"]:
            result["current"]["sections_observed"].pop()
            result["current"]["section_index_partial"] = True
    # Leave room for exact accounting even for the smallest supported budget.
    if len(encoded(result)) > max_chars - 80:
        result["current"] = {"asin": asin[:100], "live_choices_source": "action_environment.state"}
        result["history"] = {}
    candidate_budget = min(3000, max(0, max_chars - len(encoded(result)) - 80))
    for key in ordered:
        title = store.title(key)
        card = {"asin": key, "title": title[:240], "product_page": store.fact(key)["observation_status"]}
        if len(title) > 240:
            card.update(title_excerpt=True, full_title_chars=len(title))
        if len(encoded(result["candidate_ledger"] + [card])) <= candidate_budget:
            result["candidate_ledger"].append(card)
    shown = {c["asin"] for c in result["candidate_ledger"]}
    result["memory"]["omitted_candidates"] = len(products) - len(shown)
    result["memory"]["recent_batch_partial"] = bool(set(batch) - shown)

    def add(key: str, value: Any) -> bool:
        trial = dict(result, **{key: value})
        if len(encoded(trial)) <= max_chars - 80:
            result[key] = value
            return True
        return False

    terms = _terms(store.task)
    # Comparing previously opened products requires their public price/variants,
    # including after leaving the page. Provide these automatically; "seen" alone
    # cannot carry the information that justified returning to a candidate.
    history_keys = list(dict.fromkeys(pinned + list(reversed(list(products)))))
    observed_products = []
    request_text = " ".join(store.task.casefold().split())
    for key in history_keys:
        product_record = products[key]
        if not product_record.get("opened") or (key == asin and state.get("page_type") == "product"):
            continue
        snapshot = product_record.get("last_product_observation", {})
        price = {k: v for k, v in snapshot.get("product", {}).items()
                 if k in {"price", "price_min", "price_max", "price_text"}}
        entries = product_record.get("options", [])
        entry = entries[-1] if entries else {}
        values = store._get(entry["ref"]) if entry else {}
        item = {"asin": key, "price_at_observation": price,
                "state_version": snapshot.get("state_version"), "option_values": {},
                "group_counts": {name: len(v) for name, v in values.items()},
                "evidence_in_prompt": "omitted" if values else "full"}
        candidate = observed_products + [item]
        if len(encoded(candidate)) > 1700 or not add("observed_products", candidate):
            continue
        observed_products = candidate
        # Fill every group in rounds. Exact public option strings in the request
        # are literal quotes, not a semantic match/selection recommendation.
        pending = {name: sorted(enumerate(choices), key=lambda pair: (
            -int(" ".join(pair[1].casefold().split()) in request_text),
            -sum(t in pair[1].casefold() for t in terms), pair[0])) for name, choices in values.items()}
        progress = True
        per_product = max(400, 1700 // max(1, min(3, sum(bool(p.get("opened")) for p in products.values()))))
        while progress:
            progress = False
            for name, choices in pending.items():
                if not choices:
                    continue
                _, value = choices.pop(0)
                expanded = copy.deepcopy(observed_products)
                last = expanded[-1]
                last["option_values"].setdefault(name, []).append(value)
                last["evidence_in_prompt"] = "full" if all(
                    len(last["option_values"].get(n, [])) == len(v) for n, v in values.items()) else "excerpt"
                if len(encoded(last)) <= per_product and len(encoded(expanded)) <= 1700 and add("observed_products", expanded):
                    observed_products = expanded
                    progress = True
    # Prioritize current/explicitly planned products. Then cover other observed
    # products in public visit order, round-robin across sections.
    evidence_asins = list(dict.fromkeys(pinned + list(reversed(list(products)))))
    evidence_records = []
    for key in evidence_asins:
        for name, versions in products[key].get("sections", {}).items():
            if not versions:
                continue
            entry = versions[-1]
            text = store._get(entry["cleaned"])
            record = {"asin": key, "section": name, **store.fact(key, name),
                      "source": entry["source"], "full_chars": len(text), "excerpts": []}
            effect = state.get("action_effect", {})
            on_page = (key == asin and state.get("page_type") == "product_section"
                       and effect.get("kind") == "view_" + name
                       and len(str(state.get("page_text", ""))) <= 8000)
            if on_page:
                record.update(evidence_in_prompt="full", current_page_ref="action_environment.state.page_text")
            evidence_records.append((record, text, [] if on_page else _excerpts(text, terms)))
    # Facts and omitted evidence references fit first; chunks share the remainder.
    selected = []
    for record, text, windows in evidence_records:
        if len(selected) >= 3:
            break
        if add("evidence", result["evidence"] + [record]):
            selected.append((len(result["evidence"]) - 1, text, windows))
    made_progress = True
    while made_progress:
        made_progress = False
        for index, text, windows in selected:
            if not windows:
                continue
            start, end = windows.pop(0)
            records = copy.deepcopy(result["evidence"])
            record = records[index]
            record["excerpts"].append({"range": [start, end], "text": text[start:end]})
            record["excerpts"].sort(key=lambda e: e["range"][0])
            retained = sum(e["range"][1] - e["range"][0] for e in record["excerpts"])
            record["evidence_in_prompt"] = "full" if retained == len(text) else "excerpt"
            if add("evidence", records):
                made_progress = True
    result["memory"]["omitted_section_records"] = len(evidence_records) - len(result["evidence"])
    result["memory"]["projection_chars"] = 0
    for _ in range(3):
        result["memory"]["projection_chars"] = len(encoded(result))
    assert len(encoded(result)) <= max_chars
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
