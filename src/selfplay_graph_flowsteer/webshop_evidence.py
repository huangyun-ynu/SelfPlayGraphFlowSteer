"""Opt-in public evidence ledger. References are checked; semantic claims are not scored.

This module deliberately accepts public task text and projected observations only.
It never receives a TaskSpec, a hidden goal, a verifier payload, or a reward.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

EVIDENCE_PROFILE = "public_evidence_v1"
REVIEW_PROFILE = "public_evidence_review_v1"
EVIDENCE_V2_PROFILE = "public_evidence_v2"
REVIEW_V2_PROFILE = "public_evidence_review_v2"
REVIEW_PROFILES = frozenset({REVIEW_PROFILE, REVIEW_V2_PROFILE})
DEFERRED_ANNOTATION_PROFILES = frozenset({EVIDENCE_V2_PROFILE, REVIEW_V2_PROFILE})
EVIDENCE_PROFILES = frozenset({EVIDENCE_PROFILE, REVIEW_PROFILE, *DEFERRED_ANNOTATION_PROFILES})


def _text(value: object) -> str:
    return " ".join(str(value).split())


def _asin(state: dict) -> str:
    return str((state.get("product") or {}).get("asin", ""))


def public_price_check(quote: str, state: dict) -> dict | None:
    match = re.search(r"(under|below|(?:less|lower)\s+than|no\s+more\s+than|at\s+most|maximum(?:\s+of)?)"
                      r"\s*(?:\$|usd\s*)?(\d+(?:\.\d+)?)", quote, re.I)
    if not match:
        return None  # Other price expressions remain a Worker semantic judgment.
    ceiling = float(match[2])
    inclusive = match[1].lower().startswith(("no", "at", "maximum"))
    product = state.get("product") or {}
    low, high = product.get("price_min"), product.get("price_max")
    # A range takes precedence: its lower endpoint cannot certify affordability.
    if not all(type(value) in (int, float) for value in (low, high)):
        low = high = product.get("price")
    status = "unknown"
    if all(type(value) in (int, float) for value in (low, high)):
        if (high <= ceiling) if inclusive else (high < ceiling):
            status = "supported"
        elif (low > ceiling) if inclusive else (low >= ceiling):
            status = "conflict"
    return {"status": status, "ceiling": ceiling, "inclusive": inclusive,
            "public_price_min": low, "public_price_max": high}


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


SHORT_TEXT = {"type": "string", "minLength": 1, "maxLength": 240}
REFERENCE_SCHEMA = _object({"source_id": SHORT_TEXT, "quote": SHORT_TEXT}, ["source_id", "quote"])
REQUIREMENT_SCHEMA = _object({
    "id": SHORT_TEXT, "quote": SHORT_TEXT,
    "kind": {"enum": ["type", "attribute", "option", "price"]},
    "strength": {"enum": ["required", "preference", "uncertain"]},
}, ["id", "quote", "kind", "strength"])
CHECK_SCHEMA = _object({
    "requirement_id": SHORT_TEXT,
    "status": {"enum": ["supported", "conflict", "unknown", "title_only"]},
    "references": {"type": "array", "items": REFERENCE_SCHEMA, "maxItems": 3},
    "option_group": SHORT_TEXT, "option_value": SHORT_TEXT,
}, ["requirement_id", "status", "references"])
DECISION_SCHEMA = _object({
    "requirements": {"type": "array", "items": REQUIREMENT_SCHEMA, "maxItems": 12},
    "candidate": _object({
        "asin": SHORT_TEXT,
        "checks": {"type": "array", "items": CHECK_SCHEMA, "maxItems": 12},
    }, ["asin", "checks"]),
    "best_candidate_asin": SHORT_TEXT,
    "query_reason": SHORT_TEXT,
}, [])


@dataclass
class PublicEvidenceLedger:
    public_request: str
    requirements: dict[str, dict] = field(default_factory=dict)
    candidates: dict[str, dict] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)
    current_sources: list[dict] = field(default_factory=list)
    best_candidate_asin: str = ""
    last_query: str = ""
    query_reason: str = ""

    def observe(self, state: dict) -> None:
        """Index whole public fields; selected options remain live state, not memory."""
        self.current_sources = []
        if any(state.get(k) for k in ("purchased", "terminal", "done")):
            return
        asin = _asin(state)
        if asin:
            record = self.candidates.setdefault(asin, {"asin": asin, "checks": {}})
            record["recovery_query"] = self.last_query
            record["last_page_type"] = state.get("page_type")
            record["last_state_version"] = state.get("state_version")
            # The environment repeats the shopping instruction on every page.
            # A requirement quoted from that header is not product evidence.
            page_text = str(state.get("page_text", ""))
            page_text = re.sub(r"^\s*Instruction:\s*\[SEP\].*?\[SEP\]\s*", "", page_text, count=1, flags=re.S)
            fields = {"page_text": page_text,
                      "product": json.dumps(state.get("product", {}), ensure_ascii=False)}
            for name, value in fields.items():
                source = {"asin": asin, "page_type": state.get("page_type"),
                          "state_version": state.get("state_version"), "field": name,
                          "text": str(value)}
                source_id = "ev_" + hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
                self.sources[source_id] = source
                self.current_sources.append({"source_id": source_id,
                                             **{k: v for k, v in source.items() if k != "text"}})
        # Pin current and Worker-selected best; older candidates keep their cited
        # verbatim excerpts in checks, independent of runtime prose compression.
        while len(self.candidates) > 6:
            removable = next((key for key in self.candidates
                              if key not in {asin, self.best_candidate_asin}), None)
            if removable is None:
                break
            del self.candidates[removable]
        pinned = {ref["source_id"] for c in self.candidates.values()
                  for check in c["checks"].values() for ref in check.get("references", [])}
        current = {item["source_id"] for item in self.current_sources}
        for key in list(self.sources):
            if len(self.sources) <= 64:
                break
            if key not in pinned | current:
                del self.sources[key]

    def updated(self, decision: object, state: dict) -> PublicEvidenceLedger:
        """Validate on a copy, so preflight never changes the live ledger."""
        result = copy.deepcopy(self)
        if decision is None:
            decision = {}
        if not isinstance(decision, dict):
            raise ValueError("decision must be an object")
        for item in decision.get("requirements", []):
            key = item["id"]
            if not _text(item["quote"]) or _text(item["quote"]) not in _text(self.public_request):
                raise ValueError("requirement quote must be copied from the public request")
            if key in result.requirements and result.requirements[key] != item:
                raise ValueError("requirement IDs are immutable; add a new ID instead")
            result.requirements[key] = copy.deepcopy(item)
        if not result.requirements or len(result.requirements) > 12:
            raise ValueError("declare 1..12 public requirements in decision.requirements before the first action")
        candidate = decision.get("candidate")
        if candidate is not None:
            asin = candidate["asin"]
            if asin not in result.candidates:
                raise ValueError("candidate must have been publicly opened in this session; opened: "
                                 + ", ".join(result.candidates))
            checks = result.candidates[asin]["checks"]
            for check in candidate["checks"]:
                key = check["requirement_id"]
                if key not in result.requirements:
                    raise ValueError("unknown requirement_id")
                refs = check["references"]
                if check["status"] != "unknown" and not refs:
                    raise ValueError("supported/conflict/title_only assessments require a public reference")
                for ref in refs:
                    source = result.sources.get(ref["source_id"])
                    if not source:
                        raise ValueError("unknown source_id; copy an ID from public_decision.current_sources or retained check references")
                    if source["asin"] != asin:
                        raise ValueError(f"reference source belongs to {source['asin']}, not candidate {asin}")
                    if not _text(ref["quote"]) or _text(ref["quote"]) not in _text(source["text"]):
                        raise ValueError("reference must quote an observed source for this candidate")
                group, value = check.get("option_group"), check.get("option_value")
                if bool(group) != bool(value):
                    raise ValueError("option_group and option_value must be supplied together")
                if group:
                    if asin != _asin(state):
                        raise ValueError("option mappings must use the current product")
                    valid = any(a.get("kind") == "select_option"
                                and _text(a.get("option_name", "")).casefold() == _text(group).casefold()
                                and _text(a.get("option_value", a.get("label", ""))).casefold() == _text(value).casefold()
                                for a in state.get("valid_subactions", []) if isinstance(a, dict))
                    if not valid:
                        raise ValueError("option group/value must exist on the current public action surface")
                checks[key] = copy.deepcopy(check)
        best = decision.get("best_candidate_asin")
        if best is not None:
            if best not in result.candidates:
                raise ValueError("best candidate must have been opened in this session")
            result.best_candidate_asin = best
        if decision.get("query_reason"):
            result.query_reason = str(decision["query_reason"])
        return result

    def candidate_checks(self, asin: str, state: dict) -> list[dict]:
        result = []
        record = self.candidates.get(asin, {})
        for key, requirement in self.requirements.items():
            check = copy.deepcopy(record.get("checks", {}).get(key, {
                "requirement_id": key, "status": "unknown", "references": []}))
            if requirement["kind"] == "option" and check["status"] == "supported":
                group, value = check.get("option_group"), check.get("option_value")
                selected = {_text(k).casefold(): _text(v).casefold()
                            for k, v in (state.get("selected_options") or {}).items()}
                if (asin != _asin(state) or not group or not value
                        or selected.get(_text(group).casefold()) != _text(value).casefold()):
                    check["status"] = "unknown"
                    check["readback"] = "requested option is not selected on the current product"
                else:
                    check["readback"] = {"asin": asin, "state_version": state.get("state_version"),
                                         "group": group, "selected": value}
            if requirement["kind"] == "price" and asin == _asin(state):
                price = public_price_check(requirement["quote"], state)
                if price:
                    check["public_price_check"] = price
                    if check["status"] == "supported" and price["status"] != "supported":
                        check["status"] = price["status"]
            result.append(check)
        return result

    def context(self, state: dict) -> dict:
        return {"policy": EVIDENCE_PROFILE,
                "semantics": "Worker semantic assessments; runtime verifies quotes and live option readback only",
                "requirements": list(self.requirements.values()),
                "current_sources": copy.deepcopy(self.current_sources),
                "best_candidate_asin": self.best_candidate_asin,
                "query_reason": self.query_reason,
                "candidates": [{**{k: v for k, v in c.items() if k != "checks"},
                                "checks": self.candidate_checks(asin, state)}
                               for asin, c in self.candidates.items()]}

    def purchase(self, value: object, state: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError("Buy Now requires purchase_evidence")
        asin = _asin(state)
        if not asin or value.get("asin") != asin or value.get("state_version") != state.get("state_version"):
            raise ValueError("purchase_evidence must bind the current asin and state_version")
        checks = self.candidate_checks(asin, state)
        declared = self.candidates.get(asin, {}).get("checks", {})
        if any(key not in declared for key in self.requirements):
            raise ValueError("explicitly assess every public requirement, using unknown when evidence is missing")
        unresolved = [c for c in checks if c["status"] != "supported"]
        if unresolved and not _text(value.get("accept_unresolved_reason", "")):
            raise ValueError("resolve remaining checks or give accept_unresolved_reason explaining the final choice under the action budget")
        return {"asin": asin, "state_version": state.get("state_version"),
                "checks": checks, "requirements": list(self.requirements.values()),
                "unresolved_requirement_ids": [c["requirement_id"] for c in unresolved],
                "accept_unresolved_reason": value.get("accept_unresolved_reason", ""),
                "verified_requirements": [self.requirements[c["requirement_id"]]["quote"]
                                          for c in checks if c["status"] == "supported"],
                "unresolved_constraints": [self.requirements[c["requirement_id"]]["quote"]
                                           for c in unresolved]}


def extend_tool_schema(schema: dict, *, purchase: bool, review: bool = False) -> dict:
    result = copy.deepcopy(schema)
    result["properties"]["decision"] = copy.deepcopy(DECISION_SCHEMA)
    if purchase:
        result["properties"]["purchase_evidence"] = _object({
            "asin": SHORT_TEXT, "state_version": {"type": "integer", "minimum": 0},
            "accept_unresolved_reason": SHORT_TEXT,
        }, ["asin", "state_version"])
        if review:
            result["properties"]["purchase_evidence"]["properties"].update({
                "review_receipt": SHORT_TEXT, "review_reason": SHORT_TEXT,
            })
    return result
