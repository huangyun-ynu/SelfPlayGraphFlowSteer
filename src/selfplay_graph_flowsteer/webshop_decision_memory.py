"""Optional, owner-local Worker judgments. Never environment facts or Actions."""
from __future__ import annotations

import copy
from dataclasses import replace

from jsonschema import Draft202012Validator

from .webshop_memory import digest
from .webshop_purchase_review import comparison_view

POLICY = "decision_journal_v1"
GUIDANCE = """
Continuous decision memory is supplied automatically in decision_memory. It contains
your revisable judgments, NOT environment facts. When a candidate assessment or next
check changes, you may attach decision_update to an existing native shopping tool.
Do not emit a separate think/memory Action or a long reasoning transcript. Omit the
field when nothing changes. Use brief summaries (one to three sentences).
candidate_updates entries use asin, an exact task_quote from the original public
task, assessment (verified/unknown/contradicted), disposition
(selected/retained/not_inspected/rejected), evidence_refs, and summary. Copy refs
from decision_memory.available_evidence_refs; refs establish source, not correctness.
An unread section is unknown, not a contradiction. Updates use evidence ALREADY
observed before this Action; assess newly returned evidence on the next normal call.
Use operation='retract' with asin and task_quote to withdraw a judgment. next_check
is optional {asin, section, question}, or null to clear an intention. It never runs
an Action. Replace mistaken judgments when evidence changes; needs_recheck entries
are historical judgments needing review. Only live state establishes selected
options, staged purchase or completed purchase. Memory notes do not replace normal
purchase_evidence, completion_plan or Director decisions, and cost no extra Action.
"""

ENTRY_SCHEMA = {
    "type": "object", "properties": {
        "asin": {"type": "string", "minLength": 1},
        "task_quote": {"type": "string", "minLength": 1},
        "operation": {"type": "string", "enum": ["upsert", "retract"]},
        "assessment": {"type": "string", "enum": ["verified", "unknown", "contradicted"]},
        "disposition": {"type": "string", "enum": ["selected", "retained", "not_inspected", "rejected"]},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "minLength": 1},
    }, "required": ["asin", "task_quote"], "additionalProperties": False,
}
NEXT_SCHEMA = {"type": ["object", "null"], "properties": {
    "asin": {"type": "string", "minLength": 1},
    "section": {"type": "string", "minLength": 1},
    "question": {"type": "string", "minLength": 1},
}, "required": ["asin", "section", "question"], "additionalProperties": False}
UPDATE_SCHEMA = {"type": "object", "description": "Optional brief revisable judgments; no extra environment Action.",
    "properties": {"candidate_updates": {"type": "array", "items": ENTRY_SCHEMA},
                   "next_check": NEXT_SCHEMA}, "additionalProperties": False}


def validate_policy(policy, memory_policy, execution_policy):
    if policy not in {"off", POLICY}:
        raise ValueError("unknown webshop.worker_decision_memory_policy")
    if policy != "off" and (memory_policy != "factual_memory_v2" or execution_policy != "graph_tools_v1"):
        raise ValueError("decision memory requires factual_memory_v2 and graph_tools_v1")


def action_specs_with_decisions(specs):
    result = []
    for spec in specs:
        if spec.name in {"webshop_search", "webshop_click"}:
            parameters = copy.deepcopy(spec.parameters)
            parameters.setdefault("properties", {})["decision_update"] = copy.deepcopy(UPDATE_SCHEMA)
            spec = replace(spec, parameters=parameters)
        result.append(spec)
    return result


def _normalized(text):
    return " ".join(text.split()).casefold()


def _source_fingerprint(source):
    # Observation IDs and navigation versions can change without changing evidence.
    return digest({k: source[k] for k in ("asin", "source", "value", "text", "coverage") if k in source})


class WebShopDecisionMemory:
    def __init__(self, journal, *, memory, assigned_task, state):
        self.memory = memory
        handle = journal.get(POLICY)
        if (isinstance(handle, dict) and handle.get("binding") == memory.binding
                and handle.get("store") == memory.directory.name):
            self.data = memory._get(handle["index"])
            if self.data.get("schema") != POLICY or self.data.get("binding") != memory.binding:
                raise ValueError("WebShop decision memory binding mismatch")
        else:
            self.data = {"schema": POLICY, "binding": memory.binding, "current": {},
                         "events": [], "processed": {}, "next_check": None,
                         "last_receipt": None, "last_diagnostics": []}
        assigned_hash = digest(assigned_task)
        if self.data.get("assigned_task_hash") not in {None, assigned_hash}:
            for entry in self.data["current"].values():
                self._mark(entry, "assigned_task_changed")
            if self.data["next_check"]:
                self.data["next_check"]["needs_recheck"] = True
        self.data["assigned_task_hash"] = assigned_hash
        self.refresh(state)

    @staticmethod
    def _mark(entry, reason):
        entry["needs_recheck"] = True
        reasons = entry.setdefault("recheck_reasons", [])
        if reason not in reasons:
            reasons.append(reason)

    def sources(self, state):
        return comparison_view(self.memory, state)["sources"]

    def refresh(self, state):
        sources = self.sources(state)
        live = str((state.get("product") or {}).get("asin", "")).casefold()
        for entry in self.data["current"].values():
            for ref, fingerprint in entry["source_fingerprints"].items():
                if ref not in sources or _source_fingerprint(sources[ref]) != fingerprint:
                    self._mark(entry, "referenced_evidence_changed")
            product = self.memory.data["products"].get(entry["asin"], {})
            if sorted(product.get("sections", {})) != entry["sections_observed"]:
                self._mark(entry, "new_public_section_observed")
            if live == entry["asin"] and entry["selected_options_at_judgment"] != state.get("selected_options", {}):
                self._mark(entry, "live_options_changed")
        return sources

    def accept(self, update, *, state, event_key, action_name, call_id, eligible=True):
        """Consume metadata before the Action. Invalid notes never reject the Action."""
        if event_key in self.data["processed"]:
            return copy.deepcopy(self.data["processed"][event_key])
        sequence = len(self.data["events"]) + 1
        diagnostics, accepted = [], 0
        sources = self.refresh(state)
        event = {"sequence": sequence, "event_key": event_key, "call_id": call_id,
                 "action": action_name, "state_version_before": state.get("state_version"),
                 "source": "worker_judgment", "raw_update": copy.deepcopy(update),
                 "assigned_task_hash": self.data["assigned_task_hash"]}
        if not eligible:
            diagnostics.append("update_ignored_core_action_invalid_or_deferred")
        elif not isinstance(update, dict) or set(update) - set(UPDATE_SCHEMA["properties"]):
            diagnostics.append("invalid_decision_update_structure")
        else:
            entries = update.get("candidate_updates", [])
            if not isinstance(entries, list):
                diagnostics.append("candidate_updates_must_be_array")
                entries = []
            for index, raw in enumerate(entries):
                reason = self._entry_error(raw, sources)
                if reason:
                    diagnostics.append(f"candidate_updates[{index}]: {reason}")
                    continue
                row = copy.deepcopy(raw)
                row["asin"] = row["asin"].casefold()
                key = digest([row["asin"], _normalized(row["task_quote"])])
                if row.get("operation") == "retract":
                    self.data["current"].pop(key, None)
                else:
                    row.update(source="worker_judgment", sequence=sequence,
                        state_version_at_judgment=state.get("state_version"),
                        reference_status="present" if row["evidence_refs"] else "not_supplied",
                        semantic_truth_verified_by_runtime=False, needs_recheck=False,
                        source_fingerprints={ref: _source_fingerprint(sources[ref]) for ref in row["evidence_refs"]},
                        sections_observed=sorted(self.memory.data["products"][row["asin"]].get("sections", {})),
                        selected_options_at_judgment=copy.deepcopy(state.get("selected_options", {}))
                            if str((state.get("product") or {}).get("asin", "")).casefold() == row["asin"] else None)
                    self.data["current"][key] = row
                accepted += 1
            if "next_check" in update:
                nxt = update["next_check"]
                if not Draft202012Validator(NEXT_SCHEMA).is_valid(nxt):
                    diagnostics.append("invalid_next_check_structure")
                elif nxt is not None and nxt["asin"].casefold() not in self.memory.data["products"]:
                    diagnostics.append("next_check_candidate_not_observed")
                else:
                    self.data["next_check"] = None if nxt is None else {
                        **copy.deepcopy(nxt), "asin": nxt["asin"].casefold(),
                        "source": "worker_intent", "sequence": sequence, "needs_recheck": False,
                        "execution": "not_inferred_from_intent"}
                    accepted += 1
        result = {"accepted_updates": accepted, "diagnostics": diagnostics,
                  "blocks_action": False, "sequence": sequence}
        event["result"] = copy.deepcopy(result)
        self.data["events"].append(event)
        self.data["processed"][event_key] = result
        self.data["last_diagnostics"] = diagnostics
        return copy.deepcopy(result)

    def _entry_error(self, row, sources):
        if not Draft202012Validator(ENTRY_SCHEMA).is_valid(row):
            return "invalid_entry_structure"
        asin = row["asin"].casefold()
        if asin not in self.memory.data["products"]:
            return "candidate_not_observed"
        if not _normalized(row["task_quote"]) or _normalized(row["task_quote"]) not in _normalized(self.memory.task):
            return "task_quote_not_in_original_task"
        if row.get("operation") == "retract":
            return None
        if any(k not in row for k in ("assessment", "disposition", "evidence_refs", "summary")):
            return "missing_judgment_fields"
        if any(ref not in sources or sources[ref]["asin"] != asin for ref in row["evidence_refs"]):
            return "evidence_not_observed_for_this_candidate"
        return None

    def receipt(self, *, state, observation, action_name, call_id, event_key):
        self.refresh(state)
        # Only trusted runtime/environment values; do not copy model report fields.
        receipt = {"source": "runtime_receipt", "call_id": call_id, "action": action_name,
                   "status": observation.get("status"), "state_version": state.get("state_version"),
                   "steps": state.get("steps"), "error_code": (observation.get("error") or {}).get("code"),
                   "selected_options": copy.deepcopy(state.get("selected_options", {})),
                   "purchase_review_pending": bool(state.get("purchase_review_pending")),
                   "commit_pending": bool(state.get("commit_pending")),
                   "purchased": bool(state.get("purchased"))}
        self.data["last_receipt"] = receipt
        for event in reversed(self.data["events"]):
            if event["event_key"] == event_key:
                event["attached_action_receipt"] = copy.deepcopy(receipt)
                nxt = self.data["next_check"]
                if nxt and nxt["sequence"] == event["sequence"]:
                    nxt["attached_action_receipt"] = copy.deepcopy(receipt)
                break

    def project(self, state):
        sources = self.refresh(state)
        return {"policy": POLICY, "source": "revisable_worker_judgments_not_facts",
                "delivery": "automatic", "fixed_character_quotas": False,
                "current_judgments": [{k: copy.deepcopy(v) for k, v in entry.items()
                    if k not in {"source_fingerprints", "sections_observed", "selected_options_at_judgment", "operation"}}
                    for entry in self.data["current"].values()],
                "available_evidence_refs": list(sources),
                "next_check": copy.deepcopy(self.data["next_check"]),
                "last_action_receipt": copy.deepcopy(self.data["last_receipt"]),
                "metadata_diagnostics": copy.deepcopy(self.data["last_diagnostics"]),
                "authority": "Current action_environment.state alone authorizes Actions and establishes purchase status."}

    def persist(self):
        return {"schema": POLICY, "binding": self.memory.binding,
                "store": self.memory.directory.name, "index": self.memory._put(self.data)}
