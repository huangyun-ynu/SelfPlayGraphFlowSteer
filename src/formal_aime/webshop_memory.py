"""Owner-local, lossless memory of public WebShop observations.

Content-addressed files belong to the run (never /tmp). Journals and artifacts
carry immutable index references; copying the directory preserves recovery.
Only references reachable from this owner's index can be read by the Worker.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .webshop_identity import visible_product_asin

POLICY = "factual_memory_v2"
JOURNAL_KEY = "factual_memory_v2"
CLEANER_VERSION = "public_section_prefix_v2"
PROJECTION_REVISION = "structured_facts_v1"
GUIDANCE = (
    "WebShop memory is supplied automatically from this owner's observed public information. "
    "candidate_ledger preserves all observed candidates; history distinguishes searches from navigation. "
    "Repeated searches and visits are factual counts, not instructions to avoid necessary navigation. "
    "Observed means visited, not that a requirement is satisfied. Search previews describe display "
    "variants, not your selected options. Previously observed options and sections are supplied in full. "
    "Only CURRENT state_version, selected_options and valid_subactions authorize shopping Actions. "
    "Historical options and evidence references are not executable target IDs. "
    "Unobserved and unknown are distinct; missing prompt text does not mean missing evidence."
)


def encoded(value: Any) -> str:
    # Match the actual context serializer, including its spaces and escaping.
    return json.dumps(value, ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def section_name(state: dict, arguments: dict) -> str:
    target = str(arguments.get("target_id", ""))
    if target.startswith("view_"):
        return target.split(":", 1)[0][5:].casefold()
    effect = state.get("action_effect", {})
    kind = str(effect.get("kind", "")) if isinstance(effect, dict) else ""
    return kind[5:].casefold() if kind.startswith("view_") else ""


def clean_section(raw: str, task: str) -> dict:
    """Remove an exact task prefix and contiguous known navigation only.

    Keep an exact slice of the original body, including units, negation, HTML,
    and separators. Unknown framing is retained rather than guessed away.
    """
    # A tokenizer with offsets also handles repeated whitespace and newlines.
    chunks = []
    start = 0
    for delimiter in re.finditer(r"\s*\[SEP\]\s*|\r?\n", raw):
        chunks.append((start, delimiter.start(), raw[start:delimiter.start()].strip()))
        start = delimiter.end()
    chunks.append((start, len(raw), raw[start:].strip()))
    nonempty = [(a, b, s) for a, b, s in chunks if s]
    normalize = lambda s: " ".join(s.split()).casefold()
    offset = 0
    status = "unrecognized_prefix_retained"
    if len(nonempty) >= 2 and normalize(nonempty[0][2]) == "instruction:":
        if normalize(nonempty[1][2]) == normalize(task):
            index = 2
            while index < len(nonempty) and normalize(nonempty[index][2]) in {
                "back to search", "< prev", "[button] back to search [button_]",
                "[button] < prev [button_]",
            }:
                index += 1
            offset = nonempty[index][0] if index < len(nonempty) else len(raw)
            status = "verified_prefix_removed"
    return {"text": raw[offset:], "raw_range": [offset, len(raw)],
            "cleaner": CLEANER_VERSION, "parse_status": status}


class WebShopMemory:
    def __init__(self, journal: dict, *, owner: str, task: str, session: str,
                 root: Path | None = None):
        self.owner, self.task, self.session = owner, task, session
        self.binding = digest([owner, task, session])
        self.root = (root or Path(os.environ.get("SPGFS_WEBSHOP_MEMORY_DIR", "state/webshop-memory"))).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        handle = journal.get(JOURNAL_KEY)
        if isinstance(handle, dict) and handle.get("binding") == self.binding:
            self.directory = self.root / handle["store"]
            # Relocatable relative references; neither a model nor an ASIN supplies paths.
            if self.directory.parent != self.root:
                raise ValueError("invalid WebShop memory directory")
            self.data = self._get(handle["index"])
            if self.data["binding"] != self.binding:
                raise ValueError("WebShop memory owner/session mismatch")
        else:
            self.directory = self.root / uuid.uuid4().hex
            self.directory.mkdir()
            self.data = {"schema": POLICY, "binding": self.binding, "owner": owner,
                         "coverage": "complete", "products": {}, "batches": [],
                         "observations": []}
            # A mismatched v2 handle belongs to another owner/task/session;
            # never resurrect its legacy mirrors into the new session.
            if (not handle and journal.get("schema_version")
                    and journal.get("owner_agent", owner) == owner):
                self._migrate(journal)
        self._upgrade_format()

    @staticmethod
    def query_key(query: str) -> str:
        return " ".join(query.casefold().split())

    def _upgrade_format(self) -> None:
        """Old batches are observations, not evidence of exact search counts."""
        if self.data.get("format_version") == 3:
            return
        self.data.update(format_version=3, events=[], queries=[], current_query=None)
        by_query = {}
        for batch in self.data["batches"]:
            query = str(batch.get("query", "")).strip()
            if not query:
                batch["query_id"] = None
                continue
            key = self.query_key(query)
            if key not in by_query:
                by_query[key] = len(self.data["queries"])
                self.data["queries"].append({"query": query, "key": key,
                    "search_count": None, "recorded_batches": 0, "new_search_count": 0})
            index = by_query[key]
            self.data["queries"][index]["recorded_batches"] += 1
            batch["query_id"] = index
            self.data["current_query"] = index

    def _query(self, query: str) -> int:
        key = self.query_key(query)
        for index, item in enumerate(self.data["queries"]):
            if item["key"] == key:
                return index
        self.data["queries"].append({"query": query, "key": key,
            "search_count": 0, "recorded_batches": 0})
        return len(self.data["queries"]) - 1

    @staticmethod
    def _action(action: str, arguments: dict, state: dict) -> dict:
        target = str(arguments.get("target_id", ""))
        kind = "search" if action == "webshop_search" else target.split(":", 1)[0]
        entry = {"kind": kind}
        if action == "webshop_search":
            entry["query"] = str(arguments.get("query", ""))
        asin = str((state.get("product") or {}).get("asin", "")).casefold()
        if not asin and kind == "open_product":
            match = re.search(r"([a-zA-Z0-9]{10})$", target)
            asin = match.group(1).casefold() if match else ""
        if asin:
            entry["asin"] = asin
        effect = state.get("action_effect") or {}
        if kind == "select_option":
            # The executed public action supplies these; do not parse hidden goals.
            for key in ("option_name", "option_value"):
                if key in effect:
                    entry[key] = effect[key]
            if "label" in effect:
                entry["label"] = effect["label"]
            if "option_name" not in entry:
                parts = target.split(":", 2)
                if len(parts) == 3:
                    entry.update(option_name=parts[1], option_value=parts[2])
        return entry

    def record_failure(self, *, action: str, arguments: Any, error: dict) -> None:
        # Rejected calls can carry JSON null/list/string arguments. Recording
        # their rejection must not turn a recoverable tool error into a crash.
        invalid_arguments = not isinstance(arguments, dict)
        raw_arguments = copy.deepcopy(arguments) if invalid_arguments else None
        arguments = arguments if isinstance(arguments, dict) else {}
        if action == "webshop_search" and arguments.get("query"):
            query = self.data["queries"][self._query(str(arguments["query"]))]
            query["failed_search_count"] = query.get("failed_search_count", 0) + 1
        self.data["events"].append({"sequence": len(self.data["events"]) + 1,
            **self._action(action, arguments, {}), "status": "error",
            "error_code": str(error.get("code", "unknown")),
            **({"invalid_arguments": raw_arguments} if invalid_arguments else {}),
            **({"query_id": self._query(str(arguments["query"]))}
               if action == "webshop_search" and arguments.get("query") else {})})

    def _put(self, value: Any) -> str:
        key = digest(value)
        path = self.directory / (key + ".json")
        if not path.exists():
            temporary = path.with_suffix(".writing")
            temporary.write_text(encoded(value), encoding="utf-8")
            temporary.replace(path)
        return key

    def _get(self, key: str) -> Any:
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("invalid WebShop content reference")
        value = json.loads((self.directory / (key + ".json")).read_text(encoding="utf-8"))
        if digest(value) != key:
            raise ValueError("WebShop memory content hash mismatch")
        return value

    def persist(self) -> dict:
        return {"schema": POLICY, "binding": self.binding, "store": self.directory.name,
                "index": self._put(self.data), "storage": "SPGFS_WEBSHOP_MEMORY_DIR",
                "products": len(self.data["products"]),
                "observations": len(self.data["observations"]),
                "format_version": self.data["format_version"],
                "projection_revision": PROJECTION_REVISION}

    def _product(self, asin: str) -> dict:
        return self.data["products"].setdefault(asin, {"asin": asin, "titles": [],
            "opened": False, "visit_count": 0, "sections": {}, "options": [], "sources": []})

    def _title(self, product: dict, title: Any, source: str, partial: bool = False) -> None:
        title = str(title or "").strip()
        if title:
            item = {"ref": self._put(title), "source": source, "partial": partial}
            if item not in product["titles"]:
                product["titles"].append(item)

    def _migrate(self, journal: dict) -> None:
        self.data["coverage"] = "legacy_partial"
        for record in journal.get("candidate_ledger", []):
            if isinstance(record, dict) and record.get("asin"):
                p = self._product(str(record["asin"]).casefold())
                self._title(p, record.get("preview_title"), "legacy_partial", True)
        for record in journal.get("product_inspections", []):
            if not isinstance(record, dict) or not record.get("asin"):
                continue
            p = self._product(str(record["asin"]).casefold())
            p["opened"] = True
            self._title(p, record.get("product_title"), "legacy_partial", True)
            for name in record.get("sections_viewed", []):
                p["sections"].setdefault(str(name).casefold(), [])
            for name, body in record.get("section_evidence", {}).items():
                p["sections"].setdefault(name.casefold(), []).append({
                    "cleaned": self._put(str(body)), "raw": self._put(str(body)),
                    "source": "legacy_partial", "partial": True, "raw_range": None})
            options = record.get("observed_option_values", {})
            if options:
                p["options"].append({"ref": self._put(options), "source": "legacy_partial", "partial": True})

    def observe(self, state: dict, *, action: str = "", arguments: dict | None = None) -> None:
        arguments = arguments or {}
        if state.get("error") or state.get("status") in {"error", "failed"}:
            return
        # Public environment fields only, before any v2 annotations are attached.
        public = {k: copy.deepcopy(state[k]) for k in (
            "page_type", "page_text", "product", "valid_subactions", "selected_options",
            "unselected_option_groups", "state_version", "steps", "action_effect",
            "observation_truncated", "purchase_visible", "purchased", "done") if k in state}
        if isinstance(public.get("valid_subactions"), list):
            # Strip runtime-only annotations before hashing so a restored current
            # page is exactly the same public observation, not a fresh visit.
            public["valid_subactions"] = [{k: v for k, v in a.items() if k in {
                "kind", "target_id", "label", "title", "asin", "price", "price_min", "price_max",
                "option_name", "option_value", "section", "raw_action"}}
                for a in public["valid_subactions"] if isinstance(a, dict)]
        if state.get("page_type") == "product_section":
            public["observed_section"] = section_name(state, arguments)
        source = self._put(public)
        if source in self.data["observations"] and not action:
            return  # Restoration/read has no completed tool event.
        if source not in self.data["observations"]:
            self.data["observations"].append(source)
        page = state.get("page_type")
        targets = state.get("valid_subactions", [])
        known_products = set(self.data["products"])
        event = None
        if action:
            event = {"sequence": len(self.data["events"]) + 1,
                     **self._action(action, arguments, state), "status": "ok",
                     "step": state.get("steps"), "state_version": state.get("state_version"),
                     "source": source}
            if action == "webshop_search" and str(arguments.get("query", "")).strip():
                index = self._query(str(arguments["query"]))
                self.data["current_query"] = index
                query = self.data["queries"][index]
                if query["search_count"] is None:
                    query["new_search_count"] += 1
                else:
                    query["search_count"] += 1
                query["last_search_step"] = state.get("steps")
                query["last_search_sequence"] = event["sequence"]
            event["query_id"] = self.data["current_query"]
            self.data["events"].append(event)
        if page == "search_results":
            candidates = []
            for position, item in enumerate(targets):
                if not isinstance(item, dict) or item.get("kind") != "open_product":
                    continue
                asin = visible_product_asin(item)
                if not asin:
                    continue
                p = self._product(asin)
                self._title(p, str(item.get("title") or "").strip() or item.get("label"), source)
                if source not in p["sources"]:
                    p["sources"].append(source)
                preview = {key: copy.deepcopy(item[key]) for key in
                           ("price", "price_min", "price_max") if key in item}
                if preview:
                    p.setdefault("previews", []).append({"values": preview, "source": source})
                candidates.append({"asin": asin, "position": position})
            index = self.data["current_query"]
            query = self.data["queries"][index] if index is not None else None
            self.data["batches"].append({"query": query["query"] if query else "",
                                         "query_id": index, "source": source,
                                         "candidates": candidates,
                                         "event_sequence": event["sequence"] if event else None})
            if query:
                query["recorded_batches"] += 1
            if event:
                event["new_candidate_count"] = len({c["asin"] for c in candidates} - known_products)
                if query and action == "webshop_search":
                    query["last_new_candidate_count"] = event["new_candidate_count"]
                    prior = next((b for b in reversed(self.data["batches"][:-1])
                                  if b.get("query_id") == index and b.get("event_sequence")
                                  and self.data["events"][b["event_sequence"]-1]["kind"] == "search"), None)
                    previous = [c["asin"] for c in prior["candidates"]] if prior else None
                    current = [c["asin"] for c in candidates]
                    event["result_set_changed"] = previous is None or set(previous) != set(current)
                    event["result_order_changed"] = previous is None or previous != current
                    query["last_result_set_changed"] = event["result_set_changed"]
        if page not in {"product", "product_section"}:
            return
        product = state.get("product") or {}
        asin = str(product.get("asin", "")).strip().casefold()
        if not asin:
            return
        p = self._product(asin)
        p["opened"] = True
        p["sources"].append(source)
        p["last_product_observation"] = {"source": source, "state_version": state.get("state_version"),
            "product": copy.deepcopy(product)}
        p.setdefault("product_observations", []).append(copy.deepcopy(p["last_product_observation"]))
        if str(arguments.get("target_id", "")).startswith("open_product:"):
            p["visit_count"] += 1
        self._title(p, product.get("title"), source)
        values: dict[str, list[str]] = {}
        for item in targets:
            name = item.get("option_name") if isinstance(item, dict) else None
            value = item.get("option_value") if isinstance(item, dict) else None
            if name and value is not None:
                values.setdefault(str(name), [])
                if str(value) not in values[str(name)]:
                    values[str(name)].append(str(value))
        if values:
            entry = {"ref": self._put(values), "source": source,
                     "state_version": state.get("state_version"),
                     "selected_options_at_observation": copy.deepcopy(state.get("selected_options", {})),
                     "partial": bool(state.get("observation_truncated", False))}
            p["options"].append(entry)
        section = section_name(state, arguments)
        if page == "product_section" and section:
            raw = str(state.get("page_text", ""))
            cleaned = clean_section(raw, self.task)
            entry = {k: v for k, v in cleaned.items() if k != "text"}
            entry.update(raw=self._put(raw), cleaned=self._put(cleaned["text"]), source=source,
                         partial=bool(state.get("observation_truncated", False)))
            entries = p["sections"].setdefault(section, [])
            if entry not in entries:
                entries.append(entry)

    def title(self, asin: str) -> str:
        titles = self.data["products"].get(asin, {}).get("titles", [])
        return self._get(titles[-1]["ref"]) if titles else ""

    def fact(self, asin: str, section: str | None = None) -> dict:
        p = self.data["products"].get(asin.casefold(), {})
        observed = bool(p.get("opened")) if section is None else section.casefold() in p.get("sections", {})
        entry = None
        if section and p.get("sections", {}).get(section.casefold()):
            entry = p["sections"][section.casefold()][-1]
        return {"observation_status": "observed" if observed else (
                    "unobserved" if self.data["coverage"] == "complete" else "unknown"),
                "evidence_in_store": ("partial" if entry and entry.get("partial") else
                    "full" if entry or (observed and section is None and p.get("last_product_observation")) else
                    "partial" if observed and section is None and self.data["coverage"] == "legacy_partial" else "missing"),
                "evidence_in_prompt": "omitted"}

    def annotate(self, state: dict) -> None:
        asin = str((state.get("product") or {}).get("asin", "")).casefold()
        for item in state.get("valid_subactions", []):
            if not isinstance(item, dict):
                continue
            for key in ("inspection_status", "candidate_evidence", "observed_option_values",
                        "observed_option_groups", "evidence_status", "action_semantics", "visit_count"):
                item.pop(key, None)
            kind = item.get("kind")
            if kind == "open_product":
                item["memory_fact"] = self.fact(visible_product_asin(item))
            elif kind == "view_section" or str(item.get("target_id", "")).startswith("view_"):
                name = section_name({}, item) or str(item.get("section", "")).casefold()
                item["memory_fact"] = self.fact(asin, name)
        state.pop("candidate_coverage", None)

    def read(self, arguments: dict, *, max_chars: int = 1800) -> dict:
        """Audit-only paging helper, never a Worker tool or model Action."""
        kind = arguments.get("kind", "candidates")
        asin = str(arguments.get("asin", "")).strip().casefold()
        cursor = arguments.get("cursor", 0)
        if type(cursor) is not int or cursor < 0:
            return {"error": "invalid_cursor"}
        result: dict[str, Any] = {"kind": kind, "asin": asin}
        if "section" in arguments:
            result["section"] = str(arguments["section"])
        if kind == "candidates":
            content = [{"asin": k, "title": self.title(k), **self.fact(k)} for k in self.data["products"]]
            # Text serialization permits paging even one unusually long title.
            content = encoded(content)
            result["format"] = "json_text"
        else:
            p = self.data["products"].get(asin)
            if not p:
                return {"error": "not_observed_in_this_session", "asin": asin}
            version = arguments.get("version", -1)
            if type(version) is not int:
                return {"error": "invalid_version"}
            entries = p["titles"] if kind == "title" else p["options"] if kind == "options" else (
                p["sections"].get(str(arguments.get("section", "")).casefold(), []) if kind in {"section", "raw_section"} else [])
            if not entries:
                return {"error": "not_observed_in_this_session", "asin": asin, "kind": kind}
            if not -len(entries) <= version < len(entries):
                return {"error": "invalid_version", "versions": len(entries)}
            entry = entries[version]
            key = "cleaned" if kind == "section" else "raw" if kind == "raw_section" else "ref"
            content = self._get(entry[key])
            if kind == "options":
                content = {"values": content, "state_version": entry.get("state_version"),
                           "selected_options_at_observation": entry.get("selected_options_at_observation", {}),
                           "scope": "historical_snapshot_only"}
            if not isinstance(content, str):
                content = encoded(content)
                result["format"] = "json_text"
            result.update(source=entry["source"], content_ref=entry[key],
                          evidence_in_store="partial" if entry.get("partial") else "full",
                          versions=len(entries), version=version % len(entries))
        if cursor > len(content):
            return {"error": "invalid_cursor", "full_chars": len(content)}
        result.update(full_chars=len(content), range=[cursor, cursor], text="", complete=False,
                      next_cursor=cursor, same_content=False)
        # Binary search includes actual escaping, metadata, and continuation cursor.
        low, high = 0, len(content) - cursor
        while low < high:
            size = (low + high + 1) // 2
            end = cursor + size
            trial = dict(result, text=content[cursor:end], range=[cursor, end],
                         complete=end == len(content), next_cursor=None if end == len(content) else end)
            if len(encoded(trial)) <= max_chars:
                low = size
            else:
                high = size - 1
        end = cursor + low
        result.update(text=content[cursor:end], range=[cursor, end], complete=end == len(content),
                      next_cursor=None if end == len(content) else end)
        if len(encoded(result)) > max_chars or (low == 0 and cursor < len(content)):
            return {"error": "read_budget_too_small", "retry_with_same_cursor": cursor}
        return copy.deepcopy(result)
