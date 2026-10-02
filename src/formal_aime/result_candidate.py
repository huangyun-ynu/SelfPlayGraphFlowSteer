"""Conservative final-data normalization; never repairs executable tool arguments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from .artifact_protocol import check_artifact, _unique_object

NARRATIVE_FIELDS = {"summary", "evidence", "unresolved_issues", "tool_summary"}
VERSION = "result_candidate_v1"


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class DataScanner:
    """A bounded JSON lexer/parser with one narrow extension in narrative strings.

    Illegal escape letters represent literal backslashes. Structure, keys, and
    answer strings remain strict. No regex extraction, quote insertion, missing
    brackets, duplicate keys, or guessing where a field ends is allowed.
    """
    def __init__(self, text):
        self.text, self.i, self.spans, self.repairs = text, 0, {}, []

    def space(self):
        while self.i < len(self.text) and self.text[self.i] in " \t\r\n":
            self.i += 1

    def string(self, field=None):
        start = self.i
        assert self.text[self.i] == '"'
        self.i += 1
        pieces = []
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch == '"':
                self.i += 1
                return "".join(pieces)
            if ch == "\\":
                if self.i + 1 >= len(self.text):
                    raise ValueError("unclosed_string")
                nxt = self.text[self.i + 1]
                if nxt in '"\\/bfnrtu':
                    size = 6 if nxt == "u" else 2
                    fragment = self.text[self.i:self.i + size]
                    # Invalid unicode is ambiguous, not an escape-letter repair.
                    pieces.append(json.loads('"' + fragment + '"'))
                    self.i += size
                    continue
                if field not in NARRATIVE_FIELDS:
                    raise ValueError("invalid_escape_outside_narrative")
                self.repairs.append({"field": field, "position": self.i,
                                     "operation": "literal_invalid_escape"})
                pieces.append("\\" + nxt)
                self.i += 2
                continue
            if ord(ch) < 32:
                # An unescaped line break has an unambiguous literal value.
                if field not in NARRATIVE_FIELDS or ch not in "\n\r\t":
                    raise ValueError("invalid_control_character")
                self.repairs.append({"field": field, "position": self.i,
                                     "operation": "escape_literal_whitespace"})
            pieces.append(ch)
            self.i += 1
        raise ValueError(f"unclosed_string_at_{start}")

    def value(self, field=None, depth=0):
        if depth > 24:
            raise ValueError("excessive_json_depth")
        self.space()
        if self.i >= len(self.text):
            raise ValueError("incomplete_value")
        ch = self.text[self.i]
        if ch == '"':
            return self.string(field)
        if ch in "[{":
            is_object = ch == "{"
            end = "}" if is_object else "]"
            value = {} if is_object else []
            self.i += 1
            self.space()
            if self.i < len(self.text) and self.text[self.i] == end:
                self.i += 1
                return value
            while True:
                self.space()
                key = None
                if is_object:
                    if self.i >= len(self.text) or self.text[self.i] != '"':
                        raise ValueError("invalid_object_key")
                    key = self.string()
                    if key in value:
                        raise ValueError("duplicate_json_key:" + key)
                    self.space()
                    if self.i >= len(self.text) or self.text[self.i] != ":":
                        raise ValueError("missing_colon")
                    self.i += 1
                    self.space()
                start = self.i
                item = self.value(key if depth == 0 else field, depth + 1)
                if is_object:
                    value[key] = item
                    if depth == 0:
                        self.spans[key] = [start, self.i]
                else:
                    value.append(item)
                self.space()
                if self.i >= len(self.text):
                    raise ValueError("incomplete_container")
                sep = self.text[self.i]
                self.i += 1
                if sep == end:
                    return value
                if sep != ",":
                    raise ValueError("ambiguous_field_boundary")
        value, end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(self.text, self.i)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite_json_number")
        self.i = end
        return value

    def parse(self):
        value = self.value()
        self.space()
        if self.i != len(self.text) or not isinstance(value, dict):
            raise ValueError("expected_single_complete_object")
        return value


def answer_string(value):
    if isinstance(value, str) and value.strip():
        return value
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return None


@dataclass(frozen=True)
class Candidate:
    raw: str
    source: str
    payload_json: str
    answer: str
    spans_json: str
    repairs_json: str

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def candidate_id(self):
        return digest({"raw": self.raw, "source": self.source})

    def record(self):
        payload = self.payload
        spans = json.loads(self.spans_json)
        issues = payload.get("unresolved_issues")
        return {"version": VERSION, "candidate_id": self.candidate_id,
                "origin_attempt_id": self.source, "origin_response_sha256": digest(self.raw),
                "raw_response": self.raw, "answer": self.answer,
                "answer_sha256": digest(self.answer), "answer_raw_span": spans.get("answer"),
                "raw_field_spans": spans, "source_encoding": "utf-8",
                "payload": payload, "confidence_snapshot": payload.get("confidence"),
                "issue_ids": [digest(issue) for issue in issues] if isinstance(issues, list) else None,
                "model_claims": payload.get("evidence"),
                "repair_events": json.loads(self.repairs_json),
                "parse_status": "format_valid" if check_artifact(self.payload_json)[0] else "pending_repair"}


def capture_candidate(text, *, source=""):
    if not isinstance(text, str) or len(text) > 262144:
        return None
    cleaned = re.sub(r"^\s*<think>.*?</think>", "", text, count=1, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    try:
        scanner = DataScanner(cleaned)
        payload = scanner.parse()
        offset = text.find(cleaned)
        if offset < 0:
            return None
        scanner.spans = {key: [a + offset, b + offset] for key, (a, b) in scanner.spans.items()}
        for repair in scanner.repairs:
            repair["position"] += offset
        if any(key in payload for key in ("action_call", "action_calls", "tool_call", "tool_calls")):
            return None
        answer = answer_string(payload.get("answer"))
        if answer is None or answer in {"WORKER_PROTOCOL_FAILURE", "WORKER_BACKEND_FAILURE"}:
            return None
        # A scalar narrative entry has an unambiguous array representation.
        # Preserve the entire string as ONE item: never split sentences or
        # reinterpret "None"/"unknown" as an empty, resolved issue list.
        for field in ("evidence", "unresolved_issues", "tool_summary"):
            if isinstance(payload.get(field), str):
                payload[field] = [payload[field]]
                scanner.repairs.append({"field": field,
                    "position": scanner.spans[field][0],
                    "operation": "wrap_single_array_entry"})
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        # Apply only the established safe representation conversions.
        checked, _ = check_artifact(serialized)
        if checked is not None:
            serialized = json.dumps(checked, ensure_ascii=False, allow_nan=False)
        return Candidate(text, source, serialized, answer,
                         json.dumps(scanner.spans), json.dumps(scanner.repairs))
    except (ValueError, TypeError, RecursionError, AssertionError):
        return None


def repair_violations(original: Candidate, repaired: Candidate | None):
    if repaired is None:
        return ["repair_invalid_candidate"]
    a, b = original.payload, repaired.payload
    errors = []
    if original.answer != repaired.answer:
        errors.append("format_repair_changed_answer")
    for field, code in (("summary", "repair_changed_summary"), ("evidence", "repair_changed_evidence"),
                        ("unresolved_issues", "repair_dropped_issue"), ("tool_summary", "repair_changed_tool_claims")):
        old, new = a.get(field), b.get(field)
        if field != "summary" and isinstance(old, str):
            old = [old] if old else []
        if old != new:
            errors.append(code)
    old, new = a.get("confidence"), b.get("confidence")
    try:
        if old is None or isinstance(new, bool) or not math.isfinite(float(new)) or float(new) > float(old):
            errors.append("repair_raised_confidence")
    except (ValueError, TypeError):
        errors.append("repair_unknown_confidence")
    return errors
