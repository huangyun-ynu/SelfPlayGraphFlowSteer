"""Bounded text-action decoding for the student Responses gateway.

Only a leading, complete action is executable. Later requests and premature
answers are discarded; they cannot supply observations for the first action.
Other backends retain their existing strict action protocol.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PROTOCOL = "student_text_action_v2"
MAX_RESPONSE_CHARS = 262_144
MAX_OBJECTS = 64
MAX_EXECUTION_REPAIRS = 2
MAX_QUESTION_REPAIRS = 6
_FINAL_FIELDS = frozenset({
    "answer", "summary", "confidence", "evidence", "unresolved_issues",
    "tool_summary", "swe_completion",
})
_NQ_CORPUS_FINAL_FIELDS = frozenset({"answerability", "evidence_refs"})


@dataclass(frozen=True)
class DecodedStudentResponse:
    kind: str
    text: str = ""
    reason: str | None = None
    object_count: int = 0
    discarded_objects: int = 0


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite_json_number")


def _action_object(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"action_calls"}:
        return False
    calls = value["action_calls"]
    if not isinstance(calls, list) or len(calls) != 1:
        return False
    call = calls[0]
    return (isinstance(call, dict) and set(call) == {"name", "arguments"}
            and isinstance(call["name"], str) and bool(call["name"].strip())
            and isinstance(call["arguments"], dict))


def decode_student_response(text: str, *, nq_corpus: bool = False) -> DecodedStudentResponse:
    """Decode the entire bounded stream, never extract JSON from arbitrary prose."""
    if not isinstance(text, str) or len(text) > MAX_RESPONSE_CHARS:
        return DecodedStudentResponse("repair", reason="response_size_limit")
    remaining = text.strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    objects = []
    try:
        while remaining and len(objects) < MAX_OBJECTS:
            value, end = decoder.raw_decode(remaining)
            objects.append(value)
            remaining = remaining[end:].lstrip()
    except (ValueError, RecursionError):
        return DecodedStudentResponse("repair", reason="invalid_json_stream", object_count=len(objects))
    if remaining or not objects or not all(isinstance(value, dict) for value in objects):
        return DecodedStudentResponse("repair", reason="invalid_object_stream", object_count=len(objects))
    first = objects[0]
    final_fields = _FINAL_FIELDS | _NQ_CORPUS_FINAL_FIELDS if nq_corpus else _FINAL_FIELDS
    if _action_object(first):
        for index, value in enumerate(objects[1:], 1):
            if _action_object(value):
                continue
            # A final suffix is never executed, submitted or replayed. Its
            # content quality is immaterial, but it must be a known final shape.
            if index == len(objects) - 1 and "answer" in value and set(value) <= final_fields:
                continue
            return DecodedStudentResponse("repair", reason="ambiguous_action_suffix", object_count=len(objects))
        return DecodedStudentResponse(
            "action", json.dumps(first, ensure_ascii=False), object_count=len(objects),
            discarded_objects=len(objects) - 1,
        )
    if len(objects) == 1 and "answer" in first and set(first) <= final_fields:
        return DecodedStudentResponse("final", text.strip(), object_count=1)
    return DecodedStudentResponse("repair", reason="expected_one_action_or_final", object_count=len(objects))


def response_text_parts(output_items: list[dict[str, Any]], fallback: str) -> tuple[str, list[dict]]:
    """Retain item boundaries; only assistant output_text is an executable surface."""
    if not output_items:
        return fallback, [{"source": "output_text_fallback", "text": fallback}]
    parts = []
    for item_index, item in enumerate(output_items):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        if item.get("channel") not in {None, "final", "commentary"}:
            continue
        for content_index, content in enumerate(item.get("content") or []):
            if not isinstance(content, dict) or content.get("type") != "output_text":
                continue
            value = content.get("text")
            if isinstance(value, str):
                parts.append({"item_index": item_index, "content_index": content_index,
                              "channel": item.get("channel"), "text": value})
    # Preserve block text exactly, including cases where one JSON spans blocks.
    return "".join(part["text"] for part in parts), parts
