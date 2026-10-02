"""Freeze unambiguous QA fields during schema repair, never reconsider answers."""

from __future__ import annotations

import json
import math
from typing import Any

from .qa_result_contract import QA_RESULT_FIELDS


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON number")
    return parsed


def complete_payload(text: str) -> dict[str, Any] | None:
    """Accept one complete object; do not choose among conflicting or partial JSON."""
    raw = text.strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[8:-3].strip()
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def preserved_fields(text: str) -> dict[str, Any]:
    payload = complete_payload(text)
    if payload is None:
        return {}
    result = {}
    for key in QA_RESULT_FIELDS:
        value = payload.get(key)
        if (key == "answer" and isinstance(value, str) and value.strip()
                or key in {"evidence", "unresolved_issues", "tool_summary"} and isinstance(value, list)
                or key == "summary" and isinstance(value, str)
                or key == "confidence" and isinstance(value, (int, float))
                and not isinstance(value, bool) and 0 <= value <= 1 and math.isfinite(value)):
            result[key] = value
    return result


def restore_preserved_fields(
    text: str, frozen: dict[str, Any], *, stage: str = "qa_schema_preservation",
) -> tuple[str, dict[str, Any]]:
    payload = complete_payload(text)
    if payload is None or not frozen:
        return text, {}
    changed = [key for key, value in frozen.items()
               if key not in payload or type(payload[key]) is not type(value) or payload[key] != value]
    if not changed:
        return text, {}
    proposed_answer = payload.get("answer")
    payload.update(frozen)
    return json.dumps(payload, ensure_ascii=False, allow_nan=False), {
        "stage": stage, "accepted": True, "restored_fields": changed,
        "original_answer": frozen.get("answer"), "repair_proposed_answer": proposed_answer,
        "raw_repair_response": text,
    }


SCHEMA_REPAIR_INSTRUCTION = (
    " This request repairs serialization and field types ONLY. Do not solve or verify "
    "the question again, reconsider answerability, or replace the answer with a refusal. "
    "Copy preserved_artifact_fields exactly; only repair missing or ill-typed fields. "
    "Do not invent facts or quotations. There is no later automatic answer extraction "
    "or semantic rewriting."
)
