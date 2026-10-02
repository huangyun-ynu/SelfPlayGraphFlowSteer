"""One authoritative Worker artifact syntax check, with actionable diagnostics."""

import json
import math
import re

WORKER_PROTOCOL_STATUS_VERSION = "worker_protocol_status_v1"


def summarize_worker_protocol(*, answer: str, raw_response: str, diagnostics: list) -> dict:
    """Bounded public facts from runtime-owned evidence, never model-authored advice.

    Artifact acceptance is distinct from task correctness and final submission.
    Incomplete legacy records stay unknown; raw exceptions are never copied out.
    """
    records = [item for item in diagnostics if isinstance(item, dict)]
    last = records[-1] if records else {}
    attempts = [
        item for item in records if item.get("stage") in ("finalization_1", "finalization_2")
    ]
    status = "unknown"
    if answer == "WORKER_PROTOCOL_FAILURE":
        status = "failed"
    elif answer != "WORKER_BACKEND_FAILURE":
        if last.get("accepted") is False:
            status = "failed"
        elif last.get("accepted") is True:
            status = (
                "recovered"
                if any(item.get("accepted") is False for item in records[:-1])
                else "valid"
            )
        elif not records and raw_response and check_artifact(raw_response)[0] is not None:
            status = "valid"
    stage = None
    error = None
    if status != "unknown":
        stage = (
            "finalization"
            if last.get("stage") in ("finalization_1", "finalization_2")
            else "worker_output"
        )
    if status == "failed":
        reason = last.get("rejection_reason") if last.get("accepted") is False else None
        reason = reason if isinstance(reason, str) else None
        error = {
            "truncated_final_response": "output_truncated",
            "repetitive_final_response": "output_repetition",
            "missing_final_json": "invalid_json",
            "empty_answer": "missing_answer",
            "invalid_field_type": "invalid_field_type",
            "generic_acknowledgement": "generic_acknowledgement",
            "protocol_recovery_exhausted": "protocol_recovery_exhausted",
            "duplicate_failed_request": "duplicate_failed_request",
            "format_repair_changed_answer": "format_repair_changed_answer",
            "repair_dropped_issue": "repair_dropped_issue",
            "repair_raised_confidence": "repair_raised_confidence",
        }.get(reason, "protocol_failure_unknown")
        detail = last.get("parse_error")
        if (
            reason == "missing_final_json"
            and isinstance(detail, dict)
            and detail.get("error_type") == "missing_answer"
        ):
            error = "missing_answer"
        if last.get("accepted") is False and last.get("finish_reason") in ("length", "MAX_TOKENS"):
            error = "output_truncated"
    exhausted = False if status in {"valid", "recovered"} else None
    if status == "failed" and (
        last.get("local_recovery_exhausted") is True or last.get("stage") == "finalization_2"
    ):
        exhausted = True
    return {
        "status": status,
        "stage": stage,
        "error_code": error,
        "finalization_attempts_used": len(attempts) if records or status == "valid" else None,
        "local_recovery_exhausted": exhausted,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key:" + key)
        result[key] = value
    return result


def artifact_normalizations(text: str) -> list[dict]:
    """Describe only safe representation changes; never choose an answer."""
    try:
        raw = _decode_artifact(text)
    except (ValueError, TypeError, RecursionError):
        return []
    if not isinstance(raw, dict):
        return []
    records = []
    for field in ("evidence", "tool_summary"):
        value = raw.get(field)
        if isinstance(value, str):
            records.append({"field": field, "operation": "string_to_array", "original": value,
                            "normalized": [value] if value else []})
    value = raw.get("confidence")
    if isinstance(value, str):
        try:
            converted = float(value)
            if math.isfinite(converted) and 0 <= converted <= 1:
                records.append({"field": "confidence", "operation": "string_to_number",
                                "original": value, "normalized": converted})
        except ValueError:
            pass
    return records


def _decode_artifact(text: str):
    cleaned = re.sub(r"^\s*<think>.*?</think>", "", text, count=1,
                     flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    return json.loads(cleaned, object_pairs_hook=_unique_object)


def protected_format_answer(text: str) -> str | None:
    try:
        payload = _decode_artifact(text)
    except (ValueError, TypeError, RecursionError):
        return None
    answer = payload.get("answer") if isinstance(payload, dict) else None
    if isinstance(answer, str) and answer.strip():
        return answer
    if isinstance(answer, int) and not isinstance(answer, bool):
        return str(answer)
    if isinstance(answer, float) and math.isfinite(answer) and answer.is_integer():
        return str(int(answer))
    return None


def check_artifact(text: str) -> tuple[dict | None, dict]:
    candidate = text
    try:
        payload = _decode_artifact(text)
    except (ValueError, TypeError, RecursionError) as exc:
        issue = {"reason": "missing_final_json", "error_type": type(exc).__name__}
        if isinstance(exc, json.JSONDecodeError):
            issue.update(
                message=exc.msg,
                line=exc.lineno,
                column=exc.colno,
                position=exc.pos,
                excerpt=candidate[max(0, exc.pos - 120) : exc.pos + 120],
            )
        return None, issue
    if not isinstance(payload, dict):
        return None, {"reason": "missing_final_json"}
    if "answer" not in payload:
        return None, {"reason": "missing_final_json", "error_type": "missing_answer"}
    if any(key in payload for key in ("action_call", "action_calls", "tool_call", "tool_calls")):
        return None, {"reason": "missing_final_json", "error_type": "tool_call_after_action_phase"}
    answer = payload["answer"]
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return None, {"reason": "empty_answer"}
    if isinstance(answer, bool):
        return None, {
            "reason": "invalid_field_type",
            "field": "answer",
            "expected": "task artifact, not boolean",
        }
    payload = dict(payload)
    for change in artifact_normalizations(text):
        payload[change["field"]] = change["normalized"]
    normalized = re.sub(r"[.!。！]+$", "", str(answer).strip().casefold())
    if normalized in {
        "acknowledged",
        "got it",
        "noted",
        "ok",
        "okay",
        "sure",
        "thank you",
        "thanks",
        "understood",
    }:
        return None, {"reason": "generic_acknowledgement"}
    for field in ("evidence", "unresolved_issues", "tool_summary"):
        if field in payload and not isinstance(payload[field], list):
            return None, {"reason": "invalid_field_type", "field": field, "expected": "array"}
    if "summary" in payload and not isinstance(payload["summary"], str):
        return None, {"reason": "invalid_field_type", "field": "summary", "expected": "string"}
    if "confidence" in payload:
        value = payload["confidence"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return None, {
                "reason": "invalid_field_type",
                "field": "confidence",
                "expected": "finite number in [0,1]",
            }
    return payload, {}
