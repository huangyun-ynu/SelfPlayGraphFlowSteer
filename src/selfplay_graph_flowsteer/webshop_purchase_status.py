"""Reconcile reported purchase completion with trusted episode state."""
from __future__ import annotations

import copy
import re

from .artifact_protocol import check_artifact

_STAGED = re.compile(
    r"\bstaged purchase proposal\b|"
    r"\b(?:purchase|proposal|buy(?: now)?)\s+(?:(?:is|was|has been|already|successfully|only|now)\s+){0,4}staged\b|"
    r"\b(?:i|we)\s+(?:(?:have|am|are|already|now)\s+){0,3}stag(?:e|ed|ing)\b|"
    r"\b(?:already|previously) staged\b|(?:购买|订单|提案)(?:已|已经)(?:被)?(?:暂存|准备好)", re.I)
_PURCHASED = re.compile(
    r"\b(?:purchase|order)\s+(?:(?:is|was|has been|already|successfully|now)\s+){0,4}(?:committed|completed|placed)\b|"
    r"\b(?:i|we)\s+(?:(?:have|already|successfully)\s+){0,3}(?:purchased|bought|ordered)\b|"
    r"(?:已经|已)(?:完成购买|成功购买|下单成功)|(?:购买|订单)(?:已|已经)(?:提交|完成)", re.I)
_NEGATIVE_STAGE = re.compile(r"\b(?:not|never)\s+(?:yet\s+|been\s+)?staged\b|\bno\s+(?:valid\s+)?(?:purchase|proposal)\b|\bno\s+staged\b|(?:未|没有)(?:暂存|准备)", re.I)
_NEGATIVE_BUY = re.compile(r"\b(?:not|never)\s+(?:yet\s+|been\s+)?(?:committed|completed|purchased|bought|placed)\b|\bno\s+(?:actual\s+)?(?:purchase|order)\b|(?:未|没有)(?:购买|提交|下单)", re.I)
_CONDITIONAL = re.compile(r"^\s*(?:if|once|after|when|before|until)\b|\b(?:will|would|should|could|must|can)\s+(?:be\s+|have\s+been\s+)?(?:staged|committed|completed|placed)\b|(?:如果|待购买后)", re.I)
_FIELDS = ("answer", "summary", "unresolved_issues", "tool_summary", "evidence")


def execution_status(state):
    if not isinstance(state, dict):
        return {"source": "trusted_live_environment", "phase": "unknown",
                "report_text_cannot_change_state": True}
    phase = ("purchased" if state.get("purchased") else "staged"
             if state.get("commit_pending") or state.get("purchase_review_pending") else "not_staged")
    return {"source": "trusted_live_environment", "phase": phase,
            "purchased": bool(state.get("purchased")),
            "commit_pending": bool(state.get("commit_pending")),
            "commit_ready": bool(state.get("commit_ready")),
            "state_version": state.get("state_version"),
            "report_text_cannot_change_state": True}


def statement(state):
    phase = execution_status(state)["phase"]
    if phase == "unknown":
        return "Runtime purchase state: no live episode status is available to this Agent."
    if phase == "purchased":
        return "Runtime purchase state: the environment confirms a completed purchase."
    if phase == "staged":
        return "Runtime purchase state: a real purchase proposal is staged; the purchase is not committed."
    return ("Runtime purchase state: NO purchase is staged or committed. "
            "A report, recommendation or reserved budget does not stage a purchase. "
            "Only a successful native Buy tool call can create a staged proposal.")


def _conflicting_clause(clause, state):
    phase = execution_status(state)["phase"]
    if phase in {"purchased", "unknown"} or _CONDITIONAL.search(clause):
        return False
    if _PURCHASED.search(clause) and not _NEGATIVE_BUY.search(clause):
        return True
    return phase == "not_staged" and bool(_STAGED.search(clause)) and not _NEGATIVE_STAGE.search(clause)


def _clauses(text):
    # Keep decimals and all non-conflicting product evidence intact.
    return re.split(r"(?<=[.!?。！？;；])\s+|\n", text)


def conflicting_fields(report, state):
    if not isinstance(report, dict):
        return []
    result = []
    for field in _FIELDS:
        value = report.get(field, "")
        values = value if isinstance(value, (list, tuple)) else [value]
        if any(_conflicting_clause(clause, state) for text in values if isinstance(text, str)
               for clause in _clauses(text)):
            result.append(field)
    return result


def response_conflict(text, state):
    report, _ = check_artifact(text)
    return conflicting_fields(report, state)


def grounded_packet(packet, state):
    """Remove stale execution assertions from a projection, retaining raw audit."""
    result = copy.deepcopy(packet)
    fields = conflicting_fields(result, state)
    for field in fields:
        def clean(text):
            if not isinstance(text, str):
                return text
            return " ".join(clause for clause in _clauses(text) if not _conflicting_clause(clause, state))
        value = result[field]
        result[field] = [clean(item) for item in value if clean(item)] if isinstance(value, (list, tuple)) else clean(value)
    if fields:
        result["summary"] = statement(state) + " " + result.get("summary", "")
        result["purchase_status_correction"] = {
            "source": "runtime", "conflicting_fields": fields,
            "reason": "Historical model completion claims do not match the current episode; raw report retained in audit.",
            "current": execution_status(state)}
    return result


def recovery_instruction(state, *, claim_conflict=True):
    correction = (" Your previous completion claim was rejected. No action from that report ran. "
                  if claim_conflict else " Your report did not execute an environment action. ")
    return (statement(state) + correction +
            "Use the current public state and remaining budget to choose a native tool call, "
            "or explicitly report that you are stopping without a purchase and explain why. "
            "Do not claim a purchase is staged/committed without the corresponding environment receipt. "
            "The runtime does not choose a product or execute a report's proposed action.")
