from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .contracts import RelationType, StructuralOperator


class ActionType(StrEnum):
    ADD_AGENT = "add_agent"
    SET_PROMPT = "set_prompt"
    SET_MODEL = "set_model"
    SET_LAYER = "set_layer"
    CONSIDER_RELATION = "consider_relation"
    SET_RELATION = "set_relation"
    REMOVE_RELATION = "remove_relation"
    DELETE_AGENT = "delete_agent"
    SET_OUTPUT = "set_output"
    FINISH = "finish"
    INVALID = "invalid"


class PromptRevisionBasis(StrEnum):
    UPSTREAM_ARTIFACT_CHANGED = "upstream_artifact_changed"
    PEER_ARTIFACT_CHANGED = "peer_artifact_changed"
    TOOL_ERROR = "tool_error"
    UNRESOLVED_ISSUE = "unresolved_issue"
    PROTOCOL_FAILURE = "protocol_failure"
    STRUCTURAL_ROLE_CHANGE = "structural_role_change"
    CONTROLLER_REPAIR = "controller_repair"


@dataclass
class CanvasAction:
    """One atomic graph edit, following FlowSteer's one-action-per-turn contract."""

    action_type: ActionType
    agent_id: str | None = None
    target: str | None = None
    source: str | None = None
    prompt: str | None = None
    role: str | None = None
    objective: str | None = None
    scope: str | None = None
    expected_output: str | None = None
    revision_basis: PromptRevisionBasis | None = None
    evidence_agent_ids: tuple[str, ...] = ()
    structural_operator: StructuralOperator | None = None
    layer: int | None = None
    relation: RelationType | None = None
    reasoning: str | None = None
    runtime_route: str | None = None
    expected_version: int | None = None
    raw_text: str = ""
    parse_error: str | None = None

    @property
    def valid(self) -> bool:
        return self.action_type is not ActionType.INVALID and self.parse_error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_type"] = self.action_type.value
        payload["relation"] = self.relation.value if self.relation else None
        payload["revision_basis"] = self.revision_basis.value if self.revision_basis else None
        payload["evidence_agent_ids"] = list(self.evidence_agent_ids)
        return payload


@dataclass(frozen=True)
class ParsedPolicyAction:
    """Strict boundary between an immutable model response and one Canvas action."""

    action: CanvasAction
    action_text: str | None
    character_span: tuple[int, int] | None
    candidate_count: int


class ActionParser:
    """Parse a single JSON action, with FlowSteer-style XML as a compatibility fallback."""

    def parse(self, text: str) -> CanvasAction:
        if not text or not text.strip():
            return self._invalid(text or "", "empty action")
        payload = self._json_payload(text)
        if payload is not None:
            return self._from_payload(payload, text)
        return self._from_xml(text)

    def parse_policy_output(self, text: str) -> ParsedPolicyAction:
        """Accept exactly one complete, valid JSON action from a policy response.

        Reasoning outside the JSON object is retained in the trajectory but is never
        interpreted as another action. XML remains available through ``parse`` for
        legacy/audit inputs; new policy calls deliberately use the unambiguous JSON
        contract.
        """

        raw = str(text or "")
        candidates = list(_top_level_json_objects(raw))
        if len(candidates) != 1:
            invalid = self._invalid(
                raw, f"expected exactly one JSON action; found {len(candidates)}"
            )
            return ParsedPolicyAction(invalid, None, None, len(candidates))
        start, end, candidate = candidates[0]
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            invalid = self._invalid(raw, "the single JSON action is malformed")
            return ParsedPolicyAction(invalid, None, (start, end), 1)
        if not isinstance(payload, dict):
            invalid = self._invalid(raw, "the single JSON action must be an object")
            return ParsedPolicyAction(invalid, None, (start, end), 1)
        action = self._from_payload(payload, candidate)
        return ParsedPolicyAction(
            action,
            candidate if action.valid else None,
            (start, end),
            1,
        )

    def _from_payload(self, payload: dict[str, Any], raw_text: str) -> CanvasAction:
        action_name = str(payload.get("action", payload.get("action_type", ""))).lower()
        try:
            action_type = ActionType(action_name)
        except ValueError:
            return self._invalid(raw_text, f"unknown action: {action_name or 'missing'}")
        relation = payload.get("relation")
        try:
            relation_type = RelationType(str(relation).lower()) if relation is not None else None
        except ValueError:
            return self._invalid(raw_text, f"unknown relation: {relation}")
        raw_operator = payload.get("structural_operator", payload.get("operator_type"))
        try:
            structural_operator = (
                StructuralOperator(str(raw_operator).lower()) if raw_operator is not None else None
            )
        except ValueError:
            return self._invalid(raw_text, f"unknown structural_operator: {raw_operator}")
        raw_revision_basis = payload.get("revision_basis")
        try:
            revision_basis = (
                PromptRevisionBasis(str(raw_revision_basis).lower())
                if raw_revision_basis is not None
                else None
            )
        except ValueError:
            return self._invalid(
                raw_text,
                f"unknown prompt revision_basis: {raw_revision_basis}",
            )
        raw_evidence_ids = payload.get("evidence_agent_ids", ())
        if raw_evidence_ids is None:
            evidence_agent_ids: tuple[str, ...] = ()
        elif isinstance(raw_evidence_ids, (list, tuple)):
            evidence_agent_ids = tuple(
                dict.fromkeys(
                    value for item in raw_evidence_ids if (value := _clean(item)) is not None
                )
            )
        else:
            return self._invalid(raw_text, "evidence_agent_ids must be an array")
        layer = payload.get("layer")
        try:
            parsed_layer = int(layer) if layer is not None else None
        except (TypeError, ValueError):
            return self._invalid(raw_text, "layer must be an integer")
        expected_version = payload.get("expected_version")
        try:
            parsed_expected_version = (
                int(expected_version) if expected_version is not None else None
            )
        except (TypeError, ValueError):
            return self._invalid(raw_text, "expected_version must be an integer")
        action = CanvasAction(
            action_type=action_type,
            agent_id=_clean(payload.get("agent_id")),
            target=_clean(payload.get("target")),
            source=_clean(payload.get("source")),
            prompt=_clean(payload.get("prompt")),
            role=_clean(payload.get("role")),
            objective=_clean(payload.get("objective")),
            scope=_clean(payload.get("scope")),
            expected_output=_clean(payload.get("expected_output")),
            revision_basis=revision_basis,
            evidence_agent_ids=evidence_agent_ids,
            structural_operator=structural_operator,
            layer=parsed_layer,
            relation=relation_type,
            reasoning=_clean(payload.get("reasoning")),
            runtime_route=_clean(
                payload.get("runtime_route", payload.get("model_route", payload.get("runtime")))
            ),
            expected_version=parsed_expected_version,
            raw_text=raw_text,
        )
        error = self._required_field_error(action)
        if error:
            return self._invalid(raw_text, error, reasoning=action.reasoning)
        return action

    def _from_xml(self, text: str) -> CanvasAction:
        action_name = self._tag(text, "action")
        if not action_name:
            return self._invalid(text, "expected one JSON object or an <action> tag")
        payload: dict[str, Any] = {
            "action": action_name,
            "agent_id": self._tag(text, "agent_id"),
            "target": self._tag(text, "target"),
            "source": self._tag(text, "source"),
            "prompt": self._tag(text, "prompt"),
            "role": self._tag(text, "role"),
            "objective": self._tag(text, "objective"),
            "scope": self._tag(text, "scope"),
            "expected_output": self._tag(text, "expected_output"),
            "revision_basis": self._tag(text, "revision_basis"),
            "evidence_agent_ids": [
                value.strip()
                for value in (self._tag(text, "evidence_agent_ids") or "").split(",")
                if value.strip()
            ],
            "structural_operator": self._tag(text, "structural_operator")
            or self._tag(text, "operator_type"),
            "layer": self._tag(text, "layer"),
            "relation": self._tag(text, "relation"),
            "reasoning": self._tag(text, "reasoning") or self._tag(text, "thought"),
            "runtime_route": self._tag(text, "runtime_route")
            or self._tag(text, "model_route")
            or self._tag(text, "runtime"),
            "expected_version": self._tag(text, "expected_version"),
        }
        return self._from_payload(payload, text)

    @staticmethod
    def _json_payload(text: str) -> dict[str, Any] | None:
        without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = [without_think.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", without_think, re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1))
        first, last = without_think.find("{"), without_think.rfind("}")
        if first >= 0 and last > first:
            candidates.append(without_think[first : last + 1])
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError):
                try:
                    payload = json.loads(_repair_json_string_backslashes(candidate))
                except (TypeError, ValueError):
                    continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _tag(text: str, name: str) -> str | None:
        match = re.search(
            rf"[<\[]\s*{re.escape(name)}\s*[>\]](.*?)[<\[]\s*/{re.escape(name)}\s*[>\]]",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def _required_field_error(action: CanvasAction) -> str | None:
        if action.action_type is ActionType.ADD_AGENT:
            return None
        if (
            action.action_type
            in {
                ActionType.SET_PROMPT,
                ActionType.SET_MODEL,
                ActionType.SET_LAYER,
                ActionType.SET_OUTPUT,
            }
            and not action.target
        ):
            return f"{action.action_type.value} requires target"
        if action.action_type is ActionType.SET_PROMPT:
            structured = (
                action.role,
                action.objective,
                action.scope,
                action.expected_output,
            )
            if not action.prompt and not all(structured):
                return "set_prompt requires role, objective, scope, and expected_output"
        if action.action_type is ActionType.SET_LAYER and action.layer is None:
            return "set_layer requires layer"
        if action.action_type is ActionType.SET_MODEL and not action.runtime_route:
            return "set_model requires runtime_route"
        if action.runtime_route is not None and action.action_type is not ActionType.SET_MODEL:
            return "runtime_route is only valid in set_model"
        if action.action_type is ActionType.CONSIDER_RELATION and (
            not action.source or not action.target
        ):
            return "consider_relation requires source and target"
        if action.action_type in {
            ActionType.SET_RELATION,
            ActionType.REMOVE_RELATION,
        } and (not action.source or not action.target or action.relation is None):
            return f"{action.action_type.value} requires source, target, and relation"
        if action.action_type is ActionType.DELETE_AGENT and not (action.target or action.agent_id):
            return "delete_agent requires target or agent_id"
        return None

    @staticmethod
    def _invalid(raw_text: str, error: str, *, reasoning: str | None = None) -> CanvasAction:
        return CanvasAction(
            action_type=ActionType.INVALID,
            raw_text=raw_text,
            parse_error=error,
            reasoning=reasoning,
        )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _top_level_json_objects(text: str):
    """Yield complete JSON-object candidates and their exact character spans."""

    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            return
        depth = 0
        in_string = False
        escaped = False
        cursor = start
        while cursor < len(text):
            character = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    yield start, end, text[start:end]
                    index = end
                    break
            cursor += 1
        else:
            return


def _repair_json_string_backslashes(value: str) -> str:
    """Escape model-emitted LaTeX slashes only after strict JSON parsing fails.

    A retry is intentionally narrow: outside JSON strings nothing changes, and
    quote, slash, backslash and valid Unicode escapes retain JSON semantics.
    Other string escapes become literal backslashes. This recovers ``\angle``
    and also prevents ``\frac`` from silently becoming a form-feed escape in
    a candidate that already required repair.
    """

    repaired: list[str] = []
    in_string = False
    index = 0
    while index < len(value):
        character = value[index]
        if character == '"':
            backslashes = 0
            cursor = len(repaired) - 1
            while cursor >= 0 and repaired[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
            repaired.append(character)
            index += 1
            continue
        if character != "\\" or not in_string or index + 1 >= len(value):
            repaired.append(character)
            index += 1
            continue
        following = value[index + 1]
        unicode_escape = (
            following == "u"
            and index + 5 < len(value)
            and all(char in "0123456789abcdefABCDEF" for char in value[index + 2 : index + 6])
        )
        if following in {'"', "\\", "/"} or unicode_escape:
            repaired.append(character)
        else:
            repaired.extend(("\\", "\\"))
        index += 1
    return "".join(repaired)
