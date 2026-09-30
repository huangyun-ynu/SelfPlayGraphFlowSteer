"""Public-state purchase action reservation. No scoring data or implicit purchases."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from threading import RLock

from .webshop_identity import visible_product_asin
from .webshop_navigation import annotate_navigation, navigation_effect

POLICY = "completion_reserve_v1"
FLEXIBLE_POLICY = "completion_reserve_v2"
POLICIES = {POLICY, FLEXIBLE_POLICY}
PLAN_SCHEMA = {
    "type": "object",
    "description": "Reserve a publicly inspected purchase path, or abandon with a reason. Metadata costs no environment action.",
    "properties": {
        "decision": {"type": "string", "enum": ["reserve", "abandon"]},
        "asin": {"type": "string"},
        "options": {"type": "object", "additionalProperties": {"type": "string"}},
        "requirement_quotes": {"type": "array", "items": {"type": "string"}},
        "verified_requirements": {"type": "array", "items": {"type": "string"}},
        "unresolved_constraints": {"type": "array", "items": {"type": "string"}},
        "accept_partial": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["decision"], "additionalProperties": False,
}
GUIDANCE = """
Purchase action reservation (completion_reserve_v1): the ENTIRE graph shares 16 actions.
As soon as you have a purchase candidate, attach completion_plan to an existing action's
arguments: {"decision":"reserve","asin":"PUBLIC_ASIN","options":{"group":"desired public value"},
"requirement_quotes":["exact phrase from the original task"],"verified_requirements":["observed support"],
"unresolved_constraints":[],"accept_partial":false}. Include every requested option that this
candidate offers (including already selected values); omit unrequested groups. Runtime checks
public legality and action costs only; YOU choose the product and assess semantic match.
Unknown/unsatisfied requirements must be listed. Set accept_partial=true only if you actually
want to buy that partial match. The plan persists: do not repeat it unless updating it.
Q = known navigation + missing/wrong desired option selections + one Buy. FINISH is free.
At R <= Q+1, use current legal targets to return, select planned options, then Buy with normal
purchase_evidence. Do not reopen sections or search away. A reserved rejection costs zero
Actions: use its details to repair within 3 attempts. Starting at R=4, an inspected candidate
requires a feasible plan before further exploration. Do not invent a candidate or evidence.
To stop without buying, attach {"decision":"abandon","reason":"specific reason"} to a valid
current action; the attached environment action will NOT execute and you will report the reason.
A subtask pauses before using the protected closing path: report its candidate to the Director,
who can explicitly set THIS SAME node's result_scope to task_result and run it to finish.
Other nodes cannot spend the reservation or inherit this node's private shopping session.
"""

FLEXIBLE_GUIDANCE = """
Purchase completion (completion_reserve_v2): the entire question shares its action
budget. Complete the original shopping request, not just a research report.
Identify the requested PRODUCT first. Nearby words can describe its variant, color,
pattern or size rather than another product. If searches return the wrong product
type or only unaffordable items, change the query's product noun/attributes; do not
repeat the same query. Retain useful earlier candidates and compare before switching.

completion_plan is OPTIONAL metadata for a chosen public purchase path:
{"decision":"reserve","asin":"INSPECTED_ASIN","options":{"group":"desired value"}}.
It reserves navigation, desired selections and Buy; it is NOT a claim that every
requirement is already verified. Unresolved notes do not prevent inspection or
selection. Omit unrequested option groups unless the environment requires them.
Already selected desired values need no additional action. A public price range
entirely below the ceiling establishes affordability without an exact variant price.
Do not invent additional requirements or treat an unrequested option as a mismatch.

Without a plan, legal actions remain available when they leave the minimum public
path to Buy. Read error details and choose an affordable action; metadata is never
required merely to return to a product, select an option, inspect or purchase.
With a plan, follow its navigation/options budget; update it when YOUR intended
candidate or requested option values change. Semantic uncertainty is your judgment,
not a runtime quota. At Buy, report truthful verified_requirements and remaining
uncertainties in purchase_evidence. Choose the best observed relevant candidate if
no affordable useful investigation remains; do not refuse solely because of an
unrequested group or harmless price uncertainty. Never claim a purchase without Buy.
The runtime does not pick a product, choose requested options, or purchase for you.
FINISH commits a staged purchase without another Worker call or environment action.

For a non-purchase action, an invalid or unreachable optional reserve update may
be ignored with plan_update_feedback. The action is still checked against the
existing valid reservation and the remaining budget. Read that feedback; an
ignored update did NOT establish a new plan. Buy and explicit abandonment retain
their normal validation and are never executed by this recovery.
"""


def action_specs_with_plan(specs, policy=POLICY):
    result = []
    for spec in specs:
        params = copy.deepcopy(spec.parameters)
        params.setdefault("properties", {})["completion_plan"] = copy.deepcopy(PLAN_SCHEMA)
        if policy == FLEXIBLE_POLICY:
            params["properties"]["completion_plan"]["description"] = (
                "Optional public navigation/option/Buy budget plan. Only reserve decision, inspected asin and legal options are needed; unresolved semantic notes do not block actions.")
        result.append(replace(spec, parameters=params))
    return result


def actions(state):
    return [a for a in state.get("valid_subactions", []) if isinstance(a, dict)]


def asin(state):
    return str(state.get("product", {}).get("asin", "")).casefold()


def minimum_completion_cost(state, name, arguments):
    """Public lower bound including this action; requested options may cost more."""
    target = next((a for a in actions(state) if a.get("target_id") == arguments.get("target_id")), {})
    kind = target.get("kind") if name == "webshop_click" else None
    selected = dict(state.get("selected_options") or {})
    required = state.get("required_option_groups") or []
    if kind == "purchase":
        return 1
    if kind == "select_option":
        selected[target["option_name"]] = target["option_value"]
        return 2 + sum(k not in selected for k in required)
    if navigation_effect(state, target) == "return_to_current_product_page":
        return 2 + sum(k not in selected for k in required)
    if kind == "view_section":
        return 3 + sum(k not in selected for k in required)
    return 2 if kind == "open_product" else 3


def recovery_feedback(state, remaining, *, last_error=None, no_progress=0, searches=0):
    """A short final request reminder, computed only from current public state.

    The usual context rebuild drops previous assistant deliberation. Put an
    actionable correction after that large context instead of burying it in
    counters. This neither executes an action nor changes admission or budgets.
    """
    error = (last_error or {}).get("error") or {}
    if not error and no_progress < 2 and searches < 2 and remaining > 4:
        return None
    instructions = []
    if error:
        instructions.append("The preceding action was rejected without environment execution. Do not repeat it unchanged; use the error and current legal targets to repair it.")
        if error.get("code") == "invalid_completion_plan":
            instructions.append("completion_plan is optional. For an uninspected product, omit completion_plan from the open/inspection call; reserve only after observing that product. Do not invent an ASIN or option map to satisfy metadata.")
    if no_progress >= 2 or searches >= 2:
        instructions.append("Repeated actions/searches are not adding public evidence. Change strategy now: inspect an unvisited relevant result, or shorten/reformulate the query around the product noun and one distinguishing attribute. Reordering the same words or adding a price ceiling is not a useful new search. Consider retained candidates before spending the closing actions.")
    if remaining <= 4 or error.get("code") == "purchase_budget_reserved":
        instructions.append("Use the remaining actions for a feasible completion. Only requested or environment-required options need selection. Reading a section costs three steps including return and Buy. If no useful affordable inspection remains, YOU decide whether to Buy the best relevant observed candidate with honest unresolved constraints, or report why none is acceptable. Do not repeatedly request an unaffordable inspection/search.")
    choices = []
    for action in actions(state):
        target_id = action.get("target_id")
        if not target_id:
            continue
        cost = minimum_completion_cost(state, "webshop_click", {"target_id": target_id})
        if cost <= remaining:
            choices.append({k: action[k] for k in ("target_id", "kind", "option_name", "option_value") if k in action} | {"minimum_actions_including_buy": cost})
    return {"remaining_actions": remaining, "last_error_code": error.get("code"),
        "instructions": instructions, "public_completion_choices": choices[:48],
        "choices_truncated": len(choices) > 48,
        "semantics": "These are public minimum costs, not candidate rankings or semantic approval. A chosen plan's desired options may need more steps; its reservation and normal preflight still apply. Buy requires purchase_evidence. No action has been selected or executed."}


def quote(plan, state):
    """Exact cost only for a path justified by current public legal actions."""
    if any(state.get(k) for k in ("commit_pending", "purchased", "done")):
        return 0
    same = asin(state) == plan["asin"]
    page = state.get("page_type")
    if same and page == "product" and any(a.get("kind") == "purchase" for a in actions(state)):
        navigation, selected = 0, state.get("selected_options", {})
    elif same and page == "product_section" and any(
        navigation_effect(state, a) == "return_to_current_product_page" for a in actions(state)
    ):
        navigation, selected = 1, state.get("selected_options", {})
    elif page == "search_results" and any(visible_product_asin(a) == plan["asin"] for a in actions(state)):
        navigation, selected = 1, {}  # Opening a result clears selected options.
    else:
        return None
    if same and page == "product":
        legal = {(a.get("option_name"), a.get("option_value")) for a in actions(state) if a.get("kind") == "select_option"}
        if any(selected.get(k) != v and (k, v) not in legal for k, v in plan["options"].items()):
            return None
    return navigation + sum(selected.get(k) != v for k, v in plan["options"].items()) + 1


@dataclass
class PurchaseReservation:
    # A rollout serializes stateful executions under this lock, including check+consume.
    lock: object = field(default_factory=RLock, repr=False)
    plan: dict | None = None
    states: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    rejects: dict = field(default_factory=dict)
    plan_feedback: dict = field(default_factory=dict)
    blocked: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    version: int = 0
    policy: str = POLICY

    def event(self, kind, **details):
        self.events.append({"event": kind, **details})
        del self.events[:-128]

    def observe(self, owner, binding, state):
        state = copy.deepcopy(state)
        annotate_navigation(state)
        previous = self.bindings.get(owner)
        if previous is not None and previous != binding:
            self.seen.pop(owner, None)
            self.blocked.pop(owner, None)
            self.plan_feedback.pop(owner, None)
            if self.plan and self.plan["owner"] == owner:
                self.event("invalidated", owner=owner, reason="session_changed")
                self.plan = None
        self.bindings[owner] = binding
        self.states[owner] = state
        if binding and state.get("page_type") == "product" and asin(state):
            self.seen.setdefault(owner, {})[asin(state)] = copy.deepcopy(state)
        if self.plan and self.plan["owner"] == owner:
            if state.get("resource_status") == "unknown" or state.get("termination_reason") == "environment_step_failed":
                self.event("invalidated", owner=owner, reason="environment_state_unknown")
                self.plan = None
            elif any(state.get(k) for k in ("commit_pending", "purchased")):
                self.plan["phase"] = "staged" if state.get("commit_pending") else "submitted"
            elif self.policy == FLEXIBLE_POLICY and quote(self.plan, state) is None:
                # An old route cannot reserve fictitious actions on a new page.
                # Keep the observed candidate history; the model chooses a new plan.
                self.event("released", owner=owner, reason="return_path_no_longer_public")
                self.plan = None

    def reserved(self):
        return quote(self.plan, self.states.get(self.plan["owner"], {})) if self.plan else None

    def snapshot(self, remaining, viewer=None):
        q = self.reserved()
        if self.plan and q and remaining <= q + 1:
            self.plan["phase"] = "complete"
        result = {"policy": self.policy, "remaining": remaining, "reserved": q,
                  "owner": self.plan["owner"] if self.plan else None,
                  "version": self.version, "phase": self.plan["phase"] if self.plan else "explore",
                  "other_nodes_available": max(0, remaining - (q or 0)),
                  "plan_requested": self.policy == POLICY and self.plan is None and remaining <= 4,
                  "paused_nodes": copy.deepcopy(self.blocked),
                  "next_step": ("FINISH the staged owner; no Worker request or extra action is needed."
                    if self.plan and self.plan["phase"] == "staged" else
                    "Promote the reserved owner to task_result and RUN_AGENT; after Buy, FINISH that node."
                    if self.plan and q else "Establish a publicly executable purchase plan, or report why no acceptable candidate exists.")}
        if self.plan and viewer == self.plan["owner"]:
            result["completion_plan"] = {k: copy.deepcopy(v) for k, v in self.plan.items() if k != "binding"}
        if viewer in self.plan_feedback:
            result["plan_update_feedback"] = copy.deepcopy(self.plan_feedback[viewer])
        if self.policy == FLEXIBLE_POLICY:
            result.update(plan_required=False, semantic_notes_are_not_admission_checks=True)
            if not self.plan:
                result["next_step"] = "Choose a legal affordable shopping action; an optional plan can protect desired options and Buy. No plan is required to act or purchase."
            elif q:
                result["next_step"] = "The full-task owner can continue the reserved path, including affordable verification. A local subtask must hand off to that same owner with task_result scope. Buy remains the model's explicit decision."
        return result

    def blocker(self, owner, task_result, remaining):
        official = self.states.get(owner, {}).get("remaining_steps")
        if type(official) is int:
            remaining = min(remaining, max(0, official))
        self.snapshot(remaining)
        prior = self.blocked.get(owner)
        if prior and prior["reason"] in {"purchase_plan_abandoned", "purchase_plan_repair_exhausted"}:
            # Local failure can be reviewed once after explicit responsibility promotion.
            if prior["task_result"] or not task_result:
                return prior["reason"]
        if not self.plan:
            return None
        q = self.reserved()
        if q is None or q == 0:
            return None
        if owner != self.plan["owner"] and remaining <= q:
            return "purchase_reserve_other_owner"
        if owner == self.plan["owner"] and not task_result and self.plan["phase"] == "complete":
            return "purchase_reserve_promote_owner"
        return None

    def fail(self, owner, task_result, remaining, code, message):
        self.rejects[owner] = self.rejects.get(owner, 0) + 1
        stop = None
        if code == "purchase_plan_abandoned" or self.rejects[owner] >= 3:
            stop = code if code == "purchase_plan_abandoned" else "purchase_plan_repair_exhausted"
            self.blocked[owner] = {"reason": stop, "task_result": task_result}
        self.event("rejected", owner=owner, code=code, remaining=remaining, reserved=self.reserved())
        return {"code": code, "message": message,
                "details": {**self.snapshot(remaining, owner), "stop_reason": stop}}, False

    def preflight(self, *, owner, binding, state, name, arguments, remaining, task_result, task):
        self.observe(owner, binding, state)
        state = self.states[owner]
        self.plan_feedback.pop(owner, None)
        raw = arguments.get("completion_plan")
        target = next((a for a in actions(state) if a.get("target_id") == arguments.get("target_id")), {})
        # Advisory metadata may not block an otherwise valid navigation or
        # inspection. Never discard an existing reservation, a Buy's intended
        # selections, or an explicit decision to stop.
        recover_plan = (self.policy == FLEXIBLE_POLICY and isinstance(raw, dict)
            and raw.get("decision") == "reserve"
            and ((name == "webshop_search" and isinstance(arguments.get("query"), str)
                  and bool(arguments["query"].strip()))
                 or (name == "webshop_click" and target.get("kind") in
                     {"navigate", "open_product", "view_section", "select_option"})))

        def fail(code, message):
            if not recover_plan:
                return self.fail(owner, task_result, remaining, code, message)
            feedback = {"status": "ignored", "code": code, "message": message,
                "existing_plan_retained": self.plan is not None,
                "instruction": "The optional reserve update was not applied. The attached action is checked separately against the existing reservation and remaining budget. Do not repeat this plan unchanged."}
            self.plan_feedback[owner] = feedback
            self.event("plan_update_ignored", owner=owner, code=code, remaining=remaining,
                       existing_plan_retained=self.plan is not None)
            return self._admit_action(owner, task_result, state, name, arguments, remaining)

        if raw is not None:
            if not isinstance(raw, dict):
                return fail("invalid_completion_plan", "completion_plan must be an object.")
            if raw.get("decision") == "abandon":
                if not str(raw.get("reason", "")).strip():
                    return fail("invalid_completion_plan", "Abandon requires a specific reason.")
                if self.plan and self.plan["owner"] == owner:
                    self.plan = None
                self.event("abandoned", owner=owner, reason=raw["reason"])
                return fail("purchase_plan_abandoned", "Explicit abandonment recorded; attached action was not executed.")
            options = raw.get("options")
            candidate = str(raw.get("asin", "")).casefold()
            public = self.seen.get(owner, {}).get(candidate)
            quotes = raw.get("requirement_quotes")
            verified = raw.get("verified_requirements")
            unresolved = raw.get("unresolved_constraints")
            if (raw.get("decision") != "reserve" or not binding or not public
                    or not isinstance(options, dict) or len(options) > 32):
                return fail("invalid_completion_plan", "Use reserve, an inspected ASIN and a legal option map.")
            if self.policy == POLICY and (not isinstance(quotes, list) or not quotes
                    or any(not isinstance(q, str) or not q.strip() or q.casefold() not in task.casefold() for q in quotes)
                    or not isinstance(verified, list) or not verified
                    or any(not isinstance(v, str) or not v.strip() for v in verified)
                    or not isinstance(unresolved, list) or any(not isinstance(v, str) for v in unresolved)
                    or type(raw.get("accept_partial")) is not bool):
                return fail("invalid_completion_plan", "Use an inspected ASIN, legal option map, exact public task quotes, evidence, explicit unresolved list and accept_partial boolean.")
            legal = {(a.get("option_name"), a.get("option_value")) for a in actions(public) if a.get("kind") == "select_option"}
            if any(not isinstance(v, str) or (k, v) not in legal for k, v in options.items()):
                return fail("invalid_completion_plan", "Planned option group/value was not observed on this product.")
            if any(group not in options for group in public.get("required_option_groups", [])):
                return fail("invalid_completion_plan", "Choose a legal value for every environment-required option group.")
            if self.policy == POLICY and unresolved and not raw["accept_partial"]:
                return fail("invalid_completion_plan", "Resolve uncertainties before reserving, or explicitly accept this partial match; runtime does not choose for you.")
            candidate_plan = {**copy.deepcopy(raw), "asin": candidate, "owner": owner,
                              "binding": binding, "phase": "explore"}
            q = quote(candidate_plan, state)
            if q is None or q > remaining:
                return fail("purchase_budget_reserved", "The proposed path is unknown or unaffordable; existing reservation retained.")
            if self.plan and self.plan["owner"] != owner:
                return fail("purchase_budget_reserved", "Another node owns the reservation. Director must resume that owner or the owner must explicitly abandon.")
            if self.policy == POLICY and self.plan and self.plan["phase"] == "complete" and candidate != self.plan["asin"]:
                return fail("purchase_budget_reserved", "Complete the protected candidate or explicitly abandon; do not switch away during closing.")
            self.version += 1
            candidate_plan["version"] = self.version
            if self.plan and self.plan["phase"] == "complete":
                candidate_plan["phase"] = "complete"
            self.plan = candidate_plan
            self.event("reserved", owner=owner, reserved=q, remaining=remaining, version=self.version)
        return self._admit_action(owner, task_result, state, name, arguments, remaining)

    def _admit_action(self, owner, task_result, state, name, arguments, remaining):
        fail = lambda code, msg: self.fail(owner, task_result, remaining, code, msg)
        blocker = self.blocker(owner, task_result, remaining)
        if blocker:
            return {"code": "purchase_budget_reserved", "message": blocker,
                    "details": {**self.snapshot(remaining, owner), "stop_reason": blocker}}, False
        if self.plan is None:
            if self.policy == FLEXIBLE_POLICY:
                return self.unplanned_preflight(owner, task_result, state, name, arguments, remaining)
            if remaining <= 4 and self.seen.get(owner):
                return fail("completion_plan_required", "Before spending the last four actions, declare a feasible purchase plan or explicitly abandon.")
            self.rejects[owner] = 0
            return None, False
        q = self.reserved()
        if owner != self.plan["owner"]:
            if q is not None and remaining - 1 < q:
                return fail("purchase_budget_reserved", "This action would spend another node's purchase reservation.")
            self.rejects[owner] = 0
            return None, False
        selected = next((a for a in actions(state) if a.get("target_id") == arguments.get("target_id")), {})
        kind = selected.get("kind")
        after = copy.deepcopy(state)
        path_known = True
        if name == "webshop_click" and kind == "purchase":
            if asin(state) != self.plan["asin"] or q != 1:
                return fail("purchase_plan_incomplete", "Reach the reserved product and select every planned option before Buy.")
            q_after = 0
        elif name == "webshop_click" and kind == "select_option":
            after.setdefault("selected_options", {})[selected["option_name"]] = selected["option_value"]
            q_after = quote(self.plan, after)
        elif name == "webshop_click" and kind == "view_section":
            # Section page has an observed return contract, even before its target version exists.
            q_after = (q + 1) if state.get("page_type") == "product" and q is not None else q
        elif name == "webshop_click" and navigation_effect(state, selected) == "return_to_current_product_page" and state.get("page_type") == "product_section":
            q_after = q - 1 if q is not None else None
        elif name == "webshop_click" and kind == "open_product" and visible_product_asin(selected) == self.plan["asin"]:
            q_after = len(self.plan["options"]) + 1
        else:
            # Unknown search/back/other-product route: lower bound includes reopen and reselection.
            q_after = len(self.plan["options"]) + 2
            path_known = False
        progress = path_known and q is not None and q_after is not None and q_after < q
        if (q_after is None or 1 + q_after > remaining
                or (self.plan["phase"] == "complete" and not progress
                    and (self.policy == POLICY or not path_known))):
            return fail("purchase_budget_reserved", "Action plus resulting purchase path exceeds remaining actions, or does not advance the protected closing plan.")
        if not path_known:
            self.event("unknown_return_path", owner=owner, remaining=remaining,
                       message="Exploration allowed while affordable; next observation must revalidate reachability.")
        self.rejects[owner] = 0
        self.blocked.pop(owner, None)
        return None, progress

    def unplanned_preflight(self, owner, task_result, state, name, arguments, remaining):
        """Protect a public minimum route, without choosing a candidate or options.

        This floor is not a semantic/optimal plan. Unknown requested options can
        require more steps; their selection remains the Worker's responsibility.
        """
        target = next((a for a in actions(state) if a.get("target_id") == arguments.get("target_id")), {})
        kind = target.get("kind") if name == "webshop_click" else None
        cost = minimum_completion_cost(state, name, arguments)
        if cost > remaining:
            error, _ = self.fail(owner, task_result, remaining, "purchase_budget_reserved",
                "This action would leave too few steps for a public route to Buy. Choose a current affordable completion action; completion_plan is optional.")
            error["details"].update(minimum_required_actions=cost,
                current_page=state.get("page_type"), plan_required=False)
            return error, False
        self.rejects[owner] = 0
        self.blocked.pop(owner, None)
        progress = kind in {"purchase", "select_option"} or navigation_effect(state, target) == "return_to_current_product_page"
        return None, progress
