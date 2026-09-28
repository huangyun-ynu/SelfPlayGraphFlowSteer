from __future__ import annotations

import copy
import json

import pytest

from selfplay_graph_flowsteer.application import WebShopConfig
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool, WebShopSessionLifecycle
from selfplay_graph_flowsteer.webshop_evidence import EVIDENCE_PROFILE, PublicEvidenceLedger, public_price_check
from selfplay_graph_flowsteer.webshop_profiles import M02_PROFILE, section_memory_limit


REQUEST = "Buy a blue shirt, machine washable, under $30."
REQUIREMENTS = [
    {"id": "type", "quote": "shirt", "kind": "type", "strength": "required"},
    {"id": "color", "quote": "blue", "kind": "option", "strength": "required"},
    {"id": "wash", "quote": "machine washable", "kind": "attribute", "strength": "required"},
    {"id": "price", "quote": "under $30", "kind": "price", "strength": "required"},
]


class Shop:
    def __init__(self):
        self.calls = []
        self.version = 0
        self.selected = {}
        self.created = 0
        self.purchases = 0

    def state(self, page="product", asin="A"):
        return {"page_type": page, "state_version": self.version,
                "page_text": "Blue shirt. Machine washable. Price: $20. color: blue, green.",
                "product": {"asin": asin, "title": "Blue shirt", "price": 20},
                "selected_options": dict(self.selected),
                "valid_subactions": [
                    {"target_id": "buy", "kind": "purchase", "label": "Buy Now"},
                    {"target_id": "blue", "kind": "select_option", "option_name": "color", "option_value": "blue"},
                    {"target_id": "green", "kind": "select_option", "option_name": "color", "option_value": "green"},
                    {"target_id": "back", "kind": "navigate", "label": "Back to search"},
                ]}

    def create_session(self, goal_id, *, seed):
        self.created += 1
        return {"session_id": "same-session", **self.state()}

    def search(self, session_id, query):
        self.calls.append(("search", session_id, query))
        self.version += 1
        self.selected = {}
        return self.state()

    def click(self, session_id, target_id):
        self.calls.append(("click", session_id, target_id))
        self.version += 1
        if target_id in {"blue", "green"}:
            self.selected["color"] = target_id
        if target_id == "buy":
            self.purchases += 1
            return {**self.state(), "purchased": True, "done": True, "reward": 1}
        if target_id == "back":
            self.selected = {}
            return self.state("search_results", "B")
        return self.state()

    def close_session(self, session_id):
        pass


def begin(profile=EVIDENCE_PROFILE):
    shop = Shop()
    life = WebShopSessionLifecycle(shop, compatibility_profile=profile,
                                  search_observation_mode="legacy", max_observation_chars=0)
    life.bind_task(TaskSpec("task", REQUEST, metadata={"goal_id": "goal-1"}))
    state = life.begin_execution(agent_id="owner", seed=0, revision=False)
    return shop, life, state


def assessment(state):
    source = next(s for s in state["public_decision"]["current_sources"] if s["field"] == "page_text")
    quotes = {"type": "shirt", "color": "blue", "wash": "Machine washable", "price": "$20"}
    return {"requirements": REQUIREMENTS, "best_candidate_asin": "A", "candidate": {
        "asin": "A", "checks": [
            {"requirement_id": key, "status": "supported",
             "references": [{"source_id": source["source_id"], "quote": quote}],
             **({"option_group": "color", "option_value": "blue"} if key == "color" else {})}
            for key, quote in quotes.items()]}}


def test_sources_quotes_and_option_readback_have_different_authority():
    shop, life, state = begin()
    decision = assessment(state)
    updated = life._public_evidence.updated(decision, state)
    checks = {c["requirement_id"]: c for c in updated.candidate_checks("A", state)}
    assert checks["type"]["status"] == "supported"
    assert checks["color"]["status"] == "unknown"
    with pytest.raises(ValueError, match="accept_unresolved_reason"):
        updated.purchase({"asin": "A", "state_version": 0}, state)
    selected = life.click("blue", decision=decision, state_version=0)
    checks = {c["requirement_id"]: c for c in selected["public_decision"]["candidates"][0]["checks"]}
    assert checks["color"]["status"] == "supported"
    assert checks["color"]["readback"]["state_version"] == 1
    changed = life.click("green", state_version=1)
    assert next(c for c in changed["public_decision"]["candidates"][0]["checks"] if c["requirement_id"] == "color")["status"] == "unknown"
    away = life.click("back", state_version=2)
    assert next(c for c in away["public_decision"]["candidates"][0]["checks"] if c["requirement_id"] == "color")["status"] == "unknown"


@pytest.mark.parametrize("tamper", ["invent_quote", "cross_product", "invent_requirement", "invent_option"])
def test_invalid_declarations_are_read_only_preflight_rejections(tamper):
    shop, life, state = begin()
    decision = copy.deepcopy(assessment(state))
    if tamper == "invent_quote":
        decision["candidate"]["checks"][0]["references"][0]["quote"] = "definitely waterproof"
    elif tamper == "cross_product":
        life._public_evidence.observe(shop.state(asin="B"))
        decision["candidate"]["asin"] = "B"
    elif tamper == "invent_requirement":
        decision["requirements"][0]["quote"] = "buy a boat"
    else:
        decision["candidate"]["checks"][1]["option_value"] = "purple"
    before = copy.deepcopy(life._public_evidence)
    rejection = WebShopClickTool(life).preflight({"target_id": "blue", "state_version": 0, "decision": decision})
    assert rejection["code"] == "webshop_public_evidence_invalid"
    assert shop.calls == []
    assert life._public_evidence == before


def test_current_asin_version_and_full_requirement_coverage_required_for_purchase():
    shop, life, state = begin()
    decision = assessment(state)
    life.click("blue", state_version=0, decision=decision)
    for evidence in ({"asin": "B", "state_version": 1}, {"asin": "A", "state_version": 0}):
        assert WebShopClickTool(life).preflight({"target_id": "buy", "state_version": 1, "purchase_evidence": evidence})
    del life._public_evidence.candidates["A"]["checks"]["wash"]
    rejection = WebShopClickTool(life).preflight({"target_id": "buy", "state_version": 1, "purchase_evidence": {"asin": "A", "state_version": 1}})
    assert "every public requirement" in rejection["message"]
    assert shop.purchases == 0


def test_unknown_final_choice_allowed_and_early_committer_cannot_skip_validation():
    shop, life, state = begin()
    life.set_committer("owner")
    evidence = {"asin": "A", "state_version": 0, "accept_unresolved_reason": "Final action; partial choice recorded honestly"}
    decision = assessment(state)
    decision["candidate"]["checks"][2] = {"requirement_id": "wash", "status": "unknown", "references": []}
    with pytest.raises(ValueError):
        life.click("buy", state_version=0, purchase_evidence=evidence)
    bought = life.click("buy", state_version=0, decision=decision, purchase_evidence=evidence)
    assert bought["purchased"]
    assert set(bought["purchase_evidence_status"]["evidence"]["unresolved_requirement_ids"]) == {"wash", "color"}
    life.click("buy", state_version=0, decision=decision, purchase_evidence=evidence)
    assert shop.purchases == 1
    assert shop.created == 1


def test_memory_survives_revision_pins_best_quotes_and_resets_with_new_task():
    shop, life, state = begin()
    life.click("blue", state_version=0, decision=assessment(state))
    for number in range(8):
        life._public_evidence.observe(shop.state(asin=str(number)))
    ledger = life._public_evidence
    assert "A" in ledger.candidates and len(ledger.candidates) == 6
    assert ledger.candidates["A"]["checks"]["wash"]["references"][0]["quote"] == "Machine washable"
    life.end_execution()
    resumed = life.begin_execution(agent_id="owner", seed=0, revision=True)
    assert resumed["session_reused"]
    assert shop.created == 1
    life.bind_task(TaskSpec("next", "Buy pants", metadata={"goal_id": "goal-2"}))
    assert life._public_evidence.requirements == {}
    assert life._public_evidence.sources == {}


@pytest.mark.parametrize("price,expected", [
    ({"price": 20}, "supported"), ({"price": 30}, "conflict"),
    ({"price": 20, "price_min": 20, "price_max": 40}, "unknown"),
    ({"price_min": 40, "price_max": 50}, "conflict"), ({}, "unknown"),
])
def test_public_price_ranges_not_reduced_to_low_endpoint(price, expected):
    assert public_price_check("under $30", {"product": price})["status"] == expected
    assert public_price_check("at most $30", {"product": {"price": 30}})["status"] == "supported"


def test_opt_in_contract_preserves_frozen_profile_and_schema():
    shop, life, _ = begin()
    assert "decision" in WebShopSearchTool(life).parameters["properties"]
    assert "asin" in WebShopClickTool(life).parameters["properties"]["purchase_evidence"]["properties"]
    baseline = WebShopSessionLifecycle(shop, compatibility_profile=M02_PROFILE)
    assert "decision" not in WebShopSearchTool(baseline).parameters["properties"]
    assert "verified_requirements" in WebShopClickTool(baseline).parameters["properties"]["purchase_evidence"]["properties"]
    assert section_memory_limit(M02_PROFILE) == section_memory_limit(EVIDENCE_PROFILE) == 1400
    with pytest.raises(ValueError, match="enabled together"):
        WebShopConfig(worker_guidance_policy="public_evidence_v1").validate()


def test_runtime_invalid_evidence_does_not_consume_action_or_change_session():
    from selfplay_graph_flowsteer.contracts import AgentNode
    from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
    from selfplay_graph_flowsteer.llm import MockBackend
    from selfplay_graph_flowsteer.runtime import ModelAgentExecutor

    shop, life, state = begin()
    life.end_execution()
    tools = {"webshop_search": WebShopSearchTool(life), "webshop_click": WebShopClickTool(life)}
    actions = [
        ("webshop_search", {"query": "shirt", "decision": {"requirements": [
            {**REQUIREMENTS[0], "quote": "invented request"}]}}),
        ("webshop_click", {"target_id": "blue", "state_version": 0, "decision": assessment(state)}),
        ("webshop_click", {"target_id": "buy", "state_version": 1,
                           "purchase_evidence": {"asin": "A", "state_version": 1}}),
    ]
    backend = MockBackend([json.dumps({"action_call": {"name": name, "arguments": args}})
                           for name, args in actions] + [json.dumps({"answer": "Purchase staged"})])
    executor = ModelAgentExecutor(backend, tools=tools,
                                 action_registry=default_dataset_action_registry(available_actions=tuple(tools)),
                                 webshop_worker_guidance_policy="public_evidence_v1",
                                 webshop_compatibility_profile=EVIDENCE_PROFILE)
    node = AgentNode("owner", "Find the requested product", allowed_tools=tuple(tools),
                     operation_policy_configured=True, initial_tool_budget=12,
                     revision_tool_budget=4, total_tool_budget=16,
                     metadata={"action_adapter": "webshop", "runtime_route": "deepseek"})
    artifact = executor.execute(task=REQUEST, node=node, upstream=[], peers=[], revision=False, seed=0)
    assert life.commit_ready_agents() == ("owner",)
    assert shop.calls == [("click", "same-session", "blue")]
    assert shop.created == 1 and shop.purchases == 0
    assert artifact.webshop_progress["action_budget"]["total_used"] == 2


@pytest.mark.parametrize("profile", ["public_evidence_v2", "public_evidence_review_v2"])
def test_annotation_failure_does_not_block_information_action_but_purchase_stays_strict(profile):
    shop, life, state = begin(profile)
    decision = assessment(state)
    decision["candidate"]["checks"][0]["references"][0]["quote"] = "invented waterproof claim"
    tool = WebShopClickTool(life)
    assert tool.preflight({"target_id": "blue", "state_version": 0, "decision": decision}) is None
    result = life.click("blue", state_version=0, decision=decision)
    assert shop.calls == [("click", "same-session", "blue")]
    assert result["decision_feedback"]["assessment_accepted"] is False
    assert life._public_evidence.requirements == {r["id"]: r for r in REQUIREMENTS}
    assert life._public_evidence.candidates["A"]["checks"] == {}
    args = {"target_id": "buy", "state_version": 1, "decision": decision,
            "purchase_evidence": {"asin": "A", "state_version": 1}}
    assert tool.preflight(args)["code"] == "webshop_public_evidence_invalid"
    with pytest.raises(ValueError, match="observed source"):
        life.click("buy", state_version=1, decision=decision, purchase_evidence=args["purchase_evidence"])
    assert shop.purchases == 0
    corrected = assessment(state)
    assert tool.preflight({**args, "decision": corrected}) is None
    assert life.click("buy", state_version=1, decision=corrected,
                      purchase_evidence=args["purchase_evidence"])["commit_ready"]


def test_missing_requirements_allow_search_in_v2_and_invalid_targets_still_reject():
    shop, life, state = begin("public_evidence_v2")
    search = WebShopSearchTool(life)
    assert search.preflight({"query": "shirt"}) is None
    result = json.loads(search.execute({"query": "shirt"}))
    assert result["decision_feedback"]["assessment_accepted"] is False
    assert WebShopClickTool(life).preflight({"target_id": "stale", "state_version": 1})["code"] == "webshop_target_not_in_current_state"
