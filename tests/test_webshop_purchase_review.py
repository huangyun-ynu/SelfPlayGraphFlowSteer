import json

import pytest

from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.webshop_evidence import REVIEW_PROFILE
from .test_webshop_public_evidence import REQUEST, assessment, begin


def partial(state):
    decision = assessment(state)
    decision["candidate"]["checks"][2] = {"requirement_id": "wash", "status": "unknown", "references": []}
    return decision


def call(name, args):
    return json.dumps({"action_call": {"name": name, "arguments": args}})


@pytest.mark.parametrize("early_committer", [False, True])
def test_runtime_yields_before_staging_and_same_owner_resumes_without_extra_action(early_committer):
    shop, life, state = begin(REVIEW_PROFILE)
    if early_committer:
        life.set_committer("owner")
    life.end_execution()
    tools = {"webshop_search": WebShopSearchTool(life), "webshop_click": WebShopClickTool(life)}
    evidence = {"asin": "A", "state_version": 1, "accept_unresolved_reason": "No more useful evidence on this public page"}
    backend = MockBackend([
        call("webshop_click", {"target_id": "blue", "state_version": 0, "decision": partial(state)}),
        call("webshop_click", {"target_id": "buy", "state_version": 1, "purchase_evidence": evidence}),
        json.dumps({"answer": "Review needed", "unresolved_issues": ["washability unknown"]}),
    ])
    executor = ModelAgentExecutor(backend, tools=tools,
        action_registry=default_dataset_action_registry(available_actions=tuple(tools)),
        webshop_worker_guidance_policy="public_evidence_v1", webshop_compatibility_profile=REVIEW_PROFILE)
    node = AgentNode("owner", "Find the requested product", allowed_tools=tuple(tools),
        operation_policy_configured=True, initial_tool_budget=12, revision_tool_budget=4, total_tool_budget=16,
        metadata={"action_adapter": "webshop", "runtime_route": "deepseek"})
    artifact = executor.execute(task=REQUEST, node=node, upstream=[], peers=[], revision=False, seed=0)
    assert artifact.webshop_progress["state"] == "review_pending"
    assert artifact.webshop_progress["action_budget"]["total_used"] == 1
    assert artifact.webshop_progress["semantic_no_progress_count"] == 0
    assert all(t["observation"]["status"] == "ok" for t in artifact.react_trace)
    assert not life.commit_ready_agents() and shop.purchases == 0
    review = artifact.webshop_progress["purchase_review"]
    assert review["proposal"]["unresolved_requirement_ids"] == ["wash"]
    assert "reward" not in json.dumps(review)
    backend.responses.extend([
        call("webshop_click", {"target_id": "buy", "state_version": 1,
                              "purchase_evidence": {**evidence, "review_receipt": review["review_receipt"],
                                                    "review_reason": "Reviewed public evidence and remaining budget; final partial choice"}}),
        json.dumps({"answer": "Final choice"}),
    ])
    resumed = executor.execute(task=REQUEST, node=node, upstream=[], peers=[], revision=True, seed=0)
    assert resumed.webshop_progress["action_budget"]["total_used"] == 2
    assert shop.created == 1
    assert shop.purchases == int(early_committer)
    assert bool(life.commit_ready_agents()) is not early_committer


def test_receipt_cannot_skip_scheduled_review_or_survive_option_change():
    shop, life, state = begin(REVIEW_PROFILE)
    life.click("blue", state_version=0, decision=partial(state))
    evidence = {"asin": "A", "state_version": 1, "accept_unresolved_reason": "Partial final choice"}
    arguments = {"target_id": "buy", "state_version": 1, "purchase_evidence": evidence}
    tool = WebShopClickTool(life)
    before = life._purchase_review
    rejection = tool.preflight(arguments)
    assert life._purchase_review is before  # genuinely read-only preflight
    state = tool.defer_purchase_review(arguments, yield_to_director=True)
    acknowledged = {**evidence, "review_receipt": state["purchase_review"]["review_receipt"], "review_reason": "Rechecked"}
    assert tool.preflight({**arguments, "purchase_evidence": acknowledged})
    # Repeating the proposal inside the same execution cannot release the latch.
    tool.defer_purchase_review(arguments, yield_to_director=True)
    with pytest.raises(ValueError, match="Review"):
        life.click("buy", state_version=1, purchase_evidence=acknowledged)
    life.end_execution()
    life.begin_execution(agent_id="owner", seed=0, revision=True)
    assert tool.preflight({**arguments, "purchase_evidence": acknowledged}) is None
    life.click("green", state_version=1)
    stale = {**acknowledged, "state_version": 2}
    assert tool.preflight({"target_id": "buy", "state_version": 2, "purchase_evidence": stale})
    assert shop.purchases == 0 and shop.created == 1


def test_final_revision_can_review_inline_and_fully_supported_purchase_needs_no_review():
    shop, life, state = begin(REVIEW_PROFILE)
    life.set_committer("owner")
    life.click("blue", state_version=0, decision=partial(state))
    evidence = {"asin": "A", "state_version": 1, "accept_unresolved_reason": "Last available action"}
    args = {"target_id": "buy", "state_version": 1, "purchase_evidence": evidence}
    state = WebShopClickTool(life).defer_purchase_review(args, yield_to_director=False)
    assert state["purchase_review"]["can_acknowledge"]
    result = life.click("buy", state_version=1, purchase_evidence={**evidence,
        "review_receipt": state["purchase_review"]["review_receipt"], "review_reason": "No useful action before final purchase"})
    assert result["purchased"] and shop.purchases == 1
    life.click("buy", state_version=1)
    assert shop.purchases == 1

    shop, life, state = begin(REVIEW_PROFILE)
    life.click("blue", state_version=0, decision=assessment(state))
    assert WebShopClickTool(life).preflight({"target_id": "buy", "state_version": 1,
        "purchase_evidence": {"asin": "A", "state_version": 1}}) is None
