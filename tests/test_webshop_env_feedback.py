from __future__ import annotations

import copy

import pytest

from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import _webshop_context_for_prompt
from selfplay_graph_flowsteer.webshop import WebShopSessionLifecycle


class PublicClient:
    def __init__(self):
        self.calls = []
        self.fail_search = False

    def page(self, page_type):
        state = {
            "page_type": page_type,
            "page_text": "Visible page",
            "valid_subactions": [
                {"target_id": "back_to_search:1", "kind": "navigate", "label": "back to search"},
                {"target_id": "open_product:0:B012345678", "kind": "open_product", "label": "Product"},
            ],
        }
        if page_type == "product":
            state["product"] = {"asin": "B012345678", "title": "Product"}
        return state

    def create_session(self, goal_id, *, seed):
        return {"session_id": "session", **self.page("search")}

    def search(self, session_id, query):
        if self.fail_search:
            raise RuntimeError("transport failed")
        self.calls.append(("search", query))
        return self.page("search_results")

    def click(self, session_id, target_id):
        self.calls.append(("click", target_id))
        return self.page("product" if target_id.startswith("open_product:") else "search")

    def close_session(self, session_id):
        pass


def begin(*, enabled=True):
    client = PublicClient()
    lifecycle = WebShopSessionLifecycle(
        client, env_feedback_enabled=enabled, search_observation_mode="legacy"
    )
    lifecycle.bind_task(TaskSpec("task", "Buy a product", metadata={"goal_id": "goal-1"}))
    lifecycle.begin_execution(agent_id="agent_1", seed=0, revision=False)
    return lifecycle, client


def test_repeat_feedback_survives_revision_but_not_new_task():
    lifecycle, client = begin()
    assert "env_feedback" not in lifecycle.search("blue shirt")
    back = lifecycle.click("back_to_search:1")
    assert "immediately after a search result page" in back["env_feedback"]
    lifecycle.end_execution()
    lifecycle.begin_execution(agent_id="agent_1", seed=0, revision=True)
    repeated = lifecycle.search("blue shirt")
    assert "reused after Back" in repeated["env_feedback"]
    # Feedback is observational: the repeated search still reached the client.
    assert client.calls.count(("search", "blue shirt")) == 2
    assert "env_feedback" not in lifecycle.search("navy shirt")
    lifecycle.bind_task(TaskSpec("next", "Buy a product", metadata={"goal_id": "goal-2"}))
    lifecycle.begin_execution(agent_id="agent_1", seed=0, revision=False)
    assert "env_feedback" not in lifecycle.search("blue shirt")


def test_return_feedback_uses_public_product_and_preserves_actions():
    lifecycle, client = begin()
    lifecycle.search("shirt")
    lifecycle.click("open_product:0:B012345678")
    result = lifecycle.click("back_to_search:1")
    assert "ASIN B012345678" in result["env_feedback"]
    assert result["valid_subactions"] == client.page("search")["valid_subactions"]
    assert result["page_text"] == "Visible page"
    assert "B012345678" not in lifecycle.click("back_to_search:1")["env_feedback"]


def test_disabled_feedback_keeps_baseline_observations_identical():
    on, _ = begin()
    off, _ = begin(enabled=False)
    for lifecycle in (on, off):
        lifecycle.search("shirt")
    enabled = on.search("shirt")
    disabled = off.search("shirt")
    assert "already used earlier" in enabled.pop("env_feedback")
    assert enabled == disabled
    assert off._visible_action_history == []


def test_failed_search_is_not_reported_as_previously_executed():
    lifecycle, client = begin()
    client.fail_search = True
    with pytest.raises(RuntimeError, match="transport failed"):
        lifecycle.search("shirt")
    client.fail_search = False
    assert "env_feedback" not in lifecycle.search("shirt")


def test_feedback_reaches_worker_even_when_page_text_is_truncated():
    lifecycle, _ = begin()
    lifecycle.search("shirt")
    state = lifecycle.search("shirt")
    state["page_text"] = "x" * 12000
    context = {"action_environment": {"state": state}}
    original = copy.deepcopy(context)
    projected = _webshop_context_for_prompt(context)
    assert projected["action_environment"]["state"]["env_feedback"] == state["env_feedback"]
    assert len(projected["action_environment"]["state"]["page_text"]) < 12000
    assert context == original
