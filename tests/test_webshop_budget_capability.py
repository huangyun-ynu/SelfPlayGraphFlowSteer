from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.runtime import ActionBudgetLedger, _webshop_has_feasible_completion_path
from selfplay_graph_flowsteer.webshop import WebShopSessionLifecycle
from selfplay_graph_flowsteer.webshop_steps import PINNED_WEBSHOP_REVISION, environment_step_capacity
from selfplay_graph_flowsteer.webshop_sidecar import WebShopSession


UNBOUNDED = {"kind": "unbounded", "source_revision": PINNED_WEBSHOP_REVISION}


@pytest.mark.parametrize("state,expected", [
    ({}, ("unknown", None)),
    ({"remaining_steps": True}, ("unknown", None)),
    ({"remaining_steps": -1}, ("unknown", None)),
    ({"remaining_steps": 2}, ("finite", 2)),
    ({"environment_step_limit": UNBOUNDED}, ("unbounded", None)),
    ({"environment_step_limit": {"kind": "unbounded", "source_revision": "other"}}, ("unknown", None)),
    ({"environment_step_limit": UNBOUNDED, "remaining_steps": "unknown"}, ("unknown", None)),
    ({"environment_step_limit": UNBOUNDED, "remaining_steps": 1}, ("finite", 1)),
    ({"environment_step_limit": UNBOUNDED, "done": True}, ("finite", 0)),
])
def test_capability_never_invents_a_numeric_environment_limit(state, expected):
    assert environment_step_capacity(state) == expected


def test_sidecar_marks_only_the_verified_simulator_revision_unbounded():
    worker = SimpleNamespace(source_revision=PINNED_WEBSHOP_REVISION)
    session = WebShopSession("s", worker, SimpleNamespace(), "webshop/goal-1")
    result = session._project({"available_actions": ["search"]}, action_kind="reset")
    assert environment_step_capacity(result) == ("unbounded", None)
    assert "remaining_steps" not in result
    worker.source_revision = "unknown"
    result = session._project({"available_actions": ["search"]}, action_kind="reset")
    assert environment_step_capacity(result) == ("unknown", None)


@pytest.mark.parametrize("kind,remaining,allowance", [("finite", 2, 2), ("unbounded", None, 4)])
def test_closure_transfer_is_single_use_and_preserves_episode_cap(kind, remaining, allowance):
    node = AgentNode("owner", "Shop", initial_tool_budget=12, revision_tool_budget=4, total_tool_budget=16)
    ledger = ActionBudgetLedger()
    for _ in range(12):
        assert ledger.consume(node, revision=False)[0]
    assert ledger.begin_webshop_closure(node, session_id="same-session", official_remaining_steps=remaining,
                                        environment_step_limit_kind=kind)
    assert not ledger.begin_webshop_closure(node, session_id="other", official_remaining_steps=9)
    assert ledger.remaining(node, revision=True)["phase"] == 0
    for _ in range(allowance):
        assert ledger.consume(node, revision=True, closure_session="same-session")[0]
        ledger.observe_webshop_steps(node, remaining_steps=remaining, environment_step_limit_kind=kind)
    assert not ledger.consume(node, revision=True, closure_session="same-session")[0]
    assert ledger.usage["owner"].total_used == 12 + allowance
    assert ledger.webshop_audit(node)["environment_step_limit_kind"] == kind
    ledger.finish_webshop_closure(node)
    assert ledger.remaining(node, revision=True, closure_session="same-session")["phase"] == 0


def test_unknown_observation_after_unbounded_transfer_cannot_create_capacity():
    node = AgentNode("a", "Shop", initial_tool_budget=12, revision_tool_budget=4, total_tool_budget=16)
    ledger = ActionBudgetLedger()
    assert not ledger.begin_webshop_closure(node, session_id="s", official_remaining_steps=None)
    assert ledger.begin_webshop_closure(node, session_id="s", official_remaining_steps=None,
                                        environment_step_limit_kind="unbounded")
    ledger.observe_webshop_steps(node, remaining_steps=None)
    assert ledger.remaining(node, revision=True, closure_session="s")["phase"] == 0


def test_lifecycle_checks_owner_session_and_capability():
    life = WebShopSessionLifecycle(None)
    life._task = object()
    life._owner_agent = life._committer_agent = "a"
    life._active_session = "session"
    life._results["a"] = {"environment_step_limit": UNBOUNDED}
    assert life.closure_budget_context("a")["eligible"]
    assert not life.closure_budget_context("b")["eligible"]
    life._results["a"] = {}
    assert life.closure_budget_context("a")["reason"] == "official_remaining_steps_unknown"
    life._results["a"] = {"environment_step_limit": UNBOUNDED, "purchased": True}
    assert not life.closure_budget_context("a")["eligible"]


def test_fuse_continuation_requires_real_local_credit_when_environment_is_unbounded():
    state = {"page_type": "product", "environment_step_limit": UNBOUNDED,
             "purchase_visible": True, "unselected_option_groups": ["size"],
             "valid_subactions": [{"kind": "purchase", "target_id": "purchase:asin"}]}
    assert not _webshop_has_feasible_completion_path(state)
    assert not _webshop_has_feasible_completion_path(state, remaining_actions=1)
    assert _webshop_has_feasible_completion_path(state, remaining_actions=2)
