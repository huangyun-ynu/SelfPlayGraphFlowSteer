from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.application import (
    WebShopConfig,
    create_adaptive_application,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, RoutedModelAgentExecutor
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.webshop_guidance import LASER_PAGE_CHECKLIST

from .test_application import write_config
from .test_webshop_env_feedback import begin


def setup_worker(policy="baseline", *, routed=False):
    lifecycle, client = begin(enabled=False)
    lifecycle.end_execution()
    tools = {
        "webshop_search": WebShopSearchTool(lifecycle),
        "webshop_click": WebShopClickTool(lifecycle),
    }
    backend = MockBackend(
        [
            json.dumps({"action_call": {"name": "webshop_search", "arguments": {"query": "shirt"}}}),
            json.dumps({"action_call": {"name": "webshop_click", "arguments": {"target_id": "open_product:0:B012345678"}}}),
            json.dumps({"answer": "Inspected product", "evidence": ["Product"], "unresolved_issues": ["No purchase yet"]}),
        ]
    )
    kwargs = {
        "tools": tools,
        "action_registry": default_dataset_action_registry(available_actions=tuple(tools)),
        "webshop_worker_guidance_policy": policy,
    }
    executor = (
        RoutedModelAgentExecutor({"deepseek": backend}, ("deepseek",), **kwargs)
        if routed
        else ModelAgentExecutor(backend, **kwargs)
    )
    node = AgentNode(
        "agent_1", "Inspect a product", allowed_tools=tuple(tools),
        operation_policy_configured=True,
        initial_tool_budget=12, revision_tool_budget=4, total_tool_budget=16,
        metadata={"action_adapter": "webshop", "runtime_route": "deepseek"},
    )
    return executor, backend, node, client


def execute(executor, node, *, revision=False):
    return executor.execute(
        task="Buy a shirt", node=node, upstream=[], peers=[], revision=revision, seed=0
    )


@pytest.mark.parametrize("routed", [False, True])
def test_checklist_changes_only_owner_system_prompt_not_actions_or_observations(routed):
    off, off_backend, off_node, off_client = setup_worker(routed=routed)
    on, on_backend, on_node, on_client = setup_worker("laser_checklist_v1", routed=routed)
    off_artifact = execute(off, off_node)
    on_artifact = execute(on, on_node)

    assert len(on_backend.calls) == len(off_backend.calls) == 3
    assert on_client.calls == off_client.calls
    for baseline, candidate in zip(off_backend.calls, on_backend.calls, strict=True):
        assert candidate["messages"][0]["content"] == baseline["messages"][0]["content"] + LASER_PAGE_CHECKLIST
        assert candidate["messages"][1:] == baseline["messages"][1:]
        assert {k: v for k, v in candidate.items() if k != "messages"} == {
            k: v for k, v in baseline.items() if k != "messages"
        }
    assert on_artifact.webshop_progress["action_budget"] == off_artifact.webshop_progress["action_budget"]
    assert "worker_guidance" not in off_artifact.webshop_progress
    assert on_artifact.webshop_progress["worker_guidance"] == {
        "policy": "laser_checklist_v1", "applied": True,
    }


def test_owner_revision_keeps_guidance_but_stateless_reviewer_does_not_receive_it():
    executor, backend, owner, _ = setup_worker("laser_checklist_v1", routed=True)
    execute(executor, owner)
    backend.responses.append(json.dumps({"answer": "Inspect remaining attributes"}))
    execute(executor, owner, revision=True)
    assert LASER_PAGE_CHECKLIST in backend.calls[-1]["messages"][0]["content"]
    revision_context = json.loads(backend.calls[-1]["messages"][-1]["content"])
    assert revision_context["action_environment"]["state"]["session_reused"] is True

    reviewer = AgentNode(
        "reviewer", "Review evidence", allowed_tools=owner.allowed_tools,
        metadata=owner.metadata.copy(),
    )
    backend.responses.append(json.dumps({"answer": "Need more evidence"}))
    artifact = execute(executor, reviewer)
    assert LASER_PAGE_CHECKLIST not in backend.calls[-1]["messages"][0]["content"]
    assert artifact.webshop_progress["worker_guidance"]["applied"] is False


def test_other_datasets_do_not_receive_webshop_guidance():
    backend = MockBackend([json.dumps({"answer": "42"})])
    executor = ModelAgentExecutor(backend, webshop_worker_guidance_policy="laser_checklist_v1")
    execute(executor, AgentNode("math", "Compute"))
    assert LASER_PAGE_CHECKLIST not in backend.calls[0]["messages"][0]["content"]


def test_profile_and_application_wiring(tmp_path):
    path = write_config(tmp_path)
    config = load_adaptive_config(path)
    assert config.webshop.worker_guidance_policy == "baseline"
    with path.open("a") as handle:
        handle.write('\n[webshop]\nenabled = true\nworker_guidance_policy = "laser_checklist_v1"\n')
    config = load_adaptive_config(path)
    app = create_adaptive_application(config, mock=True)
    assert app.runtime.executor.webshop_worker_guidance_policy == "laser_checklist_v1"

    root = Path(__file__).resolve().parents[1]
    baseline = tomllib.loads((root / "configs/webshop_official_eval.toml").read_text())
    candidate = tomllib.loads((root / "configs/webshop_laser_checklist_eval.toml").read_text())
    assert candidate["webshop"].pop("worker_guidance_policy") == "laser_checklist_v1"
    assert candidate == baseline


def test_misspelled_policy_is_rejected():
    with pytest.raises(ValueError, match="worker_guidance_policy"):
        WebShopConfig(worker_guidance_policy="lasre").validate()
