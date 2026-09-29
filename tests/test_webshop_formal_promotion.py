from __future__ import annotations

import argparse
import copy
import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import application as app_module
from selfplay_graph_flowsteer.cli import _apply_fresh_route_report
from selfplay_graph_flowsteer.config import ModelGatewayConfig
from selfplay_graph_flowsteer.endpoint_pool import EndpointPoolBackend
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import (
    _update_webshop_product_inspections,
    _webshop_progress_prompt,
)
from selfplay_graph_flowsteer.webshop import WebShopSessionLifecycle
from selfplay_graph_flowsteer.webshop_profiles import M02_PROFILE

from .test_webshop_env_feedback import PublicClient


def load_formal(name, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "configs" / name
    monkeypatch.setattr(app_module, "_load_project_env", lambda _: None)
    payload = tomllib.loads(path.read_text())
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", ",".join(
        map(str, payload["resources"]["allocated_gpu_ids"])))
    for runtime in payload["runtimes"].values():
        if runtime.get("api_key_env"):
            monkeypatch.setenv(runtime["api_key_env"], "synthetic-test-key")
    config = app_module.load_adaptive_config(path, validate=False)
    # Validate the actual WebShop configuration without requiring deployment
    # assets or verifier credentials for unrelated ALFWorld/SWE adapters.
    config = replace(
        config,
        alfworld=replace(config.alfworld, enabled=False),
        swe=replace(config.swe, enabled=False),
    )
    config.validate()
    return config


@pytest.mark.parametrize("name", ["formal_training.toml", "formal_training_h200.local.toml",
                                  "formal_eval_h200.local.toml"])
def test_formal_m02_config_preserves_model_choices_and_thinking(name, monkeypatch):
    config = load_formal(name, monkeypatch)
    assert config.webshop.compatibility_profile == M02_PROFILE
    assert config.webshop.worker_guidance_policy == "merged_checklist_v1"
    assert config.worker_routes_for("webshop") == (
        "gpt", "grok", "gemini", "deepseek", "minimax")
    assert not config.dataset_route_overrides.get("webshop")
    assert config.solver_model.enable_thinking is True
    assert config.runtime_endpoint_pools["grok"] == ("grok", "grok45")
    assert config.runtime_endpoint_pools["gemini"] == ("gemini", "gemini2")
    assert config.endpoint_pool_member_queue_wait_s == 0.5
    assert config.skillbank_enabled
    assert not config.canvas.remaining_token_admission_enabled
    assert config.model_manifest()["webshop"]["compatibility_profile"] == M02_PROFILE


def test_m02_rejects_accidental_native_or_history_mix(monkeypatch):
    config = load_formal("formal_training.toml", monkeypatch)
    for change in ({"worker_execution_policy": "skillflow_native_v1"},
                   {"worker_memory_policy": "skillflow_history_v1"},
                   {"worker_guidance_policy": "laser_checklist_v1"}):
        with pytest.raises(ValueError, match="M02 requires"):
            replace(config.webshop, **change).validate()


def test_m02_memory_restores_1400_chars_in_storage_and_checkpoint():
    inspections = {}
    state = {"page_type": "product_section", "product": {"asin": "B012345678"}}
    for section, text in (("description", "D" * 1800), ("features", "F" * 2000)):
        _update_webshop_product_inspections(
            action_name="webshop_click", arguments={"target_id": f"view_{section}:B012345678"},
            output={**state, "page_text": "Instruction:\nBuy item\n" + text},
            product_inspections=inspections, section_max_chars=1400,
        )
    expected = {"description": "D" * 1400, "features": "F" * 1400}
    assert inspections["b012345678"]["section_evidence"] == expected
    prompt = _webshop_progress_prompt(
        queries=[], visited_products=[], product_inspections=inspections, candidate_ledger={},
        strategy_variant="factual_state_only", recent_actions=[], duplicate_action_count=0,
        semantic_no_progress_count=0, semantic_no_progress_streak=0,
        searches_since_last_product_open=0, current_state=state, section_max_chars=1400,
    )
    assert prompt["decision_checkpoint"]["retained_section_evidence"] == expected


def test_m02_projection_removes_new_native_metadata_without_mutating_sidecar_payload():
    payload = PublicClient().page("search_results")
    payload["raw_available_actions"] = {"clickables": ["B012345678"]}
    payload["valid_subactions"][0]["raw_action"] = "Back to Search"
    before = copy.deepcopy(payload)
    current = WebShopSessionLifecycle(None, search_observation_mode="legacy")._bounded(payload)
    m02 = WebShopSessionLifecycle(None, search_observation_mode="legacy",
                                 compatibility_profile=M02_PROFILE)._bounded(payload)
    assert "raw_available_actions" in current
    assert "raw_available_actions" not in m02
    assert all("raw_action" not in action for action in m02["valid_subactions"])
    assert payload == before


def test_route_preflight_keeps_dataset_candidates_and_filters_unhealthy_ones(tmp_path, monkeypatch):
    config = load_formal("formal_training.toml", monkeypatch)
    requested = list(config.runtime_pool())
    report = tmp_path / "routes.json"
    args = argparse.Namespace(route_report=report, mock=False, max_route_report_age_s=1800,
                              minimum_selected_routes=1, route_subset="")
    report.write_text(json.dumps({"routes_requested": requested,
                                  "usable_routes": [r for r in requested if r not in {"gemini", "gemini2"}]}))
    selected, audit = _apply_fresh_route_report(config, args)
    assert selected.worker_runtime_routes == tuple(
        route for route in config.worker_runtime_routes if route not in {"gemini", "gemini2"})
    assert "gemini" not in selected.worker_routes_for("webshop")
    assert "deepseek" in selected.worker_routes_for("webshop")
    assert selected.worker_routes_for("webshop") == tuple(audit["dataset_worker_routes"]["webshop"])
    assert set(selected.worker_routes_for("webshop")) <= selected.runtime_pool().keys()
    report.write_text(json.dumps({"routes_requested": [r for r in requested if r != "minimax"],
                                  "usable_routes": requested}))
    with pytest.raises(ValueError, match="all configured route candidates"):
        _apply_fresh_route_report(config, args)


def test_actual_request_slot_skips_full_queue_without_consuming_http_timeout(tmp_path, monkeypatch):
    from selfplay_graph_flowsteer import llm

    waits, sent = [], []

    class Gate:
        def __init__(self, route):
            self.route = route

        def acquire(self, *, priority, timeout):
            waits.append((self.route, timeout))
            return self.route == "free"

        def release(self):
            pass

    class Backend:
        def __init__(self, route):
            self.config = ModelGatewayConfig(base_url="https://" + route, route_name=route, timeout_s=240)

        def generate(self, messages, **kwargs):
            with llm._request_slot(self.config, None) as slot:
                sent.append((self.config.route_name, slot.timeout_s))
                return SimpleNamespace(metadata={}, text="ok")

    monkeypatch.setattr(llm, "_request_gate", lambda config: Gate(config.route_name))
    pool = EndpointPoolBackend("gpt", {name: Backend(name) for name in ("full", "free")},
                               tmp_path, member_queue_wait_s=0.5)
    response = pool.generate([{"role": "user", "content": "Synthetic request"}], role="worker")
    assert waits == [("full", 0.5), ("free", 0.5)]
    assert [route for route, _ in sent] == ["free"]
    assert sent[0][1] > 200  # Queue cap did not replace the HTTP response timeout.
    assert response.metadata["endpoint_pool_failovers"] == 1


@pytest.mark.parametrize("chosen", ["gpt", "deepseek"])
def test_real_application_routes_director_choice_and_enables_qwen_thinking(chosen, tmp_path, monkeypatch):
    config = load_formal("formal_training.toml", monkeypatch)
    config = replace(
        config, skillbank_enabled=False, pats=replace(config.pats, enabled=False),
        persist_runtime_updates=False, verifier="none",
        trace_path=tmp_path / "traces.jsonl", route_health_path=tmp_path / "health.json",
        retrieval=replace(config.retrieval, enabled=False, nq_evidence_mode=None,
                          hotpotqa_search_enabled=False),
        **{name: replace(getattr(config, name), enabled=False)
           for name in ("aime_actions", "alfworld", "swe")},
    )
    director = MockBackend([
        '{"action":"add_agent"}',
        json.dumps({"action": "set_prompt", "target": "agent_1", "role": "Shopper",
                    "objective": "Inspect the requested item", "scope": "Public shopping pages",
                    "expected_output": "Report findings"}),
        json.dumps({"action": "set_model", "target": "agent_1", "runtime_route": chosen}),
        '{"action":"set_output","target":"agent_1"}', '{"action":"finish"}',
    ])
    responses = [json.dumps({"action_call": {"name": "webshop_search", "arguments": {"query": "shirt"}}}),
                 json.dumps({"answer": "Inspected candidates", "evidence": ["Visible page"],
                             "unresolved_issues": ["Not purchased"]})]
    backends = {route: MockBackend(list(responses)) for route in config.runtime_pool()}
    for route, backend in backends.items():
        backend.config = app_module._runtime_gateway_config(config.runtime_pool()[route], {"worker": 0.6}, route_name=route)
    monkeypatch.setattr(app_module, "_create_runtime_backend", lambda runtime, *, route_name: backends[route_name])
    monkeypatch.setattr(app_module, "WebShopHTTPClient", lambda *a, **k: PublicClient())
    app = app_module.create_adaptive_application(config, director_backend=director,
                                               distiller_backend=MockBackend([]))
    try:
        assert app.runtime.executor.webshop_compatibility_profile == M02_PROFILE
        lifecycle = app.runtime.executor.tools["webshop_search"].lifecycle
        assert lifecycle.compatibility_profile == M02_PROFILE
        app.solve("Buy a shirt", task_id="synthetic", metadata={"dataset": "webshop", "goal_id": "goal-1"})
        assert director.calls and all(call["enable_thinking"] is True for call in director.calls)
        director_input = json.dumps([call["messages"] for call in director.calls])
        for physical_name in ("gpt_eco", "gpt_student", "grok45", "gemini2"):
            assert physical_name not in director_input
        for runtime in config.runtime_pool().values():
            assert runtime.base_url not in director_input
        used = {route for route, backend in backends.items() if backend.calls}
        expected = set(config.runtime_endpoint_pools["gpt"]) if chosen == "gpt" else {chosen}
        assert used and used <= expected
        assert set(app.solver.runtime_routes) == set(config.worker_routes_for("webshop"))
        for route in used:
            assert "merged_checklist_v1" in backends[route].calls[0]["messages"][0]["content"]
        app.configure_scoped_worker_routes(dataset="healthbench_professional")
        assert app.solver.runtime_routes == config.worker_routes_for("healthbench_professional")
    finally:
        app.close()
