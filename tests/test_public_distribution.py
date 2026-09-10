"""Distribution boundaries and the replacement graph/support integration."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from selfplay_graph_flowsteer.application import (
    create_adaptive_application,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.cli import build_parser
from selfplay_graph_flowsteer.selfplay_runtime import SelfPlayRunConfig


ROOT = Path(__file__).resolve().parents[1]


def test_public_config_keeps_outputs_inside_checkout():
    config = load_adaptive_config(ROOT / "configs/mock.toml")
    assert config.trace_path.is_relative_to(ROOT)
    assert config.proposer_model.checkpoint_path.is_relative_to(ROOT)
    assert config.solver_model.checkpoint_path.is_relative_to(ROOT)
    assert config.healthbench_judge_audit_path.is_relative_to(ROOT)
    assert config.skillbank_embedding_model_path == "intfloat/e5-base-v2"
    assert config.skill_distiller_runtime in config.runtime_pool()
    assert SelfPlayRunConfig().proposer_baseline_mode == "ema"


def test_skill_subsystem_is_packaged_and_opt_in():
    assert importlib.util.find_spec("selfplay_graph_flowsteer.skills") is not None
    assert "inspect-skillbank" in build_parser().format_help()
    config = load_adaptive_config(ROOT / "configs/mock.toml")
    assert not config.skillbank_enabled
    seeds = json.loads((ROOT / "src/selfplay_graph_flowsteer/director_seed_v2.json").read_text())
    assert len(seeds) == 8


def test_mock_application_runs_without_skill_context(tmp_path):
    config = replace(
        load_adaptive_config(ROOT / "configs/mock.toml"),
        verifier="none",
        trace_path=tmp_path / "trace.jsonl",
        persist_runtime_updates=False,
        route_health_path=tmp_path / "health.json",
    )
    app = create_adaptive_application(config, mock=True)
    try:
        result = app.solve("Demo task", task_id="public-demo")
        assert result.to_dict()["finished"]
        serialized = json.dumps(result.to_dict()).casefold()
        assert "skillbank" not in serialized
        assert not result.skills_used
    finally:
        app.close()


def test_skill_template_keeps_high_reasoning_separate_from_worker(monkeypatch):
    from selfplay_graph_flowsteer.application import _runtime_gateway_config

    monkeypatch.setenv("DEEPSEEK_SKILL_API_KEY", "offline-placeholder")
    config = load_adaptive_config(ROOT / "configs/skillbank.example.toml")
    assert config.skillbank_enabled
    assert config.skillbank_mode == "director_skill_v2"
    assert config.skillbank_activation_policy == "checked"
    assert config.skill_distiller_runtime not in config.worker_runtime_routes
    runtime = config.runtime_pool()[config.skill_distiller_runtime]
    role = _runtime_gateway_config(
        runtime, {"skill-distiller": 0}, route_name="deepseek_skill"
    ).roles["skill-distiller"]
    assert role.enable_thinking and role.reasoning_effort == "high"
    assert runtime.max_concurrency == 20
    assert role.max_tokens == 8192
    assert config.skillbank_path.is_relative_to(ROOT)
