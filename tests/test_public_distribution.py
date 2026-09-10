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
    assert config.graph_embedding_model_path == "intfloat/e5-base-v2"
    assert config.support_runtime in config.runtime_pool()
    assert SelfPlayRunConfig().proposer_baseline_mode == "ema"


def test_removed_subsystem_is_not_importable_or_exposed():
    assert importlib.util.find_spec("selfplay_graph_flowsteer.skills") is None
    assert "inspect-skillbank" not in build_parser().format_help()
    config = load_adaptive_config(ROOT / "configs/mock.toml")
    assert not any("skill" in field for field in vars(config))


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
        assert "skills_used" not in serialized
    finally:
        app.close()
