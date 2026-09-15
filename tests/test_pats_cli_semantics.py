from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import application, cli, skill_evolution_v2
from selfplay_graph_flowsteer.pats import PatsConfig


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return replace(
        application.load_adaptive_config(root / "configs/pats.example.toml"),
        skillbank_training=True,
    )


@pytest.mark.parametrize("fail_freeze", [False, True])
def test_first_real_freeze_uses_director_tokenizer_and_closes_checker(
    config, tmp_path, monkeypatch, fail_freeze
):
    config = replace(
        config,
        proposer_model=replace(config.proposer_model, base_model_path="proposer-tokenizer"),
        solver_model=replace(config.solver_model, base_model_path="director-tokenizer"),
    )
    calls = []
    tokenizer = object()
    backend = SimpleNamespace(close=lambda: calls.append("closed"))

    def make_tokenizer(path):
        assert path == "director-tokenizer"
        calls.append("tokenizer")
        return tokenizer

    def make_backend(runtime, *, route_name):
        assert route_name == config.skill_distiller_runtime
        assert runtime == config.runtime_pool()[route_name]
        calls.append("backend")
        return backend

    def freeze(actual, cycle_dir, **kwargs):
        assert actual is config and cycle_dir == tmp_path
        assert kwargs == {"step": 3, "semantic_backend": backend, "tokenizer": tokenizer}
        calls.append("freeze")
        if fail_freeze:
            raise ValueError("invalid frozen state")
        return "frozen-config"

    monkeypatch.setattr(cli, "HuggingFaceTokenizer", make_tokenizer)
    monkeypatch.setattr(application, "_create_runtime_backend", make_backend)
    monkeypatch.setattr(skill_evolution_v2, "freeze_collection", freeze)
    if fail_freeze:
        with pytest.raises(ValueError, match="invalid frozen state"):
            cli._freeze_training_skill_context(config, tmp_path, step=3, mock=False)
    else:
        assert (
            cli._freeze_training_skill_context(config, tmp_path, step=3, mock=False)
            == "frozen-config"
        )
    assert calls == ["tokenizer", "backend", "freeze", "closed"]


@pytest.mark.parametrize("mode", ["mock", "disabled", "evaluation", "frozen", "missing_snapshot"])
def test_nonreview_paths_never_construct_a_live_checker(config, tmp_path, monkeypatch, mode):
    if mode == "disabled":
        config = replace(config, pats=PatsConfig())
    elif mode == "evaluation":
        config = replace(config, skillbank_training=False)
    elif mode == "frozen":
        (tmp_path / "director_skill_snapshot.v2.json").write_text("immutable")
    elif mode == "missing_snapshot":
        (tmp_path / "solver_rollouts.jsonl").write_text("existing raw data")

    def unexpected(*args, **kwargs):
        pytest.fail("This path must not construct a semantic inference service")

    def freeze(actual, cycle_dir, **kwargs):
        assert actual is config and cycle_dir == tmp_path
        assert kwargs == {"step": 3}
        return config

    monkeypatch.setattr(cli, "HuggingFaceTokenizer", unexpected)
    monkeypatch.setattr(application, "_create_runtime_backend", unexpected)
    monkeypatch.setattr(skill_evolution_v2, "freeze_collection", freeze)
    assert cli._freeze_training_skill_context(config, tmp_path, step=3, mock=mode == "mock") is config
