"""The evaluated SWE submission contract must remain isolated in mixed training."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib

import pytest

from selfplay_graph_flowsteer.application import AdaptiveApplicationConfig, load_adaptive_config
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.endpoint_pool import EndpointPoolBackend
from selfplay_graph_flowsteer.execution_contract import (
    bind_rollout_contract, execution_semantics, require_same_semantics, validate_training_contract,
)
from selfplay_graph_flowsteer.llm import MockBackend, request_dataset
from selfplay_graph_flowsteer.pats import PatsConfig, PatsController, resolve_scope
from selfplay_graph_flowsteer.pats_refiner import review_system_prompt
from selfplay_graph_flowsteer.pats_semantics import contract_hash, semantic_approvals
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.rollouts import TrainingBatch, TrainingSample
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer
from selfplay_graph_flowsteer.skill_evolution_v2 import freeze_collection, load_bank, store_for_config

from .helpers import NumericRecordingExecutor
from .test_endpoint_pool import Backend, _transient_failure
from .test_pats_semantic_freeze import Checker, card, select


@pytest.mark.parametrize("dataset", ["swe_bench", "aime", "nq_open", "hotpotqa", "webshop", "alfworld", "healthbench_professional"])
def test_canvas_and_director_use_only_swe_override(tmp_path, dataset):
    base = CanvasConfig(submission_protocol_by_dataset={"swe_bench": "unified_task_result_v1"},
                        submission_journal_dir=str(tmp_path))
    canvas = GraphCanvas(task="public task", dataset=dataset, config=base,
                         runtime=MultiAgentRuntime(NumericRecordingExecutor()))
    director = GraphDirector(canvas=canvas, backend=MockBackend([]), prompt_variant="v2.2")
    assert canvas.unified is (dataset == "swe_bench")
    assert director.prompt_variant == ("v3" if dataset == "swe_bench" else "v2.2")
    assert base.submission_protocol == "legacy"
    assert base.submission_protocol_by_dataset == {"swe_bench": "unified_task_result_v1"}


def test_dataset_pool_fails_over_without_touching_eco_or_other_dataset_order(tmp_path):
    calls = []

    class RecordingBackend(Backend):
        def generate(self, messages, **kwargs):
            calls.append(self.config.route_name)
            if self.config.route_name == "student":
                raise _transient_failure()
            return super().generate(messages, **kwargs)

    pool = EndpointPoolBackend("gpt", {name: RecordingBackend(name) for name in ["gpt", "eco", "student"]},
                               tmp_path, members_by_dataset={"swe_bench": ("student", "gpt")})
    with request_dataset("swe-bench"):
        response = pool.generate([{"content": "same public request"}], role="worker")
    assert calls == ["student", "gpt"]
    assert response.metadata["endpoint_pool_failovers"] == 1
    calls.clear()
    with request_dataset("webshop"):
        pool.generate([{"content": "shop"}], role="worker")
        pool.generate([{"content": "shop"}], role="worker")
    assert calls == ["gpt", "eco"]


def test_mixed_training_contract_binds_swe_prompt_and_rejects_old_contract():
    base = AdaptiveApplicationConfig(director_prompt_variant="v2.2")
    mixed = replace(base, canvas=replace(base.canvas,
                    submission_protocol_by_dataset={"swe_bench": "unified_task_result_v1"}))
    observed = mixed.model_manifest()["execution_semantics"]
    assert observed["director_prompt_variant"] == "v2.2"
    assert observed["director_prompt_variant_by_dataset"] == {"swe_bench": "v3"}
    assert observed["submission_contract_version_by_dataset"] == {"swe_bench": "unified_submission_v1"}
    unified = execution_semantics("v3")
    legacy = execution_semantics("v2.2")
    for selected in (False, True):
        key = f"swe_bench:{selected}"
        assert observed["worker_and_recovery_sha256"][key] == unified["worker_and_recovery_sha256"][key]
        key = f"webshop:{selected}"
        assert observed["worker_and_recovery_sha256"][key] == legacy["worker_and_recovery_sha256"][key]
    assert observed == execution_semantics("v2.2", admission_config=observed["submission_admission_config"])
    with pytest.raises(ValueError, match="execution semantics changed"):
        require_same_semantics(base.model_manifest()["execution_semantics"], observed)


def test_scoped_pats_freeze_checks_each_real_protocol_and_keeps_selection_hash(tmp_path):
    config = AdaptiveApplicationConfig(
        director_prompt_variant="v2.2",
        canvas=CanvasConfig(submission_protocol_by_dataset={"swe_bench": "unified_task_result_v1"}),
        pats=PatsConfig(enabled=True), skillbank_mode="director_skill_v2",
        skillbank_training=True, skillbank_path=tmp_path / "bank.json",
        skill_cases_path=tmp_path / "cases.json", skillbank_embedding_model_path=None,
    )
    store = store_for_config(config)
    PatsController(store, config.pats, len)
    records = [card("learned")]
    scopes = {resolve_scope("qa", {"dataset": dataset}): {
        "cards": records, "ema": 0.1, "mode": "EXPAND", "policy_snapshot": "behavior",
    } for dataset in ("swe_bench", "nq_open")}
    state = {"config": asdict(config.pats), "run": str(tmp_path), "step": 1, "scopes": scopes}
    with store.connect() as db:
        db.execute("INSERT INTO pats_state VALUES(1,?)", (json.dumps(state),))
    checker = Checker()
    frozen = freeze_collection(config, tmp_path / "cycle2", step=2,
                               semantic_backend=checker, tokenizer=ByteTokenizer())
    assert checker.calls == 2
    for dataset, variant in [("swe_bench", "v3"), ("nq_open", "v2.2")]:
        scope = resolve_scope("qa", {"dataset": dataset})
        assert list(semantic_approvals(store, scope, records, variant).values()) == [True]
        selected, _, manifest = select(load_bank(frozen), dataset)
        assert len(selected) == 1
        assert manifest["semantic_contract_sha256"] == contract_hash(variant)
    assert not semantic_approvals(store, resolve_scope("qa", {"dataset": "swe_bench"}), records, "v2.2")
    swe_prompt = review_system_prompt(config.pats.for_scope(resolve_scope("qa", {"dataset": "swe_bench"})), "EXPAND")
    assert '"result_scope":"task_result"' in swe_prompt
    assert "There is no SET_OUTPUT action" in swe_prompt


def test_formal_swe_profile_matches_evaluated_settings_without_changing_other_routes():
    root = Path(__file__).resolve().parents[1]
    config = load_adaptive_config(root / "configs/formal_training.toml", validate=False)
    assert config.worker_routes_for("swe_bench") == ("gpt",)
    assert config.worker_routes_for("webshop") == ("gpt", "grok", "gemini", "deepseek", "minimax")
    assert config.dataset_endpoint_pools["swe_bench"]["gpt"] == ("gpt_student", "gpt")
    assert config.runtime_endpoint_pools["gpt"] == ("gpt", "gpt_eco", "gpt_student")
    assert config.additional_runtimes["gpt"].max_concurrency_by_dataset["swe_bench"] == 5
    assert config.additional_runtimes["gpt_student"].max_concurrency_by_dataset["swe_bench"] == 10
    assert config.additional_runtimes["gpt"].max_concurrency == 10
    assert config.additional_runtimes["gpt_student"].max_concurrency == 5
    assert config.canvas.worker_usage_policy("swe_bench")["start_threshold"] == 350000
    assert config.canvas.for_dataset("swe_bench").submission_protocol == "unified_task_result_v1"
    assert config.swe.cvm_auto_start and config.swe.cvm_auto_stop


def test_mixed_swe_hotpot_batch_binds_and_rejects_wrong_dataset_protocol():
    config = AdaptiveApplicationConfig(director_prompt_variant="v2.2", canvas=CanvasConfig(
        submission_protocol_by_dataset={"swe_bench": "unified_task_result_v1"}))
    semantics = config.model_manifest()["execution_semantics"]
    rows = [{"dataset": dataset, "model_roles": {"execution_semantics": semantics},
             "submission_contract_version": version, "training_eligible": False}
            for dataset, version in [("swe_bench", "unified_submission_v1"),
                                     ("hotpotqa", semantics["submission_contract_version"])]]
    rollouts = [SimpleNamespace(trajectory=SimpleNamespace(metadata=row)) for row in rows]
    samples = tuple(TrainingSample(str(i), str(i), (1, 2), (0, 1), 1, 0, metadata=row)
                    for i, row in enumerate(rows))
    batches = bind_rollout_contract((TrainingBatch("proposer", ()),
                                    TrainingBatch("solver", samples)), rollouts)
    validate_training_contract(*batches, expected=semantics)
    # Protocol correctness is independent of whether a sample has an admitted reward.
    rows[0]["submission_contract_version"] = semantics["submission_contract_version"]
    with pytest.raises(ValueError, match="does not match dataset: swe_bench"):
        bind_rollout_contract(batches, rollouts)
    with pytest.raises(ValueError, match="does not match dataset: swe_bench"):
        validate_training_contract(*batches, expected=semantics)
    rows[0]["submission_contract_version"] = "unified_submission_v1"
    rows[1]["submission_contract_version"] = "unified_submission_v1"
    with pytest.raises(ValueError, match="does not match dataset: hotpotqa"):
        bind_rollout_contract(batches, rollouts)


def test_swe_pool_preflight_rejects_eco_only_and_keeps_scoped_members(tmp_path, monkeypatch):
    from selfplay_graph_flowsteer.cli import _apply_fresh_route_report
    from .test_webshop_formal_promotion import load_formal

    config = load_formal("formal_training.toml", monkeypatch)
    # Qualify the independent judge so this test reaches the SWE Worker gate.
    config = replace(config, healthbench_judge_runtime_route="gpt_judge")
    report = tmp_path / "routes.json"
    requested = list(config.runtime_pool())
    args = argparse.Namespace(route_report=report, mock=False, max_route_report_age_s=1800,
                              minimum_selected_routes=1, route_subset="")
    usable = [route for route in requested if route not in {"gpt", "gpt_student"}]
    report.write_text(json.dumps({"routes_requested": requested, "usable_routes": usable}))
    with pytest.raises(ValueError, match="freshly qualified member: swe_bench/gpt"):
        _apply_fresh_route_report(config, args)
    report.write_text(json.dumps({"routes_requested": requested,
                                  "usable_routes": usable + ["gpt_student"]}))
    selected, _ = _apply_fresh_route_report(config, args)
    assert selected.dataset_endpoint_pools == config.dataset_endpoint_pools


def test_formal_hotpot_keeps_answer_contract_and_data_when_protocol_changes():
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "experiment_versions/promotions/hotpot-8672-20260928/validation.json").read_text())
    config = load_adaptive_config(root / "configs/formal_training.toml", validate=False)
    current = config.model_manifest()["execution_semantics"]
    legacy = execution_semantics("v2.2")
    for key, expected in provenance["promoted_worker_and_recovery_sha256"].items():
        assert legacy["worker_and_recovery_sha256"][key] == expected
        assert current["worker_and_recovery_sha256"][key] != expected  # Explicit V3 responsibility.
    source = "src/selfplay_graph_flowsteer/hotpot_answer_contract.py"
    assert hashlib.sha256((root / source).read_bytes()).hexdigest() == provenance["files"][source]["after_sha256"]
    for name in ("hotpotqa_official_test.jsonl", "hotpotqa_flowsteer_corrected_v1_128.jsonl"):
        assert hashlib.sha256((root / "data/formal/eval" / name).read_bytes()).hexdigest() == provenance["dataset_sha256"]
    assert config.canvas.for_dataset("hotpotqa").submission_protocol == "unified_task_result_v1"
    assert config.pats.variant_for_scope(resolve_scope("qa", {"dataset": "hotpotqa"})) == "v3"
