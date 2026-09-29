"""NQ corpus-mode input and configuration checks."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from selfplay_graph_flowsteer import application
from selfplay_graph_flowsteer.application import (
    NQPolicyConfig,
    RetrievalConfig,
    create_adaptive_application,
    load_adaptive_config,
)
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.nq_corpus_tasks import prepare_pool, validate_corpus_task
from selfplay_graph_flowsteer.observability import TaskSpec


IDENTITY = "f8129c07a8c438a559bc4983f11dd1c8d48bc362a6035455cb8c20d6b28c9c9e"


def test_prepared_official_128_keeps_question_ids_and_hidden_references() -> None:
    source = Path("data/formal/eval/nq_open_official_test.jsonl")
    output = Path("data/formal/eval/nq_open_corpus_tool_128.jsonl")
    before = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    after = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(before) == len(after) == 128
    assert [row["id"] for row in before] == [row["id"] for row in after]
    for original, prepared in zip(before, after, strict=True):
        assert prepared["prompt"] == original["prompt"]
        for field in ("id", "source_id", "split", "reference", "target_answers"):
            assert prepared[field] == original[field]
        assert prepared["metadata"]["source_split"] == original["metadata"]["source_split"]
        assert prepared["metadata"]["original_question"] == original["prompt"]
        validate_corpus_task(prepared)
    manifest = json.loads((output.parent / (output.name + ".manifest.json")).read_text())
    assert manifest["nq_count"] == 128
    assert manifest["identity_sha256"] == IDENTITY


def test_prepare_pool_preserves_non_nq_rows_and_rejects_inline_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    rows = [
        {"id": "nq/train/1", "dataset": "nq_open", "prompt": "Who?", "reference": "Ada",
         "target_answers": ["Ada"], "split": "train",
         "metadata": {"dataset": "nq_open", "source_split": "train"}},
        {"id": "hotpot/eval/1", "dataset": "hotpotqa", "prompt": "What?", "reference": "42",
         "split": "test", "metadata": {"dataset": "hotpotqa", "source_split": "dev", "context": "line\u2028separator"}},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = prepare_pool(source, output, identity_sha256=IDENTITY)
    prepared = [json.loads(line) for line in output.read_text().split("\n") if line]
    assert manifest["rows"] == 2 and manifest["nq_count"] == 1
    assert prepared[0]["reference"] == "Ada"
    assert prepared[0]["split"] == "train"
    assert prepared[0]["metadata"]["source_split"] == "train"
    assert prepared[1] == rows[1]
    rows[0]["metadata"]["context_documents"] = [{"text": "Ada"}]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden evidence"):
        prepare_pool(source, output, identity_sha256=IDENTITY)


def test_config_and_action_adapter_require_local_corpus_search(monkeypatch) -> None:
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "1")
    config = load_adaptive_config("configs/nq_corpus_eval.toml")
    assert config.runtime.max_concurrency == 24
    assert config.retrieval.nq_evidence_mode == "corpus_tool"
    assert config.retrieval.nq_policy.expected_identity_sha256 == IDENTITY
    with pytest.raises(ValueError, match="local HTTP"):
        replace(config.retrieval, service_url="https://en.wikipedia.org/retrieve").validate()
    with pytest.raises(ValueError, match="nq_frozen_top_k=0"):
        replace(config.retrieval, nq_frozen_top_k=8).validate()
    with pytest.raises(ValueError, match="pinned"):
        replace(config.retrieval, nq_policy=replace(config.retrieval.nq_policy, expected_identity_sha256="")).validate()
    with pytest.raises(ValueError, match="search Action"):
        default_dataset_action_registry((), nq_evidence_mode="corpus_tool")
    registry = default_dataset_action_registry(("search",), nq_evidence_mode="corpus_tool")
    adapter = registry.resolve(TaskSpec("nq", "Who?", metadata={"dataset": "nq_open", "evidence_mode": "corpus_tool"}))
    assert adapter and adapter.adapter_id == "nq_open_corpus" and adapter.action_names == ("search",)
    frozen = registry.resolve(TaskSpec("nq", "Who?", metadata={"dataset": "nq_open", "evidence_mode": "provided_context_inline"}))
    assert frozen and frozen.action_names == ()


def test_corpus_service_startup_rejects_wrong_profile_or_identity(monkeypatch) -> None:
    retrieval = RetrievalConfig(
        enabled=True, service_url="http://127.0.0.1:18011/retrieve", top_k=8,
        nq_evidence_mode="corpus_tool", nq_policy=NQPolicyConfig(expected_identity_sha256=IDENTITY),
    )
    health = {
        "status": "ok", "schema": "spgfs-searchr1-e5-faiss-v1",
        "profile_id": "nq-dense8-v1", "asset_identity": {"corpus": "fixture"},
        "identity_sha256": IDENTITY,
    }

    class FakeOpener:
        def open(self, url, timeout):
            assert url == "http://127.0.0.1:18011/health"
            return io.BytesIO(json.dumps(health).encode())

    monkeypatch.setattr(application, "build_opener", lambda *args: FakeOpener())
    assert application._validate_nq_corpus_service(retrieval)["identity_sha256"] == IDENTITY
    health["profile_id"] = "nq-other-v1"
    with pytest.raises(ValueError, match="identity/profile mismatch"):
        application._validate_nq_corpus_service(retrieval)
    health["profile_id"] = "nq-dense8-v1"
    health["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity/profile mismatch"):
        application._validate_nq_corpus_service(retrieval)


def test_solver_rejects_wrong_nq_mode_before_any_worker_call(monkeypatch) -> None:
    monkeypatch.setenv("SPGFS_ALLOWED_PHYSICAL_GPUS", "1")
    app = create_adaptive_application(load_adaptive_config("configs/nq_corpus_eval.toml"), mock=True)
    try:
        with pytest.raises(ValueError, match="conflicts with configured"):
            app.solve(
                "Who wrote it?", task_id="nq-1", reference="Ada",
                metadata={"dataset": "nq_open", "evidence_mode": "provided_context_inline"},
            )
        with pytest.raises(ValueError, match="forbidden evidence"):
            app.solve(
                "Who wrote it?", task_id="nq-1", reference="Ada",
                metadata={"dataset": "nq_open", "evidence_mode": "corpus_tool",
                          "original_question": "Who wrote it?",
                          "context_documents": [{"text": "Ada wrote it"}]},
            )
    finally:
        app.close()
