from __future__ import annotations

import copy
import json
import shutil
from itertools import permutations
from pathlib import Path

import pytest

from scripts.formal import hotpot_corrections as corrections
from scripts.formal import prepare_static_eval
from selfplay_graph_flowsteer.learning import load_fixed_jsonl
from selfplay_graph_flowsteer.qa_metrics import qa_official_metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_root(tmp_path):
    spec = json.loads((ROOT / corrections.SPEC).read_bytes())
    for relative in (corrections.SPEC, spec["original_path"], spec["source_raw_path"],
                     corrections.ACTIVE, corrections.MANIFEST):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    # Exercise initial activation from the original, regardless of checkout state.
    shutil.copyfile(tmp_path / spec["original_path"], tmp_path / corrections.ACTIVE)
    return tmp_path


def test_only_reviewed_content_changes_and_provenance_is_preserved():
    spec = json.loads((ROOT / corrections.SPEC).read_bytes())
    original = [json.loads(line) for line in (ROOT / spec["original_path"]).read_bytes().splitlines()]
    corrected, _, entry = corrections.build(ROOT)
    expected = {patch["source_id"] for patch in spec["patches"]}
    queue = json.loads((ROOT / "experiment_versions/reports/hotpot-label-audit-20260928/review_queue.json").read_bytes())
    assert expected == {f"hotpotqa/{case['id']}" for case in queue["cases"] if case["group"] == "data_quality"}
    changes = set()
    for old, new in zip(original, corrected, strict=True):
        assert new["source_id"] == old["source_id"]
        assert new["id"] != old["id"]
        assert new["metadata"] == {**old["metadata"], "dataset_version": corrections.VERSION}
        assert new["prompt"].rpartition(corrections.QUESTION_SEPARATOR)[0] == old["prompt"].rpartition(corrections.QUESTION_SEPARATOR)[0]
        restored = copy.deepcopy(new)
        restored["id"] = old["id"]
        restored["metadata"] = old["metadata"]
        if restored != old:
            changes.add(new["source_id"])
        for key in ("prompt", "reference", "target_answers"):
            restored[key] = old[key]
        assert restored == old
    assert changes == expected
    assert len({row["id"] for row in corrected}) == 128
    assert not {row["id"] for row in corrected} & {row["id"] for row in original}
    assert (entry["question_changes"], entry["reference_changes"], entry["target_answers_changes"]) == (13, 3, 3)
    # Support evidence stays in the private patch file, never in solver metadata.
    by_id = {row["source_id"]: row for row in original}
    for patch in spec["patches"]:
        prompt = " ".join(by_id[patch["source_id"]]["prompt"].split())
        for evidence in patch["evidence"]:
            assert " ".join(evidence["text"].split()) in prompt


def test_activation_is_idempotent_and_preserves_other_datasets(frozen_root, monkeypatch):
    previous_manifest = json.loads((frozen_root / corrections.MANIFEST).read_bytes())
    spec = json.loads((frozen_root / corrections.SPEC).read_bytes())
    originals = {path: (frozen_root / path).read_bytes() for path in (spec["original_path"], spec["source_raw_path"])}
    report = corrections.prepare(frozen_root)
    assert corrections.prepare(frozen_root, check=True) == report
    assert corrections.prepare(frozen_root) == report
    manifest = json.loads((frozen_root / corrections.MANIFEST).read_bytes())
    for name in previous_manifest["datasets"]:
        if name != "hotpotqa":
            assert manifest["datasets"][name] == previous_manifest["datasets"][name]
    for path, data in originals.items():
        assert (frozen_root / path).read_bytes() == data
    assert (frozen_root / corrections.ACTIVE).read_bytes() == (frozen_root / corrections.CANONICAL).read_bytes()
    monkeypatch.setattr(prepare_static_eval, "ROOT", frozen_root)
    assert prepare_static_eval.hotpotqa() == corrections.build(frozen_root)[0]


@pytest.mark.parametrize("fault", ["source_drift", "raw_drift", "duplicate", "missing_id", "before", "empty_answers", "unknown_active"])
def test_invalid_inputs_fail_before_any_output_write(frozen_root, fault):
    spec_path = frozen_root / corrections.SPEC
    spec = json.loads(spec_path.read_bytes())
    patch = spec["patches"][0]
    if fault in {"source_drift", "raw_drift"}:
        path = frozen_root / spec["original_path" if fault == "source_drift" else "source_raw_path"]
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "duplicate":
        spec["patches"][-1] = patch
    elif fault == "missing_id":
        patch["source_id"] = "hotpotqa/unknown"
    elif fault == "before":
        patch["before"]["reference"] = "unreviewed"
    elif fault == "empty_answers":
        patch["after"]["target_answers"] = []
    elif fault == "unknown_active":
        path = frozen_root / corrections.ACTIVE
        path.write_bytes(path.read_bytes() + b"\n")
    spec_path.write_bytes(corrections.json_bytes(spec))
    before = {path: path.read_bytes() for path in frozen_root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        corrections.prepare(frozen_root)
    assert {path: path.read_bytes() for path in frozen_root.rglob("*") if path.is_file()} == before


def test_real_loader_and_strict_em_accept_full_occupation_set_only(frozen_root):
    corrections.prepare(frozen_root)
    examples = load_fixed_jsonl(frozen_root / corrections.ACTIVE)
    assert len(examples) == 128
    example = next(item for item in examples if item.example_id.endswith("5a7e1f3f5542997cc2c47524"))
    assert len(example.reference) == 12
    for order in permutations(("pianist", "composer", "conductor")):
        for answer in (", ".join(order), f"{order[0]}, {order[1]} and {order[2]}"):
            assert qa_official_metrics("hotpotqa", answer, example.reference)["answer_em"] == 1
    for incomplete in ("composer and conductor", "pianist", "pianist, composer, conductor, organist"):
        assert qa_official_metrics("hotpotqa", incomplete, example.reference)["answer_em"] == 0


def test_check_detects_stale_manifest(frozen_root):
    corrections.prepare(frozen_root)
    path = frozen_root / corrections.MANIFEST
    manifest = json.loads(path.read_bytes())
    manifest["datasets"]["hotpotqa"]["dataset_version"] = "original"
    path.write_bytes(corrections.json_bytes(manifest))
    with pytest.raises(ValueError, match="stale"):
        corrections.prepare(frozen_root, check=True)


def test_static_eval_rebuild_keeps_corrected_version(frozen_root, monkeypatch):
    monkeypatch.setattr(prepare_static_eval, "ROOT", frozen_root)
    monkeypatch.setattr(prepare_static_eval, "OUT", frozen_root / "data/formal/eval")
    # Other datasets have their own preparation tests and external source assets.
    for name in ("aime", "nq_open", "alfworld"):
        monkeypatch.setattr(prepare_static_eval, name, lambda: [])
    prepare_static_eval.main()
    _, expected, entry = corrections.build(frozen_root)
    assert (frozen_root / corrections.ACTIVE).read_bytes() == expected
    assert (frozen_root / corrections.CANONICAL).read_bytes() == expected
    manifest = json.loads((frozen_root / corrections.MANIFEST).read_bytes())
    assert manifest["datasets"]["hotpotqa"] == entry
