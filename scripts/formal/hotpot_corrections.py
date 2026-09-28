"""Build the reviewed local HotpotQA version from the immutable FlowSteer snapshot.

Run directly with --check to verify, or without it to activate only HotpotQA.
No model calls, historical rescoring, or training-pool mutation are performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SPEC = "data/formal/private/hotpotqa_corrections_v1.json"
VERSION = "flowsteer-hotpotqa-corrected-v1"
CANONICAL = "data/formal/eval/hotpotqa_flowsteer_corrected_v1_128.jsonl"
ACTIVE = "data/formal/eval/hotpotqa_official_test.jsonl"
MANIFEST = "data/formal/eval/static_eval_split_manifest.json"
REPORT = "experiment_versions/reports/hotpot-corrections-20260928/build.json"
QUESTION_SEPARATOR = "\n\nQuestion: "


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build(root: Path) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    spec_bytes = (root / SPEC).read_bytes()
    spec = json.loads(spec_bytes)
    if spec["schema"] != "spgfs-hotpot-corrections-v1" or spec["dataset_version"] != VERSION:
        raise ValueError("unexpected correction schema/version")
    original_bytes = (root / spec["original_path"]).read_bytes()
    if sha256(original_bytes) != spec["original_sha256"]:
        raise ValueError("original HotpotQA SHA256 mismatch; refusing to patch")
    if sha256((root / spec["source_raw_path"]).read_bytes()) != spec["source_raw_sha256"]:
        raise ValueError("FlowSteer raw source SHA256 mismatch")
    original = [json.loads(line) for line in original_bytes.splitlines() if line.strip()]
    if len(original) != 128 or spec["expected_rows"] != 128:
        raise ValueError("expected 128 original rows")
    ids = {row["source_id"] for row in original}
    if len(ids) != len(original):
        raise ValueError("duplicate source IDs")
    patches = {patch["source_id"]: patch for patch in spec["patches"]}
    if len(patches) != len(spec["patches"]) or len(patches) != 14 or spec["expected_corrections"] != 14:
        raise ValueError("expected exactly 14 unique corrections")
    if not patches.keys() <= ids:
        raise ValueError("unknown correction source ID")
    rows = copy.deepcopy(original)
    counts = {"question_changes": 0, "reference_changes": 0, "target_answers_changes": 0}
    for row in rows:
        source_id = row["source_id"]
        # Separate task IDs prevent cached original-version results being reused.
        # Keep source_id unchanged for provenance and train/eval overlap checks.
        row["id"] = source_id.replace("hotpotqa/", "hotpotqa/flowsteer-corrected-v1/", 1)
        row["metadata"]["dataset_version"] = VERSION
        if source_id not in patches:
            continue
        patch = patches[source_id]
        context, separator, question = row["prompt"].rpartition(QUESTION_SEPARATOR)
        before = {"question": question, "reference": row["reference"], "target_answers": row["target_answers"]}
        if not separator or before != patch["before"]:
            raise ValueError(f"correction precondition mismatch: {source_id}")
        after = patch["after"]
        if set(after) != set(before) or before == after:
            raise ValueError(f"invalid correction fields: {source_id}")
        if not isinstance(after["question"], str) or not after["question"].strip() or "\n" in after["question"]:
            raise ValueError(f"invalid corrected question: {source_id}")
        answers = after["target_answers"]
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) or not answer.strip() for answer in answers)
                or len(set(answers)) != len(answers) or after["reference"] != answers[0]):
            raise ValueError(f"invalid corrected references: {source_id}")
        row["prompt"] = context + separator + after["question"]
        row["reference"] = after["reference"]
        row["target_answers"] = answers
        for key in before:
            counts[f"{key}_changes"] += before[key] != after[key]
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    entry = {
        "path": ACTIVE, "rows": len(rows), "sha256": sha256(data),
        "unique_source_rows": len(ids), "selection": "flowsteer_public_eval_128_locally_corrected",
        "dataset_version": VERSION, "canonical_path": CANONICAL,
        "source_url": "https://huggingface.co/datasets/beita6969/FlowSteer-Dataset/resolve/main/eval/hotpotqa.jsonl",
        "source_sha256": spec["source_raw_sha256"],
        "original_path": spec["original_path"], "original_sha256": spec["original_sha256"],
        "corrections_path": SPEC, "corrections_sha256": sha256(spec_bytes),
        "corrected_rows": len(patches), **counts,
        "upstream_split": "train", "official_hotpotqa_erratum": False,
    }
    return rows, data, entry


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root: Path, *, check: bool = False) -> dict[str, Any]:
    _, data, entry = build(root)
    manifest = json.loads((root / MANIFEST).read_bytes())
    manifest["datasets"]["hotpotqa"] = entry
    report = {
        "schema": "spgfs-hotpot-correction-build-v1", **entry,
        "content_unchanged_rows": 128 - entry["corrected_rows"],
        "passages_unchanged_rows": 128, "original_source_ids_preserved": True,
        "all_evaluation_ids_versioned": True, "model_api_calls": 0,
        "training_data_modified": False, "historical_results_rescored": False,
    }
    outputs = {CANONICAL: data, ACTIVE: data, MANIFEST: json_bytes(manifest), REPORT: json_bytes(report)}
    # Validate every input and all existing data artifacts before writing anything.
    for relative in (CANONICAL, ACTIVE):
        path = root / relative
        allowed = {entry["sha256"]}
        if relative == ACTIVE:
            allowed.add(entry["original_sha256"])
        if path.exists() and sha256(path.read_bytes()) not in allowed:
            raise ValueError(f"unrecognized existing dataset; refusing to overwrite {relative}")
    if check:
        mismatches = [relative for relative, content in outputs.items()
                      if not (root / relative).exists() or (root / relative).read_bytes() != content]
        if mismatches:
            raise ValueError(f"corrected artifacts missing or stale: {mismatches}")
    else:
        for relative, content in outputs.items():
            atomic_write(root / relative, content)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true", help="Verify without changing any files.")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, check=args.check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
