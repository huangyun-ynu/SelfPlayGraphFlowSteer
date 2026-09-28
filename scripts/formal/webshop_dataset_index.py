#!/usr/bin/env python3
"""Validate/repair the pinned environment's public instruction index, without scoring.

The official server shuffles the complete goal list once with seed 233. Raw
release row indices therefore cannot be sent as environment session indices.
Only instruction_text is retained from the release; private target fields are
neither returned nor written to repaired datasets or validation records.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import random
import re


def normalized(value):
    return " ".join(str(value).split())


def public_goal_order(path: Path):
    instructions = [str(json.loads(line)["instruction_text"]) for line in path.open() if line.strip()]
    order = list(range(len(instructions)))
    random.Random(233).shuffle(order)
    return instructions, order


def goal_index(record):
    identifier = str(record.get("metadata", {}).get("goal_id", record.get("id", "")))
    match = re.fullmatch(r"(?:webshop/)?goal-(\d+)", identifier)
    if not match:
        raise ValueError("invalid public goal identifier")
    return int(match[1])


def validate_records(records, instructions, order):
    mismatches = []
    for record in records:
        index = goal_index(record)
        expected = instructions[order[index]] if 0 <= index < len(order) else ""
        if not expected or normalized(expected) not in normalized(record["prompt"]):
            mismatches.append({"id": record["id"], "goal_index": index,
                               "public_request": record["prompt"], "environment_instruction": expected})
    return mismatches


def repair_raw_records(records, instructions, order):
    inverse = {raw: actual for actual, raw in enumerate(order)}
    repaired, quarantine = [], []
    for original in records:
        raw = goal_index(original)
        if not 0 <= raw < len(instructions) or normalized(instructions[raw]) != normalized(original["prompt"]):
            raise ValueError(f"cannot establish exact raw-index provenance for {original['id']}")
        index = inverse[raw]
        row = copy.deepcopy(original)
        identifier = f"webshop/goal-{index:05d}"
        for key in ("id", "source_id", "ads_sample_id"):
            if key in row:
                row[key] = identifier
        metadata = row.setdefault("metadata", {})
        metadata.update(goal_id=identifier, goal_index=index, task_id=identifier)
        metadata["webshop_index_repair"] = {"original_id": original["id"], "raw_release_index": raw,
                                          "environment_index": index, "shuffle_seed": 233}
        split = "test" if index < 500 else "eval" if index < 1500 else "train"
        metadata["source_split"] = metadata["split"] = row["split"] = split
        if isinstance(metadata.get("source_lineage"), dict):
            metadata["source_lineage"]["source_split"] = split
        if split == "train":
            repaired.append(row)
        else:
            quarantine.append(row)
    groups = collections.Counter(row.get("cluster_id") for row in repaired)
    ranks = collections.Counter()
    for row in repaired:
        key = row.get("cluster_id")
        if key is not None:
            row["cluster_size"] = groups[key]
            row["rank_in_cluster"] = ranks[key]
            ranks[key] += 1
    assert not validate_records(repaired + quarantine, instructions, order)
    return repaired, quarantine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new repair directory, never overwrite")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-training-split", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.open() if line.strip()]
    if args.validate_only:
        records = [r for r in records if r.get("dataset", r.get("metadata", {}).get("dataset")) == "webshop"]
        if not records:
            raise ValueError("no WebShop records in the declared task pool")
    instructions, order = public_goal_order(args.goals)
    failures = validate_records(records, instructions, order)
    if args.validate_only:
        if failures:
            raise ValueError(f"{len(failures)}/{len(records)} public WebShop tasks mismatch environment goal indices")
        if args.require_training_split and any(goal_index(r) < 1500 for r in records):
            raise ValueError("official eval/test WebShop goals must not enter the training task pool")
        print(json.dumps({"validated_webshop_rows": len(records), "public_goal_mismatches": 0,
                          "training_split_checked": args.require_training_split,
                          "task_pool_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}))
        return
    if args.output is None:
        parser.error("--output is required when repairing records")
    repaired, quarantine = repair_raw_records(records, instructions, order)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("webshop.jsonl", repaired), ("quarantine.jsonl", quarantine)):
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {"source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "goals_sha256": hashlib.sha256(args.goals.read_bytes()).hexdigest(),
                "source_rows": len(records), "source_mismatches": len(failures),
                "repaired_training_rows": len(repaired), "quarantined_eval_test_rows": len(quarantine),
                "repaired_sha256": hashlib.sha256((args.output / "webshop.jsonl").read_bytes()).hexdigest(),
                "mapping": "exact raw row provenance -> Random(233) permutation inverse",
                "scoring_changed": False, "prompts_changed": False,
                "cluster_note": "retained embeddings/difficulties; sizes and ranks updated after split quarantine"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "mismatches.public.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("source_rows", "source_mismatches", "repaired_training_rows", "quarantined_eval_test_rows")}))


if __name__ == "__main__":
    main()
