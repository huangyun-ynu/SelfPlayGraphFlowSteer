"""Pinned, verified SkillFlow small-catalogue releases (no model dependencies)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sqlite3

VERSION = "webshop-synthetic-1k-v2"
SUPPORTED_VERSIONS = {"webshop-synthetic-1k-v1", VERSION}
OFFICIAL_REVISION = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def reference_fingerprint(goal, product, price):
    return digest({"goal": goal, "product": product, "price": price})


def storage_order(runtime_goals):
    """Undo the disk adapter's Random(233) shuffle; retain SkillFlow goal IDs."""
    order = list(range(len(runtime_goals)))
    random.Random(233).shuffle(order)
    stored = [None] * len(order)
    for runtime_index, stored_index in enumerate(order):
        stored[stored_index] = runtime_goals[runtime_index]
    return stored


def public_binding(manifest):
    return {"version": manifest["version"], "inventory_sha256": digest(manifest["environment"])}


def validate_release(release, source_root=None):
    """Fail closed on changed assets, task/goal mismatch or scoring changes."""
    release = Path(release).resolve()
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest.get("version") not in SUPPORTED_VERSIONS or manifest.get("binding") != public_binding(manifest):
        raise ValueError("Invalid synthetic WebShop inventory binding")
    for name, expected in manifest["artifacts"].items():
        path = (release / name).resolve()
        if not path.is_relative_to(release) or file_hash(path) != expected:
            raise ValueError(f"Synthetic WebShop artifact changed: {name}")
    environment = manifest["environment"]
    if environment["scorer"] != "official" or environment["source_revision"] != OFFICIAL_REVISION:
        raise ValueError("Synthetic WebShop requires the pinned original official scorer")
    for name, expected in environment["assets"].items():
        if manifest["artifacts"].get(name) != expected:
            raise ValueError(f"Synthetic WebShop inventory asset binding differs: {name}")
    actual_index = {"index/" + str(p.relative_to(release / "index"))
                    for p in (release / "index").rglob("*") if p.is_file() and p.name != "write.lock"}
    if actual_index != {n for n in environment["assets"] if n.startswith("index/")}:
        raise ValueError("Synthetic WebShop search index file set changed")
    if source_root is not None:
        for name, expected in environment["scorer_sources"].items():
            if file_hash(Path(source_root) / name) != expected:
                raise ValueError(f"Original official scorer changed: {name}")
    goals = [json.loads(line) for line in (release / "goals.jsonl").read_text().splitlines()]
    random.Random(233).shuffle(goals)
    if len(goals) != environment["goal_count"] or any("_quality_contract" in g for g in goals):
        raise ValueError("Synthetic WebShop goal inventory differs")
    tasks = [json.loads(line) for line in (release / "tasks.jsonl").read_text().splitlines()]
    bindings = json.loads((release / "reference_bindings.private.json").read_text())
    if len(tasks) != 128 or len({t["id"] for t in tasks}) != 128 or len(bindings) != 128:
        raise ValueError("Synthetic WebShop requires 128 distinct validation tasks")
    quality = (json.loads((release / "quality_certificates.private.json").read_text())
               if manifest["version"] == VERSION else None)
    if quality is not None:
        if ("quality_certificates.private.json" not in manifest["artifacts"]
                or quality.get("purpose") != "inference_only"
                or manifest.get("purpose") != "inference_only"
                or quality.get("policy") != "webshop-public-evidence-v2"
                or quality["split"].get("train_count") != 0
                or set(quality["certificates"]) != set(bindings)):
            raise ValueError("Quality release must bind all 128 inference-only task certificates")
    eval_asins = set()
    with sqlite3.connect(f"file:{release / 'products.sqlite3'}?mode=ro", uri=True) as db:
        if db.execute("SELECT COUNT(*) FROM products").fetchone()[0] != 1000:
            raise ValueError("Synthetic WebShop requires exactly 1000 products")
        for task in tasks:
            metadata = task["metadata"]
            index = metadata["goal_index"]
            goal = goals[index]
            expected_id = f"webshop/goal-{index:05d}"
            if (task["dataset"] != "webshop" or task["id"] != expected_id
                    or metadata["goal_id"] != expected_id or task["prompt"] != goal["instruction_text"]
                    or metadata["webshop_inventory"] != {**manifest["binding"], "prompt_sha256":
                                                          hashlib.sha256(task["prompt"].encode()).hexdigest()}):
                raise ValueError(f"Synthetic WebShop task/goal binding differs: {task['id']}")
            product, price = db.execute("SELECT product_json, price FROM products WHERE asin = ?",
                                        (goal["asin"],)).fetchone()
            product = json.loads(product)
            if reference_fingerprint(goal, product, price) != bindings[str(index)]:
                raise ValueError(f"Synthetic WebShop reference fingerprint differs: {index}")
            if quality is not None:
                certificate = dict(quality["certificates"][str(index)])
                fingerprint = certificate.pop("fingerprint")
                if (certificate.get("policy") != "webshop-public-evidence-v2"
                        or certificate.get("accepted") is not True or certificate.get("reasons")
                        or fingerprint != digest({"goal": goal, "product": product, "certificate": certificate})):
                    raise ValueError(f"Public-evidence certificate differs: {index}")
                if goal["asin"] in eval_asins:
                    raise ValueError("Validation tasks must reference 128 different products")
                eval_asins.add(goal["asin"])
    if quality is not None:
        from collections import Counter
        counts = Counter(c["category"] for c in quality["certificates"].values())
        if (dict(counts) != quality["split"]["eval_category_quotas"]
                or quality["split"]["eval_product_cap"] != 1
                or quality["split"]["eval_products"] != 128
                or manifest.get("task_quality", {}).get("split") != quality["split"]):
            raise ValueError("Quality release product diversity or category quotas differ")
    return manifest


def validate_deployment(args):
    if Path(args.inventory_manifest).name != "manifest.json":
        raise ValueError("Synthetic WebShop inventory must use the release manifest.json")
    release = Path(args.inventory_manifest).resolve().parent
    manifest = validate_release(release, args.source_root)
    for field, name in (("store", "products.sqlite3"), ("goals", "goals.jsonl"), ("index", "index")):
        if Path(getattr(args, field)).resolve() != release / name:
            raise ValueError(f"Synthetic WebShop deployment uses a different {field}")
    if args.scorer != "auto" or args.source_revision != OFFICIAL_REVISION:
        raise ValueError("Synthetic WebShop deployment must use original official scoring")
    return manifest["binding"]
