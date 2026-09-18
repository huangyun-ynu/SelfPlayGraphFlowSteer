from __future__ import annotations

import importlib.util
import random
from pathlib import Path


def _load_prepare_raw_pools():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "formal" / "prepare_raw_pools.py"
    spec = importlib.util.spec_from_file_location("prepare_raw_pools", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_webshop_rows_use_official_shuffled_goal_order() -> None:
    module = _load_prepare_raw_pools()
    goals = [
        {
            "asin": f"asin-{index}",
            "goal_options": {"option": str(index)},
            "instruction_text": f"instruction {index}",
            "weight": 1,
        }
        for index in range(12)
    ]
    official = list(goals)
    random.Random(module.WEBSHOP_OFFICIAL_SHUFFLE_SEED).shuffle(official)

    rows = module.webshop_rows(goals, range(3), split="test")

    assert [row["metadata"]["goal_index"] for row in rows] == [0, 1, 2]
    assert [row["prompt"] for row in rows] == [
        official[0]["instruction_text"],
        official[1]["instruction_text"],
        official[2]["instruction_text"],
    ]
    assert [row["ads_target"]["asin"] for row in rows] == [
        official[0]["asin"],
        official[1]["asin"],
        official[2]["asin"],
    ]
    assert rows[0]["metadata"]["source_lineage"]["goal_order"] == (
        "official_random_seed_233"
    )
