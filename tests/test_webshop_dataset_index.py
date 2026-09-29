import importlib.util
from pathlib import Path
import random

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/formal/webshop_dataset_index.py"
spec = importlib.util.spec_from_file_location("webshop_dataset_index", SCRIPT)
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


def test_raw_to_session_permutation_preserves_duplicates_and_quarantines_splits():
    instructions = [f"public requirement {i}" for i in range(1600)]
    instructions[160] = instructions[1500] = "same public text for distinct release rows"
    order = list(range(len(instructions)))
    random.Random(233).shuffle(order)
    rows = [{"id": f"webshop/goal-{i:05d}", "prompt": instructions[i],
             "cluster_id": "cluster", "metadata": {"goal_id": f"goal-{i:05d}"}}
            for i in range(1600)]
    repaired, quarantine = index.repair_raw_records(rows, instructions, order)
    assert len(repaired) == 100 and len(quarantine) == 1500
    assert not index.validate_records(repaired + quarantine, instructions, order)
    duplicate = [r for r in repaired + quarantine if r["prompt"].startswith("same public")]
    assert len(duplicate) == 2 and len({r["id"] for r in duplicate}) == 2
    assert len({r["metadata"]["webshop_index_repair"]["raw_release_index"] for r in duplicate}) == 2
    assert all(r["cluster_size"] == 100 for r in repaired)
    assert [r["rank_in_cluster"] for r in repaired] == list(range(100))
    with pytest.raises(ValueError, match="provenance"):
        index.repair_raw_records([{**rows[0], "prompt": "wrong"}], instructions, order)


# Inference/evidence tests formerly below are retained on
# experiment/webshop-main-before-restore-20260929. This suite protects the
# corrected training data retained independently of the restored inference code.
