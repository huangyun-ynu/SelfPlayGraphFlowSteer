from pathlib import Path

import pytest

from selfplay_graph_flowsteer.dataset_migration import relocate_alfworld_record


def fixture_row(tmp_path):
    relative = Path("valid_seen/task/trial/game.tw-pddl")
    target = tmp_path / "assets" / relative
    target.parent.mkdir(parents=True)
    target.write_text("trusted game")
    return {
        "dataset": "alfworld", "id": "alfworld/valid_seen/task/trial",
        "metadata": {"game_path": str(Path('/old/assets') / relative), "source_split": "valid_seen"},
    }, tmp_path / "assets"


def test_relocation_preserves_identity_and_source(tmp_path):
    row, root = fixture_row(tmp_path)
    migrated, manifest = relocate_alfworld_record(row, old_root=Path("/old/assets"), new_root=root)
    assert row["metadata"]["game_path"].startswith("/old/assets/")
    assert migrated["id"] == row["id"]
    assert Path(migrated["metadata"]["game_path"]).is_file()
    assert len(manifest["game_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["wrong_root", "traversal", "identity", "split", "symlink"])
def test_relocation_rejects_untrusted_mapping(tmp_path, mutation):
    row, root = fixture_row(tmp_path)
    if mutation == "wrong_root":
        row["metadata"]["game_path"] = "/other/assets/valid_seen/task/trial/game.tw-pddl"
    elif mutation == "traversal":
        row["metadata"]["game_path"] = "/old/assets/../assets/valid_seen/task/trial/game.tw-pddl"
    elif mutation == "identity":
        row["id"] = "alfworld/valid_seen/different/trial"
    elif mutation == "split":
        row["metadata"]["source_split"] = "train"
    else:
        target = root / "valid_seen/task/trial/game.tw-pddl"
        target.unlink()
        outside = tmp_path / "game.tw-pddl"
        outside.write_text("outside")
        target.symlink_to(outside)
    with pytest.raises(ValueError):
        relocate_alfworld_record(row, old_root=Path("/old/assets"), new_root=root)
