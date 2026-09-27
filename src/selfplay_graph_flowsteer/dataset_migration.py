"""Explicit relocation of trusted ALFWorld dataset paths between deployments."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path


def relocate_alfworld_record(record: dict, *, old_root: Path, new_root: Path) -> tuple[dict, dict | None]:
    """Return a copied row and provenance; never relax the runtime root check."""
    if record.get("dataset") != "alfworld":
        return copy.deepcopy(record), None
    old_root = old_root.expanduser().absolute()
    new_root = new_root.expanduser().resolve(strict=True)
    metadata = record.get("metadata", {})
    old = Path(metadata["game_path"])
    if not old.is_absolute() or ".." in old.parts:
        raise ValueError("ALFWorld source path must be absolute without traversal")
    relative = old.relative_to(old_root)
    target = (new_root / relative).resolve(strict=True)
    target.relative_to(new_root)
    if not target.is_file() or target.name != "game.tw-pddl":
        raise ValueError("ALFWorld target must be an existing game.tw-pddl")
    expected_id = "alfworld/" + relative.parent.as_posix()
    if record.get("source_id", record.get("id")) != expected_id:
        raise ValueError("ALFWorld source identity does not match the relative game path")
    if metadata.get("source_split") != relative.parts[0]:
        raise ValueError("ALFWorld split does not match the relative game path")
    migrated = copy.deepcopy(record)
    migrated["metadata"]["game_path"] = str(target)
    return migrated, {
        "id": record.get("id"), "old": str(old), "new": str(target),
        "game_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
