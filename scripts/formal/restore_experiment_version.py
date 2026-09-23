#!/usr/bin/env python3
"""Restore an archived source version into a NEW directory; never starts a job."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(version_id: str, destination: Path, root: Path) -> int:
    archive = root / "experiment_versions"
    index = json.loads((archive / "index.json").read_text())
    version = next((v for v in index["versions"] if v["id"] == version_id), None)
    if version is None:
        raise ValueError(f"Unknown version {version_id!r}; use --list")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination must not already exist")
    patch = archive / version["patch"]
    if sha256(patch) != version["patch_sha256"]:
        raise ValueError("Patch SHA-256 mismatch")
    content = subprocess.check_output(
        ["git", "archive", "--format=tar", version["base_commit"]], cwd=root
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".spgfs-restore-", dir=destination.parent) as work:
        temp = Path(work) / "source"
        temp.mkdir()
        with tarfile.open(fileobj=io.BytesIO(content)) as tar:
            for member in tar.getmembers():
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported Git archive entry: {member.name}")
            tar.extractall(temp)
        # A separate repository prevents git apply from discovering the caller's
        # parent checkout when destination is inside its ignored state directory.
        subprocess.run(["git", "init", "--quiet", str(temp)], check=True)
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=temp, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=temp, check=True)
        for name, expected in version["files_sha256"].items():
            if sha256(temp / name) != expected:
                raise ValueError(f"Restored source SHA-256 mismatch: {name}")
        temp.rename(destination)
    return len(version["files_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?")
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.list:
        index = json.loads((root / "experiment_versions/index.json").read_text())
        for version in index["versions"]:
            print(version["id"], version["scope"])
        return
    if not args.version or args.destination is None:
        parser.error("version and a new destination are required")
    count = restore(args.version, args.destination, root)
    print(f"Restored {args.version}: {count} snapshot file hashes verified. No experiment was launched.")


if __name__ == "__main__":
    main()
