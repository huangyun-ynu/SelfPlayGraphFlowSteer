"""Create a relocated dataset plus provenance without overwriting the source."""

import argparse
import hashlib
import json
from pathlib import Path

from selfplay_graph_flowsteer.dataset_migration import relocate_alfworld_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "old-root", "new-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    manifest_path = destination.with_suffix(destination.suffix + ".migration.json")
    if destination.exists() or manifest_path.exists():
        parser.error("destination or migration manifest already exists")
    rows, changes = [], []
    with args.source.open() as source:
        for line in source:
            if not line.strip():
                continue
            row, change = relocate_alfworld_record(
                json.loads(line), old_root=args.old_root, new_root=args.new_root,
            )
            rows.append(row)
            if change:
                changes.append(change)
    # Validate every game before writing either output.
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "destination_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "rows": len(rows), "relocated": len(changes), "changes": changes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(serialized)
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "relocated": len(changes), "destination": str(destination)}))


if __name__ == "__main__":
    main()
