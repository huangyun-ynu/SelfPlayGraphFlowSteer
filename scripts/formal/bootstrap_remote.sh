#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  printf 'Missing %s/.env; restore it from the private configuration document.\n' "$ROOT" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${SPGFS_DATA_ARCHIVE_KEY:?Set SPGFS_DATA_ARCHIVE_KEY in .env}"
export SPGFS_ALLOWED_PHYSICAL_GPUS="${SPGFS_ALLOWED_PHYSICAL_GPUS:-0,1}"

ARCHIVE=data/formal/private/formal_data.tar.gz.enc
ARCHIVE_SHA=data/formal/private/formal_data.tar.gz.enc.sha256
if [[ ! -f "$ARCHIVE" || ! -f "$ARCHIVE_SHA" ]]; then
  printf 'Encrypted formal data package is missing; run git pull first.\n' >&2
  exit 2
fi
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum --check "$(basename "$ARCHIVE_SHA")"
)

mkdir -p state/formal-data
openssl enc -d -aes-256-cbc -pbkdf2 \
  -pass env:SPGFS_DATA_ARCHIVE_KEY \
  -in "$ARCHIVE" | tar --warning=no-timestamp -xzf - -C state/formal-data

chmod 700 state/formal-data
find state/formal-data -type d -exec chmod 700 {} +
find state/formal-data -type f -exec chmod 600 {} +

missing=0
for required in \
  models/Qwen3.5-9B \
  models/e5-base-v2 \
  assets/alfworld-data/json_2.1.1 \
  assets/webshop/source \
  assets/webshop/venv/bin/python \
  assets/webshop/prepared/products.sqlite3 \
  assets/webshop/prepared/goals.jsonl \
  assets/webshop/private-evaluation \
  state/formal-training/private/swe/identity \
  state/formal-training/private/swe/known_hosts; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing required asset: %s\n' "$ROOT/$required" >&2
    missing=1
  fi
done
if (( missing )); then
  exit 3
fi

python - <<'PY'
from pathlib import Path

from selfplay_graph_flowsteer.application import load_adaptive_config
from selfplay_graph_flowsteer.curriculum import FixedTaskPool

root = Path.cwd()
config = load_adaptive_config(root / "configs/formal_training.toml")
pool = FixedTaskPool.from_jsonl(
    [root / "state/formal-data/validated_task_pool.jsonl"],
    require_ads_metadata=True,
    require_validation_manifest=True,
)
if len(pool.ids) != 3584:
    raise SystemExit(f"expected 3584 formal tasks, found {len(pool.ids)}")
print(f"configuration valid; formal task pool rows={len(pool.ids)}")
PY

cat <<EOF
Formal configuration and encrypted task data are ready.
Before training, provision:
  $ROOT/models/Qwen3.5-9B
  $ROOT/models/e5-base-v2
  $ROOT/assets/alfworld-data/json_2.1.1
  $ROOT/assets/webshop/
  $ROOT/state/formal-training/private/swe/identity
  $ROOT/state/formal-training/private/swe/known_hosts
Then generate a fresh state/formal-training/route_report.json and run:
  scripts/formal/run_experiment.sh
EOF
