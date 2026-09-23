#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
set -a
source .env
set +a
# Evaluation uses a separately provisioned Director; it does not bind a GPU.
# Resource IDs in the TOML are training metadata, not this service's placement.
export SPGFS_ALLOWED_PHYSICAL_GPUS=${SPGFS_ALLOWED_PHYSICAL_GPUS:-0,1,2,3,4,5,6,7}
source scripts/formal/environment.sh
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${SPGFS_PYTHON:-$SPGFS_VENV/bin/python}
OUTPUT=${SPGFS_WEBSHOP_EVAL_OUTPUT:-state/formal-eval/webshop-laser-checklist-c24-$(date +%Y%m%d-%H%M%S)}
DIRECTOR_URL=${SPGFS_WEBSHOP_DIRECTOR_URL:-http://127.0.0.1:18623/v1}
WORKERS=${SPGFS_WEBSHOP_WORKERS:-24}
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi
curl --noproxy '*' --fail --silent --show-error --max-time 10 \
  -H 'Authorization: Bearer EMPTY' "$DIRECTOR_URL/models" >/dev/null
curl --noproxy '*' --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:18020/health >/dev/null

exec "$PYTHON" -m selfplay_graph_flowsteer benchmark \
  --config configs/webshop_laser_checklist_eval.toml \
  --dataset data/formal/eval/webshop_official_test_128.jsonl \
  --output "$OUTPUT" --seed 0 --workers "$WORKERS" --disable-swe \
  --worker-route deepseek --director-base-url "$DIRECTOR_URL" \
  --director-api-key EMPTY --director-model Qwen3.5-9B \
  --director-thinking --skill-context off --wandb-mode offline
