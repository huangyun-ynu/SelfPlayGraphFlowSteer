#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
set -a
source .env
set +a
# This workspace is assigned GPU 1; this entrypoint does not start GPU services.
export SPGFS_ALLOWED_PHYSICAL_GPUS=1
export CUDA_VISIBLE_DEVICES=1
source scripts/formal/environment.sh

PYTHON=${SPGFS_PYTHON:-$ROOT/.venv/bin/python}
OUTPUT=${SPGFS_WEBSHOP_EVAL_OUTPUT:-state/formal-eval/webshop-official-m02-$(date +%Y%m%d-%H%M%S)}
DIRECTOR_URL=${SPGFS_WEBSHOP_DIRECTOR_URL:-http://127.0.0.1:18603/v1}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\nChoose a new output directory.\n' "$OUTPUT" >&2
  exit 2
fi

exec "$PYTHON" -m selfplay_graph_flowsteer benchmark \
  --config configs/webshop_official_eval.toml \
  --dataset data/formal/eval/webshop_official_test_128.jsonl \
  --output "$OUTPUT" \
  --seed 0 --workers 24 --disable-swe \
  --worker-route deepseek \
  --director-base-url "$DIRECTOR_URL" \
  --director-api-key EMPTY --director-model Qwen3.5-9B \
  --director-thinking --skill-context off --wandb-mode offline
