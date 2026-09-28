#!/usr/bin/env bash
# Validated v2.2 evaluation against already running Director/WebShop services.
# Supply a fresh run-specific config (trace/route-health paths) and output path.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESOURCE_ROOT=${SPGFS_RESOURCE_ROOT:-$ROOT}
CODE_ROOT=${SPGFS_WEBSHOP_CODE_ROOT:-$ROOT}
CONFIG=${SPGFS_WEBSHOP_EVAL_CONFIG:?Set a run-specific v2.2 config path}
OUTPUT=${SPGFS_WEBSHOP_EVAL_OUTPUT:?Set a fresh output path}
DATASET=${SPGFS_WEBSHOP_EVAL_DATASET:-$RESOURCE_ROOT/data/formal/eval/webshop_official_test_128.jsonl}
PROFILE=${SPGFS_WEBSHOP_PROFILE:-m02_merged_identity_v1}
DIRECTOR_URL=${SPGFS_WEBSHOP_DIRECTOR_URL:-http://127.0.0.1:18603/v1}
PYTHON=${SPGFS_PYTHON:-$RESOURCE_ROOT/.venv/bin/python}
COMMIT=${SPGFS_WEBSHOP_EXPECTED_COMMIT:-$(git -C "$CODE_ROOT" rev-parse HEAD)}
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.log" && ! -e "$OUTPUT.preflight.json" ]] || {
  echo 'Run output or log already exists; use a fresh run path' >&2
  exit 2
}
cd "$RESOURCE_ROOT"
set -a
source .env
set +a
export PYTHONPATH="$CODE_ROOT/src"
mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON" "$CODE_ROOT/scripts/formal/check_webshop_run.py" \
  --source "$CODE_ROOT" --commit "$COMMIT" --config "$CONFIG" \
  --dataset "$DATASET" --goals "$RESOURCE_ROOT/assets/webshop/prepared/goals.jsonl" \
  --output "$OUTPUT" --profile "$PROFILE" --record "$OUTPUT.preflight.json"
# This entrypoint does not start or stop a GPU process. Services have their own
# attempt directories and cleanup records (see the prepared full-run launchers).
set -C
exec "$PYTHON" -m selfplay_graph_flowsteer benchmark \
  --config "$CONFIG" --dataset "$DATASET" --output "$OUTPUT" \
  --seed 0 --workers 40 --disable-swe --worker-route deepseek \
  --director-base-url "$DIRECTOR_URL" --director-api-key EMPTY --director-model Qwen3.5-9B \
  --director-thinking --skill-context off --wandb-mode disabled >"$OUTPUT.log" 2>&1
