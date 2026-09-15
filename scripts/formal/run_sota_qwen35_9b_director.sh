#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

set -a
source .env
set +a
source scripts/formal/environment.sh

PYTHON=${SPGFS_PYTHON:-python}
OUTPUT=${SPGFS_SOTA_OUTPUT:-state/sota-20260916/qwen35-9b-director-full}
DIRECTOR_URL=${SPGFS_SOTA_DIRECTOR_URL:-http://127.0.0.1:18603/v1}
HEALTHBENCH_TEST=${SPGFS_HEALTHBENCH_TEST:-state/formal-data/healthbench_professional_test_128.jsonl}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=${SPGFS_SOTA_TMPDIR:-/tmp}
export WANDB_NAME=${WANDB_NAME:-qwen35-9b-director-sota-six-datasets-20260916}

curl --noproxy '*' --fail --silent --show-error \
  -H 'Authorization: Bearer EMPTY' "$DIRECTOR_URL/models" >/dev/null
curl --noproxy '*' --fail --silent --show-error \
  http://127.0.0.1:8020/health >/dev/null
test -s "$HEALTHBENCH_TEST"

exec "$PYTHON" scripts/formal/wandb_direct_exec.py -- \
  "$PYTHON" -m selfplay_graph_flowsteer benchmark \
  --config configs/formal_training.toml \
  --dataset data/formal/eval/aime_official_test.jsonl \
  --additional-dataset data/formal/eval/nq_open_official_test.jsonl \
  --additional-dataset data/formal/eval/hotpotqa_official_test.jsonl \
  --additional-dataset data/formal/eval/webshop_official_test_128.jsonl \
  --additional-dataset data/formal/eval/alfworld_official_test.jsonl \
  --additional-dataset "$HEALTHBENCH_TEST" \
  --output "$OUTPUT" \
  --workers 24 --historical-duration-priority --disable-swe \
  --director-base-url "$DIRECTOR_URL" \
  --director-api-key EMPTY --director-model Qwen3.5-9B --no-director-thinking \
  --director-skill-root configs/director_skills/sota_qwen35_9b \
  --skill-context off --wandb-mode online \
  "$@"
