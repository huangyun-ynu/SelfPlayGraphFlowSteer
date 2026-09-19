#!/usr/bin/env bash
set -euo pipefail

# Reproducible fixed-context NQ evaluation. The evidence cache is prepared
# offline; benchmark execution exposes no search Action for these rows.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV="${SPGFS_VENV:-${ROOT}/../.venvs/spgfs-pats-gpu}"
DATASET="${1:-${ROOT}/state/experiments/nq-frozen-v1/nq_open_128.jsonl}"
OUTPUT="${2:-${ROOT}/state/experiments/nq-frozen-v1/run-nq-frozen}"
WORKERS="${SPGFS_WORKERS:-24}"

cd "${ROOT}"
source .env
source scripts/formal/environment.sh

PYTHONPATH=src "${VENV}/bin/python" -m selfplay_graph_flowsteer benchmark \
  --config configs/formal_training.toml \
  --dataset "${DATASET}" \
  --output "${OUTPUT}" \
  --verifier flowsteer_qa \
  --workers "${WORKERS}" \
  --limit-per-dataset 128 \
  --wandb-mode disabled \
  --worker-route deepseek \
  --director-base-url http://127.0.0.1:18603/v1 \
  --director-api-key EMPTY \
  --director-model Qwen3.5-9B \
  --no-director-thinking \
  --disable-swe
