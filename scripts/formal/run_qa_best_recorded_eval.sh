#!/usr/bin/env bash
set -euo pipefail

# Explicit historical control. Formal training never invokes this script.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
DATASET_NAME="${1:-hotpotqa}"
case "$DATASET_NAME" in
  hotpotqa)
    DEFAULT_DATASET="$ROOT/data/formal/eval/hotpotqa_official_test.jsonl"
    DEFAULT_WORKERS=10
    ;;
  nq_open)
    DEFAULT_DATASET="$ROOT/state/formal-training/qa-baseline/eval/nq_open_frozen_128.jsonl"
    DEFAULT_WORKERS=24
    ;;
  *) printf 'Usage: %s {hotpotqa|nq_open} [dataset.jsonl] [output]\n' "$0" >&2; exit 2 ;;
esac
DATASET="${2:-$DEFAULT_DATASET}"
OUTPUT="${3:-$ROOT/state/formal-eval/qa-best-recorded/$DATASET_NAME}"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export SPGFS_ALLOWED_PHYSICAL_GPUS="${SPGFS_ALLOWED_PHYSICAL_GPUS:-0}"
source scripts/formal/environment.sh
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
test -s "$DATASET"

# Qwen and (for HotpotQA) the retrieval service must already be running.
# Per-dataset thinking is specified only in this control profile.
exec "$SPGFS_VENV/bin/python" -m selfplay_graph_flowsteer benchmark \
  --config configs/qa_best_recorded_eval.toml \
  --dataset "$DATASET" --output "$OUTPUT" \
  --verifier flowsteer_qa --workers "${SPGFS_WORKERS:-$DEFAULT_WORKERS}" \
  --limit-per-dataset 128 --wandb-mode disabled \
  --skill-context off --disable-swe
