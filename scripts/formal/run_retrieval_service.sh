#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source scripts/formal/environment.sh

export PYTHONPATH="$ROOT/src:$SPGFS_RETRIEVAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$SPGFS_RETRIEVAL_BACKEND" == "wikipedia" ]]; then
  exec python -m selfplay_graph_flowsteer.wikipedia_retrieval_service \
    --cache "$SPGFS_WIKIPEDIA_CACHE" --host 127.0.0.1 --port "$SPGFS_RETRIEVAL_PORT" "$@"
fi
if [[ "$SPGFS_RETRIEVAL_BACKEND" == "faiss" ]]; then
  for path in "$SPGFS_SEARCHR1_INDEX" "$SPGFS_SEARCHR1_CORPUS"; do
    if [[ ! -s "$path" ]]; then
      printf 'Configured retrieval asset missing: %s\nSee docs/NQ_R2D2_FORMAL_PROMOTION_20260929.zh-CN.md for the pinned R2D2 assets.\n' "$path" >&2
      exit 2
    fi
  done
  profile="${SPGFS_RETRIEVAL_PROFILE:-nq-dense8-v1}"
  reranker_args=()
  if [[ "$profile" == "nq-minilm-rrf8-v1" ]]; then
    reranker_args=(--reranker "${SPGFS_RETRIEVAL_RERANKER_MODEL:-$ROOT/../models/ms-marco-MiniLM-L6-v2}")
  fi
  exec "$SPGFS_RETRIEVAL_PYTHON" -m selfplay_graph_flowsteer.dense_retrieval_service \
    --host 127.0.0.1 --port "$SPGFS_RETRIEVAL_PORT" --index "$SPGFS_SEARCHR1_INDEX" \
    --corpus "$SPGFS_SEARCHR1_CORPUS" --model "$SPGFS_SEARCHR1_MODEL" \
    --threads "${SPGFS_RETRIEVAL_THREADS:-4}" \
    --profile "$profile" --candidate-k "${SPGFS_RETRIEVAL_CANDIDATE_K:-128}" \
    --batch-size "${SPGFS_RETRIEVAL_BATCH_SIZE:-8}" \
    --max-wait-ms "${SPGFS_RETRIEVAL_MAX_WAIT_MS:-20}" \
    --max-pending "${SPGFS_RETRIEVAL_MAX_PENDING:-128}" \
    --timeout-s "${SPGFS_RETRIEVAL_TIMEOUT_S:-240}" \
    "${reranker_args[@]}" "$@"
fi
if [[ "$SPGFS_RETRIEVAL_BACKEND" != "sqlite" ]]; then
  printf 'Unknown retrieval backend: %s\n' "$SPGFS_RETRIEVAL_BACKEND" >&2
  exit 2
fi

if [[ ! -s "$SPGFS_RETRIEVAL_INDEX" ]]; then
  printf 'Retrieval index does not exist: %s\n' "$SPGFS_RETRIEVAL_INDEX" >&2
  printf 'Run scripts/formal/prepare_nq_retrieval.py first.\n' >&2
  exit 2
fi

exec python -m selfplay_graph_flowsteer.retrieval_service \
  --host 127.0.0.1 --port "$SPGFS_RETRIEVAL_PORT" --index "$SPGFS_RETRIEVAL_INDEX" "$@"
