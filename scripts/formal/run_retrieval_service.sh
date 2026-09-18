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
      printf 'Search-R1 data missing: %s\nRun scripts/formal/prepare_searchr1_retrieval.py first.\n' "$path" >&2
      exit 2
    fi
  done
  exec "$SPGFS_RETRIEVAL_PYTHON" -m selfplay_graph_flowsteer.dense_retrieval_service \
    --host 127.0.0.1 --port "$SPGFS_RETRIEVAL_PORT" --index "$SPGFS_SEARCHR1_INDEX" \
    --corpus "$SPGFS_SEARCHR1_CORPUS" --model "$SPGFS_SEARCHR1_MODEL" \
    --threads "${SPGFS_RETRIEVAL_THREADS:-4}" "$@"
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
