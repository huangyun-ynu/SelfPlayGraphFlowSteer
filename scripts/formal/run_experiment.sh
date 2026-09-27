#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ ! -f .env ]]; then
  printf 'Missing %s/.env\n' "$ROOT" >&2
  exit 2
fi
set -a
source .env
set +a
source scripts/formal/environment.sh
# Always collect with the current checkout, including promoted dataset fixes.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR"
mkdir -p \
  state/formal-training-output-contract-v2/swe/repo-cache \
  state/formal-training-output-contract-v2/swe/workspaces \
  state/formal-training-output-contract-v2/private/swe/artifacts

: "${SPGFS_FORMAL_TASK_POOL:?Set SPGFS_FORMAL_TASK_POOL to the validated seven-dataset JSONL pool}"
if [[ ! -f "$SPGFS_FORMAL_TASK_POOL" ]]; then
  printf 'Formal task pool does not exist: %s\n' "$SPGFS_FORMAL_TASK_POOL" >&2
  exit 2
fi

IFS=, read -r DEFAULT_PROPOSER_GPU DEFAULT_SOLVER_GPU DEFAULT_ASYNC_ROLLOUT_GPU _ \
  <<<"$SPGFS_ALLOWED_PHYSICAL_GPUS"
PROPOSER_GPU_ID="${SPGFS_PROPOSER_GPU_ID:-$DEFAULT_PROPOSER_GPU}"
SOLVER_GPU_ID="${SPGFS_SOLVER_GPU_ID:-$DEFAULT_SOLVER_GPU}"
ASYNC_ROLLOUT_GPU_ID="${SPGFS_ASYNC_ROLLOUT_GPU_ID:-$DEFAULT_ASYNC_ROLLOUT_GPU}"
if [[ -z "$PROPOSER_GPU_ID" || -z "$SOLVER_GPU_ID" || -z "$ASYNC_ROLLOUT_GPU_ID" ]]; then
  printf 'Set three GPU IDs in SPGFS_ALLOWED_PHYSICAL_GPUS or set role GPU variables.\n' >&2
  exit 2
fi
if [[ "$ASYNC_ROLLOUT_GPU_ID" == "$PROPOSER_GPU_ID" \
  || "$ASYNC_ROLLOUT_GPU_ID" == "$SOLVER_GPU_ID" ]]; then
  printf 'The async rollout GPU must differ from both training GPUs.\n' >&2
  exit 2
fi

WEBSHOP_PID=""
RETRIEVAL_PID=""
cleanup_services() {
  if [[ -n "$WEBSHOP_PID" ]] && kill -0 "$WEBSHOP_PID" 2>/dev/null; then
    kill "$WEBSHOP_PID" 2>/dev/null || true
    wait "$WEBSHOP_PID" 2>/dev/null || true
  fi
  if [[ -n "$RETRIEVAL_PID" ]] && kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
    kill "$RETRIEVAL_PID" 2>/dev/null || true
    wait "$RETRIEVAL_PID" 2>/dev/null || true
  fi
}
trap cleanup_services EXIT

retrieval_healthy() {
  python scripts/formal/check_retrieval_service.py
}

if [[ "${SPGFS_ENABLE_LOCAL_RETRIEVAL:-0}" == "1" ]] && ! retrieval_healthy; then
  mkdir -p state/formal-training-output-contract-v2/retrieval
  scripts/formal/run_retrieval_service.sh \
    >state/formal-training-output-contract-v2/retrieval/service.log 2>&1 &
  RETRIEVAL_PID=$!
  for _ in $(seq 1 "${SPGFS_RETRIEVAL_STARTUP_SECONDS:-1800}"); do
    retrieval_healthy && break
    kill -0 "$RETRIEVAL_PID" 2>/dev/null || break
    sleep 1
  done
  if ! retrieval_healthy; then
    printf 'Retrieval service failed to start; see retrieval/service.log.\n' >&2
    exit 3
  fi
fi

webshop_healthy() {
  python - <<'PY'
import json
import os
from urllib.request import urlopen

try:
    port = int(os.environ["SPGFS_WEBSHOP_PORT"])
    with urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
raise SystemExit(
    0
    if payload.get("status") == "ok"
    and payload.get("idempotency_protocol") == "webshop-request-v1"
    and payload.get("index_path") == os.path.realpath(os.environ["SPGFS_WEBSHOP_INDEX"])
    else 1
)
PY
}

if ! webshop_healthy; then
  mkdir -p state/formal-training-output-contract-v2/webshop_sidecar
  python -m selfplay_graph_flowsteer.webshop_sidecar \
    --host 127.0.0.1 --port "$SPGFS_WEBSHOP_PORT" \
    --interpreter "$SPGFS_WEBSHOP_INTERPRETER" \
    --worker-script "$SPGFS_WEBSHOP_WORKER" \
    --source-root "$SPGFS_WEBSHOP_SOURCE_ROOT" \
    --source-revision "$SPGFS_WEBSHOP_SOURCE_REVISION" \
    --store "$SPGFS_WEBSHOP_STORE" \
    --goals "$SPGFS_WEBSHOP_GOALS" \
    --index "$SPGFS_WEBSHOP_INDEX" \
    --java-home "$SPGFS_WEBSHOP_JAVA_HOME" \
    --worker-timeout 180 --max-sessions 48 --max-initializers 4 \
    >state/formal-training-output-contract-v2/webshop_sidecar/service.log 2>&1 &
  WEBSHOP_PID=$!
  for _ in $(seq 1 120); do
    if webshop_healthy; then
      break
    fi
    if ! kill -0 "$WEBSHOP_PID" 2>/dev/null; then
      printf 'WebShop sidecar exited during startup. See webshop_sidecar/service.log.\n' >&2
      exit 3
    fi
    sleep 1
  done
  if ! webshop_healthy; then
    printf 'WebShop sidecar did not become healthy within 120 seconds.\n' >&2
    exit 3
  fi
fi

# Freeze NQ evidence once per public question, before Proposer/Solver collection.
# All other rows and their train/eval identities are preserved by the preparer.
FORMAL_QA_TASK_POOL="state/formal-training-output-contract-v2/qa-baseline/task_pool.jsonl"
python -m selfplay_graph_flowsteer.nq_frozen_context \
  --config configs/formal_training.toml \
  --input "$SPGFS_FORMAL_TASK_POOL" \
  --output "$FORMAL_QA_TASK_POOL"

python scripts/formal/wandb_direct_exec.py -- \
python -m selfplay_graph_flowsteer selfplay-experiment \
  --config configs/formal_training.toml \
  --task-pool "$FORMAL_QA_TASK_POOL" \
  --curriculum-profile configs/curriculum/formal_3500.toml \
  --output state/formal-training-output-contract-v2/experiment \
  --route-report state/formal-training-output-contract-v2/route_report.json \
  --minimum-selected-routes 1 \
  --cycles "${SPGFS_FORMAL_CYCLES:-256}" --final-cycle-evaluation-only \
  --workers 24 --pipeline-counterfactuals --historical-duration-priority \
  --enable-swe \
  --frontier-reverify-workers 8 \
  --pipeline-frontier-by-dataset \
  --async-next-cycle-rollouts --async-rollout-gpu-id "$ASYNC_ROLLOUT_GPU_ID" \
  --parallel-role-training --max-sequence-length 32768 \
  --proposer-gpu-id "$PROPOSER_GPU_ID" --solver-gpu-id "$SOLVER_GPU_ID" \
  --max-micro-batch-tokens 32768 --micro-batch-size 1 \
  --raw-policy-backward-mode "${SPGFS_RAW_POLICY_BACKWARD_MODE:-timeline}" \
  --activation-cpu-offload --activation-cpu-offload-min-tokens 4096 \
  --manage-services --service-state-dir state/formal-training-output-contract-v2/policy_services \
  "$@"
