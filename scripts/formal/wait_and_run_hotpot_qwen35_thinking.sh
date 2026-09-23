#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

POLL_SECONDS="$SPGFS_HOTPOT_POLL_SECONDS"
MIN_FREE_MB="$SPGFS_HOTPOT_MIN_FREE_MB"
WORKERS="$SPGFS_HOTPOT_WORKERS"
MAX_MODEL_LEN="$SPGFS_HOTPOT_MAX_MODEL_LEN"
MODEL="$SPGFS_HOTPOT_MODEL"
DATASET="$SPGFS_HOTPOT_DATASET"
BASE_PORT="$SPGFS_HOTPOT_PORT"
[[ -n "$POLL_SECONDS" ]] || POLL_SECONDS=30
[[ -n "$MIN_FREE_MB" ]] || MIN_FREE_MB=60000
[[ -n "$WORKERS" ]] || WORKERS=20
[[ -n "$MAX_MODEL_LEN" ]] || MAX_MODEL_LEN=262144
[[ -n "$MODEL" ]] || MODEL="$ROOT/models/Qwen3.5-9B"
[[ -n "$DATASET" ]] || DATASET="$ROOT/data/formal/eval/hotpotqa_official_test.jsonl"
[[ -n "$BASE_PORT" ]] || BASE_PORT=18603

if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
  # Re-read optional values supplied by the project environment after .env.
  [[ -n "$SPGFS_HOTPOT_POLL_SECONDS" ]] && POLL_SECONDS="$SPGFS_HOTPOT_POLL_SECONDS"
  [[ -n "$SPGFS_HOTPOT_MIN_FREE_MB" ]] && MIN_FREE_MB="$SPGFS_HOTPOT_MIN_FREE_MB"
  [[ -n "$SPGFS_HOTPOT_WORKERS" ]] && WORKERS="$SPGFS_HOTPOT_WORKERS"
  [[ -n "$SPGFS_HOTPOT_MAX_MODEL_LEN" ]] && MAX_MODEL_LEN="$SPGFS_HOTPOT_MAX_MODEL_LEN"
  [[ -n "$SPGFS_HOTPOT_MODEL" ]] && MODEL="$SPGFS_HOTPOT_MODEL"
  [[ -n "$SPGFS_HOTPOT_DATASET" ]] && DATASET="$SPGFS_HOTPOT_DATASET"
  [[ -n "$SPGFS_HOTPOT_PORT" ]] && BASE_PORT="$SPGFS_HOTPOT_PORT"
fi

candidate_gpu=""
candidate_free=""
candidate_util=""
gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';'
}
find_candidate() {
  candidate_gpu=""
  candidate_free=""
  candidate_util=""
  local id uuid free util pids
  while IFS= read -r id; do
    id=$(echo "$id" | tr -d ' ')
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    uuid=$(nvidia-smi --id="$id" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    free=$(nvidia-smi --id="$id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n1)
    util=$(nvidia-smi --id="$id" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n1)
    [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && -n "$uuid" ]] || continue
    pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' -v wanted="$uuid" '$1 == wanted {print $2}')
    if [[ -z "$pids" && "$free" -ge "$MIN_FREE_MB" && "$util" -le 5 ]]; then
      candidate_gpu="$id"
      candidate_free="$free"
      candidate_util="$util"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
  return 1
}

echo "[$(date --iso-8601=seconds)] waiting for an exclusive GPU (free >= $MIN_FREE_MB MiB, utilization <= 5%)"
while ! find_candidate; do
  echo "[$(date --iso-8601=seconds)] still waiting; GPUs: $(gpu_snapshot)"
  sleep "$POLL_SECONDS"
done
GPU_ID="$candidate_gpu"
echo "[$(date --iso-8601=seconds)] selected GPU $GPU_ID (free=$candidate_free MiB, util=$candidate_util%)"

RUN_TAG=$(date +%Y%m%d-%H%M%S)
OUTPUT="$SPGFS_HOTPOT_OUTPUT"
[[ -n "$OUTPUT" ]] || OUTPUT="$ROOT/state/formal-eval/hotpot-flowsteer-aligned-no-skill-deepseek-c20-$RUN_TAG-qwen-thinking"
mkdir -p "$OUTPUT/logs"

# The config's logical role declaration includes 0,1,2.  Expose all physical
# IDs for validation while CUDA_VISIBLE_DEVICES still isolates the local server.
export SPGFS_ALLOWED_PHYSICAL_GPUS=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="$GPU_ID"
. scripts/formal/environment.sh

PYTHON="$SPGFS_PYTHON"
[[ -n "$PYTHON" ]] || PYTHON="$SPGFS_VENV/bin/python"
if [[ -n "$PYTHONPATH" ]]; then
  export PYTHONPATH="$ROOT/src:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT/src"
fi
test -x "$PYTHON"
test -d "$MODEL"
test -s "$DATASET"

PORT=""
for try_port in "$BASE_PORT" 18613 18614 18615 18616 18617 18618 18619; do
  if ! (echo >/dev/tcp/127.0.0.1/"$try_port") 2>/dev/null; then
    PORT="$try_port"
    break
  fi
done
if [[ -z "$PORT" ]]; then
  echo "No free Director port found" >&2
  exit 5
fi
DIRECTOR_URL="http://127.0.0.1:$PORT/v1"
export ROOT GPU_ID MODEL PORT DIRECTOR_URL DATASET OUTPUT WORKERS MAX_MODEL_LEN

MONITOR_PID=""
DIRECTOR_PID=""
cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "$DIRECTOR_PID" ]] && kill -0 "$DIRECTOR_PID" 2>/dev/null; then
    kill "$DIRECTOR_PID" 2>/dev/null || true
    wait "$DIRECTOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

nvidia-smi --id="$GPU_ID" \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu \
  --format=csv --loop-ms=5000 >"$OUTPUT/logs/gpu.csv" 2>&1 &
MONITOR_PID=$!

echo "[$(date --iso-8601=seconds)] starting Qwen Director on GPU $GPU_ID, port $PORT"
"$PYTHON" -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$WORKERS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 --generation-config auto --trust-remote-code \
  >"$OUTPUT/logs/director.log" 2>&1 &
DIRECTOR_PID=$!

ready=0
for _ in $(seq 1 1800); do
  if curl --noproxy '*' --fail --silent --show-error \
      -H 'Authorization: Bearer EMPTY' "$DIRECTOR_URL/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$DIRECTOR_PID" 2>/dev/null; then
    echo "Qwen Director exited during startup; see $OUTPUT/logs/director.log" >&2
    exit 6
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "Qwen Director did not become ready within 1800 seconds" >&2
  exit 7
fi
echo "[$(date --iso-8601=seconds)] Qwen Director ready; starting HotpotQA benchmark"

set +e
"$PYTHON" scripts/formal/wandb_direct_exec.py -- \
  "$PYTHON" -m selfplay_graph_flowsteer benchmark \
  --config configs/formal_training.toml \
  --dataset "$DATASET" --output "$OUTPUT" \
  --workers "$WORKERS" --verifier token_f1 \
  --historical-duration-priority --wandb-mode offline \
  --director-base-url "$DIRECTOR_URL" --director-api-key EMPTY \
  --director-model Qwen3.5-9B --director-thinking \
  --worker-route deepseek --disable-swe --skill-context off \
  >"$OUTPUT/logs/benchmark.log" 2>&1
BENCHMARK_RC=$?
set -e
cat "$OUTPUT/logs/benchmark.log"
echo "[$(date --iso-8601=seconds)] benchmark finished rc=$BENCHMARK_RC; output=$OUTPUT"
exit "$BENCHMARK_RC"
