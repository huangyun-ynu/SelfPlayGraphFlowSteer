#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

if [[ ! -f .env ]]; then
  printf 'Missing %s/.env\n' "$ROOT" >&2
  exit 2
fi
set -a
source .env
set +a

GPU_ID=${SPGFS_SWE_SOTA_GPU_ID:?Set SPGFS_SWE_SOTA_GPU_ID to one exclusively available physical GPU}
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  printf 'SPGFS_SWE_SOTA_GPU_ID must be a non-negative integer\n' >&2
  exit 2
fi

GPU_UUID=$(nvidia-smi --id="$GPU_ID" --query-gpu=uuid --format=csv,noheader,nounits)
GPU_PIDS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
  | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}')
if [[ -n "$GPU_PIDS" ]]; then
  printf 'GPU %s is not exclusive; refusing to stop or share existing processes:\n%s\n' \
    "$GPU_ID" "$GPU_PIDS" >&2
  exit 3
fi

# Config validation describes the three training roles, but this fixed benchmark
# starts only the one explicitly selected Director GPU. No proposer/solver
# training service is launched here.
export SPGFS_ALLOWED_PHYSICAL_GPUS=${SPGFS_ALLOWED_PHYSICAL_GPUS:-0,1,2}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
source scripts/formal/environment.sh

PYTHON=${SPGFS_PYTHON:-$SPGFS_VENV/bin/python}
MODEL=${SPGFS_SWE_SOTA_MODEL:-$ROOT/models/Qwen3.5-9B}
PORT=${SPGFS_SWE_SOTA_PORT:-18603}
DIRECTOR_URL="http://127.0.0.1:$PORT/v1"
DATASET=${SPGFS_SWE_SOTA_DATASET:-$ROOT/data/formal/eval/swe_bench_verified_test_128.jsonl}
OUTPUT=${SPGFS_SWE_SOTA_OUTPUT:-$ROOT/state/sota-20260917/swebench-verified-qwen35-9b-reasoning-no-skill-v1}
WORKERS=${SPGFS_SWE_SOTA_WORKERS:-24}
WANDB_MODE=${SPGFS_SWE_SOTA_WANDB_MODE:-online}
MAX_MODEL_LEN=${SPGFS_SWE_SOTA_MAX_MODEL_LEN:-262144}

test -d "$MODEL"
test -s "$DATASET"
mkdir -p "$OUTPUT/logs"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=${SPGFS_SWE_SOTA_TMPDIR:-/tmp/spgfs-swe-sota}
mkdir -p "$TMPDIR"
export WANDB_NAME=${WANDB_NAME:-swebench-verified-qwen35-9b-reasoning-no-skill-20260917}

SWE_VERIFIER_LOG="$ROOT/state/formal-training/private/swe/verifier-client.jsonl"
SWE_LIFECYCLE_LOG="$ROOT/state/formal-training/private/swe/lifecycle.jsonl"
VERIFIER_LOG_OFFSET=$(stat -c %s "$SWE_VERIFIER_LOG" 2>/dev/null || printf 0)
LIFECYCLE_LOG_OFFSET=$(stat -c %s "$SWE_LIFECYCLE_LOG" 2>/dev/null || printf 0)

snapshot_swe_logs() {
  "$PYTHON" - "$SWE_VERIFIER_LOG" "$VERIFIER_LOG_OFFSET" \
    "$OUTPUT/logs/swe-verifier-client.jsonl" \
    "$SWE_LIFECYCLE_LOG" "$LIFECYCLE_LOG_OFFSET" \
    "$OUTPUT/logs/swe-lifecycle.jsonl" <<'PY'
import sys
from pathlib import Path

for source, offset, target in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3], strict=True):
    path = Path(source)
    data = b""
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(int(offset))
            data = stream.read()
    Path(target).write_bytes(data)
PY
}

DIRECTOR_PID=""
MONITOR_PID=""
cleanup() {
  snapshot_swe_logs || true
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

export ROOT GPU_ID GPU_UUID MODEL PORT DIRECTOR_URL DATASET OUTPUT WORKERS WANDB_MODE MAX_MODEL_LEN
export VERIFIER_LOG_OFFSET LIFECYCLE_LOG_OFFSET
"$PYTHON" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

def run(*args):
    return subprocess.run(args, capture_output=True, check=False, text=True).stdout.strip()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

root = Path(os.environ["ROOT"])
dataset = Path(os.environ["DATASET"])
manifest = {
    "schema_version": 1,
    "created_at": datetime.now(UTC).isoformat(),
    "scope": "SWE-bench Verified fixed test evaluation; no parameter updates",
    "dataset": str(dataset.resolve()),
    "dataset_sha256": sha256(dataset),
    "dataset_rows": sum(1 for line in dataset.open() if line.strip()),
    "director": {
        "model": "Qwen3.5-9B",
        "model_path": str(Path(os.environ["MODEL"]).resolve()),
        "thinking": True,
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
        "static_dataset_skill": False,
        "skill_context": "off",
    },
    "workers": {
        "logical_route": "gpt",
        "endpoint_pool": ["gpt", "gpt_eco"],
        "reasoning_effort": "low",
        "parallelism": int(os.environ["WORKERS"]),
    },
    "swe": {
        "remote_verifier": True,
        "synthetic": False,
        "dataset_revision": "78f471bf655a3137b2e8a75af1501690ec009ec3",
        "verifier_log_start_byte": int(os.environ["VERIFIER_LOG_OFFSET"]),
        "lifecycle_log_start_byte": int(os.environ["LIFECYCLE_LOG_OFFSET"]),
    },
    "gpu": {
        "physical_id": int(os.environ["GPU_ID"]),
        "uuid": os.environ["GPU_UUID"],
        "snapshot": run(
            "nvidia-smi", "--id=" + os.environ["GPU_ID"],
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader",
        ),
    },
    "wandb_mode": os.environ["WANDB_MODE"],
    "git": {
        "head": run("git", "-C", str(root), "rev-parse", "HEAD"),
        "status_porcelain": run("git", "-C", str(root), "status", "--porcelain=v1"),
        "diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "-C", str(root), "diff", "--binary"],
                capture_output=True,
                check=False,
            ).stdout
        ).hexdigest(),
    },
    "expected_outputs": [
        "records.jsonl", "summary.json", "trajectories/", "logs/director.log",
        "logs/benchmark.log", "logs/gpu.csv", "logs/swe-verifier-client.jsonl",
        "logs/swe-lifecycle.jsonl", "launch_manifest.json",
    ],
}
target = Path(os.environ["OUTPUT"]) / "launch_manifest.json"
target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
PY

nvidia-smi --id="$GPU_ID" \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu \
  --format=csv --loop-ms=5000 >"$OUTPUT/logs/gpu.csv" 2>&1 &
MONITOR_PID=$!

"$PYTHON" -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$WORKERS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 --generation-config auto --trust-remote-code \
  >"$OUTPUT/logs/director.log" 2>&1 &
DIRECTOR_PID=$!

for _ in $(seq 1 900); do
  if curl --noproxy '*' --fail --silent --show-error \
    -H 'Authorization: Bearer EMPTY' "$DIRECTOR_URL/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$DIRECTOR_PID" 2>/dev/null; then
    printf 'Qwen Director exited during startup; see %s/logs/director.log\n' "$OUTPUT" >&2
    exit 4
  fi
  sleep 1
done
curl --noproxy '*' --fail --silent --show-error \
  -H 'Authorization: Bearer EMPTY' "$DIRECTOR_URL/models" >/dev/null

set -o pipefail
"$PYTHON" scripts/formal/wandb_direct_exec.py -- \
  "$PYTHON" -m selfplay_graph_flowsteer benchmark \
  --config configs/formal_training.toml \
  --dataset "$DATASET" --output "$OUTPUT" \
  --workers "$WORKERS" --historical-duration-priority \
  --director-base-url "$DIRECTOR_URL" \
  --director-api-key EMPTY --director-model Qwen3.5-9B \
  --director-thinking --worker-route gpt --skill-context off \
  --wandb-mode "$WANDB_MODE" \
  2>&1 | tee "$OUTPUT/logs/benchmark.log"
