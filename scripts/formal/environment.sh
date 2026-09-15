ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export SPGFS_ALLOWED_PHYSICAL_GPUS="${SPGFS_ALLOWED_PHYSICAL_GPUS:-0,1,2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$SPGFS_ALLOWED_PHYSICAL_GPUS}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# Fast Downward extracts a native library into TMPDIR and requires local,
# POSIX-consistent directory cleanup; the shared filesystem can race here.
export TMPDIR="${TMPDIR:-/tmp/spgfs-pats}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache/runtime}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ROOT/.cache/cuda}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="${SPGFS_VENV:-$ROOT/.venv}/bin:$PATH"
export SPGFS_FORMAL_TASK_POOL="${SPGFS_FORMAL_TASK_POOL:-$ROOT/state/formal-data/validated_task_pool.jsonl}"
export SPGFS_WEBSHOP_SOURCE_ROOT="${SPGFS_WEBSHOP_SOURCE_ROOT:-$ROOT/assets/webshop/source}"
export SPGFS_WEBSHOP_SOURCE_REVISION=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
export SPGFS_WEBSHOP_INTERPRETER="${SPGFS_WEBSHOP_INTERPRETER:-$ROOT/assets/webshop/venv/bin/python}"
export SPGFS_WEBSHOP_WORKER="${SPGFS_WEBSHOP_WORKER:-$ROOT/assets/webshop/private-evaluation/src/skillev_private/benchmarks/official_environment_worker.py}"
export SPGFS_WEBSHOP_STORE="${SPGFS_WEBSHOP_STORE:-$ROOT/assets/webshop/prepared/products.sqlite3}"
export SPGFS_WEBSHOP_GOALS="${SPGFS_WEBSHOP_GOALS:-$ROOT/assets/webshop/prepared/goals.jsonl}"
export SPGFS_WEBSHOP_INDEX=$SPGFS_WEBSHOP_SOURCE_ROOT/search_engine/indexes_100k
export SPGFS_WEBSHOP_JAVA_HOME="${SPGFS_WEBSHOP_JAVA_HOME:-${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}}"
