ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export SPGFS_CUDA_COMPAT_ROOT="${SPGFS_CUDA_COMPAT_ROOT:-$ROOT/state/cuda-compat-13-0/usr/local/cuda-13.0/compat}"
if [[ -d "$SPGFS_CUDA_COMPAT_ROOT" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$SPGFS_CUDA_COMPAT_ROOT:"*) ;;
    *) export LD_LIBRARY_PATH="$SPGFS_CUDA_COMPAT_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

export SPGFS_ALLOWED_PHYSICAL_GPUS="${SPGFS_ALLOWED_PHYSICAL_GPUS:-1}"
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
export SPGFS_VENV="${SPGFS_VENV:-$ROOT/../.venvs/spgfs-pats-gpu}"
export PATH="$SPGFS_VENV/bin:$PATH"
export SPGFS_FORMAL_TASK_POOL="${SPGFS_FORMAL_TASK_POOL:-$ROOT/state/formal-data/validated_task_pool.jsonl}"
export SPGFS_RETRIEVAL_INDEX="${SPGFS_RETRIEVAL_INDEX:-$ROOT/state/formal-data/retrieval/nq_open_wikipedia.sqlite3}"
# Formal NQ-open runs use the pinned Search-R1 Wiki-18
# E5/FAISS corpus.  The online Wikipedia backend remains available only when
# explicitly requested for diagnostics.
export SPGFS_RETRIEVAL_BACKEND="${SPGFS_RETRIEVAL_BACKEND:-faiss}"
export SPGFS_RETRIEVAL_PORT="${SPGFS_RETRIEVAL_PORT:-18010}"
export SPGFS_WIKIPEDIA_CACHE="${SPGFS_WIKIPEDIA_CACHE:-$ROOT/state/retrieval/wikipedia-v1}"
export SPGFS_SEARCHR1_DATA="${SPGFS_SEARCHR1_DATA:-$ROOT/state/formal-data/retrieval/searchr1}"
export SPGFS_SEARCHR1_INDEX="${SPGFS_SEARCHR1_INDEX:-$SPGFS_SEARCHR1_DATA/e5_Flat.index}"
export SPGFS_SEARCHR1_CORPUS="${SPGFS_SEARCHR1_CORPUS:-$SPGFS_SEARCHR1_DATA/wiki-18.jsonl}"
export SPGFS_SEARCHR1_MODEL="${SPGFS_SEARCHR1_MODEL:-$ROOT/../models/e5-base-v2}"
export SPGFS_RETRIEVAL_PYTHON="${SPGFS_RETRIEVAL_PYTHON:-$ROOT/../.venvs/spgfs-pats-gpu/bin/python}"
export SPGFS_RETRIEVAL_DEPS="${SPGFS_RETRIEVAL_DEPS:-$ROOT/state/retrieval-deps}"
export SPGFS_RETRIEVAL_THREADS="${SPGFS_RETRIEVAL_THREADS:-4}"
export SPGFS_WEBSHOP_SOURCE_ROOT="${SPGFS_WEBSHOP_SOURCE_ROOT:-$ROOT/assets/webshop/source}"
export SPGFS_WEBSHOP_SOURCE_REVISION=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
export SPGFS_WEBSHOP_INTERPRETER="${SPGFS_WEBSHOP_INTERPRETER:-$ROOT/assets/webshop/venv/bin/python}"
export SPGFS_WEBSHOP_WORKER="${SPGFS_WEBSHOP_WORKER:-$ROOT/assets/webshop/private-evaluation/src/skillev_private/benchmarks/official_environment_worker.py}"
export SPGFS_WEBSHOP_STORE="${SPGFS_WEBSHOP_STORE:-$ROOT/assets/webshop/prepared/products.sqlite3}"
export SPGFS_WEBSHOP_GOALS="${SPGFS_WEBSHOP_GOALS:-$ROOT/assets/webshop/prepared/goals.jsonl}"
export SPGFS_WEBSHOP_PORT="${SPGFS_WEBSHOP_PORT:-18020}"
export SPGFS_WEBSHOP_INDEX=$SPGFS_WEBSHOP_SOURCE_ROOT/search_engine/indexes
export SPGFS_WEBSHOP_JAVA_HOME="${SPGFS_WEBSHOP_JAVA_HOME:-${JAVA_HOME:-$ROOT/assets/java/jdk-11.0.32.1+1}}"
