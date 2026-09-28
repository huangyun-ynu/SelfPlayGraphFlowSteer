#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer-hotpot-answer-contract
set -a
source /mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer/.env
set +a
export SPGFS_CUDA_COMPAT_ROOT=/mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer/state/cuda-compat-13-0/usr/local/cuda-13.0/compat
export SPGFS_ALLOWED_PHYSICAL_GPUS=1
export CUDA_VISIBLE_DEVICES=1
export SPGFS_RELEASE_PROJECT_GPU_RESERVATIONS=0
source scripts/formal/environment.sh
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
exec "$SPGFS_VENV/bin/python" /mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer-hotpot-answer-contract/state/hotpot128-aime30-c50-20260928-204627/run.py
