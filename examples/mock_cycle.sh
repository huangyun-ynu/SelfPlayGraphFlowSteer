#!/usr/bin/env bash
# Run after installing this checkout; all policy backends and updates are mocked.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="${1:-state/mock-cycle}"
if [[ -e "$run_dir" ]]; then
  echo "Choose a new output directory: $run_dir already exists." >&2
  exit 1
fi
spgfs selfplay-rollout \
  --config configs/mock.toml \
  --mock --seed demo --rollouts 5 --verifier none \
  --output "$run_dir"
spgfs train-cycle \
  --config configs/mock.toml \
  --run-dir "$run_dir" --mock-trainer
