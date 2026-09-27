#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for the isolated, fixed-route NQ control.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$ROOT/scripts/formal/run_qa_best_recorded_eval.sh" nq_open "$@"
