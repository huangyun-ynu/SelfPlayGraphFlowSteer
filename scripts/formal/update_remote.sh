#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="${SPGFS_GIT_BRANCH:-feat/pats-skill-scaffold}"

cd "$ROOT"
git fetch origin "$BRANCH"
git merge --ff-only "origin/$BRANCH"
exec scripts/formal/bootstrap_remote.sh
