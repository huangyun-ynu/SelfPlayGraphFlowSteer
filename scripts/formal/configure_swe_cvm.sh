#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

read -rsp "Tencent SecretId: " secret_id
echo
read -rsp "Tencent SecretKey: " secret_key
echo

export SPGFS_CONFIG_SECRET_ID="$secret_id"
export SPGFS_CONFIG_SECRET_KEY="$secret_key"

python3 - <<'PY'
from pathlib import Path
import os

path = Path(".env")
old = path.read_text(encoding="utf-8") if path.exists() else ""
updates = {
    "TENCENTCLOUD_SECRET_ID": os.environ["SPGFS_CONFIG_SECRET_ID"],
    "TENCENTCLOUD_SECRET_KEY": os.environ["SPGFS_CONFIG_SECRET_KEY"],
    "SPGFS_SWE_CVM_AUTO_START": "1",
    "SPGFS_SWE_CVM_AUTO_STOP": "1",
}
lines = [
    line for line in old.splitlines()
    if not any(line.startswith(name + "=") for name in updates)
]
lines.extend(f"{name}={value}" for name, value in updates.items())
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

unset SPGFS_CONFIG_SECRET_ID SPGFS_CONFIG_SECRET_KEY
echo "SWE CVM 自动开关机配置已写入 .env"
