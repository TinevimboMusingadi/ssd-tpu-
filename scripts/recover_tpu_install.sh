#!/usr/bin/env bash
# Recover from failed install (disk full, broken venv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Disk before cleanup ==="
df -h /

pip cache purge 2>/dev/null || python3 -m pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

if [ -d .venv ]; then
  echo "Removing .venv for clean reinstall..."
  rm -rf .venv
fi

echo "=== Disk after cleanup ==="
df -h /

chmod +x scripts/bootstrap_vm.sh
./scripts/bootstrap_vm.sh "$@"
