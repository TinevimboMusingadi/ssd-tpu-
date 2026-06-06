#!/usr/bin/env bash
# Recover from a failed install (e.g. disk full from torch). Run on TPU VM.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Disk before cleanup ==="
df -h /

pip cache purge 2>/dev/null || python3 -m pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

# Remove broken venv if install failed mid-way
if [ -d .venv ]; then
  echo "Removing .venv for clean reinstall..."
  rm -rf .venv
fi

echo "=== Disk after cleanup ==="
df -h /

chmod +x scripts/setup_tpu_vm.sh
./scripts/setup_tpu_vm.sh
