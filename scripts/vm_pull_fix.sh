#!/usr/bin/env bash
# Run on TPU VM when git pull fails or mesh/TUI fixes are missing.
set -euo pipefail
cd ~/ssd-tpu-

echo "=== Stash local edits and pull ==="
git stash push -m "vm-local" 2>/dev/null || true
git pull origin main

echo "=== Reinstall ==="
source .venv/bin/activate
export JAX_PLATFORMS=tpu
pip install "textual>=0.58.0,<0.80.0"
pip install -e .

echo "=== Doctor ==="
python -m connect.diagnostics

echo "=== Tests ==="
pytest tests/ -q

echo "DONE"
