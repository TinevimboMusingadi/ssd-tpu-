#!/usr/bin/env bash
# Copy-paste friendly commands for the TPU VM. Run: bash scripts/vm_commands.sh
# Or copy sections below manually.

set -euo pipefail
cd ~/ssd-tpu-

echo "=== Activate venv ==="
source .venv/bin/activate
export JAX_PLATFORMS=tpu

echo "=== Pull latest code ==="
git pull

echo "=== Doctor (TPU health) ==="
python -m connect.diagnostics

echo "=== Tests ==="
pytest tests/ -q

echo "=== Benchmark (all modes) ==="
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode all --num-prompts 3

echo "=== Live TUI ==="
python -m tui.app --prompt "Refactor getUserById to fetchUserById"
