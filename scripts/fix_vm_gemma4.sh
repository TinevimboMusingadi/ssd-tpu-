#!/usr/bin/env bash
# One-shot repair: sync repo, fix venv, auth HF, download Gemma 4 weights.
set -euo pipefail

cd ~/ssd-tpu-

echo "=== sync repo from GitHub ==="
git fetch origin
git reset --hard origin/main

echo "=== fix Python 3.11 venv ==="
bash scripts/setup_py311_easydel.sh

source .venv311/bin/activate
sed -i 's/\r$//' .env 2>/dev/null || true
set -a && source .env && set +a

export JAX_PLATFORMS=tpu
export ENABLE_DISTRIBUTED_INIT=0
export EASYDEL_AUTO=1
export SSD_USE_TOY_MODEL=0

echo "=== HF auth + download Gemma 4 JAX weights ==="
hf auth login --token "$HF_TOKEN"
hf download EasyDeL/gemma-4-E2B-it

echo "=== versions ==="
pip show jax jaxlib libtpu easydel transformers | grep -E '^Name|^Version'

echo "=== smoke (first compile ~2 min) ==="
python scripts/gemma4_easydel_smoke.py "What is gravity in one sentence?"

echo "DONE"
