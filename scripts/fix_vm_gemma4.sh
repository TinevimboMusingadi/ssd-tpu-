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

# Strip BOM/whitespace from token (common .env issue on Windows-edited files)
HF_TOKEN="$(printf '%s' "${HF_TOKEN:-}" | tr -d '\r\n\ufeff' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
export HF_TOKEN

export JAX_PLATFORMS=tpu
export ENABLE_DISTRIBUTED_INIT=0
export EASYDEL_AUTO=1
export SSD_USE_TOY_MODEL=0
export SSD_TPU_ROLE=target

echo "=== HF auth + download Gemma 4 JAX weights ==="
if [[ -z "${HF_TOKEN}" ]]; then
  echo "ERROR: HF_TOKEN is empty in .env"
  echo "Create a Read token at https://huggingface.co/settings/tokens and set HF_TOKEN=hf_..."
  exit 1
fi

if ! hf auth whoami --token "$HF_TOKEN" >/dev/null 2>&1; then
  echo "ERROR: HF_TOKEN in .env is invalid or expired."
  echo "  1. Open https://huggingface.co/settings/tokens"
  echo "  2. Create a new token (type: Read)"
  echo "  3. Accept Gemma license: https://huggingface.co/google/gemma-4-E2B-it"
  echo "  4. Update HF_TOKEN in ~/ssd-tpu-/.env (no quotes, no spaces)"
  echo "  5. Re-run: bash scripts/fix_vm_gemma4.sh"
  exit 1
fi

hf auth login --token "$HF_TOKEN"

# Drop stale partial downloads from unauthenticated fetches
rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--EasyDeL--gemma-4-E2B-it"

hf download EasyDeL/gemma-4-E2B-it

echo "=== versions ==="
pip show jax jaxlib libtpu easydel transformers | grep -E '^Name|^Version'

echo "=== smoke (first compile ~2 min) ==="
python scripts/gemma4_easydel_smoke.py "What is gravity in one sentence?"

echo "DONE"
