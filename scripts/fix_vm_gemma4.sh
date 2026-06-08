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

python scripts/diagnose_hf_token.py || {
  echo ""
  echo "VM .env is out of sync with your working token (Colab uses its own login)."
  echo "From your Windows PC (in this repo):  python scripts/push_hf_token.py"
  echo "Or:  gcloud compute scp .env ssd-tpu-v6e-4-vm:~/ssd-tpu-/.env --zone=us-east5-b --project=tpu-builder1"
  exit 1
}

unset HF_TOKEN
set -a && source .env && set +a
HF_TOKEN="$(printf '%s' "${HF_TOKEN:-}" | tr -d '\r\n\ufeff' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
export HF_TOKEN
hf auth login --token "$HF_TOKEN"

# Resume/complete download (skip rm — partial cache can block rm -rf on VM)
hf download EasyDeL/gemma-4-E2B-it

echo "=== versions ==="
pip show jax jaxlib libtpu easydel transformers | grep -E '^Name|^Version'

echo "=== smoke (first compile ~2 min) ==="
python scripts/gemma4_easydel_smoke.py "What is gravity in one sentence?"

echo "DONE"
