#!/usr/bin/env bash
# Run Gemma 4 E2B AR on TPU VM (Python 3.11 + EasyDeL from git).
set -euo pipefail
cd ~/ssd-tpu-

if [[ -d .venv311 ]]; then
  source .venv311/bin/activate
else
  source .venv/bin/activate
fi

sed -i 's/\r$//' .env 2>/dev/null || true
set -a
source .env
set +a

HF_TOKEN="$(printf '%s' "${HF_TOKEN:-}" | tr -d '\r\n\ufeff' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
export HF_TOKEN

export JAX_PLATFORMS=tpu
export EASYDEL_AUTO=1
export ENABLE_DISTRIBUTED_INIT=0
export SSD_USE_TOY_MODEL=0
# AR uses all 4 chips for target (avoid 3+1 SD split reshaping Gemma4)
export SSD_TPU_ROLE=target
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Optional: restore weights from GCS once (see scripts/snapshot_gemma4_to_gcs.sh)
if [[ -n "${GEMMA4_GCS_CACHE:-}" ]] || [[ -n "${GCS_BUCKET:-}" ]]; then
  bash scripts/restore_gemma4_from_gcs.sh 2>/dev/null || true
fi

python scripts/verify_gemma4_weights.py || true

python -m jax_ssd.benchmarks.stream_prompt \
  --mode ar \
  --prompt "${1:-What is gravity?}" \
  --max-tokens "${2:-24}"
