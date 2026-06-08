#!/usr/bin/env bash
# One-time: copy EasyDeL Gemma4 HF cache to GCS so future VMs boot fast.
set -euo pipefail

cd ~/ssd-tpu-
set -a && source .env && set +a

BUCKET="${GCS_BUCKET:?Set GCS_BUCKET in .env}"
PREFIX="${GCS_MODEL_PREFIX:-models}"
DEST="${BUCKET%/}/${PREFIX}/EasyDeL_gemma-4-E2B-it-hf-cache"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SRC="${HF_HOME}/hub/models--EasyDeL--gemma-4-E2B-it"
if [ ! -d "$SRC" ]; then
  echo "ERROR: missing $SRC — run: hf download EasyDeL/gemma-4-E2B-it"
  exit 1
fi

python scripts/verify_gemma4_weights.py || echo "WARN: weights check failed; uploading anyway"

echo "=== uploading $(du -sh "$SRC" | cut -f1) to $DEST ==="
gcloud storage rsync -r "$SRC" "$DEST"

echo "DONE. Set on new VMs:"
echo "  GEMMA4_GCS_CACHE=$DEST"
echo "  bash scripts/restore_gemma4_from_gcs.sh"
