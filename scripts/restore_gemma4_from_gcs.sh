#!/usr/bin/env bash
# Restore EasyDeL Gemma4 weights from GCS into HF hub cache (skip HF re-download).
set -euo pipefail

cd ~/ssd-tpu-
set -a && source .env && set +a

BUCKET="${GCS_BUCKET:?Set GCS_BUCKET in .env}"
PREFIX="${GCS_MODEL_PREFIX:-models}"
SRC="${GEMMA4_GCS_CACHE:-${BUCKET%/}/${PREFIX}/EasyDeL_gemma-4-E2B-it-hf-cache}"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DEST="${HF_HOME}/hub/models--EasyDeL--gemma-4-E2B-it"
mkdir -p "$DEST"

if [ -d "$DEST/snapshots" ] && [ "$(find "$DEST/snapshots" -name config.json 2>/dev/null | head -1)" ]; then
  echo "OK: cache already present at $DEST"
  python scripts/verify_gemma4_weights.py
  exit 0
fi

echo "=== restoring $SRC -> $DEST ==="
gcloud storage rsync -r "$SRC" "$DEST"
python scripts/verify_gemma4_weights.py
