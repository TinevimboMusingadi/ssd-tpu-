#!/usr/bin/env bash
# Download default SD pair for current MODEL_PROFILE (sd-pair-7b on v6e-16).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

PY=python3
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
fi

PRESET="${MODEL_PROFILE:-sd-pair-7b}"
GCS_URI="${GCS_BUCKET:-}"
ARGS=(--preset "$PRESET")
if [ -n "$GCS_URI" ]; then
  ARGS+=(--gcs-uri "$GCS_URI")
fi

$PY scripts/download_models.py "${ARGS[@]}"
