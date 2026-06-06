#!/usr/bin/env bash
# Download default Gemma-2B-IT weights for SSD-TPU (fits v6e-4 slice).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

PY=python3
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
fi

$PY scripts/download_models.py --preset gemma-2b
