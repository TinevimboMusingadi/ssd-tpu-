#!/usr/bin/env bash
# Download default SD pair: Gemma-2.2B target + Gemma-2B draft (v6e-4 slice).
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

$PY scripts/download_models.py --preset sd-pair
