#!/usr/bin/env bash
# Install JAX TPU support and project deps on a TPU VM.
# Does NOT install PyTorch (saves ~2GB disk). Use .[parity] locally for JAX/torch parity tests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Installing python3..."
  sudo apt-get update -qq
  sudo apt-get install -y python3 python3-pip python3-venv
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "Installing python3-pip..."
  sudo apt-get update -qq
  sudo apt-get install -y python3-pip python3-venv
fi

PY=python3
PIP="$PY -m pip"

if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Freeing pip cache before install..."
$PIP cache purge 2>/dev/null || true

$PIP install -U pip
$PIP install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Core + pytest only — no torch/CUDA (TPU VM boot disk is ~10GB)
$PIP install -e .
$PIP install -e ".[dev]"

export JAX_PLATFORMS=tpu
python -m connect.diagnostics

if [ ! -d "./models/google_gemma-2b-it" ] && [ "${DOWNLOAD_MODELS:-0}" = "1" ]; then
  echo "Downloading Gemma-2B-IT weights (set HF_TOKEN in .env)..."
  python scripts/download_models.py --preset gemma-2b
fi
