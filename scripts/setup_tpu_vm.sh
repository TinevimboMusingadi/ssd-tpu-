#!/usr/bin/env bash
# Install JAX TPU support and project deps on a TPU VM.
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

$PIP install -U pip
$PIP install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
$PIP install -e ".[dev]"

export JAX_PLATFORMS=tpu
python -m connect.diagnostics
