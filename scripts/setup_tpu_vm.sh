#!/usr/bin/env bash
# Install JAX TPU support and project deps on a TPU VM.
set -euo pipefail

pip install -U pip
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e ".[dev]"

python -m connect.diagnostics
