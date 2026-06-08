#!/usr/bin/env bash
set -euo pipefail

cd ~/ssd-tpu-

if ! command -v python3.11 >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

if [[ ! -d .venv311 ]]; then
  python3.11 -m venv .venv311
fi
source .venv311/bin/activate
python -m pip install -U pip wheel

# JAX TPU + libtpu (JAX 0.9.x needs libtpu >=0.0.40 for Mosaic v11)
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -U "libtpu>=0.0.40" -f https://storage.googleapis.com/libtpu-releases/index.html

# Project + EasyDeL main (Gemma 4 + transformers 5)
python -m pip install -e ".[tpu]"
python -m pip install "git+https://github.com/erfanzar/EasyDeL.git" pillow

echo "OK: $(python --version), easydel $(python -c 'import easydel; print(easydel.__version__)')"
