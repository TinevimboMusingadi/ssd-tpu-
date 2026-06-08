#!/usr/bin/env bash
# Python 3.11 venv for Gemma 4 + EasyDeL git on TPU VM.
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

# Base project deps only (avoid pip easydel 0.2.x fighting git easydel 0.3.x)
python -m pip install -e .

# EasyDeL git (Gemma 4) + CPU torch for import hooks
python -m pip install "git+https://github.com/erfanzar/EasyDeL.git" pillow
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# JAX 0.9.2 + libtpu>=0.0.40 (ejkernel pins jax~=0.9; jax 0.10 breaks EasyDeL stack)
python -m pip install -U "jax[tpu]==0.9.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -U "libtpu>=0.0.40" -f https://storage.googleapis.com/libtpu-releases/index.html

export ENABLE_DISTRIBUTED_INIT=0
python -c "import easydel; print('OK:', __import__('sys').version, 'easydel', easydel.__version__)"
pip show jax jaxlib libtpu | grep -E '^Name|^Version'
