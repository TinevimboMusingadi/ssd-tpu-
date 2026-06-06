#!/usr/bin/env bash
# Master bootstrap for TPU VM — JAX, deps, GCS model sync, diagnostics, tests.
# Usage: ./scripts/bootstrap_vm.sh [--profile sd-pair-7b] [--skip-models] [--skip-real-test]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="sd-pair-7b"
SKIP_MODELS=0
SKIP_REAL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-real-test) SKIP_REAL=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"
export MODEL_PROFILE="${MODEL_PROFILE:-$PROFILE}"
export SSD_TPU_ROLE="${SSD_TPU_ROLE:-both}"
echo "TPU role: $SSD_TPU_ROLE"

echo "=== SSD-TPU Bootstrap ==="
echo "Profile: $MODEL_PROFILE"
echo "JAX_PLATFORMS: $JAX_PLATFORMS"
df -h / | tail -1

if ! command -v python3 >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y python3 python3-pip python3-venv
fi

PY=python3
if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
PY=python
PIP="$PY -m pip"

$PIP cache purge 2>/dev/null || true
$PIP install -U pip
$PIP install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
$PIP install -e .
$PIP install -e ".[dev]"

if [ "$SKIP_MODELS" -eq 0 ] && [ -n "${GCS_BUCKET:-}" ]; then
  echo "=== Syncing models from GCS ==="
  $PY - <<'PY' || true
import os
from connect.gcs_storage import is_gcs_path, sync_gcs_to_local
for key in ("TARGET_MODEL_PATH", "DRAFT_MODEL_PATH"):
    path = os.getenv(key)
    if path and is_gcs_path(path):
        try:
            local = sync_gcs_to_local(path)
            print(f"{key} -> {local}")
        except Exception as e:
            print(f"WARN: could not sync {key}: {e}")
PY
fi

echo "=== Diagnostics ==="
$PY -m connect.diagnostics

echo "=== Unit tests (toy) ==="
$PY -m pytest tests/ -q -m "not tpu"

if [ "$SKIP_REAL" -eq 0 ] && [ "${SSD_USE_TOY_MODEL:-0}" != "1" ]; then
  if $PY -c "import jax; assert any('tpu' in str(d).lower() for d in jax.devices())" 2>/dev/null; then
    echo "=== Real model smoke ==="
    export SSD_USE_REAL_MODEL=1
    export SSD_USE_TOY_MODEL=0
    $PY -m pytest tests/test_real_gemma_tpu.py -q -m tpu 2>/dev/null || \
      $PY -m jax_ssd.benchmarks.stream_prompt \
        --mode ar --prompt "Hello" --max-tokens 8 || true
  fi
fi

echo "=== Bootstrap complete ==="
