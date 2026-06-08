#!/usr/bin/env bash
# Full Gemma4 pipeline for unattended VM runs. Logs to ~/ssd-tpu-/logs/
set -euo pipefail

cd ~/ssd-tpu-
mkdir -p logs
LOG="logs/overnight-$(date -u +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== overnight Gemma4 started $(date -u) ==="
echo "log: $PWD/$LOG"

git fetch origin
git reset --hard origin/main
chmod +x scripts/*.sh 2>/dev/null || true

bash scripts/stop_gemma4_vm.sh || true

if [[ -d .venv311 ]]; then
  source .venv311/bin/activate
else
  bash scripts/setup_py311_easydel.sh
  source .venv311/bin/activate
fi

sed -i 's/\r$//' .env 2>/dev/null || true
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
set -a && source .env && set +a
export JAX_PLATFORMS=tpu
export ENABLE_DISTRIBUTED_INIT=0
export EASYDEL_AUTO=1
export SSD_USE_TOY_MODEL=0
export SSD_TPU_ROLE=target

python scripts/diagnose_hf_token.py || {
  echo "HF token bad — fix .env on PC and run: py scripts/push_hf_token.py"
  exit 1
}

bash scripts/restore_gemma4_from_gcs.sh 2>/dev/null || true

if ! python scripts/verify_gemma4_weights.py; then
  echo "=== downloading / refreshing EasyDeL weights ==="
  hf download EasyDeL/gemma-4-E2B-it
  python scripts/verify_gemma4_weights.py || true
fi

echo "=== smoke test ==="
python scripts/gemma4_easydel_smoke.py "What is gravity in one sentence?"

echo "=== AR benchmark ==="
bash scripts/run_gemma4_vm.sh "Explain Newton's second law in plain English." 32

echo "=== overnight Gemma4 finished OK $(date -u) ==="
echo "Full log: $PWD/$LOG"
