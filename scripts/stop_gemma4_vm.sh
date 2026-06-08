#!/usr/bin/env bash
# Free TPU + stop EasyDeL eSurge worker processes. Model weights stay on disk.
set -euo pipefail

echo "=== stopping Gemma4 / EasyDeL workers ==="
pkill -f 'easydel/workers/esurge' 2>/dev/null || true
pkill -f 'gemma4_easydel_smoke' 2>/dev/null || true
pkill -f 'jax_ssd.benchmarks.stream_prompt' 2>/dev/null || true
pkill -f 'scripts/which_model' 2>/dev/null || true
sleep 3

if pgrep -af 'easydel/workers/esurge' >/dev/null 2>&1; then
  echo "WARN: some eSurge workers still running; try: killall -9 -u \"$(whoami)\" python"
else
  echo "OK: no eSurge workers"
fi

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE="$HF_HOME/hub/models--EasyDeL--gemma-4-E2B-it"
if [ -d "$CACHE" ]; then
  echo "Model cache kept at: $CACHE"
  du -sh "$CACHE" 2>/dev/null || true
else
  echo "No EasyDeL gemma-4 cache yet (run hf download EasyDeL/gemma-4-E2B-it)"
fi

echo "Safe to start a new session — weights persist on this VM disk until you delete the VM."
