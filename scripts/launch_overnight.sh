#!/usr/bin/env bash
# Start overnight_gemma4.sh detached (safe for gcloud ssh --command).
set -euo pipefail
cd ~/ssd-tpu-
mkdir -p logs
PIDFILE=logs/overnight.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ALREADY_RUNNING pid=$(cat "$PIDFILE")"
  tail -n 10 logs/overnight-nohup.log 2>/dev/null || true
  exit 0
fi

git fetch origin -q && git reset --hard origin/main -q
chmod +x scripts/*.sh

setsid bash scripts/overnight_gemma4.sh >> logs/overnight-nohup.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "STARTED pid=$(cat "$PIDFILE")"
sleep 2
tail -n 15 logs/overnight-nohup.log 2>/dev/null || echo "log starting..."
