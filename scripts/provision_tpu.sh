#!/usr/bin/env bash
# Provision TPU VM via FLEX_START.
# Usage: ./scripts/provision_tpu.sh [v6e|v5p] [zone] [chip_count] [vm_name]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

FAMILY="${1:-v6e}"
ZONE="${2:-${TPU_ZONE:-us-east5-a}}"
CHIPS="${3:-${TPU_SLICE_CHIPS:-16}}"
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT in .env}"

if [ -n "${4:-}" ]; then
  NAME="$4"
elif [ -n "${TPU_VM_NAME:-}" ]; then
  NAME="$TPU_VM_NAME"
elif [ "$FAMILY" = "v5p" ]; then
  NAME="ssd-tpu-v5p-${CHIPS}-vm"
else
  NAME="ssd-tpu-v6e-${CHIPS}-vm"
fi

_update_env() {
  local key="$1" val="$2"
  touch .env
  if grep -q "^${key}=" .env 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" .env
  else
    echo "${key}=${val}" >> .env
  fi
}

case "$FAMILY" in
  v6e)
    MACHINE="ct6e-standard-${CHIPS}t"
    gcloud compute instances create "$NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="$MACHINE" \
      --provisioning-model=FLEX_START \
      --request-valid-for-duration=2h \
      --max-run-duration=4h \
      --instance-termination-action=DELETE \
      --image-project=ubuntu-os-accelerator-images \
      --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
      --maintenance-policy=TERMINATE
    ;;
  v5p)
    MACHINE="ct5p-hightpu-${CHIPS}t"
    gcloud compute instances create "$NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="$MACHINE" \
      --provisioning-model=FLEX_START \
      --request-valid-for-duration=2h \
      --max-run-duration=4h \
      --instance-termination-action=DELETE \
      --image-project=ubuntu-os-accelerator-images \
      --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
      --maintenance-policy=TERMINATE
    ;;
  *)
    echo "Unknown family: $FAMILY (use v6e or v5p)"
    exit 1
    ;;
esac

echo "Waiting for external IP..."
IP=""
for _ in $(seq 1 30); do
  IP=$(gcloud compute instances describe "$NAME" \
    --zone="$ZONE" --project="$PROJECT" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || true)
  if [ -n "$IP" ]; then break; fi
  sleep 5
done

SSH_USER="${USER:-$(whoami)}"
_update_env "TPU_VM_NAME" "$NAME"
_update_env "TPU_ZONE" "$ZONE"
_update_env "TPU_SLICE_CHIPS" "$CHIPS"
if [ -n "$IP" ]; then
  _update_env "TPU_SSH_HOST" "$IP"
  _update_env "TPU_SSH_USER" "$SSH_USER"
fi

echo "Created: $NAME ($MACHINE) in $ZONE"
echo "IP: ${IP:-pending}"
echo ".env updated"
