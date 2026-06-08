#!/usr/bin/env bash
# SSH into the TPU VM (reads GCP_PROJECT, TPU_ZONE, TPU_VM_NAME from .env).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a && source "$ENV_FILE" && set +a
fi

PROJECT="${GCP_PROJECT:-tpu-builder1}"
ZONE="${TPU_ZONE:-us-east5-b}"
VM="${TPU_VM_NAME:-ssd-tpu-v6e-4-vm}"

echo "Connecting: $VM ($ZONE, $PROJECT)"
exec gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" "$@"
