#!/usr/bin/env bash
# Delete a TPU VM. Usage: ./scripts/teardown_vm.sh [vm-name]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

VM_NAME="${1:-${TPU_VM_NAME:-ssd-tpu-v6e-vm}}"
ZONE="${TPU_ZONE:?Set TPU_ZONE in .env}"
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT in .env}"

echo "=== Teardown TPU VM ==="
echo "Project: $PROJECT"
echo "Zone:    $ZONE"
echo "VM:      $VM_NAME"
echo ""
echo "WARNING: Deletes VM and boot disk. GCS weights are kept."
read -r -p "Type VM name to confirm: " confirm
if [ "$confirm" != "$VM_NAME" ]; then
  echo "Aborted."
  exit 0
fi

gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
echo "Deleted $VM_NAME"
