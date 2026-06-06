#!/usr/bin/env bash
# Provision two v6e-8 VMs: target + draft (16 chips total).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZONE="${1:-${TPU_ZONE:-us-east5-a}}"
TARGET_VM="${TARGET_VM_NAME:-ssd-tpu-target-8-vm}"
DRAFT_VM="${DRAFT_VM_NAME:-ssd-tpu-draft-8-vm}"

echo "=== Dual 8-chip: $TARGET_VM + $DRAFT_VM in $ZONE ==="

"$ROOT/scripts/provision_tpu.sh" v6e "$ZONE" 8 "$TARGET_VM"
"$ROOT/scripts/provision_tpu.sh" v6e "$ZONE" 8 "$DRAFT_VM"

touch .env
for kv in "TARGET_VM_NAME=$TARGET_VM" "DRAFT_VM_NAME=$DRAFT_VM" "TPU_TOPOLOGY=dual8" "MODEL_PROFILE=sd-pair-7b-dual8"; do
  k="${kv%%=*}"; v="${kv#*=}"
  if grep -q "^${k}=" .env 2>/dev/null; then sed -i "s|^${k}=.*|${k}=${v}|" .env; else echo "${k}=${v}" >> .env; fi
done

echo "Set SSD_TPU_ROLE=target on $TARGET_VM and SSD_TPU_ROLE=draft on $DRAFT_VM"
