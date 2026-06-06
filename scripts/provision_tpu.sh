#!/usr/bin/env bash
# Provision TPU VM via FLEX_START. Usage: ./scripts/provision_tpu.sh v6e|v5p|v5e [zone]
set -euo pipefail

FAMILY="${1:-v6e}"
ZONE="${2:-us-east5-a}"
NAME="ssd-tpu-${FAMILY}-$(date +%s)"

case "$FAMILY" in
  v6e)
    gcloud compute instances create "$NAME" \
      --zone="$ZONE" \
      --machine-type=ct6e-standard-4t \
      --provisioning-model=FLEX_START \
      --request-valid-for-duration=2h \
      --max-run-duration=4h \
      --instance-termination-action=DELETE \
      --image-project=ubuntu-os-accelerator-images \
      --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
      --maintenance-policy=TERMINATE
    ;;
  v5p)
    gcloud compute instances create "$NAME" \
      --zone="$ZONE" \
      --machine-type=ct5p-hightpu-4t \
      --provisioning-model=FLEX_START \
      --request-valid-for-duration=2h \
      --max-run-duration=4h \
      --instance-termination-action=DELETE \
      --image-project=ubuntu-os-accelerator-images \
      --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
      --maintenance-policy=TERMINATE
    ;;
  v5e)
    gcloud alpha compute tpus queued-resources create "$NAME" \
      --zone=us-west4-a \
      --accelerator-type=v5litepod-4 \
      --runtime-version=v2-alpha-tpuv5-lite \
      --node-id="${NAME}-node" \
      --provisioning-model=flex-start \
      --max-run-duration=4h \
      --valid-until-duration=4h
    ;;
  *)
    echo "Unknown family: $FAMILY (use v6e, v5p, or v5e)"
    exit 1
    ;;
esac

echo "Created: $NAME in $ZONE"
