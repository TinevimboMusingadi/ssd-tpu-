#!/usr/bin/env bash
# Create GCS bucket and grant compute SA access for TPU VM model storage.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT in .env}"
ZONE="${TPU_ZONE:-us-east5-a}"
REGION="${GCS_REGION:-${ZONE%-*}}"
BUCKET="${GCS_BUCKET:-gs://${PROJECT}-ssd-tpu}"

# strip gs:// prefix for gcloud
BUCKET_NAME="${BUCKET#gs://}"

echo "=== Setup GCS ==="
echo "Project: $PROJECT"
echo "Region:  $REGION"
echo "Bucket:  gs://$BUCKET_NAME"

if gcloud storage buckets describe "gs://$BUCKET_NAME" --project="$PROJECT" >/dev/null 2>&1; then
  echo "Bucket already exists."
else
  gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project="$PROJECT" \
    --location="$REGION"
  echo "Created bucket."
fi

PROJECT_NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
  --member="serviceAccount:${SA}" \
  --role="roles/storage.objectAdmin" \
  --quiet

touch .env
_update() {
  local k="$1" v="$2"
  if grep -q "^${k}=" .env 2>/dev/null; then
    sed -i "s|^${k}=.*|${k}=${v}|" .env
  else
    echo "${k}=${v}" >> .env
  fi
}

_update "GCS_BUCKET" "gs://$BUCKET_NAME"
_update "GCS_MODEL_PREFIX" "${GCS_MODEL_PREFIX:-models}"
_update "TARGET_MODEL_PATH" "gs://$BUCKET_NAME/models/google_gemma-7b-it"
_update "DRAFT_MODEL_PATH" "gs://$BUCKET_NAME/models/google_gemma-2b-it"
_update "MODEL_PROFILE" "${MODEL_PROFILE:-sd-pair-7b}"

echo "GCS ready. Compute SA $SA has objectAdmin on gs://$BUCKET_NAME"
echo ".env updated with GCS_BUCKET and model paths"
