# Create GCS bucket and grant compute SA access.
param(
    [string]$Region = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

if (-not $GCP_PROJECT) { throw "GCP_PROJECT missing in .env" }

if (-not $Region) {
    $zone = if ($TPU_ZONE) { $TPU_ZONE } else { "us-east5-a" }
    $Region = $zone.Substring(0, $zone.LastIndexOf("-"))
}

$bucketName = if ($GCS_BUCKET) { $GCS_BUCKET.Replace("gs://", "") } else { "$GCP_PROJECT-ssd-tpu" }
$bucketUri = "gs://$bucketName"

Write-Host "=== Setup GCS ==="
Write-Host "Project: $GCP_PROJECT"
Write-Host "Region:  $Region"
Write-Host "Bucket:  $bucketUri"

gcloud config set project $GCP_PROJECT | Out-Null

$exists = $false
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = gcloud storage buckets describe $bucketUri 2>&1
$exists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prev

if (-not $exists) {
    gcloud storage buckets create $bucketUri --location=$Region
    Write-Host "Created bucket."
} else {
    Write-Host "Bucket already exists."
}

$projectNum = (gcloud projects describe $GCP_PROJECT --format="value(projectNumber)").Trim()
$sa = "${projectNum}-compute@developer.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding $bucketUri `
    --member="serviceAccount:$sa" `
    --role="roles/storage.objectAdmin" `
    --quiet

& (Join-Path $PSScriptRoot "update_env.ps1") -Key "GCS_BUCKET" -Value $bucketUri
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "GCS_MODEL_PREFIX" -Value "models"
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TARGET_MODEL_PATH" -Value "$bucketUri/models/google_gemma-7b-it"
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "DRAFT_MODEL_PATH" -Value "$bucketUri/models/google_gemma-2b-it"
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "MODEL_PROFILE" -Value "sd-pair-7b"

Write-Host "GCS ready. Compute SA $sa has objectAdmin on $bucketUri"
