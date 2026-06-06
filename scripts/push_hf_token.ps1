# Push HF_TOKEN from local .env to the TPU VM (does not print the token).
param(
    [string]$VmName = "ssd-tpu-v6e-vm",
    [string]$Zone = "us-east5-a",
    [string]$Project = "tpu-builder1"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvFile = Join-Path $Root ".env"

if (-not (Test-Path $EnvFile)) {
    Write-Error "Missing $EnvFile — add HF_TOKEN=hf_... first."
}

$tokenLine = Get-Content $EnvFile | Where-Object { $_ -match '^\s*HF_TOKEN=' } | Select-Object -First 1
if (-not $tokenLine) {
    Write-Error "HF_TOKEN not found in .env"
}

$token = ($tokenLine -replace '^\s*HF_TOKEN=', '').Trim()
if (-not $token.StartsWith("hf_")) {
    Write-Error "HF_TOKEN in .env should start with hf_"
}

$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($token))
$remote = @'
cd ~/ssd-tpu- && TOKEN=$(echo __B64__ | base64 -d) && touch .env && if grep -q '^HF_TOKEN=' .env; then sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$TOKEN|" .env; else echo "HF_TOKEN=$TOKEN" >> .env; fi && grep -q '^TARGET_MODEL_PATH=' .env || echo 'TARGET_MODEL_PATH=./models/google_gemma-2-2b-it' >> .env && grep -q '^DRAFT_MODEL_PATH=' .env || echo 'DRAFT_MODEL_PATH=./models/google_gemma-2b-it' >> .env && echo HF_TOKEN synced
'@
$remote = $remote.Replace('__B64__', $b64)

gcloud compute ssh $VmName --zone=$Zone --project=$Project --command=$remote
Write-Host "Done. On VM run: python scripts/download_models.py --preset gemma-2b"
