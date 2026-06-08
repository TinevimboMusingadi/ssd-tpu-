# Open SSH session to TPU VM using values from .env
param(
    [switch]$ConfigOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

if (-not $GCP_PROJECT) { throw "GCP_PROJECT missing in .env" }
if (-not $TPU_ZONE) { throw "TPU_ZONE missing in .env" }

$VmName = if ($TPU_VM_NAME) { $TPU_VM_NAME } else { "ssd-tpu-v6e-vm" }

if (-not $TPU_SSH_HOST) {
    Write-Host "TPU_SSH_HOST empty - fetching from gcloud..."
    $TPU_SSH_HOST = gcloud compute instances describe $VmName `
        --zone=$TPU_ZONE `
        --project=$GCP_PROJECT `
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
    if ($TPU_SSH_HOST) {
        & (Join-Path $PSScriptRoot "update_env.ps1") -Key "TPU_SSH_HOST" -Value $TPU_SSH_HOST
    }
}

$sshDir = Join-Path $env:USERPROFILE ".ssh"
$configPath = Join-Path $sshDir "config"
$hostBlock = @"

Host ssd-tpu
    HostName $TPU_SSH_HOST
    User $TPU_SSH_USER
    IdentityFile ~/.ssh/google_compute_engine
    StrictHostKeyChecking accept-new

"@

if (-not (Test-Path $configPath) -or -not (Select-String -Path $configPath -Pattern "Host ssd-tpu" -Quiet)) {
    New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
    Add-Content -Path $configPath -Value $hostBlock
    Write-Host "Added 'ssd-tpu' host to $configPath"
} else {
    Write-Host "SSH config already has 'ssd-tpu' entry"
}

if ($ConfigOnly) { exit 0 }

if (-not $TPU_SSH_HOST) {
    throw "No TPU VM IP. Run .\scripts\provision_tpu.ps1 first."
}

Write-Host "Connecting via gcloud compute ssh (recommended first time)..."
gcloud compute ssh $VmName --zone=$TPU_ZONE --project=$GCP_PROJECT
