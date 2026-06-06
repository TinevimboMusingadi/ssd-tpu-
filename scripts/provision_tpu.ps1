# Provision TPU VM from .env and write TPU_SSH_HOST / TPU_SSH_USER back to .env
# Usage: .\scripts\provision_tpu.ps1 [-Family v6e] [-VmName ssd-tpu-v6e-vm]
param(
    [ValidateSet("v6e", "v5p")]
    [string]$Family = "v6e",
    [string]$VmName = "ssd-tpu-v6e-vm"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

if (-not $GCP_PROJECT) { throw "GCP_PROJECT missing in .env" }
if (-not $TPU_ZONE) { throw "TPU_ZONE missing in .env" }

if ($Family -eq "v5p") {
    $MachineType = "ct5p-hightpu-4t"
    if ($VmName -eq "ssd-tpu-v6e-vm") { $VmName = "ssd-tpu-v5p-vm" }
} else {
    $MachineType = "ct6e-standard-4t"
}

Write-Host "=== Provision TPU VM ==="
Write-Host "Project: $GCP_PROJECT"
Write-Host "Zone:    $TPU_ZONE"
Write-Host "VM:      $VmName ($Family / $MachineType)"
Write-Host ""
Write-Host "Note: request-valid-for-duration max is 2h for FLEX_START."

gcloud config set project $GCP_PROJECT | Out-Null

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = gcloud compute instances describe $VmName --zone=$TPU_ZONE --project=$GCP_PROJECT 2>&1
$vmExists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

if ($vmExists) {
    Write-Host "VM '$VmName' already exists in $TPU_ZONE - skipping create."
} else {
    gcloud compute instances create $VmName `
        --project=$GCP_PROJECT `
        --zone=$TPU_ZONE `
        --machine-type=$MachineType `
        --provisioning-model=FLEX_START `
        --request-valid-for-duration=2h `
        --max-run-duration=4h `
        --instance-termination-action=DELETE `
        --image-project=ubuntu-os-accelerator-images `
        --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e `
        --maintenance-policy=TERMINATE
    if ($LASTEXITCODE -ne 0) { throw "gcloud create failed (exit $LASTEXITCODE)" }
}

Write-Host "Waiting for VM external IP..."
$ip = ""
for ($i = 0; $i -lt 30; $i++) {
    $ip = gcloud compute instances describe $VmName `
        --zone=$TPU_ZONE `
        --project=$GCP_PROJECT `
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
    if ($ip) { break }
    Start-Sleep -Seconds 5
}
if (-not $ip) { throw "Could not get external IP for $VmName" }

$account = (gcloud config get-value account 2>$null).ToString().Trim()
$sshUser = $env:USERNAME
if ($account -and $account.Contains("@")) {
    $sshUser = $account.Split("@")[0]
}

Write-Host "External IP: $ip"
Write-Host "SSH user:    $sshUser"

& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TPU_SSH_HOST" -Value $ip
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TPU_SSH_USER" -Value $sshUser
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TPU_VM_NAME" -Value $VmName

Write-Host ""
Write-Host "=== Next steps ==="
Write-Host "1. SSH:  gcloud compute ssh $VmName --zone=$TPU_ZONE --project=$GCP_PROJECT"
Write-Host "2. Or:   ssh ${sshUser}@${ip}"
Write-Host "3. Test: python -m connect.diagnostics   (on VM after setup)"
Write-Host ""
Write-Host ".env updated with TPU_SSH_HOST and TPU_SSH_USER"
