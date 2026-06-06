# Provision TPU VM from .env and write SSH settings back to .env
# Usage: .\scripts\provision_tpu.ps1 [-Family v6e] [-ChipCount 8] [-VmName ssd-tpu-v6e-8-vm]
# v6e in us-east5-a/b: ct6e-standard-1t, 4t, 8t only (no 16t).
param(
    [ValidateSet("v6e", "v5p")]
    [string]$Family = "v6e",
    [ValidateSet(1, 4, 8, 16, 32, 64, 128)]
    [int]$ChipCount = 8,
    [string]$VmName = "",
    [string]$Zone = "",
    [int]$MaxRunHours = 4
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

if (-not $GCP_PROJECT) { throw "GCP_PROJECT missing in .env" }
if ($Zone) { $TPU_ZONE = $Zone }
if (-not $TPU_ZONE) { throw "TPU_ZONE missing in .env (or pass -Zone)" }

if (-not $VmName) {
    if ($env:TPU_VM_NAME) {
        $VmName = $env:TPU_VM_NAME
    } elseif ($Family -eq "v5p") {
        $VmName = "ssd-tpu-v5p-${ChipCount}-vm"
    } else {
        $VmName = "ssd-tpu-v6e-${ChipCount}-vm"
    }
}

if ($Family -eq "v5p") {
    $MachineType = "ct5p-hightpu-${ChipCount}t"
} else {
    $V6eAllowed = @(1, 4, 8)
    if ($ChipCount -notin $V6eAllowed) {
        throw "v6e in $TPU_ZONE supports chip counts: $($V6eAllowed -join ', '). Got $ChipCount. Use -ChipCount 8 for Gemma 7B+2B."
    }
    $MachineType = "ct6e-standard-${ChipCount}t"
}

$maxRun = "${MaxRunHours}h"

Write-Host "=== Provision TPU VM ==="
Write-Host "Project:   $GCP_PROJECT"
Write-Host "Zone:      $TPU_ZONE"
Write-Host "VM:        $VmName"
Write-Host "Family:    $Family"
Write-Host "Chips:     $ChipCount"
Write-Host "Machine:   $MachineType"
Write-Host "Max run:   $maxRun"
Write-Host ""
Write-Host "Note: request-valid-for-duration max is 2h for FLEX_START."
Write-Host "v6e max per VM is 8 chips (no ct6e-standard-16t). Run .\scripts\list_tpu_capacity.ps1 for zones."

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
        --max-run-duration=$maxRun `
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
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TPU_SLICE_CHIPS" -Value "$ChipCount"

Write-Host ""
Write-Host "=== Next steps ==="
Write-Host "1. SSH:  gcloud compute ssh $VmName --zone=$TPU_ZONE --project=$GCP_PROJECT"
Write-Host "2. VM:    ./scripts/bootstrap_vm.sh --profile sd-pair-7b"
Write-Host ""
Write-Host ".env updated with TPU_SSH_HOST, TPU_VM_NAME, TPU_SLICE_CHIPS"
