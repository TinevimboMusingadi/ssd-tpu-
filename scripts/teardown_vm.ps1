# Delete a TPU VM instance. Usage: .\scripts\teardown_vm.ps1 [-VmName ssd-tpu-v6e-vm] [-Force]
param(
    [string]$VmName = "ssd-tpu-v6e-vm",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

if (-not $GCP_PROJECT) { throw "GCP_PROJECT missing in .env" }
if (-not $TPU_ZONE) { throw "TPU_ZONE missing in .env" }

if ($env:TPU_VM_NAME -and $VmName -eq "ssd-tpu-v6e-vm") {
    $VmName = $env:TPU_VM_NAME
}

Write-Host "=== Teardown TPU VM ==="
Write-Host "Project: $GCP_PROJECT"
Write-Host "Zone:    $TPU_ZONE"
Write-Host "VM:      $VmName"
Write-Host ""
Write-Host "WARNING: This deletes the VM and its boot disk. GCS weights are kept."
Write-Host "Billing stops when the instance is deleted."

if (-not $Force) {
    $confirm = Read-Host "Type the VM name to confirm deletion"
    if ($confirm -ne $VmName) {
        Write-Host "Aborted."
        exit 0
    }
}

gcloud config set project $GCP_PROJECT | Out-Null
gcloud compute instances delete $VmName --zone=$TPU_ZONE --project=$GCP_PROJECT --quiet
Write-Host "Deleted $VmName"
