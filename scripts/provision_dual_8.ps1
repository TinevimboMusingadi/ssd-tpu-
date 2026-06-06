# Provision TWO v6e-8 VMs: target (8 chips for Gemma 7B) + draft (8 chips for Gemma 2B).
# Usage: .\scripts\provision_dual_8.ps1 [-Zone us-east5-a]
param(
    [string]$Zone = "",
    [string]$TargetVm = "ssd-tpu-target-8-vm",
    [string]$DraftVm = "ssd-tpu-draft-8-vm"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

if ($Zone) { $TPU_ZONE = $Zone }
if (-not $GCP_PROJECT) { throw "GCP_PROJECT missing in .env" }
if (-not $TPU_ZONE) { throw "TPU_ZONE missing in .env" }

Write-Host "=== Provision dual 8-chip VMs (16 chips total) ==="
Write-Host "Target VM: $TargetVm  (all 8 chips -> Gemma 7B)"
Write-Host "Draft VM:  $DraftVm   (all 8 chips -> Gemma 2B)"
Write-Host "Zone:      $TPU_ZONE"
Write-Host ""

& (Join-Path $PSScriptRoot "provision_tpu.ps1") `
    -ChipCount 8 -VmName $TargetVm -Zone $TPU_ZONE -MaxRunHours 4

& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TARGET_VM_NAME" -Value $TargetVm
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "DRAFT_VM_NAME" -Value $DraftVm
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "TPU_TOPOLOGY" -Value "dual8"
& (Join-Path $PSScriptRoot "update_env.ps1") -Key "MODEL_PROFILE" -Value "sd-pair-7b-dual8"

Write-Host ""
Write-Host "Provisioning draft VM..."
& (Join-Path $PSScriptRoot "provision_tpu.ps1") `
    -ChipCount 8 -VmName $DraftVm -Zone $TPU_ZONE -MaxRunHours 4

Write-Host ""
Write-Host "=== Dual 8-chip VMs ready ==="
Write-Host "Target VM SSH: gcloud compute ssh $TargetVm --zone=$TPU_ZONE --project=$GCP_PROJECT"
Write-Host "Draft VM SSH:  gcloud compute ssh $DraftVm --zone=$TPU_ZONE --project=$GCP_PROJECT"
Write-Host ""
Write-Host "On TARGET VM .env set:  SSD_TPU_ROLE=target"
Write-Host "On DRAFT VM .env set:    SSD_TPU_ROLE=draft"
Write-Host "Run bootstrap on each VM after git clone."
