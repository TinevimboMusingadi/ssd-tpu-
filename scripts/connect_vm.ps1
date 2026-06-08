# SSH into the TPU VM from Windows (reads .env in repo root).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $Root ".env"

$Project = "tpu-builder1"
$Zone = "us-east5-b"
$Vm = "ssd-tpu-v6e-4-vm"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $k = $Matches[1].Trim()
            $v = $Matches[2].Trim()
            switch ($k) {
                "GCP_PROJECT" { $Project = $v }
                "TPU_ZONE"    { $Zone = $v }
                "TPU_VM_NAME" { $Vm = $v }
            }
        }
    }
}

Write-Host "Connecting: $Vm ($Zone, $Project)"
& gcloud compute ssh $Vm --zone=$Zone --project=$Project @args
