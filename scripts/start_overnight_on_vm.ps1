# From Windows: push HF token, sync repo on VM, start overnight job in background.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== push HF_TOKEN from .env file ==="
py -3 scripts/push_hf_token.py

$EnvFile = Join-Path $Root ".env"
$Project = "tpu-builder1"
$Zone = "us-east5-b"
$Vm = "ssd-tpu-v6e-4-vm"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            switch ($Matches[1].Trim()) {
                "GCP_PROJECT" { $Project = $Matches[2].Trim() }
                "TPU_ZONE"    { $Zone = $Matches[2].Trim() }
                "TPU_VM_NAME" { $Vm = $Matches[2].Trim() }
            }
        }
    }
}

$Remote = @"
cd ~/ssd-tpu-
git fetch origin && git reset --hard origin/main
chmod +x scripts/*.sh
mkdir -p logs
if pgrep -f overnight_gemma4.sh >/dev/null; then
  echo 'Already running. tail -f logs/overnight-*.log'
  exit 0
fi
nohup bash scripts/overnight_gemma4.sh > logs/overnight-nohup.log 2>&1 &
echo "Started PID \$!"
echo "Watch: tail -f ~/ssd-tpu-/logs/overnight-*.log"
sleep 2
tail -n 20 logs/overnight-nohup.log 2>/dev/null || true
"@

Write-Host "=== starting overnight job on $Vm ==="
& gcloud compute ssh $Vm --zone=$Zone --project=$Project --command=$Remote

Write-Host ""
Write-Host "Job running on VM while you sleep."
Write-Host "  Connect:  .\scripts\connect_vm.ps1"
Write-Host "  Logs:     tail -f ~/ssd-tpu-/logs/overnight-*.log"
