# Parse .env from repo root into PowerShell variables.
param(
    [string]$EnvFile = (Join-Path $PSScriptRoot "..\.env")
)

if (-not (Test-Path $EnvFile)) {
    throw "Missing .env at $EnvFile - copy .env.example first."
}

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $idx = $line.IndexOf("=")
    if ($idx -lt 1) { return }
    $key = $line.Substring(0, $idx).Trim()
    $val = $line.Substring($idx + 1).Trim()
    Set-Variable -Name $key -Value $val -Scope Script
    Set-Item -Path "env:$key" -Value $val
}

Write-Host "Loaded .env: GCP_PROJECT=$GCP_PROJECT TPU_ZONE=$TPU_ZONE"
