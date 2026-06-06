# List v6e/v5p FLEX_START machine types available in your project.
# Usage: .\scripts\list_tpu_capacity.ps1 [-Family v6e]
param(
    [ValidateSet("v6e", "v5p", "all")]
    [string]$Family = "v6e"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "load_env.ps1")

$project = if ($GCP_PROJECT) { $GCP_PROJECT } else { (gcloud config get-value project 2>$null) }
if (-not $project) { throw "Set GCP_PROJECT in .env" }

Write-Host "=== TPU capacity for project: $project ==="
Write-Host ""

$filter = if ($Family -eq "v6e") { "name~ct6e-standard" } elseif ($Family -eq "v5p") { "name~ct5p-hightpu" } else { "name~ct6e-standard OR name~ct5p-hightpu" }

$rows = gcloud compute machine-types list `
    --project=$project `
    --filter=$filter `
    --format="csv[no-heading](name,zone)" 2>$null | Where-Object { $_ -and $_ -notmatch "^name," }

$byZone = @{}
foreach ($row in $rows) {
    $parts = $row -split ","
    if ($parts.Count -lt 2) { continue }
    $name = $parts[0]
    $zone = $parts[1]
    if ($name -match "-tpu$") { continue }  # skip duplicate -tpu suffix types
    if (-not $byZone.ContainsKey($zone)) { $byZone[$zone] = @() }
    if ($byZone[$zone] -notcontains $name) { $byZone[$zone] += $name }
}

Write-Host "Recommended FLEX_START zones (TPU Builders):"
Write-Host "  v6e: us-east5-a, us-east5-b, us-central1-a"
Write-Host "  v5p: us-east5-a (up to 128 chips via multislice — not single 16t VM)"
Write-Host ""
Write-Host "IMPORTANT: ct6e-standard-16t does NOT exist. Max v6e GCE VM = 8 chips (ct6e-standard-8t)."
Write-Host "Builders guide '8x16' = pod multislice topology (128 chips), not one 16-chip VM."
Write-Host ""

foreach ($zone in ($byZone.Keys | Sort-Object)) {
    $types = $byZone[$zone] | Sort-Object { [int]($_ -replace '\D+','') }
    $chips = ($types | ForEach-Object { if ($_ -match '(\d+)t') { $matches[1] } }) -join ", "
    Write-Host "$zone : chips [$chips]"
    foreach ($t in $types) { Write-Host "    $t" }
}

Write-Host ""
Write-Host "For Gemma 7B+2B use: -Family v6e -ChipCount 8 -Zone us-east5-a"
Write-Host "Need >8 chips? Email tpu-builders-support@google.com or use v5p multislice."
