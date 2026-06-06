# Update a single key in .env (creates key if missing)
param(
    [Parameter(Mandatory)]
    [string]$Key,
    [Parameter(Mandatory)]
    [string]$Value,
    [string]$EnvFile = (Join-Path $PSScriptRoot "..\.env")
)

$lines = Get-Content $EnvFile
$found = $false
$newLines = foreach ($line in $lines) {
    if ($line -match "^\s*$([regex]::Escape($Key))\s*=") {
        $found = $true
        "$Key=$Value"
    } else {
        $line
    }
}
if (-not $found) {
    $newLines += "$Key=$Value"
}
$newLines | Set-Content $EnvFile -Encoding utf8
