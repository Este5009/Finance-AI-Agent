<# Thin developer wrapper around the canonical Python desktop runtime. #>

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "No se encontró el entorno virtual del proyecto en .venv."
}

Set-Location $RepoRoot
& $Python -m finance_agent.desktop @args
exit $LASTEXITCODE
