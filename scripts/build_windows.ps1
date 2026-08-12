<# Build the Windows onedir executable with the project virtual environment. #>

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Cree .venv antes de compilar." }

Set-Location $RepoRoot
& $Python scripts\cleanup_windows_build_locks.py --repo-root $RepoRoot --timeout-seconds 10
if ($LASTEXITCODE -ne 0) {
    Write-Error "No se pudo preparar build/dist para recompilar. Revise el diagnóstico anterior."
    exit $LASTEXITCODE
}
& $Python -m pip install -r requirements.txt -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m PyInstaller --noconfirm --clean packaging\finance_ai_agent.spec
exit $LASTEXITCODE
