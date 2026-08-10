<# 
Manual foreground launcher for the Finance AI Agent on Windows.

This script is intentionally user-owned: it may start Ollama and Streamlit only
when a person runs it from a terminal. Codex must not invoke it during normal
development tasks.
#>

param(
    [string]$Model = "qwen3:30b-a3b",
    [int]$Port = 8501,
    [string]$Address = "localhost"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "Finance AI Agent launcher" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"

if (-not (Test-Path $Python)) {
    throw "No se encontró el entorno virtual en .venv. Cree o restaure el entorno antes de iniciar la aplicación."
}

$streamlitPort = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($streamlitPort) {
    throw "El puerto $Port ya está en uso. Cierre el proceso existente o seleccione otro puerto."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama no está instalado o no está en PATH. Instale Ollama y descargue el modelo $Model."
}

$ollamaPort = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
if (-not $ollamaPort) {
    Write-Host "Ollama no está activo. Iniciando 'ollama serve' en segundo plano..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep -Seconds 3
}

$tags = & ollama list
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo consultar Ollama. Verifique que el servicio esté iniciado."
}
if ($tags -notmatch [regex]::Escape($Model)) {
    $answer = Read-Host "El modelo $Model no está instalado. ¿Desea descargarlo ahora con 'ollama pull'? (s/N)"
    if ($answer -match "^[sS]") {
        & ollama pull $Model
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo descargar el modelo $Model."
        }
    } else {
        throw "El modelo $Model es requerido para el modo normal de IA."
    }
}

Write-Host "Iniciando Streamlit en primer plano..." -ForegroundColor Green
Write-Host "URL local: http://$Address`:$Port"
Set-Location $RepoRoot
& $Python -m streamlit run "finance_agent\ui\streamlit_app.py" --server.port $Port --server.address $Address
