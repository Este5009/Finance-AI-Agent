<#
.SYNOPSIS
Starts the Finance AI Agent Streamlit UI in the foreground on Windows.

.DESCRIPTION
This script is intended for a human developer or demo operator. It never
starts Streamlit as a hidden or detached process, so Ctrl+C cleanly stops the
server and no long-running process is left behind for Codex to wait on.
#>

$ErrorActionPreference = "Stop"

function Get-RepositoryRoot {
    <#
    .SYNOPSIS
    Resolves the repository root from this script location.

    .OUTPUTS
    Absolute repository root path.
    #>
    $scriptDirectory = Split-Path -Parent $MyInvocation.ScriptName
    return (Resolve-Path (Join-Path $scriptDirectory "..")).Path
}

function Test-PortAvailable {
    <#
    .SYNOPSIS
    Checks whether the Streamlit port is free before starting the server.

    .PARAMETER Port
    TCP port number to inspect.

    .OUTPUTS
    True when the port is available; false when another process is listening.
    #>
    param([int]$Port)

    try {
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -First 1
        if ($null -ne $listener) {
            Write-Host "El puerto $Port ya está en uso por PID $($listener.OwningProcess)." -ForegroundColor Yellow
            try {
                $process = Get-Process -Id $listener.OwningProcess -ErrorAction Stop
                Write-Host "Proceso: $($process.ProcessName)" -ForegroundColor Yellow
            }
            catch {
                Write-Host "No se pudo identificar el nombre del proceso." -ForegroundColor Yellow
            }
            return $false
        }
    }
    catch {
        # Some locked-down Windows environments block Get-NetTCPConnection.
        # In that case, continue and let Streamlit report a bind error if needed.
        Write-Host "No se pudo verificar el puerto con Get-NetTCPConnection; se intentará iniciar Streamlit." -ForegroundColor Yellow
    }

    return $true
}

$repoRoot = Get-RepositoryRoot
$pythonExecutable = Join-Path $repoRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $repoRoot "finance_agent\ui\streamlit_app.py"
$port = 8501

if (-not (Test-Path $pythonExecutable)) {
    Write-Error "No se encontró el entorno virtual en .venv. Cree o active el entorno del proyecto antes de iniciar la interfaz."
    exit 1
}

if (-not (Test-Path $appPath)) {
    Write-Error "No se encontró la aplicación Streamlit en finance_agent\ui\streamlit_app.py."
    exit 1
}

if (-not (Test-PortAvailable -Port $port)) {
    Write-Host "Cierre ese proceso o use otro puerto antes de iniciar la interfaz." -ForegroundColor Yellow
    exit 1
}

Write-Host "Iniciando Finance AI Agent en primer plano..." -ForegroundColor Cyan
Write-Host "URL local: http://localhost:$port" -ForegroundColor Green
Write-Host "Presione Ctrl+C para detener Streamlit." -ForegroundColor Cyan

Set-Location $repoRoot
& $pythonExecutable -m streamlit run "finance_agent\ui\streamlit_app.py" --server.port $port --server.address localhost
exit $LASTEXITCODE
