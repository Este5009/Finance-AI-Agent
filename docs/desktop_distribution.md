# Distribución de escritorio

## Arquitectura

`finance_agent.desktop` es el punto de entrada compartido para Windows y macOS.
Inicializa almacenamiento escribible, carga la configuración, comprueba/inicia
Ollama, verifica el modelo y una inferencia mínima, elige un puerto libre, inicia
Streamlit, espera su endpoint de salud, abre el navegador y termina limpiamente
los procesos hijos que inició.

La aplicación instalada trata los recursos empaquetados como solo lectura. Los
datos del usuario no dependen del directorio actual ni se guardan junto al
ejecutable:

- Windows: `%LOCALAPPDATA%\FinanceAIAgent\`
- macOS: `~/Library/Application Support/FinanceAIAgent/`

Debajo de esa raíz se crean `database`, `imports`, `reports`, `cache`, `logs`,
`config`, `runtime` y `runtime/outputs`. La base histórica es
`database/finance_memory.db`; la configuración es `config/config.json`.
Las actualizaciones no reemplazan ni eliminan estos archivos.

## Primera ejecución y Ollama

El modelo predeterminado es `qwen3:30b-a3b`, configurable en `config.json` para
equipos con distinta memoria. El ejecutable busca Ollama en `PATH` y en las
ubicaciones estándar del instalador. Si está detenido, intenta iniciar
`ollama serve`. Después verifica el modelo mediante la API existente y ejecuta
la comprobación de salud compartida.

Si Ollama falta, se muestra orientación en español. Si falta el modelo, se
informa que ocupa varios GB y se solicita consentimiento explícito antes de
ejecutar `ollama pull`. Nunca se incluye ni descarga silenciosamente el LLM.
Los detalles técnicos se escriben en `logs/finance-ai-agent.log`; el usuario no
recibe tracebacks de Python.

## Compilar el ejecutable de Windows

Desde PowerShell x64, con Python 3.12 y el entorno `.venv`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
.\scripts\build_windows.ps1
```

Resultado esperado:

```text
dist\Finance AI Agent\Finance AI Agent.exe
```

PyInstaller empaqueta el intérprete; el equipo ejecutivo no necesita Python del
sistema. La especificación incluye código y recursos propios (`schema.sql`) más
recursos de Streamlit/ReportLab. Excluye `data/`, `outputs/`, bases locales,
subidas, reportes generados, cachés, secretos y `.env`.

## Compilar el instalador de Windows

Instale Inno Setup 6, compile primero el ejecutable y ejecute:

```powershell
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" installer\windows\FinanceAIAgent.iss
```

El instalador crea el acceso del menú Inicio, ofrece acceso de Escritorio,
conserva `%LOCALAPPDATA%\FinanceAIAgent`, detecta la instalación de Ollama y
abre el flujo de primera ejecución. No contiene el modelo.

## Preparar la aplicación de macOS

En un Mac con Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-build.txt
bash scripts/build_macos.sh
```

Resultado preparado: `dist/Finance AI Agent.app`. La configuración usa el mismo
punto de entrada y almacenamiento de producción. El artefacto todavía no está
firmado ni notarizado.

## Antes de una versión de producción

Quedan expresamente pendientes:

- pruebas en máquinas Windows/macOS limpias y sin Python;
- iconos/identidad visual finales;
- firma Authenticode del ejecutable e instalador de Windows;
- firma, hardened runtime y notarización de macOS;
- prueba de actualización preservando bases/configuración reales;
- validación de requisitos de RAM/disco y modelos recomendados por hardware;
- revisión corporativa de privacidad, licencias y distribución de Ollama;
- sustitución futura del navegador por un shell nativo, sin cambiar el motor financiero.
