# Finance AI Agent

Agente de análisis financiero universitario. Python conserva la autoridad sobre
ingesta, normalización, cálculos, validación y reportes; Ollama interpreta la
evidencia financiera verificada y redacta recomendaciones ejecutivas.

## Desarrollo local

Instale las dependencias en el entorno virtual del proyecto:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Inicio completo (comprueba Ollama, modelo, puerto y abre la interfaz):

```bash
bash scripts/start_app_macos.sh
```

En Windows:

```powershell
.\scripts\start_app_windows.ps1
```

Los launchers son envoltorios mínimos de `python -m finance_agent.desktop`.
No descargan modelos grandes sin consentimiento explícito.

Para validación acotada sin iniciar servicios:

```bash
.venv/bin/python scripts/run_ui_tests.py
.venv/bin/python scripts/run_project_tests.py
```

Consulte [docs/desktop_distribution.md](docs/desktop_distribution.md) para la
compilación del ejecutable, instalador, aplicación macOS y flujo de primera ejecución.
