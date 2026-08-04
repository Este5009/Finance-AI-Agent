# Finance-AI-Agent
AI-powered financial analyst and strategy agent that analyzes financial reports, compares performance against business goals, detects trends and anomalies, and generates actionable recommendations using structured data, Python analytics, and LLM-based reasoning.

## Local Streamlit UI

Starting Streamlit is a manual developer/demo action, not part of automated
Codex task completion.

Windows:

```powershell
.\scripts\start_streamlit_windows.ps1
```

macOS/Linux:

```bash
bash scripts/start_streamlit_macos.sh
```

Both launchers run Streamlit in the foreground and print:

```text
http://localhost:8501
```

For bounded diagnostics without starting services:

```bash
python scripts/check_local_services.py
```

For bounded validation without launching Streamlit:

```bash
python scripts/run_ui_tests.py
python scripts/run_project_tests.py
```
