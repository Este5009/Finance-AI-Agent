#!/usr/bin/env bash
# Start the Finance AI Agent Streamlit UI in the foreground on macOS/Linux.
#
# This script is intended for a human developer or demo operator. It avoids
# hidden/detached process behavior so Ctrl+C cleanly stops Streamlit.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_executable="${repo_root}/.venv/bin/python"
app_path="${repo_root}/finance_agent/ui/streamlit_app.py"
port="8501"

if [[ ! -x "${python_executable}" ]]; then
  echo "No se encontró el entorno virtual en .venv. Cree o active el entorno del proyecto antes de iniciar la interfaz." >&2
  exit 1
fi

if [[ ! -f "${app_path}" ]]; then
  echo "No se encontró la aplicación Streamlit en finance_agent/ui/streamlit_app.py." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:${port} -sTCP:LISTEN >/tmp/finance_agent_streamlit_port_check.txt 2>/dev/null; then
    echo "El puerto ${port} ya está en uso:" >&2
    cat /tmp/finance_agent_streamlit_port_check.txt >&2
    exit 1
  fi
elif command -v nc >/dev/null 2>&1; then
  if nc -z localhost "${port}" >/dev/null 2>&1; then
    echo "El puerto ${port} ya está en uso. Cierre ese proceso o use otro puerto." >&2
    exit 1
  fi
else
  echo "No se encontró lsof ni nc; Streamlit reportará el error si el puerto está ocupado." >&2
fi

echo "Iniciando Finance AI Agent en primer plano..."
echo "URL local: http://localhost:${port}"
echo "Presione Ctrl+C para detener Streamlit."

cd "${repo_root}"
exec "${python_executable}" -m streamlit run "finance_agent/ui/streamlit_app.py" --server.port "${port}" --server.address localhost
