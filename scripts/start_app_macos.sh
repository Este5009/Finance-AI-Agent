#!/usr/bin/env bash
# Manual foreground launcher for the Finance AI Agent on macOS/Linux.
#
# This script is user-owned. It may start Ollama and Streamlit only when a
# person runs it from a terminal; Codex must not invoke it during development.

set -euo pipefail

MODEL="${MODEL:-qwen3:30b-a3b}"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-localhost}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

echo "Finance AI Agent launcher"
echo "Repository: $REPO_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "No se encontró el entorno virtual en .venv. Cree o restaure el entorno antes de iniciar la aplicación." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "El puerto $PORT ya está en uso. Cierre el proceso existente o seleccione otro puerto." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama no está instalado o no está en PATH. Instale Ollama y descargue el modelo $MODEL." >&2
  exit 1
fi

if ! curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama no está activo. Iniciando 'ollama serve' en segundo plano..."
  ollama serve >/tmp/finance-ai-agent-ollama.log 2>&1 &
  sleep 3
fi

if ! ollama list | grep -Fq "$MODEL"; then
  read -r -p "El modelo $MODEL no está instalado. ¿Desea descargarlo ahora con 'ollama pull'? (s/N) " answer
  if [[ "$answer" =~ ^[sS]$ ]]; then
    ollama pull "$MODEL"
  else
    echo "El modelo $MODEL es requerido para el modo normal de IA." >&2
    exit 1
  fi
fi

echo "Iniciando Streamlit en primer plano..."
echo "URL local: http://$ADDRESS:$PORT"
cd "$REPO_ROOT"
exec "$PYTHON" -m streamlit run "finance_agent/ui/streamlit_app.py" --server.port "$PORT" --server.address "$ADDRESS"
