#!/usr/bin/env bash
# Manual foreground launcher for the Finance AI Agent on macOS/Linux.
#
# This script is user-owned. It may start Ollama and Streamlit only when a
# person runs it from a terminal; Codex must not invoke it during development.

set -euo pipefail

MODEL="${MODEL:-qwen3:30b-a3b}"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-localhost}"
OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
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

echo "Ollama executable: $(command -v ollama)"
echo "Ollama API endpoint: $OLLAMA_ENDPOINT"

if ! curl -fsS --max-time 2 "$OLLAMA_ENDPOINT/api/tags" >/dev/null 2>&1; then
  echo "Ollama no está activo. Iniciando 'ollama serve' en segundo plano..."
  ollama serve >/tmp/finance-ai-agent-ollama.log 2>&1 &
  sleep 3
fi

MODEL_CHECK_SCRIPT="$REPO_ROOT/scripts/check_ollama_model.py"
run_model_check() {
  local no_cli_fallback="${1:-false}"
  local args=("$MODEL_CHECK_SCRIPT" --model "$MODEL" --endpoint "$OLLAMA_ENDPOINT" --timeout 10)
  if [[ "$no_cli_fallback" == "true" ]]; then
    args+=(--no-cli-fallback)
  fi

  set +e
  model_check_json="$("$PYTHON" "${args[@]}")"
  model_check_code=$?
  set -e

  if ! printf '%s' "$model_check_json" | "$PYTHON" -m json.tool >/dev/null 2>&1; then
    echo "No se pudo interpretar la salida del verificador de Ollama." >&2
    echo "Ejecute: $PYTHON $MODEL_CHECK_SCRIPT --model $MODEL --endpoint $OLLAMA_ENDPOINT" >&2
    exit 1
  fi

  model_available="$(printf '%s' "$model_check_json" | "$PYTHON" -c 'import json,sys; print(str(json.load(sys.stdin).get("model_available", False)).lower())')"
  installed_models="$(printf '%s' "$model_check_json" | "$PYTHON" -c 'import json,sys; print(", ".join(json.load(sys.stdin).get("installed_model_names", [])))')"
  detector_status="$(printf '%s' "$model_check_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("detector_status", "checker_error"))')"
  detector_error="$(printf '%s' "$model_check_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("error") or "")')"

  echo "Modelo requerido: $MODEL"
  echo "Modelos detectados: $installed_models"
  echo "Estado del verificador: $detector_status"
  echo "Resultado de comparación exacta: $model_available"

  if [[ "$model_check_code" -eq 3 || "$detector_status" == "checker_error" ]]; then
    echo "El verificador de modelos falló. Revise Python, el entorno virtual y el paquete finance_agent. Detalle: $detector_error" >&2
    exit 1
  fi
  if [[ "$model_check_code" -eq 4 || "$detector_status" == "ollama_unreachable" ]]; then
    echo "No se pudo consultar Ollama en $OLLAMA_ENDPOINT. Confirme que Ollama esté ejecutándose y vuelva a intentar. Detalle: $detector_error" >&2
    exit 1
  fi
}

run_model_check

if [[ "$model_check_code" -eq 2 || "$detector_status" == "model_missing" ]]; then
  read -r -p "El modelo $MODEL no está instalado. ¿Desea descargarlo ahora con 'ollama pull'? (s/N) " answer
  if [[ "$answer" =~ ^[sS]$ ]]; then
    ollama pull "$MODEL"
    run_model_check true
    if [[ "$model_check_code" -ne 0 || "$model_available" != "true" ]]; then
      echo "El modelo $MODEL no fue detectado por Ollama después de la descarga." >&2
      exit 1
    fi
  else
    echo "El modelo $MODEL es requerido para el modo normal de IA." >&2
    exit 1
  fi
elif [[ "$model_check_code" -ne 0 || "$model_available" != "true" ]]; then
  echo "No se pudo confirmar el modelo $MODEL. Estado: $detector_status" >&2
  exit 1
fi

echo "Iniciando Streamlit en primer plano..."
echo "URL local: http://$ADDRESS:$PORT"
cd "$REPO_ROOT"
exec "$PYTHON" -m streamlit run "finance_agent/ui/streamlit_app.py" --server.port "$PORT" --server.address "$ADDRESS"
