#!/usr/bin/env bash
# Thin developer wrapper around the canonical Python desktop runtime.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_executable="${repo_root}/.venv/bin/python"

if [[ ! -x "${python_executable}" ]]; then
  echo "No se encontró el entorno virtual del proyecto en .venv." >&2
  exit 1
fi

cd "${repo_root}"
exec "${python_executable}" -m finance_agent.desktop "$@"
