#!/usr/bin/env bash
# Prepare the unsigned macOS .app using the shared PyInstaller entry point.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_executable="${repo_root}/.venv/bin/python"

cd "${repo_root}"
"${python_executable}" -m pip install -r requirements.txt -r requirements-build.txt
"${python_executable}" -m PyInstaller --noconfirm --clean packaging/finance_ai_agent.spec
