"""Bounded Ollama model-detection helper for manual app launchers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

# When this file is executed directly, Python puts ``scripts/`` on sys.path
# instead of the repository root.  Resolve the root from this file so launchers
# work from PowerShell, bash, the repo root, or any arbitrary current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finance_agent.llm.ollama_client import OllamaClient, OllamaError
from finance_agent.llm.ollama_models import model_name_exists, parse_ollama_list_output


EXIT_MODEL_PRESENT = 0
EXIT_MODEL_MISSING = 2
EXIT_CHECKER_ERROR = 3
EXIT_OLLAMA_UNREACHABLE = 4

UNREACHABLE_CATEGORIES = {
    "service_unavailable",
    "connection_timeout",
    "connection_refused",
    "inference_timeout",
    "read_timeout",
}


def _status_exit_code(diagnostics: dict[str, Any]) -> int:
    """Return the launcher-facing exit code for model-check diagnostics.

    Inputs: diagnostics from ``check_model``.
    Outputs: stable process exit code.
    Assumptions: launchers use code 2 only for a completed "model missing"
    check, never for checker crashes or Ollama reachability failures.
    """

    status = diagnostics.get("detector_status")
    if diagnostics.get("model_available") is True or status == "model_present":
        return EXIT_MODEL_PRESENT
    if status == "model_missing":
        return EXIT_MODEL_MISSING
    if status == "ollama_unreachable":
        return EXIT_OLLAMA_UNREACHABLE
    return EXIT_CHECKER_ERROR


def check_model(
    *,
    model: str,
    endpoint: str,
    timeout_seconds: float,
    allow_cli_fallback: bool = True,
) -> dict[str, Any]:
    """Check whether the required Ollama model is installed.

    Inputs: required model, endpoint, timeout, and fallback flag.
    Outputs: diagnostic dictionary safe for launcher display.
    Assumptions: the Ollama API is authoritative; CLI output is fallback only.
    """

    ollama_path = shutil.which("ollama")
    result: dict[str, Any] = {
        "ollama_executable": ollama_path,
        "endpoint": endpoint,
        "required_model": model.strip(),
        "installed_model_names": [],
        "model_available": False,
        "detector_status": "checker_error",
        "source": "api",
        "error": None,
    }
    client = OllamaClient(
        endpoint=endpoint,
        model=model,
        connect_timeout_seconds=min(float(timeout_seconds), 10.0),
        read_timeout_seconds=float(timeout_seconds),
    )
    try:
        names = client.list_models()
        result["installed_model_names"] = names
        result["model_available"] = model_name_exists(model, names)
        result["detector_status"] = "model_present" if result["model_available"] else "model_missing"
        return result
    except OllamaError as exc:
        result["error"] = str(exc)
        result["error_category"] = exc.category
    if not allow_cli_fallback or ollama_path is None:
        result["detector_status"] = (
            "ollama_unreachable"
            if result.get("error_category") in UNREACHABLE_CATEGORIES
            else "checker_error"
        )
        return result
    try:
        completed = subprocess.run(
            [ollama_path, "list"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        api_error_category = result.get("error_category")
        result["source"] = "cli_fallback"
        result["error"] = str(exc)
        result["error_category"] = "cli_fallback_failed"
        result["detector_status"] = (
            "ollama_unreachable"
            if api_error_category in UNREACHABLE_CATEGORIES
            else "checker_error"
        )
        return result
    result["source"] = "cli_fallback"
    result["cli_return_code"] = completed.returncode
    result["cli_stderr"] = completed.stderr.strip()[:500]
    names = parse_ollama_list_output(completed.stdout)
    result["installed_model_names"] = names
    result["model_available"] = completed.returncode == 0 and model_name_exists(model, names)
    if completed.returncode == 0:
        result["detector_status"] = "model_present" if result["model_available"] else "model_missing"
    else:
        result["detector_status"] = "checker_error"
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the model check from the command line.

    Inputs: optional argv.
    Outputs: process status code; JSON diagnostics are written to stdout.
    Assumptions: launchers read stdout and decide whether to offer installation.
    """

    parser = argparse.ArgumentParser(description="Check installed Ollama model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--no-cli-fallback", action="store_true")
    args = parser.parse_args(argv)
    try:
        diagnostics = check_model(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout,
            allow_cli_fallback=not args.no_cli_fallback,
        )
    except Exception as exc:  # pragma: no cover - defensive launcher boundary.
        diagnostics = {
            "ollama_executable": shutil.which("ollama"),
            "endpoint": args.endpoint,
            "required_model": args.model.strip(),
            "installed_model_names": [],
            "model_available": False,
            "detector_status": "checker_error",
            "source": "checker",
            "error": str(exc),
            "error_category": exc.__class__.__name__,
        }
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return _status_exit_code(diagnostics)


if __name__ == "__main__":
    raise SystemExit(main())
