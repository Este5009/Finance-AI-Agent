"""Bounded Ollama model-detection helper for manual app launchers."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any

from finance_agent.llm.ollama_client import OllamaClient, OllamaError
from finance_agent.llm.ollama_models import model_name_exists, parse_ollama_list_output


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
        return result
    except OllamaError as exc:
        result["error"] = str(exc)
        result["error_category"] = exc.category
    if not allow_cli_fallback or ollama_path is None:
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
        result["source"] = "cli_fallback"
        result["error"] = str(exc)
        result["error_category"] = "cli_fallback_failed"
        return result
    result["source"] = "cli_fallback"
    result["cli_return_code"] = completed.returncode
    result["cli_stderr"] = completed.stderr.strip()[:500]
    names = parse_ollama_list_output(completed.stdout)
    result["installed_model_names"] = names
    result["model_available"] = completed.returncode == 0 and model_name_exists(model, names)
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
    diagnostics = check_model(
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout,
        allow_cli_fallback=not args.no_cli_fallback,
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0 if diagnostics.get("model_available") else 2


if __name__ == "__main__":
    raise SystemExit(main())
