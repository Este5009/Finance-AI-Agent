"""Shared Ollama model-name detection helpers."""

from __future__ import annotations

from typing import Any


def normalize_ollama_model_name(value: Any) -> str:
    """Normalize an Ollama model name for exact operational comparison.

    Inputs: arbitrary model-name value from API/CLI metadata.
    Outputs: trimmed model name, or an empty string for invalid values.
    Assumptions: model names remain case-sensitive; only accidental outer
    whitespace/newlines are removed.
    """

    if not isinstance(value, str):
        return ""
    return value.strip()


def extract_ollama_model_names(tags_response: dict[str, Any]) -> list[str]:
    """Extract installed model names from an Ollama ``/api/tags`` response.

    Inputs: decoded JSON response from Ollama.
    Outputs: de-duplicated model names preserving service order.
    Assumptions: modern Ollama uses ``name``; tests also cover ``model`` and
    ``model_name`` for compatibility with adjacent metadata shapes.
    """

    models = tags_response.get("models", [])
    if not isinstance(models, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model", "model_name"):
            name = normalize_ollama_model_name(item.get(key))
            if name and name not in seen:
                names.append(name)
                seen.add(name)
                break
    return names


def parse_ollama_list_output(output: str) -> list[str]:
    """Parse model names from human-readable ``ollama list`` output.

    Inputs: stdout text from ``ollama list``.
    Outputs: first-column model names preserving order.
    Assumptions: this is a fallback for launch diagnostics only; API JSON is
    preferred whenever the Ollama service is reachable.
    """

    names: list[str] = []
    seen: set[str] = set()
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        first = line.split()[0].strip()
        if first.casefold() == "name":
            continue
        if first and first not in seen:
            names.append(first)
            seen.add(first)
    return names


def model_name_exists(model_name: str, installed_names: list[str] | tuple[str, ...]) -> bool:
    """Return whether an exact Ollama model name is installed.

    Inputs: required model name and installed names.
    Outputs: True only for exact case-sensitive match after trimming whitespace.
    Assumptions: ``qwen3:30b-a3b`` and ``qwen3:30b-a3b:latest`` are distinct
    tags unless Ollama reports both explicitly.
    """

    required = normalize_ollama_model_name(model_name)
    if not required:
        return False
    return required in {normalize_ollama_model_name(name) for name in installed_names}
