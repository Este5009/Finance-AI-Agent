"""Tests for canonical Ollama model-name detection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from finance_agent.llm.ollama_client import OllamaClient
from finance_agent.llm.ollama_models import (
    extract_ollama_model_names,
    model_name_exists,
    normalize_ollama_model_name,
    parse_ollama_list_output,
)
from scripts.check_ollama_model import check_model


ROOT = Path(__file__).resolve().parents[1]


def test_extract_model_names_from_realistic_tags_response() -> None:
    """Verify Ollama API ``name`` fields are extracted exactly."""

    response = {
        "models": [
            {"name": "qwen3:30b-a3b", "model": "ignored-if-name-present"},
            {"name": "qwen3:latest"},
        ]
    }

    assert extract_ollama_model_names(response) == ["qwen3:30b-a3b", "qwen3:latest"]


def test_extract_model_names_accepts_adjacent_metadata_fields() -> None:
    """Verify compatibility with model/model_name fields."""

    response = {
        "models": [
            {"model": "qwen3:30b-a3b"},
            {"model_name": "qwen3:latest"},
        ]
    }

    assert extract_ollama_model_names(response) == ["qwen3:30b-a3b", "qwen3:latest"]


def test_model_detection_trims_whitespace_but_remains_case_sensitive() -> None:
    """Verify accidental whitespace is ignored but tag case is not changed."""

    assert normalize_ollama_model_name(" qwen3:30b-a3b\r\n") == "qwen3:30b-a3b"
    assert model_name_exists("qwen3:30b-a3b", [" qwen3:30b-a3b "])
    assert not model_name_exists("QWEN3:30B-A3B", ["qwen3:30b-a3b"])


def test_exact_tag_comparison_does_not_confuse_latest_suffix() -> None:
    """Verify exact names are required for similarly named Ollama tags."""

    assert model_name_exists("qwen3:30b-a3b", ["qwen3:30b-a3b"])
    assert not model_name_exists("qwen3:30b-a3b", ["qwen3:30b-a3b:latest"])


def test_parse_ollama_list_output_is_cli_fallback_only() -> None:
    """Verify fallback parser extracts first-column model names."""

    output = (
        "NAME             ID              SIZE\n"
        "qwen3:30b-a3b    ad815644918f    18 GB\n"
        "qwen3:latest     500a1f067a9f    5 GB\n"
    )

    assert parse_ollama_list_output(output) == ["qwen3:30b-a3b", "qwen3:latest"]
    assert model_name_exists("qwen3:30b-a3b", parse_ollama_list_output(output))


class FakeModelClient(OllamaClient):
    """OllamaClient test double with in-memory API response."""

    def __init__(self, response: dict[str, Any]) -> None:
        """Store the fake API tags response."""

        super().__init__()
        self.response = response

    def _request(self, path: str, *, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the fake response for model-list calls."""

        assert path == "/api/tags"
        assert method == "GET"
        return self.response


def test_ollama_client_model_exists_uses_canonical_api_names() -> None:
    """Verify ``OllamaClient.model_exists`` accepts exact reported model names."""

    client = FakeModelClient({"models": [{"name": "qwen3:30b-a3b"}, {"name": "qwen3:latest"}]})

    assert client.list_models() == ["qwen3:30b-a3b", "qwen3:latest"]
    assert client.model_exists("qwen3:30b-a3b")
    assert not client.model_exists("missing:model")


def test_check_model_uses_api_result_before_cli_fallback(monkeypatch: object) -> None:
    """Verify launcher helper succeeds from API tags without CLI parsing."""

    class FakeClient:
        """Fake client used to replace the real OllamaClient constructor."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Accept arbitrary constructor arguments."""

        def list_models(self) -> list[str]:
            """Return a realistic installed model list."""

            return ["qwen3:30b-a3b", "qwen3:latest"]

    import scripts.check_ollama_model as module

    monkeypatch.setattr(module, "OllamaClient", FakeClient)
    result = check_model(
        model="qwen3:30b-a3b",
        endpoint="http://localhost:11434",
        timeout_seconds=1,
        allow_cli_fallback=False,
    )

    assert result["model_available"] is True
    assert result["source"] == "api"
    assert result["installed_model_names"] == ["qwen3:30b-a3b", "qwen3:latest"]
    assert result["detector_status"] == "model_present"


def test_check_model_reports_model_missing_after_successful_detection(monkeypatch: object) -> None:
    """Verify a completed check can distinguish a missing model from checker failure."""

    class FakeClient:
        """Fake client that reports a different installed model."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Accept arbitrary constructor arguments."""

        def list_models(self) -> list[str]:
            """Return installed models that do not include the required tag."""

            return ["qwen3:latest"]

    import scripts.check_ollama_model as module

    monkeypatch.setattr(module, "OllamaClient", FakeClient)
    result = check_model(
        model="qwen3:30b-a3b",
        endpoint="http://localhost:11434",
        timeout_seconds=1,
        allow_cli_fallback=False,
    )

    assert result["model_available"] is False
    assert result["detector_status"] == "model_missing"


def test_check_ollama_model_direct_execution_imports_package_from_repo_root() -> None:
    """Verify direct script execution no longer fails importing ``finance_agent``."""

    script = ROOT / "scripts" / "check_ollama_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "qwen3:30b-a3b",
            "--endpoint",
            "http://127.0.0.1:9",
            "--timeout",
            "0.2",
            "--no-cli-fallback",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode in {0, 2, 4}
    payload = json.loads(result.stdout)
    assert payload["required_model"] == "qwen3:30b-a3b"
    assert payload["detector_status"] in {"model_present", "model_missing", "ollama_unreachable"}


def test_check_ollama_model_direct_execution_imports_package_from_arbitrary_cwd(tmp_path: Path) -> None:
    """Verify launchers can call the checker when their working directory differs."""

    script = ROOT / "scripts" / "check_ollama_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "qwen3:30b-a3b",
            "--endpoint",
            "http://127.0.0.1:9",
            "--timeout",
            "0.2",
            "--no-cli-fallback",
        ],
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode in {0, 2, 4}
    payload = json.loads(result.stdout)
    assert payload["endpoint"] == "http://127.0.0.1:9"


def test_checker_internal_failure_uses_checker_error_exit_code(monkeypatch: object, capsys: object) -> None:
    """Verify checker crashes are not reported as missing models."""

    import scripts.check_ollama_model as module

    def boom(**_kwargs: object) -> dict[str, Any]:
        """Raise like an unexpected checker bug."""

        raise RuntimeError("synthetic checker crash")

    monkeypatch.setattr(module, "check_model", boom)

    exit_code = module.main(["--model", "qwen3:30b-a3b"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == module.EXIT_CHECKER_ERROR
    assert payload["detector_status"] == "checker_error"
    assert "synthetic checker crash" in payload["error"]
