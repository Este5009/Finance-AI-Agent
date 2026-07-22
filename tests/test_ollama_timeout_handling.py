"""Tests for differentiated Ollama timeout/error handling."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import URLError

import pytest

import finance_agent.llm.ollama_client as client_module
from finance_agent.llm.ollama_client import OllamaClient, OllamaError
from finance_agent.orchestration.pipeline_models import PipelineConfig
from finance_agent.orchestration.pipeline_orchestrator import _ollama_client_for_stage


class _FakeSocket:
    """Context-manager socket placeholder for connection checks."""

    def __enter__(self) -> "_FakeSocket":
        """Enter the fake socket context."""

        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the fake socket context."""


class _FakeResponse:
    """Tiny urllib response stand-in."""

    def __init__(self, payload: dict[str, Any] | str) -> None:
        """Store response payload.

        Inputs: dict encoded as JSON or raw string.
        Outputs: fake response object.
        Assumptions: tests need only ``read`` and context-manager methods.
        """

        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        """Enter response context."""

        return self

    def __exit__(self, *_args: object) -> None:
        """Exit response context."""

    def read(self) -> bytes:
        """Return payload bytes."""

        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return json.dumps(self.payload).encode("utf-8")


def test_connection_refused_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify refused TCP connections are not reported as read timeouts."""

    def refused(*_args: object, **_kwargs: object) -> _FakeSocket:
        """Raise a refused connection."""

        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(client_module.socket, "create_connection", refused)
    client = OllamaClient(connect_timeout_seconds=1, read_timeout_seconds=2)

    with pytest.raises(OllamaError) as exc:
        client.generate("{}")

    assert exc.value.category == "service_unavailable"
    assert "connection refused" in str(exc.value)


def test_read_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify read/inference timeout is classified separately from reachability."""

    monkeypatch.setattr(client_module.socket, "create_connection", lambda *_a, **_k: _FakeSocket())

    def timeout(*_args: object, **_kwargs: object) -> _FakeResponse:
        """Raise a urllib read timeout."""

        raise URLError(socket.timeout("timed out"))

    monkeypatch.setattr(client_module, "urlopen", timeout)
    client = OllamaClient(connect_timeout_seconds=1, read_timeout_seconds=2)

    with pytest.raises(OllamaError) as exc:
        client.generate("{}")

    assert exc.value.category == "inference_timeout"
    assert "read timed out" in str(exc.value) or "inference" in str(exc.value).casefold()


def test_malformed_response_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify malformed API responses have a stable category."""

    monkeypatch.setattr(client_module.socket, "create_connection", lambda *_a, **_k: _FakeSocket())
    monkeypatch.setattr(client_module, "urlopen", lambda *_a, **_k: _FakeResponse("not json"))
    client = OllamaClient()

    with pytest.raises(OllamaError) as exc:
        client.generate("{}")

    assert exc.value.category == "malformed_response"


def test_keep_alive_and_telemetry_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate sends keep_alive and returns HTTP timing telemetry."""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(client_module.socket, "create_connection", lambda *_a, **_k: _FakeSocket())

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        """Capture request payload and timeout."""

        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"response": "{\"ok\": true}", "eval_count": 1})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = OllamaClient(read_timeout_seconds=123, keep_alive="15m")
    result = client.generate_with_metadata("{}")

    assert captured["timeout"] == 123
    assert captured["payload"]["keep_alive"] == "15m"
    assert result["telemetry"]["http_elapsed_time_seconds"] >= 0
    assert result["telemetry"]["read_timeout_seconds"] == 123


def test_warm_up_uses_generate_with_keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify warm-up performs a generation request with configured keep_alive."""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(client_module.socket, "create_connection", lambda *_a, **_k: _FakeSocket())

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        """Capture warm-up payload."""

        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"response": "{\"ok\": true}", "eval_count": 1})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    OllamaClient(keep_alive="15m").warm_up()

    assert captured["payload"]["keep_alive"] == "15m"


def test_pipeline_config_routes_new_timeout_settings() -> None:
    """Verify pipeline config passes connect/read/keep_alive to Ollama client."""

    config = PipelineConfig.from_project_root(
        ".",
        python_executable="python",
        connect_timeout_seconds=7,
        read_timeout_seconds=321,
        stage_timeout_seconds=654,
        ollama_keep_alive="20m",
    )
    client = _ollama_client_for_stage(config, "strategic_analysis")

    assert client.connect_timeout_seconds == 7
    assert client.read_timeout_seconds == 321
    assert client.effective_read_timeout_seconds() == 321
    assert client.keep_alive == "20m"
    assert config.stage_timeout_seconds == 654
