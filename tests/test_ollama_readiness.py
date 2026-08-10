"""Tests for production Ollama AI-readiness checks."""

from __future__ import annotations

import pytest

from finance_agent.llm.ollama_client import OllamaError
from finance_agent.llm.ollama_readiness import check_ollama_readiness


class FakeReadinessClient:
    """Small Ollama-like client for readiness tests."""

    def __init__(
        self,
        *,
        available: bool = True,
        models: tuple[str, ...] = ("qwen3:30b-a3b",),
        health_error: Exception | None = None,
    ) -> None:
        """Store fake service/model/health behavior."""

        self.available = available
        self.models = models
        self.health_error = health_error

    def is_available(self) -> bool:
        """Return fake service reachability."""

        return self.available

    def model_exists(self, model: str) -> bool:
        """Return fake model availability."""

        return model in self.models

    def health_prompt(self) -> dict[str, object]:
        """Return or raise a fake health-prompt result."""

        if self.health_error is not None:
            raise self.health_error
        return {"response": '{"ok": true}', "telemetry": {"http_elapsed_time_seconds": 0.01}}


def test_readiness_accepts_reachable_model_and_health_prompt() -> None:
    """Verify all readiness checks must pass for normal AI mode."""

    result = check_ollama_readiness(
        FakeReadinessClient(),
        model="qwen3:30b-a3b",
        connect_timeout_seconds=10,
        read_timeout_seconds=600,
        stage_timeout_seconds=900,
    )

    assert result.is_ready is True
    assert result.reachable is True
    assert result.model_available is True
    assert result.health_prompt_ok is True
    assert result.active_model == "qwen3:30b-a3b"


def test_readiness_reports_ollama_unavailable_in_spanish() -> None:
    """Verify service failures do not masquerade as valid AI readiness."""

    result = check_ollama_readiness(
        FakeReadinessClient(available=False),
        model="qwen3:30b-a3b",
        connect_timeout_seconds=10,
        read_timeout_seconds=600,
        stage_timeout_seconds=900,
    )

    assert result.is_ready is False
    assert result.reachable is False
    assert "El motor de IA no está disponible" in result.message_es


def test_readiness_reports_missing_model() -> None:
    """Verify missing configured models block normal AI mode."""

    result = check_ollama_readiness(
        FakeReadinessClient(models=("qwen3:latest",)),
        model="qwen3:30b-a3b",
        connect_timeout_seconds=10,
        read_timeout_seconds=600,
        stage_timeout_seconds=900,
    )

    assert result.is_ready is False
    assert result.model_available is False
    assert "modelo qwen3:30b-a3b no está instalado" in result.message_es


def test_readiness_reports_health_prompt_failure_category() -> None:
    """Verify a model that cannot generate fails the health contract."""

    result = check_ollama_readiness(
        FakeReadinessClient(health_error=OllamaError("read timeout", category="inference_timeout")),
        model="qwen3:30b-a3b",
        connect_timeout_seconds=10,
        read_timeout_seconds=600,
        stage_timeout_seconds=900,
    )

    assert result.is_ready is False
    assert result.health_prompt_ok is False
    assert result.telemetry["error_category"] == "inference_timeout"


@pytest.mark.parametrize(
    ("connect_timeout", "read_timeout", "stage_timeout"),
    [(0, 600, 900), (10, -1, 900), (10, 600, 1)],
)
def test_readiness_rejects_invalid_timeout_configuration(
    connect_timeout: float,
    read_timeout: float,
    stage_timeout: float,
) -> None:
    """Verify bad timeout settings are caught before an Ollama request."""

    result = check_ollama_readiness(
        FakeReadinessClient(),
        model="qwen3:30b-a3b",
        connect_timeout_seconds=connect_timeout,
        read_timeout_seconds=read_timeout,
        stage_timeout_seconds=stage_timeout,
    )

    assert result.is_ready is False
    assert result.telemetry["model"] == "qwen3:30b-a3b"
