"""Production readiness checks for Ollama-backed strategic reasoning."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from finance_agent.llm.ollama_client import OllamaError


@dataclass(frozen=True)
class OllamaReadinessResult:
    """Result of one bounded Ollama AI-readiness check.

    Inputs: service/model/health statuses and diagnostic telemetry.
    Outputs: serializable readiness object for pipeline summaries and UI.
    Assumptions: this check runs before strategic analysis in normal AI mode.
    """

    is_ready: bool
    reachable: bool
    model_available: bool
    health_prompt_ok: bool
    active_model: str
    message_es: str
    issues: tuple[str, ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the readiness result.

        Inputs: this readiness result.
        Outputs: JSON-compatible dictionary.
        Assumptions: tuples are converted for report/pipeline diagnostics.
        """

        data = asdict(self)
        data["issues"] = list(self.issues)
        return data


def check_ollama_readiness(
    client: Any,
    *,
    model: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    stage_timeout_seconds: float,
) -> OllamaReadinessResult:
    """Verify that Ollama can support production strategic reasoning.

    Inputs: Ollama-like client, model name, and timeout configuration.
    Outputs: readiness result with precise Spanish diagnostics.
    Assumptions: callers pass a bounded client; no persistent process is
    started by this function.
    """

    started = time.perf_counter()
    telemetry: dict[str, Any] = {
        "model": model,
        "connect_timeout_seconds": connect_timeout_seconds,
        "read_timeout_seconds": read_timeout_seconds,
        "stage_timeout_seconds": stage_timeout_seconds,
    }
    timeout_errors = _timeout_config_errors(
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        stage_timeout_seconds=stage_timeout_seconds,
    )
    if timeout_errors:
        return OllamaReadinessResult(
            is_ready=False,
            reachable=False,
            model_available=False,
            health_prompt_ok=False,
            active_model=str(model or ""),
            message_es="El motor de IA no está disponible: la configuración de tiempos de espera no es válida.",
            issues=timeout_errors,
            telemetry={**telemetry, "elapsed_seconds": time.perf_counter() - started},
        )
    if not str(model or "").strip():
        return OllamaReadinessResult(
            is_ready=False,
            reachable=False,
            model_available=False,
            health_prompt_ok=False,
            active_model="",
            message_es="El motor de IA no está disponible: no hay un modelo configurado.",
            issues=("configured model is empty",),
            telemetry={**telemetry, "elapsed_seconds": time.perf_counter() - started},
        )

    try:
        reachable = bool(client.is_available())
    except OllamaError as exc:
        return _failed_readiness(
            model=model,
            message_es="El motor de IA no está disponible: no se pudo contactar el servicio de Ollama.",
            issue=str(exc),
            category=exc.category,
            telemetry=telemetry,
            started=started,
        )
    except Exception as exc:  # noqa: BLE001 - readiness must classify unexpected adapters.
        return _failed_readiness(
            model=model,
            message_es="El motor de IA no está disponible: la verificación del servicio falló.",
            issue=str(exc),
            category="readiness_error",
            telemetry=telemetry,
            started=started,
        )
    if not reachable:
        return _failed_readiness(
            model=model,
            message_es="El motor de IA no está disponible: Ollama no respondió.",
            issue="Ollama service is not reachable.",
            category="service_unavailable",
            telemetry=telemetry,
            started=started,
        )

    try:
        model_available = bool(client.model_exists(model)) if hasattr(client, "model_exists") else model in set(client.list_models())
    except OllamaError as exc:
        return _failed_readiness(
            model=model,
            reachable=True,
            message_es="El motor de IA no está disponible: no se pudo verificar el modelo configurado.",
            issue=str(exc),
            category=exc.category,
            telemetry=telemetry,
            started=started,
        )
    if not model_available:
        return OllamaReadinessResult(
            is_ready=False,
            reachable=True,
            model_available=False,
            health_prompt_ok=False,
            active_model=model,
            message_es=f"El motor de IA no está disponible: el modelo {model} no está instalado.",
            issues=(f"model not installed: {model}",),
            telemetry={
                **telemetry,
                "error_category": "model_missing",
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    try:
        health_response = client.health_prompt() if hasattr(client, "health_prompt") else client.generate('{"ok": true}')
    except OllamaError as exc:
        return _failed_readiness(
            model=model,
            reachable=True,
            model_available=True,
            message_es="El motor de IA no está disponible: el modelo no completó la prueba de salud.",
            issue=str(exc),
            category=exc.category,
            telemetry=telemetry,
            started=started,
        )
    except Exception as exc:  # noqa: BLE001 - keep readiness diagnostic explicit.
        return _failed_readiness(
            model=model,
            reachable=True,
            model_available=True,
            message_es="El motor de IA no está disponible: la prueba de salud falló.",
            issue=str(exc),
            category="health_prompt_failed",
            telemetry=telemetry,
            started=started,
        )
    return OllamaReadinessResult(
        is_ready=True,
        reachable=True,
        model_available=True,
        health_prompt_ok=True,
        active_model=model,
        message_es=f"Motor de IA disponible con el modelo {model}.",
        issues=(),
        telemetry={
            **telemetry,
            "health_prompt_telemetry": health_response.get("telemetry", {})
            if isinstance(health_response, dict)
            else {},
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _timeout_config_errors(
    *,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    stage_timeout_seconds: float,
) -> tuple[str, ...]:
    """Return timeout-configuration validation errors.

    Inputs: connection, read, and stage timeouts.
    Outputs: tuple of errors, empty when valid.
    Assumptions: stage timeout should cover at least one model response.
    """

    errors: list[str] = []
    for name, value in (
        ("connect_timeout_seconds", connect_timeout_seconds),
        ("read_timeout_seconds", read_timeout_seconds),
        ("stage_timeout_seconds", stage_timeout_seconds),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric")
            continue
        if numeric <= 0:
            errors.append(f"{name} must be positive")
    try:
        if float(stage_timeout_seconds) < float(connect_timeout_seconds):
            errors.append("stage_timeout_seconds must be greater than or equal to connect_timeout_seconds")
    except (TypeError, ValueError):
        pass
    return tuple(errors)


def _failed_readiness(
    *,
    model: str,
    message_es: str,
    issue: str,
    category: str,
    telemetry: dict[str, Any],
    started: float,
    reachable: bool = False,
    model_available: bool = False,
) -> OllamaReadinessResult:
    """Build one failed readiness result with consistent telemetry.

    Inputs: diagnostic metadata and partial readiness flags.
    Outputs: failed OllamaReadinessResult.
    Assumptions: the Spanish message is safe for normal UI display.
    """

    return OllamaReadinessResult(
        is_ready=False,
        reachable=reachable,
        model_available=model_available,
        health_prompt_ok=False,
        active_model=model,
        message_es=message_es,
        issues=(issue,),
        telemetry={
            **telemetry,
            "error_category": category,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
