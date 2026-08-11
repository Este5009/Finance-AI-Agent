"""Persistent configuration for the desktop launcher."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finance_agent.llm.ollama_client import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_OLLAMA_MODEL


@dataclass(frozen=True)
class DesktopConfig:
    """User-editable desktop runtime settings.

    Inputs: Ollama, readiness, and local UI startup preferences.
    Outputs: validated immutable configuration.
    Assumptions: large model downloads always require separate explicit consent.
    """

    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    model: str = DEFAULT_OLLAMA_MODEL
    address: str = "127.0.0.1"
    preferred_port: int = 8501
    startup_timeout_seconds: float = 45.0
    ollama_start_timeout_seconds: float = 20.0
    connect_timeout_seconds: float = 5.0
    health_read_timeout_seconds: float = 120.0
    health_stage_timeout_seconds: float = 180.0
    open_browser: bool = True

    @classmethod
    def load(cls, path: Path) -> "DesktopConfig":
        """Load configuration, creating a default file on first run.

        Inputs: configuration JSON path.
        Outputs: validated DesktopConfig.
        Assumptions: unknown keys from newer versions are safely ignored.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            config = cls()
            config.save(path)
            return config
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("La configuración debe ser un objeto JSON.")
        allowed = cls.__dataclass_fields__.keys()
        config = cls(**{key: value for key, value in raw.items() if key in allowed})
        config.validate()
        return config

    def save(self, path: Path) -> Path:
        """Persist configuration without overwriting any other user data."""

        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def validate(self) -> None:
        """Validate settings that could otherwise create unsafe startup behavior."""

        if not self.model.strip():
            raise ValueError("No hay un modelo de Ollama configurado.")
        if not self.ollama_endpoint.startswith(("http://", "https://")):
            raise ValueError("El servidor de Ollama debe usar http:// o https://.")
        if not 1 <= int(self.preferred_port) <= 65535:
            raise ValueError("El puerto preferido no es válido.")
        for name in (
            "startup_timeout_seconds",
            "ollama_start_timeout_seconds",
            "connect_timeout_seconds",
            "health_read_timeout_seconds",
            "health_stage_timeout_seconds",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} debe ser mayor que cero.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration dictionary."""

        return asdict(self)
