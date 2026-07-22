"""Structured models for modular reasoning stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningValidationResult:
    """Validation result for one reasoning-stage response.

    Inputs: validity flag, cleaned payload, and immutable error messages.
    Outputs: serializable metadata for tests, scripts, and orchestration.
    Assumptions: accepted payloads are still model-authored and evidence-bound.
    """

    is_valid: bool
    payload: dict[str, Any] | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ReasoningStageResult:
    """One completed modular reasoning stage.

    Inputs: stage identity, accepted payload, validation errors, prompt metrics,
    and runtime telemetry.
    Outputs: JSON-compatible stage result for reasoning-state snapshots.
    Assumptions: failed stages may have an empty payload but remain auditable.
    """

    stage_id: str
    stage_name: str
    accepted: bool
    payload: dict[str, Any]
    validation_errors: tuple[str, ...]
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage result.

        Inputs: this result.
        Outputs: JSON-compatible dictionary.
        Assumptions: payload and telemetry are already JSON-compatible.
        """

        data = asdict(self)
        data["validation_errors"] = list(self.validation_errors)
        return data
