"""Structured models for Ollama strategic financial analysis outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisValidationResult:
    """Validation result for one untrusted strategic-analysis response.

    Inputs: validation status, cleaned analysis, and errors.
    Outputs: immutable validation metadata for CLI/scripts/tests.
    Assumptions: accepted analysis is safe to serialize but still model-authored.
    """

    is_valid: bool
    analysis: dict[str, Any] | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class StrategicAnalysisResult:
    """Result of one Step 9 strategic analysis attempt.

    Inputs: output document, acceptance flag, and validation errors.
    Outputs: object used by the CLI to save and summarize results.
    Assumptions: the document is JSON-compatible whether accepted or rejected.
    """

    analysis_document: dict[str, Any]
    accepted: bool
    validation_errors: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisRunSummary:
    """Cross-scope summary of strategic-analysis generation.

    Inputs: generated/rejected counters and accepted analysis metadata.
    Outputs: compact JSON summary artifact.
    Assumptions: confidence values come from validated model responses only.
    """

    summary_id: str
    analyses_requested: int
    analyses_generated: int
    analyses_rejected: int
    average_confidence: float | None
    recommendations_generated: int
    scopes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the analysis summary.

        Inputs: this summary object.
        Outputs: JSON-compatible dictionary.
        Assumptions: scopes are already JSON-compatible.
        """

        data = asdict(self)
        data["scopes"] = list(self.scopes)
        return data
