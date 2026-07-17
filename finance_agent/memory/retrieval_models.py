"""Structured models returned by historical memory retrieval tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryToolResult:
    """Structured result returned by one historical memory retrieval tool.

    Inputs: tool metadata, success state, compact data, sources, warnings, and confidence.
    Outputs: JSON-compatible object for Python callers and registry adapters.
    Assumptions: missing history is explicit and non-exceptional.
    """

    tool_name: str
    success: bool
    data: dict[str, Any]
    source_references: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unavailable_data: tuple[str, ...] = ()
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retrieval result.

        Inputs: this result.
        Outputs: dictionary with list-valued tuple fields.
        Assumptions: data is already JSON-compatible.
        """

        payload = asdict(self)
        payload["source_references"] = list(self.source_references)
        payload["warnings"] = list(self.warnings)
        payload["unavailable_data"] = list(self.unavailable_data)
        return payload


@dataclass(frozen=True)
class HistoricalPeriod:
    """Stored pipeline period metadata.

    Inputs: period fields from pipeline_runs.
    Outputs: immutable period summary.
    Assumptions: period identifiers are normalized by retrieval validation.
    """

    run_id: str
    period: str
    period_type: str
    completed_at_utc: str
    status: str
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize period metadata."""

        return asdict(self)


@dataclass(frozen=True)
class HistoricalMetricPoint:
    """One historical metric/KPI value.

    Inputs: period, metric, value, unit, status, and optional department.
    Outputs: immutable metric point.
    Assumptions: values were calculated before storage.
    """

    period: str
    metric: str
    value: float | None
    unit: str | None
    status: str | None
    department: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metric point."""

        return asdict(self)
