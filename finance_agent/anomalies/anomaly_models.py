"""Structured anomaly records and deterministic identifier generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


@dataclass(frozen=True)
class Anomaly:
    """One deterministic financial anomaly or data-quality risk flag."""

    anomaly_id: str
    title: str
    description: str
    metric: str
    observed_value: float | int | str | None
    threshold_value: float | int | str | None
    severity: str
    period: str
    source_file: str
    evidence: str
    recommended_next_check: str
    detection_method: str
    rule_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the anomaly as a JSON/CSV-compatible dictionary.

        Inputs: this anomaly.
        Outputs: dictionary preserving every evidence and prioritization field.
        Assumptions: observed/threshold values are scalar calculation outputs.
        """

        return asdict(self)


class AnomalyIdGenerator:
    """Generate stable sequential identifiers within one anomaly report."""

    def __init__(self, prefix: str) -> None:
        """Initialize a report-specific identifier sequence.

        Inputs: readable report prefix such as ANOM-JUNE-2026.
        Outputs: generator ready to create identifiers.
        Assumptions: one generator instance is used per report.
        """

        self.prefix = prefix
        self._counter = 0

    def next_id(self) -> str:
        """Return the next zero-padded anomaly identifier.

        Inputs: current generator state.
        Outputs: unique sequential identifier.
        Assumptions: detection rules execute in deterministic order.
        """

        self._counter += 1
        return f"{self.prefix}-{self._counter:03d}"


def severity_counts(anomalies: list[Anomaly]) -> dict[str, int]:
    """Count anomalies by severity in standard priority order.

    Inputs: anomaly list.
    Outputs: critical/high/medium/low count dictionary.
    Assumptions: every anomaly severity is one of the supported labels.
    """

    return {
        severity: sum(anomaly.severity == severity for anomaly in anomalies)
        for severity in ("critical", "high", "medium", "low")
    }


def sort_anomalies(anomalies: list[Anomaly]) -> list[Anomaly]:
    """Sort anomalies by severity while preserving deterministic ID order.

    Inputs: anomaly list.
    Outputs: new highest-priority-first list.
    Assumptions: anomaly IDs end in a sortable numeric sequence.
    """

    return sorted(
        anomalies,
        key=lambda anomaly: (
            -SEVERITY_ORDER[anomaly.severity],
            anomaly.anomaly_id,
        ),
    )
