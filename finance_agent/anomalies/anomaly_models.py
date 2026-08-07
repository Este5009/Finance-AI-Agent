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


FINDING_TYPES = {
    "institutional_violation",
    "statistical_anomaly",
    "system_review_rule",
    "potential_duplicate",
    "data_quality_finding",
    "informational_observation",
}

REFERENCE_ORIGINS = {
    "institutional/workbook",
    "approved_budget",
    "historical/statistical",
    "system-derived/default",
    "synthetic/test",
    "none",
}

SYSTEM_REFERENCE_NOTICE_ES = (
    "Referencia analítica del sistema. No corresponde a una meta, límite o política institucional."
)
STATISTICAL_REFERENCE_NOTICE_ES = (
    "Referencia histórica/estadística. No corresponde a una regla institucional."
)


@dataclass(frozen=True)
class Anomaly:
    """One deterministic financial anomaly, finding, or data-quality flag."""

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
    finding_type: str = "system_review_rule"
    reference_type: str = "threshold"
    reference_origin: str = "system-derived/default"
    reference_source: str = "AnomalyThresholds configuration"
    is_institutional_reference: bool = False
    reason_for_flagging: str = ""
    supporting_evidence: str = ""
    recommended_action: str = ""
    reference_notice_es: str = SYSTEM_REFERENCE_NOTICE_ES

    def to_dict(self) -> dict[str, Any]:
        """Serialize the anomaly as a JSON/CSV-compatible dictionary.

        Inputs: this anomaly.
        Outputs: dictionary preserving every evidence and prioritization field.
        Assumptions: observed/threshold values are scalar calculation outputs.
        """

        data = asdict(self)
        # Keep newer canonical fields populated even for legacy constructor
        # calls. This protects downstream renderers and LLM prompts from having
        # to guess provenance from English wording.
        if not data.get("supporting_evidence"):
            data["supporting_evidence"] = self.evidence
        if not data.get("recommended_action"):
            data["recommended_action"] = self.recommended_next_check
        if data.get("finding_type") not in FINDING_TYPES:
            data["finding_type"] = "system_review_rule"
        if data.get("reference_origin") not in REFERENCE_ORIGINS:
            data["reference_origin"] = "system-derived/default"
        if data.get("is_institutional_reference") or data.get("reference_origin") == "none":
            data["reference_notice_es"] = ""
        elif data.get("finding_type") == "statistical_anomaly":
            data["reference_notice_es"] = STATISTICAL_REFERENCE_NOTICE_ES
        elif data.get("reference_origin") == "system-derived/default":
            data["reference_notice_es"] = SYSTEM_REFERENCE_NOTICE_ES
        return data


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
