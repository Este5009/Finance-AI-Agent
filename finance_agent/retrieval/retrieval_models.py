"""Structured models for deterministic evidence retrieval packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalRequest:
    """One validated queue item converted into a retrieval request.

    Inputs: queue metadata, public retrieval tool name, and validated arguments.
    Outputs: immutable request object used by retrieval functions.
    Assumptions: arguments were validated by Step 7 before queue creation.
    """

    execution_id: str
    task_id: str
    anomaly_id: str | None
    question: str
    priority: str
    tool_name: str
    arguments: dict[str, Any]
    expected_output: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request to JSON-compatible metadata.

        Inputs: this request.
        Outputs: dictionary representation.
        Assumptions: nested argument values are already JSON-compatible.
        """

        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """Structured output returned by one retrieval function.

    Inputs: retrieval function status, data, sources, warnings, and confidence.
    Outputs: JSON-compatible evidence payload after serialization.
    Assumptions: data is processed evidence only, never raw workbook/PDF content.
    """

    retrieval_name: str
    success: bool
    data: dict[str, Any]
    source_references: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    unavailable_data: tuple[str, ...] = ()
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize one retrieval result.

        Inputs: this retrieval result.
        Outputs: dictionary with lists for tuple fields.
        Assumptions: confidence is bounded by the retrieval function.
        """

        return {
            "retrieval_name": self.retrieval_name,
            "success": self.success,
            "data": self.data,
            "source_references": list(self.source_references),
            "warnings": list(self.warnings),
            "unavailable_data": list(self.unavailable_data),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class EvidencePackage:
    """Evidence attached to one investigation task after retrieval.

    Inputs: original task metadata and retrieval result.
    Outputs: task-level package for later strategic analysis.
    Assumptions: evidence_summary describes availability, not recommendations.
    """

    task_id: str
    execution_id: str
    anomaly_id: str | None
    priority: str
    investigation_question: str
    retrieved_evidence: RetrievalResult
    evidence_summary: str
    source_references: tuple[str, ...]
    retrieval_warnings: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize one task evidence package.

        Inputs: this evidence package.
        Outputs: JSON-compatible dictionary.
        Assumptions: no downstream analysis has been performed.
        """

        return {
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "anomaly_id": self.anomaly_id,
            "priority": self.priority,
            "investigation_question": self.investigation_question,
            "retrieved_evidence": self.retrieved_evidence.to_dict(),
            "evidence_summary": self.evidence_summary,
            "source_references": list(self.source_references),
            "retrieval_warnings": list(self.retrieval_warnings),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RetrievalRunSummary:
    """Execution statistics for one retrieval queue run.

    Inputs: package identity and retrieval counters.
    Outputs: compact summary used by the CLI and annual summary artifact.
    Assumptions: unavailable evidence can occur even when the retrieval call succeeds.
    """

    package_id: str
    period_slug: str
    tasks_executed: int
    successful_retrievals: int
    failed_retrievals: int
    unavailable_evidence: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize run counters.

        Inputs: this summary object.
        Outputs: JSON-compatible dictionary.
        Assumptions: counters are non-negative integers.
        """

        return asdict(self)
