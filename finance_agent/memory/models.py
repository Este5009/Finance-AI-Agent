"""Structured models for SQLite historical storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_DB_PATH = Path("data") / "memory" / "finance_memory.db"
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ArtifactRecord:
    """Reference to one persisted pipeline artifact.

    Inputs: artifact type, path, and checksum.
    Outputs: immutable artifact reference used by the repository.
    Assumptions: artifact content remains on disk, not in SQLite blobs.
    """

    artifact_type: str
    path: str
    checksum: str | None


@dataclass(frozen=True)
class KpiRecord:
    """Normalized KPI row for historical lookup.

    Inputs: period, optional department, metric, value, unit, and status.
    Outputs: immutable KPI record.
    Assumptions: KPI values were calculated by Python before storage.
    """

    period: str | None
    department: str | None
    metric: str
    value: float | None
    unit: str | None
    status: str | None


@dataclass(frozen=True)
class AnomalyRecord:
    """Normalized anomaly row for historical lookup.

    Inputs: anomaly identity, scope, metric, severity, values JSON, and text.
    Outputs: immutable anomaly record.
    Assumptions: anomaly facts come from deterministic anomaly detection.
    """

    anomaly_id: str
    period: str | None
    department: str | None
    type: str | None
    severity: str | None
    metric: str | None
    values_json: str
    description: str | None


@dataclass(frozen=True)
class RecommendationRecord:
    """Structured recommendation row from accepted strategic analysis.

    Inputs: recommendation metadata and follow-up status.
    Outputs: immutable recommendation record.
    Assumptions: recommendation wording is model-authored but Python-validated.
    """

    recommendation_id: str
    priority: str | None
    department: str | None
    action: str
    expected_impact: str | None
    status: str = "unknown"
    follow_up_required: bool = False


@dataclass(frozen=True)
class GoalRecord:
    """Goal progress row derived from processed outputs when available.

    Inputs: goal metric, target, actual, unit, and status.
    Outputs: immutable goal record.
    Assumptions: absence of goal rows is acceptable when processed outputs lack them.
    """

    goal_id: str
    metric: str
    target: float | None
    actual: float | None
    unit: str | None
    progress_status: str | None
    gap: float | None = None
    score: float | None = None
    direction: str | None = None
    source_provenance_json: str | None = None


@dataclass(frozen=True)
class MemoryFactRecord:
    """Compact memory fact for efficient future retrieval.

    Inputs: category, subject, fact, and confidence.
    Outputs: immutable compact memory row.
    Assumptions: facts are summaries, not raw report duplication.
    """

    category: str
    subject: str
    fact: str
    confidence: float | None


@dataclass(frozen=True)
class StoredPipelineRun:
    """Complete storage payload for one accepted pipeline run.

    Inputs: run metadata plus child record collections.
    Outputs: immutable repository payload.
    Assumptions: idempotency key uniquely identifies equivalent reprocessing.
    """

    run_id: str
    idempotency_key: str
    period: str
    period_type: str
    started_at_utc: str | None
    completed_at_utc: str
    report_hash: str
    goals_hash: str
    report_path: str
    goals_path: str
    language: str
    model: str
    confidence: float | None
    cache_hit: bool
    cache_key: str | None
    status: str
    artifact_directory: str
    configuration_json: str
    artifacts: tuple[ArtifactRecord, ...] = field(default_factory=tuple)
    kpis: tuple[KpiRecord, ...] = field(default_factory=tuple)
    anomalies: tuple[AnomalyRecord, ...] = field(default_factory=tuple)
    recommendations: tuple[RecommendationRecord, ...] = field(default_factory=tuple)
    goals: tuple[GoalRecord, ...] = field(default_factory=tuple)
    memory_facts: tuple[MemoryFactRecord, ...] = field(default_factory=tuple)
    source_documents: tuple[SourceDocumentRecord, ...] = field(default_factory=tuple)
    source_revision_confirmed: bool = False


@dataclass(frozen=True)
class StorageResult:
    """Summary returned after storing or skipping a pipeline run.

    Inputs: status, database path, run id, and count metadata.
    Outputs: immutable storage summary for pipeline/scripts/tests.
    Assumptions: skipped runs are expected for rejected strategy or draft reports.
    """

    stored: bool
    run_id: str | None
    database_path: Path
    idempotency_key: str | None
    table_counts: dict[str, int]
    reason: str | None = None
    updated_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize storage result.

        Inputs: this storage result.
        Outputs: JSON-compatible dictionary.
        Assumptions: paths are stringified for CLI output.
        """

        return {
            "stored": self.stored,
            "run_id": self.run_id,
            "database_path": str(self.database_path),
            "idempotency_key": self.idempotency_key,
            "table_counts": self.table_counts,
            "reason": self.reason,
            "updated_existing": self.updated_existing,
        }


@dataclass(frozen=True)
class SourceDocumentRecord:
    """Stored source-document metadata for uploaded reports and goals.

    Inputs: content identity, document type, period metadata, and version fields.
    Outputs: immutable record used by the repository layer.
    Assumptions: file bytes stay on disk; SQLite stores metadata and hashes only.
    """

    document_id: str
    content_sha256: str
    original_filename: str
    document_type: str
    size_bytes: int
    detected_period: str | None
    effective_period: str
    upload_time_utc: str
    processing_status: str
    source_metadata_json: str
    version_number: int = 1
    supersedes_document_id: str | None = None
    is_current: bool = True


@dataclass(frozen=True)
class DocumentClassification:
    """Pre-processing classification for one uploaded source document.

    Inputs: status and optional existing/current records.
    Outputs: UI-safe classification used before pipeline execution.
    Assumptions: caller decides whether a revision is explicitly confirmed.
    """

    status: str
    message: str
    document_type: str
    content_sha256: str
    effective_period: str
    existing_document_id: str | None = None
    current_document_id: str | None = None
    next_version_number: int = 1
    requires_revision_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the classification for UI/session diagnostics."""

        return {
            "status": self.status,
            "message": self.message,
            "document_type": self.document_type,
            "content_sha256": self.content_sha256,
            "effective_period": self.effective_period,
            "existing_document_id": self.existing_document_id,
            "current_document_id": self.current_document_id,
            "next_version_number": self.next_version_number,
            "requires_revision_confirmation": self.requires_revision_confirmation,
        }


@dataclass(frozen=True)
class DocumentRegistrationResult:
    """Result returned after registering an accepted source document.

    Inputs: persisted document ID, status, and version metadata.
    Outputs: immutable summary for UI and pipeline diagnostics.
    Assumptions: duplicate content reuses the existing document record.
    """

    document_id: str
    status: str
    message: str
    version_number: int
    reused_existing: bool = False
    registered_revision: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registration result for UI/session diagnostics."""

        return {
            "document_id": self.document_id,
            "status": self.status,
            "message": self.message,
            "version_number": self.version_number,
            "reused_existing": self.reused_existing,
            "registered_revision": self.registered_revision,
        }
