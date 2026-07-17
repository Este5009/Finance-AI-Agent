"""Structured models for SQLite historical storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_DB_PATH = Path("data") / "memory" / "finance_memory.db"
SCHEMA_VERSION = 1


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
