"""Repository layer for transactional Finance AI Agent memory storage."""

from __future__ import annotations

import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from finance_agent.memory.database import connect_database, initialize_database
from finance_agent.memory.models import (
    DEFAULT_MEMORY_DB_PATH,
    DocumentClassification,
    DocumentRegistrationResult,
    SourceDocumentRecord,
    StoredPipelineRun,
    StorageResult,
)


CHILD_TABLES = (
    "artifacts",
    "kpis",
    "anomalies",
    "recommendations",
    "goals",
    "memory_facts",
)

DOCUMENT_TABLES = (
    "source_documents",
    "pipeline_run_documents",
)


class MemoryRepository:
    """Transactional repository for historical run and memory records."""

    def __init__(self, database_path: str | Path = DEFAULT_MEMORY_DB_PATH) -> None:
        """Create a repository bound to one SQLite file.

        Inputs: database path.
        Outputs: repository instance.
        Assumptions: schema initialization is safe to run repeatedly.
        """

        self.database_path = initialize_database(database_path)

    def table_counts(self) -> dict[str, int]:
        """Return row counts for all memory tables.

        Inputs: none.
        Outputs: dictionary of table name to row count.
        Assumptions: used for diagnostics and tests, not business logic.
        """

        tables = ("pipeline_runs", *CHILD_TABLES, *DOCUMENT_TABLES)
        with connect_database(self.database_path) as connection:
            return {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in tables
            }

    def classify_source_document(
        self,
        *,
        document_type: str,
        effective_period: str,
        raw_bytes: bytes | None = None,
        content_sha256: str | None = None,
    ) -> DocumentClassification:
        """Classify an upload before analysis using content identity.

        Inputs: document role, raw bytes or SHA-256 hash, and effective period.
        Outputs: duplicate/new/revision classification.
        Assumptions: filename is intentionally ignored for identity checks.
        """

        self._validate_document_type(document_type)
        if raw_bytes is not None:
            content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if not content_sha256:
            raise ValueError("raw_bytes or content_sha256 is required for source document classification")
        with connect_database(self.database_path) as connection:
            duplicate = connection.execute(
                """
                SELECT * FROM source_documents
                WHERE document_type = ? AND content_sha256 = ?
                LIMIT 1
                """,
                (document_type, content_sha256),
            ).fetchone()
            current = connection.execute(
                """
                SELECT * FROM source_documents
                WHERE document_type = ? AND effective_period = ?
                  AND is_current = 1 AND processing_status = 'accepted'
                ORDER BY version_number DESC
                LIMIT 1
                """,
                (document_type, effective_period),
            ).fetchone()
            max_version = connection.execute(
                """
                SELECT MAX(version_number) AS version
                FROM source_documents
                WHERE document_type = ? AND effective_period = ?
                """,
                (document_type, effective_period),
            ).fetchone()["version"]

        if duplicate:
            return DocumentClassification(
                status="duplicate",
                message="Este archivo ya fue registrado anteriormente. Se reutilizará el registro existente.",
                document_type=document_type,
                content_sha256=content_sha256,
                effective_period=effective_period,
                existing_document_id=str(duplicate["document_id"]),
                current_document_id=str(current["document_id"]) if current else None,
                next_version_number=int(duplicate["version_number"]),
                requires_revision_confirmation=False,
            )
        if current:
            return DocumentClassification(
                status="revision",
                message="Existe una versión aceptada para este período. Confirme antes de registrar una nueva versión.",
                document_type=document_type,
                content_sha256=content_sha256,
                effective_period=effective_period,
                current_document_id=str(current["document_id"]),
                next_version_number=int(max_version or 1) + 1,
                requires_revision_confirmation=True,
            )
        return DocumentClassification(
            status="new",
            message="Archivo nuevo listo para registrar después de un análisis aceptado.",
            document_type=document_type,
            content_sha256=content_sha256,
            effective_period=effective_period,
            next_version_number=1,
            requires_revision_confirmation=False,
        )

    def register_source_document(
        self,
        record: SourceDocumentRecord,
        *,
        revision_confirmed: bool = False,
    ) -> DocumentRegistrationResult:
        """Persist one source document after an accepted pipeline run.

        Inputs: document metadata and explicit revision confirmation flag.
        Outputs: registration result with duplicate/revision status.
        Assumptions: callers register only after strategy/report validation succeeds.
        """

        self._validate_document_type(record.document_type)
        classification = self.classify_source_document(
            document_type=record.document_type,
            content_sha256=record.content_sha256,
            effective_period=record.effective_period,
        )
        if classification.status == "duplicate" and classification.existing_document_id:
            return DocumentRegistrationResult(
                document_id=classification.existing_document_id,
                status="duplicate",
                message=classification.message,
                version_number=classification.next_version_number,
                reused_existing=True,
            )
        if classification.requires_revision_confirmation and not revision_confirmed:
            raise ValueError("Revision confirmation is required before registering this document.")

        try:
            with connect_database(self.database_path) as connection:
                with connection:
                    supersedes = classification.current_document_id if classification.status == "revision" else None
                    version = classification.next_version_number
                    if supersedes:
                        connection.execute(
                            """
                            UPDATE source_documents
                            SET is_current = 0
                            WHERE document_type = ? AND effective_period = ?
                              AND is_current = 1
                            """,
                            (record.document_type, record.effective_period),
                        )
                    connection.execute(
                        """
                        INSERT INTO source_documents (
                            document_id, content_sha256, original_filename,
                            document_type, size_bytes, detected_period,
                            effective_period, upload_time_utc, processing_status,
                            source_metadata_json, version_number,
                            supersedes_document_id, is_current
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_type, content_sha256) DO NOTHING
                        """,
                        (
                            record.document_id,
                            record.content_sha256,
                            record.original_filename,
                            record.document_type,
                            record.size_bytes,
                            record.detected_period,
                            record.effective_period,
                            record.upload_time_utc,
                            record.processing_status,
                            record.source_metadata_json,
                            version,
                            supersedes,
                            int(record.is_current),
                        ),
                    )
                    row = connection.execute(
                        """
                        SELECT document_id, version_number
                        FROM source_documents
                        WHERE document_type = ? AND content_sha256 = ?
                        """,
                        (record.document_type, record.content_sha256),
                    ).fetchone()
        except sqlite3.IntegrityError:
            # Concurrent identical inserts are resolved by rereading the unique
            # content record; conflicting revisions still surface as errors.
            with connect_database(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT document_id, version_number
                    FROM source_documents
                    WHERE document_type = ? AND content_sha256 = ?
                    """,
                    (record.document_type, record.content_sha256),
                ).fetchone()
                if row is None:
                    raise
        if row is None:
            raise RuntimeError("Document registration did not return a row.")
        registered_revision = classification.status == "revision"
        return DocumentRegistrationResult(
            document_id=str(row["document_id"]),
            status="revision" if registered_revision else "new",
            message=(
                "Nueva versión registrada."
                if registered_revision
                else "Nuevo período registrado."
            ),
            version_number=int(row["version_number"]),
            registered_revision=registered_revision,
        )

    def link_run_document(self, *, run_id: str, document_id: str, document_role: str) -> None:
        """Link one accepted run to the source document it consumed.

        Inputs: run ID, source document ID, and role.
        Outputs: none.
        Assumptions: run and document rows already exist.
        """

        self._validate_document_type(document_role)
        with connect_database(self.database_path) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO pipeline_run_documents(run_id, document_id, document_role)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id, document_role) DO UPDATE SET
                        document_id = excluded.document_id
                    """,
                    (run_id, document_id, document_role),
                )

    def fetch_current_document(
        self,
        *,
        document_type: str,
        effective_period: str,
    ) -> sqlite3.Row | None:
        """Return the current accepted document for a period and role.

        Inputs: document type and effective period.
        Outputs: source_documents row or None.
        Assumptions: unique partial index permits at most one current accepted row.
        """

        self._validate_document_type(document_type)
        with connect_database(self.database_path) as connection:
            return connection.execute(
                """
                SELECT * FROM source_documents
                WHERE document_type = ? AND effective_period = ?
                  AND is_current = 1 AND processing_status = 'accepted'
                LIMIT 1
                """,
                (document_type, effective_period),
            ).fetchone()

    @staticmethod
    def _validate_document_type(document_type: str) -> None:
        """Validate source document roles before SQL writes.

        Inputs: document type.
        Outputs: none; raises ValueError for unsupported roles.
        Assumptions: callers use the same role names in UI and storage.
        """

        if document_type not in {"financial_report", "goals_document"}:
            raise ValueError(f"Unsupported document_type: {document_type}")

    def existing_run_id(self, idempotency_key: str) -> str | None:
        """Return an existing run ID for an idempotency key.

        Inputs: idempotency key.
        Outputs: run ID or None.
        Assumptions: unique constraint enforces at most one match.
        """

        with connect_database(self.database_path) as connection:
            row = connection.execute(
                "SELECT run_id FROM pipeline_runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return str(row["run_id"]) if row else None

    def fetch_periods(
        self,
        *,
        limit: int | None = None,
        before_period: str | None = None,
        include_current: bool = False,
    ) -> tuple[sqlite3.Row, ...]:
        """Fetch stored run periods for read-only memory retrieval.

        Inputs: optional limit, before-period filter, and inclusivity flag.
        Outputs: tuple of sqlite rows from pipeline_runs.
        Assumptions: chronological sorting is finalized by the retrieval layer.
        """

        where = ""
        params: list[object] = []
        if before_period is not None:
            operator = "<=" if include_current else "<"
            where = f"WHERE period {operator} ?"
            params.append(before_period)
        query = f"SELECT * FROM pipeline_runs {where} ORDER BY period"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with connect_database(self.database_path) as connection:
            return tuple(connection.execute(query, params).fetchall())

    def fetch_period_run(self, period: str) -> sqlite3.Row | None:
        """Fetch one stored pipeline run by period.

        Inputs: period identifier.
        Outputs: pipeline_runs row or None.
        Assumptions: current storage keeps one idempotent run per period/config.
        """

        with connect_database(self.database_path) as connection:
            return connection.execute(
                "SELECT * FROM pipeline_runs WHERE period = ? ORDER BY updated_at_utc DESC LIMIT 1",
                (period,),
            ).fetchone()

    def fetch_rows_for_periods(
        self,
        table: str,
        periods: tuple[str, ...],
        *,
        extra_where: str = "",
        params: tuple[object, ...] = (),
    ) -> tuple[sqlite3.Row, ...]:
        """Fetch child-table rows joined to runs for selected periods.

        Inputs: allowed table name, periods, optional SQL predicate and params.
        Outputs: read-only sqlite rows.
        Assumptions: table names are allowlisted before interpolation.
        """

        if table not in CHILD_TABLES:
            raise ValueError(f"Unsupported memory child table: {table}")
        if not periods:
            return ()
        placeholders = ",".join("?" for _ in periods)
        query = (
            f"SELECT child.*, runs.period AS run_period, runs.period_type, runs.updated_at_utc "
            f"FROM {table} AS child "
            "JOIN pipeline_runs AS runs ON child.run_id = runs.run_id "
            f"WHERE runs.period IN ({placeholders})"
        )
        query_params: list[object] = list(periods)
        if extra_where:
            query += f" AND ({extra_where})"
            query_params.extend(params)
        query += " ORDER BY runs.period"
        with connect_database(self.database_path) as connection:
            return tuple(connection.execute(query, query_params).fetchall())

    def save_pipeline_run(self, payload: StoredPipelineRun) -> StorageResult:
        """Store one accepted pipeline run transactionally and idempotently.

        Inputs: complete stored-run payload.
        Outputs: storage summary including whether an existing run was updated.
        Assumptions: child collections are rebuilt from current artifacts each run.
        """

        updated_existing = self.existing_run_id(payload.idempotency_key) is not None
        now = datetime.now(timezone.utc).isoformat()
        try:
            with connect_database(self.database_path) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO pipeline_runs (
                            run_id, idempotency_key, period, period_type,
                            started_at_utc, completed_at_utc, report_hash, goals_hash,
                            report_path, goals_path, language, model, confidence,
                            cache_hit, cache_key, status, artifact_directory,
                            configuration_json, updated_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(idempotency_key) DO UPDATE SET
                            period=excluded.period,
                            period_type=excluded.period_type,
                            completed_at_utc=excluded.completed_at_utc,
                            language=excluded.language,
                            model=excluded.model,
                            confidence=excluded.confidence,
                            cache_hit=excluded.cache_hit,
                            cache_key=excluded.cache_key,
                            status=excluded.status,
                            artifact_directory=excluded.artifact_directory,
                            configuration_json=excluded.configuration_json,
                            updated_at_utc=excluded.updated_at_utc
                        """,
                        (
                            payload.run_id,
                            payload.idempotency_key,
                            payload.period,
                            payload.period_type,
                            payload.started_at_utc,
                            payload.completed_at_utc,
                            payload.report_hash,
                            payload.goals_hash,
                            payload.report_path,
                            payload.goals_path,
                            payload.language,
                            payload.model,
                            payload.confidence,
                            int(payload.cache_hit),
                            payload.cache_key,
                            payload.status,
                            payload.artifact_directory,
                            payload.configuration_json,
                            now,
                        ),
                    )
                    row = connection.execute(
                        "SELECT run_id FROM pipeline_runs WHERE idempotency_key = ?",
                        (payload.idempotency_key,),
                    ).fetchone()
                    run_id = str(row["run_id"]) if row else payload.run_id
                    document_links = [
                        (
                            self._register_source_document_in_transaction(
                                connection,
                                item,
                                revision_confirmed=payload.source_revision_confirmed,
                            ),
                            item.document_type,
                        )
                        for item in payload.source_documents
                    ]
                    self._replace_children(connection, run_id, payload)
                    for document_id, document_role in document_links:
                        connection.execute(
                            """
                            INSERT INTO pipeline_run_documents(run_id, document_id, document_role)
                            VALUES (?, ?, ?)
                            ON CONFLICT(run_id, document_role) DO UPDATE SET
                                document_id = excluded.document_id
                            """,
                            (run_id, document_id, document_role),
                        )
        except sqlite3.DatabaseError:
            # Let callers/tests observe rollback behavior rather than hiding corruption.
            raise

        return StorageResult(
            stored=True,
            run_id=self.existing_run_id(payload.idempotency_key) or payload.run_id,
            database_path=self.database_path,
            idempotency_key=payload.idempotency_key,
            table_counts=self.table_counts(),
            updated_existing=updated_existing,
        )

    def _classify_source_document_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        document_type: str,
        content_sha256: str,
        effective_period: str,
    ) -> DocumentClassification:
        """Classify a source document using an existing transaction.

        Inputs: active connection, document role, hash, and period.
        Outputs: document classification.
        Assumptions: caller has already validated the document type.
        """

        duplicate = connection.execute(
            """
            SELECT * FROM source_documents
            WHERE document_type = ? AND content_sha256 = ?
            LIMIT 1
            """,
            (document_type, content_sha256),
        ).fetchone()
        current = connection.execute(
            """
            SELECT * FROM source_documents
            WHERE document_type = ? AND effective_period = ?
              AND is_current = 1 AND processing_status = 'accepted'
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (document_type, effective_period),
        ).fetchone()
        max_version = connection.execute(
            """
            SELECT MAX(version_number) AS version
            FROM source_documents
            WHERE document_type = ? AND effective_period = ?
            """,
            (document_type, effective_period),
        ).fetchone()["version"]
        if duplicate:
            return DocumentClassification(
                status="duplicate",
                message="Este archivo ya fue registrado anteriormente. Se reutilizará el registro existente.",
                document_type=document_type,
                content_sha256=content_sha256,
                effective_period=effective_period,
                existing_document_id=str(duplicate["document_id"]),
                current_document_id=str(current["document_id"]) if current else None,
                next_version_number=int(duplicate["version_number"]),
                requires_revision_confirmation=False,
            )
        if current:
            return DocumentClassification(
                status="revision",
                message="Existe una versión aceptada para este período. Confirme antes de registrar una nueva versión.",
                document_type=document_type,
                content_sha256=content_sha256,
                effective_period=effective_period,
                current_document_id=str(current["document_id"]),
                next_version_number=int(max_version or 1) + 1,
                requires_revision_confirmation=True,
            )
        return DocumentClassification(
            status="new",
            message="Archivo nuevo listo para registrar después de un análisis aceptado.",
            document_type=document_type,
            content_sha256=content_sha256,
            effective_period=effective_period,
            next_version_number=1,
            requires_revision_confirmation=False,
        )

    def _register_source_document_in_transaction(
        self,
        connection: sqlite3.Connection,
        record: SourceDocumentRecord,
        *,
        revision_confirmed: bool,
    ) -> str:
        """Register a source document inside the accepted-run transaction.

        Inputs: active connection, source-document metadata, and confirmation flag.
        Outputs: persisted document ID.
        Assumptions: exact duplicate content must be reused safely.
        """

        self._validate_document_type(record.document_type)
        classification = self._classify_source_document_in_transaction(
            connection,
            document_type=record.document_type,
            content_sha256=record.content_sha256,
            effective_period=record.effective_period,
        )
        if classification.existing_document_id:
            return classification.existing_document_id
        if classification.requires_revision_confirmation and not revision_confirmed:
            raise ValueError("Revision confirmation is required before registering this document.")
        supersedes = classification.current_document_id if classification.status == "revision" else None
        if supersedes:
            connection.execute(
                """
                UPDATE source_documents
                SET is_current = 0
                WHERE document_type = ? AND effective_period = ?
                  AND is_current = 1
                """,
                (record.document_type, record.effective_period),
            )
        connection.execute(
            """
            INSERT INTO source_documents (
                document_id, content_sha256, original_filename,
                document_type, size_bytes, detected_period, effective_period,
                upload_time_utc, processing_status, source_metadata_json,
                version_number, supersedes_document_id, is_current
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_type, content_sha256) DO NOTHING
            """,
            (
                record.document_id,
                record.content_sha256,
                record.original_filename,
                record.document_type,
                record.size_bytes,
                record.detected_period,
                record.effective_period,
                record.upload_time_utc,
                record.processing_status,
                record.source_metadata_json,
                classification.next_version_number,
                supersedes,
                int(record.is_current),
            ),
        )
        row = connection.execute(
            """
            SELECT document_id FROM source_documents
            WHERE document_type = ? AND content_sha256 = ?
            """,
            (record.document_type, record.content_sha256),
        ).fetchone()
        if row is None:
            raise RuntimeError("Document registration failed inside transaction.")
        return str(row["document_id"])

    def _replace_children(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        payload: StoredPipelineRun,
    ) -> None:
        """Delete and reinsert child records for an idempotent run update.

        Inputs: active connection, run ID, and payload.
        Outputs: None.
        Assumptions: caller owns an open transaction.
        """

        for table in CHILD_TABLES:
            connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM pipeline_run_documents WHERE run_id = ?", (run_id,))
        now = datetime.now(timezone.utc).isoformat()
        connection.executemany(
            """
            INSERT INTO artifacts(run_id, artifact_type, path, checksum, created_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (run_id, item.artifact_type, item.path, item.checksum, now)
                for item in payload.artifacts
            ],
        )
        connection.executemany(
            """
            INSERT INTO kpis(run_id, period, department, metric, value, unit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.period,
                    item.department,
                    item.metric,
                    item.value,
                    item.unit,
                    item.status,
                )
                for item in payload.kpis
            ],
        )
        connection.executemany(
            """
            INSERT INTO anomalies(
                run_id, anomaly_id, period, department, type, severity,
                metric, values_json, description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.anomaly_id,
                    item.period,
                    item.department,
                    item.type,
                    item.severity,
                    item.metric,
                    item.values_json,
                    item.description,
                )
                for item in payload.anomalies
            ],
        )
        connection.executemany(
            """
            INSERT INTO recommendations(
                run_id, recommendation_id, priority, department, action,
                expected_impact, status, follow_up_required
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.recommendation_id,
                    item.priority,
                    item.department,
                    item.action,
                    item.expected_impact,
                    item.status,
                    int(item.follow_up_required),
                )
                for item in payload.recommendations
            ],
        )
        connection.executemany(
            """
            INSERT INTO goals(run_id, goal_id, metric, target, actual, unit, progress_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.goal_id,
                    item.metric,
                    item.target,
                    item.actual,
                    item.unit,
                    item.progress_status,
                )
                for item in payload.goals
            ],
        )
        connection.executemany(
            """
            INSERT INTO memory_facts(run_id, category, subject, fact, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (run_id, item.category, item.subject, item.fact, item.confidence)
                for item in payload.memory_facts
            ],
        )
