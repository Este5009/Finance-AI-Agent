"""Repository layer for transactional Finance AI Agent memory storage."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from finance_agent.memory.database import connect_database, initialize_database
from finance_agent.memory.models import (
    DEFAULT_MEMORY_DB_PATH,
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

        tables = ("pipeline_runs", *CHILD_TABLES)
        with connect_database(self.database_path) as connection:
            return {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in tables
            }

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
            f"SELECT child.*, runs.period AS run_period, runs.period_type "
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
                    self._replace_children(connection, run_id, payload)
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
