"""Explicitly import accepted synthetic runs into a selected runtime database.

This developer/demo utility never runs automatically. It merges only completed
periods named by the caller, leaving production databases isolated unless the
operator deliberately supplies their path and the confirmation flag.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.memory.database import initialize_database  # noqa: E402


CHILD_TABLES = (
    "kpis", "anomalies", "recommendations", "goals", "memory_facts", "artifacts",
)
INTEGER_PRIMARY_KEYS = {
    "kpis": "kpi_id", "anomalies": "row_id", "memory_facts": "fact_id", "artifacts": "artifact_id",
}


def import_synthetic_history(
    source_database: str | Path,
    target_database: str | Path,
    *,
    periods: tuple[str, ...],
) -> dict[str, Any]:
    """Merge selected completed synthetic periods into one explicit target DB.

    Inputs: source DB, target DB, and exact monthly period slugs.
    Outputs: imported run/child counts and resulting period list.
    Assumptions: source and target use compatible Finance AI memory schemas;
    synthetic rows retain their provenance and are never imported implicitly.
    """

    source_path = Path(source_database).expanduser().resolve()
    target_path = Path(target_database).expanduser().resolve()
    if source_path == target_path:
        raise ValueError("source and target databases must be different")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    normalized_periods = tuple(sorted({_validate_period(period) for period in periods}))
    if not normalized_periods:
        raise ValueError("at least one --period is required")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(target_path)

    imported_runs = 0
    imported_children = {table: 0 for table in CHILD_TABLES}
    with sqlite3.connect(source_path) as source, sqlite3.connect(target_path) as target:
        source.row_factory = sqlite3.Row
        source_runs = source.execute(
            f"SELECT * FROM pipeline_runs WHERE status='completed' AND period IN ({','.join('?' for _ in normalized_periods)}) ORDER BY period, updated_at_utc",
            normalized_periods,
        ).fetchall()
        latest_by_period = {str(row["period"]): row for row in source_runs}
        with target:
            for period in normalized_periods:
                row = latest_by_period.get(period)
                if row is None:
                    raise ValueError(f"source database has no completed synthetic run for {period}")
                run_added = _insert_row(target, "pipeline_runs", dict(row), replace=False)
                imported_runs += run_added
                run_id = str(row["run_id"])
                if not run_added:
                    continue
                for table in CHILD_TABLES:
                    if not _table_exists(source, table) or not _table_exists(target, table):
                        continue
                    for child in source.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)).fetchall():
                        values = dict(child)
                        values.pop(INTEGER_PRIMARY_KEYS.get(table, ""), None)
                        imported_children[table] += _insert_row(target, table, values, replace=False)
    with sqlite3.connect(target_path) as target:
        resulting_periods = [row[0] for row in target.execute(
            "SELECT DISTINCT period FROM pipeline_runs WHERE status='completed' ORDER BY period"
        )]
    return {
        "source_database": str(source_path), "target_database": str(target_path),
        "requested_periods": list(normalized_periods), "imported_runs": imported_runs,
        "imported_children": imported_children, "resulting_completed_periods": resulting_periods,
    }


def _insert_row(connection: sqlite3.Connection, table: str, values: dict[str, Any], *, replace: bool) -> int:
    """Insert one schema-compatible row and return whether it was added."""

    target_columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    compatible = {key: value for key, value in values.items() if key in target_columns}
    columns = list(compatible)
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    cursor = connection.execute(
        f"{verb} INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        tuple(compatible[column] for column in columns),
    )
    return max(cursor.rowcount, 0)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    """Return whether an exact allowlisted table exists."""

    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _validate_period(period: str) -> str:
    """Normalize and validate a monthly 20xx period slug."""

    normalized = str(period).replace("-", "_")
    if re.fullmatch(r"20\d{2}_(0[1-9]|1[0-2])", normalized) is None:
        raise ValueError(f"invalid monthly period: {period}")
    return normalized


def main() -> int:
    """Run the guarded synthetic import command."""

    parser = argparse.ArgumentParser(description="Explicitly import synthetic Finance AI history.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--period", action="append", required=True)
    parser.add_argument("--confirm-synthetic-data", action="store_true", required=True)
    args = parser.parse_args()
    summary = import_synthetic_history(args.source, args.target, periods=tuple(args.period))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
