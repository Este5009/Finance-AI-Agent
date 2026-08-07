"""SQLite database initialization and connection helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from finance_agent.memory.models import DEFAULT_MEMORY_DB_PATH, SCHEMA_VERSION


def schema_path() -> Path:
    """Return the bundled SQLite schema file path.

    Inputs: none.
    Outputs: path to schema.sql.
    Assumptions: schema.sql is packaged beside this module.
    """

    return Path(__file__).with_name("schema.sql")


def connect_database(database_path: str | Path = DEFAULT_MEMORY_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled.

    Inputs: database path.
    Outputs: sqlite3 connection.
    Assumptions: callers close the connection or use repository context helpers.
    """

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path = DEFAULT_MEMORY_DB_PATH) -> Path:
    """Create or migrate the memory database schema.

    Inputs: database path.
    Outputs: resolved database path.
    Assumptions: current migrations are additive and represented by schema.sql.
    """

    path = Path(database_path).resolve()
    with connect_database(path) as connection:
        with connection:
            connection.executescript(schema_path().read_text(encoding="utf-8"))
            _ensure_goal_comparison_columns(connection)
            version = connection.execute(
                "SELECT MAX(version) AS version FROM schema_version"
            ).fetchone()["version"]
            if version is None:
                connection.execute(
                    "INSERT INTO schema_version(version, applied_at_utc) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
                )
            elif int(version) < SCHEMA_VERSION:
                # Current migrations are additive and idempotent through
                # schema.sql. Recording the new version makes old databases
                # visibly upgraded without rewriting existing accepted runs.
                connection.execute(
                    "INSERT OR IGNORE INTO schema_version(version, applied_at_utc) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
                )
            elif int(version) > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {version} is newer than supported {SCHEMA_VERSION}."
                )
    return path


def _ensure_goal_comparison_columns(connection: sqlite3.Connection) -> None:
    """Add nullable goal-comparison columns to existing databases.

    Inputs: open SQLite connection.
    Outputs: schema migrated in place when needed.
    Assumptions: columns are additive and therefore safe for accepted history.
    """

    existing = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(goals)").fetchall()
    }
    for column_name, column_type in {
        "gap": "REAL",
        "score": "REAL",
        "direction": "TEXT",
        "source_provenance_json": "TEXT",
    }.items():
        if column_name not in existing:
            connection.execute(f"ALTER TABLE goals ADD COLUMN {column_name} {column_type}")
