"""Regression tests for explicit synthetic-history import into app-data DBs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from finance_agent.memory.database import initialize_database
from finance_agent.memory.retrieval import get_metric_history
from scripts.import_synthetic_history import import_synthetic_history


def _seed_source(path: Path) -> None:
    """Create two accepted synthetic months with KPI rows."""

    initialize_database(path)
    with sqlite3.connect(path) as connection:
        for month, value in (("2026_04", 104.0), ("2026_05", 105.0)):
            run_id = f"run-{month}"
            connection.execute(
                """INSERT INTO pipeline_runs (
                    run_id,idempotency_key,period,period_type,completed_at_utc,report_hash,goals_hash,
                    report_path,goals_path,language,model,status,artifact_directory,configuration_json,updated_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, f"key-{month}", month, "monthly", "2026-06-01T00:00:00Z", "r", "g",
                 "synthetic.xlsx", "", "es", "qwen3:8b", "completed", "outputs", "{}", "2026-06-01T00:00:00Z"),
            )
            connection.execute(
                "INSERT INTO kpis (run_id,period,metric,value,unit,status) VALUES (?,?,?,?,?,?)",
                (run_id, month, "total_revenue", value, "USD", "actual"),
            )


def test_explicit_import_merges_only_requested_completed_periods(tmp_path: Path) -> None:
    """Synthetic rows enter an explicitly selected target without fabrication."""

    source = tmp_path / "source.db"
    target = tmp_path / "app-data" / "finance_memory.db"
    _seed_source(source)

    summary = import_synthetic_history(source, target, periods=("2026_04", "2026_05"))
    history = get_metric_history(
        "total_revenue", 5, before_period="2026_09", database_path=target,
    )

    assert summary["resulting_completed_periods"] == ["2026_04", "2026_05"]
    assert [(row["period"], row["value"]) for row in history.data["records"]] == [
        ("2026_04", 104.0), ("2026_05", 105.0),
    ]


def test_explicit_import_is_idempotent(tmp_path: Path) -> None:
    """Repeating the same import does not duplicate runs or KPI points."""

    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _seed_source(source)
    import_synthetic_history(source, target, periods=("2026_04",))
    second = import_synthetic_history(source, target, periods=("2026_04",))

    assert second["imported_runs"] == 0
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM kpis").fetchone()[0] == 1
