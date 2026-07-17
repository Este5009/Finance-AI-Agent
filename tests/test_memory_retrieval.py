"""Tests for Phase 11B historical memory retrieval tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finance_agent.memory.models import (
    AnomalyRecord,
    ArtifactRecord,
    GoalRecord,
    KpiRecord,
    MemoryFactRecord,
    RecommendationRecord,
    StoredPipelineRun,
)
from finance_agent.memory.repository import MemoryRepository
from finance_agent.memory.retrieval import (
    get_artifact_references,
    get_department_history,
    get_full_period_record,
    get_goal_progress,
    get_memory_facts,
    get_metric_history,
    get_period_history,
    get_previous_period,
    get_previous_recommendations,
    get_repeated_anomalies,
)
from finance_agent.retrieval.retrieval_registry import create_default_registry


SEEDED_PERIODS = tuple(f"2026_{month:02d}" for month in range(1, 7))


def _payload(period: str, index: int) -> StoredPipelineRun:
    """Build one stored run payload for controlled history tests."""

    payroll_ratio = 0.36 + (index * 0.01)
    collection_rate = 0.84 + (index * 0.015)
    recommendations: tuple[RecommendationRecord, ...] = ()
    if period == "2026_03":
        recommendations = (
            RecommendationRecord(
                recommendation_id="REC-OT-001",
                priority="high",
                department="Health Sciences",
                action="Review overtime approvals.",
                expected_impact="Reduce overtime variance.",
                status="unknown",
                follow_up_required=True,
            ),
        )
    return StoredPipelineRun(
        run_id=f"RUN-{period}",
        idempotency_key=f"key-{period}",
        period=period,
        period_type="monthly",
        started_at_utc=None,
        completed_at_utc=f"2026-0{index + 1}-28T00:00:00+00:00",
        report_hash=f"report-{period}",
        goals_hash=f"goals-{period}",
        report_path=f"reports/{period}.xlsx",
        goals_path="goals/goals.pdf",
        language="es",
        model="qwen3:30b-a3b",
        confidence=0.75 + index * 0.01,
        cache_hit=False,
        cache_key=None,
        status="completed",
        artifact_directory="outputs",
        configuration_json=json.dumps({"period": period}),
        artifacts=(
            ArtifactRecord(
                "report_pdf",
                f"outputs/report/financial_report_{period}.pdf",
                f"checksum-pdf-{period}",
            ),
            ArtifactRecord(
                "finance_summary",
                f"outputs/calculations/finance_summary_{period}.json",
                f"checksum-finance-{period}",
            ),
        ),
        kpis=(
            KpiRecord(period, None, "payroll_percentage_of_revenue", payroll_ratio, "ratio", "available"),
            KpiRecord(period, None, "collection_rate", collection_rate, "ratio", "available"),
            KpiRecord(period, "Health Sciences", "department_budget_variance", 0.13 + index * 0.005, "ratio", "flag"),
        ),
        anomalies=(
            AnomalyRecord(
                f"ANOM-HS-{period}",
                period,
                "Health Sciences",
                "DEPARTMENT_OVERSPEND",
                "high",
                "department_budget_variance",
                json.dumps({"observed_value": 0.13 + index * 0.005}),
                "Health Sciences overspending exceeded threshold.",
            ),
            AnomalyRecord(
                f"ANOM-VENDOR-{period}",
                period,
                None,
                "VENDOR_ANOMALY",
                "high",
                "vendor_payment_amount",
                json.dumps({"vendor": "MedSupply Co", "amount": 50000 + index}),
                "Recurring vendor anomaly for MedSupply Co.",
            ),
        ),
        recommendations=recommendations,
        goals=(
            GoalRecord(
                f"GOAL-COLL-{period}",
                "collection_rate",
                0.94,
                collection_rate,
                "ratio",
                "improving",
            ),
        ),
        memory_facts=(
            MemoryFactRecord(
                "root_cause",
                "overtime",
                "Health Sciences overtime is a recurring pressure.",
                0.8,
            ),
            MemoryFactRecord(
                "strategic_priority",
                "collections",
                "Collection rate is improving but below target.",
                0.82,
            ),
        ),
    )


@pytest.fixture()
def seeded_database(tmp_path: Path) -> Path:
    """Create a temp SQLite memory database with six controlled periods."""

    db_path = tmp_path / "memory.db"
    repository = MemoryRepository(db_path)
    for index, period in enumerate(SEEDED_PERIODS):
        repository.save_pipeline_run(_payload(period, index))
    return db_path


def test_previous_period_and_period_history_ordering(seeded_database: Path) -> None:
    """Verify chronological period navigation excludes future periods."""

    previous = get_previous_period("2026_06", database_path=seeded_database)
    history = get_period_history(3, before_period="2026_06", database_path=seeded_database)

    assert previous.success is True
    assert previous.data["record"]["period"] == "2026_05"
    assert [row["period"] for row in history.data["records"]] == [
        "2026_03",
        "2026_04",
        "2026_05",
    ]
    assert "2026_06" not in [row["period"] for row in history.data["records"]]


def test_metric_history_filters_and_orders(seeded_database: Path) -> None:
    """Verify metric history returns exact payroll ratio values in order."""

    result = get_metric_history(
        "payroll_percentage_of_revenue",
        6,
        database_path=seeded_database,
    )

    assert result.success is True
    assert [row["period"] for row in result.data["records"]] == list(SEEDED_PERIODS)
    assert result.data["records"][0]["value"] == pytest.approx(0.36)
    assert result.data["records"][-1]["value"] == pytest.approx(0.41)


def test_department_history_summary_and_full(seeded_database: Path) -> None:
    """Verify department history supports summary and full detail levels."""

    summary = get_department_history(
        "Health Sciences",
        6,
        database_path=seeded_database,
    )
    full = get_department_history(
        "Health Sciences",
        6,
        detail_level="full",
        database_path=seeded_database,
    )

    assert summary.success is True
    assert summary.data["counts"]["kpis"] == 6
    assert summary.data["counts"]["anomalies"] == 6
    assert "records" not in summary.data
    assert full.data["records"]["anomalies"][0]["department"] == "Health Sciences"


def test_repeated_anomalies_and_filters(seeded_database: Path) -> None:
    """Verify repeated anomaly grouping and department filters."""

    result = get_repeated_anomalies(6, min_occurrences=2, database_path=seeded_database)
    filtered = get_repeated_anomalies(
        6,
        department="Health Sciences",
        min_occurrences=2,
        database_path=seeded_database,
    )

    metrics = {row["metric"] for row in result.data["records"]}
    assert {"department_budget_variance", "vendor_payment_amount"} <= metrics
    assert [row["metric"] for row in filtered.data["records"]] == [
        "department_budget_variance"
    ]


def test_previous_recommendations_and_goal_progress(seeded_database: Path) -> None:
    """Verify prior recommendations and measurable goal progress retrieval."""

    recs = get_previous_recommendations(
        6,
        department="Health Sciences",
        before_period="2026_06",
        database_path=seeded_database,
    )
    goals = get_goal_progress(
        "collection_rate",
        6,
        database_path=seeded_database,
    )

    assert recs.success is True
    assert recs.data["records"][0]["action"] == "Review overtime approvals."
    assert goals.success is True
    assert len(goals.data["records"]) == 6
    assert goals.data["records"][-1]["actual"] > goals.data["records"][0]["actual"]


def test_memory_facts_full_record_and_artifacts(seeded_database: Path) -> None:
    """Verify fact, full-period, and artifact-reference retrieval."""

    facts = get_memory_facts(category="root_cause", periods=6, database_path=seeded_database)
    full = get_full_period_record("2026_06", database_path=seeded_database)
    artifacts = get_artifact_references(
        "2026_06",
        artifact_type="report_pdf",
        database_path=seeded_database,
    )

    assert facts.success is True
    assert facts.data["record_count"] == 6
    assert full.success is True
    assert len(full.data["kpis"]) == 3
    assert len(full.data["anomalies"]) == 2
    assert artifacts.success is True
    assert artifacts.data["records"][0]["checksum"] == "checksum-pdf-2026_06"


def test_missing_invalid_and_read_only_behavior(seeded_database: Path) -> None:
    """Verify missing history is explicit, invalid input fails, and reads do not write."""

    repository = MemoryRepository(seeded_database)
    before = repository.table_counts()
    missing = get_metric_history("net_cash_flow", 6, database_path=seeded_database)

    with pytest.raises(ValueError):
        get_metric_history("metric; DROP TABLE kpis", 6, database_path=seeded_database)
    with pytest.raises(ValueError):
        get_department_history("Health Sciences", 6, detail_level="everything", database_path=seeded_database)

    after = repository.table_counts()
    assert missing.success is False
    assert missing.unavailable_data
    assert before == after


def test_registry_integration_preserves_current_period_tools() -> None:
    """Verify memory tools are registered without replacing existing retrieval names."""

    registry = create_default_registry()

    assert "get_payroll_history" in registry.names()
    assert "get_department_history" in registry.names()
    assert registry.get("get_department_history").function.__name__ == "retrieve_department_history"
    assert "get_metric_history" in registry.names()
    assert "get_memory_department_history" in registry.names()
