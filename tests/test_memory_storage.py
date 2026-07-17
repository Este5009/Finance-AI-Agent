"""Tests for Phase 11A SQLite historical storage and memory index."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas

from finance_agent.memory.database import initialize_database
from finance_agent.memory.models import (
    AnomalyRecord,
    ArtifactRecord,
    StoredPipelineRun,
)
from finance_agent.memory.repository import MemoryRepository
from finance_agent.memory.run_storage import persist_pipeline_run
from finance_agent.orchestration.pipeline_models import (
    DetectedPeriod,
    PipelineConfig,
    PipelineInputModel,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
)
from finance_agent.orchestration.pipeline_orchestrator import (
    _cache_manifest_path,
    _pipeline_cache_key,
    run_pipeline_for_report,
)


def _input_model(tmp_path: Path) -> PipelineInputModel:
    """Build a generic pipeline input fixture with existing files."""

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"financial report")
    goals.write_bytes(b"%PDF goals")
    return PipelineInputModel(
        financial_report_path=report,
        goals_document_path=goals,
        detected_period=DetectedPeriod(
            period_type="monthly",
            label="2026-06",
            confidence=0.95,
            year=2026,
            month=6,
        ),
        period_type="monthly",
        period_override="2026-06",
        report_language="es",
    )


def _config(tmp_path: Path, input_model: PipelineInputModel) -> PipelineConfig:
    """Build a temp pipeline config pointing storage at a temp DB."""

    return PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        input_model=input_model,
        memory_database_path=tmp_path / "data" / "memory" / "finance_memory.db",
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON fixture file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_pdf(path: Path, text: str) -> None:
    """Write a simple text PDF fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, text)
    pdf.save()


def _valid_report_model(period_slug: str) -> dict[str, Any]:
    """Return a minimal report model that passes quality validation."""

    sections = [
        {
            "section_id": section_id,
            "title": section_id,
            "content": {},
            "source_references": [],
            "warnings": [],
        }
        for section_id in (
            "cover",
            "executive_summary",
            "financial_health_overview",
            "kpi_overview",
            "revenue_analysis",
            "expense_analysis",
            "department_analysis",
            "anomaly_summary",
            "investigation_evidence",
            "strategic_recommendations",
            "missing_information",
            "appendix",
        )
    ]
    for section in sections:
        if section["section_id"] == "executive_summary":
            section["content"] = {
                "analysis_status": "accepted",
                "summary": "Accepted strategy summary.",
            }
        if section["section_id"] == "strategic_recommendations":
            section["content"] = {
                "recommendations": [{"action": "Control payroll variance."}],
                "root_causes": ["Payroll pressure."],
                "strategic_priorities": ["Stabilize cash."],
            }
    return {
        "report_id": "REPORT-MODEL-TEST",
        "period_slug": period_slug,
        "report_period": "2026-06",
        "language": "es",
        "section_count": len(sections),
        "sections": sections,
        "source_references": [],
    }


def _write_artifacts(config: PipelineConfig, period_slug: str = "2026_06") -> tuple[str, ...]:
    """Write processed artifacts required by memory storage."""

    outputs = config.output_directory
    _write_json(
        outputs / "calculations" / f"finance_summary_{period_slug}.json",
        {
            "report_period": "2026-06",
            "goal_progress": [
                {
                    "goal_id": "GOAL-1",
                    "metric": "collection_rate",
                    "target": 0.9,
                    "actual": 0.84,
                    "unit": "ratio",
                    "status": "behind",
                }
            ],
        },
    )
    kpi_path = outputs / "calculations" / f"kpi_summary_{period_slug}.csv"
    kpi_path.parent.mkdir(parents=True, exist_ok=True)
    kpi_path.write_text(
        "metric,value,unit,availability\ncollection_rate,0.84,ratio,available\n",
        encoding="utf-8",
    )
    _write_json(
        outputs / "anomalies" / f"anomaly_report_{period_slug}.json",
        {
            "report_period": "2026-06",
            "anomalies": [
                {
                    "anomaly_id": "ANOM-1",
                    "period": "2026-06",
                    "department": "Health Sciences",
                    "rule_id": "PAYROLL_SPIKE",
                    "severity": "high",
                    "metric": "payroll_total",
                    "observed_value": 120,
                    "threshold_value": 100,
                    "evidence": "Payroll exceeded threshold.",
                }
            ],
        },
    )
    _write_json(outputs / "anomalies" / f"risk_summary_{period_slug}.json", {"top_risks": []})
    _write_json(
        outputs / "evidence" / f"evidence_package_{period_slug}.json",
        {
            "evidence_packages": [
                {
                    "task_id": "STEP-1",
                    "evidence_summary": "Retrieved payroll evidence.",
                    "confidence": 0.9,
                }
            ]
        },
    )
    _write_json(outputs / "evidence" / f"retrieval_summary_{period_slug}.json", {})
    analysis = {
        "validation_status": "accepted",
        "analysis": {
            "executive_summary": "Accepted summary.",
            "key_findings": ["Payroll is elevated."],
            "root_causes": ["Overtime pressure."],
            "strategic_priorities": ["Stabilize payroll."],
            "missing_information": ["Approval notes."],
            "confidence": 0.8,
            "recommendations": [
                {
                    "priority": "high",
                    "action": "Review overtime approvals.",
                    "expected_impact": "Lower payroll variance.",
                }
            ],
        },
        "recommendation_count": 1,
    }
    _write_json(outputs / "analysis" / f"strategic_analysis_{period_slug}.json", analysis)
    report_model_path = outputs / "report" / f"report_model_{period_slug}.json"
    _write_json(report_model_path, _valid_report_model(period_slug))
    html_path = outputs / "report" / f"financial_report_{period_slug}.html"
    html_path.write_text("<html>Accepted strategy summary.</html>", encoding="utf-8")
    _write_pdf(outputs / "report" / f"financial_report_{period_slug}.pdf", "Accepted strategy summary.")
    normalized = outputs / "intermediate" / period_slug / "normalized_tables"
    normalized.mkdir(parents=True, exist_ok=True)
    (normalized / "Payroll.csv").write_text("department,total_payroll\nHealth,120\n", encoding="utf-8")
    return tuple(str(path) for path in outputs.rglob("*") if path.is_file())


def _pipeline_result(config: PipelineConfig, output_files: tuple[str, ...]) -> PipelineRunResult:
    """Build a successful pipeline result fixture."""

    stage = PipelineStageResult(
        stage_name="report_generation",
        display_name="Report generation",
        critical=False,
        success=True,
        skipped=False,
        output_files=output_files,
        warnings=(),
        error=None,
        runtime_seconds=0.1,
    )
    return PipelineRunResult(
        success=True,
        stages=(stage,),
        output_files=output_files,
        warnings=(),
        runtime_summary=RuntimeSummary(0.1, 1, 1, 1, 0, 0),
        config=config,
    )


def test_schema_creation_and_migration(tmp_path: Path) -> None:
    """Verify schema initialization creates versioned tables."""

    db_path = initialize_database(tmp_path / "memory.db")

    with sqlite3.connect(db_path) as connection:
        version = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert version == 1
    assert "pipeline_runs" in tables
    assert "memory_facts" in tables


def test_successful_run_persistence_and_table_counts(tmp_path: Path) -> None:
    """Verify an accepted run stores all requested record categories."""

    input_model = _input_model(tmp_path)
    config = _config(tmp_path, input_model)
    output_files = _write_artifacts(config)

    storage = persist_pipeline_run(
        _pipeline_result(config, output_files),
        period_slug="2026_06",
        database_path=config.memory_database_path,
    )

    assert storage.stored is True
    assert storage.table_counts["pipeline_runs"] == 1
    assert storage.table_counts["artifacts"] >= 7
    assert storage.table_counts["kpis"] == 1
    assert storage.table_counts["anomalies"] == 1
    assert storage.table_counts["recommendations"] == 1
    assert storage.table_counts["goals"] == 1
    assert storage.table_counts["memory_facts"] >= 5


def test_artifact_references_include_checksums(tmp_path: Path) -> None:
    """Verify artifacts are referenced by path and checksum, not blobs."""

    input_model = _input_model(tmp_path)
    config = _config(tmp_path, input_model)
    output_files = _write_artifacts(config)
    storage = persist_pipeline_run(
        _pipeline_result(config, output_files),
        period_slug="2026_06",
        database_path=config.memory_database_path,
    )

    with sqlite3.connect(storage.database_path) as connection:
        rows = connection.execute("SELECT path, checksum FROM artifacts").fetchall()

    assert rows
    assert all(row[0] for row in rows)
    assert all(row[1] for row in rows)


def test_idempotent_repeated_runs_update_existing_record(tmp_path: Path) -> None:
    """Verify reprocessing equivalent inputs does not create duplicate runs."""

    input_model = _input_model(tmp_path)
    config = _config(tmp_path, input_model)
    output_files = _write_artifacts(config)
    result = _pipeline_result(config, output_files)

    first = persist_pipeline_run(result, period_slug="2026_06", database_path=config.memory_database_path)
    second = persist_pipeline_run(result, period_slug="2026_06", database_path=config.memory_database_path)

    assert first.run_id == second.run_id
    assert second.updated_existing is True
    assert second.table_counts["pipeline_runs"] == 1
    assert second.table_counts["recommendations"] == 1


def test_rollback_on_failure(tmp_path: Path) -> None:
    """Verify child insert failure rolls back the whole transaction."""

    repository = MemoryRepository(tmp_path / "memory.db")
    payload = StoredPipelineRun(
        run_id="RUN-ROLLBACK",
        idempotency_key="rollback",
        period="2026_06",
        period_type="monthly",
        started_at_utc=None,
        completed_at_utc="2026-07-17T00:00:00+00:00",
        report_hash="report",
        goals_hash="goals",
        report_path="report.xlsx",
        goals_path="goals.pdf",
        language="es",
        model="qwen3:30b-a3b",
        confidence=0.8,
        cache_hit=False,
        cache_key=None,
        status="completed",
        artifact_directory="outputs",
        configuration_json="{}",
        artifacts=(ArtifactRecord("json", "missing.json", None),),
        anomalies=(
            AnomalyRecord("DUP", None, None, None, "high", "metric", "{}", "one"),
            AnomalyRecord("DUP", None, None, None, "high", "metric", "{}", "two"),
        ),
    )

    try:
        repository.save_pipeline_run(payload)
    except sqlite3.IntegrityError:
        pass

    assert repository.table_counts()["pipeline_runs"] == 0


def test_rejected_strategy_is_not_stored(tmp_path: Path) -> None:
    """Verify unavailable/rejected strategy artifacts are skipped."""

    input_model = _input_model(tmp_path)
    config = _config(tmp_path, input_model)
    output_files = _write_artifacts(config)
    analysis_path = config.output_directory / "analysis" / "strategic_analysis_2026_06.json"
    _write_json(analysis_path, {"validation_status": "unavailable"})

    storage = persist_pipeline_run(
        _pipeline_result(config, output_files),
        period_slug="2026_06",
        database_path=config.memory_database_path,
    )

    assert storage.stored is False
    assert storage.reason == "strategic analysis was not accepted"
    assert storage.table_counts["pipeline_runs"] == 0


def test_pipeline_cache_hit_integration_invokes_memory_storage(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Verify run_pipeline_for_report calls storage for valid cache-hit runs."""

    input_model = _input_model(tmp_path)
    config = _config(tmp_path, input_model)
    _write_artifacts(config)
    cache_key = _pipeline_cache_key(input_model, config)
    manifest = _cache_manifest_path(config, cache_key)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"cache_key": cache_key}), encoding="utf-8")
    calls: list[str] = []

    def fake_persist(
        result: PipelineRunResult,
        *,
        period_slug: str,
        database_path: Path,
    ) -> object:
        """Capture integration call without touching SQLite."""

        calls.append(period_slug)
        return object()

    monkeypatch.setattr(
        "finance_agent.memory.run_storage.persist_pipeline_run",
        fake_persist,
    )

    result = run_pipeline_for_report(input_model, config)

    assert result.cache_hit is True
    assert calls == ["2026_06"]
