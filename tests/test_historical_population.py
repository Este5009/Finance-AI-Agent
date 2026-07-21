"""Tests for Phase 12B synthetic historical pipeline population."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas

from finance_agent.memory.repository import MemoryRepository
from finance_agent.orchestration import PipelineConfig, PipelineInputModel, PipelineRunResult, PipelineStageResult, RuntimeSummary
from finance_agent.synthetic_history import SyntheticHistoryConfig, generate_synthetic_history
from scripts.populate_synthetic_history import (
    discover_synthetic_period_inputs,
    populate_synthetic_history,
    validate_population_against_manifest,
)


def _copy_history(tmp_path: Path) -> Path:
    """Generate and copy a synthetic history into an isolated project layout.

    Inputs:
        tmp_path: Pytest temporary directory.
    Outputs:
        Synthetic history root.
    Assumptions:
        The generated dataset is small enough for tests to copy locally.
    """

    generated = generate_synthetic_history(SyntheticHistoryConfig(output_directory=tmp_path / "source"))
    project_history = tmp_path / "project" / "data" / "synthetic_history" / "recovery_2026"
    shutil.copytree(generated.root_directory, project_history)
    return project_history


def _period_slug_from_input(input_model: PipelineInputModel) -> str:
    """Extract the period slug from a pipeline input model.

    Inputs:
        input_model: Generic pipeline input.
    Outputs:
        Period slug such as ``2026_06``.
    Assumptions:
        Population always supplies a ``YYYY-MM`` period override.
    """

    return str(input_model.period_override).replace("-", "_")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON fixture file for mocked pipeline output.

    Inputs:
        path: Target path.
        data: JSON-compatible dictionary.
    Outputs:
        None.
    Assumptions:
        Parent directories may be created.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_pdf(path: Path, text: str) -> None:
    """Write a minimal valid PDF for report-quality validation.

    Inputs:
        path: Target PDF path.
        text: Text to draw.
    Outputs:
        None.
    Assumptions:
        Report-quality tests require a parseable PDF artifact.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, text)
    pdf.save()


def _valid_report_model(period_slug: str) -> dict[str, Any]:
    """Return a minimal strategy-backed report model.

    Inputs:
        period_slug: Period slug for the report model.
    Outputs:
        Report model dictionary accepted by report-quality validation.
    Assumptions:
        Tests focus on storage/population, not report rendering fidelity.
    """

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
            section["content"] = {"analysis_status": "accepted", "summary": "Strategy accepted."}
        if section["section_id"] == "strategic_recommendations":
            section["content"] = {
                "recommendations": [{"action": "Reduce Health Sciences overtime."}],
                "root_causes": ["Overtime pressure."],
                "strategic_priorities": ["Payroll control."],
            }
    return {
        "report_id": f"REPORT-{period_slug}",
        "period_slug": period_slug,
        "language": "es",
        "section_count": len(sections),
        "sections": sections,
        "source_references": [],
    }


def _mock_runner_from_manifest(manifest_path: Path):
    """Build a mocked pipeline runner that writes accepted period artifacts.

    Inputs:
        manifest_path: Scenario manifest path.
    Outputs:
        Callable compatible with ``run_pipeline_for_report``.
    Assumptions:
        Persisting happens in the population script after the runner returns.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def runner(input_model: PipelineInputModel, config: PipelineConfig) -> PipelineRunResult:
        """Write storage-compatible artifacts and return a successful result."""

        period_slug = _period_slug_from_input(input_model)
        totals = manifest["monthly_totals"][period_slug]
        outputs = config.output_directory
        kpi_rows = [
            ("total_revenue", totals["actual_revenue"], "USD"),
            ("total_expenses", totals["actual_expense"], "USD"),
            ("payroll_percentage_of_revenue", totals["payroll_ratio"], "ratio"),
            ("student_payment_collection_rate", manifest["collection_rate_trend"][period_slug], "ratio"),
            ("net_cash_flow", totals["net_cash_flow"], "USD"),
        ]
        kpi_path = outputs / "calculations" / f"kpi_summary_{period_slug}.csv"
        kpi_path.parent.mkdir(parents=True, exist_ok=True)
        kpi_path.write_text(
            "metric,value,unit,availability,source\n"
            + "\n".join(f"{metric},{value},{unit},available,mock" for metric, value, unit in kpi_rows),
            encoding="utf-8",
        )
        _write_json(
            outputs / "calculations" / f"finance_summary_{period_slug}.json",
            {
                "report_period": period_slug.replace("_", "-"),
                "finance_summary": {
                    "total_revenue": totals["actual_revenue"],
                    "total_expenses": totals["actual_expense"],
                    "payroll_percentage_of_revenue": totals["payroll_ratio"],
                    "student_payments": {"collection_rate": manifest["collection_rate_trend"][period_slug]},
                },
                "goal_progress": [
                    {
                        "metric": "payroll_percentage_of_revenue",
                        "target": 0.42,
                        "actual": totals["payroll_ratio"],
                        "unit": "ratio",
                        "status": "attention" if totals["payroll_ratio"] > 0.42 else "ok",
                    }
                ],
            },
        )
        anomalies = []
        if period_slug in manifest["health_sciences_overspending_periods"]:
            anomalies.append(
                {
                    "anomaly_id": f"HS-{period_slug}",
                    "period": period_slug,
                    "department": "Health Sciences",
                    "rule_id": "DEPARTMENT_OVERSPEND_FLAG",
                    "severity": "high",
                    "metric": "department_expense_variance_pct",
                    "description": "Health Sciences payroll overspend from overtime.",
                }
            )
        if period_slug in manifest["recurring_vendor_anomaly_periods"]:
            anomalies.append(
                {
                    "anomaly_id": f"VENDOR-{period_slug}",
                    "period": period_slug,
                    "department": "Health Sciences",
                    "rule_id": "VENDOR_PAYMENT_REVIEW",
                    "severity": "high",
                    "metric": "maximum_vendor_payment",
                    "description": "Recurring MedSupply vendor duplicate payment.",
                }
            )
        _write_json(
            outputs / "anomalies" / f"anomaly_report_{period_slug}.json",
            {"report_period": period_slug.replace("_", "-"), "period_slug": period_slug, "anomalies": anomalies},
        )
        recommendations = []
        if period_slug == manifest["recommendation_milestone"]["period"]:
            recommendations.append(
                {
                    "priority": "high",
                    "department": "Health Sciences",
                    "action": "Reduce Health Sciences overtime before September.",
                    "expected_impact": "Lower payroll ratio.",
                }
            )
        else:
            recommendations.append(
                {
                    "priority": "medium",
                    "department": "University",
                    "action": "Monitor monthly recovery milestones.",
                    "expected_impact": "Sustain trend visibility.",
                }
            )
        _write_json(
            outputs / "analysis" / f"strategic_analysis_{period_slug}.json",
            {
                "validation_status": "accepted",
                "recommendation_count": len(recommendations),
                "analysis": {
                    "executive_summary": "Accepted strategy.",
                    "recommendations": recommendations,
                    "confidence": 0.9,
                    "key_findings": [],
                    "root_causes": [],
                    "strategic_priorities": [],
                    "missing_information": [],
                    "reasoning_summary": "Mocked.",
                },
            },
        )
        _write_json(outputs / "evidence" / f"evidence_package_{period_slug}.json", {"packages": []})
        report_model = outputs / "report" / f"report_model_{period_slug}.json"
        html = outputs / "report" / f"financial_report_{period_slug}.html"
        pdf = outputs / "report" / f"financial_report_{period_slug}.pdf"
        _write_json(report_model, _valid_report_model(period_slug))
        html.write_text("<html><body>Strategy accepted. Reduce Health Sciences overtime.</body></html>", encoding="utf-8")
        _write_pdf(pdf, "Strategy accepted. Reduce Health Sciences overtime.")
        stage = PipelineStageResult(
            stage_name="mock",
            display_name="Mock pipeline",
            critical=True,
            success=True,
            skipped=False,
            output_files=(str(report_model), str(html), str(pdf)),
            warnings=(),
            error=None,
            runtime_seconds=0.01,
        )
        return PipelineRunResult(
            success=True,
            stages=(stage,),
            output_files=(str(report_model), str(html), str(pdf)),
            warnings=(),
            runtime_summary=RuntimeSummary(0.01, 1, 1, 1, 0, 0),
            config=config,
        )

    return runner


def test_full_12_period_population_and_idempotent_rerun(tmp_path: Path) -> None:
    """Verify all 12 periods are processed, stored, and idempotent."""

    history = _copy_history(tmp_path)
    database = tmp_path / "project" / "data" / "memory" / "recovery_2026_memory.db"
    output = tmp_path / "project" / "outputs" / "history_population"
    runner = _mock_runner_from_manifest(history / "scenario_manifest.json")

    summary = populate_synthetic_history(
        history_root=history,
        database_path=database,
        output_directory=output,
        project_root=tmp_path / "project",
        verify_idempotency=True,
        runner=runner,
    )

    assert len(summary["successful_periods"]) == 12
    assert not summary["failed_periods"]
    assert summary["idempotency_verified"] is True
    assert summary["table_counts"]["pipeline_runs"] == 12
    assert (output / "population_summary.json").is_file()
    assert (output / "validation_report.json").is_file()


def test_manifest_validation(tmp_path: Path) -> None:
    """Verify stored history patterns match the scenario manifest."""

    history = _copy_history(tmp_path)
    database = tmp_path / "project" / "data" / "memory" / "recovery_2026_memory.db"
    runner = _mock_runner_from_manifest(history / "scenario_manifest.json")
    populate_synthetic_history(
        history_root=history,
        database_path=database,
        output_directory=tmp_path / "project" / "outputs" / "history_population",
        project_root=tmp_path / "project",
        verify_idempotency=False,
        runner=runner,
    )

    validation = validate_population_against_manifest(database, history / "scenario_manifest.json")

    assert validation["valid"] is True
    assert validation["checks"]["payroll_trend_matches"] is True
    assert validation["checks"]["collection_trend_matches"] is True
    assert validation["observed"]["health_sciences_overspending_periods"] == ["2026_04", "2026_05", "2026_06", "2026_07"]
    assert validation["observed"]["recurring_vendor_anomaly_periods"] == ["2026_07", "2026_08", "2026_09"]


def test_isolated_test_database_and_no_production_modification(tmp_path: Path) -> None:
    """Verify population writes only to the dedicated recovery database."""

    history = _copy_history(tmp_path)
    project = tmp_path / "project"
    production_db = project / "data" / "memory" / "finance_memory.db"
    production_db.parent.mkdir(parents=True, exist_ok=True)
    production_db.write_bytes(b"production sentinel")
    database = project / "data" / "memory" / "recovery_2026_memory.db"

    populate_synthetic_history(
        history_root=history,
        database_path=database,
        output_directory=project / "outputs" / "history_population",
        project_root=project,
        verify_idempotency=False,
        runner=_mock_runner_from_manifest(history / "scenario_manifest.json"),
    )

    assert production_db.read_bytes() == b"production sentinel"
    assert MemoryRepository(database).table_counts()["pipeline_runs"] == 12


def test_discovery_pairs_periods_chronologically(tmp_path: Path) -> None:
    """Verify report/goals discovery returns chronological paired inputs."""

    history = _copy_history(tmp_path)

    periods = discover_synthetic_period_inputs(history)

    assert [period.period_slug for period in periods] == [f"2026_{month:02d}" for month in range(1, 13)]
    assert all(period.report_path.is_file() and period.goals_path.is_file() for period in periods)


def test_validation_fails_when_database_is_missing_periods(tmp_path: Path) -> None:
    """Verify manifest validation catches incomplete historical storage."""

    history = _copy_history(tmp_path)
    database = tmp_path / "partial.db"
    runner = _mock_runner_from_manifest(history / "scenario_manifest.json")
    periods = discover_synthetic_period_inputs(history)[:1]
    from scripts.populate_synthetic_history import _run_population_pass

    _run_population_pass(
        periods=periods,
        database_path=database,
        project_root=tmp_path / "project",
        language="es",
        model="qwen3:30b-a3b",
        ollama_timeout_seconds=1,
        stage_timeout_seconds=1,
        runner=runner,
        pass_name="partial",
    )

    validation = validate_population_against_manifest(database, history / "scenario_manifest.json")

    assert validation["valid"] is False
    assert validation["checks"]["all_periods_stored"] is False
