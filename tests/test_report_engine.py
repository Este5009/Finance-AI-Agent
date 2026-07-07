"""Tests for renderer-agnostic report model generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finance_agent.reporting import (
    REQUIRED_SECTION_IDS,
    ReportInputBundle,
    build_report_model,
    save_report_model,
    validate_report_model,
)


def _bundle() -> ReportInputBundle:
    """Build a minimal processed-input report fixture.

    Inputs: none.
    Outputs: ReportInputBundle with representative processed artifacts.
    Assumptions: values are copied from upstream stages, not recalculated here.
    """

    finance_summary = {
        "report_period": "June 2026",
        "source_workbook": "monthly_financial_report_june_2026.xlsx",
        "finance_summary": {
            "total_revenue": 1000,
            "total_expenses": 1200,
            "net_operating_result": -200,
            "payroll_total": 500,
            "payroll_percentage_of_revenue": 0.5,
            "budget_vs_actual": {
                "revenue_budget": 1100,
                "revenue_variance": -100,
                "expense_budget": 1000,
                "expense_variance": 200,
            },
            "student_payments": {"collection_rate": 0.9},
            "cash_flow": {"net_cash_flow": -300, "ending_cash": 5000},
        },
        "department_summary": [{"department": "Engineering", "actual_expense": 600}],
        "category_summary": [{"category": "Payroll", "actual_amount": 500}],
        "calculation_warnings": [],
    }
    anomaly_report = {
        "total_anomalies": 1,
        "anomalies_by_severity": {"critical": 1},
        "anomalies": [
            {
                "anomaly_id": "ANOM-1",
                "title": "Operating deficit",
                "severity": "critical",
            }
        ],
    }
    evidence_package = {
        "summary": {"tasks_executed": 1, "successful_retrievals": 1},
        "evidence_packages": [
            {
                "task_id": "TASK-1",
                "priority": "critical",
                "investigation_question": "What caused the deficit?",
                "evidence_summary": "Retrieved processed report.",
                "retrieved_evidence": {
                    "retrieval_name": "financial_report",
                    "success": True,
                    "data": {"record_count": 1},
                    "source_references": ["outputs/calculations/finance_summary.json"],
                    "warnings": [],
                    "unavailable_data": [],
                },
            }
        ],
    }
    strategic_analysis = {
        "validation_status": "accepted",
        "validation_errors": [],
        "analysis": {
            "executive_summary": "Performance requires management attention.",
            "key_findings": ["Operating result is negative."],
            "root_causes": ["Expenses grew faster than revenue."],
            "recommendations": [{"action": "Review spending approvals."}],
            "strategic_priorities": ["Stabilize cash flow."],
            "missing_information": ["Vendor invoice notes."],
            "confidence": 0.8,
            "reasoning_summary": "Evidence supports the deficit concern.",
        },
    }
    return ReportInputBundle(
        period_slug="june_2026",
        finance_summary=finance_summary,
        kpi_summary=(
            {
                "metric": "total_revenue",
                "value": "1000",
                "unit": "USD",
                "availability": "available",
                "source": "Revenue",
            },
        ),
        anomaly_report=anomaly_report,
        evidence_package=evidence_package,
        strategic_analysis=strategic_analysis,
        source_files=(
            "finance_summary_june_2026.json",
            "kpi_summary_june_2026.csv",
            "anomaly_report_june_2026.json",
            "evidence_package_june_2026.json",
            "strategic_analysis_june_2026.json",
        ),
    )


def test_report_model_generation_contains_required_sections() -> None:
    """Verify report generation creates every required section in order."""

    model = build_report_model(_bundle())

    assert [section.section_id for section in model.sections] == list(REQUIRED_SECTION_IDS)
    assert model.report_id == "REPORT-MODEL-JUNE-2026"
    assert model.report_period == "June 2026"


def test_required_section_validation_accepts_valid_model() -> None:
    """Verify a generated model satisfies the internal schema validator."""

    model = build_report_model(_bundle())

    validate_report_model(model.to_dict())


def test_missing_section_handling_raises_clear_error() -> None:
    """Verify missing required sections are rejected."""

    data = build_report_model(_bundle()).to_dict()
    data["sections"] = [
        section for section in data["sections"] if section["section_id"] != "appendix"
    ]
    data["section_count"] = len(data["sections"])

    with pytest.raises(ValueError, match="missing required sections"):
        validate_report_model(data)


def test_source_reference_preservation() -> None:
    """Verify source references are preserved at section and report levels."""

    model = build_report_model(_bundle())
    data = model.to_dict()
    section_by_id = {section["section_id"]: section for section in data["sections"]}

    assert "finance_summary_june_2026.json" in section_by_id["cover"]["source_references"]
    assert "evidence_package_june_2026.json" in section_by_id["investigation_evidence"]["source_references"]
    assert "strategic_analysis_june_2026.json" in data["source_references"]


def test_strategic_analysis_fields_are_preserved_for_renderers() -> None:
    """Verify report models keep strategic analysis fields needed by renderers."""

    data = build_report_model(_bundle()).to_dict()
    section_by_id = {section["section_id"]: section for section in data["sections"]}

    executive = section_by_id["executive_summary"]["content"]
    recommendations = section_by_id["strategic_recommendations"]["content"]
    missing = section_by_id["missing_information"]["content"]

    assert executive["summary"] == "Performance requires management attention."
    assert executive["key_findings"] == ["Operating result is negative."]
    assert executive["root_causes"] == ["Expenses grew faster than revenue."]
    assert recommendations["recommendations"] == [{"action": "Review spending approvals."}]
    assert recommendations["strategic_priorities"] == ["Stabilize cash flow."]
    assert recommendations["root_causes"] == ["Expenses grew faster than revenue."]
    assert missing["missing_information"] == ["Vendor invoice notes."]


def test_json_schema_validation_and_save(tmp_path: Path) -> None:
    """Verify saved report model JSON keeps the expected schema."""

    model = build_report_model(_bundle())
    output_path = save_report_model(model, tmp_path / "report_model.json")
    data = json.loads(output_path.read_text(encoding="utf-8"))

    validate_report_model(data)
    assert set(data) == {
        "report_id",
        "period_slug",
        "report_period",
        "renderer_contract_version",
        "section_count",
        "sections",
        "source_references",
    }
    assert data["section_count"] == len(REQUIRED_SECTION_IDS)
