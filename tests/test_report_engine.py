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
from finance_agent.reporting.presentation import build_presentation_view
from finance_agent.reporting.report_engine import _previous_month_slug, load_report_inputs


def test_january_monthly_comparison_uses_previous_december() -> None:
    """Verify monthly previous-period mapping crosses the year boundary."""

    assert _previous_month_slug("2027_01") == "2026_12"
    assert _previous_month_slug("2027-01") == "2026_12"
    assert _previous_month_slug("2027") is None


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


def _trend_points(report_model: dict[str, object], metric_id: str) -> list[tuple[str, float]]:
    """Return period/value pairs for one report-model historical trend."""

    for section in report_model.get("sections", []):  # type: ignore[union-attr]
        if isinstance(section, dict) and section.get("section_id") == "historical_trends":
            for series in section.get("content", {}).get("trend_series", []):  # type: ignore[union-attr]
                if isinstance(series, dict) and series.get("metric_id") == metric_id:
                    return [
                        (str(point.get("period")), float(point.get("value")))
                        for point in series.get("points", [])
                        if isinstance(point, dict)
                    ]
    return []


def _september_bundle_with_explicit_history() -> ReportInputBundle:
    """Build a September bundle with exact Jun-Sep trend values.

    Inputs: none.
    Outputs: report input bundle carrying deterministic historical context.
    Assumptions: this fixture mirrors the recovery_2026 September values and
    does not depend on Ollama.
    """

    bundle = _bundle()
    finance = json.loads(json.dumps(bundle.finance_summary))
    finance["report_period"] = "2026_09"
    finance["source_workbook"] = "outputs/calculations/finance_summary_2026_09.json"
    finance["finance_summary"].update(
        {
            "total_revenue": 2_123_856.0,
            "total_expenses": 2_096_356.0,
            "net_operating_result": 27_500.0,
        }
    )
    historical_context = {
        "current_period": "2026_09",
        "purpose": "report_model",
        "summary": {"available_retrievals": 2, "unavailable_retrievals": 0},
        "retrievals": [
            {
                "tool_name": "get_metric_history",
                "success": True,
                "metric": "total_revenue",
                "records": [
                    {"period": "2026_06", "metric": "total_revenue", "value": 1_992_060.0},
                    {"period": "2026_07", "metric": "total_revenue", "value": 2_021_376.0},
                    {"period": "2026_08", "metric": "total_revenue", "value": 2_072_448.0},
                ],
            },
            {
                "tool_name": "get_metric_history",
                "success": True,
                "metric": "total_expenses",
                "records": [
                    {"period": "2026_06", "metric": "total_expenses", "value": 2_366_060.0},
                    {"period": "2026_07", "metric": "total_expenses", "value": 2_213_876.0},
                    {"period": "2026_08", "metric": "total_expenses", "value": 2_138_448.0},
                ],
            },
        ],
        "derived_context": {
            "kpi_trends": {
                "total_revenue": {
                    "periods": ["2026_06", "2026_07", "2026_08"],
                    "first_value": 1_992_060.0,
                    "latest_value": 2_072_448.0,
                    "direction": "improving",
                },
                "total_expenses": {
                    "periods": ["2026_06", "2026_07", "2026_08"],
                    "first_value": 2_366_060.0,
                    "latest_value": 2_138_448.0,
                    "direction": "improving",
                },
            }
        },
    }
    strategy = json.loads(json.dumps(bundle.strategic_analysis))
    strategy["historical_context"] = historical_context
    return ReportInputBundle(
        period_slug="2026_09",
        finance_summary=finance,
        kpi_summary=bundle.kpi_summary,
        anomaly_report=bundle.anomaly_report,
        evidence_package=bundle.evidence_package,
        strategic_analysis=strategy,
        source_files=(
            "outputs/calculations/finance_summary_2026_09.json",
            "outputs/calculations/kpi_summary_2026_09.csv",
            "outputs/anomalies/anomaly_report_2026_09.json",
            "outputs/evidence/evidence_package_2026_09.json",
            "outputs/analysis/strategic_analysis_2026_09.json",
        ),
    )


def test_september_report_model_preserves_explicit_intermediate_trend_points() -> None:
    """Verify Jun-Jul-Aug-Sep values reach the report model exactly."""

    report_model = build_report_model(_september_bundle_with_explicit_history()).to_dict()

    assert _trend_points(report_model, "total_revenue") == [
        ("2026_06", 1_992_060.0),
        ("2026_07", 2_021_376.0),
        ("2026_08", 2_072_448.0),
        ("2026_09", 2_123_856.0),
    ]
    assert _trend_points(report_model, "total_expenses") == [
        ("2026_06", 2_366_060.0),
        ("2026_07", 2_213_876.0),
        ("2026_08", 2_138_448.0),
        ("2026_09", 2_096_356.0),
    ]
    assert len(_trend_points(report_model, "total_revenue")) == 4
    assert len(_trend_points(report_model, "total_expenses")) == 4


def test_load_report_inputs_supports_generic_period_with_refreshed_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify generic monthly report inputs can refresh stale historical context."""

    root = tmp_path
    for folder in ("calculations", "anomalies", "evidence", "analysis"):
        (root / "outputs" / folder).mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "calculations" / "finance_summary_2026_09.json").write_text(
        json.dumps(_september_bundle_with_explicit_history().finance_summary),
        encoding="utf-8",
    )
    (root / "outputs" / "calculations" / "kpi_summary_2026_09.csv").write_text(
        "metric,value,unit\n"
        "total_revenue,2123856,USD\n",
        encoding="utf-8",
    )
    (root / "outputs" / "anomalies" / "anomaly_report_2026_09.json").write_text("{}", encoding="utf-8")
    (root / "outputs" / "evidence" / "evidence_package_2026_09.json").write_text("{}", encoding="utf-8")
    (root / "outputs" / "analysis" / "strategic_analysis_2026_09.json").write_text(
        json.dumps({"validation_status": "accepted", "analysis": {}, "historical_context": {"retrievals": []}}),
        encoding="utf-8",
    )

    class FakeHistory:
        """Fake deterministic context builder result."""

        context = {"retrievals": [{"tool_name": "get_metric_history", "metric": "total_revenue", "records": [{"period": "2026_08", "value": 1}]}]}
        telemetry = {"database_queries": 1}

    import finance_agent.reporting.report_engine as report_engine

    monkeypatch.setattr(report_engine, "build_historical_context", lambda **kwargs: FakeHistory())

    bundle = load_report_inputs(root, "2026_09", memory_database_path=tmp_path / "memory.db")

    assert bundle.period_slug == "2026_09"
    assert bundle.strategic_analysis["historical_context"] == FakeHistory.context
    assert bundle.strategic_analysis["historical_context_refresh"]["refreshed_for_report_model"] is True


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


def test_report_model_removes_false_arts_humanities_missing_information() -> None:
    """Verify report mapping checks processed department evidence before publishing missing claims."""

    bundle = _bundle()
    finance_summary = {
        **bundle.finance_summary,
        "report_period": "2026-11",
        "department_summary": [
            {
                "department": "Arts & Humanities",
                "budget_revenue": 305760.0,
                "actual_revenue": 311875.2,
                "budget_expenses": 284356.8,
                "actual_expenses": 287235.21,
                "net_operating_result": 24639.99,
                "expense_variance": 2878.41,
            }
        ],
    }
    strategic_analysis = {
        **bundle.strategic_analysis,
        "analysis": {
            **bundle.strategic_analysis["analysis"],
            "department_analysis": (
                "El departamento de Artes y Humanidades reporta gastos, "
                "pero no se proporcionan ingresos para este departamento."
            ),
            "missing_information": [
                "Ingresos del departamento de Artes y Humanidades para completar el análisis financiero.",
                "Actas de aprobación de proveedores.",
            ],
        },
    }
    model = build_report_model(
        ReportInputBundle(
            period_slug="2026_11",
            finance_summary=finance_summary,
            kpi_summary=bundle.kpi_summary,
            anomaly_report=bundle.anomaly_report,
            evidence_package=bundle.evidence_package,
            strategic_analysis=strategic_analysis,
            source_files=(
                "outputs/calculations/finance_summary_2026_11.json",
                "outputs/calculations/kpi_summary_2026_11.csv",
                "outputs/anomalies/anomaly_report_2026_11.json",
                "outputs/evidence/evidence_package_2026_11.json",
                "outputs/analysis/strategic_analysis_2026_11.json",
            ),
        )
    ).to_dict()
    section_by_id = {section["section_id"]: section for section in model["sections"]}

    missing = section_by_id["missing_information"]["content"]
    department = section_by_id["department_analysis"]["content"]

    assert missing["missing_information"] == ["Actas de aprobación de proveedores."]
    assert missing["missing_information_provenance"][0]["checked_sources"][0]["field"] == "actual_revenue"
    assert "no se proporcionan ingresos" not in department["analysis"]
    assert "Por resultado operativo" in department["analysis"]
    assert department["department_summary"][0]["actual_revenue"] == 311875.2


def test_zero_anomaly_message_names_configured_thresholds() -> None:
    """Verify zero-anomaly reports explain threshold-based decisions."""

    bundle = _bundle()
    zero_anomalies = {
        "report_period": "2026-11",
        "total_anomalies": 0,
        "anomalies_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "anomalies": [],
    }
    model = build_report_model(
        ReportInputBundle(
            period_slug="2026_11",
            finance_summary={**bundle.finance_summary, "report_period": "2026-11"},
            kpi_summary=bundle.kpi_summary,
            anomaly_report=zero_anomalies,
            evidence_package=bundle.evidence_package,
            strategic_analysis=bundle.strategic_analysis,
            source_files=bundle.source_files,
        )
    ).to_dict()

    view = build_presentation_view(model)

    assert "umbrales configurados" in view["anomalies"]["current_period_status"]
    assert view["anomalies"]["top_rows"] == []


def test_university_wide_anomalies_are_not_hidden_without_department_rows() -> None:
    """Verify current-period anomaly display does not depend on department anomalies."""

    bundle = _bundle()
    anomaly_report = {
        "report_period": "2026-11",
        "total_anomalies": 1,
        "anomalies_by_severity": {"critical": 1, "high": 0, "medium": 0, "low": 0},
        "anomalies": [
            {
                "anomaly_id": "ANOM-UNIV-1",
                "title": "Negative or low cash flow",
                "description": "Net cash flow is at or below the configured minimum.",
                "severity": "critical",
                "metric": "net_cash_flow",
                "observed_value": -1.0,
                "threshold_value": 0.0,
                "period": "2026-11",
                "source_file": "finance_summary_2026_11.json",
                "evidence": "Net cash flow is $-1; ending cash is $100.",
            }
        ],
    }

    model = build_report_model(
        ReportInputBundle(
            period_slug="2026_11",
            finance_summary={**bundle.finance_summary, "report_period": "2026-11", "department_summary": []},
            kpi_summary=bundle.kpi_summary,
            anomaly_report=anomaly_report,
            evidence_package=bundle.evidence_package,
            strategic_analysis=bundle.strategic_analysis,
            source_files=bundle.source_files,
        )
    ).to_dict()
    view = build_presentation_view(model)

    assert view["anomalies"]["top_rows"]
    assert view["anomalies"]["top_rows"][0]["title"] == "Flujo de caja bajo o negativo"


def test_generic_period_source_files_are_preserved() -> None:
    """Verify period-slugged artifacts flow into the report model references."""

    bundle = _bundle()
    generic_bundle = ReportInputBundle(
        period_slug="2026_06",
        finance_summary={**bundle.finance_summary, "report_period": "2026-06"},
        kpi_summary=bundle.kpi_summary,
        anomaly_report=bundle.anomaly_report,
        evidence_package=bundle.evidence_package,
        strategic_analysis=bundle.strategic_analysis,
        source_files=(
            "outputs/calculations/finance_summary_2026_06.json",
            "outputs/calculations/kpi_summary_2026_06.csv",
            "outputs/anomalies/anomaly_report_2026_06.json",
            "outputs/evidence/evidence_package_2026_06.json",
            "outputs/analysis/strategic_analysis_2026_06.json",
        ),
    )

    model = build_report_model(generic_bundle).to_dict()

    assert "outputs/anomalies/anomaly_report_2026_06.json" in model["source_references"]
    assert "outputs/analysis/strategic_analysis_2026_06.json" in model["source_references"]


def test_kpi_comparisons_use_previous_processed_summary(tmp_path: Path) -> None:
    """Verify KPI card comparisons are populated from deterministic previous outputs."""

    current_path = tmp_path / "finance_summary_2026_12.json"
    previous_path = tmp_path / "finance_summary_2026_11.json"
    previous_path.write_text(
        json.dumps(
            {
                "report_period": "2026-11",
                "finance_summary": {
                    "total_revenue": 900,
                    "total_expenses": 950,
                    "net_operating_result": -50,
                    "student_payments": {"collection_rate": 0.88},
                    "cash_flow": {"net_cash_flow": 25, "ending_cash": 4800},
                },
            }
        ),
        encoding="utf-8",
    )
    current_path.write_text("{}", encoding="utf-8")
    bundle = _bundle()
    generic_bundle = ReportInputBundle(
        period_slug="2026_12",
        finance_summary={**bundle.finance_summary, "report_period": "2026-12"},
        kpi_summary=bundle.kpi_summary,
        anomaly_report=bundle.anomaly_report,
        evidence_package=bundle.evidence_package,
        strategic_analysis=bundle.strategic_analysis,
        source_files=(
            str(current_path),
            "outputs/calculations/kpi_summary_2026_12.csv",
            "outputs/anomalies/anomaly_report_2026_12.json",
            "outputs/evidence/evidence_package_2026_12.json",
            "outputs/analysis/strategic_analysis_2026_12.json",
        ),
    )

    model = build_report_model(generic_bundle).to_dict()
    section = next(section for section in model["sections"] if section["section_id"] == "financial_health_overview")
    comparisons = section["content"]["kpi_comparisons"]["items"]

    assert comparisons["total_revenue"]["previous_value"] == 900
    assert comparisons["total_revenue"]["absolute_change"] == 100
    assert comparisons["total_revenue"]["budget_value"] == 1100
    assert comparisons["net_cash_flow"]["previous_value"] == 25
    assert comparisons["ending_cash"]["previous_value"] == 4800


def test_kpi_comparison_provenance_separates_current_previous_and_change(tmp_path: Path) -> None:
    """Verify report-model KPI provenance keeps deterministic values distinct."""

    current_path = tmp_path / "finance_summary_2026_06.json"
    previous_path = tmp_path / "finance_summary_2026_05.json"
    previous_path.write_text(
        json.dumps(
            {
                "report_period": "2026-05",
                "finance_summary": {
                    "total_revenue": 2_005_584,
                    "total_expenses": 2_126_584,
                    "net_operating_result": -121_000,
                    "payroll_percentage_of_revenue": 0.47,
                    "student_payments": {"collection_rate": 0.86},
                    "cash_flow": {"net_cash_flow": -220_000, "ending_cash": 2_410_000},
                },
            }
        ),
        encoding="utf-8",
    )
    current_path.write_text("{}", encoding="utf-8")
    bundle = _bundle()
    finance_summary = {
        **bundle.finance_summary,
        "report_period": "2026-06",
        "finance_summary": {
            **bundle.finance_summary["finance_summary"],
            "total_revenue": 1_992_060,
            "total_expenses": 2_366_060,
            "net_operating_result": -374_000,
            "payroll_percentage_of_revenue": 0.53,
            "student_payments": {"collection_rate": 0.84},
            "cash_flow": {"net_cash_flow": -680_000, "ending_cash": 1_730_000},
        },
    }
    model = build_report_model(
        ReportInputBundle(
            period_slug="2026_06",
            finance_summary=finance_summary,
            kpi_summary=bundle.kpi_summary,
            anomaly_report=bundle.anomaly_report,
            evidence_package=bundle.evidence_package,
            strategic_analysis=bundle.strategic_analysis,
            source_files=(
                str(current_path),
                "outputs/calculations/kpi_summary_2026_06.csv",
                "outputs/anomalies/anomaly_report_2026_06.json",
                "outputs/evidence/evidence_package_2026_06.json",
                "outputs/analysis/strategic_analysis_2026_06.json",
            ),
        )
    ).to_dict()
    section = next(section for section in model["sections"] if section["section_id"] == "financial_health_overview")
    comparisons = section["content"]["kpi_comparisons"]["items"]

    revenue = comparisons["total_revenue"]
    expenses = comparisons["total_expenses"]
    payroll_ratio = comparisons["payroll_percentage_of_revenue"]

    assert revenue["current_value"] == 1_992_060
    assert revenue["previous_value"] == 2_005_584
    assert revenue["absolute_change"] == -13_524
    assert revenue["provenance"]["computed_change"] == -13_524
    assert revenue["provenance"]["calculation_method"].startswith("current_minus_previous")
    assert expenses["absolute_change"] == 239_476
    assert payroll_ratio["percentage_point_change"] == pytest.approx(0.06)
    assert payroll_ratio["percent_change"] is None
    assert payroll_ratio["provenance"]["unit"] == "ratio"


def test_rejected_modular_synthesis_is_not_used_as_visible_report_text() -> None:
    """Verify rejected strategy cannot leak unsupported Ollama prose into reports."""

    bundle = _bundle()
    rejected_strategy = {
        "validation_status": "rejected",
        "validation_errors": ["unsupported quantitative claim"],
        "reasoning_state": {
            "reasoning_outputs": {
                "strategic_synthesis": {
                    "executive_summary": "Los gastos aumentaron 18.8% frente al periodo anterior.",
                    "recommendations": [{"action": "Unsupported"}],
                }
            }
        },
        "analysis": {
            "executive_summary": "Este texto tampoco debe aparecer.",
        },
    }
    model = build_report_model(
        ReportInputBundle(
            period_slug=bundle.period_slug,
            finance_summary=bundle.finance_summary,
            kpi_summary=bundle.kpi_summary,
            anomaly_report=bundle.anomaly_report,
            evidence_package=bundle.evidence_package,
            strategic_analysis=rejected_strategy,
            source_files=bundle.source_files,
        )
    ).to_dict()
    executive = next(section for section in model["sections"] if section["section_id"] == "executive_summary")

    assert "18.8%" not in executive["content"]["summary"]
    assert "Este texto tampoco" not in executive["content"]["summary"]
    assert "Reporte financiero determinístico" in executive["content"]["summary"]


def test_rejected_strategy_preserves_deterministic_analysis_and_anomaly_rows() -> None:
    """Verify rejected strategy still leaves useful deterministic report content.

    Inputs: a report bundle with rejected strategic analysis.
    Outputs: report model with fallback analysis, anomaly detail rows, and
    attention items.
    Assumptions: invalid Ollama prose must not be shown, but Python-derived
    facts remain safe to render.
    """

    bundle = _bundle()
    rejected_strategy = {
        "validation_status": "rejected",
        "validation_errors": ["risks[2] contains unsupported number: 15.0%"],
        "analysis": {
            "executive_summary": "Unsupported text must not be shown.",
            "recommendations": [{"action": "Unsupported"}],
        },
    }

    model = build_report_model(
        ReportInputBundle(
            period_slug=bundle.period_slug,
            finance_summary=bundle.finance_summary,
            kpi_summary=bundle.kpi_summary,
            anomaly_report=bundle.anomaly_report,
            evidence_package=bundle.evidence_package,
            strategic_analysis=rejected_strategy,
            source_files=bundle.source_files,
        )
    ).to_dict()
    sections = {section["section_id"]: section for section in model["sections"]}

    health = sections["financial_health_overview"]["content"]
    anomaly = sections["anomaly_summary"]["content"]
    recommendations = sections["strategic_recommendations"]["content"]

    assert "Ingresos:" in health["analysis"]
    assert anomaly["anomalies"][0]["anomaly_id"] == "ANOM-1"
    assert anomaly["top_anomalies"][0]["title"] == "Operating deficit"
    assert anomaly["analysis"].startswith("El detector determinístico registró 1 anomalías")
    assert recommendations["recommendations"] == []
    assert recommendations["deterministic_attention_items"][0]["title"] == "Operating deficit"


def test_unsupported_previous_period_percentage_claim_uses_deterministic_summary(tmp_path: Path) -> None:
    """Verify misleading model prose cannot relabel budget variance as prior change."""

    current_path = tmp_path / "finance_summary_2026_06.json"
    previous_path = tmp_path / "finance_summary_2026_05.json"
    previous_path.write_text(
        json.dumps(
            {
                "report_period": "2026-05",
                "finance_summary": {
                    "total_revenue": 2_005_584,
                    "total_expenses": 2_126_584,
                    "net_operating_result": -121_000,
                    "student_payments": {"collection_rate": 0.86},
                    "cash_flow": {"net_cash_flow": -220_000, "ending_cash": 2_410_000},
                },
            }
        ),
        encoding="utf-8",
    )
    current_path.write_text("{}", encoding="utf-8")
    bundle = _bundle()
    strategic_analysis = {
        **bundle.strategic_analysis,
        "analysis": {
            **bundle.strategic_analysis["analysis"],
            "executive_summary": (
                "Los gastos aumentaron un 18.8% en comparación con el periodo anterior, "
                "mientras que los ingresos disminuyeron un 7.0%."
            ),
        },
    }
    finance_summary = {
        **bundle.finance_summary,
        "report_period": "2026-06",
        "finance_summary": {
            **bundle.finance_summary["finance_summary"],
            "total_revenue": 1_992_060,
            "total_expenses": 2_366_060,
            "net_operating_result": -374_000,
            "budget_vs_actual": {
                "revenue_budget": 2_142_000,
                "revenue_variance": -149_940,
                "revenue_variance_pct": -0.07,
                "expense_budget": 1_992_060,
                "expense_variance": 374_000,
                "expense_variance_pct": 0.188,
            },
            "student_payments": {"collection_rate": 0.84},
            "cash_flow": {"net_cash_flow": -680_000, "ending_cash": 1_730_000},
        },
    }

    model = build_report_model(
        ReportInputBundle(
            period_slug="2026_06",
            finance_summary=finance_summary,
            kpi_summary=bundle.kpi_summary,
            anomaly_report=bundle.anomaly_report,
            evidence_package=bundle.evidence_package,
            strategic_analysis=strategic_analysis,
            source_files=(
                str(current_path),
                "outputs/calculations/kpi_summary_2026_06.csv",
                "outputs/anomalies/anomaly_report_2026_06.json",
                "outputs/evidence/evidence_package_2026_06.json",
                "outputs/analysis/strategic_analysis_2026_06.json",
            ),
        )
    ).to_dict()
    executive = next(section for section in model["sections"] if section["section_id"] == "executive_summary")

    assert "18.8%" not in executive["content"]["summary"]
    assert "7.0%" not in executive["content"]["summary"]
    assert any("quantitative comparison" in warning for warning in executive["warnings"])


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
