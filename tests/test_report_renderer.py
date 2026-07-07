"""Tests for Spanish HTML and PDF report renderers."""

from __future__ import annotations

from pathlib import Path

from finance_agent.reporting.renderers import (
    load_report_model,
    render_report_html,
    render_report_pdf,
    save_report_html,
)
from finance_agent.reporting.report_models import REQUIRED_SECTION_IDS


def _sample_report_model() -> dict[str, object]:
    """Build a compact report model fixture.

    Inputs: none.
    Outputs: JSON-compatible report model dictionary.
    Assumptions: values represent already-processed pipeline outputs.
    """

    sections = [
        {
            "section_id": "cover",
            "title": "Cover",
            "content": {"title": "Reporte financiero", "report_period": "June 2026"},
            "source_references": ["outputs/report/report_model_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "executive_summary",
            "title": "Executive summary",
            "content": {
                "summary": "La operación requiere atención ejecutiva.",
                "key_findings": ["El resultado operativo es negativo."],
                "root_causes": ["El gasto creció más rápido que los ingresos."],
                "confidence": 0.8,
                "analysis_status": "accepted",
            },
            "source_references": ["outputs/analysis/strategic_analysis_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "financial_health_overview",
            "title": "Financial health",
            "content": {
                "total_revenue": 1000,
                "total_expenses": 1200,
                "net_operating_result": -200,
                "net_cash_flow": -300,
                "ending_cash": 5000,
                "collection_rate": 0.9,
            },
            "source_references": ["outputs/calculations/finance_summary_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "kpi_overview",
            "title": "KPI overview",
            "content": {
                "kpis": [
                    {
                        "metric": "collection_rate",
                        "value": 0.9,
                        "unit": "ratio",
                        "availability": "available",
                        "source": "student_payments",
                    }
                ]
            },
            "source_references": ["outputs/calculations/kpi_summary_june_2026.csv"],
            "warnings": [],
        },
        {
            "section_id": "revenue_analysis",
            "title": "Revenue",
            "content": {"total_revenue": 1000, "revenue_budget": 1100, "revenue_variance": -100},
            "source_references": ["outputs/calculations/finance_summary_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "expense_analysis",
            "title": "Expense",
            "content": {"total_expenses": 1200, "expense_budget": 1000, "expense_variance": 200},
            "source_references": ["outputs/calculations/finance_summary_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "department_analysis",
            "title": "Departments",
            "content": {
                "department_summary": [
                    {
                        "department": "Engineering",
                        "actual_revenue": 300,
                        "actual_expenses": 450,
                        "net_operating_result": -150,
                        "expense_variance_pct": 0.2,
                    }
                ]
            },
            "source_references": ["outputs/evidence/evidence_package_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "anomaly_summary",
            "title": "Anomalies",
            "content": {
                "anomalies_by_severity": {"critical": 1},
                "top_anomalies": [
                    {
                        "anomaly_id": "ANOM-1",
                        "title": "Gasto elevado",
                        "severity": "critical",
                        "evidence": "Variance above threshold.",
                    }
                ],
            },
            "source_references": ["outputs/anomalies/anomaly_report_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "investigation_evidence",
            "title": "Evidence",
            "content": {
                "evidence_items": [
                    {
                        "task_id": "TASK-1",
                        "priority": "critical",
                        "retrieval_name": "department_history",
                        "record_count": 3,
                        "evidence_summary": "Se recuperó evidencia departamental.",
                    }
                ]
            },
            "source_references": ["outputs/evidence/evidence_package_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "strategic_recommendations",
            "title": "Recommendations",
            "content": {
                "root_causes": ["El gasto creció más rápido que los ingresos."],
                "strategic_priorities": ["Estabilizar el flujo de caja."],
                "reasoning_summary": "La evidencia soporta acciones de control de gasto.",
                "recommendations": [
                    {
                        "priority": "high",
                        "action": "Revisar aprobaciones de gasto.",
                        "rationale": "El gasto supera presupuesto.",
                        "expected_impact": "Mejor control operacional.",
                    }
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "missing_information",
            "title": "Missing",
            "content": {"missing_information": []},
            "source_references": ["outputs/analysis/strategic_analysis_june_2026.json"],
            "warnings": [],
        },
        {
            "section_id": "appendix",
            "title": "Appendix",
            "content": {"source_files": ["finance_summary_june_2026.json"]},
            "source_references": ["outputs/report/report_model_june_2026.json"],
            "warnings": [],
        },
    ]
    return {
        "report_id": "REPORT-MODEL-JUNE-2026",
        "period_slug": "june_2026",
        "report_period": "June 2026",
        "renderer_contract_version": "1.0",
        "section_count": len(sections),
        "sections": sections,
        "source_references": ["outputs/report/report_model_june_2026.json"],
    }


def test_html_generation_contains_required_sections_and_spanish_labels() -> None:
    """Verify HTML rendering includes all report sections with Spanish labels."""

    html = render_report_html(_sample_report_model())

    for section_id in REQUIRED_SECTION_IDS:
        assert f"id='{section_id}'" in html or f'id="{section_id}"' in html
    assert "Resumen ejecutivo" in html
    assert "Salud financiera" in html
    assert "Análisis por departamento" in html
    assert "Recomendaciones estratégicas" in html
    assert "Fuentes:" in html


def test_html_generation_renders_strategic_analysis_fields() -> None:
    """Verify accepted strategy fields appear in the HTML report."""

    html = render_report_html(_sample_report_model())

    assert "La operación requiere atención ejecutiva." in html
    assert "El resultado operativo es negativo." in html
    assert "El gasto creció más rápido que los ingresos." in html
    assert "Estabilizar el flujo de caja." in html
    assert "Revisar aprobaciones de gasto." in html
    assert "No hay recomendaciones estratégicas generadas." not in html


def test_save_html_writes_document(tmp_path: Path) -> None:
    """Verify HTML output is written as a complete document."""

    output_path = save_report_html(_sample_report_model(), tmp_path / "report.html")

    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<html lang='es'>" in text


def test_pdf_generation_writes_pdf_file(tmp_path: Path) -> None:
    """Verify PDF rendering creates a non-empty PDF artifact."""

    output_path = render_report_pdf(_sample_report_model(), tmp_path / "report.pdf")

    data = output_path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 1000


def test_missing_and_empty_sections_render_gracefully(tmp_path: Path) -> None:
    """Verify missing sections and empty tables produce readable placeholders."""

    model = _sample_report_model()
    model["sections"] = [section for section in model["sections"] if section["section_id"] != "kpi_overview"]  # type: ignore[index]
    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "missing.pdf")

    assert "Sección faltante en el modelo: kpi_overview" not in html
    assert "Sin datos disponibles" in html
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_strategy_validation_rejects_unavailable_analysis() -> None:
    """Verify final rendering guard rejects report models without accepted strategy."""

    from finance_agent.reporting.renderers import report_strategy_warnings, validate_strategy_available

    model = _sample_report_model()
    sections = model["sections"]  # type: ignore[assignment]
    for section in sections:  # type: ignore[union-attr]
        if section["section_id"] == "executive_summary":
            section["content"]["analysis_status"] = "unavailable"
        if section["section_id"] == "strategic_recommendations":
            section["content"]["recommendations"] = []

    warnings = report_strategy_warnings(model)
    assert warnings
    try:
        validate_strategy_available(model)
    except ValueError as exc:
        assert "Strategic analysis is unavailable" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unavailable strategy")


def test_load_report_model_rejects_non_object_json(tmp_path: Path) -> None:
    """Verify report model loading rejects JSON roots that are not objects."""

    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    try:
        load_report_model(path)
    except ValueError as exc:
        assert "root must be an object" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-object report model")
