"""Tests for Spanish HTML and PDF report renderers."""

from __future__ import annotations

from pathlib import Path
import inspect

from finance_agent.reporting.renderers import (
    load_report_model,
    render_report_html,
    render_report_pdf,
    save_report_html,
)
from finance_agent.reporting.presentation import REPORT_SECTION_TEMPLATES, build_presentation_view
import finance_agent.reporting.presentation as presentation
from finance_agent.reporting.report_quality import (
    require_report_quality,
    validate_report_artifacts,
    validate_report_model_quality,
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
                "summary": "La operaciÃ³n requiere atenciÃ³n ejecutiva.",
                "key_findings": ["El resultado operativo es negativo."],
                "root_causes": ["El gasto creciÃ³ mÃ¡s rÃ¡pido que los ingresos."],
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
                        "evidence_summary": "Se recuperÃ³ evidencia departamental.",
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
                "root_causes": ["El gasto creciÃ³ mÃ¡s rÃ¡pido que los ingresos."],
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

    assert "La operaciÃ³n requiere atenciÃ³n ejecutiva." in html
    assert "El resultado operativo es negativo." in html
    assert "El gasto creciÃ³ mÃ¡s rÃ¡pido que los ingresos." in html
    assert "Estabilizar el flujo de caja." in html
    assert "Revisar aprobaciones de gasto." in html
    assert "No hay recomendaciones estratÃ©gicas generadas." not in html


def test_section_templates_define_evidence_contracts() -> None:
    """Verify section templates define objectives and required evidence."""

    expected = {
        "executive_summary",
        "financial_health_overview",
        "kpi_overview",
        "historical_summary",
        "historical_trends",
        "revenue_expense_analysis",
        "department_analysis",
        "anomaly_summary",
        "recommendation_follow_up",
        "longitudinal_risk_assessment",
        "strategic_recommendations",
        "missing_information",
        "appendix",
    }

    assert expected.issubset(REPORT_SECTION_TEMPLATES)
    for template in REPORT_SECTION_TEMPLATES.values():
        assert template.objective
        assert template.visibility_rule
        assert template.narrative_fields or template.section_id == "appendix"


def test_presentation_layer_has_no_analytical_sentence_generator() -> None:
    """Verify presentation.py does not hardcode analytical conclusions."""

    source = inspect.getsource(presentation)

    forbidden_fragments = (
        "Strategic analysis was unavailable",
        "No hay recomendaciones estratégicas generadas",
        "se deterioró de",
        "mejoró de",
        "Las tendencias históricas",
        "hardcoded_values",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source


def test_ollama_section_narrative_populates_html_sections() -> None:
    """Verify final HTML displays section narrative from the report model."""

    model = _sample_report_model()
    section_content = {
        section["section_id"]: section["content"]
        for section in model["sections"]  # type: ignore[index]
        if isinstance(section, dict)
    }
    section_content["financial_health_overview"]["analysis"] = "La salud financiera se deteriora por un resultado operativo negativo de -200."
    section_content["kpi_overview"]["analysis"] = "La tasa de cobranza de 90.0% exige seguimiento ejecutivo."
    section_content["department_analysis"]["analysis"] = "Engineering concentra un déficit departamental de -150."

    html = render_report_html(model)

    assert "La salud financiera se deteriora por un resultado operativo negativo de -200." in html
    assert "La tasa de cobranza de 90.0% exige seguimiento ejecutivo." in html
    assert "Engineering concentra un déficit departamental de -150." in html


def test_report_model_quality_accepts_strategy_backed_model() -> None:
    """Verify quality validation accepts current strategy-backed report models."""

    result = validate_report_model_quality(_sample_report_model())

    assert result.is_valid is True
    assert result.recommendation_count == 1


def test_report_model_quality_rejects_missing_strategy() -> None:
    """Verify missing strategy and recommendations are blocking errors."""

    model = _sample_report_model()
    sections = model["sections"]  # type: ignore[assignment]
    for section in sections:  # type: ignore[union-attr]
        if section["section_id"] == "executive_summary":
            section["content"]["analysis_status"] = "unavailable"
            section["content"]["summary"] = "Strategic analysis was unavailable; use processed metrics."
        if section["section_id"] == "strategic_recommendations":
            section["content"]["recommendations"] = []

    result = validate_report_model_quality(model)

    assert result.is_valid is False
    assert any("not accepted" in error for error in result.errors)
    assert any("placeholder" in error for error in result.errors)


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

    assert "SecciÃ³n faltante en el modelo: kpi_overview" not in html
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


def test_report_artifact_quality_detects_html_placeholder(tmp_path: Path) -> None:
    """Verify rendered artifact validation catches missing-strategy placeholders."""

    model_path = tmp_path / "report_model.json"
    model_path.write_text(__import__("json").dumps(_sample_report_model()), encoding="utf-8")
    html_path = tmp_path / "report.html"
    html_path.write_text("Strategic analysis was unavailable", encoding="utf-8")

    result = validate_report_artifacts(model_path, html_path=html_path)

    assert result.is_valid is False
    assert any("HTML contains" in error for error in result.errors)


def test_executive_html_hides_internal_identifiers_and_paths() -> None:
    """Verify executive HTML does not expose raw implementation details."""

    model = _sample_report_model()
    html = render_report_html(model)

    assert "total_revenue" not in html
    assert "collection_rate" not in html
    assert "get_department_history" not in html
    assert "C:\\" not in html
    assert "########" not in html
    assert "<svg" in html


def test_presentation_view_contains_recommendation_cards() -> None:
    """Verify the shared presentation view prepares recommendation cards."""

    view = build_presentation_view(_sample_report_model())

    cards = view["recommendations"]["cards"]
    assert cards
    assert cards[0]["action"] == "Revisar aprobaciones de gasto."


def test_presentation_layer_has_no_narrative_translation_dictionary() -> None:
    """Verify presentation does not contain report-specific translation mappings."""

    source = inspect.getsource(presentation)

    assert "The financial performance shows mixed results" not in source
    assert "Payroll variance shows" not in source
    assert "+4%" not in source
    assert "replacements =" not in source


def test_rendered_quality_rejects_internal_tool_names(tmp_path: Path) -> None:
    """Verify artifact validation blocks tool-name leaks in executive HTML."""

    import json

    model_path = tmp_path / "report_model.json"
    model_path.write_text(json.dumps(_sample_report_model()), encoding="utf-8")
    html_path = tmp_path / "report.html"
    html_path.write_text("Resumen get_metric_history total_revenue C:\\temp\\file.json", encoding="utf-8")

    result = validate_report_artifacts(model_path, html_path=html_path)

    assert result.is_valid is False
    assert any("internal retrieval tool" in error for error in result.errors)
    assert any("absolute Windows path" in error for error in result.errors)
    assert any("canonical KPI" in error for error in result.errors)


def test_require_report_quality_detects_stale_artifact(tmp_path: Path) -> None:
    """Verify quality validation rejects artifacts older than the report model."""

    import os
    import time
    import json

    html_path = tmp_path / "report.html"
    html_path.write_text(render_report_html(_sample_report_model()), encoding="utf-8")
    old_time = time.time() - 20
    os.utime(html_path, (old_time, old_time))
    model_path = tmp_path / "report_model.json"
    model_path.write_text(json.dumps(_sample_report_model()), encoding="utf-8")

    try:
        require_report_quality(model_path, html_path=html_path)
    except ValueError as exc:
        assert "older than report model" in str(exc)
    else:
        raise AssertionError("Expected ValueError for stale artifact")

