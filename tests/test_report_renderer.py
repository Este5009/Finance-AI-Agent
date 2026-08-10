"""Tests for Spanish HTML and PDF report renderers."""

from __future__ import annotations

from pathlib import Path
import inspect

import pytest

from finance_agent.reporting.renderers import (
    load_report_model,
    render_report_html,
    render_report_pdf,
    save_report_html,
)
from finance_agent.reporting.renderers import pdf_renderer as pdf_renderer_module
from finance_agent.reporting.presentation import (
    REPORT_SECTION_TEMPLATES,
    adaptive_axis_domain,
    build_anomaly_summary,
    build_presentation_view,
    display_anomaly_text,
    display_metric_name,
    find_spanish_executive_localization_leaks,
)
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
                "kpi_comparisons": {
                    "items": {
                        "total_revenue": {
                            "unit": "USD",
                            "current_value": 1000,
                            "previous_value": 900,
                            "absolute_change": 100,
                            "percent_change": 0.1111111111,
                            "budget_value": 1100,
                            "budget_change": -100,
                            "budget_change_pct": -0.0909090909,
                        },
                        "total_expenses": {
                            "unit": "USD",
                            "current_value": 1200,
                            "previous_value": 1000,
                            "absolute_change": 200,
                            "percent_change": 0.2,
                            "budget_value": 1000,
                            "budget_change": 200,
                            "budget_change_pct": 0.2,
                        },
                    },
                    "unavailable": [],
                },
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
            "section_id": "goal_budget_performance",
            "title": "Goal and budget performance",
            "content": {
                "overall_score": 84.2,
                "valid_goal_count": 3,
                "met_goal_count": 1,
                "risk_goal_count": 1,
                "critical_goal_count": 1,
                "weighting_method": "Equal weighting across comparable deterministic goals.",
                "deterministic_conclusion": (
                    "1 de 3 metas determinísticas se cumplieron; 2 requieren seguimiento ejecutivo."
                ),
                "status_distribution": {
                    "Cumplida": 1,
                    "Cerca de cumplir": 0,
                    "En riesgo": 1,
                    "Crítica": 1,
                    "Sin datos": 0,
                },
                "items": [
                    {
                        "goal_id": "GOAL-001",
                        "metric_id": "total_revenue",
                        "display_label": "Ingresos totales",
                        "actual_value": 1000,
                        "target_value": 1100,
                        "absolute_gap": -100,
                        "relative_gap": -0.0909090909,
                        "percentage_point_gap": None,
                        "unit": "USD",
                        "achievement_score": 90.9,
                        "status": "Cerca de cumplir",
                        "direction": "higher_is_better",
                        "budget_classification": "Desfavorable",
                        "source_provenance_actual": {
                            "artifact": "outputs/calculations/finance_summary_june_2026.json",
                            "path": "total_revenue",
                            "source_metric_id": "total_revenue",
                        },
                        "source_provenance_target": {
                            "artifact": "outputs/calculations/finance_summary_june_2026.json",
                            "path": "budget_vs_actual.revenue_budget",
                            "source_metric_id": "total_revenue",
                        },
                    },
                    {
                        "goal_id": "GOAL-002",
                        "metric_id": "payroll_percentage_of_revenue",
                        "display_label": "Nómina / ingresos",
                        "actual_value": 0.46,
                        "target_value": 0.42,
                        "absolute_gap": 0.04,
                        "relative_gap": 0.095238095,
                        "percentage_point_gap": 0.04,
                        "unit": "ratio",
                        "achievement_score": 91.3,
                        "status": "Cerca de cumplir",
                        "direction": "lower_is_better",
                        "budget_classification": "Desfavorable",
                        "source_provenance_actual": {
                            "artifact": "outputs/calculations/finance_summary_june_2026.json",
                            "path": "payroll_percentage_of_revenue",
                            "source_metric_id": "payroll_percentage_of_revenue",
                        },
                        "source_provenance_target": {
                            "artifact": "policy_threshold",
                            "path": "goal_thresholds.payroll_percentage_of_revenue",
                            "source_metric_id": "payroll_percentage_of_revenue",
                        },
                    },
                ],
            },
            "source_references": ["outputs/calculations/finance_summary_june_2026.json"],
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
                        "finding_type": "system_review_rule",
                        "reference_origin": "system-derived/default",
                        "reference_type": "system_threshold",
                        "reference_source": "AnomalyThresholds configuration",
                        "is_institutional_reference": False,
                        "reason_for_flagging": "El valor observado supera la referencia analítica configurada por el sistema.",
                        "reference_notice_es": (
                            "Referencia analítica del sistema. No corresponde a una meta, límite o política institucional."
                        ),
                        "observed_value": 1_200,
                        "threshold_value": 1_000,
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


def test_html_anomaly_section_shows_finding_provenance() -> None:
    """Verify executive anomaly cards distinguish finding and reference origin."""

    html = render_report_html(_sample_report_model())

    assert "Hallazgos y anomalías" in html
    assert "Revisión sugerida por el sistema" in html
    assert "Referencia analítica del sistema" in html
    assert "No corresponde a una meta, límite o política institucional" in html


def test_budget_variance_finding_presentation_uses_exact_comparison_fields() -> None:
    """Verify budget findings lead with budget, actual, and exact variance facts."""

    model = _sample_report_model()
    for section in model["sections"]:
        if section["section_id"] == "anomaly_summary":
            section["content"] = {
                "report_period": "Sep 2026",
                "anomalies_by_severity": {"medium": 1},
                "top_anomalies": [
                    {
                        "title": "Business outside budget target range",
                        "description": "Department expense variance is outside the system analytical +/- range.",
                        "metric": "department_expense_variance_pct",
                        "department": "Business",
                        "observed_value": 4.0119,
                        "threshold_value": 8.0,
                        "severity": "medium",
                        "period": "2026_09",
                        "finding_type": "system_review_rule",
                        "reference_origin": "system-derived/default",
                        "reference_notice_es": "Referencia analítica del sistema. No corresponde a una meta, límite o política institucional.",
                        "reason_for_flagging": "Business variance of 4.01% is outside the +/-8.00% system review range.",
                        "evidence": "Business expense variance is 4.01% versus the +/-8.00% system review reference.",
                        "comparison_details": {
                            "budget_expense": 402_631.41,
                            "actual_expense": 418_784.70,
                            "expense_variance": 16_153.29,
                            "expense_variance_pct": 0.0401,
                        },
                    }
                ],
            }
            break

    summary = build_anomaly_summary(model)
    row = summary["top_rows"][0]

    assert row["title"] == "Gastos de Negocios por encima del presupuesto"
    assert row["budget_expense"] == "$402,631"
    assert row["actual_expense"] == "$418,785"
    assert row["expense_variance"] == "$16,153"
    assert row["expense_variance_pct"] == "4.0%"
    assert row["reference_value"] == "8.0%"


def test_budget_variance_text_and_metric_labels_are_spanish() -> None:
    """Verify legacy detector phrases and canonical metric IDs display in Spanish."""

    rendered = [
        display_anomaly_text("Engineering expense variance is 10.75% versus a +/-8.00% target."),
        display_anomaly_text("Business expense variance is 10.75% versus the +/-8.00% system review reference."),
        display_anomaly_text("Business variance of 10.75% is outside the +/-8.00% system review range."),
        display_anomaly_text("Arts & Humanities spent $328,089 against $275,851 budget (18.94% variance)."),
        display_anomaly_text("Technology actual $112,295 versus budget $103,124; variance 8.89%."),
        display_anomaly_text("Collection rate is 92.00% from $1,406,842 paid against $1,529,176 due."),
        display_anomaly_text("Maximum payment is $74,500 versus a $50,000 threshold."),
    ]

    assert display_metric_name("department_expense_variance_pct") == "Variación porcentual del gasto departamental"
    assert display_metric_name("category_expense_variance_pct") == "Variación porcentual del gasto por categoría"
    assert display_metric_name("maximum_vendor_payment") == "Pago máximo a proveedor"
    for text in rendered:
        assert find_spanish_executive_localization_leaks(text) == []
    assert "Artes y Humanidades gastó $328,089 frente a un presupuesto de $275,851" in rendered[3]
    assert "Tecnología: gasto real $112,295 frente a presupuesto $103,124" in rendered[4]
    assert "La tasa de cobranza es 92.00%" in rendered[5]
    assert "referencia de revisión de $50,000" in rendered[6]


def test_known_deterministic_anomaly_strings_do_not_leak_to_html_or_pdf(tmp_path: Path) -> None:
    """Verify known Python detector titles/descriptions are localized in final artifacts."""

    model = _sample_report_model()
    known = [
        ("Negative or low cash flow", "Net cash flow is $-1; ending cash is $100."),
        ("Overdue student payments above limit", "10 of 100 invoices are overdue."),
        ("Negative or zero operating result", "Net operating result is $-5 on $100 of revenue."),
        ("Vendor payment exceeds review threshold", "Maximum payment is $60,000 versus a $50,000 threshold."),
        ("Tuition collection below target", "Collection rate is 90.00% from $900 paid against $1,000 due."),
        ("Technology category outside budget target", "Technology actual $112,295 versus budget $103,124; variance 8.89%."),
        ("Arts & Humanities outside budget target range", "Arts & Humanities spent $328,089 against $275,851 budget (18.94% variance)."),
    ]
    for section in model["sections"]:
        if section["section_id"] == "anomaly_summary":
            section["content"] = {
                "anomalies_by_severity": {"medium": len(known)},
                "top_anomalies": [
                    {
                        "anomaly_id": f"ANOM-{index}",
                        "title": title,
                        "description": description,
                        "evidence": description,
                        "metric": "net_cash_flow" if "cash" in title.lower() else "collection_rate",
                        "observed_value": 1,
                        "threshold_value": 0,
                        "severity": "medium",
                        "finding_type": "system_review_rule",
                        "reference_origin": "system-derived/default",
                    }
                    for index, (title, description) in enumerate(known)
                ],
            }
            break

    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "spanish_anomalies.pdf")

    from pypdf import PdfReader

    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    combined = html + "\n" + pdf_text
    for english, _description in known:
        assert english not in combined
    for phrase in (
        "Net cash flow is",
        "Overdue student payments above limit",
        "Negative or zero operating result",
        "Vendor payment exceeds review threshold",
        "Tuition collection below target",
    ):
        assert phrase not in combined
    assert "Flujo de caja negativo o insuficiente" in combined
    assert "Pagos estudiantiles vencidos por encima de la referencia" in combined
    assert "Pago a proveedor supera la referencia de revisión" in combined
    assert find_spanish_executive_localization_leaks(combined) == []


def test_july_deterministic_attention_items_are_spanish_in_html_pdf(tmp_path: Path) -> None:
    """Verify July fallback findings do not render raw English detector fields."""

    path = Path("outputs/report/report_model_2026_07.json")
    if not path.is_file():
        pytest.skip("July report model artifact is not available.")

    model = load_report_model(path)
    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "july_attention_items.pdf")

    from pypdf import PdfReader

    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    combined = html + "\n" + pdf_text
    forbidden = (
        "Negative or",
        "Overdue student",
        "Net cash flow is",
        "Vendor payment",
        "Maximum payment is",
        "Tuition collection",
        "Collection rate is",
        "operating result is",
    )
    expected = (
        "Flujo de caja negativo o insuficiente",
        "El flujo neto de caja es de $-350,000",
        "Pagos estudiantiles vencidos por encima de la referencia",
        "24 de 24 facturas están vencidas",
        "Resultado operativo negativo o nulo",
        "Pago a proveedor supera la referencia de revisión",
        "El pago máximo es de $74,500",
        "Tasa de cobranza por debajo de la referencia",
        "La tasa de cobranza es 85.00%",
    )
    for phrase in forbidden:
        assert phrase not in combined
    for phrase in expected:
        assert phrase in combined
    assert find_spanish_executive_localization_leaks(combined) == []


def test_august_and_september_html_pdf_have_no_deterministic_english_leaks(tmp_path: Path) -> None:
    """Verify synthetic report artifacts render Spanish executive strings end to end."""

    from pypdf import PdfReader

    for period in ("2026_08", "2026_09"):
        path = Path(f"outputs/report/report_model_{period}.json")
        if not path.is_file():
            pytest.skip(f"{period} report model artifact is not available.")
        model = load_report_model(path)
        html = render_report_html(model)
        pdf_path = render_report_pdf(model, tmp_path / f"{period}.pdf")
        pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)

        assert find_spanish_executive_localization_leaks(html) == []
        assert find_spanish_executive_localization_leaks(pdf_text) == []


def test_september_goal_chart_groups_are_grouped_and_preserve_exact_values() -> None:
    """Verify September goal charts compare values side-by-side without sums."""

    path = Path("outputs/report/report_model_2026_09.json")
    if not path.is_file():
        pytest.skip("September report model artifact is not available.")
    view = build_presentation_view(load_report_model(path))
    groups = view["goal_budget"]["chart_groups"]
    currency = next(group for group in groups if group["unit"] == "USD")
    ratio = next(group for group in groups if group["unit"] == "ratio")
    values_by_key = {
        (row["metric"], row["series"]): row["value"]
        for group in groups
        for row in group["rows"]
    }

    assert currency["encoding"] == "grouped_actual_reference"
    assert ratio["encoding"] == "grouped_actual_reference"
    assert {row["series"] for row in currency["rows"]} >= {"Real", "Meta", "Presupuesto"}
    assert {row["series"] for row in ratio["rows"]} >= {"Real", "Límite máximo", "Meta mínima"}
    assert values_by_key[("Ingresos totales", "Real")] == 2_123_856.0
    assert values_by_key[("Ingresos totales", "Presupuesto")] == 2_167_200.0
    assert values_by_key[("Gastos totales", "Real")] == 2_096_356.0
    assert values_by_key[("Gastos totales", "Presupuesto")] == 2_015_496.0
    assert values_by_key[("Resultado operativo", "Real")] == 27_500.0
    assert values_by_key[("Resultado operativo", "Presupuesto")] == 151_704.0
    assert values_by_key[("Flujo neto de caja", "Real")] == 50_000.0
    assert values_by_key[("Flujo neto de caja", "Meta")] == 0.0
    assert values_by_key[("Nómina / ingresos", "Real")] == 0.46
    assert values_by_key[("Nómina / ingresos", "Límite máximo")] == 0.42
    assert values_by_key[("Tasa de cobranza", "Real")] == pytest.approx(0.9199999665)
    assert values_by_key[("Tasa de cobranza", "Meta mínima")] == 0.94
    assert 2_123_856.0 + 2_167_200.0 not in [row["value"] for row in currency["rows"]]
    assert set(group["unit"] for group in groups) >= {"USD", "ratio"}


def test_html_goal_charts_use_grouped_semantics_and_reference_labels() -> None:
    """Verify HTML goal charts expose grouped comparison labels, not stacked rows."""

    path = Path("outputs/report/report_model_2026_09.json")
    if not path.is_file():
        pytest.skip("September report model artifact is not available.")
    html = render_report_html(load_report_model(path))

    assert "grouped-goal-chart" in html
    assert "Límite máximo" in html
    assert "Meta mínima" in html
    assert "Tipo de referencia" in html
    assert "Real vs Objetivo" not in html


def test_html_generation_renders_strategic_analysis_fields() -> None:
    """Verify accepted strategy fields appear in the HTML report."""

    html = render_report_html(_sample_report_model())

    assert "La operación requiere atención ejecutiva." in html
    assert "El resultado operativo es negativo." in html
    assert "El gasto creció más rápido que los ingresos." in html
    assert "Estabilizar el flujo de caja." in html
    assert "Revisar aprobaciones de gasto." in html
    assert "No hay recomendaciones estratégicas generadas." not in html


def test_section_templates_define_evidence_contracts() -> None:
    """Verify section templates define objectives and required evidence."""

    expected = {
        "executive_summary",
        "financial_health_overview",
        "kpi_overview",
        "goal_budget_performance",
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
        assert template.narrative_fields or template.section_id in {"appendix", "goal_budget_performance"}


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
    assert "Ingeniería concentra un déficit departamental de -150." in html


def test_current_anomalies_are_separate_from_historical_risks() -> None:
    """Verify current anomaly status does not suppress recurring historical risks."""

    model = _sample_report_model()
    for section in model["sections"]:  # type: ignore[union-attr]
        if section["section_id"] == "anomaly_summary":
            section["content"] = {
                "report_period": "2026-12",
                "total_anomalies": 0,
                "anomalies_by_severity": {"critical": 0, "high": 0},
                "top_anomalies": [],
                "analysis": "No se identificaron anomalías en el período 2026-12, con un total de 0 anomalías reportadas.",
            }
    historical_context = {
        "current_period": "2026_12",
        "summary": {"available_retrievals": 2, "topics": ["get_repeated_anomalies", "get_previous_recommendations"]},
        "retrievals": [
            {
                "tool_name": "get_repeated_anomalies",
                "records": [
                    {"type": "recurring_vendor_duplicate", "latest_severity": "high"},
                    {"type": "negative_cash_flow", "latest_severity": "high"},
                    {"type": "payroll_overtime_overspend", "latest_severity": "medium"},
                ],
            },
            {
                "tool_name": "get_previous_recommendations",
                "records": [
                    {
                        "run_period": "2026_09",
                        "action": "Review overtime and benefits allocation in Health Sciences.",
                        "expected_impact": "Reduce payroll overruns by 20% within Q4.",
                    },
                    {
                        "run_period": "2026_09",
                        "action": "Audit vendor payment process.",
                        "expected_impact": "Ensure compliance and prevent duplicate payments.",
                    },
                ],
            },
        ],
        "derived_context": {
            "kpi_trends": [
                {
                    "metric": "payroll_percentage_of_revenue",
                    "periods": ["2026_04", "2026_11"],
                    "first_value": 0.45,
                    "latest_value": 0.41,
                    "direction": "improving",
                }
            ],
            "recommendation_effectiveness": [
                {
                    "topic": "payroll_overtime",
                    "related_trend": {
                        "periods": ["2026_04", "2026_11"],
                        "first_value": 0.45,
                        "latest_value": 0.41,
                        "direction": "improving",
                    },
                },
                {"topic": "vendor_controls", "related_trend": None},
            ],
            "artifact_anomaly_patterns": [
                {
                    "department": "Health Sciences",
                    "anomaly_type": "payroll_overtime_overspend",
                    "occurrences": 4,
                    "periods": ["2026_04", "2026_05", "2026_06", "2026_07"],
                },
                {
                    "department": "Health Sciences",
                    "anomaly_type": "recurring_vendor_duplicate",
                    "occurrences": 3,
                    "periods": ["2026_07", "2026_08", "2026_09"],
                },
                {
                    "department": "University",
                    "anomaly_type": "negative_cash_flow",
                    "occurrences": 2,
                    "periods": ["2026_06", "2026_07"],
                },
            ],
        },
    }
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_summary",
            "title": "Historical Summary",
            "content": {"historical_context": historical_context},
            "source_references": ["outputs/analysis/strategic_analysis_2026_12.json"],
            "warnings": [],
        }
    )
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "longitudinal_risk_assessment",
            "title": "Longitudinal Risk Assessment",
            "content": {
                "recurring_risks": [
                    {
                        "risk": "Riesgo recurrente",
                        "department": "Health Sciences",
                        "occurrences": "4",
                        "periods": "Abr 2026, May 2026, Jun 2026",
                    },
                    {
                        "risk": "Riesgo recurrente",
                        "department": "University",
                        "occurrences": "2",
                        "periods": "Jun 2026, Jul 2026",
                    },
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_12.json"],
            "warnings": [],
        }
    )

    html = render_report_html(model)

    assert "Hallazgos y anomalías" in html
    assert "No se detectaron desviaciones que superaran los umbrales configurados en Dic 2026." in html
    assert "Riesgos históricos recurrentes" in html
    assert "Seguimiento de recomendaciones emitidas anteriormente" in html
    assert "Las siguientes recomendaciones fueron emitidas en informes financieros de meses anteriores" in html
    assert "Ciencias de la Salud" in html
    assert "Universidad" in html
    assert "Sobrecosto recurrente de nómina y horas extra" in html
    assert "Riesgo recurrente en pagos a proveedores" in html
    assert "Flujo de caja negativo recurrente" in html
    assert "Emitida en" in html
    assert "Estado de seguimiento" in html
    assert "Objetivo alcanzado" in html
    assert "Por qué:" in html
    assert "Por qué es recurrente" in html
    assert "Por qué importa" in html
    assert "Objetivo original" in html
    assert "Próxima acción sugerida" in html
    assert "Sin anomalías relevantes" not in html
    assert "0 anomalías reportadas" not in html
    assert "Riesgo recurrente</h3>" not in html
    assert "Periodo de origen" not in html
    assert "Progreso:" not in html
    assert "Se requiere un análisis adicional" not in html
    assert "Health Sciences" not in html
    assert "University" not in html


def test_report_model_quality_accepts_strategy_backed_model() -> None:
    """Verify quality validation accepts current strategy-backed report models."""

    result = validate_report_model_quality(_sample_report_model())

    assert result.is_valid is True
    assert result.recommendation_count == 1


def test_report_model_quality_warns_for_missing_strategy() -> None:
    """Verify missing strategy is warned about but deterministic reports remain valid."""

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
    assert any("placeholder" in error for error in result.errors)
    assert any("Deterministic strategic synthesis" in warning for warning in result.warnings)


def test_html_pdf_do_not_render_old_recommendations_unavailable_state(tmp_path: Path) -> None:
    """Verify legacy empty recommendation models render deterministic synthesis wording."""

    model = _sample_report_model()
    sections = model["sections"]  # type: ignore[assignment]
    for section in sections:  # type: ignore[union-attr]
        if section["section_id"] == "executive_summary":
            section["content"]["analysis_status"] = "unavailable"
        if section["section_id"] == "strategic_recommendations":
            section["content"]["recommendations"] = []
            section["content"]["strategy_unavailable_note"] = ""

    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "deterministic_synthesis.pdf")

    from pypdf import PdfReader

    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    combined = html + "\n" + pdf_text
    forbidden = (
        "Recomendaciones estratégicas no validadas",
        "No hay recomendaciones estratégicas validadas",
        "Las recomendaciones estratégicas validadas no están disponibles",
        "Strategic analysis was unavailable",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "Modo degradado" in combined


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


def test_historical_trend_series_append_current_period_for_all_renderers(tmp_path: Path) -> None:
    """Verify report-model trend data reaches HTML/PDF presentation consistently."""

    model = _sample_report_model()
    model["period_slug"] = "2026_08"  # type: ignore[index]
    model["report_period"] = "2026_08"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"].update(
                {
                    "total_revenue": 1300,
                    "total_expenses": 1000,
                    "net_operating_result": 300,
                    "net_cash_flow": 200,
                    "ending_cash": 900,
                    "payroll_percentage_of_revenue": 0.40,
                    "collection_rate": 0.93,
                }
            )
    metrics = (
        ("total_revenue", "USD", 1000, 1100),
        ("total_expenses", "USD", 900, 950),
        ("net_operating_result", "USD", 100, 150),
        ("payroll_percentage_of_revenue", "ratio", 0.50, 0.45),
        ("collection_rate", "ratio", 0.84, 0.88),
        ("net_cash_flow", "USD", -100, 50),
        ("ending_cash", "USD", 700, 800),
    )
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": metric,
                        "metric": metric,
                        "unit": unit,
                        "direction": "stable",
                        "points": [
                            {"period": "2026_06", "value": first},
                            {"period": "2026_07", "value": second},
                        ],
                    }
                    for metric, unit, first, second in metrics
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_08.json"],
            "warnings": [],
        }
    )

    view = build_presentation_view(model)
    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "report.pdf")

    assert len(view["historical"]["trends"]) == 7
    assert all(series["points"][-1]["period"] == "2026_08" for series in view["historical"]["trends"])
    assert "Ago 2026" in html
    assert pdf_path.exists()


def test_one_point_historical_series_renders_insufficient_history_card() -> None:
    """Verify one-point trend series do not render as large empty line charts."""

    model = _sample_report_model()
    model["period_slug"] = "2026_06"  # type: ignore[index]
    model["report_period"] = "2026_06"  # type: ignore[index]
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "collection_rate",
                        "metric": "Tasa de cobranza",
                        "unit": "ratio",
                        "direction": "stable",
                        "points": [{"period": "2026_06", "value": 0.9}],
                    }
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_06.json"],
            "warnings": [],
        }
    )

    html = render_report_html(model)

    assert "Historial insuficiente para graficar una tendencia" in html


def test_historical_trend_window_preserves_intermediate_months() -> None:
    """Verify trend charts keep each real month inside the rolling window."""

    model = _sample_report_model()
    model["period_slug"] = "2026_09"  # type: ignore[index]
    model["report_period"] = "2026_09"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"]["collection_rate"] = 0.92
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "collection_rate",
                        "metric": "Tasa de cobranza",
                        "unit": "ratio",
                        "direction": "improving",
                        "points": [
                            {"period": "2026_04", "value": 0.89},
                            {"period": "2026_05", "value": 0.86},
                            {"period": "2026_06", "value": 0.84},
                            {"period": "2026_07", "value": 0.85},
                            {"period": "2026_08", "value": 0.90},
                        ],
                    }
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )

    view = build_presentation_view(model)
    series = view["historical"]["trends"][0]

    assert [point["period"] for point in series["points"]] == [
        "2026_04",
        "2026_05",
        "2026_06",
        "2026_07",
        "2026_08",
        "2026_09",
    ]
    assert series["window"]["missing_periods"] == []


def test_historical_trend_window_limits_to_six_months() -> None:
    """Verify older months are dropped while all months inside the window remain."""

    model = _sample_report_model()
    model["period_slug"] = "2026_09"  # type: ignore[index]
    model["report_period"] = "2026_09"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"]["total_revenue"] = 1_090_000
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "total_revenue",
                        "metric": "Ingresos",
                        "unit": "USD",
                        "points": [
                            {"period": f"2026_{month:02d}", "value": 1_000_000 + month}
                            for month in range(1, 9)
                        ],
                    }
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )

    series = build_presentation_view(model)["historical"]["trends"][0]

    assert [point["period"] for point in series["points"]] == [
        "2026_04",
        "2026_05",
        "2026_06",
        "2026_07",
        "2026_08",
        "2026_09",
    ]
    assert "2026_03" not in [point["period"] for point in series["points"]]
    assert len(series["points"]) == 6


def test_february_historical_trend_uses_two_available_months() -> None:
    """Verify early-year reports show the available Jan-Feb progression."""

    model = _sample_report_model()
    model["period_slug"] = "2026_02"  # type: ignore[index]
    model["report_period"] = "2026_02"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"]["payroll_percentage_of_revenue"] = 0.39
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "payroll_percentage_of_revenue",
                        "metric": "Nómina / ingresos",
                        "unit": "ratio",
                        "points": [{"period": "2026_01", "value": 0.38}],
                    }
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )

    series = build_presentation_view(model)["historical"]["trends"][0]

    assert [point["period"] for point in series["points"]] == ["2026_01", "2026_02"]
    assert series["window"]["missing_periods"] == ["2025_09", "2025_10", "2025_11", "2025_12"]


def test_missing_months_are_not_fabricated_in_historical_trends() -> None:
    """Verify gaps are metadata only and do not create artificial chart points."""

    model = _sample_report_model()
    model["period_slug"] = "2026_09"  # type: ignore[index]
    model["report_period"] = "2026_09"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"]["net_cash_flow"] = -20_000
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "net_cash_flow",
                        "metric": "Flujo neto de caja",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_06", "value": -680_000},
                            {"period": "2026_08", "value": -120_000},
                        ],
                    }
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )

    series = build_presentation_view(model)["historical"]["trends"][0]

    assert [point["period"] for point in series["points"]] == ["2026_06", "2026_08", "2026_09"]
    assert "2026_07" in series["window"]["missing_periods"]
    assert all(point["period"] != "2026_07" for point in series["points"])


def test_pdf_chart_input_preserves_exact_september_revenue_and_expense_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify PDF chart input receives exact Apr-Sep points, not endpoints."""

    model = _sample_report_model()
    model["period_slug"] = "2026_09"  # type: ignore[index]
    model["report_period"] = "2026_09"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"].update({"total_revenue": 2_123_856.0, "total_expenses": 2_096_356.0})
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "total_revenue",
                        "metric": "Ingresos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_018_940.0},
                            {"period": "2026_05", "value": 2_005_584.0},
                            {"period": "2026_06", "value": 1_992_060.0},
                            {"period": "2026_07", "value": 2_021_376.0},
                            {"period": "2026_08", "value": 2_072_448.0},
                        ],
                    },
                    {
                        "metric_id": "total_expenses",
                        "metric": "Gastos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_084_940.0},
                            {"period": "2026_05", "value": 2_126_584.0},
                            {"period": "2026_06", "value": 2_366_060.0},
                            {"period": "2026_07", "value": 2_213_876.0},
                            {"period": "2026_08", "value": 2_138_448.0},
                        ],
                    },
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )
    captured: dict[str, list[tuple[str, float]]] = {}
    original_line_chart = pdf_renderer_module.LineChart

    class SpyLineChart(original_line_chart):
        """Capture exact PDF chart input points."""

        def __init__(self, series: dict[str, object], width: float = 6.7 * 72) -> None:
            """Record series points before delegating to the real PDF flowable."""

            super().__init__(series, width=width)
            metric = str(series.get("metric_id") or "")
            if metric in {"total_revenue", "total_expenses"}:
                captured[metric] = [
                    (str(point.get("period")), float(point.get("value")))
                    for point in series.get("points", [])  # type: ignore[union-attr]
                    if isinstance(point, dict)
                ]
                captured[f"{metric}_dimensions"] = [("width", self.width), ("height", self.height)]

    monkeypatch.setattr(pdf_renderer_module, "LineChart", SpyLineChart)

    render_report_pdf(model, tmp_path / "september.pdf")

    assert captured["total_revenue"] == [
        ("2026_04", 2_018_940.0),
        ("2026_05", 2_005_584.0),
        ("2026_06", 1_992_060.0),
        ("2026_07", 2_021_376.0),
        ("2026_08", 2_072_448.0),
        ("2026_09", 2_123_856.0),
    ]
    assert captured["total_expenses"] == [
        ("2026_04", 2_084_940.0),
        ("2026_05", 2_126_584.0),
        ("2026_06", 2_366_060.0),
        ("2026_07", 2_213_876.0),
        ("2026_08", 2_138_448.0),
        ("2026_09", 2_096_356.0),
    ]
    assert len(captured["total_revenue"]) == 6
    assert len(captured["total_expenses"]) == 6
    assert dict(captured["total_revenue_dimensions"])["width"] <= 3.25 * 72
    assert 170 <= dict(captured["total_revenue_dimensions"])["height"] <= 210


def test_html_chart_input_preserves_exact_september_revenue_and_expense_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HTML chart helper receives all six September trend points."""

    model = _sample_report_model()
    model["period_slug"] = "2026_09"  # type: ignore[index]
    model["report_period"] = "2026_09"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"].update({"total_revenue": 2_123_856.0, "total_expenses": 2_096_356.0})
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "total_revenue",
                        "metric": "Ingresos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_018_940.0},
                            {"period": "2026_05", "value": 2_005_584.0},
                            {"period": "2026_06", "value": 1_992_060.0},
                            {"period": "2026_07", "value": 2_021_376.0},
                            {"period": "2026_08", "value": 2_072_448.0},
                        ],
                    },
                    {
                        "metric_id": "total_expenses",
                        "metric": "Gastos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_084_940.0},
                            {"period": "2026_05", "value": 2_126_584.0},
                            {"period": "2026_06", "value": 2_366_060.0},
                            {"period": "2026_07", "value": 2_213_876.0},
                            {"period": "2026_08", "value": 2_138_448.0},
                        ],
                    },
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )
    from finance_agent.reporting.renderers import html_renderer as html_renderer_module

    captured: dict[str, list[tuple[str, float]]] = {}
    original_line_chart = html_renderer_module._line_chart

    def spy_line_chart(series: dict[str, object]) -> str:
        """Capture exact HTML chart helper points."""

        metric = str(series.get("metric_id") or "")
        if metric in {"total_revenue", "total_expenses"}:
            captured[metric] = [
                (str(point.get("period_label")), float(point.get("value")))
                for point in series.get("points", [])  # type: ignore[union-attr]
                if isinstance(point, dict)
            ]
        return original_line_chart(series)

    monkeypatch.setattr(html_renderer_module, "_line_chart", spy_line_chart)

    html = render_report_html(model)

    assert captured["total_revenue"] == [
        ("Abr 2026", 2_018_940.0),
        ("May 2026", 2_005_584.0),
        ("Jun 2026", 1_992_060.0),
        ("Jul 2026", 2_021_376.0),
        ("Ago 2026", 2_072_448.0),
        ("Sep 2026", 2_123_856.0),
    ]
    assert captured["total_expenses"] == [
        ("Abr 2026", 2_084_940.0),
        ("May 2026", 2_126_584.0),
        ("Jun 2026", 2_366_060.0),
        ("Jul 2026", 2_213_876.0),
        ("Ago 2026", 2_138_448.0),
        ("Sep 2026", 2_096_356.0),
    ]
    for label in ("Abr 2026", "May 2026", "Jun 2026", "Jul 2026", "Ago 2026", "Sep 2026"):
        assert label in html
    assert html.count("<circle") >= 12


def test_generated_september_pdf_contains_all_month_labels_and_compact_charts(tmp_path: Path) -> None:
    """Verify the final PDF artifact exposes all labels and compact chart dimensions."""

    model = _sample_report_model()
    model["period_slug"] = "2026_09"  # type: ignore[index]
    model["report_period"] = "2026_09"  # type: ignore[index]
    for section in model["sections"]:  # type: ignore[index]
        if section["section_id"] == "financial_health_overview":
            section["content"].update({"total_revenue": 2_123_856.0, "total_expenses": 2_096_356.0})
    model["sections"].append(  # type: ignore[union-attr]
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "total_revenue",
                        "metric": "Ingresos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_018_940.0},
                            {"period": "2026_05", "value": 2_005_584.0},
                            {"period": "2026_06", "value": 1_992_060.0},
                            {"period": "2026_07", "value": 2_021_376.0},
                            {"period": "2026_08", "value": 2_072_448.0},
                        ],
                    }
                ]
            },
            "source_references": [],
            "warnings": [],
        }
    )
    pdf_path = render_report_pdf(model, tmp_path / "september_labels.pdf")

    from pypdf import PdfReader

    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)

    for label in ("Abr 2026", "May 2026", "Jun 2026", "Jul 2026", "Ago 2026", "Sep 2026"):
        assert label in text
    chart = pdf_renderer_module.LineChart({"points": [{"period": "2026_04", "value": 1}, {"period": "2026_05", "value": 2}]})
    assert chart.width <= 3.25 * 72
    assert 170 <= chart.height <= 210


def test_adaptive_axis_domain_uses_local_range_without_forcing_zero() -> None:
    """Verify high-value financial charts emphasize local variation truthfully."""

    assert adaptive_axis_domain([2_018_940.0, 2_005_584.0, 1_992_060.0, 2_123_856.0]) == (
        pytest.approx(1_978_880.4),
        pytest.approx(2_137_035.6),
    )
    ratio_min, ratio_max = adaptive_axis_domain([0.91, 0.92, 0.93, 0.94])
    assert ratio_min == pytest.approx(0.907)
    assert ratio_max == pytest.approx(0.943)
    assert ratio_min > 0
    assert ratio_max < 1


def test_adaptive_axis_domain_keeps_zero_when_series_crosses_zero() -> None:
    """Verify cross-zero charts keep zero visible without forcing all-positive charts to zero."""

    y_min, y_max = adaptive_axis_domain([-150_000.0, -80_000.0, 20_000.0, 60_000.0])

    assert y_min < 0
    assert y_max > 0
    assert y_min == pytest.approx(-171_000.0)
    assert y_max == pytest.approx(81_000.0)


def test_html_pdf_and_streamlit_historical_charts_share_axis_domain() -> None:
    """Verify every renderer uses the same adaptive y-axis limits."""

    from finance_agent.reporting.renderers import html_renderer as html_renderer_module
    from finance_agent.ui import streamlit_app

    revenue_series = {
        "metric_id": "total_revenue",
        "metric": "Ingresos totales",
        "unit": "USD",
        "points": [
            {"period": "2026_04", "value": 2_018_940.0},
            {"period": "2026_05", "value": 2_005_584.0},
            {"period": "2026_06", "value": 1_992_060.0},
            {"period": "2026_07", "value": 2_021_376.0},
            {"period": "2026_08", "value": 2_072_448.0},
            {"period": "2026_09", "value": 2_123_856.0},
        ],
    }
    expected = adaptive_axis_domain([point["value"] for point in revenue_series["points"]])

    streamlit_domain = tuple(streamlit_app._trend_chart_spec(revenue_series)["encoding"]["y"]["scale"]["domain"])
    pdf_domain = pdf_renderer_module.LineChart(revenue_series).y_axis_domain
    html = html_renderer_module._line_chart(revenue_series)

    assert streamlit_domain == pytest.approx(expected)
    assert pdf_domain == pytest.approx(expected)
    assert "data-y-min='1978880.4'" in html
    assert "data-y-max='213703" in html
    assert expected[0] > 0


def test_august_report_renders_all_historical_charts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify August's rolling-window series render as seven HTML/PDF charts."""

    path = Path("outputs/report/report_model_2026_08.json")
    if not path.is_file():
        pytest.skip("August report model artifact is not available.")
    model = load_report_model(path)
    chart_calls: list[str] = []
    original_line_chart = pdf_renderer_module.LineChart

    class SpyLineChart(original_line_chart):
        """Count PDF historical chart flowables created by the renderer."""

        def __init__(self, series: dict[str, object], width: float = 6.7 * 72) -> None:
            """Record the metric and delegate to the real chart flowable."""

            chart_calls.append(str(series.get("metric_id") or series.get("metric")))
            super().__init__(series, width=width)

    monkeypatch.setattr(pdf_renderer_module, "LineChart", SpyLineChart)

    html = render_report_html(model)
    render_report_pdf(model, tmp_path / "august.pdf")

    assert html.count("line-chart") >= 7
    assert len(chart_calls) == 7
    for label in ("Mar 2026", "Abr 2026", "May 2026", "Jun 2026", "Jul 2026", "Ago 2026"):
        assert label in html
    assert "historial insuficiente" not in html.casefold()


def test_missing_and_empty_sections_render_gracefully(tmp_path: Path) -> None:
    """Verify missing sections and empty tables produce readable placeholders."""

    model = _sample_report_model()
    model["sections"] = [section for section in model["sections"] if section["section_id"] != "kpi_overview"]  # type: ignore[index]
    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "missing.pdf")

    assert "Sección faltante en el modelo: kpi_overview" not in html
    assert "No hay KPIs disponibles para este periodo." in html
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_strategy_validation_warns_for_unavailable_analysis() -> None:
    """Verify final rendering guard allows deterministic fallback reports."""

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
    validate_strategy_available(model)


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


def test_executive_html_uses_dashboard_visuals_and_insights() -> None:
    """Verify upgraded report presentation includes cards, badges, axes, and insights."""

    html = render_report_html(_sample_report_model())

    assert "Insight ejecutivo" not in html
    assert "Conclusión ejecutiva:" in html
    assert "Periodo anterior:" in html
    assert "Variación respecto al presupuesto:" in html
    assert "class='grid-line'" in html
    assert "class='axis-label'" in html
    assert "current-bar" in html
    assert "recommendation-card" in html


def test_executive_html_replaces_low_value_tables_with_status_cards() -> None:
    """Verify sparse evidence tables become compact executive status cards."""

    model = _sample_report_model()
    for section in model["sections"]:  # type: ignore[union-attr]
        if section["section_id"] == "investigation_evidence":
            section["content"]["evidence_items"] = [
                {"priority": "medium", "retrieval_name": "", "record_count": "", "evidence_summary": ""}
            ]
        if section["section_id"] == "kpi_overview":
            section["content"]["kpis"] = []

    html = render_report_html(model)

    assert "No hay KPIs disponibles para este periodo." in html
    assert "La evidencia recuperada está disponible" not in html
    assert "<table><thead><tr><th>Prioridad</th><th>Evidencia</th><th>Registros</th><th>Resumen</th>" not in html


def test_executive_html_uses_executive_spanish_kpi_labels() -> None:
    """Verify KPI cards avoid technical labels and compact unavailable abbreviations."""

    html = render_report_html(_sample_report_model())

    assert "Delta" not in html
    assert "N/D" not in html
    assert "Variación respecto al periodo anterior" in html
    assert "No disponible</small>" not in html


def test_executive_html_has_no_mojibake_text() -> None:
    """Verify user-facing HTML renders corrected UTF-8 Spanish text."""

    html = render_report_html(_sample_report_model())

    for fragment in ("Ã", "Â", "â†", "SÃ­ntesis", "DescripciÃ³n"):
        assert fragment not in html
    assert "Síntesis ejecutiva" in html
    assert "Descripción" in html


def test_final_outputs_reject_known_english_presentation_labels(tmp_path: Path) -> None:
    """Verify HTML/PDF presentation helper labels stay in Spanish."""

    from pypdf import PdfReader

    model = _sample_report_model()
    html = render_report_html(model)
    pdf_path = render_report_pdf(model, tmp_path / "report.pdf")
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)

    forbidden_labels = (
        "Executive Insight",
        "Current period",
        "Direction:",
        "Previous period",
        "Budget delta",
        "Suggested owner",
        "Status:",
    )
    for label in forbidden_labels:
        assert label not in html
        assert label not in pdf_text
    assert "Conclusión ejecutiva:" in html
    assert "Conclusión ejecutiva:" in pdf_text


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

