"""PDF renderer for Finance AI Agent report models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from finance_agent.reporting.renderers.html_renderer import (
    SECTION_LABELS_ES,
    _format_value,
    _number,
    _section,
    report_strategy_warnings,
    validate_strategy_available,
)


def _styles() -> dict[str, ParagraphStyle]:
    """Build report styles for PDF output.

    Inputs: none.
    Outputs: style dictionary.
    Assumptions: built-in Helvetica supports required Spanish text.
    """

    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#17324d"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "SectionHeading",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#4c5968"),
        ),
    }


def _para(value: Any, style: ParagraphStyle) -> Paragraph:
    """Create a ReportLab paragraph from a value.

    Inputs: any display value and paragraph style.
    Outputs: Paragraph flowable.
    Assumptions: XML-sensitive characters are escaped by reportlab paragraph parser minimally.
    """

    text = "" if value is None else str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, style)


def _table(headers: list[str], rows: list[list[Any]], styles: dict[str, ParagraphStyle]) -> Table:
    """Create a compact ReportLab table.

    Inputs: headers, row values, and style dictionary.
    Outputs: styled Table flowable.
    Assumptions: tables are summary-sized in the report model.
    """

    data = [[_para(header, styles["small"]) for header in headers]]
    if not rows:
        rows = [["Sin datos disponibles"] + [""] * max(0, len(headers) - 1)]
    for row in rows:
        data.append([_para(value, styles["small"]) for value in row])
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4fb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8e1ea")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _metric_table(content: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    """Render key financial health values as a table.

    Inputs: section content and PDF styles.
    Outputs: ReportLab Table.
    Assumptions: values are already calculated by upstream stages.
    """

    rows = [
        ["Ingresos totales", _format_value(content.get("total_revenue"), "USD")],
        ["Gastos totales", _format_value(content.get("total_expenses"), "USD")],
        ["Resultado operativo", _format_value(content.get("net_operating_result"), "USD")],
        ["Flujo neto de caja", _format_value(content.get("net_cash_flow"), "USD")],
        ["Caja final", _format_value(content.get("ending_cash"), "USD")],
        ["Cobranza", _format_value(content.get("collection_rate"), "ratio")],
    ]
    return _table(["Métrica", "Valor"], rows, styles)


def _simple_bar_rows(items: list[tuple[str, float]]) -> list[list[str]]:
    """Build text-based bar chart rows for PDF tables.

    Inputs: label/value pairs.
    Outputs: rows with ASCII bars and formatted values.
    Assumptions: this keeps PDF dependencies simple and renderer-neutral.
    """

    if not items:
        return [["Sin datos", "", ""]]
    max_value = max(abs(value) for _, value in items) or 1.0
    rows = []
    for label, value in items:
        bar_length = max(1, int(abs(value) / max_value * 22))
        rows.append([label, "#" * bar_length, _format_value(value, "USD")])
    return rows


def _add_section_title(story: list[Any], section_id: str, styles: dict[str, ParagraphStyle]) -> None:
    """Append a Spanish section title to the PDF story.

    Inputs: story list, section ID, and styles.
    Outputs: mutates story list.
    Assumptions: section labels come from the shared Spanish label map.
    """

    story.append(_para(SECTION_LABELS_ES[section_id], styles["h1"]))


def _add_source_note(story: list[Any], section: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    """Append compact source references for one section.

    Inputs: story list, section dictionary, and styles.
    Outputs: mutates story list when sources exist.
    Assumptions: source references are processed-output paths, not raw evidence.
    """

    sources = section.get("source_references", [])
    if sources:
        story.append(_para(f"Fuentes: {'; '.join(str(source) for source in sources[:5])}", styles["small"]))


def _add_warning_note(story: list[Any], section: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    """Append visible section warnings to the PDF story.

    Inputs: story list, section dictionary, and styles.
    Outputs: mutates story list when warnings exist.
    Assumptions: warnings should be visible in draft reports.
    """

    warnings = section.get("warnings", [])
    for warning in warnings[:5]:
        story.append(_para(f"Advertencia: {warning}", styles["small"]))


def _build_story(report_model: dict[str, Any]) -> list[Any]:
    """Build ReportLab flowables from a report model.

    Inputs: report model dictionary.
    Outputs: list of ReportLab flowables.
    Assumptions: content is summarized enough for a readable PDF.
    """

    styles = _styles()
    story: list[Any] = []
    cover = _section(report_model, "cover").get("content", {})
    story.append(_para(cover.get("title", "Reporte financiero"), styles["title"]))
    story.append(_para(f"Periodo: {report_model.get('report_period')}", styles["body"]))
    story.append(_para("Generado por Finance AI Agent a partir de datos procesados.", styles["body"]))
    story.append(Spacer(1, 0.18 * inch))

    executive_section = _section(report_model, "executive_summary")
    executive = executive_section.get("content", {})
    _add_section_title(story, "executive_summary", styles)
    _add_warning_note(story, executive_section, styles)
    story.append(_para(executive.get("summary", "Sin resumen disponible."), styles["body"]))
    findings = executive.get("key_findings") or ["No hay hallazgos estratégicos generados."]
    story.append(_para("Hallazgos clave", styles["body"]))
    for item in findings[:6]:
        story.append(_para(f"- {item}", styles["body"]))
    story.append(_para("Causas raíz probables", styles["body"]))
    for item in (executive.get("root_causes") or ["Sin causas raíz estratégicas disponibles."])[:6]:
        story.append(_para(f"- {item}", styles["body"]))
    _add_source_note(story, executive_section, styles)

    health_section = _section(report_model, "financial_health_overview")
    health = health_section.get("content", {})
    _add_section_title(story, "financial_health_overview", styles)
    story.append(_metric_table(health, styles))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        _table(
            ["Concepto", "Barra", "Valor"],
            _simple_bar_rows(
                [
                    ("Ingresos", _number(health.get("total_revenue")) or 0.0),
                    ("Gastos", _number(health.get("total_expenses")) or 0.0),
                    ("Resultado", _number(health.get("net_operating_result")) or 0.0),
                ]
            ),
            styles,
        )
    )
    _add_source_note(story, health_section, styles)

    kpi_section = _section(report_model, "kpi_overview")
    kpis = kpi_section.get("content", {}).get("kpis", [])
    _add_section_title(story, "kpi_overview", styles)
    story.append(
        _table(
            ["Indicador", "Valor", "Unidad", "Estado"],
            [
                [
                    item.get("metric"),
                    _format_value(item.get("value"), item.get("unit")),
                    item.get("unit"),
                    item.get("availability"),
                ]
                for item in kpis[:12]
                if isinstance(item, dict)
            ],
            styles,
        )
    )
    _add_source_note(story, kpi_section, styles)

    for section_id in ("revenue_analysis", "expense_analysis"):
        section = _section(report_model, section_id)
        content = section.get("content", {})
        _add_section_title(story, section_id, styles)
        rows = [
            [key, _format_value(value, "ratio" if "pct" in key else "USD")]
            for key, value in content.items()
            if not key.endswith("summary") and not isinstance(value, list)
        ]
        story.append(_table(["Métrica", "Valor"], rows, styles))
        _add_source_note(story, section, styles)

    department_section = _section(report_model, "department_analysis")
    departments = department_section.get("content", {}).get("department_summary", [])
    _add_section_title(story, "department_analysis", styles)
    story.append(
        _table(
            ["Departamento", "Ingresos", "Gastos", "Resultado"],
            [
                [
                    item.get("department"),
                    _format_value(item.get("actual_revenue"), "USD"),
                    _format_value(item.get("actual_expenses"), "USD"),
                    _format_value(item.get("net_operating_result"), "USD"),
                ]
                for item in departments
                if isinstance(item, dict)
            ],
            styles,
        )
    )
    _add_source_note(story, department_section, styles)

    anomaly_section = _section(report_model, "anomaly_summary")
    anomalies = anomaly_section.get("content", {})
    _add_section_title(story, "anomaly_summary", styles)
    severity = anomalies.get("anomalies_by_severity", {})
    severity = severity if isinstance(severity, dict) else {}
    story.append(_table(["Severidad", "Cantidad"], [[k, v] for k, v in severity.items()], styles))
    top_rows = [
        [item.get("anomaly_id"), item.get("title"), item.get("severity")]
        for item in anomalies.get("top_anomalies", [])[:8]
        if isinstance(item, dict)
    ]
    story.append(_table(["ID", "Título", "Severidad"], top_rows, styles))
    _add_source_note(story, anomaly_section, styles)

    evidence_section = _section(report_model, "investigation_evidence")
    evidence = evidence_section.get("content", {}).get("evidence_items", [])
    _add_section_title(story, "investigation_evidence", styles)
    story.append(
        _table(
            ["Tarea", "Prioridad", "Evidencia", "Registros"],
            [
                [
                    item.get("task_id"),
                    item.get("priority"),
                    item.get("retrieval_name"),
                    item.get("record_count"),
                ]
                for item in evidence[:12]
                if isinstance(item, dict)
            ],
            styles,
        )
    )
    _add_source_note(story, evidence_section, styles)

    recommendation_section = _section(report_model, "strategic_recommendations")
    recommendations = recommendation_section.get("content", {}).get("recommendations", [])
    recommendation_content = recommendation_section.get("content", {})
    _add_section_title(story, "strategic_recommendations", styles)
    _add_warning_note(story, recommendation_section, styles)
    priorities = recommendation_content.get("strategic_priorities", [])
    story.append(_para("Prioridades estratégicas", styles["body"]))
    story.append(
        _table(
            ["Prioridad"],
            [[item] for item in priorities if isinstance(item, str)],
            styles,
        )
    )
    root_causes = recommendation_content.get("root_causes", [])
    story.append(_para("Causas raíz", styles["body"]))
    story.append(
        _table(
            ["Causa probable"],
            [[item] for item in root_causes if isinstance(item, str)],
            styles,
        )
    )
    recommendation_rows = []
    for item in recommendations:
        if isinstance(item, dict):
            recommendation_rows.append([item.get("priority", ""), item.get("action", ""), item.get("expected_impact", "")])
        else:
            recommendation_rows.append(["", item, ""])
    if not recommendation_rows:
        recommendation_rows = [["", "No hay recomendaciones estratégicas generadas.", ""]]
    story.append(_para("Acciones recomendadas", styles["body"]))
    story.append(_table(["Prioridad", "Acción", "Impacto"], recommendation_rows, styles))
    if recommendation_content.get("reasoning_summary"):
        story.append(_para(recommendation_content.get("reasoning_summary"), styles["body"]))
    _add_source_note(story, recommendation_section, styles)

    for section_id in (
        "historical_summary",
        "historical_trends",
        "recommendation_follow_up",
        "longitudinal_risk_assessment",
    ):
        if not any(
            isinstance(section, dict) and section.get("section_id") == section_id
            for section in report_model.get("sections", [])
        ):
            continue
        historical_section = _section(report_model, section_id)
        _add_section_title(story, section_id, styles)
        content = historical_section.get("content", {})
        rows = [
            [key, str(value)[:900]]
            for key, value in content.items()
        ]
        story.append(_table(["Campo", "Resumen"], rows, styles))
        _add_source_note(story, historical_section, styles)

    missing_section = _section(report_model, "missing_information")
    missing = missing_section.get("content", {})
    _add_section_title(story, "missing_information", styles)
    missing_items = missing.get("missing_information") or ["Sin información faltante declarada."]
    for item in missing_items[:8]:
        story.append(_para(f"- {item}", styles["body"]))
    _add_source_note(story, missing_section, styles)

    story.append(PageBreak())
    appendix_section = _section(report_model, "appendix")
    appendix = appendix_section.get("content", {})
    _add_section_title(story, "appendix", styles)
    story.append(
        _table(
            ["Campo", "Detalle"],
            [
                [key, "; ".join(str(item) for item in value[:12]) if isinstance(value, list) else value]
                for key, value in appendix.items()
            ],
            styles,
        )
    )
    _add_source_note(story, appendix_section, styles)
    return story


def _draw_footer(canvas: Any, document: Any) -> None:
    """Draw page footer with page number.

    Inputs: ReportLab canvas and document.
    Outputs: mutates PDF canvas.
    Assumptions: footer is presentation-only metadata.
    """

    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(0.75 * inch, 0.45 * inch, "Finance AI Agent - Reporte financiero")
    canvas.drawRightString(7.75 * inch, 0.45 * inch, f"Página {document.page}")
    canvas.restoreState()


def render_report_pdf(report_model: dict[str, Any], output_path: str | Path) -> Path:
    """Render and save a report model as PDF.

    Inputs: report model dictionary and PDF output path.
    Outputs: resolved written path.
    Assumptions: PDF is a clean summary renderer, not a pixel-perfect final design.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
    )
    document.build(_build_story(report_model), onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return path
