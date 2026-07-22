"""Professional PDF renderer for Finance AI Agent report models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from finance_agent.reporting.presentation import SECTION_LABELS_ES, build_presentation_view, format_value


NAVY = colors.HexColor("#17324d")
BLUE = colors.HexColor("#245b89")
GREEN = colors.HexColor("#1f7a5b")
RED = colors.HexColor("#b84242")
AMBER = colors.HexColor("#b7791f")
LINE = colors.HexColor("#d8e1ea")
LIGHT = colors.HexColor("#f4f7fb")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#647084")


class HorizontalBarChart(Flowable):
    """Small vector bar chart flowable.

    Inputs: display-ready chart items, title, width, and height.
    Outputs: ReportLab flowable drawing real bars.
    Assumptions: values were calculated upstream; this only scales them visually.
    """

    def __init__(self, items: list[dict[str, Any]], title: str, width: float = 6.8 * inch) -> None:
        """Initialize a bar chart flowable.

        Inputs: chart item dictionaries and title.
        Outputs: configured flowable.
        Assumptions: long labels are truncated for PDF readability.
        """

        super().__init__()
        self.items = items
        self.title = title
        self.width = width
        self.height = 32 + max(1, len(items)) * 24

    def draw(self) -> None:
        """Draw the bar chart onto the PDF canvas.

        Inputs: flowable state.
        Outputs: mutates ReportLab canvas.
        Assumptions: canvas coordinate system starts at flowable origin.
        """

        canvas = self.canv
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(NAVY)
        canvas.drawString(0, self.height - 12, self.title)
        if not self.items:
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(0, self.height - 34, "Sin datos para graficar.")
            return
        max_value = max(abs(float(item.get("value") or 0.0)) for item in self.items) or 1.0
        label_width = 1.75 * inch
        value_width = 1.0 * inch
        bar_width = self.width - label_width - value_width - 0.2 * inch
        for index, item in enumerate(self.items):
            y = self.height - 36 - index * 24
            value = float(item.get("value") or 0.0)
            label = _truncate(str(item.get("label") or ""), 24)
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(INK)
            canvas.drawString(0, y + 3, label)
            canvas.setFillColor(colors.HexColor("#e8eef5"))
            canvas.roundRect(label_width, y, bar_width, 10, 4, fill=1, stroke=0)
            canvas.setFillColor(GREEN if value >= 0 else RED)
            scaled = max(2.0, abs(value) / max_value * bar_width)
            canvas.roundRect(label_width, y, scaled, 10, 4, fill=1, stroke=0)
            canvas.setFillColor(INK)
            canvas.drawRightString(self.width, y + 2, format_value(value, item.get("unit")))


class LineChart(Flowable):
    """Small vector line chart flowable.

    Inputs: trend series and width.
    Outputs: ReportLab flowable drawing a line chart.
    Assumptions: points are ordered chronologically upstream.
    """

    def __init__(self, series: dict[str, Any], width: float = 3.2 * inch) -> None:
        """Initialize a line chart.

        Inputs: trend series dictionary and width.
        Outputs: configured flowable.
        Assumptions: empty series will show an empty-state note.
        """

        super().__init__()
        self.series = series
        self.width = width
        self.height = 1.65 * inch

    def draw(self) -> None:
        """Draw the line chart onto the PDF canvas.

        Inputs: flowable state.
        Outputs: mutates ReportLab canvas.
        Assumptions: values are normalized only for visual scale.
        """

        canvas = self.canv
        points = self.series.get("points", [])
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(0, self.height - 10, _truncate(str(self.series.get("metric") or ""), 34))
        if not points:
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)
            canvas.drawString(0, self.height - 28, "Sin puntos históricos.")
            return
        values = [float(point.get("value") or 0.0) for point in points]
        min_value = min(values)
        max_value = max(values)
        span = max(max_value - min_value, 1e-9)
        left = 12
        bottom = 24
        chart_width = self.width - 24
        chart_height = self.height - 52
        coords = []
        for index, point in enumerate(points):
            x = left + chart_width * (index / max(1, len(points) - 1))
            y = bottom + ((float(point.get("value") or 0.0) - min_value) / span) * chart_height
            coords.append((x, y))
        canvas.setStrokeColor(LINE)
        canvas.line(left, bottom, left + chart_width, bottom)
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(1.4)
        path = canvas.beginPath()
        path.moveTo(coords[0][0], coords[0][1])
        for x, y in coords[1:]:
            path.lineTo(x, y)
        canvas.drawPath(path)
        canvas.setFillColor(GREEN)
        for x, y in coords:
            canvas.circle(x, y, 2.5, fill=1, stroke=0)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0, 2, f"{points[0].get('period')} → {points[-1].get('period')}")


def _styles() -> dict[str, ParagraphStyle]:
    """Build PDF paragraph styles.

    Inputs: none.
    Outputs: style dictionary.
    Assumptions: Helvetica handles Spanish text generated here.
    """

    sample = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle("CoverTitle", parent=sample["Title"], fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=colors.white, spaceAfter=16),
        "cover": ParagraphStyle("CoverText", parent=sample["BodyText"], fontName="Helvetica", fontSize=13, leading=18, textColor=colors.white),
        "h1": ParagraphStyle("SectionHeading", parent=sample["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=10, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("SubHeading", parent=sample["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NAVY, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=sample["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.5, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("Small", parent=sample["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9, textColor=MUTED),
        "card_value": ParagraphStyle("CardValue", parent=sample["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=NAVY),
    }


def _para(value: Any, style: ParagraphStyle) -> Paragraph:
    """Create a safe ReportLab paragraph.

    Inputs: scalar display value and style.
    Outputs: Paragraph flowable.
    Assumptions: text is plain and should be XML escaped.
    """

    text = "" if value is None else str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, style)


def _truncate(text: str, limit: int) -> str:
    """Truncate text for compact chart labels.

    Inputs: source text and character limit.
    Outputs: shortened text.
    Assumptions: full wording is available in adjacent tables/cards.
    """

    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _table(headers: list[str], rows: list[list[Any]], styles: dict[str, ParagraphStyle], *, widths: list[float] | None = None) -> Table:
    """Create a styled PDF table with repeating headers.

    Inputs: headers, rows, styles, and optional column widths.
    Outputs: ReportLab Table.
    Assumptions: rows are bounded for executive reports.
    """

    data = [[_para(header, styles["small"]) for header in headers]]
    if not rows:
        rows = [["Sin datos disponibles."] + [""] * max(0, len(headers) - 1)]
    for row in rows:
        data.append([_para(value, styles["small"]) for value in row])
    table = Table(data, repeatRows=1, hAlign="LEFT", colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4fb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
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


def _section_title(story: list[Any], section_id: str, styles: dict[str, ParagraphStyle]) -> None:
    """Append a section title.

    Inputs: story list, section ID, and styles.
    Outputs: mutates story.
    Assumptions: Spanish labels are centrally defined.
    """

    story.append(_para(SECTION_LABELS_ES.get(section_id, section_id), styles["h1"]))


def _append_narrative(story: list[Any], view: dict[str, Any], section_id: str, styles: dict[str, ParagraphStyle]) -> None:
    """Append Step-9-authored section narrative when present.

    Inputs: story list, presentation view, section ID, and styles.
    Outputs: mutates story with one paragraph.
    Assumptions: narrative has already passed Spanish/evidence validation.
    """

    text = view.get("section_narratives", {}).get(section_id, "")
    if text:
        story.append(_para(text, styles["body"]))


def _bullet_list(items: list[str], styles: dict[str, ParagraphStyle], *, limit: int = 8) -> list[Any]:
    """Build bullet paragraphs.

    Inputs: text items, styles, and limit.
    Outputs: paragraph flowables.
    Assumptions: empty lists should not create empty tables.
    """

    return [_para(f"- {item}", styles["body"]) for item in items[:limit]]


def _metric_cards(view: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    """Render KPI cards as a two-row table.

    Inputs: presentation view and styles.
    Outputs: card-style table.
    Assumptions: cards are display-ready.
    """

    cells = []
    for card in view["financial_health"]["cards"][:6]:
        cells.append([
            _para(card["label"], styles["small"]),
            _para(card["value"], styles["card_value"]),
            _para(card["description"], styles["small"]),
        ])
    rows = []
    for index in range(0, len(cells), 3):
        rows.append(cells[index:index + 3])
    while rows and len(rows[-1]) < 3:
        rows[-1].append([_para("", styles["small"])])
    table = Table(rows, colWidths=[2.15 * inch, 2.15 * inch, 2.15 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.25, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, LINE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfdff")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _recommendation_cards(view: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Render recommendations as card-like tables.

    Inputs: presentation view and styles.
    Outputs: flowables.
    Assumptions: recommendations are strategic-analysis outputs.
    """

    flowables: list[Any] = []
    for card in view["recommendations"]["cards"][:6]:
        data = [
            [_para(f"Prioridad: {card['priority']}", styles["h2"])],
            [_para(card["action"], styles["body"])],
            [_para(f"Racional: {card['rationale']}", styles["small"])],
            [_para(f"Impacto esperado: {card['expected_impact']}", styles["small"])],
            [_para(f"Responsable/estado: {card['owner_status']}", styles["small"])],
        ]
        table = Table(data, colWidths=[6.7 * inch], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfdff")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        flowables.extend([KeepTogether(table), Spacer(1, 0.08 * inch)])
    return flowables


def _build_story(report_model: dict[str, Any], *, mode: str = "executive") -> list[Any]:
    """Build ReportLab flowables from a presentation view.

    Inputs: report model and render mode.
    Outputs: list of ReportLab flowables.
    Assumptions: presentation layer has already sanitized executive content.
    """

    view = build_presentation_view(report_model, mode=mode)
    styles = _styles()
    story: list[Any] = []

    story.append(_para(view["title"], styles["cover_title"]))
    story.append(_para(f"Periodo: {view.get('period')}", styles["cover"]))
    story.append(_para(view.get("organization"), styles["cover"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(_para("Síntesis ejecutiva generada desde salidas procesadas, validadas y trazables.", styles["cover"]))
    story.append(PageBreak())

    _section_title(story, "executive_summary", styles)
    story.append(_para(view["executive_summary"]["summary"], styles["body"]))
    story.append(_para("Hallazgos clave", styles["h2"]))
    story.extend(_bullet_list(view["executive_summary"]["key_findings"], styles, limit=6))
    story.append(_para("Causas raíz probables", styles["h2"]))
    story.extend(_bullet_list(view["executive_summary"]["root_causes"], styles, limit=6))
    story.append(_para(f"Confianza del análisis: {view['executive_summary']['confidence']}", styles["small"]))

    _section_title(story, "financial_health_overview", styles)
    _append_narrative(story, view, "financial_health_overview", styles)
    story.append(_metric_cards(view, styles))
    chart_items = [
        {"label": card["label"], "value": card["numeric_value"] or 0.0, "unit": card["unit"]}
        for card in view["financial_health"]["cards"]
        if card["id"] in {"total_revenue", "total_expenses", "net_operating_result", "net_cash_flow"}
    ]
    story.append(Spacer(1, 0.1 * inch))
    story.append(HorizontalBarChart(chart_items, "Resumen financiero principal"))

    _section_title(story, "kpi_overview", styles)
    _append_narrative(story, view, "kpi_overview", styles)
    story.append(
        _table(
            ["Indicador", "Valor", "Estado", "Descripción"],
            [[row["indicator"], row["value"], row["status"], row["description"]] for row in view["kpis"][:10]],
            styles,
            widths=[1.55 * inch, 0.9 * inch, 0.9 * inch, 3.15 * inch],
        )
    )

    historical = view["historical"]
    if historical.get("available"):
        _section_title(story, "historical_trends", styles)
        _append_narrative(story, view, "historical_summary", styles)
        _append_narrative(story, view, "historical_trends", styles)
        chart_cells = [[LineChart(series) for series in historical.get("trends", [])[:2]]]
        if chart_cells[0]:
            story.append(Table(chart_cells, colWidths=[3.3 * inch] * len(chart_cells[0])))
        _section_title(story, "recommendation_follow_up", styles)
        _append_narrative(story, view, "recommendation_follow_up", styles)
        story.append(
            _table(
                ["Recomendación", "Periodo", "Evidencia actual", "Estado"],
                [
                    [row["recommendation"], row["issued_period"], row["current_evidence"], row["status"]]
                    for row in historical.get("recommendation_follow_up", [])[:6]
                ],
                styles,
                widths=[1.7 * inch, 0.75 * inch, 3.0 * inch, 1.0 * inch],
            )
        )
        _section_title(story, "longitudinal_risk_assessment", styles)
        _append_narrative(story, view, "longitudinal_risk_assessment", styles)
        story.append(
            _table(
                ["Riesgo", "Departamento", "Ocurrencias", "Periodos"],
                [
                    [row["risk"], row["department"], row["occurrences"], row["periods"]]
                    for row in historical.get("recurring_risks", [])[:6]
                ],
                styles,
                widths=[2.1 * inch, 1.4 * inch, 0.8 * inch, 2.0 * inch],
            )
        )

    _section_title(story, "revenue_expense_analysis", styles)
    _append_narrative(story, view, "revenue_expense_analysis", styles)
    story.append(HorizontalBarChart(view["revenue_expense"]["chart"], "Ingresos, gastos y resultado"))
    story.append(HorizontalBarChart(view["revenue_expense"]["budget_chart"], "Comparación contra presupuesto"))
    story.append(
        _table(
            ["Métrica", "Valor", "Descripción"],
            [[row["metric"], row["value"], row["description"]] for row in view["revenue_expense"]["rows"]],
            styles,
            widths=[1.7 * inch, 1.0 * inch, 3.8 * inch],
        )
    )

    _section_title(story, "department_analysis", styles)
    _append_narrative(story, view, "department_analysis", styles)
    story.append(
        _table(
            ["Departamento", "Ingresos", "Gastos", "Resultado", "Var. gasto"],
            [[row["department"], row["revenue"], row["expenses"], row["result"], row["variance"]] for row in view["departments"]],
            styles,
            widths=[1.55 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch, 0.9 * inch],
        )
    )
    story.append(HorizontalBarChart(
        [{"label": row["department"], "value": row["numeric_result"], "unit": "USD"} for row in view["departments"][:6]],
        "Resultado operativo por departamento",
    ))

    _section_title(story, "anomaly_summary", styles)
    _append_narrative(story, view, "anomaly_summary", styles)
    anomalies = view["anomalies"]
    if anomalies.get("positive_status"):
        story.append(_para(anomalies["positive_status"], styles["body"]))
    else:
        story.append(_table(["Severidad", "Cantidad"], [[row["severity"], row["count"]] for row in anomalies["severity_rows"]], styles))
        story.append(
            _table(
                ["Anomalía", "Severidad", "Evidencia"],
                [[row["title"], row["severity"], row["evidence"]] for row in anomalies["top_rows"]],
                styles,
                widths=[2.1 * inch, 0.9 * inch, 3.4 * inch],
            )
        )

    _section_title(story, "investigation_evidence", styles)
    story.append(
        _table(
            ["Prioridad", "Evidencia", "Registros", "Resumen"],
            [[row["priority"], row["evidence"], row["records"], row["summary"]] for row in view["evidence"][:8]],
            styles,
            widths=[0.75 * inch, 1.35 * inch, 0.65 * inch, 3.7 * inch],
        )
    )

    _section_title(story, "strategic_recommendations", styles)
    _append_narrative(story, view, "strategic_recommendations", styles)
    story.append(_para("Prioridades estratégicas", styles["h2"]))
    story.extend(_bullet_list(view["recommendations"]["priorities"], styles, limit=6))
    story.extend(_recommendation_cards(view, styles))
    if view["recommendations"]["reasoning_summary"]:
        story.append(_para(view["recommendations"]["reasoning_summary"], styles["small"]))

    _section_title(story, "missing_information", styles)
    _append_narrative(story, view, "missing_information", styles)
    story.extend(_bullet_list(view["missing_information"], styles, limit=8))

    _section_title(story, "appendix", styles)
    story.append(_para("Metodología", styles["h2"]))
    story.extend(_bullet_list(view["appendix"]["methodology"], styles, limit=6))
    story.append(_para("Fuentes procesadas", styles["h2"]))
    story.extend(_bullet_list(view["appendix"]["sources"], styles, limit=18))
    return story


def _draw_page_frame(canvas: Any, document: Any) -> None:
    """Draw page header/footer and cover background.

    Inputs: ReportLab canvas and document.
    Outputs: mutates PDF canvas.
    Assumptions: page one is the cover.
    """

    canvas.saveState()
    if document.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#245b89"))
        canvas.circle(letter[0] - 80, letter[1] - 80, 120, fill=1, stroke=0)
    else:
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.65 * inch, letter[1] - 0.4 * inch, "Reporte financiero ejecutivo")
        canvas.drawRightString(letter[0] - 0.65 * inch, 0.42 * inch, f"Página {document.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(0.65 * inch, letter[1] - 0.48 * inch, letter[0] - 0.65 * inch, letter[1] - 0.48 * inch)
    canvas.restoreState()


def render_report_pdf(report_model: dict[str, Any], output_path: str | Path, *, mode: str = "executive") -> Path:
    """Render and save a report model as a polished PDF.

    Inputs: report model dictionary, PDF output path, and rendering mode.
    Outputs: resolved written path.
    Assumptions: renderers do not change business logic or financial values.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.65 * inch,
    )
    document.build(_build_story(report_model, mode=mode), onFirstPage=_draw_page_frame, onLaterPages=_draw_page_frame)
    return path
