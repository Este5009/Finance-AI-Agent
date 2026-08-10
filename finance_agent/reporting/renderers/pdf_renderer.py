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

from finance_agent.reporting.presentation import (
    SECTION_LABELS_ES,
    adaptive_axis_domain,
    build_presentation_view,
    deterministic_chart_insight,
    display_or_unavailable,
    format_compact_axis_value,
    format_period_label,
    format_value,
    historical_chart_series,
    table_has_useful_detail,
    trim_low_value_columns,
    validate_historical_chart_rendering,
)


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
        chart_top = self.height - 28
        chart_bottom = 12
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.35)
        canvas.setFont("Helvetica", 5.8)
        canvas.setFillColor(MUTED)
        for tick in range(5):
            x = label_width + bar_width * tick / 4
            canvas.line(x, chart_bottom, x, chart_top)
            canvas.drawCentredString(x, 2, format_compact_axis_value(max_value * tick / 4, self.items[0].get("unit")))
        abs_values = [abs(float(item.get("value") or 0.0)) for item in self.items]
        max_index = abs_values.index(max(abs_values))
        min_index = abs_values.index(min(abs_values))
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
            if index in {len(self.items) - 1, max_index, min_index}:
                canvas.setStrokeColor(NAVY if index == len(self.items) - 1 else AMBER)
                canvas.setLineWidth(0.65)
                canvas.roundRect(label_width, y, scaled, 10, 4, fill=0, stroke=1)
            canvas.setFillColor(INK)
            canvas.drawRightString(self.width, y + 2, format_value(value, item.get("unit")))


class LineChart(Flowable):
    """Small vector line chart flowable.

    Inputs: trend series and width.
    Outputs: ReportLab flowable drawing a line chart.
    Assumptions: points are ordered chronologically upstream.
    """

    def __init__(self, series: dict[str, Any], width: float = 3.15 * inch) -> None:
        """Initialize a line chart.

        Inputs: trend series dictionary and width.
        Outputs: configured flowable.
        Assumptions: empty series will show an empty-state note.
        """

        super().__init__()
        self.series = series
        self.width = width
        values = [
            float(point.get("value") or 0.0)
            for point in series.get("points", [])
            if isinstance(point, dict)
        ]
        self.y_axis_domain = adaptive_axis_domain(values) if values else (0.0, 1.0)
        # Keep historical charts compact enough for a two-column executive
        # layout while preserving readable axes and point markers.
        self.height = 2.6 * inch

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
        data_min = min(values)
        data_max = max(values)
        min_value, max_value = self.y_axis_domain
        span = max(max_value - min_value, 1e-9)
        left = 0.58 * inch
        bottom = 0.56 * inch
        right = 0.16 * inch
        top = 0.34 * inch
        chart_width = self.width - left - right
        chart_height = self.height - bottom - top
        coords = []
        for index, point in enumerate(points):
            x = left + chart_width * (index / max(1, len(points) - 1))
            y = bottom + ((float(point.get("value") or 0.0) - min_value) / span) * chart_height
            coords.append((x, y, point))
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.35)
        for tick in range(5):
            y = bottom + chart_height * tick / 4
            canvas.line(left, y, left + chart_width, y)
            value = min_value + span * tick / 4
            canvas.setFont("Helvetica", 6)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(left - 4, y - 2, format_value(value, self.series.get("unit")))
        canvas.setStrokeColor(MUTED)
        canvas.line(left, bottom, left + chart_width, bottom)
        canvas.line(left, bottom, left, bottom + chart_height)
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(1.4)
        path = canvas.beginPath()
        path.moveTo(coords[0][0], coords[0][1])
        for x, y, _ in coords[1:]:
            path.lineTo(x, y)
        canvas.drawPath(path)
        min_index = values.index(data_min)
        max_index = values.index(data_max)
        for index, (x, y, _) in enumerate(coords):
            canvas.setFillColor(
                NAVY if index == len(coords) - 1 else (GREEN if index == max_index else (RED if index == min_index else BLUE))
            )
            canvas.circle(x, y, 2.5, fill=1, stroke=0)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(MUTED)
        for x, _, point in coords:
            label = str(point.get("period_label") or format_period_label(point.get("period")))
            canvas.saveState()
            canvas.translate(x, bottom - 10)
            # Six monthly labels fit much better when slightly rotated; every
            # supplied point still receives a visible marker and month label.
            if len(coords) <= 6:
                canvas.rotate(32)
                canvas.drawCentredString(0, 0, label)
            else:
                canvas.drawCentredString(0, 0, label)
            canvas.restoreState()
        canvas.setFont("Helvetica-Bold", 6.3)
        canvas.setFillColor(NAVY)
        canvas.drawCentredString(left + chart_width / 2, 6, "Periodo")
        canvas.saveState()
        canvas.translate(8, bottom + chart_height / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "Porcentaje" if self.series.get("unit") == "ratio" else "Valor")
        canvas.restoreState()


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
        "h1": ParagraphStyle("SectionHeading", parent=sample["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=18, spaceAfter=12, keepWithNext=True),
        "h2": ParagraphStyle("SubHeading", parent=sample["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NAVY, spaceBefore=12, spaceAfter=7, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=sample["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.2, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=sample["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.8, textColor=MUTED, spaceAfter=4),
        "source_badge": ParagraphStyle("SourceBadge", parent=sample["BodyText"], fontName="Helvetica-Bold", fontSize=6.9, leading=8.5, textColor=MUTED, backColor=colors.HexColor("#f3f7fb"), borderColor=LINE, borderWidth=0.3, borderPadding=3, spaceAfter=6),
        "card_value": ParagraphStyle("CardValue", parent=sample["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=NAVY),
        "insight": ParagraphStyle("ExecutiveConclusion", parent=sample["BodyText"], fontName="Helvetica", fontSize=8, leading=11.2, textColor=colors.HexColor("#24364a"), backColor=colors.HexColor("#eef7ff"), borderColor=BLUE, borderWidth=0.4, borderPadding=7, spaceBefore=8, spaceAfter=14),
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


def _rich_para(value: Any, style: ParagraphStyle) -> Paragraph:
    """Create a paragraph with renderer-owned inline markup.

    Inputs: trusted renderer text with limited tags and paragraph style.
    Outputs: ReportLab paragraph.
    Assumptions: callers pass fixed labels, not untrusted narrative markup.
    """

    return Paragraph(str(value or ""), style)


def _info_card(message: str, styles: dict[str, ParagraphStyle], *, title: str = "Estado actual") -> Table:
    """Create a compact PDF information card.

    Inputs: message, styles, and title.
    Outputs: card-style ReportLab table.
    Assumptions: used instead of empty or low-value tables.
    """

    table = Table(
        [[_rich_para(f"<b>{title}:</b> {message}", styles["small"])]],
        colWidths=[6.6 * inch],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.45, LINE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfdff")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _insight_para(value: Any, style: ParagraphStyle) -> Paragraph:
    """Create a deterministic conclusion paragraph with bold Spanish label.

    Inputs: conclusion text and paragraph style.
    Outputs: ReportLab paragraph with bold label.
    Assumptions: value has no markup and is already presentation-safe.
    """

    text = "" if value is None else str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(f"<b>Conclusión ejecutiva:</b> {text}", style)


def _truncate(text: str, limit: int) -> str:
    """Truncate text for compact chart labels.

    Inputs: source text and character limit.
    Outputs: shortened text.
    Assumptions: full wording is available in adjacent tables/cards.
    """

    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _table(
    headers: list[str],
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    *,
    widths: list[float] | None = None,
    empty: str = "No hay datos suficientes para mostrar una tabla útil.",
    force: bool = False,
) -> Table:
    """Create an adaptive styled PDF table with repeating headers.

    Inputs: headers, rows, styles, and optional column widths.
    Outputs: ReportLab Table or card-like empty state.
    Assumptions: source artifacts retain full detail outside the executive PDF.
    """

    if not rows:
        return _info_card(empty, styles)
    headers, rows = trim_low_value_columns(headers, rows, protected_headers=(headers[0],))
    if not force and not table_has_useful_detail(headers, rows):
        return _info_card("La evidencia existe, pero no contiene suficiente detalle tabular para esta sección.", styles)
    if widths and len(widths) != len(headers):
        total_width = sum(widths)
        widths = [total_width / len(headers)] * len(headers)
    data = [[_para(header, styles["small"]) for header in headers]]
    for row in rows:
        padded = list(row[: len(headers)]) + [""] * max(0, len(headers) - len(row))
        data.append([_para(display_or_unavailable(value), styles["small"]) for value in padded])
    table = Table(data, repeatRows=1, hAlign="LEFT", colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4fb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _compact_cards(items: list[dict[str, Any]], styles: dict[str, ParagraphStyle], *, kind: str) -> list[Table]:
    """Create compact executive cards for short historical lists.

    Inputs: display-ready presentation items, PDF styles, and card kind.
    Outputs: ReportLab card tables.
    Assumptions: cards are presentation-only; source artifacts keep full details.
    """

    cards: list[Table] = []
    for item in items:
        if kind == "follow_up":
            lines = [
                f"<b>{display_or_unavailable(item.get('recommendation'))}</b>",
                f"<b>Emitida en:</b> {display_or_unavailable(item.get('issued_period'))}",
                f"<b>Estado de seguimiento:</b> {display_or_unavailable(item.get('progress') or item.get('status'))}",
                f"<b>Por qué:</b> {display_or_unavailable(item.get('status_reason'))}",
                f"<b>Objetivo original:</b> {display_or_unavailable(item.get('objective'))}",
                f"<b>Evidencia actual:</b> {display_or_unavailable(item.get('current_evidence'))}",
                f"<b>Próxima acción sugerida:</b> {display_or_unavailable(item.get('next_action'))}",
            ]
        else:
            lines = [
                f"<b>{display_or_unavailable(item.get('risk'))}</b>",
                f"<b>Qué pasó:</b> {display_or_unavailable(item.get('what_happened'))}",
                f"<b>Departamento:</b> {display_or_unavailable(item.get('department'))}",
                f"<b>Por qué es recurrente:</b> {display_or_unavailable(item.get('recurrence_reason'))}",
                f"<b>Tendencia de recurrencia:</b> {display_or_unavailable(item.get('recurrence_direction'))}",
                f"<b>Por qué importa:</b> {display_or_unavailable(item.get('management_relevance'))}",
                f"<b>Frecuencia:</b> {display_or_unavailable(item.get('frequency') or item.get('occurrences'))}",
                f"<b>Estado de recurrencia:</b> {display_or_unavailable(item.get('status'))}",
                f"<b>Períodos afectados:</b> {display_or_unavailable(item.get('periods'))}",
            ]
        table = Table(
            [[_rich_para("<br/>".join(lines), styles["small"])]],
            colWidths=[6.6 * inch],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.45, LINE),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfdff")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        cards.append(table)
    return cards


def _section_title(
    story: list[Any],
    section_id: str,
    styles: dict[str, ParagraphStyle],
    view: dict[str, Any] | None = None,
) -> None:
    """Append a section title.

    Inputs: story list, section ID, styles, and optional presentation view.
    Outputs: mutates story.
    Assumptions: Spanish labels and generation provenance are centrally defined.
    """

    story.append(_para(SECTION_LABELS_ES.get(section_id, section_id), styles["h1"]))
    badge = (view or {}).get("generation_sources", {}).get(section_id, {})
    if isinstance(badge, dict) and badge.get("label"):
        story.append(_para(str(badge.get("label")), styles["source_badge"]))


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
        comparison_rows = card.get("comparison_rows", [])
        comparison_flowables = [
            _para(f"{row.get('label')}: {row.get('value')}", styles["small"])
            for row in comparison_rows
            if row.get("value")
        ]
        cells.append([
            _para(f"{card['label']} - {card.get('badge', {}).get('label', '')}", styles["small"]),
            _para(f"{card['value']}", styles["card_value"]),
            *comparison_flowables,
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
            [_para(f"Responsable sugerido: {card.get('owner')}", styles["small"])],
            [_para(f"Estado: {card.get('status')}", styles["small"])],
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

    _section_title(story, "executive_summary", styles, view)
    story.append(_para(view["executive_summary"]["summary"], styles["body"]))
    story.append(_para("Hallazgos clave", styles["h2"]))
    story.extend(_bullet_list(view["executive_summary"]["key_findings"], styles, limit=6))
    story.append(_para("Causas raíz probables", styles["h2"]))
    story.extend(_bullet_list(view["executive_summary"]["root_causes"], styles, limit=6))
    story.append(_para(f"Confianza del análisis: {view['executive_summary']['confidence']}", styles["small"]))

    _section_title(story, "financial_health_overview", styles, view)
    _append_narrative(story, view, "financial_health_overview", styles)
    story.append(_metric_cards(view, styles))
    chart_items = [
        {"label": card["label"], "value": card["numeric_value"] or 0.0, "unit": card["unit"]}
        for card in view["financial_health"]["cards"]
        if card["id"] in {"total_revenue", "total_expenses", "net_operating_result", "net_cash_flow"}
    ]
    story.append(Spacer(1, 0.1 * inch))
    story.append(HorizontalBarChart(chart_items, "Resumen financiero principal"))
    story.append(_insight_para(view["financial_health"].get("chart_insight", ""), styles["insight"]))

    _section_title(story, "kpi_overview", styles, view)
    _append_narrative(story, view, "kpi_overview", styles)
    if len(view["kpis"]) > 6:
        story.append(
            _table(
                ["Indicador", "Valor", "Estado", "Descripción"],
                [[row["indicator"], row["value"], row["status"], row["description"]] for row in view["kpis"][:10]],
                styles,
                widths=[1.55 * inch, 0.9 * inch, 0.9 * inch, 3.15 * inch],
                force=True,
            )
        )
    else:
        story.append(_info_card("Los KPIs principales ya están resumidos en las tarjetas de salud financiera.", styles, title="Lectura ejecutiva"))

    goal_budget = view.get("goal_budget", {})
    if goal_budget.get("available"):
        _section_title(story, "goal_budget_performance", styles, view)
        story.append(
            _info_card(
                goal_budget.get("conclusion") or "Cumplimiento calculado con datos procesados.",
                styles,
                title="Conclusión ejecutiva",
            )
        )
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            _table(
                ["Indicador", "Valor"],
                [
                    ["Cumplimiento general", goal_budget.get("overall_score", "")],
                    [
                        "Metas cumplidas",
                        f"{goal_budget.get('met_goal_count', 0)}/{goal_budget.get('valid_goal_count', 0)}",
                    ],
                    ["Metas en riesgo o críticas", str(goal_budget.get("risk_goal_count", 0) + goal_budget.get("critical_goal_count", 0))],
                    ["Método de ponderación", goal_budget.get("weighting_method", "")],
                ],
                styles,
                widths=[2.4 * inch, 3.8 * inch],
                force=True,
            )
        )
        for group in goal_budget.get("chart_groups", [])[:2]:
            chart_rows: list[dict[str, Any]] = []
            for row in group.get("rows", [])[:8]:
                chart_rows.append(
                    {
                        "label": f"{row.get('metric')} · {row.get('series')}",
                        "value": row.get("value"),
                        "unit": group.get("unit") or "",
                    }
                )
            if chart_rows:
                story.append(Spacer(1, 0.08 * inch))
                story.append(HorizontalBarChart(chart_rows, str(group.get("title") or "Real vs referencia")))
        rows = [
            [
                item["label"],
                item["actual"],
                item["target"],
                item.get("reference_label", "Meta"),
                item["gap"],
                item["score"],
                item["status"],
            ]
            for item in goal_budget.get("items", [])[:8]
        ]
        story.append(
            _table(
                ["Meta", "Real", "Referencia", "Tipo", "Brecha", "Puntaje", "Estado"],
                rows,
                styles,
                widths=[1.35 * inch, 0.8 * inch, 0.85 * inch, 0.8 * inch, 0.85 * inch, 0.75 * inch, 1.0 * inch],
                force=True,
            )
        )

    historical = view["historical"]
    if historical.get("available"):
        _section_title(story, "historical_trends", styles, view)
        _append_narrative(story, view, "historical_summary", styles)
        _append_narrative(story, view, "historical_trends", styles)
        chartable_ids = {id(series) for series in historical_chart_series(historical)}
        rendered_chart_count = 0
        chart_cells: list[list[Any]] = []
        for series in historical.get("trends", []):
            points = series.get("points", []) if isinstance(series, dict) else []
            if id(series) in chartable_ids:
                chart_cells.append(
                    [
                        LineChart(series, width=3.15 * inch),
                        _insight_para(series.get("insight", ""), styles["insight"]),
                    ]
                )
                rendered_chart_count += 1
            else:
                available = points[0] if points else {}
                message = (
                    f"{series.get('metric') or 'Indicador histórico'}: historial insuficiente para graficar "
                    f"una tendencia. Dato disponible: "
                    f"{available.get('period_label') or format_period_label(available.get('period'))} — "
                    f"{available.get('display') or format_value(available.get('value'), series.get('unit'))}."
                )
                chart_cells.append([_info_card(message, styles, title="Historial insuficiente")])
        if chart_cells:
            rows = []
            for index in range(0, len(chart_cells), 2):
                row = chart_cells[index : index + 2]
                if len(row) == 1:
                    row.append("")
                rows.append(row)
            chart_table = Table(
                rows,
                colWidths=[3.22 * inch, 3.22 * inch],
                hAlign="LEFT",
                spaceBefore=4,
                spaceAfter=10,
            )
            chart_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(chart_table)
        validate_historical_chart_rendering(
            historical,
            rendered_chart_count,
            renderer_name="PDF renderer",
        )
        _section_title(story, "recommendation_follow_up", styles, view)
        if historical.get("recommendation_intro"):
            story.append(_para(historical.get("recommendation_intro"), styles["body"]))
        if historical.get("recommendation_summary"):
            story.append(_info_card(historical.get("recommendation_summary"), styles, title="Lectura ejecutiva"))
            story.append(Spacer(1, 0.08 * inch))
        follow_items = historical.get("recommendation_follow_up", [])[:6]
        if 0 < len(follow_items) <= 5:
            for card in _compact_cards(follow_items, styles, kind="follow_up"):
                story.append(card)
                story.append(Spacer(1, 0.08 * inch))
        else:
            follow_rows = [
                [row["recommendation"], row["issued_period"], row.get("progress", row.get("status", "")), row.get("status_reason", ""), row.get("objective", ""), row["current_evidence"]]
                for row in follow_items
            ]
            story.append(
                _table(
                    ["Recomendación", "Emitida en", "Estado de seguimiento", "Por qué", "Objetivo original", "Evidencia actual"],
                    follow_rows,
                    styles,
                    widths=[1.15 * inch, 0.7 * inch, 0.9 * inch, 1.45 * inch, 1.25 * inch, 1.15 * inch],
                )
            )
        _section_title(story, "longitudinal_risk_assessment", styles, view)
        if historical.get("risk_summary"):
            story.append(_info_card(historical.get("risk_summary"), styles, title="Lectura ejecutiva"))
            story.append(Spacer(1, 0.08 * inch))
        risk_items = historical.get("recurring_risks", [])[:6]
        if 0 < len(risk_items) <= 3:
            for card in _compact_cards(risk_items, styles, kind="risk"):
                story.append(card)
                story.append(Spacer(1, 0.08 * inch))
        else:
            story.append(
                _table(
                    ["Riesgo", "Departamento", "Frecuencia", "Estado de recurrencia", "Períodos afectados"],
                    [
                        [row["risk"], row["department"], row.get("frequency", row.get("occurrences", "")), row.get("status", ""), row["periods"]]
                        for row in risk_items
                    ],
                    styles,
                    widths=[1.75 * inch, 1.3 * inch, 0.8 * inch, 0.9 * inch, 1.85 * inch],
                )
            )

    _section_title(story, "revenue_expense_analysis", styles, view)
    _append_narrative(story, view, "revenue_expense_analysis", styles)
    story.append(HorizontalBarChart(view["revenue_expense"]["chart"], "Ingresos, gastos y resultado"))
    story.append(_insight_para(view["revenue_expense"].get("chart_insight", ""), styles["insight"]))
    story.append(HorizontalBarChart(view["revenue_expense"]["budget_chart"], "Comparación contra presupuesto"))
    story.append(_insight_para(view["revenue_expense"].get("budget_chart_insight", ""), styles["insight"]))
    story.append(
        _table(
            ["Métrica", "Valor", "Descripción"],
            [[row["metric"], row["value"], row["description"]] for row in view["revenue_expense"]["rows"]],
            styles,
            widths=[1.7 * inch, 1.0 * inch, 3.8 * inch],
        )
    )

    _section_title(story, "department_analysis", styles, view)
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
    if view["departments"]:
        story.append(_insight_para(
            deterministic_chart_insight(
                [{"label": row["department"], "value": row["numeric_result"], "unit": "USD"} for row in view["departments"][:6]],
                title="resultado operativo por departamento",
                chart_kind="department",
                unit="USD",
            ),
            styles["insight"],
        ))

    _section_title(story, "anomaly_summary", styles, view)
    anomalies = view["anomalies"]
    if anomalies.get("current_period_status") or anomalies.get("positive_status"):
        story.append(_info_card(anomalies.get("current_period_status") or anomalies.get("positive_status"), styles))
        if anomalies.get("distinction_note"):
            story.append(_para(anomalies.get("distinction_note"), styles["body"]))
    else:
        _append_narrative(story, view, "anomaly_summary", styles)
        story.append(HorizontalBarChart(anomalies.get("severity_chart", []), "Hallazgos por severidad"))
        story.append(_insight_para(anomalies.get("chart_insight", ""), styles["insight"]))
        if len(anomalies["severity_rows"]) > 4:
            story.append(_table(["Severidad", "Cantidad"], [[row["severity"], row["count"]] for row in anomalies["severity_rows"]], styles, force=True))
        if len(anomalies["top_rows"]) > 3:
            story.append(
                _table(
                    ["Hallazgo", "Clasificación", "Sev.", "Valor", "Referencia", "Origen", "Motivo"],
                    [
                        [
                            row["title"],
                            row.get("classification"),
                            row["severity"],
                            row.get("expense_variance") or row.get("observed_value"),
                            row.get("expense_variance_pct") or row.get("reference_value"),
                            row.get("reference_origin"),
                            row.get("reason_for_flagging") or row["evidence"],
                        ]
                        for row in anomalies["top_rows"]
                    ],
                    styles,
                    widths=[1.2 * inch, 1.05 * inch, 0.45 * inch, 0.75 * inch, 0.75 * inch, 1.05 * inch, 1.7 * inch],
                    force=True,
                )
            )

    _section_title(story, "investigation_evidence", styles, view)
    evidence_rows = [[row["priority"], row["evidence"], row["records"], row["summary"]] for row in view["evidence"][:8]]
    story.append(
        _table(
            ["Prioridad", "Evidencia", "Registros", "Resumen"],
            evidence_rows,
            styles,
            widths=[0.75 * inch, 1.35 * inch, 0.65 * inch, 3.7 * inch],
            empty="No se solicitó evidencia adicional para este periodo.",
        )
    )

    _section_title(story, "strategic_recommendations", styles, view)
    _append_narrative(story, view, "strategic_recommendations", styles)
    if view["recommendations"]["priorities"]:
        story.append(_para("Prioridades estratégicas", styles["h2"]))
        story.extend(_bullet_list(view["recommendations"]["priorities"], styles, limit=6))
    if not view["recommendations"]["cards"]:
        story.append(
            _info_card(
                view["recommendations"].get("strategy_unavailable_note")
                or (
                    "El reporte conserva hallazgos verificados, KPIs, anomalías, historial y evidencia procesada "
                    "para orientar la revisión ejecutiva."
                ),
                styles,
                title="Modo degradado: análisis determinístico",
            )
        )
        attention_items = view["recommendations"].get("attention_items", [])
        if attention_items:
            story.append(_para("Hallazgos determinísticos que requieren atención", styles["h2"]))
            story.append(
                _table(
                    ["Hallazgo", "Severidad", "Indicador", "Periodo", "Evidencia"],
                    [
                        [
                            item.get("display_title_es") or item.get("title"),
                            item.get("severity"),
                            item.get("metric"),
                            item.get("period"),
                            item.get("display_evidence_es") or item.get("evidence"),
                        ]
                        for item in attention_items[:6]
                    ],
                    styles,
                    widths=[1.4 * inch, 0.75 * inch, 1.1 * inch, 0.75 * inch, 2.5 * inch],
                    force=True,
                )
            )
    elif len(view["recommendations"]["cards"]) <= 5:
        story.extend(_recommendation_cards(view, styles))
    else:
        story.append(
            _table(
                ["Prioridad", "Acción", "Impacto esperado", "Responsable", "Estado"],
                [
                    [card["priority"], card["action"], card["expected_impact"], card.get("owner"), card.get("status")]
                    for card in view["recommendations"]["cards"]
                ],
                styles,
                widths=[0.75 * inch, 2.3 * inch, 1.6 * inch, 1.2 * inch, 0.75 * inch],
                force=True,
            )
        )
    _section_title(story, "missing_information", styles, view)
    _append_narrative(story, view, "missing_information", styles)
    if view["missing_information"]:
        story.extend(_bullet_list(view["missing_information"], styles, limit=8))
    else:
        story.append(_info_card("No se reportan brechas de información relevantes.", styles, title="Estado de información"))

    _section_title(story, "appendix", styles, view)
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
