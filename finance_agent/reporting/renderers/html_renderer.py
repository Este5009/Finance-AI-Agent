"""Professional HTML renderer for Finance AI Agent report models."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from finance_agent.reporting.presentation import (
    SECTION_LABELS_ES,
    build_presentation_view,
    deterministic_chart_insight,
    display_or_unavailable,
    format_compact_axis_value,
    format_period_label,
    format_value,
    get_section,
    historical_chart_series,
    is_empty_display_value,
    number_value,
    table_has_useful_detail,
    trim_low_value_columns,
    validate_historical_chart_rendering,
)


def _escape(value: Any) -> str:
    """Escape a value for safe HTML output.

    Inputs: any scalar value.
    Outputs: HTML-escaped string.
    Assumptions: report content is plain text, not trusted markup.
    """

    return html.escape("" if value is None else str(value))


def _info_card(title: str, message: str, *, klass: str = "neutral") -> str:
    """Render a compact informational card.

    Inputs: title, message, and visual class.
    Outputs: HTML card markup.
    Assumptions: cards replace tables that would waste executive attention.
    """

    return (
        f"<div class='status-card {klass} info-panel'>"
        f"<strong>{_escape(title)}</strong><p>{_escape(message)}</p>"
        "</div>"
    )


def _badge_cell(value: Any, klass: str = "neutral") -> str:
    """Render a compact status badge cell.

    Inputs: display value and CSS class.
    Outputs: badge HTML.
    Assumptions: color is paired with readable text.
    """

    return f"<span class='badge {klass}'>{_escape(display_or_unavailable(value, compact=False))}</span>"


def _cell_html(header: str, value: Any) -> str:
    """Render one table cell with status badges where useful.

    Inputs: table header and cell value.
    Outputs: escaped cell HTML.
    Assumptions: table values are display-only and already bounded.
    """

    text = display_or_unavailable(value, compact=False)
    normalized = str(text).lower()
    if any(word in header.lower() for word in ("estado", "severidad", "prioridad")):
        klass = "neutral"
        if any(word in normalized for word in ("crítica", "alta", "riesgo", "desfavorable")):
            klass = "risk"
        elif any(word in normalized for word in ("media", "atención", "seguimiento")):
            klass = "amber"
        elif any(word in normalized for word in ("baja", "disponible", "en rango", "favorable")):
            klass = "good"
        return _badge_cell(text, klass)
    return _escape(text)


def _table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    empty: str = "No hay datos suficientes para mostrar una tabla útil.",
    low_value_title: str = "Información disponible",
    low_value_message: str = "Los artefactos procesados no contienen suficiente detalle tabular para esta sección.",
    force: bool = False,
) -> str:
    """Render an adaptive responsive HTML table.

    Inputs: headers, rows, and empty-state text.
    Outputs: HTML table markup or a compact informational card.
    Assumptions: source artifacts retain full details even when tables are hidden.
    """

    if not rows:
        return _info_card("Estado actual", empty, klass="positive")
    headers, rows = trim_low_value_columns(headers, rows, protected_headers=(headers[0],))
    if not force and not table_has_useful_detail(headers, rows):
        return _info_card(low_value_title, low_value_message)
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{_cell_html(headers[index], value)}</td>"
            for index, value in enumerate(row[: len(headers)])
        ) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _summary_cards(items: list[dict[str, Any]], *, kind: str) -> str:
    """Render compact adaptive cards for evidence, risks, or follow-up rows.

    Inputs: item dictionaries and semantic kind.
    Outputs: card grid HTML.
    Assumptions: used for small datasets where cards beat tables.
    """

    cards: list[str] = []
    if kind == "evidence":
        for item in items:
            summary = display_or_unavailable(item.get("summary"))
            records = display_or_unavailable(item.get("records"))
            cards.append(
                "<article class='status-card evidence-card'>"
                f"<em class='badge neutral'>{_escape(item.get('priority'))}</em>"
                f"<h3>{_escape(item.get('evidence'))}</h3>"
                + (f"<p>{_escape(summary)}</p>" if summary != "No disponible" else "")
                + (f"<small>Registros: {_escape(records)}</small>" if records != "No disponible" else "")
                + "</article>"
            )
    elif kind == "follow_up":
        for item in items:
            period = display_or_unavailable(item.get("issued_period"))
            evidence = display_or_unavailable(item.get("current_evidence"))
            progress = display_or_unavailable(item.get("progress") or item.get("status"))
            klass = _status_class(progress)
            cards.append(
                "<article class='status-card'>"
                f"<em class='badge {klass}'>{_escape(progress)}</em>"
                f"<h3>{_escape(item.get('recommendation'))}</h3>"
                + (f"<p><strong>Emitida en:</strong> {_escape(period)}</p>" if period != "No disponible" else "")
                + f"<p><strong>Estado de seguimiento:</strong> {_escape(progress)}</p>"
                + (
                    f"<p><strong>Por qué:</strong> {_escape(display_or_unavailable(item.get('status_reason')))}</p>"
                    if display_or_unavailable(item.get("status_reason")) != "No disponible"
                    else ""
                )
                + (
                    f"<p><strong>Objetivo original:</strong> {_escape(display_or_unavailable(item.get('objective')))}</p>"
                    if display_or_unavailable(item.get("objective")) != "No disponible"
                    else ""
                )
                + (f"<p><strong>Evidencia actual:</strong> {_escape(evidence)}</p>" if evidence != "No disponible" else "")
                + (
                    f"<p><strong>Próxima acción sugerida:</strong> {_escape(display_or_unavailable(item.get('next_action')))}</p>"
                    if display_or_unavailable(item.get("next_action")) != "No disponible"
                    else ""
                )
                + "</article>"
            )
    elif kind == "risk":
        for item in items:
            severity = display_or_unavailable(item.get("severity"))
            status = display_or_unavailable(item.get("status"))
            cards.append(
                "<article class='risk-card'>"
                f"<em class='badge risk'>{_escape(display_or_unavailable(item.get('frequency')))}</em>"
                + (f"<em class='badge amber'>{_escape(status)}</em>" if status != "No disponible" else "")
                + (f"<em class='badge neutral'>{_escape(severity)}</em>" if severity != "No disponible" else "")
                + f"<h3>{_escape(item.get('risk'))}</h3>"
                + f"<p><strong>Qué pasó:</strong> {_escape(display_or_unavailable(item.get('what_happened')))}</p>"
                + f"<p><strong>Departamento:</strong> {_escape(display_or_unavailable(item.get('department')))}</p>"
                + f"<p><strong>Por qué es recurrente:</strong> {_escape(display_or_unavailable(item.get('recurrence_reason')))}</p>"
                + f"<p><strong>Estado de recurrencia:</strong> {_escape(display_or_unavailable(item.get('status')))}</p>"
                + f"<p><strong>Tendencia de recurrencia:</strong> {_escape(display_or_unavailable(item.get('recurrence_direction')))}</p>"
                + f"<p><strong>Por qué importa:</strong> {_escape(display_or_unavailable(item.get('management_relevance')))}</p>"
                + f"<p><strong>Períodos afectados:</strong> {_escape(display_or_unavailable(item.get('periods')))}</p>"
                + "</article>"
            )
    return f"<div class='recommendation-grid'>{''.join(cards)}</div>" if cards else ""


def _source_note(labels: list[str]) -> str:
    """Render compact source labels without local paths.

    Inputs: source filenames.
    Outputs: HTML source note.
    Assumptions: full source references remain in report model JSON.
    """

    if not labels:
        return ""
    return f"<p class='sources'>Fuentes: {_escape('; '.join(labels))}</p>"


def _status_class(status: str) -> str:
    """Map metric status to a CSS class.

    Inputs: normalized card status.
    Outputs: CSS class suffix.
    Assumptions: color is paired with text labels for accessibility.
    """

    normalized = str(status or "").lower()
    if normalized in {"good", "resuelto", "objetivo alcanzado"}:
        return "good"
    if normalized in {"amber", "en seguimiento", "parcialmente resuelto", "mejora parcial"}:
        return "amber"
    if normalized in {"risk", "no iniciado", "sin evidencia suficiente"}:
        return "risk"
    return "neutral"


def _narrative(view: dict[str, Any], section_id: str) -> str:
    """Render model-authored section narrative when present.

    Inputs: presentation view and section ID.
    Outputs: HTML paragraph or empty string.
    Assumptions: Step 9 generated and validated narrative in Spanish.
    """

    text = view.get("section_narratives", {}).get(section_id, "")
    return f"<p class='section-analysis'>{_escape(text)}</p>" if text else ""


def _insight_box(text: Any) -> str:
    """Render one deterministic chart conclusion box.

    Inputs: insight text from the presentation layer.
    Outputs: HTML with bold Spanish label and regular body text.
    Assumptions: insight text is deterministic and already sanitized.
    """

    body = _escape(text)
    if not body:
        return ""
    return f"<p class='executive-insight'><strong>Conclusión ejecutiva:</strong> {body}</p>"


def _bar_chart(items: list[dict[str, Any]], *, title: str) -> str:
    """Render a real SVG horizontal bar chart.

    Inputs: chart item dictionaries with label/value/unit.
    Outputs: SVG chart markup.
    Assumptions: bars visualize existing values only; no finance math occurs.
    """

    values = [abs(float(item.get("value") or 0.0)) for item in items]
    if not items or not any(values):
        return f"<div class='status-card'>{_escape(title)}: sin datos para graficar.</div>"
    width = 820
    row_height = 42
    label_width = 210
    chart_width = width - label_width - 170
    height = 72 + row_height * len(items)
    max_value = max(values) or 1.0
    rows = [
        f"<text x='0' y='20' class='chart-title'>{_escape(title)}</text>",
        f"<text x='{label_width}' y='42' class='axis-label'>Valor</text>",
    ]
    for tick in range(0, 5):
        x = label_width + chart_width * tick / 4
        rows.append(f"<line x1='{x:.1f}' y1='50' x2='{x:.1f}' y2='{height - 18}' class='grid-line' />")
        rows.append(f"<text x='{x:.1f}' y='{height - 4}' class='tick-label'>{_escape(format_compact_axis_value(max_value * tick / 4, items[0].get('unit')))}</text>")
    max_index = values.index(max(values))
    min_index = values.index(min(values))
    for index, item in enumerate(items):
        y = 58 + index * row_height
        value = float(item.get("value") or 0.0)
        bar_width = max(2.0, abs(value) / max_value * chart_width)
        color = "#1f7a5b" if value >= 0 else "#b84242"
        klass = " current-bar" if index == len(items) - 1 else ""
        marker = "Actual" if index == len(items) - 1 else ("Máx." if index == max_index else ("Mín." if index == min_index else ""))
        rows.append(f"<text x='0' y='{y + 14}' class='axis-label'>{_escape(item.get('label'))}</text>")
        rows.append(f"<rect x='{label_width}' y='{y}' width='{chart_width}' height='18' rx='5' class='track' />")
        rows.append(f"<rect class='bar{klass}' x='{label_width}' y='{y}' width='{bar_width:.1f}' height='18' rx='5' fill='{color}' />")
        rows.append(
            f"<text x='{label_width + chart_width + 12}' y='{y + 14}' class='value-label'>"
            f"{_escape(format_value(value, item.get('unit')))} {_escape(marker)}</text>"
        )
    return f"<svg class='svg-chart' viewBox='0 0 {width} {height}' role='img' aria-label='{_escape(title)}'>{''.join(rows)}</svg>"


def _line_chart(series: dict[str, Any]) -> str:
    """Render a compact SVG line chart for one historical KPI.

    Inputs: one trend series from presentation view.
    Outputs: SVG line chart markup.
    Assumptions: trend points are already sorted by historical retrieval.
    """

    points = series.get("points", [])
    if not points:
        return ""
    if len(points) < 2:
        label = _escape(series.get("metric") or "Indicador histórico")
        period = _escape(points[0].get("period_label") or format_period_label(points[0].get("period")))
        value = _escape(points[0].get("display") or format_value(points[0].get("value"), series.get("unit")))
        return (
            "<div class='trend-card status-card'>"
            f"<h4>{label}</h4>"
            "<p class='muted chart-caption'>Historial insuficiente para graficar una tendencia.</p>"
            f"<p><strong>Dato disponible:</strong> {period} — {value}</p>"
            f"{_insight_box(series.get('insight'))}"
            "</div>"
        )
    values = [float(point.get("value") or 0.0) for point in points]
    width = 600
    height = 280
    left_pad = 72
    right_pad = 28
    top_pad = 42
    bottom_pad = 64
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    coords = []
    for index, point in enumerate(points):
        x = left_pad + (width - left_pad - right_pad) * (index / max(1, len(points) - 1))
        y = height - bottom_pad - ((float(point.get("value") or 0.0) - min_value) / span) * (height - top_pad - bottom_pad)
        coords.append((x, y, point))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
    min_index = values.index(min_value)
    max_index = values.index(max_value)
    dots = "".join(
        f"<circle class='{'current-point' if index == len(coords)-1 else ('max-point' if index == max_index else ('min-point' if index == min_index else ''))}' cx='{x:.1f}' cy='{y:.1f}' r='4.2'><title>{_escape(point.get('period_label') or format_period_label(point.get('period')))}: {_escape(point.get('display'))}</title></circle>"
        for index, (x, y, point) in enumerate(coords)
    )
    grid_parts = []
    axis_labels = [
        f"<text x='{left_pad + (width-left_pad-right_pad)/2:.1f}' y='{height-8}' text-anchor='middle' class='axis-title'>Periodo</text>",
        f"<text x='16' y='{top_pad + (height-top_pad-bottom_pad)/2:.1f}' transform='rotate(-90 16 {top_pad + (height-top_pad-bottom_pad)/2:.1f})' text-anchor='middle' class='axis-title'>{_escape('Porcentaje' if series.get('unit') == 'ratio' else 'Valor')}</text>",
    ]
    for tick in range(5):
        value = min_value + span * tick / 4
        y = height - bottom_pad - (tick / 4) * (height - top_pad - bottom_pad)
        grid_parts.append(f"<line x1='{left_pad}' x2='{width-right_pad}' y1='{y:.1f}' y2='{y:.1f}' class='grid-line' />")
        axis_labels.append(f"<text x='{left_pad-8}' y='{y+4:.1f}' text-anchor='end' class='tick-label'>{_escape(format_value(value, series.get('unit')))}</text>")
    for x, _, point in coords:
        axis_labels.append(
            f"<text x='{x:.1f}' y='{height-bottom_pad+20}' text-anchor='middle' class='tick-label x-period'>{_escape(point.get('period_label') or format_period_label(point.get('period')))}</text>"
        )
    grid = "".join(grid_parts)
    return (
        "<div class='trend-card'>"
        f"<h4>{_escape(series.get('metric'))}</h4>"
        f"<svg viewBox='0 0 {width} {height}' class='line-chart'>"
        f"{grid}<line x1='{left_pad}' y1='{height-bottom_pad}' x2='{width-right_pad}' y2='{height-bottom_pad}' class='axis-line' />"
        f"<line x1='{left_pad}' y1='{top_pad}' x2='{left_pad}' y2='{height-bottom_pad}' class='axis-line' />"
        f"<polyline points='{polyline}' />{dots}{''.join(axis_labels)}</svg>"
        f"<p class='muted chart-caption'>Dirección: {_escape(series.get('direction_label') or series.get('direction'))}</p>"
        f"{_insight_box(series.get('insight'))}"
        "</div>"
    )


def _render_cover(view: dict[str, Any]) -> str:
    """Render the executive cover.

    Inputs: presentation view.
    Outputs: cover section HTML.
    Assumptions: organization can be replaced by future configuration.
    """

    return (
        "<section class='cover' id='cover'>"
        "<div class='cover-mark'>Finance AI Agent</div>"
        f"<h1>{_escape(view['title'])}</h1>"
        f"<p class='period'>Periodo: {_escape(view.get('period'))}</p>"
        f"<p>{_escape(view.get('organization'))}</p>"
        "<p class='cover-note'>Síntesis ejecutiva generada desde salidas procesadas, validadas y trazables.</p>"
        "</section>"
    )


def _render_summary(view: dict[str, Any]) -> str:
    """Render executive summary, findings, and root causes.

    Inputs: presentation view.
    Outputs: HTML section.
    Assumptions: text was localized by the presentation layer.
    """

    summary = view["executive_summary"]
    findings = "".join(f"<li>{_escape(item)}</li>" for item in summary["key_findings"])
    roots = "".join(f"<li>{_escape(item)}</li>" for item in summary["root_causes"])
    return (
        "<section id='executive_summary'>"
        f"<h2>{SECTION_LABELS_ES['executive_summary']}</h2>"
        f"<p class='lead'>{_escape(summary['summary'])}</p>"
        "<div class='two-col'>"
        f"<div><h3>Hallazgos clave</h3><ul>{findings or '<li>Sin hallazgos materiales.</li>'}</ul></div>"
        f"<div><h3>Causas raíz probables</h3><ul>{roots or '<li>Sin causas raíz materiales.</li>'}</ul></div>"
        "</div>"
        f"<p class='confidence'>Confianza del análisis: {_escape(summary['confidence'])}</p>"
        "</section>"
    )


def _render_health(view: dict[str, Any]) -> str:
    """Render financial health dashboard.

    Inputs: presentation view.
    Outputs: HTML section with KPI cards and chart.
    Assumptions: card values come from finance summary outputs.
    """

    card_html: list[str] = []
    missing_comparison_count = 0
    for card in view["financial_health"]["cards"]:
        klass = _status_class(card.get("status", "neutral"))
        comparison_rows = card.get("comparison_rows", [])
        missing_comparison_count += max(0, 4 - len(comparison_rows))
        comparison_html = "".join(
            f"<small><span>{_escape(row.get('label'))}:</span> {_escape(row.get('value'))}</small>"
            for row in comparison_rows
            if row.get("value")
        )
        card_html.append(
            f"<article class='kpi-card {klass}'>"
            f"<div class='card-head'><span>{_escape(card['label'])}</span>"
            f"<em class='badge {klass}'>{_escape(card.get('badge', {}).get('icon', 'i'))} "
            f"{_escape(card.get('badge', {}).get('label', 'Info'))}</em></div>"
            f"<strong>{_escape(card['trend_arrow'])} {_escape(card['value'])}</strong>"
            + (f"<div class='card-deltas'>{comparison_html}</div>" if comparison_html else "")
            + f"<small>{_escape(card['description'])}</small>"
            "</article>"
        )
    cards = "".join(card_html)
    chart_items = [
        {"label": card["label"], "value": card["numeric_value"] or 0.0, "unit": card["unit"]}
        for card in view["financial_health"]["cards"]
        if card["id"] in {"total_revenue", "total_expenses", "net_operating_result", "net_cash_flow"}
    ]
    return (
        "<section id='financial_health_overview'>"
        f"<h2>{SECTION_LABELS_ES['financial_health_overview']}</h2>"
        + _narrative(view, "financial_health_overview")
        + f"<div class='kpi-grid'>{cards}</div>"
        + (
            _info_card(
                "Comparaciones no disponibles",
                "Algunas metas o presupuestos no existen en los artefactos procesados para estos KPIs; por eso se omiten en las tarjetas.",
            )
            if missing_comparison_count >= len(view["financial_health"]["cards"])
            else ""
        )
        + _bar_chart(chart_items, title="Resumen financiero principal")
        + _insight_box(view["financial_health"].get("chart_insight"))
        + _source_note(view["financial_health"]["sources"])
        + "</section>"
    )


def _render_kpis(view: dict[str, Any]) -> str:
    """Render KPI and goal compliance section.

    Inputs: presentation view.
    Outputs: HTML KPI table.
    Assumptions: no KPI calculations are performed here.
    """

    mini_cards: list[str] = []
    for item in view["kpis"]:
        klass = item.get("badge", {}).get("class", "neutral")
        mini_cards.append(
            "<article class='mini-card'>"
            f"<span>{_escape(item['indicator'])}</span>"
            f"<strong>{_escape(item['value'])}</strong>"
            f"<em class='badge {klass}'>{_escape(item.get('badge', {}).get('label', item['status']))}</em>"
            f"<small>{_escape(item['description'])}</small>"
            "</article>"
        )
    cards = "".join(mini_cards)
    rows = [[item["indicator"], item["value"], item["status"], item["description"]] for item in view["kpis"]]
    detail = ""
    if not view["kpis"]:
        detail = _info_card("KPIs", "No hay KPIs disponibles para este periodo.", klass="positive")
    elif len(view["kpis"]) > 6:
        detail = _table(["Indicador", "Valor", "Estado", "Descripción"], rows, force=True)
    return (
        "<section id='kpi_overview'>"
        f"<h2>{SECTION_LABELS_ES['kpi_overview']}</h2>"
        + _narrative(view, "kpi_overview")
        + (f"<div class='mini-grid'>{cards}</div>" if cards else "")
        + detail
        + "</section>"
    )


def _render_historical(view: dict[str, Any]) -> str:
    """Render historical trends and longitudinal sections.

    Inputs: presentation view.
    Outputs: HTML sections or empty string.
    Assumptions: empty history is omitted gracefully.
    """

    historical = view["historical"]
    if not historical.get("available"):
        return ""
    chartable = historical_chart_series(historical)
    chartable_ids = {id(series) for series in chartable}
    charts = "".join(_line_chart(series) for series in chartable)
    fallback_cards = "".join(
        _line_chart(series)
        for series in historical.get("trends", [])
        if isinstance(series, dict) and id(series) not in chartable_ids
    )
    validate_historical_chart_rendering(
        historical,
        charts.count("line-chart"),
        renderer_name="HTML renderer",
    )
    risks = [
        [row["risk"], row["department"], row.get("frequency", row.get("occurrences", "")), row.get("status", ""), row["periods"]]
        for row in historical.get("recurring_risks", [])
    ]
    follow = [
        [row["recommendation"], row["issued_period"], row.get("progress", row.get("status", "")), row.get("status_reason", ""), row.get("objective", ""), row["current_evidence"]]
        for row in historical.get("recommendation_follow_up", [])
    ]
    follow_markup = (
        _summary_cards(historical.get("recommendation_follow_up", []), kind="follow_up")
        if 0 < len(follow) <= 5
        else _table(
            ["Recomendación", "Emitida en", "Estado de seguimiento", "Por qué", "Objetivo original", "Evidencia actual"],
            follow,
            low_value_title="Seguimiento disponible",
            low_value_message="La evidencia histórica no contiene suficiente detalle para una tabla de seguimiento.",
        )
    )
    risk_markup = (
        _summary_cards(historical.get("recurring_risks", []), kind="risk")
        if 0 < len(risks) <= 3
        else _table(
            ["Riesgo", "Departamento", "Frecuencia", "Estado de recurrencia", "Períodos afectados"],
            risks,
            low_value_title="Riesgos longitudinales",
            low_value_message="No hay riesgos recurrentes con suficiente detalle tabular.",
        )
    )
    return (
        "<section id='historical_trends'><span id='historical_summary'></span>"
        f"<h2>{SECTION_LABELS_ES['historical_trends']}</h2>"
        + _narrative(view, "historical_summary")
        + _narrative(view, "historical_trends")
        + f"<div class='trend-grid'>{charts}{fallback_cards}</div>"
        "</section>"
        "<section id='recommendation_follow_up'>"
        f"<h2>{SECTION_LABELS_ES['recommendation_follow_up']}</h2>"
        + (f"<p class='section-analysis'>{_escape(historical.get('recommendation_intro'))}</p>" if historical.get("recommendation_intro") else "")
        + (f"<div class='info-card neutral'><p>{_escape(historical.get('recommendation_summary'))}</p></div>" if historical.get("recommendation_summary") else "")
        + follow_markup
        + "</section>"
        "<section id='longitudinal_risk_assessment'>"
        f"<h2>{SECTION_LABELS_ES['longitudinal_risk_assessment']}</h2>"
        + (f"<div class='info-card neutral'><p>{_escape(historical.get('risk_summary'))}</p></div>" if historical.get("risk_summary") else "")
        + risk_markup
        + "</section>"
    )


def _render_revenue_expense(view: dict[str, Any]) -> str:
    """Render revenue and expense comparison.

    Inputs: presentation view.
    Outputs: HTML section.
    Assumptions: values are from processed summaries.
    """

    data = view["revenue_expense"]
    rows = [[row["metric"], row["value"], row["description"]] for row in data["rows"]]
    return (
        "<section id='revenue_expense_analysis'><span id='revenue_analysis'></span><span id='expense_analysis'></span>"
        f"<h2>{SECTION_LABELS_ES['revenue_expense_analysis']}</h2>"
        + _narrative(view, "revenue_expense_analysis")
        + _bar_chart(data["chart"], title="Ingresos, gastos y resultado")
        + _insight_box(data.get("chart_insight"))
        + _bar_chart(data["budget_chart"], title="Comparación contra presupuesto")
        + _insight_box(data.get("budget_chart_insight"))
        + _table(["Métrica", "Valor", "Descripción"], rows)
        + "</section>"
    )


def _render_departments(view: dict[str, Any]) -> str:
    """Render department analysis.

    Inputs: presentation view.
    Outputs: HTML section.
    Assumptions: department rows are pre-aggregated upstream.
    """

    rows = [
        [item["department"], item["revenue"], item["expenses"], item["result"], item["variance"]]
        for item in view["departments"]
    ]
    chart = [
        {"label": item["department"], "value": item["numeric_result"], "unit": "USD"}
        for item in view["departments"]
    ]
    insight = ""
    if chart:
        insight = deterministic_chart_insight(chart, title="resultado operativo por departamento", chart_kind="department", unit="USD")
    department_cards: list[str] = []
    for item in view["departments"]:
        klass = item.get("variance_class", "neutral")
        department_cards.append(
            f"<article class='department-card {klass}'>"
            f"<span>{_escape(item['department'])}</span>"
            f"<strong>{_escape(item['result'])}</strong>"
            f"<em class='badge {klass}'>{_escape(item.get('rank_badge') or 'Monitoreo')}</em>"
            f"<small>Gastos: {_escape(item['expenses'])} · Variación de gasto: {_escape(display_or_unavailable(item['variance']))}</small>"
            "</article>"
        )
    cards = "".join(department_cards)
    return (
        "<section id='department_analysis'>"
        f"<h2>{SECTION_LABELS_ES['department_analysis']}</h2>"
        + _narrative(view, "department_analysis")
        + f"<div class='mini-grid'>{cards}</div>"
        + _bar_chart(chart, title="Resultado operativo por departamento")
        + _insight_box(insight)
        + _table(["Departamento", "Ingresos", "Gastos", "Resultado", "Var. gasto"], rows)
        + "</section>"
    )


def _render_anomalies(view: dict[str, Any]) -> str:
    """Render anomalies with positive empty state.

    Inputs: presentation view.
    Outputs: HTML anomaly section.
    Assumptions: no anomaly logic is executed here.
    """

    anomalies = view["anomalies"]
    severity_rows = [[row["severity"], row["count"]] for row in anomalies["severity_rows"]]
    top_rows = [
        [
            row["title"],
            row["severity"],
            row.get("metric"),
            row.get("observed_value"),
            row.get("reference_value"),
            row["evidence"],
        ]
        for row in anomalies["top_rows"]
    ]
    severity_chart = anomalies.get("severity_chart", [])
    risk_card_items: list[str] = []
    for row in anomalies["top_rows"]:
        klass = row.get("severity_class", "info")
        chips = "".join(f"<span>{_escape(chip)}</span>" for chip in row.get("period_chips", []))
        meta = "".join(
            f"<p><strong>{_escape(label)}:</strong> {_escape(value)}</p>"
            for label, value in (
                ("Indicador afectado", row.get("metric")),
                ("Entidad/departamento", row.get("entity")),
                ("Valor observado", row.get("observed_value")),
                ("Referencia", row.get("reference_value")),
                ("Periodo", row.get("period")),
                ("Próxima verificación", row.get("recommended_next_check")),
            )
            if value
        )
        risk_card_items.append(
            "<article class='risk-card'>"
            f"<div><em class='badge {klass}'>{_escape(row['severity'])}</em> "
            f"<em class='badge neutral'>{_escape(row.get('recurrence'))}</em></div>"
            f"<h3>{_escape(row['title'])}</h3>"
            f"<p>{_escape(row['evidence'])}</p>"
            f"{meta}"
            f"<div class='chips'>{chips}</div>"
            "</article>"
        )
    risk_cards = "".join(risk_card_items)
    positive = anomalies.get("positive_status")
    current_status = anomalies.get("current_period_status") or positive
    if current_status:
        return (
            "<section id='anomaly_summary'>"
            f"<h2>{SECTION_LABELS_ES['anomaly_summary']}</h2>"
            + f"<div class='status-card positive'>{_escape(current_status)}</div>"
            + (
                f"<p class='section-analysis'>{_escape(anomalies.get('distinction_note'))}</p>"
                if anomalies.get("distinction_note")
                else ""
            )
            + "</section>"
        )
    detail_markup = ""
    if len(top_rows) > 3:
        detail_markup = _table(
            ["Anomalía", "Severidad", "Indicador", "Valor observado", "Referencia", "Evidencia"],
            top_rows,
            force=True,
        )
    return (
        "<section id='anomaly_summary'>"
        f"<h2>{SECTION_LABELS_ES['anomaly_summary']}</h2>"
        + _narrative(view, "anomaly_summary")
        + _bar_chart(severity_chart, title="Anomalías por severidad")
        + _insight_box(anomalies.get("chart_insight"))
        + f"<div class='recommendation-grid'>{risk_cards}</div>"
        + (_table(["Severidad", "Cantidad"], severity_rows, force=True) if len(severity_rows) > 4 else "")
        + detail_markup
        + "</section>"
    )


def _render_evidence(view: dict[str, Any]) -> str:
    """Render concise investigation evidence for executive readers.

    Inputs: presentation view.
    Outputs: HTML evidence section.
    Assumptions: internal task IDs and tool names are intentionally hidden.
    """

    rows = [
        [item["priority"], item["evidence"], item["records"], item["summary"]]
        for item in view["evidence"]
    ]
    if not view["evidence"]:
        content = _info_card("Evidencia", "No se solicitó evidencia adicional para este periodo.", klass="positive")
    elif len(view["evidence"]) <= 3:
        content = _summary_cards(view["evidence"], kind="evidence")
    else:
        content = _table(
            ["Prioridad", "Evidencia", "Registros", "Resumen"],
            rows,
            low_value_title="Evidencia procesada",
            low_value_message="La evidencia recuperada está disponible, pero no contiene suficiente detalle resumible para una tabla.",
        )
    return (
        "<section id='investigation_evidence'>"
        f"<h2>{SECTION_LABELS_ES['investigation_evidence']}</h2>"
        + content
        + "</section>"
    )


def _render_recommendations(view: dict[str, Any]) -> str:
    """Render strategic priorities and recommendation cards.

    Inputs: presentation view.
    Outputs: HTML recommendation section.
    Assumptions: recommendations were validated upstream.
    """

    recs = view["recommendations"]
    priorities = "".join(f"<li>{_escape(item)}</li>" for item in recs["priorities"])
    cards = "".join(
        "<article class='recommendation-card'>"
        f"<div class='badge {card['priority'].lower()}'>{_escape(card['priority'])}</div>"
        f"<h3>{_escape(card['action'])}</h3>"
        f"<p><strong>Racional:</strong> {_escape(card['rationale'])}</p>"
        f"<p><strong>Impacto esperado:</strong> {_escape(card['expected_impact'])}</p>"
        f"<p><strong>Responsable sugerido:</strong> {_escape(card.get('owner'))}</p>"
        f"<p><strong>Estado:</strong> {_escape(card.get('status'))}</p>"
        "</article>"
        for card in recs["cards"]
    )
    rows = [
        [card["priority"], card["action"], card["expected_impact"], card.get("owner"), card.get("status")]
        for card in recs["cards"]
    ]
    recommendation_display = (
        f"<div class='recommendation-grid'>{cards}</div>"
        if len(recs["cards"]) <= 5
        else _table(["Prioridad", "Acción", "Impacto esperado", "Responsable", "Estado"], rows, force=True)
    )
    if not recs["cards"]:
        attention_cards = "".join(
            "<article class='risk-card'>"
            f"<div><em class='badge warning'>{_escape(item.get('severity') or 'Atención')}</em></div>"
            f"<h3>{_escape(item.get('title') or 'Hallazgo determinístico')}</h3>"
            f"<p>{_escape(item.get('evidence') or '')}</p>"
            f"<p><strong>Indicador:</strong> {_escape(item.get('metric') or '')}</p>"
            f"<p><strong>Departamento/entidad:</strong> {_escape(item.get('department') or '')}</p>"
            f"<p><strong>Periodo:</strong> {_escape(item.get('period') or '')}</p>"
            "</article>"
            for item in recs.get("attention_items", [])[:6]
        )
        recommendation_display = _info_card(
            "Recomendaciones estratégicas",
            recs.get("strategy_unavailable_note")
            or "No hay recomendaciones estratégicas validadas para este período. El reporte conserva los hallazgos determinísticos, KPIs, anomalías, historial y evidencia procesada.",
            klass="warning",
        ) + (
            f"<h3>Hallazgos determinísticos que requieren atención</h3><div class='recommendation-grid'>{attention_cards}</div>"
            if attention_cards
            else ""
        )
    return (
        "<section id='strategic_recommendations'>"
        f"<h2>{SECTION_LABELS_ES['strategic_recommendations']}</h2>"
        + _narrative(view, "strategic_recommendations")
        + (f"<h3>Prioridades estratégicas</h3><ul>{priorities}</ul>" if priorities else "")
        + recommendation_display
        + "</section>"
    )


def _render_missing_and_appendix(view: dict[str, Any]) -> str:
    """Render missing information and appendix methodology.

    Inputs: presentation view.
    Outputs: HTML sections.
    Assumptions: appendix source labels are filenames, not local paths.
    """

    missing = "".join(f"<li>{_escape(item)}</li>" for item in view["missing_information"])
    appendix = view["appendix"]
    methodology = "".join(f"<li>{_escape(item)}</li>" for item in appendix["methodology"])
    sources = "".join(f"<li>{_escape(item)}</li>" for item in appendix["sources"])
    return (
        "<section id='missing_information'>"
        f"<h2>{SECTION_LABELS_ES['missing_information']}</h2>"
        + _narrative(view, "missing_information")
        + (f"<ul>{missing}</ul>" if missing else _info_card("Estado de información", "No se reportan brechas de información relevantes.", klass="positive"))
        + "</section>"
        + "<section id='appendix'>"
        f"<h2>{SECTION_LABELS_ES['appendix']}</h2>"
        f"<h3>Metodología</h3><ul>{methodology}</ul>"
        f"<h3>Fuentes:</h3><ul>{sources}</ul>"
        "</section>"
    )


def report_strategy_warnings(report_model: dict[str, Any]) -> list[str]:
    """Identify whether a report model lacks full accepted strategic analysis.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: warning strings; empty when final strategy is present.
    Assumptions: deterministic reports remain renderable without full strategy.
    """

    executive = get_section(report_model, "executive_summary").get("content", {})
    recommendations = get_section(report_model, "strategic_recommendations").get("content", {})
    warnings: list[str] = []
    if executive.get("analysis_status") not in {"accepted", "sanitized"}:
        warnings.append(f"Strategic analysis is unavailable or not accepted for {report_model.get('report_id', 'report')}.")
    if not recommendations.get("recommendations"):
        warnings.append("No accepted strategic recommendations are present; deterministic report sections remain available.")
    return warnings


def validate_strategy_available(report_model: dict[str, Any]) -> None:
    """Validate the report has enough deterministic content to render.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: None; raises ValueError only when deterministic executive content is missing.
    Assumptions: strategic enrichment is optional and should not block rendering.
    """

    executive = get_section(report_model, "executive_summary").get("content", {})
    if not str(executive.get("summary") or "").strip():
        raise ValueError("Executive summary is missing.")


def _styles() -> str:
    """Return embedded CSS for a responsive printable executive report.

    Inputs: none.
    Outputs: CSS string.
    Assumptions: the HTML report is self-contained for downloads.
    """

    return """
    :root { --navy:#17324d; --blue:#245b89; --green:#1f7a5b; --red:#b84242; --amber:#b7791f; --ink:#172033; --muted:#647084; --line:#d8e1ea; --bg:#f4f7fb; --soft:#fbfdff; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Arial, Helvetica, sans-serif; color:var(--ink); background:var(--bg); line-height:1.52; }
    main { max-width: 1180px; margin:0 auto; padding:38px 32px; }
    section { background:#fff; margin:42px 0; padding:38px; border-radius:20px; box-shadow:0 8px 28px rgba(23,50,77,.08); break-inside: avoid; page-break-inside: avoid; }
    .cover { min-height: 360px; display:flex; flex-direction:column; justify-content:center; color:#fff; background:linear-gradient(135deg,var(--navy),var(--blue)); }
    .cover-mark { letter-spacing:.16em; text-transform:uppercase; font-size:13px; opacity:.82; }
    h1 { margin:.35em 0; font-size:44px; line-height:1.05; }
    h2 { margin:0 0 24px; color:var(--navy); font-size:25px; border-left:6px solid var(--blue); padding-left:12px; break-after: avoid; page-break-after: avoid; }
    h3 { color:var(--navy); margin:22px 0 12px; }
    .cover h1, .cover h2 { color:#fff; border:0; padding:0; }
    .period { font-size:22px; font-weight:700; }
    .cover-note, .lead { font-size:18px; max-width:850px; }
    .section-analysis { font-size:15px; color:#263244; background:var(--soft); border-left:4px solid var(--blue); padding:15px 17px; border-radius:12px; margin:14px 0 24px; }
    .two-col { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; }
    .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(235px,1fr)); gap:22px; margin:22px 0 34px; align-items:stretch; }
    .mini-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:18px; margin:22px 0 32px; align-items:stretch; }
    .kpi-card, .mini-card, .department-card { border:1px solid var(--line); border-radius:18px; padding:24px; min-height:160px; background:linear-gradient(180deg,#fff,var(--soft)); box-shadow:0 8px 18px rgba(23,50,77,.06); }
    .card-head { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .kpi-card span { display:block; color:var(--muted); font-size:13px; }
    .kpi-card strong { display:block; font-size:30px; line-height:1.12; margin:11px 0 12px; color:var(--navy); letter-spacing:-.02em; }
    .card-deltas { display:grid; grid-template-columns:1fr; gap:5px; margin:9px 0 12px; color:var(--muted); }
    .mini-card strong, .department-card strong { display:block; color:var(--navy); font-size:21px; margin:5px 0; }
    .kpi-card.good { border-top:5px solid var(--green); }
    .kpi-card.amber { border-top:5px solid var(--amber); }
    .kpi-card.risk { border-top:5px solid var(--red); }
    .kpi-card.neutral { border-top:5px solid var(--blue); }
    .department-card.good { border-left:5px solid var(--green); }
    .department-card.risk { border-left:5px solid var(--red); }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:separate; border-spacing:0; margin:28px 0 14px; font-size:13px; border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    th { background:#eef4fb; color:var(--navy); text-align:left; }
    th, td { border-bottom:1px solid var(--line); padding:13px 14px; vertical-align:middle; }
    th:not(:last-child), td:not(:last-child) { border-right:1px solid var(--line); }
    tbody tr:last-child td { border-bottom:0; }
    tr:nth-child(even) td { background:#fbfdff; }
    .svg-chart, .line-chart { width:100%; height:auto; margin:26px 0 12px; background:var(--soft); border:1px solid var(--line); border-radius:16px; padding:14px; }
    .chart-title { font-weight:700; fill:var(--navy); font-size:16px; }
    .axis-label, .value-label { fill:#263244; font-size:12px; }
    .axis-title { fill:var(--navy); font-size:11px; font-weight:700; }
    .tick-label { fill:var(--muted); font-size:10.5px; }
    .grid-line { stroke:#dfe8f2; stroke-width:1; }
    .axis-line { stroke:#9fb2c7; stroke-width:1.1; }
    .track { fill:#e8eef5; }
    .current-bar { stroke:var(--navy); stroke-width:1.5; }
    .line-chart polyline { fill:none; stroke:var(--blue); stroke-width:3; }
    .line-chart circle { fill:var(--green); stroke:#fff; stroke-width:1; }
    .line-chart .current-point { fill:var(--navy); r:5; }
    .line-chart .max-point { fill:var(--green); }
    .line-chart .min-point { fill:var(--red); }
    .trend-grid, .recommendation-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:22px; margin:22px 0 30px; align-items:start; }
    .trend-card, .recommendation-card, .risk-card, .status-card { border:1px solid var(--line); border-radius:18px; padding:24px; background:var(--soft); box-shadow:0 6px 16px rgba(23,50,77,.05); break-inside: avoid; page-break-inside: avoid; }
    .info-panel strong { display:block; color:var(--navy); margin-bottom:6px; }
    .info-panel p { margin:0; color:var(--muted); }
    .status-card.positive { border-left:5px solid var(--green); }
    .badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#e9f2fb; color:var(--navy); font-weight:700; font-size:12px; font-style:normal; }
    .badge.good, .badge.baja { background:#e8f6ef; color:var(--green); }
    .badge.amber, .badge.media { background:#fff3d6; color:var(--amber); }
    .badge.risk, .badge.alta, .badge.crítica, .badge.critical, .badge.high { background:#fdebea; color:var(--red); }
    .chips span { display:inline-block; margin:2px 4px 2px 0; padding:3px 8px; border-radius:999px; background:#eef4fb; color:var(--navy); font-size:11px; }
    .executive-insight { margin:14px 0 28px; padding:16px 18px; border-radius:14px; background:#eef7ff; border-left:5px solid var(--blue); color:#24364a; font-size:13.5px; line-height:1.55; }
    .executive-insight strong { color:var(--navy); font-weight:800; }
    .chart-caption { margin:10px 0 8px; }
    .muted, .sources, .confidence { color:var(--muted); font-size:12px; }
    footer { text-align:center; color:var(--muted); font-size:12px; padding:26px; }
    @media (max-width:760px) { main { padding:14px; } .two-col { grid-template-columns:1fr; } h1 { font-size:34px; } section { padding:20px; } }
    @media print { body { background:#fff; } main { padding:0; } section { box-shadow:none; border-radius:0; page-break-inside:avoid; } .cover { page-break-after:always; } }
    """


def render_report_html(report_model: dict[str, Any], *, mode: str = "executive") -> str:
    """Render a report model to a complete Spanish HTML document.

    Inputs: renderer-agnostic report model dictionary and rendering mode.
    Outputs: complete HTML string.
    Assumptions: presentation transformation handles localization and sanitizing.
    """

    view = build_presentation_view(report_model, mode=mode)
    body = [
        _render_cover(view),
        _render_summary(view),
        _render_health(view),
        _render_kpis(view),
        _render_historical(view),
        _render_revenue_expense(view),
        _render_departments(view),
        _render_anomalies(view),
        _render_evidence(view),
        _render_recommendations(view),
        _render_missing_and_appendix(view),
    ]
    return (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_escape(view.get('title'))} - {_escape(view.get('period'))}</title>"
        f"<style>{_styles()}</style></head><body><main>{''.join(body)}</main>"
        "<footer>Finance AI Agent · Reporte ejecutivo generado desde datos procesados y validados.</footer>"
        "</body></html>"
    )


def load_report_model(path: str | Path) -> dict[str, Any]:
    """Load a report model JSON file.

    Inputs: report model path.
    Outputs: parsed report model dictionary.
    Assumptions: root must be a JSON object.
    """

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Report model root must be an object: {path}")
    return value


def save_report_html(report_model: dict[str, Any], output_path: str | Path, *, mode: str = "executive") -> Path:
    """Render and save a report model as HTML.

    Inputs: report model dictionary, output path, and rendering mode.
    Outputs: resolved written path.
    Assumptions: parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_html(report_model, mode=mode), encoding="utf-8")
    return path

