"""Professional HTML renderer for Finance AI Agent report models."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from finance_agent.reporting.presentation import (
    SECTION_LABELS_ES,
    build_presentation_view,
    format_value,
    get_section,
    number_value,
)


def _escape(value: Any) -> str:
    """Escape a value for safe HTML output.

    Inputs: any scalar value.
    Outputs: HTML-escaped string.
    Assumptions: report content is plain text, not trusted markup.
    """

    return html.escape("" if value is None else str(value))


def _table(headers: list[str], rows: list[list[Any]], *, empty: str = "Sin datos disponibles.") -> str:
    """Render a responsive HTML table.

    Inputs: headers, rows, and empty-state text.
    Outputs: HTML table markup.
    Assumptions: values are display-ready and escaped here.
    """

    if not rows:
        return f"<div class='status-card positive'>{_escape(empty)}</div>"
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


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

    return {"good": "good", "risk": "risk"}.get(status, "neutral")


def _narrative(view: dict[str, Any], section_id: str) -> str:
    """Render model-authored section narrative when present.

    Inputs: presentation view and section ID.
    Outputs: HTML paragraph or empty string.
    Assumptions: Step 9 generated and validated narrative in Spanish.
    """

    text = view.get("section_narratives", {}).get(section_id, "")
    return f"<p class='section-analysis'>{_escape(text)}</p>" if text else ""


def _bar_chart(items: list[dict[str, Any]], *, title: str) -> str:
    """Render a real SVG horizontal bar chart.

    Inputs: chart item dictionaries with label/value/unit.
    Outputs: SVG chart markup.
    Assumptions: bars visualize existing values only; no finance math occurs.
    """

    values = [abs(float(item.get("value") or 0.0)) for item in items]
    if not items or not any(values):
        return f"<div class='status-card'>{_escape(title)}: sin datos para graficar.</div>"
    width = 720
    row_height = 34
    label_width = 180
    chart_width = width - label_width - 150
    height = 42 + row_height * len(items)
    max_value = max(values) or 1.0
    rows = [
        f"<text x='0' y='20' class='chart-title'>{_escape(title)}</text>"
    ]
    for index, item in enumerate(items):
        y = 42 + index * row_height
        value = float(item.get("value") or 0.0)
        bar_width = max(2.0, abs(value) / max_value * chart_width)
        color = "#1f7a5b" if value >= 0 else "#b84242"
        rows.append(f"<text x='0' y='{y + 14}' class='axis-label'>{_escape(item.get('label'))}</text>")
        rows.append(f"<rect x='{label_width}' y='{y}' width='{bar_width:.1f}' height='18' rx='5' fill='{color}' />")
        rows.append(
            f"<text x='{label_width + chart_width + 12}' y='{y + 14}' class='value-label'>"
            f"{_escape(format_value(value, item.get('unit')))}</text>"
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
    values = [float(point.get("value") or 0.0) for point in points]
    width = 420
    height = 180
    pad = 34
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    coords = []
    for index, point in enumerate(points):
        x = pad + (width - pad * 2) * (index / max(1, len(points) - 1))
        y = height - pad - ((float(point.get("value") or 0.0) - min_value) / span) * (height - pad * 2)
        coords.append((x, y, point))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5'><title>{_escape(point.get('period'))}: {_escape(point.get('display'))}</title></circle>"
        for x, y, point in coords
    )
    return (
        "<div class='trend-card'>"
        f"<h4>{_escape(series.get('metric'))}</h4>"
        f"<svg viewBox='0 0 {width} {height}' class='line-chart'>"
        f"<polyline points='{polyline}' />{dots}</svg>"
        f"<p class='muted'>DirecciÃ³n: {_escape(series.get('direction'))}</p>"
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
        "<p class='cover-note'>SÃ­ntesis ejecutiva generada desde salidas procesadas, validadas y trazables.</p>"
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
        f"<div><h3>Causas raÃ­z probables</h3><ul>{roots or '<li>Sin causas raÃ­z materiales.</li>'}</ul></div>"
        "</div>"
        f"<p class='confidence'>Confianza del anÃ¡lisis: {_escape(summary['confidence'])}</p>"
        "</section>"
    )


def _render_health(view: dict[str, Any]) -> str:
    """Render financial health dashboard.

    Inputs: presentation view.
    Outputs: HTML section with KPI cards and chart.
    Assumptions: card values come from finance summary outputs.
    """

    cards = "".join(
        "<article class='kpi-card {klass}'>"
        f"<span>{_escape(card['label'])}</span>"
        f"<strong>{_escape(card['value'])}</strong>"
        f"<small>{_escape(card['description'])}</small>"
        "</article>".format(klass=_status_class(card.get("status", "neutral")))
        for card in view["financial_health"]["cards"]
    )
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
        + _bar_chart(chart_items, title="Resumen financiero principal")
        + _source_note(view["financial_health"]["sources"])
        + "</section>"
    )


def _render_kpis(view: dict[str, Any]) -> str:
    """Render KPI and goal compliance section.

    Inputs: presentation view.
    Outputs: HTML KPI table.
    Assumptions: no KPI calculations are performed here.
    """

    rows = [
        [item["indicator"], item["value"], item["status"], item["description"]]
        for item in view["kpis"]
    ]
    return (
        "<section id='kpi_overview'>"
        f"<h2>{SECTION_LABELS_ES['kpi_overview']}</h2>"
        + _narrative(view, "kpi_overview")
        + _table(["Indicador", "Valor", "Estado", "DescripciÃ³n"], rows)
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
    charts = "".join(_line_chart(series) for series in historical.get("trends", []))
    risks = [
        [row["risk"], row["department"], row["occurrences"], row["periods"]]
        for row in historical.get("recurring_risks", [])
    ]
    follow = [
        [row["recommendation"], row["issued_period"], row["current_evidence"], row["status"]]
        for row in historical.get("recommendation_follow_up", [])
    ]
    return (
        "<section id='historical_trends'><span id='historical_summary'></span>"
        f"<h2>{SECTION_LABELS_ES['historical_trends']}</h2>"
        + _narrative(view, "historical_summary")
        + _narrative(view, "historical_trends")
        + f"<div class='trend-grid'>{charts}</div>"
        "</section>"
        "<section id='recommendation_follow_up'>"
        f"<h2>{SECTION_LABELS_ES['recommendation_follow_up']}</h2>"
        + _narrative(view, "recommendation_follow_up")
        + _table(["RecomendaciÃ³n", "Periodo", "Evidencia actual", "Estado inferido"], follow)
        + "</section>"
        "<section id='longitudinal_risk_assessment'>"
        f"<h2>EvaluaciÃ³n longitudinal de riesgos</h2>"
        + _narrative(view, "longitudinal_risk_assessment")
        + _table(["Riesgo", "Departamento", "Ocurrencias", "Periodos"], risks)
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
        + _bar_chart(data["budget_chart"], title="ComparaciÃ³n contra presupuesto")
        + _table(["MÃ©trica", "Valor", "DescripciÃ³n"], rows)
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
    return (
        "<section id='department_analysis'>"
        f"<h2>{SECTION_LABELS_ES['department_analysis']}</h2>"
        + _narrative(view, "department_analysis")
        + _bar_chart(chart, title="Resultado operativo por departamento")
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
    top_rows = [[row["title"], row["severity"], row["evidence"]] for row in anomalies["top_rows"]]
    positive = anomalies.get("positive_status")
    if positive:
        return (
            "<section id='anomaly_summary'>"
            f"<h2>{SECTION_LABELS_ES['anomaly_summary']}</h2>"
            + _narrative(view, "anomaly_summary")
            + f"<div class='status-card positive'>{_escape(positive)}</div>"
            "</section>"
        )
    return (
        "<section id='anomaly_summary'>"
        f"<h2>{SECTION_LABELS_ES['anomaly_summary']}</h2>"
        + _narrative(view, "anomaly_summary")
        + _table(["Severidad", "Cantidad"], severity_rows)
        + _table(["AnomalÃ­a", "Severidad", "Evidencia"], top_rows)
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
    return (
        "<section id='investigation_evidence'>"
        f"<h2>{SECTION_LABELS_ES['investigation_evidence']}</h2>"
        + _table(["Prioridad", "Evidencia", "Registros", "Resumen"], rows)
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
        f"<div class='badge'>{_escape(card['priority'])}</div>"
        f"<h3>{_escape(card['action'])}</h3>"
        f"<p><strong>Racional:</strong> {_escape(card['rationale'])}</p>"
        f"<p><strong>Impacto esperado:</strong> {_escape(card['expected_impact'])}</p>"
        f"<p><strong>Responsable/estado:</strong> {_escape(card['owner_status'])}</p>"
        "</article>"
        for card in recs["cards"]
    )
    return (
        "<section id='strategic_recommendations'>"
        f"<h2>{SECTION_LABELS_ES['strategic_recommendations']}</h2>"
        + _narrative(view, "strategic_recommendations")
        + f"<h3>Prioridades</h3><ul>{priorities}</ul>"
        + f"<div class='recommendation-grid'>{cards}</div>"
        + f"<p class='muted'>{_escape(recs['reasoning_summary'])}</p>"
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
        + f"<ul>{missing}</ul>"
        + "</section>"
        + "<section id='appendix'>"
        f"<h2>{SECTION_LABELS_ES['appendix']}</h2>"
        f"<h3>MetodologÃ­a</h3><ul>{methodology}</ul>"
        f"<h3>Fuentes:</h3><ul>{sources}</ul>"
        "</section>"
    )


def report_strategy_warnings(report_model: dict[str, Any]) -> list[str]:
    """Identify whether a report model lacks accepted strategic analysis.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: warning strings; empty when final strategy is present.
    Assumptions: Step 9 writes analysis_status='accepted' into executive content.
    """

    executive = get_section(report_model, "executive_summary").get("content", {})
    recommendations = get_section(report_model, "strategic_recommendations").get("content", {})
    warnings: list[str] = []
    if executive.get("analysis_status") != "accepted":
        warnings.append(f"Strategic analysis is unavailable or not accepted for {report_model.get('report_id', 'report')}.")
    if not recommendations.get("recommendations"):
        warnings.append("No accepted strategic recommendations are present in the report model.")
    return warnings


def validate_strategy_available(report_model: dict[str, Any]) -> None:
    """Raise when final rendering would omit accepted strategic analysis.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: None; raises ValueError on missing strategy.
    Assumptions: draft rendering must be explicitly allowed by the caller.
    """

    warnings = report_strategy_warnings(report_model)
    if warnings:
        raise ValueError("; ".join(warnings))


def _styles() -> str:
    """Return embedded CSS for a responsive printable executive report.

    Inputs: none.
    Outputs: CSS string.
    Assumptions: the HTML report is self-contained for downloads.
    """

    return """
    :root { --navy:#17324d; --blue:#245b89; --green:#1f7a5b; --red:#b84242; --amber:#b7791f; --ink:#172033; --muted:#647084; --line:#d8e1ea; --bg:#f4f7fb; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Arial, Helvetica, sans-serif; color:var(--ink); background:var(--bg); line-height:1.45; }
    main { max-width: 1180px; margin:0 auto; padding:32px; }
    section { background:#fff; margin:22px 0; padding:30px; border-radius:18px; box-shadow:0 8px 28px rgba(23,50,77,.08); break-inside: avoid; }
    .cover { min-height: 360px; display:flex; flex-direction:column; justify-content:center; color:#fff; background:linear-gradient(135deg,var(--navy),var(--blue)); }
    .cover-mark { letter-spacing:.16em; text-transform:uppercase; font-size:13px; opacity:.82; }
    h1 { margin:.35em 0; font-size:44px; line-height:1.05; }
    h2 { margin:0 0 18px; color:var(--navy); font-size:25px; border-left:6px solid var(--blue); padding-left:12px; }
    h3 { color:var(--navy); margin-bottom:8px; }
    .cover h1, .cover h2 { color:#fff; border:0; padding:0; }
    .period { font-size:22px; font-weight:700; }
    .cover-note, .lead { font-size:18px; max-width:850px; }
    .section-analysis { font-size:15px; color:#263244; background:#fbfdff; border-left:4px solid var(--blue); padding:10px 12px; border-radius:8px; }
    .two-col { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; }
    .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }
    .kpi-card { border:1px solid var(--line); border-radius:14px; padding:16px; min-height:128px; background:#fbfdff; }
    .kpi-card span { display:block; color:var(--muted); font-size:13px; }
    .kpi-card strong { display:block; font-size:24px; margin:6px 0; }
    .kpi-card.good { border-top:5px solid var(--green); }
    .kpi-card.risk { border-top:5px solid var(--red); }
    .kpi-card.neutral { border-top:5px solid var(--blue); }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; margin:16px 0; font-size:13px; }
    th { background:#eef4fb; color:var(--navy); text-align:left; }
    th, td { border:1px solid var(--line); padding:9px; vertical-align:top; }
    tr:nth-child(even) td { background:#fbfdff; }
    .svg-chart, .line-chart { width:100%; height:auto; margin:14px 0; background:#fbfdff; border:1px solid var(--line); border-radius:12px; padding:10px; }
    .chart-title { font-weight:700; fill:var(--navy); font-size:16px; }
    .axis-label, .value-label { fill:#263244; font-size:12px; }
    .line-chart polyline { fill:none; stroke:var(--blue); stroke-width:3; }
    .line-chart circle { fill:var(--green); stroke:#fff; stroke-width:1; }
    .trend-grid, .recommendation-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }
    .trend-card, .recommendation-card, .status-card { border:1px solid var(--line); border-radius:14px; padding:16px; background:#fbfdff; }
    .status-card.positive { border-left:5px solid var(--green); }
    .badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#e9f2fb; color:var(--navy); font-weight:700; font-size:12px; }
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
        "<footer>Finance AI Agent Â· Reporte ejecutivo generado desde datos procesados y validados.</footer>"
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

