"""HTML renderer for Finance AI Agent report models."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


SECTION_LABELS_ES: dict[str, str] = {
    "cover": "Portada",
    "executive_summary": "Resumen ejecutivo",
    "financial_health_overview": "Salud financiera",
    "kpi_overview": "KPIs principales",
    "revenue_analysis": "Ingresos",
    "expense_analysis": "Gastos",
    "department_analysis": "Análisis por departamento",
    "anomaly_summary": "Anomalías detectadas",
    "investigation_evidence": "Evidencia de investigación",
    "strategic_recommendations": "Recomendaciones estratégicas",
    "missing_information": "Información faltante",
    "appendix": "Apéndice",
}


def _escape(value: Any) -> str:
    """Escape a value for safe HTML output.

    Inputs: any scalar value from the report model.
    Outputs: escaped string.
    Assumptions: renderers never mutate source data.
    """

    return html.escape("" if value is None else str(value))


def _number(value: Any) -> float | None:
    """Convert a model value to float when possible.

    Inputs: scalar model value.
    Outputs: float or None.
    Assumptions: conversion is for display scale only, not recalculation.
    """

    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _format_value(value: Any, unit: str | None = None) -> str:
    """Format common report values for Spanish-facing display.

    Inputs: value and optional unit label.
    Outputs: readable text.
    Assumptions: formatting does not change underlying calculations.
    """

    number = _number(value)
    if number is None:
        return _escape(value)
    if unit == "ratio" or (abs(number) <= 1 and unit is None):
        return f"{number:.1%}"
    if unit == "USD":
        return f"${number:,.0f}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def _section(report_model: dict[str, Any], section_id: str) -> dict[str, Any]:
    """Find one report section by ID.

    Inputs: report model and section ID.
    Outputs: section dictionary or an empty placeholder section.
    Assumptions: missing sections should render gracefully for robustness.
    """

    for section in report_model.get("sections", []):
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return section
    return {
        "section_id": section_id,
        "title": SECTION_LABELS_ES.get(section_id, section_id),
        "content": {},
        "source_references": [],
        "warnings": [f"Sección faltante en el modelo: {section_id}"],
    }


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple HTML table.

    Inputs: header labels and row values.
    Outputs: HTML table string.
    Assumptions: all values are escaped before insertion.
    """

    header_html = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    row_html = []
    if not rows:
        rows = [["Sin datos disponibles"] + [""] * max(0, len(headers) - 1)]
    for row in rows:
        row_html.append(
            "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def _source_note(section: dict[str, Any]) -> str:
    """Render compact source references for a section.

    Inputs: one report section dictionary.
    Outputs: HTML source note or empty string.
    Assumptions: source references come from processed pipeline artifacts only.
    """

    sources = section.get("source_references", [])
    if not sources:
        return ""
    visible = "; ".join(str(source) for source in sources[:6])
    return f"<p class='sources'>Fuentes: {_escape(visible)}</p>"


def _warning_note(section: dict[str, Any]) -> str:
    """Render section warnings when strategy or source data is incomplete.

    Inputs: one report section dictionary.
    Outputs: HTML warning note or empty string.
    Assumptions: warnings are produced by upstream validators, not renderers.
    """

    warnings = section.get("warnings", [])
    if not warnings:
        return ""
    items = "".join(f"<li>{_escape(warning)}</li>" for warning in warnings[:6])
    return f"<div class='warning'><strong>Advertencia:</strong><ul>{items}</ul></div>"


def report_strategy_warnings(report_model: dict[str, Any]) -> list[str]:
    """Identify whether a report model lacks accepted strategic analysis.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: warning strings; empty when strategic analysis is ready for final rendering.
    Assumptions: Step 9 marks accepted analysis with analysis_status='accepted'.
    """

    executive = _section(report_model, "executive_summary")
    recommendations = _section(report_model, "strategic_recommendations")
    executive_content = executive.get("content", {})
    recommendation_content = recommendations.get("content", {})
    warnings: list[str] = []
    if executive_content.get("analysis_status") != "accepted":
        warnings.append(
            "Strategic analysis is unavailable or not accepted for "
            f"{report_model.get('report_id', 'report')}."
        )
    recs = recommendation_content.get("recommendations", [])
    if not isinstance(recs, list) or not recs:
        warnings.append(
            "No accepted strategic recommendations are present in the report model."
        )
    return warnings


def validate_strategy_available(report_model: dict[str, Any]) -> None:
    """Raise when final rendering would omit accepted strategic analysis.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: None; raises ValueError on missing strategy.
    Assumptions: callers may explicitly bypass this guard for draft reports.
    """

    warnings = report_strategy_warnings(report_model)
    if warnings:
        raise ValueError("; ".join(warnings))


def _bar_chart(items: list[tuple[str, float]], *, title: str) -> str:
    """Render a lightweight CSS bar chart.

    Inputs: label/value pairs and chart title.
    Outputs: HTML chart markup.
    Assumptions: values are already calculated upstream.
    """

    if not items:
        return "<p class='muted'>Sin datos disponibles para el gráfico.</p>"
    max_value = max(abs(value) for _, value in items) or 1.0
    bars = []
    for label, value in items:
        width = max(4.0, abs(value) / max_value * 100)
        css_class = "negative" if value < 0 else "positive"
        bars.append(
            "<div class='bar-row'>"
            f"<span class='bar-label'>{_escape(label)}</span>"
            "<div class='bar-track'>"
            f"<div class='bar {css_class}' style='width:{width:.1f}%'></div>"
            "</div>"
            f"<span class='bar-value'>{_escape(_format_value(value, 'USD'))}</span>"
            "</div>"
        )
    return f"<div class='chart'><h4>{_escape(title)}</h4>{''.join(bars)}</div>"


def _render_cover(report_model: dict[str, Any]) -> str:
    """Render the cover section.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: cover metadata comes from the report model.
    """

    cover = _section(report_model, "cover")
    content = cover.get("content", {})
    return (
        "<section class='cover' id='cover'>"
        "<div class='eyebrow'>Finance AI Agent</div>"
        f"<h1>{_escape(content.get('title', 'Reporte financiero'))}</h1>"
        f"<h2>{_escape(report_model.get('report_period'))}</h2>"
        "<p>Reporte estructurado generado a partir de salidas procesadas del pipeline.</p>"
        "</section>"
    )


def _render_executive_summary(report_model: dict[str, Any]) -> str:
    """Render executive summary content.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: summary may be unavailable when Ollama was unavailable.
    """

    section = _section(report_model, "executive_summary")
    content = section.get("content", {})
    findings = content.get("key_findings") or []
    findings_html = "".join(f"<li>{_escape(item)}</li>" for item in findings)
    if not findings_html:
        findings_html = "<li>No hay hallazgos estratégicos generados por LLM.</li>"
    root_causes = content.get("root_causes") or []
    root_html = "".join(f"<li>{_escape(item)}</li>" for item in root_causes)
    if not root_html:
        root_html = "<li>Sin causas raíz estratégicas disponibles.</li>"
    return (
        "<section id='executive_summary'>"
        f"<h2>{SECTION_LABELS_ES['executive_summary']}</h2>"
        f"{_warning_note(section)}"
        f"<p>{_escape(content.get('summary', 'Sin resumen disponible.'))}</p>"
        "<h3>Hallazgos clave</h3>"
        f"<ul>{findings_html}</ul>"
        "<h3>Causas raíz probables</h3>"
        f"<ul>{root_html}</ul>"
        f"<p class='muted'>Confianza: {_escape(content.get('confidence', 'N/D'))}</p>"
        f"{_source_note(section)}"
        "</section>"
    )


def _render_financial_health(report_model: dict[str, Any]) -> str:
    """Render financial health overview cards and comparison chart.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: all metrics were calculated upstream.
    """

    section = _section(report_model, "financial_health_overview")
    content = section.get("content", {})
    metrics = [
        ("Ingresos totales", content.get("total_revenue"), "USD"),
        ("Gastos totales", content.get("total_expenses"), "USD"),
        ("Resultado operativo", content.get("net_operating_result"), "USD"),
        ("Flujo neto de caja", content.get("net_cash_flow"), "USD"),
        ("Caja final", content.get("ending_cash"), "USD"),
        ("Cobranza", content.get("collection_rate"), "ratio"),
    ]
    cards = "".join(
        "<div class='metric-card'>"
        f"<span>{_escape(label)}</span><strong>{_escape(_format_value(value, unit))}</strong>"
        "</div>"
        for label, value, unit in metrics
    )
    chart = _bar_chart(
        [
            ("Ingresos", _number(content.get("total_revenue")) or 0.0),
            ("Gastos", _number(content.get("total_expenses")) or 0.0),
            ("Resultado", _number(content.get("net_operating_result")) or 0.0),
        ],
        title="Comparación ingresos/gastos",
    )
    return (
        "<section id='financial_health_overview'>"
        f"<h2>{SECTION_LABELS_ES['financial_health_overview']}</h2>"
        f"<div class='metric-grid'>{cards}</div>{chart}{_source_note(section)}</section>"
    )


def _render_kpis(report_model: dict[str, Any]) -> str:
    """Render KPI table.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: KPI values are copied from processed CSV outputs.
    """

    section = _section(report_model, "kpi_overview")
    kpis = section.get("content", {}).get("kpis", [])
    rows = [
        [
            item.get("metric"),
            _format_value(item.get("value"), item.get("unit")),
            item.get("unit"),
            item.get("availability"),
            item.get("source"),
        ]
        for item in kpis
        if isinstance(item, dict)
    ]
    return (
        "<section id='kpi_overview'>"
        f"<h2>{SECTION_LABELS_ES['kpi_overview']}</h2>"
        + _html_table(["Indicador", "Valor", "Unidad", "Estado", "Fuente"], rows)
        + _source_note(section)
        + "</section>"
    )


def _render_revenue_expense(report_model: dict[str, Any], section_id: str) -> str:
    """Render revenue or expense analysis section.

    Inputs: report model and section ID.
    Outputs: HTML section.
    Assumptions: section content shape follows Step 10A report model.
    """

    section = _section(report_model, section_id)
    content = section.get("content", {})
    rows = []
    for key, value in content.items():
        if key.endswith("summary"):
            continue
        rows.append([key, _format_value(value, "USD" if "pct" not in key else "ratio")])
    chart_items = []
    if section_id == "revenue_analysis":
        chart_items = [
            ("Presupuesto", _number(content.get("revenue_budget")) or 0.0),
            ("Actual", _number(content.get("total_revenue")) or 0.0),
            ("Variación", _number(content.get("revenue_variance")) or 0.0),
        ]
    else:
        chart_items = [
            ("Presupuesto", _number(content.get("expense_budget")) or 0.0),
            ("Actual", _number(content.get("total_expenses")) or 0.0),
            ("Variación", _number(content.get("expense_variance")) or 0.0),
        ]
    return (
        f"<section id='{section_id}'>"
        f"<h2>{SECTION_LABELS_ES[section_id]}</h2>"
        + _html_table(["Métrica", "Valor"], rows)
        + _bar_chart(chart_items, title=SECTION_LABELS_ES[section_id])
        + _source_note(section)
        + "</section>"
    )


def _render_departments(report_model: dict[str, Any]) -> str:
    """Render department summary table and chart.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: department rows come from processed finance summary.
    """

    section = _section(report_model, "department_analysis")
    rows_data = section.get("content", {}).get("department_summary", [])
    rows = [
        [
            item.get("department"),
            _format_value(item.get("actual_revenue"), "USD"),
            _format_value(item.get("actual_expenses"), "USD"),
            _format_value(item.get("net_operating_result"), "USD"),
            _format_value(item.get("expense_variance_pct"), "ratio"),
        ]
        for item in rows_data
        if isinstance(item, dict)
    ]
    chart_items = [
        (str(item.get("department")), _number(item.get("net_operating_result")) or 0.0)
        for item in rows_data
        if isinstance(item, dict)
    ]
    return (
        "<section id='department_analysis'>"
        f"<h2>{SECTION_LABELS_ES['department_analysis']}</h2>"
        + _html_table(["Departamento", "Ingresos", "Gastos", "Resultado", "Var. gastos"], rows)
        + _bar_chart(chart_items, title="Resultado por departamento")
        + _source_note(section)
        + "</section>"
    )


def _render_anomalies(report_model: dict[str, Any]) -> str:
    """Render anomaly severity summary and top anomaly table.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: anomalies are copied from deterministic anomaly outputs.
    """

    section = _section(report_model, "anomaly_summary")
    content = section.get("content", {})
    severity = content.get("anomalies_by_severity", {})
    severity = severity if isinstance(severity, dict) else {}
    severity_rows = [[key, value] for key, value in severity.items()]
    top_rows = [
        [item.get("anomaly_id"), item.get("title"), item.get("severity"), item.get("evidence")]
        for item in content.get("top_anomalies", [])
        if isinstance(item, dict)
    ]
    return (
        "<section id='anomaly_summary'>"
        f"<h2>{SECTION_LABELS_ES['anomaly_summary']}</h2>"
        + _html_table(["Severidad", "Cantidad"], severity_rows)
        + _html_table(["ID", "Título", "Severidad", "Evidencia"], top_rows)
        + _source_note(section)
        + "</section>"
    )


def _render_evidence(report_model: dict[str, Any]) -> str:
    """Render investigation evidence summary.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: evidence details remain bounded in the report model.
    """

    section = _section(report_model, "investigation_evidence")
    items = section.get("content", {}).get("evidence_items", [])
    rows = [
        [
            item.get("task_id"),
            item.get("priority"),
            item.get("retrieval_name"),
            item.get("record_count"),
            item.get("evidence_summary"),
        ]
        for item in items
        if isinstance(item, dict)
    ]
    return (
        "<section id='investigation_evidence'>"
        f"<h2>{SECTION_LABELS_ES['investigation_evidence']}</h2>"
        + _html_table(["Tarea", "Prioridad", "Evidencia", "Registros", "Resumen"], rows)
        + _source_note(section)
        + "</section>"
    )


def _render_recommendations(report_model: dict[str, Any]) -> str:
    """Render strategic recommendations.

    Inputs: report model.
    Outputs: HTML section.
    Assumptions: recommendations may be empty if strategic analysis was unavailable.
    """

    section = _section(report_model, "strategic_recommendations")
    content = section.get("content", {})
    recommendations = content.get("recommendations", [])
    root_causes = content.get("root_causes", [])
    priorities = content.get("strategic_priorities", [])
    rows = []
    for item in recommendations:
        if isinstance(item, dict):
            rows.append([
                item.get("priority", ""),
                item.get("action", item.get("recommendation", "")),
                item.get("rationale", ""),
                item.get("expected_impact", ""),
            ])
        else:
            rows.append(["", item, "", ""])
    if not rows:
        rows = [["", "No hay recomendaciones estratégicas generadas.", "", ""]]
    priority_rows = [[item] for item in priorities if isinstance(item, str)]
    root_rows = [[item] for item in root_causes if isinstance(item, str)]
    return (
        "<section id='strategic_recommendations'>"
        f"<h2>{SECTION_LABELS_ES['strategic_recommendations']}</h2>"
        f"{_warning_note(section)}"
        "<h3>Prioridades estratégicas</h3>"
        + _html_table(["Prioridad"], priority_rows)
        + "<h3>Causas raíz</h3>"
        + _html_table(["Causa probable"], root_rows)
        + "<h3>Acciones recomendadas</h3>"
        + _html_table(["Prioridad", "Acción", "Razonamiento", "Impacto esperado"], rows)
        + f"<p>{_escape(content.get('reasoning_summary', ''))}</p>"
        + _source_note(section)
        + "</section>"
    )


def _render_missing_and_appendix(report_model: dict[str, Any], section_id: str) -> str:
    """Render missing information or appendix sections.

    Inputs: report model and section ID.
    Outputs: HTML section.
    Assumptions: generic key/value rendering is acceptable for appendix content.
    """

    section = _section(report_model, section_id)
    content = section.get("content", {})
    rows = []
    for key, value in content.items():
        if isinstance(value, list):
            rendered = "; ".join(str(item) for item in value[:20])
        else:
            rendered = str(value)
        rows.append([key, rendered])
    if not rows:
        rows = [["Estado", "Sin información adicional."]]
    return (
        f"<section id='{section_id}'>"
        f"<h2>{SECTION_LABELS_ES[section_id]}</h2>"
        + _html_table(["Campo", "Detalle"], rows)
        + _source_note(section)
        + "</section>"
    )


def _styles() -> str:
    """Return embedded CSS for the HTML report.

    Inputs: none.
    Outputs: CSS string.
    Assumptions: HTML is self-contained for easy local viewing.
    """

    return """
    body { font-family: Arial, Helvetica, sans-serif; margin: 0; color: #172033; background: #f5f7fb; }
    main { max-width: 1120px; margin: 0 auto; padding: 32px; }
    section { background: #fff; margin: 22px 0; padding: 28px; border-radius: 14px; box-shadow: 0 2px 10px rgba(23,32,51,0.08); }
    .cover { background: linear-gradient(135deg, #17324d, #245b89); color: white; padding: 56px 36px; }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.14em; opacity: 0.8; font-size: 13px; }
    h1 { margin: 12px 0; font-size: 40px; }
    h2 { margin-top: 0; color: #17324d; }
    .cover h2 { color: white; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 18px 0; }
    .metric-card { border: 1px solid #dbe4ee; padding: 14px; border-radius: 10px; background: #fbfdff; }
    .metric-card span { display: block; color: #5f6b7a; font-size: 13px; }
    .metric-card strong { display: block; margin-top: 6px; font-size: 20px; }
    table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }
    th { background: #eef4fb; text-align: left; color: #17324d; }
    th, td { border: 1px solid #d8e1ea; padding: 8px; vertical-align: top; }
    tr:nth-child(even) td { background: #fbfdff; }
    .chart { margin: 20px 0; }
    .bar-row { display: grid; grid-template-columns: 150px 1fr 120px; gap: 10px; align-items: center; margin: 8px 0; }
    .bar-track { background: #e8eef5; height: 14px; border-radius: 999px; overflow: hidden; }
    .bar { height: 14px; border-radius: 999px; }
    .positive { background: #1f8a70; }
    .negative { background: #c94c4c; }
    .bar-value { text-align: right; font-variant-numeric: tabular-nums; }
    .muted { color: #667085; }
    .sources { color: #667085; font-size: 12px; margin-top: 12px; }
    .warning { background: #fff7e6; border: 1px solid #ffd591; color: #5f3b00; padding: 10px 12px; border-radius: 8px; margin: 12px 0; }
    .warning ul { margin: 6px 0 0 18px; padding: 0; }
    footer { color: #667085; text-align: center; padding: 24px; font-size: 12px; }
    """


def render_report_html(report_model: dict[str, Any]) -> str:
    """Render a report model to a complete Spanish HTML document.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: complete HTML string.
    Assumptions: section IDs use the Step 10A report contract.
    """

    body = [
        _render_cover(report_model),
        _render_executive_summary(report_model),
        _render_financial_health(report_model),
        _render_kpis(report_model),
        _render_revenue_expense(report_model, "revenue_analysis"),
        _render_revenue_expense(report_model, "expense_analysis"),
        _render_departments(report_model),
        _render_anomalies(report_model),
        _render_evidence(report_model),
        _render_recommendations(report_model),
        _render_missing_and_appendix(report_model, "missing_information"),
        _render_missing_and_appendix(report_model, "appendix"),
    ]
    return (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_escape(report_model.get('report_id', 'Reporte financiero'))}</title>"
        f"<style>{_styles()}</style></head><body><main>{''.join(body)}</main>"
        "<footer>Generado por Finance AI Agent usando datos procesados existentes.</footer>"
        "</body></html>"
    )


def load_report_model(path: str | Path) -> dict[str, Any]:
    """Load a report model JSON file.

    Inputs: report model path.
    Outputs: parsed report model dictionary.
    Assumptions: report model root is a JSON object.
    """

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Report model root must be an object: {path}")
    return value


def save_report_html(report_model: dict[str, Any], output_path: str | Path) -> Path:
    """Render and save a report model as HTML.

    Inputs: report model dictionary and output path.
    Outputs: resolved written path.
    Assumptions: parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_html(report_model), encoding="utf-8")
    return path
