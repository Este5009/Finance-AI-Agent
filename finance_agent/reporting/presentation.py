"""Presentation adapters for executive financial reports.

This module formats already-validated report data for HTML/PDF renderers.  It
does not translate model-authored narrative; strategic prose must be generated
in Spanish by the analysis stage before report generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    "Ã¡": "á",
    "Ã©": "é",
    "Ã­": "í",
    "Ã³": "ó",
    "Ãº": "ú",
    "Ã±": "ñ",
    "Ã¼": "ü",
    "Ã": "Á",
    "Ã‰": "É",
    "Ã": "Í",
    "Ã“": "Ó",
    "Ãš": "Ú",
    "Ã‘": "Ñ",
    "Â·": "·",
    "â†’": "→",
    "â€¦": "...",
}

MONTH_LABELS_ES: dict[str, str] = {
    "01": "Ene",
    "02": "Feb",
    "03": "Mar",
    "04": "Abr",
    "05": "May",
    "06": "Jun",
    "07": "Jul",
    "08": "Ago",
    "09": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dic",
}

EMPTY_DISPLAY_VALUES: tuple[str, ...] = (
    "",
    "-",
    "N/D",
    "n/d",
    "No disponible",
    "no disponible",
    "None",
    "none",
    "null",
    "NULL",
)


@dataclass(frozen=True)
class ReportSectionTemplate:
    """Reusable presentation contract for one executive report section.

    Inputs: section metadata, evidence requirements, visual specifications, and
    the Step 9 narrative field that should populate analytical text.
    Outputs: immutable template used by report builders and renderers.
    Assumptions: templates describe structure only; they do not contain
    analytical conclusions or report-specific prose.
    """

    section_id: str
    title_es: str
    objective: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    chart_specs: tuple[str, ...]
    table_specs: tuple[str, ...]
    narrative_fields: tuple[str, ...]
    validation_rules: tuple[str, ...]
    visibility_rule: str


EXECUTIVE_SECTION_ORDER: tuple[str, ...] = (
    "cover",
    "executive_summary",
    "financial_health_overview",
    "kpi_overview",
    "historical_trends",
    "revenue_expense_analysis",
    "department_analysis",
    "anomaly_summary",
    "recommendation_follow_up",
    "strategic_recommendations",
    "missing_information",
    "appendix",
)

SECTION_LABELS_ES: dict[str, str] = {
    "cover": "Portada",
    "executive_summary": "Resumen ejecutivo",
    "financial_health_overview": "Salud financiera ejecutiva",
    "kpi_overview": "KPIs y cumplimiento de metas",
    "historical_trends": "Tendencias históricas",
    "revenue_expense_analysis": "Análisis de ingresos y gastos",
    "department_analysis": "Análisis por departamento",
    "anomaly_summary": "Anomalías del período",
    "investigation_evidence": "Evidencia de investigación",
    "recommendation_follow_up": "Seguimiento de recomendaciones emitidas anteriormente",
    "longitudinal_risk_assessment": "Riesgos históricos recurrentes",
    "strategic_recommendations": "Recomendaciones estratégicas actuales",
    "missing_information": "Información faltante / supuestos",
    "appendix": "Metodología y fuentes",
}

ENTITY_LABELS_ES: dict[str, str] = {
    "Health Sciences": "Ciencias de la Salud",
    "University": "Universidad",
    "Engineering": "Ingeniería",
    "Business": "Negocios",
    "Arts & Humanities": "Artes y Humanidades",
    "Student Services": "Servicios Estudiantiles",
    "Administration": "Administración",
}

RISK_TYPE_LABELS_ES: dict[str, str] = {
    "payroll_overtime_overspend": "Sobrecosto recurrente de nómina y horas extra",
    "recurring_vendor_duplicate": "Riesgo recurrente en pagos a proveedores",
    "negative_cash_flow": "Flujo de caja negativo recurrente",
    "TUITION_COLLECTION_MIN": "Riesgo recurrente de cobranza estudiantil",
    "PAYROLL_RATIO_MAX": "Nómina sobre ingresos por encima de meta",
    "NET_CASH_FLOW_MIN": "Flujo neto de caja bajo o negativo",
    "OPERATING_RESULT_MIN": "Resultado operativo bajo o negativo",
    "OVERDUE_PAYMENT_MAX": "Pagos estudiantiles vencidos recurrentes",
    "VENDOR_PAYMENT_REVIEW": "Pagos a proveedores requieren revisión",
    "CATEGORY_BUDGET_RANGE": "Desviación presupuestaria por categoría",
    "CATEGORY_OVERSPEND_FLAG": "Sobregasto recurrente por categoría",
}

RECOMMENDATION_TOPIC_LABELS_ES: dict[str, str] = {
    "payroll_overtime": "Control de horas extra y nómina",
    "collections": "Gestión de cobranza estudiantil",
    "vendor_controls": "Controles de pagos a proveedores",
}

RECOMMENDATION_FOLLOW_UP_INTRO = (
    "Las siguientes recomendaciones fueron emitidas en informes financieros de meses anteriores "
    "y se evalúan utilizando la evidencia acumulada hasta el periodo actual."
)

METRIC_LABELS_ES: dict[str, tuple[str, str, str]] = {
    "total_revenue": ("Ingresos totales", "USD", "Ingreso operativo reconocido en el periodo."),
    "total_expenses": ("Gastos totales", "USD", "Gasto operativo reconocido en el periodo."),
    "net_operating_result": ("Resultado operativo", "USD", "Diferencia entre ingresos y gastos operativos."),
    "net_cash_flow": ("Flujo neto de caja", "USD", "Entrada o salida neta de efectivo del periodo."),
    "ending_cash": ("Caja final", "USD", "Saldo de caja al cierre del periodo."),
    "payroll_percentage_of_revenue": ("Nómina / ingresos", "ratio", "Peso de la nómina sobre ingresos."),
    "student_payment_collection_rate": ("Tasa de cobranza", "ratio", "Porcentaje cobrado sobre saldos estudiantiles."),
    "collection_rate": ("Tasa de cobranza", "ratio", "Porcentaje cobrado sobre saldos estudiantiles."),
    "overdue_payment_percentage": ("Pagos vencidos", "ratio", "Porcentaje de pagos estudiantiles vencidos."),
    "vendor_payment_total": ("Pagos a proveedores", "USD", "Monto total de pagos a proveedores en el periodo."),
    "scholarship_awarded_total": ("Becas otorgadas", "USD", "Monto total de becas otorgadas en el periodo."),
    "revenue_budget": ("Presupuesto de ingresos", "USD", "Meta presupuestada de ingresos."),
    "revenue_variance": ("Variación de ingresos", "USD", "Diferencia entre ingreso actual y presupuesto."),
    "revenue_variance_pct": ("Variación de ingresos", "ratio", "Variación porcentual contra presupuesto."),
    "expense_budget": ("Presupuesto de gastos", "USD", "Meta presupuestada de gastos."),
    "expense_variance": ("Variación de gastos", "USD", "Diferencia entre gasto actual y presupuesto."),
    "expense_variance_pct": ("Variación de gastos", "ratio", "Variación porcentual contra presupuesto."),
    "payroll_total": ("Nómina total", "USD", "Gasto total de nómina."),
    "revenue_budget_variance": ("Variación presupuestaria de ingresos", "USD", "Diferencia entre ingresos reales y presupuesto."),
    "revenue_budget_variance_pct": ("Variación presupuestaria de ingresos", "ratio", "Diferencia porcentual entre ingresos reales y presupuesto."),
    "expense_budget_variance": ("Variación presupuestaria de gastos", "USD", "Diferencia entre gastos reales y presupuesto."),
    "expense_budget_variance_pct": ("Variación presupuestaria de gastos", "ratio", "Diferencia porcentual entre gastos reales y presupuesto."),
}

SEVERITY_LABELS_ES: dict[str, str] = {
    "critical": "Crítica",
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
    "info": "Informativa",
}

PRIORITY_LABELS_ES: dict[str, str] = {
    "critical": "Crítica",
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
}

TOOL_LABELS_ES: dict[str, str] = {
    "department_history": "evidencia departamental",
    "payroll_history": "evidencia de nómina",
    "vendor_history": "evidencia de proveedores",
    "student_payment_history": "evidencia de cobranza estudiantil",
    "cashflow_history": "evidencia de flujo de caja",
    "transactions": "transacciones procesadas",
    "previous_cycle_memory": "memoria del ciclo previo",
    "financial_report": "reporte financiero procesado",
}

CANONICAL_IDENTIFIERS: tuple[str, ...] = tuple(METRIC_LABELS_ES)
RAW_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{[^{}]*:[^{}]*\}"),
    re.compile(r"\[[^\[\]]*\{[^\[\]]*\}"),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"\bget_[a-z_]+\b"),
)

REPORT_SECTION_TEMPLATES: dict[str, ReportSectionTemplate] = {
    "executive_summary": ReportSectionTemplate(
        "executive_summary",
        SECTION_LABELS_ES["executive_summary"],
        "Sintetizar los asuntos financieros materiales para liderazgo.",
        ("strategic_analysis", "finance_summary"),
        ("historical_context",),
        (),
        (),
        ("executive_summary", "key_findings", "root_causes"),
        ("spanish", "evidence_bound", "non_generic"),
        "show_when_strategy_accepted",
    ),
    "financial_health_overview": ReportSectionTemplate(
        "financial_health_overview",
        SECTION_LABELS_ES["financial_health_overview"],
        "Mostrar salud financiera con KPIs principales y comentario analítico.",
        ("finance_summary", "strategic_analysis.financial_health_analysis"),
        ("cash_flow",),
        ("financial_health_bar_chart",),
        ("financial_health_cards",),
        ("financial_health_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_finance_summary_available",
    ),
    "kpi_overview": ReportSectionTemplate(
        "kpi_overview",
        SECTION_LABELS_ES["kpi_overview"],
        "Presentar KPIs, estado de disponibilidad y lectura ejecutiva.",
        ("kpi_summary", "strategic_analysis.kpi_analysis"),
        ("goal_progress",),
        (),
        ("kpi_table",),
        ("kpi_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_kpis_available",
    ),
    "historical_summary": ReportSectionTemplate(
        "historical_summary",
        "Resumen histórico",
        "Resumir el contexto histórico disponible sin cargar reportes completos.",
        ("historical_context", "strategic_analysis.historical_summary"),
        (),
        (),
        (),
        ("historical_summary",),
        ("spanish", "evidence_bound"),
        "show_when_history_available",
    ),
    "historical_trends": ReportSectionTemplate(
        "historical_trends",
        SECTION_LABELS_ES["historical_trends"],
        "Mostrar tendencias cronológicas y análisis asociado.",
        ("historical_context", "strategic_analysis.historical_trend_analysis"),
        (),
        ("historical_kpi_line_charts",),
        (),
        ("historical_trend_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_historical_trends_available",
    ),
    "revenue_expense_analysis": ReportSectionTemplate(
        "revenue_expense_analysis",
        SECTION_LABELS_ES["revenue_expense_analysis"],
        "Comparar ingresos, gastos, presupuesto y resultado operativo.",
        ("finance_summary",),
        ("budget_vs_actual",),
        ("revenue_expense_bar_chart", "budget_actual_bar_chart"),
        ("revenue_expense_table",),
        ("financial_health_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_finance_summary_available",
    ),
    "department_analysis": ReportSectionTemplate(
        "department_analysis",
        SECTION_LABELS_ES["department_analysis"],
        "Comparar desempeño departamental y riesgos operativos.",
        ("department_summary", "strategic_analysis.department_analysis"),
        ("department_evidence",),
        ("department_result_chart",),
        ("department_table",),
        ("department_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_department_rows_available",
    ),
    "anomaly_summary": ReportSectionTemplate(
        "anomaly_summary",
        SECTION_LABELS_ES["anomaly_summary"],
        "Presentar anomalías relevantes y su análisis.",
        ("anomaly_report", "strategic_analysis.anomaly_analysis"),
        (),
        ("anomaly_severity_chart",),
        ("anomaly_table",),
        ("anomaly_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_anomaly_artifact_available",
    ),
    "recommendation_follow_up": ReportSectionTemplate(
        "recommendation_follow_up",
        SECTION_LABELS_ES["recommendation_follow_up"],
        "Dar seguimiento a recomendaciones previas con evidencia histórica.",
        ("historical_context", "strategic_analysis.recommendation_follow_up_analysis"),
        (),
        (),
        ("recommendation_follow_up_table",),
        ("recommendation_follow_up_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_follow_up_available",
    ),
    "longitudinal_risk_assessment": ReportSectionTemplate(
        "longitudinal_risk_assessment",
        SECTION_LABELS_ES["longitudinal_risk_assessment"],
        "Evaluar riesgos recurrentes o persistentes a través del tiempo.",
        ("historical_context", "strategic_analysis.longitudinal_risk_analysis"),
        (),
        (),
        ("recurring_risk_table",),
        ("longitudinal_risk_analysis",),
        ("spanish", "evidence_bound"),
        "show_when_recurring_risks_available",
    ),
    "strategic_recommendations": ReportSectionTemplate(
        "strategic_recommendations",
        SECTION_LABELS_ES["strategic_recommendations"],
        "Priorizar acciones ejecutivas respaldadas por evidencia.",
        ("strategic_analysis.strategic_recommendations",),
        (),
        (),
        ("recommendation_cards",),
        ("strategic_recommendations",),
        ("spanish", "evidence_bound", "non_generic"),
        "show_when_recommendations_available",
    ),
    "missing_information": ReportSectionTemplate(
        "missing_information",
        SECTION_LABELS_ES["missing_information"],
        "Listar brechas de evidencia que limitan el análisis.",
        ("strategic_analysis.missing_information",),
        (),
        (),
        ("missing_information_list",),
        ("missing_information",),
        ("spanish", "evidence_bound"),
        "show_when_missing_information_available",
    ),
    "appendix": ReportSectionTemplate(
        "appendix",
        SECTION_LABELS_ES["appendix"],
        "Documentar metodología, validación y fuentes procesadas.",
        ("source_references",),
        ("validation_status",),
        (),
        ("source_file_list",),
        (),
        ("no_absolute_paths",),
        "always_show",
    ),
}


@dataclass(frozen=True)
class PresentationValidationResult:
    """Validation result for an executive presentation view.

    Inputs: errors and warnings found during presentation validation.
    Outputs: immutable validation result.
    Assumptions: errors block executive rendering.
    """

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def get_section(report_model: dict[str, Any], section_id: str) -> dict[str, Any]:
    """Return one section from a report model.

    Inputs: report model and section ID.
    Outputs: matching section or an empty placeholder.
    Assumptions: optional sections may be absent.
    """

    for section in report_model.get("sections", []):
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return section
    return {"section_id": section_id, "content": {}, "source_references": [], "warnings": []}


def number_value(value: Any) -> float | None:
    """Convert a scalar to float when possible.

    Inputs: scalar value.
    Outputs: float or None.
    Assumptions: conversion is for display scale only, not recalculation.
    """

    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def format_value(value: Any, unit: str | None = None) -> str:
    """Format a processed value for executive display.

    Inputs: value and optional unit.
    Outputs: formatted currency, percentage, or scalar.
    Assumptions: upstream Python already calculated the value.
    """

    number = number_value(value)
    if number is None:
        return "N/D" if value in (None, "") else sanitize_text(value)
    if unit == "ratio":
        return f"{number:.1%}"
    if unit == "USD":
        sign = "-" if number < 0 else ""
        return f"{sign}${abs(number):,.0f}"
    if unit == "count":
        return f"{number:,.0f}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def display_or_unavailable(value: Any, *, compact: bool = False) -> str:
    """Return a user-facing value with a readable unavailable label.

    Inputs: any display value and whether a compact table context is required.
    Outputs: sanitized value or a Spanish unavailable label.
    Assumptions: compact contexts may still use N/D when visual space is tight.
    """

    text = sanitize_text(value)
    if is_empty_display_value(text):
        return "N/D" if compact else "No disponible"
    return text


def is_empty_display_value(value: Any) -> bool:
    """Return whether a rendered table/card value has little executive value.

    Inputs: any scalar display value.
    Outputs: True when the value is blank or an unavailable placeholder.
    Assumptions: this supports presentation choices only, never calculations.
    """

    return str(value if value is not None else "").strip() in EMPTY_DISPLAY_VALUES


def trim_low_value_columns(
    headers: list[str],
    rows: list[list[Any]],
    *,
    protected_headers: tuple[str, ...] = (),
) -> tuple[list[str], list[list[Any]]]:
    """Remove columns that contain no meaningful display values.

    Inputs: table headers, table rows, and headers that must always remain.
    Outputs: trimmed headers and rows.
    Assumptions: source JSON keeps full detail; trimming is presentation-only.
    """

    if not headers or not rows:
        return headers, rows
    protected = {header.lower() for header in protected_headers}
    keep_indices: list[int] = []
    for index, header in enumerate(headers):
        column_values = [row[index] if index < len(row) else "" for row in rows]
        if str(header).lower() in protected or any(not is_empty_display_value(value) for value in column_values):
            keep_indices.append(index)
    if not keep_indices:
        return headers[:1], [[row[0] if row else "No disponible"] for row in rows]
    return [headers[index] for index in keep_indices], [
        [row[index] if index < len(row) else "" for index in keep_indices]
        for row in rows
    ]


def table_has_useful_detail(
    headers: list[str],
    rows: list[list[Any]],
    *,
    min_useful_ratio: float = 0.45,
) -> bool:
    """Return whether a table is worth rendering for executives.

    Inputs: headers, rows, and minimum useful-cell ratio.
    Outputs: True when the table has enough non-placeholder cells.
    Assumptions: a single useful row may still be better shown as a card by the renderer.
    """

    if not headers or not rows:
        return False
    total = len(headers) * len(rows)
    if total <= 0:
        return False
    useful = 0
    for row in rows:
        for index in range(len(headers)):
            if index < len(row) and not is_empty_display_value(row[index]):
                useful += 1
    return (useful / total) >= min_useful_ratio


def format_compact_axis_value(value: Any, unit: str | None = None) -> str:
    """Format a numeric axis tick with compact executive notation.

    Inputs: value and unit.
    Outputs: compact currency/percentage/count label.
    Assumptions: exact values remain displayed beside chart bars and in tables.
    """

    number = number_value(value)
    if number is None:
        return "N/D"
    if unit == "USD":
        sign = "-" if number < 0 else ""
        absolute = abs(number)
        if absolute >= 1_000_000:
            return f"{sign}${absolute / 1_000_000:.1f} M"
        if absolute >= 1_000:
            return f"{sign}${absolute / 1_000:.0f} k"
        return f"{sign}${absolute:,.0f}"
    return format_value(number, unit)


def format_period_label(period: Any, *, include_year: bool = True) -> str:
    """Return a readable Spanish period label for reports.

    Inputs: period identifier such as 2026_04, 2026-04, or 2026.
    Outputs: Spanish display label like Abr 2026.
    Assumptions: unknown/custom periods remain sanitized text.
    """

    text = str(period or "").strip()
    match = re.fullmatch(r"(20\d{2})[-_](0[1-9]|1[0-2])", text)
    if match:
        year, month = match.groups()
        return f"{MONTH_LABELS_ES[month]} {year}" if include_year else MONTH_LABELS_ES[month]
    return text


def replace_period_identifiers(text: str) -> str:
    """Replace raw monthly period IDs in user-facing text.

    Inputs: visible text.
    Outputs: text with 2026_04/2026-04 style labels converted to Spanish.
    Assumptions: source artifact filenames are handled separately.
    """

    return re.sub(
        r"\b(20\d{2})[-_](0[1-9]|1[0-2])\b",
        lambda match: format_period_label(match.group(0)),
        text,
    )


def display_metric_name(metric: Any) -> str:
    """Return a Spanish display label for a metric identifier.

    Inputs: canonical metric name or arbitrary label.
    Outputs: Spanish label when known; readable title otherwise.
    Assumptions: unknown metrics are sanitized but not translated by dictionary.
    """

    text = str(metric or "").strip()
    normalized = text.lower().replace(" ", "_")
    if text in METRIC_LABELS_ES:
        return METRIC_LABELS_ES[text][0]
    if normalized in METRIC_LABELS_ES:
        return METRIC_LABELS_ES[normalized][0]
    return text.replace("_", " ").strip().capitalize() or "Indicador"


def compact_source_label(source: Any) -> str:
    """Return a compact source label without absolute paths.

    Inputs: source reference.
    Outputs: filename or short label.
    Assumptions: full paths remain in JSON artifacts for audit.
    """

    text = str(source or "").strip()
    return Path(text).name if text else ""


def source_labels(section: dict[str, Any], *, limit: int = 4) -> list[str]:
    """Return compact source labels for a section.

    Inputs: section dictionary and maximum labels.
    Outputs: deduplicated source filenames.
    Assumptions: source labels are presentation-only provenance.
    """

    labels = [compact_source_label(item) for item in section.get("source_references", [])]
    return list(dict.fromkeys(label for label in labels if label))[:limit]


def section_templates_payload() -> dict[str, dict[str, Any]]:
    """Serialize section templates for renderer diagnostics.

    Inputs: none.
    Outputs: dictionary keyed by section ID with display/evidence metadata.
    Assumptions: templates are structural contracts only and contain no
    analytical conclusions.
    """

    return {
        section_id: {
            "section_id": template.section_id,
            "title_es": template.title_es,
            "objective": template.objective,
            "required_inputs": list(template.required_inputs),
            "optional_inputs": list(template.optional_inputs),
            "chart_specs": list(template.chart_specs),
            "table_specs": list(template.table_specs),
            "narrative_fields": list(template.narrative_fields),
            "validation_rules": list(template.validation_rules),
            "visibility_rule": template.visibility_rule,
        }
        for section_id, template in REPORT_SECTION_TEMPLATES.items()
    }


def section_narratives(report_model: dict[str, Any]) -> dict[str, str]:
    """Return Step-9-authored narrative for report sections.

    Inputs: report model.
    Outputs: section ID to sanitized Spanish narrative text.
    Assumptions: strategic analysis validated narrative language and evidence;
    this function only exposes already-authored prose to renderers.
    """

    narratives: dict[str, str] = {}
    for template in REPORT_SECTION_TEMPLATES.values():
        candidate_ids = (template.section_id,)
        if template.section_id == "revenue_expense_analysis":
            candidate_ids = ("revenue_analysis", "expense_analysis")
        for section_id in candidate_ids:
            content = get_section(report_model, section_id).get("content", {})
            if not isinstance(content, dict):
                continue
            for field in ("analysis", *template.narrative_fields):
                value = content.get(field)
                if isinstance(value, list):
                    text = " ".join(sanitize_items(value, limit=4))
                else:
                    text = sanitize_text(value)
                if text:
                    narratives[template.section_id] = text
                    break
            if template.section_id in narratives:
                break
    return narratives


def sanitize_text(value: Any) -> str:
    """Sanitize user-facing text without translating narrative.

    Inputs: text from validated analysis or deterministic summaries.
    Outputs: text with paths, tool names, and canonical metric IDs hidden.
    Assumptions: strategic prose is already professional Spanish.
    """

    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    sanitized = re.sub(r"[A-Za-z]:\\[^\s,;)]*", "[archivo procesado]", text)
    sanitized = re.sub(r"\bget_[a-z_]+\b", "herramienta de recuperación", sanitized)
    for metric, (label, _, _) in METRIC_LABELS_ES.items():
        sanitized = re.sub(rf"\b{re.escape(metric)}\b", label, sanitized)
    for entity, label in ENTITY_LABELS_ES.items():
        sanitized = re.sub(rf"\b{re.escape(entity)}\b", label, sanitized)
    return replace_period_identifiers(sanitized)


def display_entity_name(value: Any) -> str:
    """Return a Spanish display label for organization entities.

    Inputs: source entity/department name.
    Outputs: Spanish display name when known; sanitized original otherwise.
    Assumptions: internal canonical names remain unchanged in JSON sources.
    """

    text = str(value or "").strip()
    return ENTITY_LABELS_ES.get(text, sanitize_text(text))


def display_risk_name(value: Any, metric: Any = None) -> str:
    """Return a Spanish risk name from deterministic anomaly metadata.

    Inputs: anomaly type/risk label and optional metric.
    Outputs: executive Spanish risk name.
    Assumptions: unknown types are sanitized, not interpreted by Ollama.
    """

    text = str(value or "").strip()
    if text in RISK_TYPE_LABELS_ES:
        return RISK_TYPE_LABELS_ES[text]
    metric_text = str(metric or "").strip()
    if metric_text in METRIC_LABELS_ES:
        return f"Riesgo recurrente en {METRIC_LABELS_ES[metric_text][0].lower()}"
    return sanitize_text(text.replace("_", " ").capitalize() or "Riesgo recurrente")


def status_badge(status: str) -> dict[str, str]:
    """Return display metadata for a KPI or risk status badge.

    Inputs: normalized status string.
    Outputs: label, CSS class and compact symbol.
    Assumptions: color is never the only status indicator.
    """

    mapping = {
        "good": ("En rango", "good", "✓"),
        "amber": ("Atención", "amber", "!"),
        "risk": ("Riesgo", "risk", "!"),
        "neutral": ("Informativo", "neutral", "i"),
    }
    label, klass, icon = mapping.get(str(status or "").lower(), mapping["neutral"])
    return {"label": label, "class": klass, "icon": icon}


def trend_arrow(delta: float | None, *, inverse: bool = False) -> str:
    """Return a visual trend arrow for a delta.

    Inputs: numeric delta and whether lower values are favorable.
    Outputs: one of ▲, ▼ or →.
    Assumptions: arrow direction describes value movement, not business meaning.
    """

    if delta is None or abs(delta) < 1e-9:
        return "→"
    if delta > 0:
        return "▲"
    return "▼"


def deterministic_chart_insight(
    items: list[dict[str, Any]],
    *,
    title: str,
    chart_kind: str = "ranking",
    value_key: str = "value",
    label_key: str = "label",
    unit: str | None = None,
) -> str:
    """Build a deterministic executive insight from chart data.

    Inputs: chart items and display metadata.
    Outputs: concise Spanish summary of ranking, variance, or top observation.
    Assumptions: this describes only visible values and does not infer causes.
    """

    numeric_items = [
        (str(item.get(label_key) or "Dato"), number_value(item.get(value_key)))
        for item in items
        if isinstance(item, dict) and number_value(item.get(value_key)) is not None
    ]
    if not numeric_items:
        return f"{title} no tiene datos numéricos suficientes para resumir."
    values = [value for _, value in numeric_items if value is not None]
    labels = [label for label, value in numeric_items if value is not None]
    max_index = max(range(len(values)), key=lambda index: values[index])
    min_index = min(range(len(values)), key=lambda index: values[index])
    spread = values[max_index] - values[min_index]
    if chart_kind == "budget":
        pairs = []
        for budget_label, actual_label in (
            ("Presupuesto ingresos", "Ingresos reales"),
            ("Presupuesto gastos", "Gastos reales"),
        ):
            budget = next((value for label, value in numeric_items if label == budget_label), None)
            actual = next((value for label, value in numeric_items if label == actual_label), None)
            if budget is not None and actual is not None:
                variance = actual - budget
                status = "favorable" if (actual >= budget if "ingresos" in actual_label.lower() else actual <= budget) else "desfavorable"
                pairs.append(
                    f"{actual_label}: {format_value(actual, unit)} vs {format_value(budget, unit)} "
                    f"({format_value(variance, unit)}, {status})"
                )
        return "; ".join(pairs) + "." if pairs else deterministic_chart_insight(items, title=title, unit=unit)
    if chart_kind == "revenue_expense":
        return (
            f"El mayor monto visible es {labels[max_index]} ({format_value(values[max_index], unit)}) "
            f"y el menor es {labels[min_index]} ({format_value(values[min_index], unit)}); "
            f"la diferencia entre ambos es {format_value(spread, unit)}."
        )
    if chart_kind == "department":
        return (
            f"El resultado más alto corresponde a {labels[max_index]} ({format_value(values[max_index], unit)}) "
            f"y el más bajo a {labels[min_index]} ({format_value(values[min_index], unit)}); "
            f"la brecha visible es {format_value(spread, unit)}."
        )
    return (
        f"En {title}, el mayor valor visible es {labels[max_index]} "
        f"({format_value(values[max_index], unit)}) y el menor es {labels[min_index]} "
        f"({format_value(values[min_index], unit)}); la brecha es {format_value(spread, unit)}."
    )


def line_chart_insight(series: dict[str, Any]) -> str:
    """Build a deterministic insight for one trend line.

    Inputs: trend series with period/value points.
    Outputs: concise Spanish trend description.
    Assumptions: trend direction is based only on first and last visible points.
    """

    points = [point for point in series.get("points", []) if isinstance(point, dict)]
    if len(points) < 2:
        return "La serie no contiene suficientes periodos para describir una tendencia."
    first = points[0]
    last = points[-1]
    values = [float(point["value"]) for point in points]
    min_index = min(range(len(points)), key=lambda index: values[index])
    max_index = max(range(len(points)), key=lambda index: values[index])
    unit = str(series.get("unit") or "")
    delta = float(last["value"]) - float(first["value"])
    direction = "al alza" if delta > 0 else ("a la baja" if delta < 0 else "sin cambio visible")
    return (
        f"{sanitize_text(series.get('metric'))} pasa de {format_value(first['value'], unit)} "
        f"en {first.get('period_label', format_period_label(first.get('period')))} a "
        f"{format_value(last['value'], unit)} en {last.get('period_label', format_period_label(last.get('period')))}; "
        f"cambio visible: {format_value(delta, unit)} ({direction}). Mínimo: "
        f"{points[min_index].get('period_label', format_period_label(points[min_index].get('period')))} "
        f"({format_value(points[min_index]['value'], unit)}); máximo: "
        f"{points[max_index].get('period_label', format_period_label(points[max_index].get('period')))} "
        f"({format_value(points[max_index]['value'], unit)})."
    )


def sanitize_items(items: Any, *, limit: int = 8) -> list[str]:
    """Return a bounded list of sanitized text items.

    Inputs: list-like value.
    Outputs: sanitized strings.
    Assumptions: no English-to-Spanish translation is performed.
    """

    raw_items = items if isinstance(items, list) else ([items] if items else [])
    return [sanitize_text(item) for item in raw_items[:limit] if sanitize_text(item)]


def localize_evidence_summary(value: Any) -> str:
    """Localize deterministic retrieval status text for executive reports.

    Inputs: retrieval/evidence summary text from processed artifacts.
    Outputs: Spanish helper text without changing strategic narrative.
    Assumptions: this handles operational labels only, not analytical conclusions.
    """

    raw_text = "" if value is None else str(value).strip()
    if not raw_text:
        return ""
    raw_match = re.fullmatch(r"Retrieved processed ([\w-]+) finance summary\.", raw_text, flags=re.IGNORECASE)
    if raw_match:
        return f"Resumen financiero procesado disponible para {format_period_label(raw_match.group(1))}."
    text = sanitize_text(raw_text)
    match = re.fullmatch(r"Retrieved processed ([\w-]+) finance summary\.", text, flags=re.IGNORECASE)
    if match:
        return f"Resumen financiero procesado disponible para {format_period_label(match.group(1))}."
    fixed_status_labels = {
        "No evidence available.": "No hay evidencia disponible en los artefactos procesados.",
        "Evidence unavailable.": "Evidencia no disponible en los artefactos procesados.",
    }
    return fixed_status_labels.get(text, text)


def build_metric_cards(report_model: dict[str, Any]) -> list[dict[str, Any]]:
    """Build financial health metric cards.

    Inputs: report model.
    Outputs: display-ready card dictionaries.
    Assumptions: values come from processed finance outputs.
    """

    content = get_section(report_model, "financial_health_overview").get("content", {})
    comparison_block = content.get("kpi_comparisons", {}) if isinstance(content, dict) else {}
    comparison_items = comparison_block.get("items", {}) if isinstance(comparison_block, dict) else {}
    trends = {
        str(series.get("metric")): series
        for series in build_historical_presentation(report_model).get("trends", [])
        if isinstance(series, dict)
    }
    keys = (
        "total_revenue",
        "total_expenses",
        "net_operating_result",
        "net_cash_flow",
        "ending_cash",
        "payroll_percentage_of_revenue",
        "collection_rate",
    )
    cards: list[dict[str, Any]] = []
    for key in keys:
        label, unit, description = METRIC_LABELS_ES.get(key, (display_metric_name(key), "", ""))
        numeric = number_value(content.get(key))
        status = "neutral"
        if key in {"net_operating_result", "net_cash_flow"} and numeric is not None:
            status = "good" if numeric >= 0 else "risk"
        if key == "payroll_percentage_of_revenue" and numeric is not None:
            status = "good" if numeric <= 0.42 else "risk"
        if key == "collection_rate" and numeric is not None:
            status = "good" if numeric >= 0.94 else "risk"
        label = label
        trend_series = trends.get(label)
        delta_value: float | None = None
        previous_display = "No disponible"
        if trend_series and len(trend_series.get("points", [])) >= 2:
            points = trend_series["points"]
            previous = number_value(points[-2].get("value"))
            current = number_value(points[-1].get("value"))
            if previous is not None and current is not None:
                delta_value = current - previous
                previous_display = format_value(previous, unit)
        if status == "neutral" and delta_value is not None:
            status = "amber" if abs(delta_value) > 0 else "neutral"
        comparison = comparison_items.get(key, {}) if isinstance(comparison_items, dict) else {}
        if isinstance(comparison, dict) and number_value(comparison.get("previous_value")) is not None:
            previous_numeric = number_value(comparison.get("previous_value"))
            current_numeric = number_value(comparison.get("current_value"))
            previous_display = format_value(previous_numeric, unit)
            if previous_numeric is not None and current_numeric is not None:
                delta_value = current_numeric - previous_numeric
        previous_delta = format_value(delta_value, unit) if delta_value is not None else ""
        percent_change = number_value(comparison.get("percent_change")) if isinstance(comparison, dict) else None
        percentage_point_change = number_value(comparison.get("percentage_point_change")) if isinstance(comparison, dict) else None
        if percent_change is not None:
            previous_delta = f"{previous_delta} ({format_value(percent_change, 'ratio')})"
        elif percentage_point_change is not None:
            previous_delta = f"{previous_delta} ({format_value(percentage_point_change, 'ratio')} p.p.)"
        budget_value = number_value(comparison.get("budget_value")) if isinstance(comparison, dict) else None
        budget_delta_value = number_value(comparison.get("budget_change")) if isinstance(comparison, dict) else None
        budget_delta_pct = number_value(comparison.get("budget_change_pct")) if isinstance(comparison, dict) else None
        budget_delta = format_value(budget_delta_value, unit) if budget_delta_value is not None else ""
        if budget_delta and budget_delta_pct is not None:
            budget_delta = f"{budget_delta} ({format_value(budget_delta_pct, 'ratio')})"
        elif not budget_delta:
            budget_delta = _budget_delta_for_metric(key, report_model)
        comparison_rows = []
        if not is_empty_display_value(previous_display):
            comparison_rows.append(
                {
                    "label": "Periodo anterior",
                    "value": previous_display,
                }
            )
        if previous_delta:
            comparison_rows.append(
                {
                    "label": "Variación respecto al periodo anterior",
                    "value": previous_delta,
                }
            )
        if budget_value is not None:
            comparison_rows.append(
                {
                    "label": "Presupuesto / meta",
                    "value": format_value(budget_value, unit),
                }
            )
        if budget_delta:
            comparison_rows.append(
                {
                    "label": "Variación respecto al presupuesto",
                    "value": budget_delta,
                }
            )
        cards.append(
            {
                "id": key,
                "label": label,
                "value": display_or_unavailable(format_value(content.get(key), unit)),
                "numeric_value": numeric,
                "unit": unit,
                "description": description,
                "status": status,
                "badge": status_badge(status),
                "trend_arrow": trend_arrow(delta_value),
                "previous_value": previous_display,
                "previous_delta": previous_delta,
                "budget_value": format_value(budget_value, unit) if budget_value is not None else "",
                "budget_delta": budget_delta,
                "comparison_rows": comparison_rows,
            }
        )
    return cards


def build_kpi_rows(report_model: dict[str, Any]) -> list[dict[str, str]]:
    """Build localized KPI table rows.

    Inputs: report model.
    Outputs: display-ready KPI rows.
    Assumptions: KPI values are already calculated upstream.
    """

    kpis = get_section(report_model, "kpi_overview").get("content", {}).get("kpis", [])
    rows: list[dict[str, str]] = []
    for item in kpis if isinstance(kpis, list) else []:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "")
        normalized = metric.lower().replace(" ", "_")
        _, default_unit, description = METRIC_LABELS_ES.get(
            metric,
            METRIC_LABELS_ES.get(normalized, (display_metric_name(metric), "", "")),
        )
        rows.append(
            {
                "indicator": display_metric_name(metric),
                "value": display_or_unavailable(format_value(item.get("value"), str(item.get("unit") or default_unit or ""))),
                "status": _localize_status(item.get("availability")),
                "badge": status_badge("good" if str(item.get("availability")).lower() == "available" else "amber"),
                "description": description or sanitize_text(item.get("source") or ""),
            }
        )
    return rows


def build_revenue_expense_summary(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build revenue/expense display rows and chart values.

    Inputs: report model.
    Outputs: display-ready summary dictionary.
    Assumptions: all values are sourced from processed summaries.
    """

    revenue = get_section(report_model, "revenue_analysis").get("content", {})
    expense = get_section(report_model, "expense_analysis").get("content", {})
    keys = (
        "total_revenue",
        "revenue_budget",
        "revenue_variance",
        "revenue_variance_pct",
        "total_expenses",
        "expense_budget",
        "expense_variance",
        "expense_variance_pct",
        "payroll_total",
    )
    rows: list[dict[str, str]] = []
    for key in keys:
        value = revenue.get(key, expense.get(key))
        if value is None:
            continue
        label, unit, description = METRIC_LABELS_ES.get(key, (display_metric_name(key), "USD", ""))
        rows.append({"metric": label, "value": format_value(value, unit), "description": description})
    chart = [
        {"label": "Ingresos", "value": number_value(revenue.get("total_revenue")) or 0.0, "unit": "USD"},
        {"label": "Gastos", "value": number_value(expense.get("total_expenses")) or 0.0, "unit": "USD"},
        {"label": "Resultado", "value": _net_result_value(report_model), "unit": "USD"},
    ]
    budget_chart = [
        {"label": "Presupuesto ingresos", "value": number_value(revenue.get("revenue_budget")) or 0.0, "unit": "USD"},
        {"label": "Ingresos reales", "value": number_value(revenue.get("total_revenue")) or 0.0, "unit": "USD"},
        {"label": "Presupuesto gastos", "value": number_value(expense.get("expense_budget")) or 0.0, "unit": "USD"},
        {"label": "Gastos reales", "value": number_value(expense.get("total_expenses")) or 0.0, "unit": "USD"},
    ]
    return {
        "rows": rows,
        "chart": chart,
        "budget_chart": budget_chart,
        "chart_insight": deterministic_chart_insight(
            chart,
            title="ingresos, gastos y resultado",
            chart_kind="revenue_expense",
            unit="USD",
        ),
        "budget_chart_insight": deterministic_chart_insight(
            budget_chart,
            title="comparación contra presupuesto",
            chart_kind="budget",
            unit="USD",
        ),
    }


def build_department_rows(report_model: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    """Build department comparison rows.

    Inputs: report model and row limit.
    Outputs: display-ready department rows.
    Assumptions: department data is already aggregated upstream.
    """

    items = get_section(report_model, "department_analysis").get("content", {}).get("department_summary", [])
    rows: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        expenses = item.get("actual_expenses", item.get("actual_expense"))
        result = item.get("net_operating_result")
        rows.append(
            {
                "department": display_entity_name(item.get("department") or "Sin departamento"),
                "revenue": display_or_unavailable(format_value(item.get("actual_revenue"), "USD")),
                "expenses": display_or_unavailable(format_value(expenses, "USD")),
                "result": display_or_unavailable(format_value(result, "USD")),
                "variance": format_value(item.get("expense_variance_pct"), "ratio") if item.get("expense_variance_pct") is not None else "No disponible",
                "numeric_result": number_value(result) or 0.0,
                "numeric_expenses": number_value(expenses) or 0.0,
                "numeric_variance": number_value(item.get("expense_variance_pct")),
            }
        )
    ranked = sorted(rows, key=lambda row: abs(float(row["numeric_expenses"])), reverse=True)[:limit]
    if ranked:
        best = max(ranked, key=lambda row: float(row["numeric_result"]))
        worst = min(ranked, key=lambda row: float(row["numeric_result"]))
        for row in ranked:
            row["rank_badge"] = "Mejor" if row is best else ("Mayor presión" if row is worst else "")
            row["variance_class"] = "risk" if (row.get("numeric_variance") or 0.0) > 0 else "good"
    return ranked


def build_anomaly_summary(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build display-ready anomaly summary.

    Inputs: report model.
    Outputs: severity rows, top anomaly rows, or positive status.
    Assumptions: anomaly detection happened upstream.
    """

    content = get_section(report_model, "anomaly_summary").get("content", {})
    severity = content.get("anomalies_by_severity", {})
    severity = severity if isinstance(severity, dict) else {}
    severity_rows = [
        {"severity": SEVERITY_LABELS_ES.get(str(key).lower(), str(key)), "count": int(value or 0)}
        for key, value in severity.items()
    ]
    top_rows: list[dict[str, str]] = []
    for item in (content.get("top_anomalies", []) or [])[:8]:
        if isinstance(item, dict):
            top_rows.append(
                {
                    "title": sanitize_text(item.get("title") or item.get("description") or "Anomalía detectada"),
                    "severity": SEVERITY_LABELS_ES.get(str(item.get("severity", "")).lower(), str(item.get("severity", ""))),
                    "severity_class": str(item.get("severity", "")).lower() or "info",
                    "recurrence": "Recurrente" if item.get("recurrence_count") or item.get("periods") else "Periodo actual",
                    "period_chips": [str(period) for period in (item.get("periods") or [])[:6]],
                    "evidence": sanitize_text(item.get("evidence") or item.get("description") or ""),
                }
            )
    if not top_rows and (not severity_rows or not any(row["count"] for row in severity_rows)):
        return {
            "positive_status": "",
            "current_period_status": f"No se detectaron anomalías en {format_period_label(content.get('report_period') or get_section(report_model, 'cover').get('content', {}).get('report_period') or report_model.get('report_period'))}.",
            "severity_rows": [],
            "top_rows": [],
        }
    severity_chart = [
        {"label": row["severity"], "value": row["count"], "unit": "count"}
        for row in severity_rows
    ]
    return {
        "positive_status": "",
        "severity_rows": severity_rows,
        "top_rows": top_rows,
        "severity_chart": severity_chart,
        "chart_insight": deterministic_chart_insight(
            severity_chart,
            title="anomalías por severidad",
            chart_kind="ranking",
            unit="count",
        ),
    }


def build_evidence_summary(report_model: dict[str, Any], *, limit: int = 8) -> list[dict[str, str]]:
    """Build concise evidence rows without internal task/tool identifiers.

    Inputs: report model and row limit.
    Outputs: display-ready evidence rows.
    Assumptions: detailed evidence remains in source artifacts.
    """

    items = get_section(report_model, "investigation_evidence").get("content", {}).get("evidence_items", [])
    rows: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "priority": PRIORITY_LABELS_ES.get(str(item.get("priority", "")).lower(), str(item.get("priority", ""))),
                "evidence": TOOL_LABELS_ES.get(str(item.get("retrieval_name")), "evidencia procesada"),
                "records": str(item.get("record_count") or "No disponible"),
                "summary": localize_evidence_summary(item.get("evidence_summary") or ""),
            }
        )
    return rows[:limit]


def build_recommendation_cards(report_model: dict[str, Any]) -> list[dict[str, str]]:
    """Build executive recommendation cards.

    Inputs: report model.
    Outputs: display-ready recommendation cards.
    Assumptions: recommendation text is already Spanish from Step 9.
    """

    content = get_section(report_model, "strategic_recommendations").get("content", {})
    cards: list[dict[str, str]] = []
    for item in content.get("recommendations", []) or []:
        if isinstance(item, dict):
            cards.append(
                {
                    "priority": PRIORITY_LABELS_ES.get(str(item.get("priority", "")).lower(), str(item.get("priority", ""))) or "Media",
                    "action": sanitize_text(item.get("action") or item.get("recommendation") or ""),
                    "rationale": sanitize_text(item.get("rationale") or item.get("supporting_evidence") or ""),
                    "expected_impact": sanitize_text(item.get("expected_impact") or ""),
                    "owner": sanitize_text(item.get("owner") or "Responsable por asignar"),
                    "status": sanitize_text(item.get("status") or "Planificada"),
                    "owner_status": sanitize_text(
                        item.get("owner_status")
                        or f"{item.get('owner') or 'Responsable por asignar'} / {item.get('status') or 'Planificada'}"
                    ),
                }
            )
        elif item:
            cards.append(
                {
                    "priority": "Media",
                    "action": sanitize_text(item),
                    "rationale": "",
                    "expected_impact": "",
                    "owner": "Responsable por asignar",
                    "status": "Planificada",
                    "owner_status": "Responsable por asignar / Planificada",
                }
            )
    return cards


def build_historical_presentation(report_model: dict[str, Any]) -> dict[str, Any]:
    """Convert compact historical data into readable exhibits.

    Inputs: report model.
    Outputs: trend series, recurring risks, follow-up rows, and narratives.
    Assumptions: no raw historical reports are included.
    """

    clean = _clean_historical_sections(report_model)
    if clean["available"]:
        context = _historical_context(report_model)
        context_risks = _recurring_risk_rows(context)
        context_follow_up = _recommendation_follow_up_rows(context)
        # Older report models may contain sparse, database-shaped rows while
        # still carrying the compact historical context needed to build an
        # executive view. Prefer the richer deterministic mapping when present.
        if context_risks and _risk_rows_are_generic(clean.get("recurring_risks", [])):
            clean["recurring_risks"] = context_risks[:8]
            clean["risk_summary"] = _risk_summary(context_risks)
        if context_follow_up and _follow_up_rows_are_sparse(clean.get("recommendation_follow_up", [])):
            clean["recommendation_follow_up"] = context_follow_up[:8]
            clean["recommendation_intro"] = RECOMMENDATION_FOLLOW_UP_INTRO
            clean["recommendation_summary"] = _recommendation_follow_up_summary(context_follow_up)
        return clean
    context = _historical_context(report_model)
    derived = context.get("derived_context", {}) if isinstance(context, dict) else {}
    kpi_trends = derived.get("kpi_trends", []) if isinstance(derived, dict) else []
    if isinstance(kpi_trends, dict):
        kpi_trends = _trend_items_from_retrievals(context, kpi_trends)
    trends = [_trend_series(item) for item in kpi_trends if isinstance(item, dict)]
    trends = [item for item in trends if item["points"]]
    risks = _recurring_risk_rows(context)
    follow_up = _recommendation_follow_up_rows(context)
    return {
        "available": bool(trends or risks or follow_up),
        "narrative": [],
        "trends": trends[:4],
        "recurring_risks": risks[:8],
        "risk_summary": _risk_summary(risks),
        "recommendation_intro": RECOMMENDATION_FOLLOW_UP_INTRO,
        "recommendation_summary": _recommendation_follow_up_summary(follow_up),
        "recommendation_follow_up": follow_up[:8],
        "longitudinal_conclusions": _longitudinal_conclusions(trends, risks, follow_up),
    }


def build_missing_information(report_model: dict[str, Any]) -> list[str]:
    """Build missing-information display items.

    Inputs: report model.
    Outputs: sanitized missing-information strings or a positive status.
    Assumptions: false missing-info filtering happens upstream.
    """

    content = get_section(report_model, "missing_information").get("content", {})
    items = sanitize_items(content.get("missing_information"), limit=8)
    return items


def build_appendix(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build methodology and compact source notes.

    Inputs: report model.
    Outputs: appendix data for renderers.
    Assumptions: full paths remain in machine-readable artifacts only.
    """

    sources = [compact_source_label(source) for source in report_model.get("source_references", [])]
    return {
        "methodology": [
            "Los cálculos financieros, KPIs y anomalías fueron generados por reglas determinísticas de Python.",
            "Ollama se utiliza únicamente para interpretación estratégica sobre evidencia ya calculada y validada.",
            "Las cifras se muestran redondeadas para lectura ejecutiva; los artefactos JSON/CSV conservan los valores auditables.",
        ],
        "validation": "Análisis estratégico aceptado; cálculos y anomalías provienen de salidas procesadas.",
        "sources": list(dict.fromkeys(source for source in sources if source)),
    }


def build_presentation_view(report_model: dict[str, Any], *, mode: str = "executive") -> dict[str, Any]:
    """Build the renderer-facing presentation view.

    Inputs: report model and mode.
    Outputs: display-ready dictionary shared by HTML and PDF renderers.
    Assumptions: executive mode hides implementation details.
    """

    if mode not in {"executive", "technical"}:
        raise ValueError("Report rendering mode must be 'executive' or 'technical'.")
    executive = get_section(report_model, "executive_summary").get("content", {})
    recommendations = get_section(report_model, "strategic_recommendations").get("content", {})
    view = {
        "mode": mode,
        "report_id": report_model.get("report_id"),
        "period_slug": report_model.get("period_slug"),
        "period": format_period_label(report_model.get("report_period")),
        "title": "Reporte financiero ejecutivo",
        "organization": "Universidad / Institución",
        "sections": EXECUTIVE_SECTION_ORDER,
        "labels": SECTION_LABELS_ES,
        "templates": section_templates_payload(),
        "section_narratives": section_narratives(report_model),
        "executive_summary": {
            "summary": sanitize_text(executive.get("summary") or ""),
            "key_findings": sanitize_items(executive.get("key_findings"), limit=6),
            "root_causes": sanitize_items(executive.get("root_causes"), limit=6),
            "confidence": format_value(executive.get("confidence"), "ratio"),
            "analysis_status": executive.get("analysis_status"),
        },
        "financial_health": {
            "cards": build_metric_cards(report_model),
            "sources": source_labels(get_section(report_model, "financial_health_overview")),
        },
        "kpis": build_kpi_rows(report_model),
        "revenue_expense": build_revenue_expense_summary(report_model),
        "departments": build_department_rows(report_model),
        "anomalies": build_anomaly_summary(report_model),
        "evidence": build_evidence_summary(report_model),
        "historical": build_historical_presentation(report_model),
        "recommendations": {
            "priorities": sanitize_items(recommendations.get("strategic_priorities"), limit=6),
            "reasoning_summary": sanitize_text(recommendations.get("reasoning_summary") or ""),
            "cards": build_recommendation_cards(report_model),
        },
        "missing_information": build_missing_information(report_model),
        "appendix": build_appendix(report_model),
    }
    view["anomalies"]["historical_risks_present"] = bool(
        view.get("historical", {}).get("recurring_risks")
    )
    if view["anomalies"].get("current_period_status") and view["anomalies"]["historical_risks_present"]:
        view["anomalies"]["distinction_note"] = (
            f"{view['anomalies']['current_period_status']} Sin embargo, permanecen riesgos "
            "recurrentes identificados en períodos anteriores."
        )
    health_chart = [
        {"label": card["label"], "value": card["numeric_value"] or 0.0, "unit": card["unit"]}
        for card in view["financial_health"]["cards"]
        if card["id"] in {"total_revenue", "total_expenses", "net_operating_result", "net_cash_flow"}
    ]
    view["financial_health"]["chart_insight"] = deterministic_chart_insight(
        health_chart,
        title="salud financiera principal",
        chart_kind="revenue_expense",
        unit="USD",
    )
    if mode == "technical":
        view["technical_sources"] = report_model.get("source_references", [])
    validate_presentation_view(view, mode=mode)
    return view


def validate_presentation_view(view: dict[str, Any], *, mode: str = "executive") -> PresentationValidationResult:
    """Validate the executive presentation view.

    Inputs: presentation view and mode.
    Outputs: validation result.
    Assumptions: validation checks leaks, not language translation.
    """

    errors: list[str] = []
    text = "\n".join(_visible_strings(view))
    if mode == "executive":
        for pattern in RAW_TEXT_PATTERNS:
            if pattern.search(text):
                errors.append(f"Executive presentation contains raw/internal pattern: {pattern.pattern}")
        for identifier in CANONICAL_IDENTIFIERS:
            if re.search(rf"\b{re.escape(identifier)}\b", text):
                errors.append(f"Executive presentation exposes canonical identifier: {identifier}")
        analysis_status = view.get("executive_summary", {}).get("analysis_status")
        if (
            not view.get("recommendations", {}).get("cards")
            and analysis_status in {"accepted", "sanitized"}
        ):
            errors.append("Executive presentation is missing strategic recommendation cards.")
        if not view.get("executive_summary", {}).get("summary"):
            errors.append("Executive presentation is missing the executive summary.")
    return PresentationValidationResult(not errors, tuple(errors), ())


def _visible_strings(value: Any) -> list[str]:
    """Collect user-facing string values from a presentation view.

    Inputs: nested presentation value.
    Outputs: visible string leaves.
    Assumptions: internal IDs and units are not rendered as prose.
    """

    strings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"id", "unit", "mode", "period_slug", "report_id", "labels", "sections"}:
                continue
            strings.extend(_visible_strings(child))
    elif isinstance(value, list):
        for child in value:
            strings.extend(_visible_strings(child))
    elif isinstance(value, str):
        strings.append(value)
    return strings


def _localize_status(value: Any) -> str:
    """Translate simple availability/status labels.

    Inputs: raw status value.
    Outputs: Spanish status label.
    Assumptions: unknown statuses remain readable.
    """

    mapping = {"available": "Disponible", "unavailable": "No disponible", "planned": "Planificado"}
    return mapping.get(str(value or "").lower(), str(value or "No disponible"))


def _net_result_value(report_model: dict[str, Any]) -> float:
    """Return net operating result for charts.

    Inputs: report model.
    Outputs: numeric result or 0.
    Assumptions: value is copied from processed output.
    """

    content = get_section(report_model, "financial_health_overview").get("content", {})
    return number_value(content.get("net_operating_result")) or 0.0


def _budget_delta_for_metric(metric_id: str, report_model: dict[str, Any]) -> str:
    """Return a display-ready budget delta for a KPI card when available.

    Inputs: metric identifier and report model.
    Outputs: formatted budget delta or readable unavailable label.
    Assumptions: delta values are copied from processed report sections.
    """

    revenue = get_section(report_model, "revenue_analysis").get("content", {})
    expense = get_section(report_model, "expense_analysis").get("content", {})
    mapping = {
        "total_revenue": (revenue.get("revenue_variance"), "USD"),
        "total_expenses": (expense.get("expense_variance"), "USD"),
        "net_operating_result": (_net_result_value(report_model), "USD"),
        "payroll_percentage_of_revenue": (expense.get("payroll_variance_pct"), "ratio"),
        "collection_rate": (revenue.get("collection_variance_pct"), "ratio"),
    }
    value, unit = mapping.get(metric_id, (None, None))
    return format_value(value, unit) if value is not None else ""


def _historical_context(report_model: dict[str, Any]) -> dict[str, Any]:
    """Return embedded historical context when present.

    Inputs: report model.
    Outputs: compact historical context dictionary.
    Assumptions: backward compatibility may expose older historical sections.
    """

    for section_id in ("historical_summary", "historical_trends", "recommendation_follow_up"):
        content = get_section(report_model, section_id).get("content", {})
        if isinstance(content, dict) and isinstance(content.get("historical_context"), dict):
            return content["historical_context"]
    content = get_section(report_model, "historical_trends").get("content", {})
    return {"retrievals": content.get("metric_trends", []) if isinstance(content, dict) else []}


def _trend_series(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one KPI trend into chart-ready points.

    Inputs: trend dictionary.
    Outputs: display-ready series.
    Assumptions: values are processed historical KPI records.
    """

    metric = str(item.get("metric") or "")
    label, unit, _ = METRIC_LABELS_ES.get(metric, (display_metric_name(metric), "", ""))
    points = []
    for point in item.get("points", []) or []:
        if isinstance(point, dict):
            numeric = number_value(point.get("value"))
            if numeric is not None:
                period = str(point.get("period") or "")
                points.append(
                    {
                        "period": period,
                        "period_label": format_period_label(period),
                        "value": numeric,
                        "display": format_value(numeric, unit),
                    }
                )
    direction = str(item.get("direction") or "stable")
    if points and metric == "payroll_percentage_of_revenue":
        direction = "improving" if points[-1]["value"] <= points[0]["value"] else "worsening"
    if points and metric == "student_payment_collection_rate":
        direction = "improving" if points[-1]["value"] >= points[0]["value"] else "worsening"
    if points and metric == "net_cash_flow":
        direction = "improving" if points[-1]["value"] >= points[0]["value"] else "worsening"
    series = {
        "metric": label,
        "unit": unit,
        "direction": direction,
        "direction_label": _localize_direction(direction),
        "points": points,
    }
    series["insight"] = line_chart_insight(series)
    return series


def _trend_items_from_retrievals(context: dict[str, Any], trend_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct trend items from compact retrieval records.

    Inputs: historical context and derived trend summary.
    Outputs: trend item dictionaries.
    Assumptions: retrieval records contain processed KPI values only.
    """

    items: list[dict[str, Any]] = []
    retrievals = context.get("retrievals", []) if isinstance(context, dict) else []
    for retrieval in retrievals if isinstance(retrievals, list) else []:
        if not isinstance(retrieval, dict) or retrieval.get("tool_name") != "get_metric_history":
            continue
        metric = str(retrieval.get("metric") or retrieval.get("arguments", {}).get("metric") or "")
        records = retrieval.get("records", [])
        if metric and isinstance(records, list):
            items.append(
                {
                    "metric": metric,
                    "direction": (trend_summary.get(metric, {}) or {}).get("direction", "stable"),
                    "points": [
                        {"period": record.get("period"), "value": record.get("value")}
                        for record in records
                        if isinstance(record, dict)
                    ],
                }
            )
    return items


def _trend_narrative(series: dict[str, Any]) -> str:
    """Return no deterministic analytical trend narrative.

    Inputs: trend series.
    Outputs: empty string.
    Assumptions: section analysis must come from validated Step 9 narrative.
    """

    del series
    return ""


def _recurring_risk_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build recurring-risk rows from historical context.

    Inputs: historical context.
    Outputs: display-ready recurring-risk rows.
    Assumptions: pattern names are sanitized, not translated by dictionary.
    """

    derived = context.get("derived_context", {}) if isinstance(context, dict) else {}
    current_period = str(context.get("current_period") or "")
    repeated = _repeated_anomaly_records(context)
    severity_by_type = {
        str(item.get("type") or item.get("anomaly_type") or ""): str(item.get("latest_severity") or "")
        for item in repeated
        if isinstance(item, dict)
    }
    patterns = derived.get("artifact_anomaly_patterns", []) if isinstance(derived, dict) else []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in patterns if isinstance(patterns, list) else []:
        if not isinstance(item, dict):
            continue
        risk_type = str(item.get("anomaly_type") or item.get("type") or item.get("pattern") or "")
        department = str(item.get("department") or "University")
        key = (risk_type, department)
        periods = [str(period) for period in item.get("periods", []) if period]
        existing = grouped.setdefault(
            key,
            {
                "risk_type": risk_type,
                "department": department,
                "periods": [],
                "occurrences": 0,
                "severity": severity_by_type.get(risk_type, ""),
            },
        )
        # Merge only when deterministic metadata proves the same risk type and
        # department. Different risks in the same department intentionally stay separate.
        existing["periods"] = sorted(set(existing["periods"]) | set(periods))
        existing["occurrences"] = max(int(existing.get("occurrences") or 0), int(item.get("occurrences") or len(periods)))
    rows = []
    for item in grouped.values():
        periods = list(item.get("periods", []))
        severity = item.get("severity") or _severity_from_risk_type(item.get("risk_type"))
        rows.append(
            {
                "risk": display_risk_name(item.get("risk_type")),
                "department": display_entity_name(item.get("department") or "University"),
                "occurrences": str(item.get("occurrences") or len(periods)),
                "frequency": f"{item.get('occurrences') or len(periods)} períodos",
                "periods": ", ".join(format_period_label(period) for period in periods[:8]),
                "status": _recurring_risk_status(periods, current_period),
                "what_happened": _risk_what_happened(item.get("risk_type"), periods),
                "recurrence_reason": _risk_recurrence_reason(item.get("occurrences") or len(periods), periods),
                "recurrence_direction": _risk_recurrence_direction(periods, current_period),
                "management_relevance": _risk_management_relevance(item.get("risk_type")),
                "severity": _localize_severity(severity),
                "severity_class": str(severity or "medium").lower(),
                "sort_score": _risk_sort_score(severity, int(item.get("occurrences") or len(periods))),
            }
        )
    return sorted(rows, key=lambda row: int(row.get("sort_score") or 0), reverse=True)


def _recommendation_follow_up_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build previous-recommendation follow-up rows.

    Inputs: historical context.
    Outputs: display-ready follow-up rows.
    Assumptions: topic labels are short fixed Spanish status labels.
    """

    derived = context.get("derived_context", {}) if isinstance(context, dict) else {}
    effectiveness = derived.get("recommendation_effectiveness", []) if isinstance(derived, dict) else []
    previous = _previous_recommendation_records(context)
    rows: list[dict[str, str]] = []
    for item in effectiveness if isinstance(effectiveness, list) else []:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "Recomendación previa")
        records = sorted(
            [record for record in previous if _recommendation_topic(record) == topic],
            key=lambda record: str(record.get("run_period") or ""),
        )
        periods = sorted(str(record.get("run_period") or "") for record in records if record.get("run_period"))
        latest = records[-1] if records else {}
        trend = item.get("related_trend") if isinstance(item.get("related_trend"), dict) else None
        progress = _recommendation_progress(topic, trend)
        rows.append(
            {
                "recommendation": RECOMMENDATION_TOPIC_LABELS_ES.get(topic, sanitize_text(topic.replace("_", " ").capitalize())),
                "issued_period": display_or_unavailable(format_period_label(periods[0] if periods else "")),
                "current_evidence": _trend_evidence(topic, trend),
                "status": progress,
                "progress": progress,
                "status_reason": _recommendation_status_reason(topic, progress, trend),
                "objective": _recommendation_objective(topic, latest),
                "next_action": _recommendation_next_action(topic, progress),
            }
        )
    return rows


def _clean_historical_sections(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build historical presentation from already-clean report sections.

    Inputs: report model.
    Outputs: historical presentation dictionary.
    Assumptions: newer report models store presentation-ready historical rows.
    """

    trends_content = get_section(report_model, "historical_trends").get("content", {})
    follow_content = get_section(report_model, "recommendation_follow_up").get("content", {})
    risk_content = get_section(report_model, "longitudinal_risk_assessment").get("content", {})
    trends = trends_content.get("trend_series", []) if isinstance(trends_content, dict) else []
    follow_up = follow_content.get("follow_up", []) if isinstance(follow_content, dict) else []
    follow_intro = follow_content.get("intro", "") if isinstance(follow_content, dict) else ""
    follow_summary = follow_content.get("summary", "") if isinstance(follow_content, dict) else ""
    risks = risk_content.get("recurring_risks", []) if isinstance(risk_content, dict) else []
    risk_summary = risk_content.get("risk_summary", "") if isinstance(risk_content, dict) else ""
    conclusions = risk_content.get("conclusions", []) if isinstance(risk_content, dict) else []
    narrative = trends_content.get("narrative", []) if isinstance(trends_content, dict) else []
    if not any((trends, follow_up, risks, narrative, conclusions)):
        return {
            "available": False,
            "narrative": [],
            "trends": [],
            "recurring_risks": [],
            "risk_summary": "",
            "recommendation_intro": "",
            "recommendation_summary": "",
            "recommendation_follow_up": [],
            "longitudinal_conclusions": [],
        }
    normalized_trends = []
    for series in trends if isinstance(trends, list) else []:
        if not isinstance(series, dict):
            continue
        series = {
                "metric": sanitize_text(series.get("metric") or ""),
                "unit": str(series.get("unit") or ""),
                "direction": str(series.get("direction") or "stable"),
                "direction_label": _localize_direction(series.get("direction") or "stable"),
                "points": [
                    {
                        "period": str(point.get("period") or ""),
                        "period_label": format_period_label(point.get("period")),
                        "value": number_value(point.get("value")) or 0.0,
                        "display": str(point.get("display") or format_value(point.get("value"), series.get("unit"))),
                    }
                    for point in series.get("points", []) or []
                    if isinstance(point, dict) and number_value(point.get("value")) is not None
                ],
            }
        series["insight"] = line_chart_insight(series)
        normalized_trends.append(series)
    normalized_risks = _normalize_recurring_risk_rows(risks)
    return {
        "available": True,
        "narrative": sanitize_items(narrative),
        "trends": normalized_trends,
        "recurring_risks": normalized_risks,
        "risk_summary": sanitize_text(risk_summary) or _risk_summary(normalized_risks),
        "recommendation_intro": sanitize_text(follow_intro) or (RECOMMENDATION_FOLLOW_UP_INTRO if follow_up else ""),
        "recommendation_summary": sanitize_text(follow_summary) or _recommendation_follow_up_summary(follow_up),
        "recommendation_follow_up": [
            {
                "recommendation": sanitize_text(item.get("recommendation") or ""),
                "issued_period": display_or_unavailable(format_period_label(item.get("issued_period"))),
                "current_evidence": sanitize_text(item.get("current_evidence") or ""),
                "status": sanitize_text(item.get("status") or "En seguimiento"),
                "progress": sanitize_text(item.get("progress") or item.get("status") or "En seguimiento"),
                "objective": sanitize_text(item.get("objective") or ""),
                "next_action": sanitize_text(item.get("next_action") or ""),
                "status_reason": sanitize_text(item.get("status_reason") or ""),
            }
            for item in follow_up
            if isinstance(item, dict)
        ],
        "longitudinal_conclusions": sanitize_items(conclusions),
    }


def _longitudinal_conclusions(trends: list[dict[str, Any]], risks: list[dict[str, str]], follow_up: list[dict[str, str]]) -> list[str]:
    """Build concise longitudinal-risk conclusions.

    Inputs: trend series, recurring risks, and follow-up rows.
    Outputs: Spanish conclusions.
    Assumptions: conclusions summarize deterministic context only.
    """

    del trends, risks, follow_up
    return []


def _normalize_recurring_risk_rows(risks: Any) -> list[dict[str, str]]:
    """Normalize report-model recurring risk rows for renderers.

    Inputs: recurring-risk rows from the report model.
    Outputs: display-ready rows with Spanish labels and status fields.
    Assumptions: rows were built from deterministic historical metadata.
    """

    rows = []
    for item in risks if isinstance(risks, list) else []:
        if not isinstance(item, dict):
            continue
        periods = _format_period_list(item.get("periods"))
        occurrences = item.get("occurrences") or item.get("frequency") or ""
        severity = str(item.get("severity_class") or item.get("severity") or "").lower()
        occurrence_text = str(occurrences).replace(" períodos", "").strip()
        occurrence_count = int(occurrence_text.split()[0]) if occurrence_text.split() and occurrence_text.split()[0].isdigit() else 0
        rows.append(
            {
                "risk": display_risk_name(item.get("risk_type") or item.get("risk") or "", item.get("metric")),
                "department": display_entity_name(item.get("department") or "University"),
                "occurrences": occurrence_text,
                "frequency": sanitize_text(item.get("frequency") or f"{occurrence_text} períodos").strip(),
                "periods": periods,
                "status": sanitize_text(item.get("status") or "En seguimiento"),
                "what_happened": sanitize_text(item.get("what_happened") or _risk_what_happened(item.get("risk_type") or item.get("risk"), periods)),
                "recurrence_reason": sanitize_text(item.get("recurrence_reason") or _risk_recurrence_reason(occurrence_count, periods.split(",") if periods else [])),
                "recurrence_direction": sanitize_text(item.get("recurrence_direction") or item.get("status") or "En seguimiento"),
                "management_relevance": sanitize_text(item.get("management_relevance") or _risk_management_relevance(item.get("risk_type") or item.get("risk"))),
                "severity": _localize_severity(severity or item.get("severity")),
                "severity_class": severity or "medium",
                "sort_score": _risk_sort_score(severity, occurrence_count),
            }
        )
    return sorted(rows, key=lambda row: int(row.get("sort_score") or 0), reverse=True)


def _risk_rows_are_generic(rows: Any) -> bool:
    """Return whether recurring-risk rows lack specific risk names.

    Inputs: presentation risk rows.
    Outputs: boolean.
    Assumptions: generic labels are safe to replace from deterministic context.
    """

    if not isinstance(rows, list) or not rows:
        return True
    return any(str(row.get("risk") or "").strip().lower() in {"riesgo recurrente", "recurrent risk"} for row in rows if isinstance(row, dict))


def _follow_up_rows_are_sparse(rows: Any) -> bool:
    """Return whether follow-up rows lack executive evidence fields.

    Inputs: presentation recommendation rows.
    Outputs: boolean.
    Assumptions: sparse rows are upgraded only from deterministic historical context.
    """

    if not isinstance(rows, list) or not rows:
        return True
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("current_evidence") or not row.get("objective") or not row.get("next_action"):
            return True
    return False


def _risk_summary(rows: list[dict[str, Any]]) -> str:
    """Build a deterministic executive summary for recurring risks.

    Inputs: display-ready recurring-risk rows.
    Outputs: concise Spanish summary.
    Assumptions: the summary describes row metadata only; it does not infer causes.
    """

    if not rows:
        return ""
    ordered = sorted(rows, key=lambda row: int(row.get("sort_score") or 0), reverse=True)
    highest = ordered[0]
    active_count = sum(1 for row in rows if str(row.get("status") or "").lower().startswith("activo"))
    directions = {str(row.get("recurrence_direction") or row.get("status") or "").lower() for row in rows}
    if any("activa" in direction or "estable" in direction for direction in directions):
        recurrence = "activa o estable"
    elif any("disminución" in direction or "remisión" in direction for direction in directions):
        recurrence = "en disminución"
    else:
        recurrence = "sin tendencia verificable"
    return (
        f"Se identifican {len(rows)} riesgos históricos recurrentes; {active_count} muestran actividad reciente. "
        f"El riesgo de mayor prioridad es {highest.get('risk')} en {highest.get('department')}. "
        f"La recurrencia agregada está {recurrence} según los períodos afectados registrados."
    )


def _risk_what_happened(risk_type: Any, periods: list[str] | str) -> str:
    """Describe what happened for a recurring risk.

    Inputs: deterministic risk type and affected periods.
    Outputs: concise Spanish description.
    Assumptions: wording is selected from known deterministic risk categories.
    """

    risk_name = display_risk_name(risk_type)
    period_text = _format_period_list(periods)
    suffix = f" en {period_text}" if period_text else ""
    return f"{risk_name} apareció{suffix}."


def _risk_recurrence_reason(occurrences: Any, periods: list[str] | str) -> str:
    """Explain why the risk is considered recurring.

    Inputs: occurrence count and affected periods.
    Outputs: deterministic recurrence explanation.
    Assumptions: recurrence means the issue appeared in more than one period.
    """

    count = int(occurrences or 0) if str(occurrences or "").isdigit() else len(_period_values(periods))
    if count <= 1:
        return "Se muestra como antecedente histórico porque existe un registro previo documentado."
    return f"Se considera recurrente porque aparece en {count} períodos distintos."


def _risk_recurrence_direction(periods: list[str] | str, current_period: str) -> str:
    """Classify recurrence movement from affected-period recency.

    Inputs: affected periods and current period.
    Outputs: Spanish recurrence movement label.
    Assumptions: presentation uses recency only and does not infer causality.
    """

    values = _period_values(periods)
    period_indices = [_period_index(period) for period in values]
    period_indices = [index for index in period_indices if index is not None]
    current = _period_index(current_period)
    if not period_indices or current is None:
        return "Sin tendencia verificable"
    latest_gap = current - max(period_indices)
    if latest_gap <= 1:
        return "Estable o activa"
    if latest_gap <= 3:
        return "En disminución"
    return "En remisión"


def _risk_management_relevance(risk_type: Any) -> str:
    """Explain why management should care about a deterministic risk type.

    Inputs: deterministic risk type.
    Outputs: Spanish business relevance statement.
    Assumptions: statements describe operational exposure, not root causes.
    """

    key = str(risk_type or "").lower()
    if "vendor" in key or "proveedor" in key:
        return "Importa porque puede debilitar controles de pago, aprobación y soporte documental."
    if "cash" in key or "caja" in key:
        return "Importa porque reduce liquidez disponible para cubrir compromisos operativos."
    if "payroll" in key or "nómina" in key or "nomina" in key or "overtime" in key:
        return "Importa porque presiona el margen operativo y limita flexibilidad presupuestaria."
    if "collection" in key or "cobranza" in key or "overdue" in key:
        return "Importa porque retrasa entradas de efectivo y aumenta presión sobre caja."
    if "budget" in key or "presupuesto" in key or "overspend" in key:
        return "Importa porque señala disciplina presupuestaria débil en gastos recurrentes."
    return "Importa porque representa una exposición repetida que requiere seguimiento directivo."


def _format_period_list(value: Any) -> str:
    """Format a period list or comma-separated value for Spanish display.

    Inputs: list-like or string period value.
    Outputs: comma-separated Spanish period labels.
    Assumptions: invalid fragments are sanitized rather than interpreted.
    """

    if isinstance(value, list):
        periods = [str(item) for item in value if item]
    else:
        periods = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return ", ".join(format_period_label(period) for period in periods)


def _period_values(value: Any) -> list[str]:
    """Normalize period values without changing their chronology.

    Inputs: list-like or comma-separated periods.
    Outputs: period identifiers or already-formatted labels.
    Assumptions: callers use this only for display/status summaries.
    """

    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _repeated_anomaly_records(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Return repeated-anomaly records from compact historical retrievals.

    Inputs: historical context dictionary.
    Outputs: repeated anomaly records.
    Assumptions: retrievals are read-only deterministic memory outputs.
    """

    records: list[dict[str, Any]] = []
    retrievals = context.get("retrievals", []) if isinstance(context, dict) else []
    for retrieval in retrievals if isinstance(retrievals, list) else []:
        if isinstance(retrieval, dict) and retrieval.get("tool_name") == "get_repeated_anomalies":
            values = retrieval.get("records", [])
            records.extend(item for item in values if isinstance(item, dict))
    return records


def _previous_recommendation_records(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Return previous recommendation records from compact historical retrievals.

    Inputs: historical context dictionary.
    Outputs: recommendation records ordered by source retrieval.
    Assumptions: no SQL or new retrieval is performed here.
    """

    records: list[dict[str, Any]] = []
    retrievals = context.get("retrievals", []) if isinstance(context, dict) else []
    for retrieval in retrievals if isinstance(retrievals, list) else []:
        if isinstance(retrieval, dict) and retrieval.get("tool_name") == "get_previous_recommendations":
            values = retrieval.get("records", [])
            records.extend(item for item in values if isinstance(item, dict))
    return records


def _severity_from_risk_type(risk_type: Any) -> str:
    """Infer a conservative severity class from deterministic risk type.

    Inputs: anomaly risk type.
    Outputs: severity string.
    Assumptions: used only when source severity is unavailable.
    """

    text = str(risk_type or "").lower()
    if "cash" in text or "vendor" in text:
        return "high"
    if "overdue" in text:
        return "critical"
    return "medium"


def _localize_severity(value: Any) -> str:
    """Return a Spanish severity label.

    Inputs: severity identifier.
    Outputs: Spanish label.
    Assumptions: unknown severities are display-sanitized.
    """

    return SEVERITY_LABELS_ES.get(str(value or "").lower(), sanitize_text(value or "Media"))


def _risk_sort_score(severity: Any, occurrences: int) -> int:
    """Return deterministic sort score for recurring risks.

    Inputs: severity and occurrence count.
    Outputs: integer score.
    Assumptions: sorting improves presentation only.
    """

    severity_score = {"critical": 400, "high": 300, "medium": 200, "low": 100}.get(
        str(severity or "").lower(),
        200,
    )
    return severity_score + occurrences


def _period_index(period: Any) -> int | None:
    """Convert a monthly period to a sortable month index.

    Inputs: period identifier.
    Outputs: year * 12 + month, or None.
    Assumptions: only monthly identifiers can support recurrence status.
    """

    match = re.fullmatch(r"(20\d{2})[-_](0[1-9]|1[0-2])", str(period or ""))
    if not match:
        return None
    year, month = match.groups()
    return int(year) * 12 + int(month)


def _recurring_risk_status(periods: list[str], current_period: str) -> str:
    """Return deterministic recurrence status.

    Inputs: affected periods and current period.
    Outputs: Spanish status label.
    Assumptions: recent means affected within the prior two monthly periods.
    """

    period_indices = [_period_index(period) for period in periods]
    period_indices = [index for index in period_indices if index is not None]
    current = _period_index(current_period)
    if not period_indices or current is None:
        return "En seguimiento"
    latest_gap = current - max(period_indices)
    if latest_gap <= 2:
        return "Activo reciente"
    return "En remisión"


def _recommendation_topic(record: dict[str, Any]) -> str:
    """Classify a recommendation into a deterministic follow-up topic.

    Inputs: previous recommendation record.
    Outputs: topic identifier.
    Assumptions: keyword matching is presentation grouping, not reasoning.
    """

    text = f"{record.get('action', '')} {record.get('expected_impact', '')}".lower()
    if any(word in text for word in ("payroll", "nómina", "overtime", "benefits")):
        return "payroll_overtime"
    if any(word in text for word in ("collection", "cobranza", "overdue", "payment tracking")):
        return "collections"
    if any(word in text for word in ("vendor", "proveedor", "invoice", "approval")):
        return "vendor_controls"
    return "other"


def _recommendation_progress(topic: str, trend: dict[str, Any] | None) -> str:
    """Calculate deterministic recommendation progress.

    Inputs: recommendation topic and related trend summary.
    Outputs: one of the allowed Spanish progress labels.
    Assumptions: no trend means progress cannot be proven.
    """

    if trend is None:
        return "Sin evidencia suficiente"
    latest = number_value(trend.get("latest_value"))
    direction = str(trend.get("direction") or "").lower()
    if topic == "payroll_overtime" and latest is not None and latest <= 0.42:
        return "Objetivo alcanzado"
    if topic == "collections" and latest is not None and latest >= 0.94:
        return "Objetivo alcanzado"
    if direction == "improving":
        return "Mejora parcial"
    if direction in {"worsening", "deterioro"}:
        return "En seguimiento"
    return "En seguimiento"


def _trend_evidence(topic: str, trend: dict[str, Any] | None) -> str:
    """Build deterministic current evidence for recommendation follow-up.

    Inputs: topic and related trend summary.
    Outputs: concise Spanish evidence statement.
    Assumptions: evidence repeats only supplied trend values.
    """

    if trend is None:
        return "No hay indicador cuantitativo acumulado para medir avance en el periodo actual."
    metric = {
        "payroll_overtime": "Nómina / ingresos",
        "collections": "Tasa de cobranza",
        "vendor_controls": "Controles de proveedores",
    }.get(topic, "Indicador relacionado")
    first = number_value(trend.get("first_value"))
    latest = number_value(trend.get("latest_value"))
    periods = trend.get("periods", [])
    unit = "ratio" if topic in {"payroll_overtime", "collections"} else ""
    if first is None or latest is None:
        return "La tendencia vinculada no contiene valores suficientes."
    start = format_period_label(periods[0]) if isinstance(periods, list) and periods else "periodo inicial"
    end = format_period_label(periods[-1]) if isinstance(periods, list) and periods else "periodo reciente"
    return f"{metric}: {format_value(first, unit)} en {start} y {format_value(latest, unit)} en {end}."


def _recommendation_next_action(topic: str, progress: str) -> str:
    """Return deterministic next action guidance for recommendation follow-up.

    Inputs: topic and calculated progress.
    Outputs: Spanish next action.
    Assumptions: text is fixed operational guidance, not LLM reasoning.
    """

    if progress == "Objetivo alcanzado":
        return "Mantener monitoreo mensual y conservar controles actuales."
    if progress == "Mejora parcial":
        return "Mantener la acción correctiva y verificar que alcance el objetivo definido."
    if topic == "vendor_controls":
        return "Reunir evidencia de facturas, aprobaciones y posibles duplicados para medir avance."
    if progress == "Sin evidencia suficiente":
        return "Solicitar evidencia documental o un indicador de seguimiento para medir avance."
    if topic == "payroll_overtime":
        return "Revisar horas extra, beneficios y dotación por departamento."
    if topic == "collections":
        return "Continuar seguimiento de saldos vencidos y planes de pago."
    return "Mantener seguimiento ejecutivo hasta contar con evidencia concluyente."


def _recommendation_status_reason(topic: str, progress: str, trend: dict[str, Any] | None) -> str:
    """Explain why a follow-up recommendation received its status.

    Inputs: topic, calculated progress, and optional deterministic trend.
    Outputs: Spanish status explanation.
    Assumptions: no new calculations are performed here.
    """

    evidence = _trend_evidence(topic, trend)
    if progress == "Objetivo alcanzado":
        return f"{evidence} El valor reciente cumple el umbral objetivo usado para seguimiento."
    if progress == "Mejora parcial":
        return f"{evidence} La dirección registrada mejora, pero aún no confirma cierre completo."
    if progress == "Sin evidencia suficiente":
        return "No existe un indicador cuantitativo acumulado que permita comprobar si la recomendación produjo el efecto esperado."
    return f"{evidence} El seguimiento continúa porque la evidencia disponible no demuestra cierre del objetivo."


def _recommendation_follow_up_summary(rows: list[dict[str, Any]]) -> str:
    """Summarize recommendation follow-up status deterministically.

    Inputs: follow-up rows.
    Outputs: executive Spanish summary.
    Assumptions: status labels were calculated by Python.
    """

    if not rows:
        return ""
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("progress") or row.get("status") or "Sin evidencia suficiente")
        counts[status] = counts.get(status, 0) + 1
    order = ["Objetivo alcanzado", "Mejora parcial", "En seguimiento", "Sin evidencia suficiente"]
    parts = [f"{counts[status]} {status.lower()}" for status in order if status in counts]
    return f"Se evalúan {len(rows)} recomendaciones previas: {', '.join(parts)}."


def _recommendation_objective(topic: str, record: dict[str, Any]) -> str:
    """Return a Spanish objective for a previous recommendation.

    Inputs: deterministic topic and original recommendation record.
    Outputs: Spanish objective text for executive display.
    Assumptions: topic is derived from the original action/impact text.
    """

    topic_objectives = {
        "payroll_overtime": "Alinear los costos de nómina, beneficios y horas extra con el presupuesto.",
        "collections": "Elevar la tasa de cobranza y reducir saldos estudiantiles vencidos.",
        "vendor_controls": "Fortalecer controles para prevenir pagos duplicados o sin soporte suficiente.",
    }
    if topic in topic_objectives:
        return topic_objectives[topic]
    raw = sanitize_text(record.get("expected_impact") or "")
    return raw or "Objetivo no especificado en la recomendación original."


def _localize_direction(value: Any) -> str:
    """Return a short Spanish label for trend direction metadata.

    Inputs: trend direction from retrieval/presentation data.
    Outputs: Spanish display label.
    Assumptions: unknown values are treated as stable/informational.
    """

    mapping = {
        "improving": "Mejora",
        "worsening": "Deterioro",
        "stable": "Estable",
        "increasing": "Al alza",
        "decreasing": "A la baja",
    }
    return mapping.get(str(value or "").lower(), "Estable")

