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
    "anomaly_summary": "Riesgos y anomalías relevantes",
    "investigation_evidence": "Evidencia de investigación",
    "recommendation_follow_up": "Seguimiento de recomendaciones anteriores",
    "longitudinal_risk_assessment": "Evaluación longitudinal de riesgos",
    "strategic_recommendations": "Recomendaciones estratégicas actuales",
    "missing_information": "Información faltante / supuestos",
    "appendix": "Metodología y fuentes",
}

METRIC_LABELS_ES: dict[str, tuple[str, str, str]] = {
    "total_revenue": ("Ingresos totales", "USD", "Ingreso operativo reconocido en el periodo."),
    "total_expenses": ("Gastos totales", "USD", "Gasto operativo reconocido en el periodo."),
    "net_operating_result": ("Resultado operativo", "USD", "Diferencia entre ingresos y gastos operativos."),
    "net_cash_flow": ("Flujo neto de caja", "USD", "Entrada o salida neta de efectivo del periodo."),
    "ending_cash": ("Caja final", "USD", "Saldo de caja al cierre del periodo."),
    "payroll_percentage_of_revenue": ("NÃ³mina / ingresos", "ratio", "Peso de la nÃ³mina sobre ingresos."),
    "student_payment_collection_rate": ("Tasa de cobranza", "ratio", "Porcentaje cobrado sobre saldos estudiantiles."),
    "collection_rate": ("Tasa de cobranza", "ratio", "Porcentaje cobrado sobre saldos estudiantiles."),
    "revenue_budget": ("Presupuesto de ingresos", "USD", "Meta presupuestada de ingresos."),
    "revenue_variance": ("VariaciÃ³n de ingresos", "USD", "Diferencia entre ingreso actual y presupuesto."),
    "revenue_variance_pct": ("VariaciÃ³n de ingresos", "ratio", "VariaciÃ³n porcentual contra presupuesto."),
    "expense_budget": ("Presupuesto de gastos", "USD", "Meta presupuestada de gastos."),
    "expense_variance": ("VariaciÃ³n de gastos", "USD", "Diferencia entre gasto actual y presupuesto."),
    "expense_variance_pct": ("VariaciÃ³n de gastos", "ratio", "VariaciÃ³n porcentual contra presupuesto."),
    "payroll_total": ("NÃ³mina total", "USD", "Gasto total de nÃ³mina."),
    "revenue_budget_variance": ("VariaciÃ³n presupuestaria de ingresos", "USD", "Diferencia entre ingresos reales y presupuesto."),
    "revenue_budget_variance_pct": ("VariaciÃ³n presupuestaria de ingresos", "ratio", "Diferencia porcentual entre ingresos reales y presupuesto."),
    "expense_budget_variance": ("VariaciÃ³n presupuestaria de gastos", "USD", "Diferencia entre gastos reales y presupuesto."),
    "expense_budget_variance_pct": ("VariaciÃ³n presupuestaria de gastos", "ratio", "Diferencia porcentual entre gastos reales y presupuesto."),
}

SEVERITY_LABELS_ES: dict[str, str] = {
    "critical": "CrÃ­tica",
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
    "info": "Informativa",
}

PRIORITY_LABELS_ES: dict[str, str] = {
    "critical": "CrÃ­tica",
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
}

TOOL_LABELS_ES: dict[str, str] = {
    "department_history": "evidencia departamental",
    "payroll_history": "evidencia de nÃ³mina",
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
        "Mostrar salud financiera con KPIs principales y comentario analÃ­tico.",
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
        "Resumen histÃ³rico",
        "Resumir el contexto histÃ³rico disponible sin cargar reportes completos.",
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
        "Mostrar tendencias cronolÃ³gicas y anÃ¡lisis asociado.",
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
        "Comparar desempeÃ±o departamental y riesgos operativos.",
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
        "Presentar anomalÃ­as relevantes y su anÃ¡lisis.",
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
        "Dar seguimiento a recomendaciones previas con evidencia histÃ³rica.",
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
        "Evaluar riesgos recurrentes o persistentes a travÃ©s del tiempo.",
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
        "Listar brechas de evidencia que limitan el anÃ¡lisis.",
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
        "Documentar metodologÃ­a, validaciÃ³n y fuentes procesadas.",
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
    return f"{number:,.2f}".rstrip("0").rstrip(".")


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
                text = sanitize_text(content.get(field))
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
    sanitized = re.sub(r"[A-Za-z]:\\[^\s,;)]*", "[archivo procesado]", text)
    sanitized = re.sub(r"\bget_[a-z_]+\b", "herramienta de recuperaciÃ³n", sanitized)
    for metric, (label, _, _) in METRIC_LABELS_ES.items():
        sanitized = re.sub(rf"\b{re.escape(metric)}\b", label, sanitized)
    return sanitized


def sanitize_items(items: Any, *, limit: int = 8) -> list[str]:
    """Return a bounded list of sanitized text items.

    Inputs: list-like value.
    Outputs: sanitized strings.
    Assumptions: no English-to-Spanish translation is performed.
    """

    raw_items = items if isinstance(items, list) else ([items] if items else [])
    return [sanitize_text(item) for item in raw_items[:limit] if sanitize_text(item)]


def build_metric_cards(report_model: dict[str, Any]) -> list[dict[str, Any]]:
    """Build financial health metric cards.

    Inputs: report model.
    Outputs: display-ready card dictionaries.
    Assumptions: values come from processed finance outputs.
    """

    content = get_section(report_model, "financial_health_overview").get("content", {})
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
        cards.append(
            {
                "id": key,
                "label": label,
                "value": format_value(content.get(key), unit),
                "numeric_value": numeric,
                "unit": unit,
                "description": description,
                "status": status,
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
                "value": format_value(item.get("value"), str(item.get("unit") or default_unit or "")),
                "status": _localize_status(item.get("availability")),
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
    return {"rows": rows, "chart": chart, "budget_chart": budget_chart}


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
                "department": sanitize_text(item.get("department") or "Sin departamento"),
                "revenue": format_value(item.get("actual_revenue"), "USD"),
                "expenses": format_value(expenses, "USD"),
                "result": format_value(result, "USD"),
                "variance": format_value(item.get("expense_variance_pct"), "ratio") if item.get("expense_variance_pct") is not None else "N/D",
                "numeric_result": number_value(result) or 0.0,
                "numeric_expenses": number_value(expenses) or 0.0,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["numeric_expenses"])), reverse=True)[:limit]


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
                    "title": sanitize_text(item.get("title") or item.get("description") or "AnomalÃ­a detectada"),
                    "severity": SEVERITY_LABELS_ES.get(str(item.get("severity", "")).lower(), str(item.get("severity", ""))),
                    "evidence": sanitize_text(item.get("evidence") or item.get("description") or ""),
                }
            )
    if not top_rows and (not severity_rows or not any(row["count"] for row in severity_rows)):
        return {
            "positive_status": "Sin anomalÃ­as relevantes.",
            "severity_rows": [],
            "top_rows": [],
        }
    return {"positive_status": "", "severity_rows": severity_rows, "top_rows": top_rows}


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
                "records": str(item.get("record_count") or "N/D"),
                "summary": sanitize_text(item.get("evidence_summary") or ""),
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
                    "owner_status": sanitize_text(item.get("owner") or item.get("status") or "Responsable por asignar"),
                }
            )
        elif item:
            cards.append(
                {
                    "priority": "Media",
                    "action": sanitize_text(item),
                    "rationale": "",
                    "expected_impact": "",
                    "owner_status": "Responsable por asignar",
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
            "Los cÃ¡lculos financieros, KPIs y anomalÃ­as fueron generados por reglas determinÃ­sticas de Python.",
            "Ollama se utiliza Ãºnicamente para interpretaciÃ³n estratÃ©gica sobre evidencia ya calculada y validada.",
            "Las cifras se muestran redondeadas para lectura ejecutiva; los artefactos JSON/CSV conservan los valores auditables.",
        ],
        "validation": "AnÃ¡lisis estratÃ©gico aceptado; cÃ¡lculos y anomalÃ­as provienen de salidas procesadas.",
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
        "period": report_model.get("report_period"),
        "title": "Reporte financiero ejecutivo",
        "organization": "Universidad / InstituciÃ³n",
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
        if not view.get("recommendations", {}).get("cards"):
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
    return mapping.get(str(value or "").lower(), str(value or "N/D"))


def _net_result_value(report_model: dict[str, Any]) -> float:
    """Return net operating result for charts.

    Inputs: report model.
    Outputs: numeric result or 0.
    Assumptions: value is copied from processed output.
    """

    content = get_section(report_model, "financial_health_overview").get("content", {})
    return number_value(content.get("net_operating_result")) or 0.0


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
                points.append({"period": str(point.get("period") or ""), "value": numeric, "display": format_value(numeric, unit)})
    direction = str(item.get("direction") or "stable")
    if points and metric == "payroll_percentage_of_revenue":
        direction = "improving" if points[-1]["value"] <= points[0]["value"] else "worsening"
    if points and metric == "student_payment_collection_rate":
        direction = "improving" if points[-1]["value"] >= points[0]["value"] else "worsening"
    if points and metric == "net_cash_flow":
        direction = "improving" if points[-1]["value"] >= points[0]["value"] else "worsening"
    return {"metric": label, "unit": unit, "direction": direction, "points": points}


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
    patterns = derived.get("artifact_anomaly_patterns", []) if isinstance(derived, dict) else []
    rows: list[dict[str, str]] = []
    for item in patterns if isinstance(patterns, list) else []:
        if isinstance(item, dict):
            rows.append(
                {
                    "risk": sanitize_text(str(item.get("pattern", "Riesgo recurrente")).replace("_", " ")).capitalize(),
                    "department": sanitize_text(item.get("department") or "Institucional"),
                    "occurrences": str(item.get("occurrences") or ""),
                    "periods": ", ".join(str(period) for period in item.get("periods", [])[:6]),
                }
            )
    return rows


def _recommendation_follow_up_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build previous-recommendation follow-up rows.

    Inputs: historical context.
    Outputs: display-ready follow-up rows.
    Assumptions: topic labels are short fixed Spanish status labels.
    """

    derived = context.get("derived_context", {}) if isinstance(context, dict) else {}
    effectiveness = derived.get("recommendation_effectiveness", []) if isinstance(derived, dict) else []
    topic_labels = {
        "payroll_overtime": "Control de horas extra de nÃ³mina",
        "collections": "GestiÃ³n de cobranza",
        "vendor_controls": "Controles de proveedores",
    }
    rows: list[dict[str, str]] = []
    for item in effectiveness if isinstance(effectiveness, list) else []:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "RecomendaciÃ³n previa")
        status = str(item.get("inferred_status") or "").replace("_", " ")
        rows.append(
            {
                "recommendation": topic_labels.get(topic, sanitize_text(topic.replace("_", " ").capitalize())),
                "issued_period": str(item.get("issued_period") or "N/D"),
                "current_evidence": sanitize_text(item.get("evidence") or ""),
                "status": status.capitalize() if status else "En seguimiento",
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
    risks = risk_content.get("recurring_risks", []) if isinstance(risk_content, dict) else []
    conclusions = risk_content.get("conclusions", []) if isinstance(risk_content, dict) else []
    narrative = trends_content.get("narrative", []) if isinstance(trends_content, dict) else []
    if not any((trends, follow_up, risks, narrative, conclusions)):
        return {"available": False, "narrative": [], "trends": [], "recurring_risks": [], "recommendation_follow_up": [], "longitudinal_conclusions": []}
    normalized_trends = []
    for series in trends if isinstance(trends, list) else []:
        if not isinstance(series, dict):
            continue
        normalized_trends.append(
            {
                "metric": sanitize_text(series.get("metric") or ""),
                "unit": str(series.get("unit") or ""),
                "direction": str(series.get("direction") or "stable"),
                "points": [
                    {
                        "period": str(point.get("period") or ""),
                        "value": number_value(point.get("value")) or 0.0,
                        "display": str(point.get("display") or format_value(point.get("value"), series.get("unit"))),
                    }
                    for point in series.get("points", []) or []
                    if isinstance(point, dict) and number_value(point.get("value")) is not None
                ],
            }
        )
    return {
        "available": True,
        "narrative": sanitize_items(narrative),
        "trends": normalized_trends,
        "recurring_risks": [item for item in risks if isinstance(item, dict)],
        "recommendation_follow_up": [item for item in follow_up if isinstance(item, dict)],
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

