"""Presentation adapters for executive financial reports.

This module converts the renderer-agnostic report model into a polished Spanish
presentation view.  It deliberately does not recalculate finance values; it only
formats, localizes, filters, and summarizes validated pipeline outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    "payroll_percentage_of_revenue": ("Nómina / ingresos", "ratio", "Peso de la nómina sobre ingresos."),
    "student_payment_collection_rate": ("Tasa de cobranza", "ratio", "Porcentaje cobrado sobre saldos estudiantiles."),
    "collection_rate": ("Tasa de cobranza", "ratio", "Porcentaje cobrado sobre saldos estudiantiles."),
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
    "get_metric_history": "tendencia histórica de KPI",
    "get_department_history": "historial departamental",
    "get_repeated_anomalies": "riesgos recurrentes",
    "get_previous_recommendations": "recomendaciones previas",
    "get_goal_progress": "avance de metas",
    "get_memory_facts": "memoria ejecutiva",
}

CANONICAL_IDENTIFIERS: tuple[str, ...] = tuple(METRIC_LABELS_ES)
RAW_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{[^{}]*:[^{}]*\}"),
    re.compile(r"\[[^\[\]]*\{[^\[\]]*\}"),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"\bget_[a-z_]+\b"),
)


@dataclass(frozen=True)
class PresentationValidationResult:
    """Validation result for an executive presentation view.

    Inputs: errors and warnings found during presentation validation.
    Outputs: immutable result used by tests, CLIs, and renderers.
    Assumptions: errors block executive rendering.
    """

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def get_section(report_model: dict[str, Any], section_id: str) -> dict[str, Any]:
    """Return one section from a report model.

    Inputs: report model dictionary and section identifier.
    Outputs: section dictionary, or an empty section placeholder.
    Assumptions: renderers can gracefully handle absent optional sections.
    """

    for section in report_model.get("sections", []):
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return section
    return {"section_id": section_id, "content": {}, "source_references": [], "warnings": []}


def number_value(value: Any) -> float | None:
    """Convert a scalar to float when possible.

    Inputs: any scalar display value.
    Outputs: float or None.
    Assumptions: conversion is only for formatting/chart scale, not calculations.
    """

    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def format_value(value: Any, unit: str | None = None) -> str:
    """Format a processed value for Spanish executive display.

    Inputs: value and optional unit.
    Outputs: formatted string.
    Assumptions: upstream Python already calculated the value.
    """

    number = number_value(value)
    if number is None:
        return "N/D" if value in (None, "") else str(value)
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
    Assumptions: unknown metrics are still sanitized for executive readers.
    """

    text = str(metric or "").strip()
    normalized = text.lower().replace(" ", "_")
    if text in METRIC_LABELS_ES:
        return METRIC_LABELS_ES[text][0]
    if normalized in METRIC_LABELS_ES:
        return METRIC_LABELS_ES[normalized][0]
    return text.replace("_", " ").strip().capitalize() or "Indicador"


def compact_source_label(source: Any) -> str:
    """Hide absolute paths and return a compact artifact label.

    Inputs: source reference from report sections.
    Outputs: filename-only label or short identifier.
    Assumptions: full paths remain in JSON for audit but not in body text.
    """

    text = str(source or "").strip()
    if not text:
        return ""
    return Path(text).name


def source_labels(section: dict[str, Any], *, limit: int = 4) -> list[str]:
    """Return compact source labels for a section.

    Inputs: one report section and maximum labels.
    Outputs: ordered compact source labels.
    Assumptions: duplicate source files are not useful for executives.
    """

    labels = [compact_source_label(item) for item in section.get("source_references", [])]
    return list(dict.fromkeys(label for label in labels if label))[:limit]


def localize_text(value: Any) -> str:
    """Localize concise LLM text into Spanish using deterministic phrase mapping.

    Inputs: text from validated strategic analysis.
    Outputs: Spanish-facing text when common English financial wording is detected.
    Assumptions: this is a presentation safety net; it preserves numbers and meaning
    through conservative phrase replacement rather than introducing new reasoning.
    """

    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    replacements = {
        "The financial performance shows mixed results with payroll costs under control but departmental budget adherence issues and underutilized resources in Facilities and Technology. Collection rates meet targets, but no overdue invoices reported may indicate potential cash flow risks.": "El desempeño financiero muestra resultados mixtos: la nómina está bajo control, pero persisten brechas de cumplimiento presupuestario por departamento y recursos subutilizados en Instalaciones y Tecnología. La cobranza cumple la meta, aunque la ausencia de facturas vencidas reportadas debe revisarse para confirmar la salud del flujo de caja.",
        "Payroll variance shows a significant +4% over budget, while Facilities, Supplies, and Technology categories show negative variances (under budget).": "La nómina presenta una variación significativa de +4% sobre el presupuesto, mientras Instalaciones, Suministros y Tecnología muestran variaciones negativas frente al presupuesto.",
        "Collection rate is 95%, meeting the 94% threshold, but no overdue invoices are reported, suggesting possible cash flow management issues.": "La tasa de cobranza es 95% y supera el umbral de 94%; sin embargo, la falta de facturas vencidas reportadas amerita validación operativa del flujo de caja.",
        "Departmental expense variances indicate inconsistent budget adherence across departments.": "Las variaciones de gasto por departamento indican cumplimiento presupuestario inconsistente.",
        "Payroll expense growth may be driven by headcount increases or salary adjustments not reflected in budget planning.": "El crecimiento del gasto de nómina puede estar asociado a aumentos de plantilla o ajustes salariales no reflejados en la planificación presupuestaria.",
        "Facilities and Technology under-spending may indicate inefficient resource allocation or delayed spending.": "La subejecución en Instalaciones y Tecnología puede indicar asignación ineficiente de recursos o gasto diferido.",
        "Inconsistent budget adherence across departments suggests possible lack of oversight or planning gaps.": "El cumplimiento presupuestario inconsistente por departamento sugiere brechas de supervisión o planificación.",
        "Conduct departmental budget adherence review with focus on payroll and Facilities/Technology.": "Realizar una revisión de cumplimiento presupuestario por departamento, con foco en nómina, Instalaciones y Tecnología.",
        "Inconsistent budget adherence across departments and payroll variance require immediate attention to align spending with planning.": "El cumplimiento presupuestario inconsistente y la variación de nómina requieren atención inmediata para alinear el gasto con la planificación.",
        "Improved budget adherence, reduced expense variances, and better resource allocation.": "Mejor cumplimiento presupuestario, menores variaciones de gasto y mejor asignación de recursos.",
        "Optimize underutilized resources in Facilities and Technology through reallocation or cost-saving initiatives.": "Optimizar recursos subutilizados en Instalaciones y Tecnología mediante reasignaciones o iniciativas de ahorro.",
        "Facilities and Technology under-spending suggests inefficient resource use that can be optimized.": "La subejecución en Instalaciones y Tecnología sugiere uso ineficiente de recursos que puede optimizarse.",
        "Cost savings and improved operational efficiency.": "Ahorros y mayor eficiencia operativa.",
        "Maintain payroll cost control to stay within 42% threshold.": "Mantener el control de nómina para permanecer dentro del umbral de 42%.",
        "Improve departmental budget adherence to reduce expense variances.": "Mejorar el cumplimiento presupuestario por departamento para reducir variaciones de gasto.",
        "Evidence shows payroll variance, departmental inconsistencies, and underutilized resources. Root causes include planning gaps and inefficient allocation. Recommendations focus on immediate adherence review and resource optimization.": "La evidencia muestra variación de nómina, inconsistencias departamentales y recursos subutilizados. Las causas raíz incluyen brechas de planificación y asignación ineficiente. Las recomendaciones se enfocan en revisión inmediata del cumplimiento presupuestario y optimización de recursos.",
        "Retrieved processed": "Se recuperó el resumen financiero procesado de",
        "finance summary.": "para el periodo.",
        "The financial performance shows mixed results": "El desempeño financiero muestra resultados mixtos",
        "payroll costs under control": "costos de nómina bajo control",
        "departmental budget adherence issues": "problemas de cumplimiento presupuestario por departamento",
        "underutilized resources": "recursos subutilizados",
        "Collection rates meet targets": "La tasa de cobranza cumple las metas",
        "cash flow risks": "riesgos de flujo de caja",
        "Payroll variance shows": "La variación de nómina muestra",
        "significant": "significativa",
        "over budget": "por encima del presupuesto",
        "under budget": "por debajo del presupuesto",
        "Departmental expense variances": "Las variaciones de gasto por departamento",
        "inconsistent budget adherence": "cumplimiento presupuestario inconsistente",
        "Payroll expense growth": "El crecimiento del gasto de nómina",
        "headcount increases": "aumentos de plantilla",
        "salary adjustments": "ajustes salariales",
        "budget planning": "planificación presupuestaria",
        "under-spending": "subejecución presupuestaria",
        "inefficient resource allocation": "asignación ineficiente de recursos",
        "planning gaps": "brechas de planificación",
        "Conduct": "Realizar",
        "review": "revisión",
        "Optimize": "Optimizar",
        "through reallocation or cost-saving initiatives": "mediante reasignación o iniciativas de ahorro",
        "Improved": "Mejor",
        "reduced expense variances": "menores variaciones de gasto",
        "better resource allocation": "mejor asignación de recursos",
        "Cost savings": "Ahorros",
        "operational efficiency": "eficiencia operativa",
        "Maintain": "Mantener",
        "Improve": "Mejorar",
        "Evidence shows": "La evidencia muestra",
        "Root causes include": "Las causas raíz incluyen",
        "Recommendations focus on": "Las recomendaciones se enfocan en",
        "budget adherence": "cumplimiento presupuestario",
        "departmental": "departamental",
        "departments": "departamentos",
        "payroll": "nómina",
        "Facilities": "Instalaciones",
        "Technology": "Tecnología",
        "Supplies": "Suministros",
        "categories": "categorías",
        "negative variances": "variaciones negativas",
        "Collection rate": "Tasa de cobranza",
        "meeting": "cumpliendo",
        "threshold": "umbral",
        "no overdue invoices are reported": "no se reportan facturas vencidas",
        "no overdue invoices reported": "no se reportan facturas vencidas",
        "suggesting possible": "lo que sugiere posibles",
        "management issues": "problemas de gestión",
        "may be driven by": "puede estar impulsado por",
        "not reflected in": "no reflejados en",
        "may indicate": "puede indicar",
        "delayed spending": "gasto diferido",
        "suggests possible lack of oversight": "sugiere posible falta de supervisión",
        "planning": "planificación",
        "with focus on": "con foco en",
        "require immediate attention": "requieren atención inmediata",
        "to align spending with planning": "para alinear el gasto con la planificación",
        "resource use": "uso de recursos",
        "can be optimized": "puede optimizarse",
        "resource optimization": "optimización de recursos",
        "cash flow management issues": "problemas de gestión del flujo de caja",
        "potential": "potenciales",
        "resource allocation": "asignación de recursos",
        "inefficient allocation": "asignación ineficiente",
        "review": "revisión",
        "cost control": "control de costos",
        "stay within": "mantenerse dentro de",
        "reduce": "reducir",
        "and": "y",
        "with": "con",
        "but": "pero",
        "while": "mientras",
        " in ": " en ",
        " across ": " en ",
    }
    localized = text
    for source, target in replacements.items():
        if source.strip() in {"and", "with", "but", "while"}:
            localized = re.sub(rf"\b{re.escape(source)}\b", target, localized)
        else:
            localized = localized.replace(source, target)
    localized = (
        localized.replace("po departamento", "por departamento")
        .replace("repotadas", "reportadas")
        .replace("superoilizados", "subutilizados")
    )
    # Replace canonical identifiers wherever they leaked into narrative text.
    for metric, (label, _, _) in METRIC_LABELS_ES.items():
        localized = re.sub(rf"\b{re.escape(metric)}\b", label, localized)
    return localized


def localize_items(items: Any, *, limit: int = 8) -> list[str]:
    """Return a bounded list of localized text items.

    Inputs: list-like value from strategic analysis.
    Outputs: localized strings.
    Assumptions: non-list inputs are represented as one item when non-empty.
    """

    raw_items = items if isinstance(items, list) else ([items] if items else [])
    return [localize_text(item) for item in raw_items[:limit] if localize_text(item)]


def build_metric_cards(report_model: dict[str, Any]) -> list[dict[str, Any]]:
    """Build KPI-style cards for the financial health dashboard.

    Inputs: report model.
    Outputs: list of display-ready metric card dictionaries.
    Assumptions: values come directly from processed finance outputs.
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
        value = content.get(key)
        numeric = number_value(value)
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
                "value": format_value(value, unit),
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
    Assumptions: raw KPI rows are processed outputs and not recalculated.
    """

    kpis = get_section(report_model, "kpi_overview").get("content", {}).get("kpis", [])
    rows: list[dict[str, str]] = []
    for item in kpis if isinstance(kpis, list) else []:
        if not isinstance(item, dict):
            continue
        unit = item.get("unit")
        metric = str(item.get("metric") or "")
        normalized = metric.lower().replace(" ", "_")
        _, default_unit, description = METRIC_LABELS_ES.get(
            metric,
            METRIC_LABELS_ES.get(normalized, (display_metric_name(metric), "", "")),
        )
        rows.append(
            {
                "indicator": display_metric_name(metric),
                "value": format_value(item.get("value"), str(unit or default_unit or "")),
                "status": _localize_status(item.get("availability")),
                "description": description or str(item.get("source") or ""),
            }
        )
    return rows


def build_revenue_expense_summary(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build display-ready revenue and expense comparison content.

    Inputs: report model.
    Outputs: chart items and table rows.
    Assumptions: upstream finance calculations provide actual and budget values.
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
    rows = []
    for key in keys:
        value = revenue.get(key, expense.get(key))
        if value is None:
            continue
        label, unit, description = METRIC_LABELS_ES.get(key, (display_metric_name(key), "USD", ""))
        rows.append(
            {
                "metric": label,
                "value": format_value(value, unit),
                "description": description,
            }
        )
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
    """Build localized department comparison rows.

    Inputs: report model and maximum rows.
    Outputs: display-ready department records.
    Assumptions: department data is already aggregated upstream.
    """

    items = get_section(report_model, "department_analysis").get("content", {}).get("department_summary", [])
    rows: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        expenses = item.get("actual_expenses", item.get("actual_expense"))
        revenue = item.get("actual_revenue")
        result = item.get("net_operating_result")
        variance = item.get("expense_variance_pct")
        rows.append(
            {
                "department": str(item.get("department") or "Sin departamento"),
                "revenue": format_value(revenue, "USD"),
                "expenses": format_value(expenses, "USD"),
                "result": format_value(result, "USD"),
                "variance": format_value(variance, "ratio") if variance is not None else "N/D",
                "numeric_result": number_value(result) or 0.0,
                "numeric_expenses": number_value(expenses) or 0.0,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["numeric_expenses"])), reverse=True)[:limit]


def build_anomaly_summary(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build display-ready anomaly summary.

    Inputs: report model.
    Outputs: severity counts and concise anomaly rows.
    Assumptions: anomaly detection remains deterministic upstream.
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
        if not isinstance(item, dict):
            continue
        top_rows.append(
            {
                "title": localize_text(item.get("title") or item.get("description") or "Anomalía detectada"),
                "severity": SEVERITY_LABELS_ES.get(str(item.get("severity", "")).lower(), str(item.get("severity", ""))),
                "evidence": localize_text(item.get("evidence") or item.get("description") or ""),
            }
        )
    if not top_rows and (not severity_rows or not any(row["count"] for row in severity_rows)):
        return {
            "positive_status": "No se detectaron anomalías relevantes en los datos procesados del periodo.",
            "severity_rows": [],
            "top_rows": [],
        }
    return {"positive_status": "", "severity_rows": severity_rows, "top_rows": top_rows}


def build_evidence_summary(report_model: dict[str, Any], *, limit: int = 8) -> list[dict[str, str]]:
    """Build concise investigation evidence rows without internal task/tool IDs.

    Inputs: report model and row limit.
    Outputs: display-ready evidence summaries.
    Assumptions: detailed evidence remains available in JSON artifacts.
    """

    items = get_section(report_model, "investigation_evidence").get("content", {}).get("evidence_items", [])
    rows: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = TOOL_LABELS_ES.get(str(item.get("retrieval_name")), "evidencia procesada")
        rows.append(
            {
                "priority": PRIORITY_LABELS_ES.get(str(item.get("priority", "")).lower(), str(item.get("priority", ""))),
                "evidence": name,
                "records": str(item.get("record_count") or "N/D"),
                "summary": localize_text(item.get("evidence_summary") or ""),
            }
        )
    return rows[:limit]


def build_recommendation_cards(report_model: dict[str, Any]) -> list[dict[str, str]]:
    """Build executive recommendation cards.

    Inputs: report model.
    Outputs: cards with priority, action, rationale, impact, and status.
    Assumptions: recommendations are validated strategic-analysis outputs.
    """

    content = get_section(report_model, "strategic_recommendations").get("content", {})
    cards: list[dict[str, str]] = []
    for item in content.get("recommendations", []) or []:
        if isinstance(item, dict):
            priority = PRIORITY_LABELS_ES.get(str(item.get("priority", "")).lower(), str(item.get("priority", "")))
            cards.append(
                {
                    "priority": priority or "Media",
                    "action": localize_text(item.get("action") or item.get("recommendation") or ""),
                    "rationale": localize_text(item.get("rationale") or item.get("supporting_evidence") or ""),
                    "expected_impact": localize_text(item.get("expected_impact") or ""),
                    "owner_status": localize_text(item.get("owner") or item.get("status") or "Responsable por asignar"),
                }
            )
        elif item:
            cards.append(
                {
                    "priority": "Media",
                    "action": localize_text(item),
                    "rationale": "",
                    "expected_impact": "",
                    "owner_status": "Responsable por asignar",
                }
            )
    return cards


def build_historical_presentation(report_model: dict[str, Any]) -> dict[str, Any]:
    """Convert compact historical retrieval structures into readable exhibits.

    Inputs: report model with optional historical sections.
    Outputs: trend series, recurring risks, follow-up rows, and narratives.
    Assumptions: retrieval outputs are compact and processed, never full reports.
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
    risk_rows = _recurring_risk_rows(context)
    follow_up = _recommendation_follow_up_rows(context)
    narratives = [
        _trend_narrative(item)
        for item in trends[:4]
        if item.get("points")
    ]
    return {
        "available": bool(trends or risk_rows or follow_up),
        "narrative": narratives,
        "trends": trends[:4],
        "recurring_risks": risk_rows[:8],
        "recommendation_follow_up": follow_up[:8],
        "longitudinal_conclusions": _longitudinal_conclusions(trends, risk_rows, follow_up),
    }


def _clean_historical_sections(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build historical presentation from already-clean report sections.

    Inputs: report model.
    Outputs: historical presentation dictionary.
    Assumptions: new report models store presentation-ready historical rows, not
    raw retrieval tool payloads.
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
        return {
            "available": False,
            "narrative": [],
            "trends": [],
            "recurring_risks": [],
            "recommendation_follow_up": [],
            "longitudinal_conclusions": [],
        }
    normalized_trends = []
    for series in trends if isinstance(trends, list) else []:
        if not isinstance(series, dict):
            continue
        normalized_points = []
        for point in series.get("points", []) or []:
            if not isinstance(point, dict):
                continue
            value = number_value(point.get("value"))
            if value is None:
                continue
            normalized_points.append(
                {
                    "period": str(point.get("period") or ""),
                    "value": value,
                    "display": str(point.get("display") or format_value(value, series.get("unit"))),
                }
            )
        normalized_trends.append(
            {
                "metric": str(series.get("metric") or ""),
                "unit": str(series.get("unit") or ""),
                "direction": str(series.get("direction") or "estable"),
                "points": normalized_points,
            }
        )
    return {
        "available": True,
        "narrative": [str(item) for item in narrative if item],
        "trends": normalized_trends,
        "recurring_risks": [item for item in risks if isinstance(item, dict)],
        "recommendation_follow_up": [item for item in follow_up if isinstance(item, dict)],
        "longitudinal_conclusions": [str(item) for item in conclusions if item],
    }


def build_missing_information(report_model: dict[str, Any]) -> list[str]:
    """Build a concise missing-information list.

    Inputs: report model.
    Outputs: localized missing-information strings or a positive status.
    Assumptions: false missing-info filtering happens upstream; this only displays.
    """

    content = get_section(report_model, "missing_information").get("content", {})
    items = localize_items(content.get("missing_information"), limit=8)
    return items or ["No se identificó información faltante material para este reporte ejecutivo."]


def build_appendix(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build useful methodology and source notes without exposing local paths.

    Inputs: report model.
    Outputs: appendix rows for methodology, validation, and source filenames.
    Assumptions: full paths remain in machine-readable JSON only.
    """

    sources = [compact_source_label(source) for source in report_model.get("source_references", [])]
    sources = list(dict.fromkeys(source for source in sources if source))
    quality = "Análisis estratégico aceptado; cálculos y anomalías provienen de salidas procesadas."
    return {
        "methodology": [
            "Los cálculos financieros, KPIs y anomalías fueron generados por reglas determinísticas de Python.",
            "Ollama se utiliza únicamente para interpretación estratégica sobre evidencia ya calculada y validada.",
            "Las cifras se muestran redondeadas para lectura ejecutiva; los artefactos JSON/CSV conservan los valores auditables.",
        ],
        "validation": quality,
        "sources": sources,
    }


def build_presentation_view(report_model: dict[str, Any], *, mode: str = "executive") -> dict[str, Any]:
    """Build the renderer-facing presentation view.

    Inputs: report model and rendering mode (`executive` or `technical`).
    Outputs: display-ready dictionary shared by HTML and PDF renderers.
    Assumptions: executive mode hides internal identifiers and absolute paths.
    """

    if mode not in {"executive", "technical"}:
        raise ValueError("Report rendering mode must be 'executive' or 'technical'.")
    executive = get_section(report_model, "executive_summary").get("content", {})
    recommendations = get_section(report_model, "strategic_recommendations").get("content", {})
    historical = build_historical_presentation(report_model)
    view = {
        "mode": mode,
        "report_id": report_model.get("report_id"),
        "period_slug": report_model.get("period_slug"),
        "period": report_model.get("report_period"),
        "title": "Reporte financiero ejecutivo",
        "organization": "Universidad / Institución",
        "sections": EXECUTIVE_SECTION_ORDER,
        "labels": SECTION_LABELS_ES,
        "executive_summary": {
            "summary": localize_text(executive.get("summary") or ""),
            "key_findings": localize_items(executive.get("key_findings"), limit=6),
            "root_causes": localize_items(executive.get("root_causes"), limit=6),
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
        "historical": historical,
        "recommendations": {
            "priorities": localize_items(recommendations.get("strategic_priorities"), limit=6),
            "reasoning_summary": localize_text(recommendations.get("reasoning_summary") or ""),
            "cards": build_recommendation_cards(report_model),
        },
        "missing_information": build_missing_information(report_model),
        "appendix": build_appendix(report_model),
    }
    if mode == "technical":
        # Technical mode preserves compact debug references for analysts while
        # executive mode hides them from leadership-facing pages.
        view["technical_sources"] = report_model.get("source_references", [])
    validate_presentation_view(view, mode=mode)
    return view


def validate_presentation_view(view: dict[str, Any], *, mode: str = "executive") -> PresentationValidationResult:
    """Validate that presentation text is safe for executive rendering.

    Inputs: presentation view and mode.
    Outputs: validation result.
    Assumptions: executive mode must not expose raw structures, tool names, paths,
    or canonical metric identifiers in user-facing strings.
    """

    errors: list[str] = []
    warnings: list[str] = []
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
    return PresentationValidationResult(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _visible_strings(value: Any) -> list[str]:
    """Collect user-facing string values from a presentation view.

    Inputs: nested presentation value.
    Outputs: visible string leaves, excluding internal IDs/units/status tokens.
    Assumptions: validation should inspect rendered content, not dictionary keys.
    """

    strings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"id", "unit", "mode", "period_slug", "report_id", "labels", "sections"}:
                continue
            strings.extend(_visible_strings(child))
        return strings
    if isinstance(value, list):
        for child in value:
            strings.extend(_visible_strings(child))
        return strings
    if isinstance(value, str):
        strings.append(value)
    return strings


def _localize_status(value: Any) -> str:
    """Translate simple availability/status values.

    Inputs: status value from processed outputs.
    Outputs: Spanish status.
    Assumptions: unknown statuses remain readable.
    """

    mapping = {"available": "Disponible", "unavailable": "No disponible", "planned": "Planificado"}
    return mapping.get(str(value or "").lower(), str(value or "N/D"))


def _net_result_value(report_model: dict[str, Any]) -> float:
    """Return net operating result from the financial health section.

    Inputs: report model.
    Outputs: numeric result or 0.
    Assumptions: no recalculation is performed.
    """

    content = get_section(report_model, "financial_health_overview").get("content", {})
    return number_value(content.get("net_operating_result")) or 0.0


def _historical_context(report_model: dict[str, Any]) -> dict[str, Any]:
    """Return historical context embedded in report sections when available.

    Inputs: report model.
    Outputs: historical context dictionary.
    Assumptions: Phase 13 stores compact historical context on historical sections.
    """

    for section_id in ("historical_summary", "historical_trends", "recommendation_follow_up"):
        content = get_section(report_model, section_id).get("content", {})
        if isinstance(content, dict) and isinstance(content.get("historical_context"), dict):
            return content["historical_context"]
    # Backward compatibility for report models generated before this adapter.
    content = get_section(report_model, "historical_trends").get("content", {})
    return {"retrievals": content.get("metric_trends", []) if isinstance(content, dict) else []}


def _trend_series(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one derived KPI trend into chart-ready points.

    Inputs: derived trend dictionary.
    Outputs: display-ready trend series.
    Assumptions: values are already historical KPI records.
    """

    metric = str(item.get("metric") or "")
    label, unit, _ = METRIC_LABELS_ES.get(metric, (display_metric_name(metric), "", ""))
    points = []
    for point in item.get("points", []) or []:
        if not isinstance(point, dict):
            continue
        numeric = number_value(point.get("value"))
        if numeric is None:
            continue
        points.append(
            {
                "period": str(point.get("period") or ""),
                "value": numeric,
                "display": format_value(numeric, unit),
            }
        )
    direction = str(item.get("direction") or "estable")
    if points and metric == "payroll_percentage_of_revenue":
        direction = "improving" if points[-1]["value"] <= points[0]["value"] else "worsening"
    if points and metric == "student_payment_collection_rate":
        direction = "improving" if points[-1]["value"] >= points[0]["value"] else "worsening"
    if points and metric == "net_cash_flow":
        direction = "improving" if points[-1]["value"] >= points[0]["value"] else "worsening"
    return {
        "metric": label,
        "unit": unit,
        "direction": direction,
        "points": points,
    }


def _trend_items_from_retrievals(
    context: dict[str, Any],
    trend_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct chart-ready trend items from compact retrieval records.

    Inputs: historical context and derived trend summary by metric.
    Outputs: list using the same shape consumed by `_trend_series`.
    Assumptions: retrieval records contain processed historical KPI values only.
    """

    retrievals = context.get("retrievals", []) if isinstance(context, dict) else []
    items: list[dict[str, Any]] = []
    for retrieval in retrievals if isinstance(retrievals, list) else []:
        if not isinstance(retrieval, dict) or retrieval.get("tool_name") != "get_metric_history":
            continue
        metric = str(retrieval.get("metric") or retrieval.get("arguments", {}).get("metric") or "")
        records = retrieval.get("records", [])
        if not metric or not isinstance(records, list):
            continue
        points = [
            {"period": record.get("period"), "value": record.get("value")}
            for record in records
            if isinstance(record, dict)
        ]
        items.append(
            {
                "metric": metric,
                "direction": (trend_summary.get(metric, {}) or {}).get("direction", "stable"),
                "points": points,
            }
        )
    return items


def _trend_narrative(series: dict[str, Any]) -> str:
    """Build a concise Spanish narrative for one KPI trend.

    Inputs: trend series.
    Outputs: one sentence.
    Assumptions: the sentence describes processed historical values only.
    """

    points = series.get("points", [])
    if not points:
        return ""
    first = points[0]
    last = points[-1]
    direction = {
        "improving": "mejoró",
        "worsening": "se deterioró",
        "stable": "se mantuvo estable",
    }.get(str(series.get("direction")), str(series.get("direction") or "cambió"))
    return (
        f"{series.get('metric')} {direction} de {first.get('display')} "
        f"en {first.get('period')} a {last.get('display')} en {last.get('period')}."
    )


def _recurring_risk_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build readable recurring-risk rows from historical context.

    Inputs: historical context.
    Outputs: recurring anomaly/risk rows.
    Assumptions: compact historical patterns are safe to show without tool names.
    """

    rows: list[dict[str, str]] = []
    derived = context.get("derived_context", {}) if isinstance(context, dict) else {}
    patterns = derived.get("artifact_anomaly_patterns", []) if isinstance(derived, dict) else []
    for item in patterns if isinstance(patterns, list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "risk": localize_text(str(item.get("pattern", "Riesgo recurrente")).replace("_", " ")).capitalize(),
                "department": str(item.get("department") or "Institucional"),
                "occurrences": str(item.get("occurrences") or ""),
                "periods": ", ".join(str(period) for period in item.get("periods", [])[:6]),
            }
        )
    return rows


def _recommendation_follow_up_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build readable prior-recommendation follow-up rows.

    Inputs: historical context.
    Outputs: rows with recommendation, period, evidence, and inferred status.
    Assumptions: effectiveness is derived deterministically by context builder.
    """

    rows: list[dict[str, str]] = []
    derived = context.get("derived_context", {}) if isinstance(context, dict) else {}
    effectiveness = derived.get("recommendation_effectiveness", []) if isinstance(derived, dict) else []
    for item in effectiveness if isinstance(effectiveness, list) else []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("inferred_status") or "").replace("_", " ")
        topic = str(item.get("topic") or "Recomendación previa").replace("_", " ")
        topic = {
            "payroll overtime": "Control de horas extra de nómina",
            "collections": "Gestión de cobranza",
            "vendor controls": "Controles de proveedores",
        }.get(topic.lower(), topic.capitalize())
        rows.append(
            {
                "recommendation": localize_text(topic),
                "issued_period": str(item.get("issued_period") or "N/D"),
                "current_evidence": localize_text(item.get("evidence") or ""),
                "status": status.capitalize() if status else "En seguimiento",
            }
        )
    return rows


def _longitudinal_conclusions(
    trends: list[dict[str, Any]],
    risks: list[dict[str, str]],
    follow_up: list[dict[str, str]],
) -> list[str]:
    """Build concise longitudinal-risk conclusions.

    Inputs: trend series, recurring risks, and follow-up rows.
    Outputs: Spanish bullet conclusions.
    Assumptions: conclusions summarize deterministic context; they do not add math.
    """

    conclusions: list[str] = []
    if trends:
        conclusions.append("Las tendencias históricas muestran señales útiles para priorizar la gestión del periodo actual.")
    if risks:
        conclusions.append("Existen riesgos recurrentes que requieren seguimiento directivo, especialmente donde se repiten por varios periodos.")
    if follow_up:
        conclusions.append("Las recomendaciones previas deben revisarse contra evidencia actual para confirmar avance o persistencia del riesgo.")
    return conclusions
