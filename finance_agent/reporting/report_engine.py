"""Build renderer-agnostic report models from processed pipeline outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finance_agent.reporting.report_models import (
    REQUIRED_SECTION_IDS,
    ReportModel,
    ReportSection,
)
from finance_agent.reporting.presentation import (
    build_anomaly_summary,
    build_department_rows,
    build_evidence_summary,
    build_historical_presentation,
    build_kpi_rows,
    build_metric_cards,
    build_missing_information,
    build_presentation_view,
    build_recommendation_cards,
    build_revenue_expense_summary,
    compact_source_label,
    number_value,
    sanitize_items,
    sanitize_text,
    validate_presentation_view,
)


class ReportInputError(RuntimeError):
    """Raised when required processed report inputs cannot be loaded."""


@dataclass(frozen=True)
class ReportInputBundle:
    """Processed inputs used to build one report model.

    Inputs: parsed finance, KPI, anomaly, evidence, and analysis artifacts.
    Outputs: immutable bundle consumed by report construction.
    Assumptions: all artifacts are processed outputs, never raw Excel/PDF inputs.
    """

    period_slug: str
    finance_summary: dict[str, Any]
    kpi_summary: tuple[dict[str, Any], ...]
    anomaly_report: dict[str, Any]
    evidence_package: dict[str, Any]
    strategic_analysis: dict[str, Any]
    source_files: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    """Read a processed JSON object.

    Inputs: JSON artifact path.
    Outputs: parsed dictionary.
    Assumptions: report inputs use object roots.
    """

    if not path.is_file():
        raise ReportInputError(f"Required report input does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"Could not read report input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"Report input JSON root must be an object: {path}")
    return value


def _read_csv_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a processed CSV artifact into row dictionaries.

    Inputs: CSV path.
    Outputs: ordered tuple of row dictionaries.
    Assumptions: values are copied as strings because calculations already happened.
    """

    if not path.is_file():
        raise ReportInputError(f"Required report input does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ReportInputError(f"Could not read report input {path}: {exc}") from exc


def load_report_inputs(project_root: str | Path, period_slug: str) -> ReportInputBundle:
    """Load processed artifacts for one report period.

    Inputs: project root and supported period slug (`june_2026` or `2026`).
    Outputs: ReportInputBundle.
    Assumptions: Step 10A consumes existing processed outputs only.
    """

    root = Path(project_root).resolve()
    if period_slug == "june_2026":
        paths = {
            "finance_summary": root / "outputs" / "calculations" / "finance_summary_june_2026.json",
            "kpi_summary": root / "outputs" / "calculations" / "kpi_summary_june_2026.csv",
            "anomaly_report": root / "outputs" / "anomalies" / "anomaly_report_june_2026.json",
            "evidence_package": root / "outputs" / "evidence" / "evidence_package_june_2026.json",
            "strategic_analysis": root / "outputs" / "analysis" / "strategic_analysis_june_2026.json",
        }
    elif period_slug == "2026":
        paths = {
            "finance_summary": root / "outputs" / "calculations" / "finance_summary_2026.json",
            "kpi_summary": root / "outputs" / "calculations" / "kpi_summary_2026.csv",
            "anomaly_report": root / "outputs" / "anomalies" / "anomaly_report_2026.json",
            "evidence_package": root / "outputs" / "evidence" / "evidence_package_2026.json",
            "strategic_analysis": root / "outputs" / "analysis" / "strategic_analysis_2026.json",
        }
    else:
        raise ReportInputError(f"Unsupported report period slug: {period_slug}")

    return ReportInputBundle(
        period_slug=period_slug,
        finance_summary=_read_json(paths["finance_summary"]),
        kpi_summary=_read_csv_records(paths["kpi_summary"]),
        anomaly_report=_read_json(paths["anomaly_report"]),
        evidence_package=_read_json(paths["evidence_package"]),
        strategic_analysis=_read_json(paths["strategic_analysis"]),
        source_files=tuple(str(path) for path in paths.values()),
    )


def _finance(document: dict[str, Any]) -> dict[str, Any]:
    """Return the calculated finance summary object.

    Inputs: finance summary document.
    Outputs: nested finance summary dictionary.
    Assumptions: missing finance data is represented as an empty dictionary.
    """

    value = document.get("finance_summary", {})
    return value if isinstance(value, dict) else {}


def _analysis_payload(document: dict[str, Any]) -> dict[str, Any]:
    """Return validated strategic-analysis payload.

    Inputs: strategic analysis output document.
    Outputs: analysis dictionary.
    Assumptions: Phase 14 modular runs expose the final validated synthesis in
    ReasoningState; legacy runs still expose the same payload under ``analysis``.
    """

    state = document.get("reasoning_state", {})
    state = state if isinstance(state, dict) else {}
    outputs = state.get("reasoning_outputs", {})
    outputs = outputs if isinstance(outputs, dict) else {}
    synthesis = outputs.get("strategic_synthesis")
    if (
        isinstance(synthesis, dict)
        and synthesis
        and not synthesis.get("_validation_failed")
        and "executive_summary" in synthesis
    ):
        return synthesis
    value = document.get("analysis", {})
    return value if isinstance(value, dict) else {}


def _analysis_text(analysis: dict[str, Any], field_name: str, fallback: str = "") -> str:
    """Return a model-authored section narrative.

    Inputs: strategic analysis payload, preferred field, and fallback field.
    Outputs: narrative text.
    Assumptions: Step 9 validated the prose as Spanish and evidence-bound.
    """

    value = analysis.get(field_name)
    if isinstance(value, str):
        return value
    fallback_value = analysis.get(fallback) if fallback else ""
    if isinstance(fallback_value, str):
        return fallback_value
    return ""


def _analysis_unavailable_warnings(document: dict[str, Any]) -> tuple[str, ...]:
    """Return report warnings when strategic analysis is not accepted.

    Inputs: Step 9 strategic-analysis document.
    Outputs: warning tuple for report model sections.
    Assumptions: final renderers should make missing strategy visible.
    """

    status = document.get("validation_status")
    errors = tuple(str(error) for error in document.get("validation_errors", []))
    if status == "accepted":
        return errors
    if status == "sanitized":
        warning = "Strategic analysis was adjusted to remove unsupported claims."
    else:
        warning = f"Strategic analysis is not accepted; status={status or 'unknown'}."
    return (warning, *errors)


def _deterministic_executive_summary(
    *,
    finance: dict[str, Any],
    anomaly_report: dict[str, Any],
    report_period: str,
    analysis_status: str | None,
) -> str:
    """Build a deterministic fallback executive summary.

    Inputs: finance summary, anomaly report, report period, and strategy status.
    Outputs: concise Spanish summary using only processed deterministic values.
    Assumptions: this is not strategic reasoning and must not fabricate causes or
    recommendations.
    """

    revenue = finance.get("total_revenue")
    expenses = finance.get("total_expenses")
    result = finance.get("net_operating_result")
    anomaly_count = anomaly_report.get("total_anomalies")
    parts = [f"Reporte financiero determinístico para {report_period}."]
    if revenue is not None and expenses is not None and result is not None:
        parts.append(
            "Ingresos, gastos y resultado operativo fueron calculados por el motor determinístico del pipeline."
        )
    if anomaly_count is not None:
        parts.append(f"El detector determinístico registró {anomaly_count} anomalías del período.")
    if analysis_status != "accepted":
        parts.append(
            "Las recomendaciones estratégicas validadas no están disponibles; el reporte conserva KPIs, comparaciones, anomalías, historial y evidencia procesada."
        )
    return " ".join(parts)


def _section(
    section_id: str,
    title: str,
    content: dict[str, Any],
    sources: tuple[str, ...],
    warnings: tuple[str, ...] = (),
) -> ReportSection:
    """Create one report section.

    Inputs: section metadata, content, source references, and warnings.
    Outputs: ReportSection.
    Assumptions: source references preserve artifact lineage for future renderers.
    """

    return ReportSection(
        section_id=section_id,
        title=title,
        content=content,
        source_references=sources,
        warnings=warnings,
    )


def _evidence_items(evidence_package: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact task evidence for the report model.

    Inputs: Step 8 evidence package.
    Outputs: bounded list of investigation evidence summaries.
    Assumptions: full evidence package remains the source artifact for audit.
    """

    items: list[dict[str, Any]] = []
    for item in evidence_package.get("evidence_packages", [])[:20]:
        if not isinstance(item, dict):
            continue
        evidence = item.get("retrieved_evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        data = evidence.get("data", {})
        data = data if isinstance(data, dict) else {}
        items.append(
            {
                "task_id": item.get("task_id"),
                "priority": item.get("priority"),
                "question": item.get("investigation_question"),
                "retrieval_name": evidence.get("retrieval_name"),
                "success": evidence.get("success"),
                "record_count": data.get("record_count"),
                "matched_tables": data.get("matched_tables") or data.get("source_tables"),
                "evidence_summary": item.get("evidence_summary"),
                "source_references": evidence.get("source_references", []),
                "warnings": evidence.get("warnings", []),
                "unavailable_data": evidence.get("unavailable_data", []),
            }
        )
    return items


def _all_section_sources(sections: tuple[ReportSection, ...]) -> tuple[str, ...]:
    """Collect unique source references across all sections.

    Inputs: report sections.
    Outputs: ordered unique source references.
    Assumptions: order of first use is useful for appendix rendering.
    """

    return tuple(
        dict.fromkeys(
            source
            for section in sections
            for source in section.source_references
        )
    )


def _previous_month_slug(period_slug: str) -> str | None:
    """Return the previous monthly slug when the current period is monthly.

    Inputs: period slug such as ``2026_12``.
    Outputs: previous period slug or None.
    Assumptions: annual/custom periods do not have a safe automatic previous month.
    """

    parts = period_slug.replace("-", "_").split("_")
    if len(parts) != 2:
        return None
    try:
        year = int(parts[0])
        month = int(parts[1])
    except ValueError:
        return None
    if month < 1 or month > 12:
        return None
    if month == 1:
        return f"{year - 1}_12"
    return f"{year}_{month - 1:02d}"


def _load_previous_finance_summary(current_finance_source: str, period_slug: str) -> tuple[dict[str, Any], str | None]:
    """Load a previous processed finance summary when it exists.

    Inputs: current finance-summary path and current period slug.
    Outputs: parsed previous summary and source path, or empty payload and None.
    Assumptions: this reads processed outputs only; raw workbooks are never opened.
    """

    previous_slug = _previous_month_slug(period_slug)
    if not previous_slug:
        return {}, None
    current_path = Path(current_finance_source)
    previous_path = current_path.with_name(f"finance_summary_{previous_slug}.json")
    if not previous_path.is_file():
        return {}, None
    try:
        payload = json.loads(previous_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    return (payload if isinstance(payload, dict) else {}), str(previous_path)


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    """Return a nested dictionary value.

    Inputs: payload and key path.
    Outputs: nested value or None.
    Assumptions: non-dict intermediates mean the value is unavailable.
    """

    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _historical_previous_values(historical_context: dict[str, Any]) -> dict[str, float]:
    """Extract latest previous KPI values from compact historical retrievals.

    Inputs: strategic-analysis historical context.
    Outputs: metric name to numeric value from the latest retrieved period.
    Assumptions: retrieval records exclude the current period by construction.
    """

    aliases = {
        "student_payment_collection_rate": "collection_rate",
    }
    values: dict[str, float] = {}
    retrievals = historical_context.get("retrievals", []) if isinstance(historical_context, dict) else []
    for retrieval in retrievals if isinstance(retrievals, list) else []:
        if not isinstance(retrieval, dict) or retrieval.get("tool_name") != "get_metric_history":
            continue
        metric = str(retrieval.get("metric") or retrieval.get("arguments", {}).get("metric") or "")
        records = retrieval.get("records", [])
        if not metric or not isinstance(records, list) or not records:
            continue
        latest = records[-1] if isinstance(records[-1], dict) else {}
        numeric = number_value(latest.get("value"))
        if numeric is not None:
            values[aliases.get(metric, metric)] = numeric
    return values


def _comparison_item(
    metric: str,
    *,
    unit: str,
    current_value: Any,
    previous_value: Any = None,
    budget_value: Any = None,
    budget_change: Any = None,
    budget_change_pct: Any = None,
    current_source: str,
    previous_source: str | None,
) -> dict[str, Any]:
    """Build one deterministic KPI comparison object.

    Inputs: metric values, budget values, units, and source references.
    Outputs: comparison payload for the report model.
    Assumptions: differences are deterministic arithmetic over processed values.
    """

    current = number_value(current_value)
    previous = number_value(previous_value)
    budget = number_value(budget_value)
    change = current - previous if current is not None and previous is not None else None
    percent_change = None
    percentage_point_change = None
    if current is not None and previous is not None:
        if unit == "ratio":
            percentage_point_change = change
        elif abs(previous) > 1e-12:
            percent_change = change / abs(previous)
    if budget_change is None and current is not None and budget is not None:
        budget_change = current - budget
    item = {
        "metric": metric,
        "unit": unit,
        "current_value": current,
        "previous_value": previous,
        "absolute_change": change,
        "percent_change": percent_change,
        "percentage_point_change": percentage_point_change,
        "budget_value": budget,
        "budget_change": number_value(budget_change),
        "budget_change_pct": number_value(budget_change_pct),
        "sources": {
            "current": current_source,
            "previous": previous_source,
            "budget": current_source if budget is not None else None,
        },
    }
    return item


def _build_kpi_comparisons(
    inputs: ReportInputBundle,
    finance: dict[str, Any],
    budget: dict[str, Any],
    payments: dict[str, Any],
    cash_flow: dict[str, Any],
    historical_context: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic current/previous/budget comparisons for KPI cards.

    Inputs: report inputs and processed finance/historical payloads.
    Outputs: comparison block keyed by canonical KPI metric.
    Assumptions: Ollama never supplies or calculates these values.
    """

    current_source = inputs.source_files[0]
    previous_document, previous_source = _load_previous_finance_summary(current_source, inputs.period_slug)
    previous_finance = _finance(previous_document)
    previous_payments = _nested_value(previous_finance, "student_payments") or {}
    previous_cash_flow = _nested_value(previous_finance, "cash_flow") or {}
    historical_previous = _historical_previous_values(historical_context)

    def previous(metric: str, *path: str) -> Any:
        """Prefer explicit history retrieval, then processed previous finance summary."""

        if metric in historical_previous:
            return historical_previous[metric]
        return _nested_value(previous_finance, *path)

    ending_cash_budget = None
    ending_cash_variance = number_value(cash_flow.get("ending_cash_variance"))
    ending_cash_current = number_value(cash_flow.get("ending_cash"))
    if ending_cash_current is not None and ending_cash_variance is not None:
        # Cash-flow calculations already produced the variance; this derives the
        # implied target solely for display comparison, not new financial logic.
        ending_cash_budget = ending_cash_current - ending_cash_variance

    items = {
        "total_revenue": _comparison_item(
            "total_revenue",
            unit="USD",
            current_value=finance.get("total_revenue"),
            previous_value=previous("total_revenue", "total_revenue"),
            budget_value=budget.get("revenue_budget"),
            budget_change=budget.get("revenue_variance"),
            budget_change_pct=budget.get("revenue_variance_pct"),
            current_source=current_source,
            previous_source=previous_source,
        ),
        "total_expenses": _comparison_item(
            "total_expenses",
            unit="USD",
            current_value=finance.get("total_expenses"),
            previous_value=previous("total_expenses", "total_expenses"),
            budget_value=budget.get("expense_budget"),
            budget_change=budget.get("expense_variance"),
            budget_change_pct=budget.get("expense_variance_pct"),
            current_source=current_source,
            previous_source=previous_source,
        ),
        "net_operating_result": _comparison_item(
            "net_operating_result",
            unit="USD",
            current_value=finance.get("net_operating_result"),
            previous_value=previous("net_operating_result", "net_operating_result"),
            budget_value=budget.get("net_budget"),
            budget_change=budget.get("net_variance"),
            current_source=current_source,
            previous_source=previous_source,
        ),
        "net_cash_flow": _comparison_item(
            "net_cash_flow",
            unit="USD",
            current_value=cash_flow.get("net_cash_flow"),
            previous_value=historical_previous.get("net_cash_flow", previous_cash_flow.get("net_cash_flow")),
            current_source=current_source,
            previous_source=previous_source,
        ),
        "ending_cash": _comparison_item(
            "ending_cash",
            unit="USD",
            current_value=cash_flow.get("ending_cash"),
            previous_value=previous_cash_flow.get("ending_cash"),
            budget_value=ending_cash_budget,
            budget_change=ending_cash_variance,
            current_source=current_source,
            previous_source=previous_source,
        ),
        "payroll_percentage_of_revenue": _comparison_item(
            "payroll_percentage_of_revenue",
            unit="ratio",
            current_value=finance.get("payroll_percentage_of_revenue"),
            previous_value=historical_previous.get(
                "payroll_percentage_of_revenue",
                previous_finance.get("payroll_percentage_of_revenue"),
            ),
            current_source=current_source,
            previous_source=previous_source,
        ),
        "collection_rate": _comparison_item(
            "collection_rate",
            unit="ratio",
            current_value=payments.get("collection_rate"),
            previous_value=historical_previous.get("collection_rate", previous_payments.get("collection_rate")),
            current_source=current_source,
            previous_source=previous_source,
        ),
    }
    unavailable = [
        {"metric": metric, "field": field}
        for metric, item in items.items()
        for field in ("previous_value", "budget_value")
        if item.get(field) is None
    ]
    return {
        "current_period": str(inputs.finance_summary.get("report_period", inputs.period_slug)),
        "previous_period": str(previous_document.get("report_period") or _previous_month_slug(inputs.period_slug) or ""),
        "source_policy": "processed_outputs_and_compact_history_only",
        "items": items,
        "unavailable": unavailable,
    }


def _historical_sections(
    historical_context: dict[str, Any],
    analysis: dict[str, Any],
    analysis_source: tuple[str, ...],
) -> tuple[ReportSection, ...]:
    """Build optional historical report sections when compact history exists.

    Inputs: historical context from strategic analysis and source references.
    Outputs: optional report sections.
    Assumptions: no sections are emitted when no historical retrieval succeeded.
    """

    if not historical_context:
        return ()
    summary = historical_context.get("summary", {})
    if not isinstance(summary, dict) or not summary.get("available_retrievals"):
        return ()
    presentation_seed = {
        "report_id": "historical-context-seed",
        "report_period": "",
        "sections": [
            {
                "section_id": "historical_summary",
                "content": {"historical_context": historical_context},
                "source_references": list(analysis_source),
                "warnings": [],
            }
        ],
        "source_references": list(analysis_source),
    }
    historical = build_historical_presentation(presentation_seed)
    trend_overview = [
        {
            "metric": trend.get("metric"),
            "unit": trend.get("unit"),
            "direction": trend.get("direction"),
            "points": [
                {
                    "period": point.get("period"),
                    "value": point.get("value"),
                    "display": point.get("display"),
                }
                for point in trend.get("points", [])
            ],
        }
        for trend in historical.get("trends", [])
    ]
    return (
        _section(
            "historical_summary",
            "Historical Summary",
            {
                "analysis": _analysis_text(analysis, "historical_summary"),
                "historical_context": historical_context,
                "narrative": historical.get("narrative", []),
                "retrieval_count": summary.get("available_retrievals", 0),
                "topics": [
                    str(topic).replace("get_", "").replace("_", " ").title()
                    for topic in summary.get("topics", [])
                ],
            },
            analysis_source,
        ),
        _section(
            "historical_trends",
            "Historical Trends",
            {
                "analysis": _analysis_text(analysis, "historical_trend_analysis"),
                "historical_context": historical_context,
                "trend_series": trend_overview,
                "narrative": historical.get("narrative", []),
            },
            analysis_source,
        ),
        _section(
            "recommendation_follow_up",
            "Recommendation Follow-up",
            {
                "analysis": _analysis_text(analysis, "recommendation_follow_up_analysis"),
                "historical_context": historical_context,
                "intro": historical.get("recommendation_intro", ""),
                "summary": historical.get("recommendation_summary", ""),
                "follow_up": historical.get("recommendation_follow_up", []),
            },
            analysis_source,
        ),
        _section(
            "longitudinal_risk_assessment",
            "Longitudinal Risk Assessment",
            {
                "analysis": _analysis_text(analysis, "longitudinal_risk_analysis"),
                "historical_context": historical_context,
                "recurring_risks": historical.get("recurring_risks", []),
                "risk_summary": historical.get("risk_summary", ""),
                "conclusions": historical.get("longitudinal_conclusions", []),
            },
            analysis_source,
        ),
    )


def _add_presentation_payload(model: ReportModel) -> None:
    """Attach display-ready payloads used by executive renderers.

    Inputs: report model object.
    Outputs: mutates shallow section content dictionaries with `presentation`.
    Assumptions: original deterministic/LLM fields remain preserved for audit and
    backward-compatible tests; renderers prefer the presentation payload.
    """

    report_data = model.to_dict()
    section_by_id = {section.section_id: section for section in model.sections}
    section_by_id["executive_summary"].content["presentation"] = {
        "summary": sanitize_text(section_by_id["executive_summary"].content.get("summary", "")),
        "key_findings": sanitize_items(section_by_id["executive_summary"].content.get("key_findings", [])),
        "root_causes": sanitize_items(section_by_id["executive_summary"].content.get("root_causes", [])),
    }
    section_by_id["financial_health_overview"].content["presentation"] = {
        "metric_cards": build_metric_cards(report_data),
    }
    section_by_id["kpi_overview"].content["presentation"] = {"rows": build_kpi_rows(report_data)}
    section_by_id["revenue_analysis"].content["presentation"] = build_revenue_expense_summary(report_data)
    section_by_id["expense_analysis"].content["presentation"] = build_revenue_expense_summary(report_data)
    section_by_id["department_analysis"].content["presentation"] = {"rows": build_department_rows(report_data)}
    section_by_id["anomaly_summary"].content["presentation"] = build_anomaly_summary(report_data)
    section_by_id["investigation_evidence"].content["presentation"] = {
        "rows": build_evidence_summary(report_data),
    }
    section_by_id["strategic_recommendations"].content["presentation"] = {
        "recommendations": build_recommendation_cards(report_data),
        "priorities": sanitize_items(
            section_by_id["strategic_recommendations"].content.get("strategic_priorities", [])
        ),
        "reasoning_summary": sanitize_text(
            section_by_id["strategic_recommendations"].content.get("reasoning_summary", "")
        ),
    }
    section_by_id["missing_information"].content["presentation"] = {
        "items": build_missing_information(report_data)
    }
    section_by_id["appendix"].content["presentation"] = {
        "source_files": [
            compact_source_label(source)
            for source in section_by_id["appendix"].content.get("source_files", [])
        ],
        "methodology": [
            "Cálculos, KPIs y anomalías fueron generados por Python a partir de salidas procesadas.",
            "El análisis estratégico fue validado antes de generar este reporte.",
            "Las fuentes completas permanecen en los artefactos JSON/CSV del pipeline.",
        ],
    }


def build_report_model(inputs: ReportInputBundle) -> ReportModel:
    """Build a renderer-agnostic report model from processed outputs.

    Inputs: report input bundle.
    Outputs: ReportModel with all required sections.
    Assumptions: business logic and calculations were completed upstream.
    """

    finance = _finance(inputs.finance_summary)
    analysis = _analysis_payload(inputs.strategic_analysis)
    historical_context = inputs.strategic_analysis.get("historical_context", {})
    historical_context = historical_context if isinstance(historical_context, dict) else {}
    analysis_warnings = _analysis_unavailable_warnings(inputs.strategic_analysis)
    budget = finance.get("budget_vs_actual", {})
    budget = budget if isinstance(budget, dict) else {}
    payments = finance.get("student_payments", {})
    payments = payments if isinstance(payments, dict) else {}
    cash_flow = finance.get("cash_flow", {})
    cash_flow = cash_flow if isinstance(cash_flow, dict) else {}
    kpi_comparisons = _build_kpi_comparisons(
        inputs,
        finance,
        budget,
        payments,
        cash_flow,
        historical_context,
    )

    finance_source = (inputs.source_files[0],)
    kpi_source = (inputs.source_files[1],)
    anomaly_source = (inputs.source_files[2],)
    evidence_source = (inputs.source_files[3],)
    analysis_source = (inputs.source_files[4],)

    report_period = str(inputs.finance_summary.get("report_period", inputs.period_slug))
    analysis_status = str(inputs.strategic_analysis.get("validation_status") or "unknown")
    executive_summary = (
        analysis.get("executive_summary")
        or _deterministic_executive_summary(
            finance=finance,
            anomaly_report=inputs.anomaly_report,
            report_period=report_period,
            analysis_status=analysis_status,
        )
    )
    strategy_recovery = inputs.strategic_analysis.get("strategic_recovery", {})
    if not isinstance(strategy_recovery, dict):
        strategy_recovery = analysis.get("_strategic_recovery", {})
    strategy_recovery = strategy_recovery if isinstance(strategy_recovery, dict) else {}
    base_sections = (
        _section(
            "cover",
            "Cover",
            {
                "title": "Finance AI Agent Report",
                "report_period": report_period,
                "period_slug": inputs.period_slug,
                "source_workbook": inputs.finance_summary.get("source_workbook"),
                "renderer_note": "Renderer-agnostic report model; no layout applied.",
            },
            finance_source,
        ),
        _section(
            "executive_summary",
            "Executive Summary",
            {
                "summary": executive_summary,
                "key_findings": analysis.get("key_findings", []),
                "root_causes": analysis.get("root_causes", []),
                "confidence": analysis.get("confidence"),
                "analysis_status": analysis_status,
                "strategy_recovery": strategy_recovery,
            },
            analysis_source,
            analysis_warnings,
        ),
        _section(
            "financial_health_overview",
            "Financial Health Overview",
            {
                "total_revenue": finance.get("total_revenue"),
                "total_expenses": finance.get("total_expenses"),
                "net_operating_result": finance.get("net_operating_result"),
                "net_cash_flow": cash_flow.get("net_cash_flow"),
                "ending_cash": cash_flow.get("ending_cash"),
                "payroll_percentage_of_revenue": finance.get("payroll_percentage_of_revenue"),
                "collection_rate": payments.get("collection_rate"),
                "kpi_comparisons": kpi_comparisons,
                "analysis": _analysis_text(analysis, "financial_health_analysis"),
            },
            finance_source,
        ),
        _section(
            "kpi_overview",
            "KPI Overview",
            {
                "kpis": list(inputs.kpi_summary),
                "analysis": _analysis_text(analysis, "kpi_analysis"),
            },
            kpi_source,
        ),
        _section(
            "revenue_analysis",
            "Revenue Analysis",
            {
                "total_revenue": finance.get("total_revenue"),
                "revenue_budget": budget.get("revenue_budget"),
                "revenue_variance": budget.get("revenue_variance"),
                "revenue_variance_pct": budget.get("revenue_variance_pct"),
                "department_summary": inputs.finance_summary.get("department_summary", []),
                "analysis": _analysis_text(analysis, "financial_health_analysis"),
            },
            finance_source,
        ),
        _section(
            "expense_analysis",
            "Expense Analysis",
            {
                "total_expenses": finance.get("total_expenses"),
                "expense_budget": budget.get("expense_budget"),
                "expense_variance": budget.get("expense_variance"),
                "expense_variance_pct": budget.get("expense_variance_pct"),
                "payroll_total": finance.get("payroll_total"),
                "category_summary": inputs.finance_summary.get("category_summary", []),
                "analysis": _analysis_text(analysis, "financial_health_analysis"),
            },
            finance_source,
        ),
        _section(
            "department_analysis",
            "Department Analysis",
            {
                "department_summary": inputs.finance_summary.get("department_summary", []),
                "department_evidence": [
                    item
                    for item in _evidence_items(inputs.evidence_package)
                    if item.get("retrieval_name") == "department_history"
                ],
                "analysis": _analysis_text(analysis, "department_analysis"),
            },
            (inputs.source_files[0], inputs.source_files[3]),
        ),
        _section(
            "anomaly_summary",
            "Anomaly Summary",
            {
                "report_period": report_period,
                "total_anomalies": inputs.anomaly_report.get("total_anomalies"),
                "anomalies_by_severity": inputs.anomaly_report.get("anomalies_by_severity", {}),
                "top_anomalies": inputs.anomaly_report.get("anomalies", [])[:10],
                "analysis": (
                    _analysis_text(analysis, "anomaly_analysis")
                    if int(inputs.anomaly_report.get("total_anomalies") or 0) > 0
                    else ""
                ),
            },
            anomaly_source,
        ),
        _section(
            "investigation_evidence",
            "Investigation Evidence",
            {
                "retrieval_summary": inputs.evidence_package.get("summary", {}),
                "evidence_items": _evidence_items(inputs.evidence_package),
            },
            evidence_source,
        ),
        _section(
            "strategic_recommendations",
            "Strategic Recommendations",
            {
                "recommendations": analysis.get("strategic_recommendations", analysis.get("recommendations", [])),
                "root_causes": analysis.get("root_causes", []),
                "strategic_priorities": analysis.get("strategic_priorities", []),
                "reasoning_summary": analysis.get("reasoning_summary", ""),
                "analysis": _analysis_text(analysis, "longitudinal_risk_analysis"),
                "strategy_recovery": strategy_recovery,
                "strategy_unavailable_note": (
                    "No hay recomendaciones estratégicas validadas para este período; "
                    "el reporte conserva los hallazgos determinísticos y la evidencia procesada."
                    if analysis_status != "accepted"
                    and not analysis.get("strategic_recommendations", analysis.get("recommendations", []))
                    else ""
                ),
            },
            analysis_source,
            analysis_warnings,
        ),
        _section(
            "missing_information",
            "Missing Information",
            {
                "missing_information": analysis.get("missing_information", []),
                "evidence_warnings": [
                    warning
                    for item in _evidence_items(inputs.evidence_package)
                    for warning in item.get("warnings", [])
                ],
                "unavailable_evidence": [
                    unavailable
                    for item in _evidence_items(inputs.evidence_package)
                    for unavailable in item.get("unavailable_data", [])
                ],
            },
            (inputs.source_files[3], inputs.source_files[4]),
            analysis_warnings,
        ),
        _section(
            "appendix",
            "Appendix",
            {
                "source_files": list(inputs.source_files),
                "calculation_warnings": inputs.finance_summary.get("calculation_warnings", []),
                "analysis_validation_errors": inputs.strategic_analysis.get("validation_errors", []),
            },
            inputs.source_files,
        ),
    )
    historical_sections = _historical_sections(historical_context, analysis, analysis_source)
    sections = (*base_sections[:-2], *historical_sections, *base_sections[-2:])
    model = ReportModel(
        report_id=f"REPORT-MODEL-{inputs.period_slug.upper().replace('_', '-')}",
        period_slug=inputs.period_slug,
        report_period=report_period,
        renderer_contract_version="1.0",
        sections=sections,
        source_references=_all_section_sources(sections),
    )
    _add_presentation_payload(model)
    validate_report_model(model.to_dict())
    presentation_result = validate_presentation_view(
        build_presentation_view(model.to_dict(), mode="executive"),
        mode="executive",
    )
    if not presentation_result.is_valid:
        raise ValueError(
            "Report presentation validation failed: "
            + "; ".join(presentation_result.errors)
        )
    return model


def validate_report_model(report_data: dict[str, Any]) -> None:
    """Validate the renderer-agnostic report model schema.

    Inputs: serialized report model.
    Outputs: None; raises ValueError when invalid.
    Assumptions: this is a lightweight internal schema check, not JSON Schema draft validation.
    """

    required_root = {
        "report_id",
        "period_slug",
        "report_period",
        "renderer_contract_version",
        "section_count",
        "sections",
        "source_references",
    }
    if set(report_data) != required_root:
        raise ValueError(f"Report model root keys are invalid: {sorted(report_data)}")
    sections = report_data["sections"]
    if not isinstance(sections, list):
        raise ValueError("Report sections must be a list")
    section_ids = [section.get("section_id") for section in sections if isinstance(section, dict)]
    missing = [section_id for section_id in REQUIRED_SECTION_IDS if section_id not in section_ids]
    if missing:
        raise ValueError(f"Report model missing required sections: {missing}")
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("Report model contains duplicate section IDs")
    if report_data["section_count"] != len(sections):
        raise ValueError("section_count does not match sections length")
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("Each report section must be an object")
        if set(section) != {"section_id", "title", "content", "source_references", "warnings"}:
            raise ValueError(f"Invalid section keys for {section.get('section_id')}")
        if not isinstance(section["content"], dict):
            raise ValueError(f"Section content must be an object: {section['section_id']}")
        if not isinstance(section["source_references"], list):
            raise ValueError(f"Section sources must be a list: {section['section_id']}")


def save_report_model(model: ReportModel, output_path: str | Path) -> Path:
    """Save a report model as readable JSON.

    Inputs: report model and output path.
    Outputs: resolved written path.
    Assumptions: parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
