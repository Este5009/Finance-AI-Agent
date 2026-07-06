"""Primary Ollama investigation planning with deterministic fallback and queues."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from finance_agent.ollama_client import OllamaError
from finance_agent.planner_models import InvestigationPlan
from finance_agent.planner_validation import (
    MAX_PLAN_STEPS,
    TOOL_INTERFACES,
    PlanValidationResult,
    validate_ollama_plan_response,
)


class OllamaPlannerClient(Protocol):
    """Minimal client contract required by the Step 7 planner."""

    def is_available(self) -> bool:
        """Return whether the local model service can be reached."""

    def generate(self, prompt: str) -> str:
        """Return one model-generated strict-JSON plan."""


@dataclass(frozen=True)
class OllamaPlannerResult:
    """Validated primary plan, fallback status, and unexecuted queue."""

    plan_document: dict[str, Any]
    execution_queue: dict[str, Any]
    ollama_plan_accepted: bool
    fallback_used: bool
    validation_errors: tuple[str, ...]


def _compact_finance_summary(finance_document: dict[str, Any]) -> dict[str, Any]:
    """Select decision-relevant calculated values without copying full reports.

    Inputs: Step 3 finance summary document.
    Outputs: compact headline finance and KPI summary.
    Assumptions: Python-calculated values are authoritative and remain unmodified.
    """

    finance = finance_document.get("finance_summary", {})
    finance = finance if isinstance(finance, dict) else {}
    budget = finance.get("budget_vs_actual", {})
    payments = finance.get("student_payments", {})
    cash = finance.get("cash_flow", {})
    return {
        "report_period": finance_document.get("report_period"),
        "headline_finance": {
            "total_revenue": finance.get("total_revenue"),
            "total_expenses": finance.get("total_expenses"),
            "net_operating_result": finance.get("net_operating_result"),
            "payroll_total": finance.get("payroll_total"),
            "payroll_percentage_of_revenue": finance.get(
                "payroll_percentage_of_revenue"
            ),
            "revenue_variance": (
                budget.get("revenue_variance")
                if isinstance(budget, dict)
                else None
            ),
            "expense_variance": (
                budget.get("expense_variance")
                if isinstance(budget, dict)
                else None
            ),
            "collection_rate": (
                payments.get("collection_rate")
                if isinstance(payments, dict)
                else None
            ),
            "overdue_invoice_count": (
                payments.get("overdue_invoice_count")
                if isinstance(payments, dict)
                else None
            ),
            "net_cash_flow": (
                cash.get("net_cash_flow") if isinstance(cash, dict) else None
            ),
            "ending_cash": (
                cash.get("ending_cash") if isinstance(cash, dict) else None
            ),
        },
        "kpi_summary": [
            {
                "metric": item.get("metric"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "availability": item.get("availability"),
            }
            for item in finance_document.get("kpi_summary", [])[:20]
            if isinstance(item, dict)
        ],
        "calculation_warnings": [
            str(warning)[:240]
            for warning in finance_document.get("calculation_warnings", [])[:10]
        ],
    }


def _compact_anomaly_report(anomaly_report: dict[str, Any]) -> dict[str, Any]:
    """Compress anomaly evidence to scalar risk facts for planning.

    Inputs: Step 4 anomaly report.
    Outputs: counts, thresholds, and bounded anomaly records.
    Assumptions: descriptions and source file paths are unnecessary for planning.
    """

    return {
        "report_period": anomaly_report.get("report_period"),
        "total_anomalies": anomaly_report.get("total_anomalies"),
        "anomalies_by_severity": anomaly_report.get("anomalies_by_severity", {}),
        "thresholds": anomaly_report.get("thresholds", {}),
        "anomalies": [
            {
                "anomaly_id": anomaly.get("anomaly_id"),
                "title": anomaly.get("title"),
                "metric": anomaly.get("metric"),
                "observed_value": anomaly.get("observed_value"),
                "threshold_value": anomaly.get("threshold_value"),
                "severity": anomaly.get("severity"),
                "period": anomaly.get("period"),
                "evidence": str(anomaly.get("evidence", ""))[:260],
                "rule_id": anomaly.get("rule_id"),
            }
            for anomaly in anomaly_report.get("anomalies", [])[:MAX_PLAN_STEPS]
            if isinstance(anomaly, dict)
        ],
    }


def _compact_risk_summary(risk_summary: dict[str, Any]) -> dict[str, Any]:
    """Compress the annual risk summary for primary planner context.

    Inputs: Step 4 annual risk summary.
    Outputs: counts and bounded top-risk entries.
    Assumptions: detector scope and full source metadata are not needed.
    """

    return {
        "total_anomalies": risk_summary.get("total_anomalies"),
        "high_priority_count": risk_summary.get("high_priority_count"),
        "anomalies_by_severity": risk_summary.get("anomalies_by_severity", {}),
        "top_risks": [
            {
                "anomaly_id": risk.get("anomaly_id"),
                "title": risk.get("title"),
                "severity": risk.get("severity"),
                "period": risk.get("period"),
                "metric": risk.get("metric"),
            }
            for risk in risk_summary.get("top_risks", [])[:10]
            if isinstance(risk, dict)
        ],
    }


def _table_matches_scope(table: dict[str, Any], period_slug: str) -> bool:
    """Match an enriched table to the requested monthly or annual scope.

    Inputs: enriched table metadata and planner period slug.
    Outputs: True when provenance belongs to the target plan.
    Assumptions: current IDs/provenance retain monthly or annual labels.
    """

    provenance = " ".join(
        str(table.get(field, "")).lower()
        for field in ("table_id", "source_workbook")
    )
    if period_slug == "2026":
        return "annual" in provenance
    return "monthly" in provenance or period_slug in provenance


def _compact_enriched_model(
    enriched_model: dict[str, Any],
    period_slug: str,
) -> dict[str, Any]:
    """Expose only unresolved table metadata, never rows or full mappings.

    Inputs: Step 5 enriched model and plan scope.
    Outputs: compact list of uncertain table identities and confidence.
    Assumptions: resolved table detail is unnecessary for investigation ordering.
    """

    tables = [
        {
            "table_id": table.get("table_id"),
            "sheet": table.get("sheet"),
            "final_table_type": table.get("final_table_type"),
            "final_confidence": table.get("final_confidence"),
            "requires_human_review": table.get("requires_human_review"),
            "llm_suggested_type": table.get("llm_suggested_type"),
        }
        for table in enriched_model.get("tables", [])
        if isinstance(table, dict)
        and _table_matches_scope(table, period_slug)
        and (
            table.get("requires_human_review")
            or table.get("final_table_type") == "Unknown"
        )
    ]
    return {
        "unresolved_table_count": len(tables),
        "unresolved_tables": tables[:10],
    }


def _compact_baseline(plan: InvestigationPlan) -> dict[str, Any]:
    """Compress the deterministic plan into validation/baseline context.

    Inputs: Step 6 deterministic plan.
    Outputs: top questions, priorities, and suggested interfaces.
    Assumptions: Ollama may reprioritize but should not ignore Python risk evidence.
    """

    return {
        "total_tasks": len(plan.tasks),
        "tasks": [
            {
                "anomaly_id": task.anomaly_id,
                "priority": task.priority.value,
                "priority_score": task.priority_score,
                "question": task.question_to_answer,
                "suggested_tool": task.suggested_tool,
            }
            for task in plan.tasks[:MAX_PLAN_STEPS]
        ],
    }


def build_ollama_planner_prompt(
    *,
    finance_document: dict[str, Any],
    anomaly_report: dict[str, Any],
    risk_summary: dict[str, Any],
    enriched_model: dict[str, Any],
    baseline_plan: InvestigationPlan,
    period_slug: str,
) -> str:
    """Build a bounded planning prompt from compressed processed summaries.

    Inputs: prior-stage outputs, baseline plan, and target scope.
    Outputs: strict-JSON planning prompt.
    Assumptions: no full report, normalized table, transaction, or sample row is sent.
    """

    context = {
        "period_slug": period_slug,
        "finance_and_kpis": _compact_finance_summary(finance_document),
        "anomaly_summary": _compact_anomaly_report(anomaly_report),
        "annual_risk_summary": _compact_risk_summary(risk_summary),
        "data_quality_summary": _compact_enriched_model(
            enriched_model,
            period_slug,
        ),
        "deterministic_baseline": _compact_baseline(baseline_plan),
        "available_tool_interfaces": TOOL_INTERFACES,
        "maximum_investigation_steps": MAX_PLAN_STEPS,
    }
    schema_example = {
        "investigation_steps": [
            {
                "step_id": "STEP-001",
                "anomaly_id": None,
                "priority": "high",
                "question": "What evidence explains the prioritized financial risk?",
                "tool_name": "get_full_report",
                "arguments": {"period": "2026-06"},
                "reasoning": "The processed report is needed to isolate the drivers.",
                "expected_output": "Evidence breakdown of the primary risk drivers.",
            }
        ]
    }
    return (
        "PLANNING_CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nINSTRUCTIONS_AFTER_CONTEXT:\n"
        "You are the primary investigation planner for a financial analysis "
        "system. Decide what evidence should be retrieved next. Do not calculate "
        "financial values, provide strategy, execute tools, or invent tool "
        "interfaces. Use only the supplied processed summaries. Prioritize severe, "
        "high-impact, repeated, goal-breaching, operational, and data-quality "
        "risks. For anomaly_id, copy a supplied identifier character-for-character "
        "from anomaly_summary.anomalies[].anomaly_id, use DATA-TABLE- followed by "
        "an exact supplied unresolved table_id, or use null for a synthesized "
        "cross-cutting step. Never shorten, rename, or invent an identifier. "
        "Return strict JSON only, with exactly the "
        "root key investigation_steps and exactly the step fields shown. Every "
        "tool_name/arguments pair must be unique. Keep the plan focused and within "
        f"{MAX_PLAN_STEPS} steps. Before responding, group proposed steps by exact "
        "tool_name and arguments and emit each equivalent call only once. Consolidate "
        "all questions supported by that call into one cross-cutting step. In "
        "particular, get_previous_cycle_memory with empty arguments may appear at "
        "most once, and get_full_report may appear at most once per period. "
        "Keep question and expected_output under 160 characters each. Keep "
        "reasoning short and factual, under 180 characters. "
        "Do not copy an anomaly object as the response. "
        "The example below demonstrates shape only; replace its contents with the "
        "best plan for the context. Your response must begin with "
        '{"investigation_steps":[ and end with ]}.\n'
        "VALID_RESPONSE_SHAPE:\n"
        + json.dumps(schema_example, ensure_ascii=False, separators=(",", ":"))
    )


def _allowed_source_ids(
    anomaly_report: dict[str, Any],
    enriched_model: dict[str, Any],
    period_slug: str,
) -> set[str]:
    """Build the source identifiers Ollama may reference.

    Inputs: anomaly report, enriched model, and target scope.
    Outputs: allowed anomaly/data-quality identifier set.
    Assumptions: null remains valid for cross-cutting synthesized questions.
    """

    identifiers = {
        str(anomaly.get("anomaly_id"))
        for anomaly in anomaly_report.get("anomalies", [])
        if isinstance(anomaly, dict) and anomaly.get("anomaly_id")
    }
    identifiers.update(
        f"DATA-TABLE-{table.get('table_id')}"
        for table in enriched_model.get("tables", [])
        if isinstance(table, dict)
        and table.get("table_id")
        and _table_matches_scope(table, period_slug)
        and (
            table.get("requires_human_review")
            or table.get("final_table_type") == "Unknown"
        )
    )
    return identifiers


def _baseline_steps(plan: InvestigationPlan) -> tuple[dict[str, Any], ...]:
    """Convert trusted deterministic tasks to the Step 7 plan-step shape.

    Inputs: Step 6 baseline plan.
    Outputs: fallback investigation steps.
    Assumptions: the first required evidence request is the task's primary next call.
    """

    steps: list[dict[str, Any]] = []
    for index, task in enumerate(plan.tasks, start=1):
        primary_request = task.required_evidence[0]
        arguments = dict(primary_request.parameters)
        if primary_request.tool_name == "get_transactions":
            # Step 6 stores readable flat filters; Step 7's public tool interface
            # wraps them in one explicit filters object.
            arguments = {"filters": arguments}
        steps.append(
            {
                "step_id": f"FALLBACK-{index:03d}",
                "anomaly_id": task.anomaly_id,
                "priority": task.priority.value,
                "question": task.question_to_answer,
                "tool_name": primary_request.tool_name,
                "arguments": arguments,
                "reasoning": (
                    f"Deterministic baseline selected this task: {task.reason}"
                )[:500],
                "expected_output": task.expected_output,
            }
        )
    return tuple(steps)


def _build_plan_document(
    *,
    period_slug: str,
    report_period: str,
    planner_source: str,
    ollama_available: bool,
    validation_status: str,
    validation_errors: tuple[str, ...],
    deduplicated_tool_calls: int,
    repaired_text_fields: int,
    baseline_plan: InvestigationPlan,
    steps: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Assemble the auditable primary/fallback plan output.

    Inputs: run metadata, baseline, validation result, and selected steps.
    Outputs: JSON-compatible Ollama plan document.
    Assumptions: selected steps have either passed validation or come from Python.
    """

    return {
        "plan_id": f"OLLAMA-PLAN-{period_slug.upper().replace('_', '-')}",
        "report_period": report_period,
        "period_slug": period_slug,
        "planner_source": planner_source,
        "ollama_available": ollama_available,
        "validation_status": validation_status,
        "fallback_used": planner_source == "deterministic_fallback",
        "validation_errors": list(validation_errors),
        "deduplicated_tool_calls": deduplicated_tool_calls,
        "repaired_text_fields": repaired_text_fields,
        "maximum_ollama_plan_steps": MAX_PLAN_STEPS,
        "deterministic_baseline_task_count": len(baseline_plan.tasks),
        "total_steps": len(steps),
        "investigation_steps": list(steps),
    }


def build_execution_queue(plan_document: dict[str, Any]) -> dict[str, Any]:
    """Convert a selected plan into an ordered, unexecuted tool-call queue.

    Inputs: accepted Ollama or deterministic-fallback plan document.
    Outputs: JSON-compatible queue whose items all have queued status.
    Assumptions: queue construction performs no imports or tool calls.
    """

    items = [
        {
            "queue_position": index,
            "execution_id": (
                f"EXEC-{plan_document['period_slug'].upper().replace('_', '-')}"
                f"-{index:03d}"
            ),
            "step_id": step["step_id"],
            "anomaly_id": step["anomaly_id"],
            "priority": step["priority"],
            "question": step["question"],
            "tool_name": step["tool_name"],
            "arguments": step["arguments"],
            "reasoning": step["reasoning"],
            "expected_output": step["expected_output"],
            "status": "queued",
            "dependencies": [],
        }
        for index, step in enumerate(
            plan_document["investigation_steps"],
            start=1,
        )
    ]
    return {
        "queue_id": (
            f"QUEUE-{plan_document['period_slug'].upper().replace('_', '-')}"
        ),
        "period_slug": plan_document["period_slug"],
        "source_plan_id": plan_document["plan_id"],
        "planner_source": plan_document["planner_source"],
        "status": "pending",
        "tools_executed": False,
        "total_items": len(items),
        "items": items,
    }


def create_ollama_investigation_plan(
    *,
    client: OllamaPlannerClient,
    finance_document: dict[str, Any],
    anomaly_report: dict[str, Any],
    risk_summary: dict[str, Any],
    enriched_model: dict[str, Any],
    baseline_plan: InvestigationPlan,
    period_slug: str,
) -> OllamaPlannerResult:
    """Create, validate, and queue a primary Ollama plan or Python fallback.

    Inputs: Ollama client, compressed-source artifacts, baseline, and scope.
    Outputs: auditable plan result and unexecuted execution queue.
    Assumptions: any unavailability, request error, or validation error falls back.
    """

    available = client.is_available()
    validation: PlanValidationResult | None = None
    errors: tuple[str, ...] = ()
    deduplicated_tool_calls = 0
    repaired_text_fields = 0
    if available:
        prompt = build_ollama_planner_prompt(
            finance_document=finance_document,
            anomaly_report=anomaly_report,
            risk_summary=risk_summary,
            enriched_model=enriched_model,
            baseline_plan=baseline_plan,
            period_slug=period_slug,
        )
        try:
            response = client.generate(prompt)
            validation = validate_ollama_plan_response(
                response,
                allowed_source_ids=_allowed_source_ids(
                    anomaly_report,
                    enriched_model,
                    period_slug,
                ),
            )
            errors = validation.errors
            deduplicated_tool_calls = validation.deduplicated_steps
            repaired_text_fields = validation.repaired_text_fields
        except OllamaError as exc:
            errors = (str(exc),)

    if validation is not None and validation.is_valid:
        steps = validation.steps
        planner_source = "ollama"
        validation_status = "accepted"
    else:
        steps = _baseline_steps(baseline_plan)
        planner_source = "deterministic_fallback"
        validation_status = "rejected" if available else "unavailable"
        if not available:
            errors = ("Ollama is unavailable.",)

    plan_document = _build_plan_document(
        period_slug=period_slug,
        report_period=baseline_plan.report_period,
        planner_source=planner_source,
        ollama_available=available,
        validation_status=validation_status,
        validation_errors=errors,
        deduplicated_tool_calls=deduplicated_tool_calls,
        repaired_text_fields=repaired_text_fields,
        baseline_plan=baseline_plan,
        steps=steps,
    )
    return OllamaPlannerResult(
        plan_document=plan_document,
        execution_queue=build_execution_queue(plan_document),
        ollama_plan_accepted=planner_source == "ollama",
        fallback_used=planner_source == "deterministic_fallback",
        validation_errors=errors,
    )


def save_json_artifact(data: dict[str, Any], output_path: str | Path) -> Path:
    """Save one plan or queue artifact as strict readable JSON.

    Inputs: JSON-compatible data and destination path.
    Outputs: resolved written path.
    Assumptions: output parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
