"""Deterministic risk prioritization and investigation task planning."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from finance_agent.planner_models import (
    PRIORITY_ORDER,
    EvidenceRequest,
    InvestigationPlan,
    InvestigationTask,
    PriorityLevel,
)


SEVERITY_BASE_SCORE = {
    "critical": 90,
    "high": 65,
    "medium": 40,
    "low": 15,
}


@dataclass(frozen=True)
class _EvidenceSpec:
    """Internal evidence request before final task identifiers are assigned."""

    tool_name: str
    parameters: dict[str, Any]
    purpose: str


@dataclass(frozen=True)
class _TaskCandidate:
    """Internal task candidate used for deterministic sorting and ID assignment."""

    anomaly_id: str
    priority: PriorityLevel
    priority_score: int
    question_to_answer: str
    reason: str
    evidence_specs: tuple[_EvidenceSpec, ...]
    expected_output: str
    prioritization_factors: tuple[str, ...]


def severity_to_priority(severity: str) -> PriorityLevel:
    """Map anomaly severity to its baseline investigation priority.

    Inputs: critical, high, medium, or low anomaly severity.
    Outputs: corresponding PriorityLevel.
    Assumptions: unknown severities are treated conservatively as medium.
    """

    return {
        "critical": PriorityLevel.CRITICAL,
        "high": PriorityLevel.HIGH,
        "medium": PriorityLevel.MEDIUM,
        "low": PriorityLevel.LOW,
    }.get(str(severity).lower(), PriorityLevel.MEDIUM)


def _priority_from_score(score: int) -> PriorityLevel:
    """Convert a deterministic risk score to a priority level.

    Inputs: integer risk score.
    Outputs: bounded PriorityLevel.
    Assumptions: score additions represent recurrence, impact, and importance.
    """

    if score >= 90:
        return PriorityLevel.CRITICAL
    if score >= 65:
        return PriorityLevel.HIGH
    if score >= 40:
        return PriorityLevel.MEDIUM
    return PriorityLevel.LOW


def _number(value: Any) -> float | None:
    """Convert a processed scalar to float when possible.

    Inputs: JSON/CSV scalar.
    Outputs: float or None.
    Assumptions: empty and non-numeric values are unavailable evidence.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _investigation_kind(anomaly: dict[str, Any]) -> str:
    """Classify an anomaly into a stable investigation rule family.

    Inputs: serialized anomaly.
    Outputs: planner rule-family identifier.
    Assumptions: Step 4 metrics and rule IDs carry more stable meaning than prose.
    """

    text = " ".join(
        str(anomaly.get(field, "")).lower()
        for field in ("rule_id", "metric", "title")
    )
    if "overdue" in text:
        return "overdue_payments"
    if "collection" in text or "tuition" in text:
        return "collection_rate"
    if "payroll" in text:
        return "payroll"
    if "cash_flow" in text or "cash flow" in text:
        return "cash_flow"
    if "operating" in text and ("result" in text or "deficit" in text):
        return "operating_result"
    if "vendor" in text:
        return "vendor_payment"
    if "department" in text and ("overspend" in text or "variance" in text):
        return "department_overspend"
    if "category" in text and ("overspend" in text or "variance" in text):
        return "category_overspend"
    if "expense_increase" in text or "expense increase" in text:
        return "expense_trend"
    if "revenue_drop" in text or "revenue drop" in text:
        return "revenue_trend"
    return "general_financial_risk"


def _goal_is_violated(anomaly: dict[str, Any]) -> bool:
    """Determine whether observed evidence breaches its deterministic threshold.

    Inputs: serialized anomaly with rule, observed value, and threshold.
    Outputs: True when direction and values show a goal/limit violation.
    Assumptions: MAX/FLAG/REVIEW are upper limits and MIN is a lower limit.
    """

    observed = _number(anomaly.get("observed_value"))
    threshold = _number(anomaly.get("threshold_value"))
    if observed is None:
        return False
    rule_id = str(anomaly.get("rule_id", "")).upper()
    if "DEFICIT" in rule_id:
        return observed < 0
    if threshold is None:
        return False
    if any(token in rule_id for token in ("MAX", "FLAG", "REVIEW", "INCREASE")):
        return observed > threshold
    if "MIN" in rule_id:
        return observed < threshold
    if "Z_SCORE" in rule_id:
        return abs(observed) > abs(threshold)
    return False


def _financial_impact_points(
    anomaly: dict[str, Any],
    total_revenue: float | None,
) -> tuple[int, str | None]:
    """Score direct monetary impact relative to report revenue.

    Inputs: anomaly and optional calculated total revenue.
    Outputs: score addition and readable factor, or zero and None.
    Assumptions: ratios, rates, percentages, and z-scores are not currency.
    """

    metric = str(anomaly.get("metric", "")).lower()
    non_currency_tokens = ("percentage", "ratio", "rate", "pct", "z_score")
    currency_tokens = (
        "cash_flow",
        "operating_result",
        "vendor_payment",
        "expense_variance",
        "revenue_variance",
    )
    if any(token in metric for token in non_currency_tokens) or not any(
        token in metric for token in currency_tokens
    ):
        return 0, None
    observed = _number(anomaly.get("observed_value"))
    if observed is None:
        return 0, None
    impact = abs(observed)
    ratio = impact / total_revenue if total_revenue and total_revenue > 0 else 0.0
    if ratio >= 0.25:
        points = 15
    elif ratio >= 0.10:
        points = 12
    elif ratio >= 0.03 or impact >= 250_000:
        points = 8
    elif impact >= 50_000:
        points = 5
    else:
        return 0, None
    return points, f"financial_impact:{impact:.2f}"


def _operational_importance(kind: str) -> int:
    """Return operational importance points for a rule family.

    Inputs: investigation kind.
    Outputs: small deterministic score addition.
    Assumptions: liquidity and solvency outrank narrower control exceptions.
    """

    return {
        "operating_result": 10,
        "cash_flow": 10,
        "payroll": 8,
        "collection_rate": 8,
        "overdue_payments": 8,
        "department_overspend": 6,
        "vendor_payment": 6,
        "revenue_trend": 6,
        "expense_trend": 6,
        "category_overspend": 4,
    }.get(kind, 2)


def _trend_recurrence_periods(
    trends: Iterable[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, set[str]]:
    """Identify repeated risk families directly from monthly processed trends.

    Inputs: monthly trend rows and Step 4 threshold configuration.
    Outputs: rule-family to affected-period sets.
    Assumptions: trend ratios are stored as decimals while thresholds are percentages.
    """

    periods: dict[str, set[str]] = defaultdict(set)
    payroll_limit = (_number(thresholds.get("payroll_percent_max")) or 0.0) / 100
    collection_min = (
        _number(thresholds.get("tuition_collection_min_percent")) or 0.0
    ) / 100
    overdue_limit = (
        _number(thresholds.get("overdue_payment_max_percent")) or 0.0
    ) / 100
    for row in trends:
        period = str(row.get("period", "unknown"))
        checks = {
            "operating_result": (
                (_number(row.get("net_operating_result")) or 0.0) < 0
            ),
            "cash_flow": ((_number(row.get("net_cash_flow")) or 0.0) < 0),
            "payroll": (
                (_number(row.get("payroll_percentage_of_revenue")) or 0.0)
                > payroll_limit
            ),
            "collection_rate": (
                (_number(row.get("student_collection_rate")) or 0.0)
                < collection_min
            ),
            "overdue_payments": (
                (_number(row.get("overdue_payment_percentage")) or 0.0)
                > overdue_limit
            ),
        }
        for kind, is_present in checks.items():
            if is_present:
                periods[kind].add(period)
    return periods


def build_recurrence_index(
    anomalies: Iterable[dict[str, Any]],
    monthly_trends: Iterable[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Build affected-period evidence for repeated investigation issues.

    Inputs: annual anomaly records, monthly processed trends, and thresholds.
    Outputs: rule-family to sorted unique period tuple.
    Assumptions: two or more periods means the issue is repeated.
    """

    periods: dict[str, set[str]] = defaultdict(set)
    for anomaly in anomalies:
        periods[_investigation_kind(anomaly)].add(
            str(anomaly.get("period", "unknown"))
        )
    for kind, trend_periods in _trend_recurrence_periods(
        monthly_trends,
        thresholds,
    ).items():
        periods[kind].update(trend_periods)
    return {
        kind: tuple(sorted(values))
        for kind, values in periods.items()
    }


def _entity_from_title(title: str, suffix: str) -> str | None:
    """Extract a named department or category from a standard anomaly title.

    Inputs: title and trailing phrase such as ' overspending'.
    Outputs: leading entity name or None.
    Assumptions: Step 4 titles place the entity before the supplied suffix.
    """

    lowered = title.lower()
    index = lowered.find(suffix)
    return title[:index].strip() if index > 0 else None


def create_evidence_requests(
    anomaly: dict[str, Any],
    kind: str,
) -> tuple[_EvidenceSpec, ...]:
    """Define future retrieval requests for one anomaly without executing tools.

    Inputs: anomaly and its planner rule family.
    Outputs: ordered evidence specifications.
    Assumptions: references prefixed with '$' are resolved by later orchestration.
    """

    period = str(anomaly.get("period", "unknown"))
    title = str(anomaly.get("title", ""))
    common_previous = _EvidenceSpec(
        "get_previous_cycle_memory",
        {},
        "Check whether prior analysis already identified this issue and actions.",
    )
    if kind == "payroll":
        return (
            _EvidenceSpec(
                "get_payroll_history",
                {"department": "all", "months": 6},
                "Compare payroll, overtime, benefits, and headcount over time.",
            ),
            common_previous,
        )
    if kind == "department_overspend":
        department = _entity_from_title(title, " overspending") or "unknown"
        return (
            _EvidenceSpec(
                "get_department_history",
                {"department": department, "months": 6},
                "Compare department budget and actual expense history.",
            ),
            _EvidenceSpec(
                "get_transactions",
                {"department": department, "period": period},
                "Identify expense categories, vendors, and transactions driving variance.",
            ),
        )
    if kind in {"collection_rate", "overdue_payments"}:
        return (
            _EvidenceSpec(
                "get_transactions",
                {
                    "type": "student_receivable",
                    "period": period,
                    "include_aging": True,
                },
                "Inspect overdue invoices, aging, and payment-plan status.",
            ),
            common_previous,
        )
    if kind in {"cash_flow", "operating_result", "expense_trend", "revenue_trend"}:
        return (
            _EvidenceSpec(
                "get_full_report",
                {"period": period},
                "Compare revenue, expense, and cash-flow components for the period.",
            ),
            common_previous,
        )
    if kind == "vendor_payment":
        return (
            _EvidenceSpec(
                "get_transactions",
                {
                    "type": "vendor_payment",
                    "period": period,
                    "minimum_amount": anomaly.get("threshold_value"),
                },
                "Locate flagged invoices, approvals, and possible duplicates.",
            ),
            _EvidenceSpec(
                "get_vendor_history",
                {"vendor": "$flagged_transaction.vendor", "months": 12},
                "Review the selected vendor's payment pattern after identification.",
            ),
        )
    if kind == "category_overspend":
        category = _entity_from_title(title, " category overspending") or "unknown"
        return (
            _EvidenceSpec(
                "get_transactions",
                {"expense_category": category, "period": period},
                "Identify transactions and vendors behind category overspending.",
            ),
        )
    return (
        _EvidenceSpec(
            "get_full_report",
            {"period": period},
            "Retrieve the processed report context needed to investigate the flag.",
        ),
    )


def _question_and_output(
    anomaly: dict[str, Any],
    kind: str,
) -> tuple[str, str]:
    """Create a deterministic investigation question and expected deliverable.

    Inputs: anomaly and planner rule family.
    Outputs: question and expected-output description.
    Assumptions: wording requests evidence, not strategy or LLM judgment.
    """

    period = str(anomaly.get("period", "the reported period"))
    title = str(anomaly.get("title", "financial anomaly"))
    templates = {
        "payroll": (
            f"What payroll components and departments caused the payroll risk in {period}?",
            "Payroll driver breakdown by department, overtime, benefits, and headcount.",
        ),
        "department_overspend": (
            f"Which expense categories, payroll items, or vendors caused {title}?",
            "Department variance bridge with the largest contributing transactions.",
        ),
        "collection_rate": (
            f"Why did the student collection rate miss its target in {period}?",
            "Receivables aging and collection shortfall breakdown.",
        ),
        "overdue_payments": (
            f"Which invoices and aging buckets caused overdue payments in {period}?",
            "Overdue invoice list grouped by aging, department, and payment status.",
        ),
        "cash_flow": (
            f"Which operating, scholarship, or capital outflows caused cash-flow risk in {period}?",
            "Cash-flow bridge showing the dominant negative drivers.",
        ),
        "operating_result": (
            f"Which revenue shortfalls and expense drivers caused the operating deficit in {period}?",
            "Operating-result bridge by major revenue and expense driver.",
        ),
        "vendor_payment": (
            f"Does the flagged vendor payment in {period} have valid invoice, approval, and duplicate evidence?",
            "Payment control review with invoice, approval, duplicate, and vendor-history evidence.",
        ),
        "category_overspend": (
            f"Which transactions caused {title}?",
            "Category variance breakdown with top transactions and vendors.",
        ),
        "expense_trend": (
            f"What caused the unusual expense increase in {period}?",
            "Month-over-month expense driver comparison.",
        ),
        "revenue_trend": (
            f"What caused the unusual revenue decline in {period}?",
            "Month-over-month revenue driver comparison.",
        ),
    }
    return templates.get(
        kind,
        (
            f"What evidence explains the detected risk '{title}' in {period}?",
            "Evidence-backed explanation of the observed anomaly.",
        ),
    )


def _score_anomaly(
    anomaly: dict[str, Any],
    *,
    recurrence_index: dict[str, tuple[str, ...]],
    total_revenue: float | None,
    top_risk_kinds: set[str],
) -> tuple[int, tuple[str, ...]]:
    """Score one anomaly using all required deterministic priority factors.

    Inputs: anomaly, recurrence evidence, revenue scale, and annual top risks.
    Outputs: bounded score and readable factor tuple.
    Assumptions: severity is the base; other factors can escalate urgency.
    """

    severity = str(anomaly.get("severity", "medium")).lower()
    score = SEVERITY_BASE_SCORE.get(severity, SEVERITY_BASE_SCORE["medium"])
    factors = [f"severity:{severity}"]
    kind = _investigation_kind(anomaly)

    periods = recurrence_index.get(kind, ())
    if len(periods) >= 2:
        recurrence_points = 20 if len(periods) >= 3 else 15
        score += recurrence_points
        factors.append(f"repeated_across_months:{len(periods)}")

    impact_points, impact_factor = _financial_impact_points(
        anomaly,
        total_revenue,
    )
    score += impact_points
    if impact_factor:
        factors.append(impact_factor)

    if _goal_is_violated(anomaly):
        score += 8
        factors.append("goal_or_threshold_violation")

    importance = _operational_importance(kind)
    score += importance
    factors.append(f"operational_importance:{importance}")
    if kind in top_risk_kinds:
        score += 3
        factors.append("listed_in_annual_top_risks")
    return min(score, 125), tuple(factors)


def _table_matches_scope(table: dict[str, Any], period_slug: str) -> bool:
    """Match an enriched table to monthly or annual planner scope.

    Inputs: serialized enriched table and target period slug.
    Outputs: True when workbook/table provenance matches the plan.
    Assumptions: current monthly IDs contain monthly/june and annual IDs contain annual.
    """

    provenance = " ".join(
        str(table.get(field, "")).lower()
        for field in ("table_id", "source_workbook")
    )
    if period_slug == "2026":
        return "annual" in provenance
    return "monthly" in provenance or period_slug in provenance


def _data_quality_candidates(
    finance_document: dict[str, Any],
    enriched_model: dict[str, Any],
    period_slug: str,
) -> list[_TaskCandidate]:
    """Create investigation candidates for uncertain tables and missing outputs.

    Inputs: calculated finance document, enriched model, and plan scope.
    Outputs: data-quality task candidates.
    Assumptions: planner requests evidence but does not reinterpret tables itself.
    """

    candidates: list[_TaskCandidate] = []
    for table in enriched_model.get("tables", []):
        unresolved = (
            bool(table.get("requires_human_review"))
            or table.get("final_table_type") == "Unknown"
        )
        if not unresolved or not _table_matches_scope(table, period_slug):
            continue
        table_id = str(table.get("table_id", "unknown_table"))
        final_type = str(table.get("final_table_type", "Unknown"))
        unknown_type = final_type == "Unknown"
        score = 75 if unknown_type else 50
        priority = PriorityLevel.HIGH if unknown_type else PriorityLevel.MEDIUM
        candidates.append(
            _TaskCandidate(
                anomaly_id=f"DATA-TABLE-{table_id}",
                priority=priority,
                priority_score=score,
                question_to_answer=(
                    f"What is the correct structure and business meaning of table "
                    f"'{table_id}'?"
                ),
                reason=(
                    f"Enriched model requires human review; final type is "
                    f"{final_type} with confidence {table.get('final_confidence')}."
                ),
                evidence_specs=(
                    _EvidenceSpec(
                        "get_full_report",
                        {"period": finance_document.get("report_period", period_slug)},
                        "Review the table in the context of its processed report.",
                    ),
                    _EvidenceSpec(
                        "get_transactions",
                        {"table_id": table_id, "limit": 100},
                        "Inspect representative processed records before schema approval.",
                    ),
                ),
                expected_output=(
                    "Confirmed table type, canonical column mappings, and review decision."
                ),
                prioritization_factors=(
                    "missing_or_uncertain_data",
                    f"final_table_type:{final_type}",
                ),
            )
        )

    warnings = finance_document.get("calculation_warnings", [])
    if isinstance(warnings, list):
        for index, warning in enumerate(warnings, start=1):
            candidates.append(
                _TaskCandidate(
                    anomaly_id=f"DATA-WARNING-{period_slug}-{index:03d}",
                    priority=PriorityLevel.MEDIUM,
                    priority_score=45,
                    question_to_answer=(
                        f"What processed evidence is missing for calculation warning "
                        f"{index}?"
                    ),
                    reason=str(warning),
                    evidence_specs=(
                        _EvidenceSpec(
                            "get_full_report",
                            {
                                "period": finance_document.get(
                                    "report_period",
                                    period_slug,
                                )
                            },
                            "Confirm which calculated source or KPI is unavailable.",
                        ),
                    ),
                    expected_output="Resolved source gap or explicit unavailable-data decision.",
                    prioritization_factors=("missing_or_uncertain_data",),
                )
            )
    return candidates


def _top_risk_kinds(risk_summary: dict[str, Any]) -> set[str]:
    """Extract planner rule families represented in the annual top-risk list.

    Inputs: annual risk summary.
    Outputs: set of stable investigation kinds.
    Assumptions: compact top risks contain metric/title fields from Step 4.
    """

    return {
        _investigation_kind(risk)
        for risk in risk_summary.get("top_risks", [])
        if isinstance(risk, dict)
    }


def build_investigation_plan(
    *,
    finance_document: dict[str, Any],
    anomaly_report: dict[str, Any],
    monthly_trends: Iterable[dict[str, Any]],
    recurrence_anomalies: Iterable[dict[str, Any]],
    enriched_model: dict[str, Any],
    risk_summary: dict[str, Any],
    period_slug: str,
    source_files: Iterable[str],
) -> InvestigationPlan:
    """Build a prioritized investigation plan from processed outputs only.

    Inputs: finance/anomaly outputs, trends, enriched model, annual risk context,
    period slug, and source filenames.
    Outputs: immutable deterministic InvestigationPlan.
    Assumptions: no evidence tools, databases, raw files, or LLMs are called here.
    """

    thresholds = anomaly_report.get("thresholds", {})
    recurrence_index = build_recurrence_index(
        recurrence_anomalies,
        monthly_trends,
        thresholds if isinstance(thresholds, dict) else {},
    )
    finance_summary = finance_document.get("finance_summary", {})
    total_revenue = _number(
        finance_summary.get("total_revenue")
        if isinstance(finance_summary, dict)
        else None
    )
    top_kinds = _top_risk_kinds(risk_summary)
    candidates: list[_TaskCandidate] = []

    for anomaly in anomaly_report.get("anomalies", []):
        if not isinstance(anomaly, dict):
            continue
        kind = _investigation_kind(anomaly)
        score, factors = _score_anomaly(
            anomaly,
            recurrence_index=recurrence_index,
            total_revenue=total_revenue,
            top_risk_kinds=top_kinds,
        )
        question, expected_output = _question_and_output(anomaly, kind)
        candidates.append(
            _TaskCandidate(
                anomaly_id=str(anomaly.get("anomaly_id", "UNKNOWN-ANOMALY")),
                priority=_priority_from_score(score),
                priority_score=score,
                question_to_answer=question,
                reason=str(
                    anomaly.get("evidence")
                    or anomaly.get("description")
                    or anomaly.get("title")
                    or "Anomaly requires investigation."
                ),
                evidence_specs=create_evidence_requests(anomaly, kind),
                expected_output=expected_output,
                prioritization_factors=factors,
            )
        )

    candidates.extend(
        _data_quality_candidates(
            finance_document,
            enriched_model,
            period_slug,
        )
    )
    candidates.sort(
        key=lambda candidate: (
            -PRIORITY_ORDER[candidate.priority],
            -candidate.priority_score,
            candidate.anomaly_id,
            candidate.question_to_answer,
        )
    )

    task_prefix = f"INV-{period_slug.upper().replace('_', '-')}"
    tasks: list[InvestigationTask] = []
    for task_index, candidate in enumerate(candidates, start=1):
        task_id = f"{task_prefix}-{task_index:03d}"
        requests = tuple(
            EvidenceRequest(
                request_id=f"{task_id}-E{evidence_index:02d}",
                tool_name=spec.tool_name,
                parameters=spec.parameters,
                purpose=spec.purpose,
            )
            for evidence_index, spec in enumerate(
                candidate.evidence_specs,
                start=1,
            )
        )
        tasks.append(
            InvestigationTask(
                task_id=task_id,
                anomaly_id=candidate.anomaly_id,
                priority=candidate.priority,
                priority_score=candidate.priority_score,
                question_to_answer=candidate.question_to_answer,
                reason=candidate.reason,
                required_evidence=requests,
                suggested_tool=requests[0].tool_name,
                expected_output=candidate.expected_output,
                prioritization_factors=candidate.prioritization_factors,
            )
        )

    report_period = str(
        anomaly_report.get("report_period")
        or finance_document.get("report_period")
        or period_slug
    )
    return InvestigationPlan(
        plan_id=f"PLAN-{period_slug.upper().replace('_', '-')}",
        report_period=report_period,
        period_slug=period_slug,
        source_files=tuple(source_files),
        tasks=tuple(tasks),
    )


def validate_plan_schema(plan_data: dict[str, Any]) -> None:
    """Validate the serialized investigation-plan output contract.

    Inputs: serialized plan dictionary.
    Outputs: none; raises ValueError for invalid shape.
    Assumptions: validation protects downstream orchestrator consumers.
    """

    required_plan_fields = {
        "plan_id",
        "report_period",
        "period_slug",
        "source_files",
        "total_tasks",
        "tasks_by_priority",
        "tasks",
    }
    if not required_plan_fields.issubset(plan_data):
        raise ValueError("Investigation plan is missing required top-level fields.")
    tasks = plan_data.get("tasks")
    if not isinstance(tasks, list) or plan_data.get("total_tasks") != len(tasks):
        raise ValueError("Investigation plan task count is invalid.")
    required_task_fields = {
        "task_id",
        "anomaly_id",
        "priority",
        "question_to_answer",
        "reason",
        "required_evidence",
        "suggested_tool",
        "expected_output",
        "status",
    }
    for task in tasks:
        if not isinstance(task, dict) or not required_task_fields.issubset(task):
            raise ValueError("Investigation task is missing required fields.")
        if task["status"] != "planned":
            raise ValueError("New investigation tasks must have planned status.")
        evidence = task["required_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Every task must contain at least one evidence request.")
        for request in evidence:
            if not {
                "request_id",
                "tool_name",
                "parameters",
                "purpose",
            }.issubset(request):
                raise ValueError("Evidence request is missing required fields.")


def save_investigation_plan(
    plan: InvestigationPlan,
    output_path: str | Path,
) -> Path:
    """Validate and save one deterministic investigation plan.

    Inputs: plan and destination path.
    Outputs: resolved written path.
    Assumptions: output parent may not exist yet.
    """

    data = plan.to_dict()
    validate_plan_schema(data)
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def build_planner_summary(
    plans: Iterable[InvestigationPlan],
    plan_files: Iterable[str],
) -> dict[str, Any]:
    """Build a compact summary across generated investigation plans.

    Inputs: generated plans and their output filenames.
    Outputs: JSON-compatible aggregate counts and top questions.
    Assumptions: incoming plans are already prioritized.
    """

    plan_list = list(plans)
    all_tasks = [task for plan in plan_list for task in plan.tasks]
    return {
        "reporting_year": "2026",
        "plan_count": len(plan_list),
        "total_tasks": len(all_tasks),
        "tasks_by_priority": {
            priority.value: sum(task.priority == priority for task in all_tasks)
            for priority in PriorityLevel
        },
        "human_review_task_count": sum(
            task.anomaly_id.startswith("DATA-") for task in all_tasks
        ),
        "plan_files": list(plan_files),
        "top_5_investigation_questions": [
            task.question_to_answer
            for task in sorted(
                all_tasks,
                key=lambda task: (
                    -PRIORITY_ORDER[task.priority],
                    -task.priority_score,
                    task.task_id,
                ),
            )[:5]
        ],
        "llm_used": False,
        "evidence_retrieved": False,
    }


def save_planner_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save the aggregate planner summary.

    Inputs: summary dictionary and destination path.
    Outputs: resolved written path.
    Assumptions: summary was built from validated plan objects.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
