"""Tests for deterministic investigation planning and evidence requests."""

from __future__ import annotations

from finance_agent.investigation_planner import (
    build_investigation_plan,
    severity_to_priority,
    validate_plan_schema,
)
from finance_agent.planner_models import PriorityLevel


def _anomaly(
    *,
    anomaly_id: str = "ANOM-TEST-001",
    severity: str = "high",
    rule_id: str = "PAYROLL_RATIO_MAX",
    metric: str = "payroll_percentage_of_revenue",
    period: str = "2026-06",
    title: str = "Payroll exceeds revenue threshold",
    observed: float = 53.0,
    threshold: float = 42.0,
) -> dict[str, object]:
    """Build one serialized anomaly fixture.

    Inputs: configurable anomaly identity, risk, metric, and threshold evidence.
    Outputs: Step 4-compatible anomaly dictionary.
    Assumptions: omitted fields contain representative planner values.
    """

    return {
        "anomaly_id": anomaly_id,
        "title": title,
        "description": title,
        "metric": metric,
        "observed_value": observed,
        "threshold_value": threshold,
        "severity": severity,
        "period": period,
        "source_file": "finance_summary_2026.json",
        "evidence": f"Observed {observed} versus {threshold}.",
        "recommended_next_check": "Review supporting records.",
        "detection_method": "rule_based",
        "rule_id": rule_id,
    }


def _finance_document() -> dict[str, object]:
    """Build a minimal processed finance-summary fixture.

    Inputs: none.
    Outputs: finance document with revenue and no warnings.
    Assumptions: only planner-consumed fields are needed.
    """

    return {
        "report_period": "2026",
        "finance_summary": {"total_revenue": 2_000_000},
        "calculation_warnings": [],
    }


def _build_plan(
    anomalies: list[dict[str, object]],
    *,
    recurrence: list[dict[str, object]] | None = None,
    enriched_model: dict[str, object] | None = None,
):
    """Build a plan with compact default test inputs.

    Inputs: plan anomalies plus optional recurrence and enriched-model evidence.
    Outputs: InvestigationPlan.
    Assumptions: annual scope is sufficient for planner unit tests.
    """

    return build_investigation_plan(
        finance_document=_finance_document(),
        anomaly_report={
            "report_period": "2026",
            "thresholds": {
                "payroll_percent_max": 42,
                "tuition_collection_min_percent": 94,
                "overdue_payment_max_percent": 6,
            },
            "anomalies": anomalies,
        },
        monthly_trends=[],
        recurrence_anomalies=recurrence or anomalies,
        enriched_model=enriched_model or {"tables": []},
        risk_summary={"top_risks": []},
        period_slug="2026",
        source_files=["finance_summary_2026.json", "anomaly_report_2026.json"],
    )


def test_severity_to_priority_mapping() -> None:
    """Verify every Step 4 severity has the expected baseline priority."""

    assert severity_to_priority("critical") is PriorityLevel.CRITICAL
    assert severity_to_priority("high") is PriorityLevel.HIGH
    assert severity_to_priority("medium") is PriorityLevel.MEDIUM
    assert severity_to_priority("low") is PriorityLevel.LOW


def test_high_risk_anomaly_generates_planned_task() -> None:
    """Verify a high-risk anomaly becomes an evidence-backed planned task."""

    plan = _build_plan([_anomaly()])
    task = plan.tasks[0]

    assert task.anomaly_id == "ANOM-TEST-001"
    assert task.priority in {PriorityLevel.HIGH, PriorityLevel.CRITICAL}
    assert task.status == "planned"
    assert "payroll" in task.question_to_answer.lower()
    assert task.required_evidence


def test_repeated_issue_escalates_priority() -> None:
    """Verify the same issue across months receives a higher priority score."""

    current = _anomaly(severity="high")
    single_plan = _build_plan([current], recurrence=[current])
    repeated = [
        _anomaly(anomaly_id=f"ANOM-{month}", period=f"2026-{month}")
        for month in ("06", "07", "08")
    ]
    repeated_plan = _build_plan([current], recurrence=repeated)

    assert repeated_plan.tasks[0].priority_score > single_plan.tasks[0].priority_score
    assert repeated_plan.tasks[0].priority is PriorityLevel.CRITICAL
    assert any(
        factor.startswith("repeated_across_months:")
        for factor in repeated_plan.tasks[0].prioritization_factors
    )


def test_uncertain_table_generates_data_review_task() -> None:
    """Verify human-review tables become scoped data-quality investigations."""

    enriched_model = {
        "tables": [
            {
                "table_id": "annual_report__unknown__table_01",
                "source_workbook": "annual_financial_report_2026.xlsx",
                "final_table_type": "Unknown",
                "final_confidence": 0.30,
                "requires_human_review": True,
            }
        ]
    }

    plan = _build_plan([], enriched_model=enriched_model)
    task = plan.tasks[0]

    assert task.anomaly_id.startswith("DATA-TABLE-")
    assert task.priority is PriorityLevel.HIGH
    assert "correct structure" in task.question_to_answer
    assert "missing_or_uncertain_data" in task.prioritization_factors


def test_department_task_creates_specific_evidence_requests() -> None:
    """Verify department overspending defines history and transaction requests."""

    anomaly = _anomaly(
        rule_id="DEPARTMENT_OVERSPEND_FLAG",
        metric="department_expense_variance_pct",
        title="Engineering overspending exceeds flag threshold",
        observed=18.0,
        threshold=12.0,
    )

    plan = _build_plan([anomaly])
    task = plan.tasks[0]

    assert task.suggested_tool == "get_department_history"
    assert [request.tool_name for request in task.required_evidence] == [
        "get_department_history",
        "get_transactions",
    ]
    assert task.required_evidence[0].parameters["department"] == "Engineering"


def test_output_schema_is_valid() -> None:
    """Verify serialized plans satisfy the downstream output contract."""

    plan = _build_plan([_anomaly()])
    serialized = plan.to_dict()

    validate_plan_schema(serialized)
    task = serialized["tasks"][0]
    assert {
        "task_id",
        "anomaly_id",
        "priority",
        "question_to_answer",
        "reason",
        "required_evidence",
        "suggested_tool",
        "expected_output",
        "status",
    }.issubset(task)
    assert task["status"] == "planned"
