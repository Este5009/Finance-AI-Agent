"""Rule-based anomaly detection over Step 3 calculation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finance_agent.anomalies.anomaly_config import AnomalyThresholds
from finance_agent.anomalies.anomaly_loader import CalculationOutputBundle
from finance_agent.anomalies.anomaly_models import Anomaly, AnomalyIdGenerator
from finance_agent.anomalies.anomaly_severity import (
    severity_for_lower_threshold,
    severity_for_negative_value,
    severity_for_threshold_multiple,
    severity_for_upper_threshold,
)


def _number(value: object) -> float | None:
    """Convert a calculation output scalar to float when possible.

    Inputs: JSON/CSV scalar value.
    Outputs: float or None.
    Assumptions: invalid and missing values must not trigger fabricated anomalies.
    """

    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _make_anomaly(
    generator: AnomalyIdGenerator,
    *,
    title: str,
    description: str,
    metric: str,
    observed_value: float | int | str | None,
    threshold_value: float | int | str | None,
    severity: str,
    period: str,
    source_file: str,
    evidence: str,
    recommended_next_check: str,
    rule_id: str,
) -> Anomaly:
    """Build one rule-based anomaly with a generated identifier.

    Inputs: generator and all required anomaly evidence fields.
    Outputs: immutable Anomaly record.
    Assumptions: this module always labels its method as rule_based.
    """

    return Anomaly(
        anomaly_id=generator.next_id(),
        title=title,
        description=description,
        metric=metric,
        observed_value=observed_value,
        threshold_value=threshold_value,
        severity=severity,
        period=period,
        source_file=source_file,
        evidence=evidence,
        recommended_next_check=recommended_next_check,
        detection_method="rule_based",
        rule_id=rule_id,
    )


def _detect_summary_rules(
    bundle: CalculationOutputBundle,
    thresholds: AnomalyThresholds,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Detect headline KPI, operating, cash, and vendor threshold breaches.

    Inputs: calculation bundle, thresholds, and ID generator.
    Outputs: headline rule anomalies.
    Assumptions: stored ratios are decimals and thresholds are percentage points.
    """

    anomalies: list[Anomaly] = []
    finance = bundle.finance_summary
    period = bundle.report_period
    source_file = Path(bundle.finance_summary_path).name

    payroll_ratio = _number(finance.get("payroll_percentage_of_revenue"))
    if payroll_ratio is not None:
        observed = payroll_ratio * 100
        if observed > thresholds.payroll_percent_max:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title="Payroll exceeds revenue threshold",
                    description=(
                        "Payroll cost is above the configured maximum share of revenue."
                    ),
                    metric="payroll_percentage_of_revenue",
                    observed_value=observed,
                    threshold_value=thresholds.payroll_percent_max,
                    severity=severity_for_upper_threshold(
                        observed,
                        thresholds.payroll_percent_max,
                    ),
                    period=period,
                    source_file=source_file,
                    evidence=(
                        f"Calculated payroll/revenue is {observed:.2f}% "
                        f"versus a {thresholds.payroll_percent_max:.2f}% maximum."
                    ),
                    recommended_next_check=(
                        "Review payroll by department, overtime, benefits, and headcount."
                    ),
                    rule_id="PAYROLL_RATIO_MAX",
                )
            )

    student = finance.get("student_payments") or {}
    collection_ratio = _number(student.get("collection_rate"))
    if collection_ratio is not None:
        observed = collection_ratio * 100
        if observed < thresholds.tuition_collection_min_percent:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title="Tuition collection below target",
                    description=(
                        "Student payment collections are below the configured minimum."
                    ),
                    metric="student_payment_collection_rate",
                    observed_value=observed,
                    threshold_value=thresholds.tuition_collection_min_percent,
                    severity=severity_for_lower_threshold(
                        observed,
                        thresholds.tuition_collection_min_percent,
                    ),
                    period=period,
                    source_file=source_file,
                    evidence=(
                        f"Collection rate is {observed:.2f}% from "
                        f"${student.get('amount_paid', 0):,.0f} paid against "
                        f"${student.get('amount_due', 0):,.0f} due."
                    ),
                    recommended_next_check=(
                        "Inspect overdue invoices, aging buckets, and payment plans."
                    ),
                    rule_id="TUITION_COLLECTION_MIN",
                )
            )

    overdue_ratio = _number(student.get("overdue_payment_percentage"))
    if overdue_ratio is not None:
        observed = overdue_ratio * 100
        if observed > thresholds.overdue_payment_max_percent:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title="Overdue student payments above limit",
                    description=(
                        "The share of overdue student invoices exceeds policy."
                    ),
                    metric="overdue_payment_percentage",
                    observed_value=observed,
                    threshold_value=thresholds.overdue_payment_max_percent,
                    severity=severity_for_upper_threshold(
                        observed,
                        thresholds.overdue_payment_max_percent,
                    ),
                    period=period,
                    source_file=source_file,
                    evidence=(
                        f"{student.get('overdue_invoice_count', 0)} of "
                        f"{student.get('invoice_count', 0)} invoices are overdue "
                        f"({observed:.2f}%)."
                    ),
                    recommended_next_check=(
                        "Review student receivables by aging and department."
                    ),
                    rule_id="OVERDUE_PAYMENT_MAX",
                )
            )

    operating_result = _number(finance.get("net_operating_result"))
    total_revenue = _number(finance.get("total_revenue"))
    if operating_result is not None and operating_result <= 0:
        anomalies.append(
            _make_anomaly(
                generator,
                title="Negative or zero operating result",
                description="Operating expenses meet or exceed operating revenue.",
                metric="net_operating_result",
                observed_value=operating_result,
                threshold_value=0,
                severity=severity_for_negative_value(
                    operating_result,
                    total_revenue,
                ),
                period=period,
                source_file=source_file,
                evidence=(
                    f"Net operating result is ${operating_result:,.0f} on "
                    f"${(total_revenue or 0):,.0f} of revenue."
                ),
                recommended_next_check=(
                    "Review revenue shortfalls and expense drivers by department."
                ),
                rule_id="OPERATING_RESULT_MIN",
            )
        )

    cash = finance.get("cash_flow") or {}
    net_cash_flow = _number(cash.get("net_cash_flow"))
    cash_inflows = _number(cash.get("cash_inflows"))
    if (
        net_cash_flow is not None
        and net_cash_flow <= thresholds.low_cash_flow_threshold
    ):
        anomalies.append(
            _make_anomaly(
                generator,
                title="Negative or low cash flow",
                description="Net cash flow is at or below the configured minimum.",
                metric="net_cash_flow",
                observed_value=net_cash_flow,
                threshold_value=thresholds.low_cash_flow_threshold,
                severity=severity_for_negative_value(
                    net_cash_flow,
                    cash_inflows,
                ),
                period=period,
                source_file=source_file,
                evidence=(
                    f"Net cash flow is ${net_cash_flow:,.0f}; ending cash is "
                    f"${float(cash.get('ending_cash') or 0):,.0f}."
                ),
                recommended_next_check=(
                    "Review operating, scholarship, and capital cash outflows."
                ),
                rule_id="NET_CASH_FLOW_MIN",
            )
        )

    vendor = finance.get("vendor_payments") or {}
    maximum_payment = _number(vendor.get("maximum_payment_amount"))
    if (
        maximum_payment is not None
        and maximum_payment > thresholds.vendor_payment_review_threshold
    ):
        anomalies.append(
            _make_anomaly(
                generator,
                title="Vendor payment exceeds review threshold",
                description=(
                    "At least one vendor payment exceeds the configured review value."
                ),
                metric="maximum_vendor_payment",
                observed_value=maximum_payment,
                threshold_value=thresholds.vendor_payment_review_threshold,
                severity=severity_for_threshold_multiple(
                    maximum_payment,
                    thresholds.vendor_payment_review_threshold,
                ),
                period=period,
                source_file=source_file,
                evidence=(
                    f"Maximum payment is ${maximum_payment:,.0f} versus a "
                    f"${thresholds.vendor_payment_review_threshold:,.0f} threshold."
                ),
                recommended_next_check=(
                    "Inspect the underlying vendor invoice, approval, and duplicate checks."
                ),
                rule_id="VENDOR_PAYMENT_REVIEW",
            )
        )
    return anomalies


def _detect_department_rules(
    bundle: CalculationOutputBundle,
    thresholds: AnomalyThresholds,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Detect department overspending and target-range exceptions.

    Inputs: calculation bundle, thresholds, and ID generator.
    Outputs: at most one prioritized budget anomaly per department.
    Assumptions: expense_variance_pct uses (actual - budget) / budget.
    """

    anomalies: list[Anomaly] = []
    dataframe = bundle.department_summary
    source_file = Path(bundle.department_summary_path).name
    if "expense_variance_pct" not in dataframe.columns:
        return anomalies

    for _, row in dataframe.iterrows():
        variance_ratio = _number(row.get("expense_variance_pct"))
        if variance_ratio is None:
            continue
        variance_percent = variance_ratio * 100
        department = str(row.get("department") or "Unknown department")
        if variance_percent > thresholds.department_overspend_flag_percent:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title=f"{department} overspending exceeds flag threshold",
                    description=(
                        "Department actual expenses materially exceed budget."
                    ),
                    metric="department_expense_variance_pct",
                    observed_value=variance_percent,
                    threshold_value=thresholds.department_overspend_flag_percent,
                    severity=severity_for_upper_threshold(
                        variance_percent,
                        thresholds.department_overspend_flag_percent,
                    ),
                    period=bundle.report_period,
                    source_file=source_file,
                    evidence=(
                        f"{department} spent ${float(row.get('actual_expenses')):,.0f} "
                        f"against ${float(row.get('budget_expenses')):,.0f} budget "
                        f"({variance_percent:.2f}% variance)."
                    ),
                    recommended_next_check=(
                        "Inspect department expense categories, payroll, and vendors."
                    ),
                    rule_id="DEPARTMENT_OVERSPEND_FLAG",
                )
            )
        elif abs(variance_percent) > thresholds.department_budget_target_range_percent:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title=f"{department} outside budget target range",
                    description=(
                        "Department expense variance is outside the configured +/- range."
                    ),
                    metric="department_expense_variance_pct",
                    observed_value=variance_percent,
                    threshold_value=thresholds.department_budget_target_range_percent,
                    severity="medium",
                    period=bundle.report_period,
                    source_file=source_file,
                    evidence=(
                        f"{department} expense variance is {variance_percent:.2f}% "
                        f"versus a +/-{thresholds.department_budget_target_range_percent:.2f}% target."
                    ),
                    recommended_next_check=(
                        "Confirm whether the variance is timing-related or structural."
                    ),
                    rule_id="DEPARTMENT_BUDGET_RANGE",
                )
            )
    return anomalies


def _detect_category_rules(
    bundle: CalculationOutputBundle,
    thresholds: AnomalyThresholds,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Detect expense-category overspending when budget evidence is available.

    Inputs: calculation bundle, thresholds, and ID generator.
    Outputs: category budget anomalies.
    Assumptions: category variance percentage follows the Step 3 aggregate formula.
    """

    anomalies: list[Anomaly] = []
    dataframe = bundle.category_summary
    source_file = Path(bundle.category_summary_path).name
    if "variance_pct" not in dataframe.columns:
        return anomalies

    for _, row in dataframe.iterrows():
        variance_ratio = _number(row.get("variance_pct"))
        if variance_ratio is None:
            continue
        variance_percent = variance_ratio * 100
        if variance_percent <= thresholds.department_budget_target_range_percent:
            continue
        category = str(row.get("category") or "Unknown category")
        is_flag = variance_percent > thresholds.department_overspend_flag_percent
        anomalies.append(
            _make_anomaly(
                generator,
                title=(
                    f"{category} category overspending"
                    if is_flag
                    else f"{category} category outside budget target"
                ),
                description="Expense category actual value exceeds its budget range.",
                metric="category_expense_variance_pct",
                observed_value=variance_percent,
                threshold_value=(
                    thresholds.department_overspend_flag_percent
                    if is_flag
                    else thresholds.department_budget_target_range_percent
                ),
                severity=(
                    severity_for_upper_threshold(
                        variance_percent,
                        thresholds.department_overspend_flag_percent,
                    )
                    if is_flag
                    else "medium"
                ),
                period=bundle.report_period,
                source_file=source_file,
                evidence=(
                    f"{category} actual ${float(row.get('actual_amount')):,.0f} "
                    f"versus budget ${float(row.get('budget_amount')):,.0f}; "
                    f"variance {variance_percent:.2f}%."
                ),
                recommended_next_check=(
                    "Review the category by department and underlying transactions."
                ),
                rule_id=(
                    "CATEGORY_OVERSPEND_FLAG"
                    if is_flag
                    else "CATEGORY_BUDGET_RANGE"
                ),
            )
        )
    return anomalies


def _detect_availability_rules(
    bundle: CalculationOutputBundle,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Convert unavailable KPIs and calculation warnings into low-severity flags.

    Inputs: calculation bundle and ID generator.
    Outputs: data-quality anomaly records.
    Assumptions: availability problems are risks but not financial threshold breaches.
    """

    anomalies: list[Anomaly] = []
    source_file = Path(bundle.kpi_summary_path).name
    if "availability" in bundle.kpi_summary.columns:
        unavailable = bundle.kpi_summary.loc[
            bundle.kpi_summary["availability"].astype(str).str.lower()
            != "available"
        ]
        for _, row in unavailable.iterrows():
            metric = str(row.get("metric") or "unknown_metric")
            anomalies.append(
                _make_anomaly(
                    generator,
                    title=f"Unavailable metric: {metric}",
                    description="A required KPI could not be calculated.",
                    metric=metric,
                    observed_value=None,
                    threshold_value=None,
                    severity="low",
                    period=bundle.report_period,
                    source_file=source_file,
                    evidence="KPI availability is marked unavailable.",
                    recommended_next_check=(
                        "Review calculation warnings and source table availability."
                    ),
                    rule_id="METRIC_UNAVAILABLE",
                )
            )

    calculation_warnings = bundle.finance_document.get("calculation_warnings")
    if isinstance(calculation_warnings, list):
        for warning in calculation_warnings:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title="Calculation warning requires review",
                    description=str(warning),
                    metric="calculation_warning",
                    observed_value=str(warning),
                    threshold_value=None,
                    severity="low",
                    period=bundle.report_period,
                    source_file=Path(bundle.finance_summary_path).name,
                    evidence=str(warning),
                    recommended_next_check=(
                        "Resolve the missing or invalid calculation input."
                    ),
                    rule_id="CALCULATION_WARNING",
                )
            )
    return anomalies


def detect_rule_based_anomalies(
    bundle: CalculationOutputBundle,
    thresholds: AnomalyThresholds,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Run all deterministic rules for one calculation scope.

    Inputs: calculation bundle, configurable thresholds, and ID generator.
    Outputs: ordered headline, department, category, and availability anomalies.
    Assumptions: no trend or statistical logic runs in this function.
    """

    return [
        *_detect_summary_rules(bundle, thresholds, generator),
        *_detect_department_rules(bundle, thresholds, generator),
        *_detect_category_rules(bundle, thresholds, generator),
        *_detect_availability_rules(bundle, generator),
    ]
