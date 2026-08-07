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
    finding_type: str = "system_review_rule",
    reference_type: str = "threshold",
    reference_origin: str = "system-derived/default",
    reference_source: str = "AnomalyThresholds configuration",
    is_institutional_reference: bool = False,
    reason_for_flagging: str | None = None,
    supporting_evidence: str | None = None,
    recommended_action: str | None = None,
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
        finding_type=finding_type,
        reference_type=reference_type,
        reference_origin=reference_origin,
        reference_source=reference_source,
        is_institutional_reference=is_institutional_reference,
        reason_for_flagging=reason_for_flagging or description,
        supporting_evidence=supporting_evidence or evidence,
        recommended_action=recommended_action or recommended_next_check,
    )


def _system_reason(observed: object, reference: object, relation: str) -> str:
    """Explain a system review rule without implying institutional policy.

    Inputs: observed value, reference value, and comparison relation.
    Outputs: concise reason for flagging.
    Assumptions: configured anomaly thresholds are analytical references unless
    an institutional source is explicitly recorded elsewhere.
    """

    return (
        f"El valor observado ({observed}) {relation} la referencia analítica "
        f"configurada por el sistema ({reference})."
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
                        "Payroll cost is above the system analytical reference for revenue share."
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
                        f"versus the system analytical reference of "
                        f"{thresholds.payroll_percent_max:.2f}%."
                    ),
                    recommended_next_check=(
                        "Review payroll by department, overtime, benefits, and headcount."
                    ),
                    rule_id="PAYROLL_RATIO_MAX",
                    reason_for_flagging=_system_reason(
                        f"{observed:.2f}%",
                        f"{thresholds.payroll_percent_max:.2f}%",
                        "supera",
                    ),
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
                        "Student payment collections are below the system analytical reference."
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
                    reason_for_flagging=_system_reason(
                        f"{observed:.2f}%",
                        f"{thresholds.tuition_collection_min_percent:.2f}%",
                        "está por debajo de",
                    ),
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
                        "The share of overdue student invoices exceeds the system analytical reference."
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
                    reason_for_flagging=_system_reason(
                        f"{observed:.2f}%",
                        f"{thresholds.overdue_payment_max_percent:.2f}%",
                        "supera",
                    ),
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
                reason_for_flagging=_system_reason(
                    f"${operating_result:,.0f}",
                    "$0",
                    "está en o por debajo de",
                ),
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
                description="Net cash flow is at or below the system analytical reference.",
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
                reason_for_flagging=_system_reason(
                    f"${net_cash_flow:,.0f}",
                    f"${thresholds.low_cash_flow_threshold:,.0f}",
                    "está en o por debajo de",
                ),
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
                    "At least one vendor payment exceeds the system analytical review reference."
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
                    f"${thresholds.vendor_payment_review_threshold:,.0f} system review reference."
                ),
                recommended_next_check=(
                    "Inspect the underlying vendor invoice, approval, and duplicate checks."
                ),
                rule_id="VENDOR_PAYMENT_REVIEW",
                reason_for_flagging=_system_reason(
                    f"${maximum_payment:,.0f}",
                    f"${thresholds.vendor_payment_review_threshold:,.0f}",
                    "supera",
                ),
            )
        )

    duplicate_candidates = vendor.get("duplicate_candidates")
    if isinstance(duplicate_candidates, list):
        for candidate in duplicate_candidates:
            if not isinstance(candidate, dict):
                continue
            vendor_name = candidate.get("vendor") or candidate.get("vendor_name") or "unknown vendor"
            invoice = candidate.get("invoice") or candidate.get("invoice_id") or candidate.get("invoice_number")
            amount = _number(candidate.get("amount"))
            payment_date = candidate.get("date") or candidate.get("payment_date")
            evidence_parts = [
                f"vendor={vendor_name}",
                f"invoice={invoice}" if invoice else "",
                f"amount=${amount:,.0f}" if amount is not None else "",
                f"date={payment_date}" if payment_date else "",
            ]
            evidence = "; ".join(part for part in evidence_parts if part)
            anomalies.append(
                _make_anomaly(
                    generator,
                    title="Potential duplicate vendor payment",
                    description=(
                        "Transaction evidence contains matching vendor/invoice/payment attributes that require verification."
                    ),
                    metric="vendor_payment_duplicate_candidate",
                    observed_value=amount,
                    threshold_value=None,
                    severity="high",
                    period=period,
                    source_file=source_file,
                    evidence=evidence or "Duplicate candidate was present in processed vendor evidence.",
                    recommended_next_check=(
                        "Verify the vendor, invoice, amount, approval, and payment date before drawing conclusions."
                    ),
                    rule_id="VENDOR_POTENTIAL_DUPLICATE",
                    finding_type="potential_duplicate",
                    reference_type="transaction_match",
                    reference_origin="none",
                    reference_source="processed vendor transaction evidence",
                    is_institutional_reference=False,
                    reason_for_flagging=(
                        "Processed transaction evidence shows matching vendor/invoice/amount/date attributes."
                    ),
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
                        "Department actual expenses exceed budget and cross the system review reference."
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
                    reference_type="budget_variance_review_threshold",
                    reference_source=(
                        "approved department budget plus AnomalyThresholds.department_overspend_flag_percent"
                    ),
                    reason_for_flagging=(
                        f"{department} variance of {variance_percent:.2f}% exceeds the "
                        f"{thresholds.department_overspend_flag_percent:.2f}% system review reference."
                    ),
                )
            )
        elif abs(variance_percent) > thresholds.department_budget_target_range_percent:
            anomalies.append(
                _make_anomaly(
                    generator,
                    title=f"{department} outside budget target range",
                    description=(
                        "Department expense variance is outside the system analytical +/- range."
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
                    reference_type="budget_variance_review_range",
                    reference_source=(
                        "approved department budget plus AnomalyThresholds.department_budget_target_range_percent"
                    ),
                    reason_for_flagging=(
                        f"{department} variance of {variance_percent:.2f}% is outside the "
                        f"+/-{thresholds.department_budget_target_range_percent:.2f}% system review range."
                    ),
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
                description="Expense category actual value exceeds its budget and crosses a system review reference.",
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
                reference_type="budget_variance_review_threshold",
                reference_source=(
                    "approved category budget plus AnomalyThresholds budget variance review settings"
                ),
                reason_for_flagging=(
                    f"{category} variance of {variance_percent:.2f}% exceeds the "
                    "configured system review reference."
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
                    finding_type="data_quality_finding",
                    reference_type="data_availability",
                    reference_origin="none",
                    reference_source="processed KPI availability",
                    reason_for_flagging="A processed KPI was marked unavailable by deterministic calculations.",
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
                    finding_type="data_quality_finding",
                    reference_type="calculation_warning",
                    reference_origin="none",
                    reference_source="finance calculation warning",
                    reason_for_flagging="A deterministic calculation warning was emitted.",
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
