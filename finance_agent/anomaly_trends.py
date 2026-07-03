"""Trend-based anomaly detection over annual monthly calculation outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finance_agent.anomaly_config import AnomalyThresholds
from finance_agent.anomaly_loader import CalculationOutputBundle
from finance_agent.anomaly_models import Anomaly, AnomalyIdGenerator
from finance_agent.anomaly_severity import (
    severity_for_lower_threshold,
    severity_for_negative_value,
    severity_for_upper_threshold,
)


def _trend_anomaly(
    generator: AnomalyIdGenerator,
    *,
    title: str,
    description: str,
    metric: str,
    observed_value: float,
    threshold_value: float,
    severity: str,
    period: str,
    source_file: str,
    evidence: str,
    recommended_next_check: str,
    rule_id: str,
) -> Anomaly:
    """Build one trend-based anomaly record.

    Inputs: generator and complete trend evidence.
    Outputs: immutable anomaly marked trend_based.
    Assumptions: observed and threshold values use the units named by metric.
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
        detection_method="trend_based",
        rule_id=rule_id,
    )


def _numeric_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric trend series with invalid values as NaN.

    Inputs: monthly trends DataFrame and column name.
    Outputs: numeric Series aligned to the input rows.
    Assumptions: the caller has confirmed the column exists.
    """

    return pd.to_numeric(dataframe[column], errors="coerce")


def detect_trend_anomalies(
    bundle: CalculationOutputBundle,
    thresholds: AnomalyThresholds,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Detect month-over-month and monthly threshold exceptions.

    Inputs: annual calculation bundle, thresholds, and ID generator.
    Outputs: ordered trend anomaly records.
    Assumptions: monthly_trends rows are chronological and periods are unique.
    """

    trends = bundle.monthly_trends.copy()
    if trends.empty:
        return []

    source_file = Path(bundle.monthly_trends_path or "").name
    anomalies: list[Anomaly] = []
    periods = trends["period"].astype(str)

    if "actual_revenue" in trends.columns:
        revenue = _numeric_column(trends, "actual_revenue")
        revenue_change = revenue.pct_change(fill_method=None) * 100
        for index in revenue_change.index[revenue_change < -thresholds.month_over_month_revenue_drop_percent]:
            observed_drop = abs(float(revenue_change.loc[index]))
            anomalies.append(
                _trend_anomaly(
                    generator,
                    title="Month-over-month revenue drop",
                    description="Monthly revenue declined beyond the configured limit.",
                    metric="month_over_month_revenue_drop_percent",
                    observed_value=observed_drop,
                    threshold_value=thresholds.month_over_month_revenue_drop_percent,
                    severity=severity_for_upper_threshold(
                        observed_drop,
                        thresholds.month_over_month_revenue_drop_percent,
                    ),
                    period=periods.loc[index],
                    source_file=source_file,
                    evidence=(
                        f"Revenue changed from ${revenue.loc[index - 1]:,.0f} to "
                        f"${revenue.loc[index]:,.0f}, a {observed_drop:.2f}% drop."
                    ),
                    recommended_next_check=(
                        "Review revenue categories, enrollment, and collection timing."
                    ),
                    rule_id="MOM_REVENUE_DROP",
                )
            )

    if "actual_expenses" in trends.columns:
        expenses = _numeric_column(trends, "actual_expenses")
        expense_change = expenses.pct_change(fill_method=None) * 100
        for index in expense_change.index[
            expense_change > thresholds.month_over_month_expense_increase_percent
        ]:
            observed_increase = float(expense_change.loc[index])
            anomalies.append(
                _trend_anomaly(
                    generator,
                    title="Month-over-month expense increase",
                    description="Monthly expenses increased beyond the configured limit.",
                    metric="month_over_month_expense_increase_percent",
                    observed_value=observed_increase,
                    threshold_value=thresholds.month_over_month_expense_increase_percent,
                    severity=severity_for_upper_threshold(
                        observed_increase,
                        thresholds.month_over_month_expense_increase_percent,
                    ),
                    period=periods.loc[index],
                    source_file=source_file,
                    evidence=(
                        f"Expenses changed from ${expenses.loc[index - 1]:,.0f} to "
                        f"${expenses.loc[index]:,.0f}, a {observed_increase:.2f}% increase."
                    ),
                    recommended_next_check=(
                        "Review monthly expense categories, departments, and vendors."
                    ),
                    rule_id="MOM_EXPENSE_INCREASE",
                )
            )

    if "net_operating_result" in trends.columns:
        operating = _numeric_column(trends, "net_operating_result")
        revenue = (
            _numeric_column(trends, "actual_revenue")
            if "actual_revenue" in trends.columns
            else pd.Series(index=trends.index, dtype=float)
        )
        for index in operating.index[operating < 0]:
            value = float(operating.loc[index])
            scale = (
                float(revenue.loc[index])
                if index in revenue.index and pd.notna(revenue.loc[index])
                else None
            )
            anomalies.append(
                _trend_anomaly(
                    generator,
                    title="Monthly operating deficit",
                    description="The month produced a negative operating result.",
                    metric="monthly_net_operating_result",
                    observed_value=value,
                    threshold_value=0,
                    severity=severity_for_negative_value(value, scale),
                    period=periods.loc[index],
                    source_file=source_file,
                    evidence=f"Monthly operating result is ${value:,.0f}.",
                    recommended_next_check=(
                        "Compare monthly revenue and expense drivers with budget."
                    ),
                    rule_id="MONTHLY_OPERATING_DEFICIT",
                )
            )

    if "payroll_percentage_of_revenue" in trends.columns:
        payroll_ratio = (
            _numeric_column(trends, "payroll_percentage_of_revenue") * 100
        )
        for index in payroll_ratio.index[
            payroll_ratio > thresholds.payroll_percent_max
        ]:
            observed = float(payroll_ratio.loc[index])
            anomalies.append(
                _trend_anomaly(
                    generator,
                    title="Monthly payroll ratio above threshold",
                    description="Payroll consumed too large a share of monthly revenue.",
                    metric="monthly_payroll_percentage_of_revenue",
                    observed_value=observed,
                    threshold_value=thresholds.payroll_percent_max,
                    severity=severity_for_upper_threshold(
                        observed,
                        thresholds.payroll_percent_max,
                    ),
                    period=periods.loc[index],
                    source_file=source_file,
                    evidence=(
                        f"Monthly payroll/revenue is {observed:.2f}% versus "
                        f"{thresholds.payroll_percent_max:.2f}% maximum."
                    ),
                    recommended_next_check=(
                        "Review payroll components and the month's revenue denominator."
                    ),
                    rule_id="MONTHLY_PAYROLL_RATIO_MAX",
                )
            )

    if "student_collection_rate" in trends.columns:
        collection_rate = _numeric_column(trends, "student_collection_rate") * 100
        for index in collection_rate.index[
            collection_rate < thresholds.tuition_collection_min_percent
        ]:
            observed = float(collection_rate.loc[index])
            anomalies.append(
                _trend_anomaly(
                    generator,
                    title="Monthly collection rate below target",
                    description="Monthly student collections are below target.",
                    metric="monthly_student_collection_rate",
                    observed_value=observed,
                    threshold_value=thresholds.tuition_collection_min_percent,
                    severity=severity_for_lower_threshold(
                        observed,
                        thresholds.tuition_collection_min_percent,
                    ),
                    period=periods.loc[index],
                    source_file=source_file,
                    evidence=(
                        f"Monthly collection rate is {observed:.2f}% versus "
                        f"{thresholds.tuition_collection_min_percent:.2f}% minimum."
                    ),
                    recommended_next_check=(
                        "Inspect that month's overdue invoices and collection actions."
                    ),
                    rule_id="MONTHLY_COLLECTION_MIN",
                )
            )
    return anomalies
