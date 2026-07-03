"""Simple z-score anomaly detection for annual monthly trends."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finance_agent.anomaly_config import AnomalyThresholds
from finance_agent.anomaly_loader import CalculationOutputBundle
from finance_agent.anomaly_models import Anomaly, AnomalyIdGenerator
from finance_agent.anomaly_severity import severity_for_z_score


STATISTICAL_METRICS = {
    "actual_revenue": "Monthly revenue",
    "actual_expenses": "Monthly expenses",
    "payroll_percentage_of_revenue": "Monthly payroll/revenue",
    "student_collection_rate": "Monthly collection rate",
    "net_operating_result": "Monthly operating result",
}


def calculate_z_scores(values: pd.Series) -> pd.Series:
    """Calculate population z-scores while safely handling zero variance.

    Inputs: numeric-like monthly values.
    Outputs: aligned z-score Series; all zeros when standard deviation is zero.
    Assumptions: population standard deviation is appropriate for the full year.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean(skipna=True)
    standard_deviation = numeric.std(skipna=True, ddof=0)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - mean) / standard_deviation


def detect_statistical_anomalies(
    bundle: CalculationOutputBundle,
    thresholds: AnomalyThresholds,
    generator: AnomalyIdGenerator,
) -> list[Anomaly]:
    """Flag monthly values whose absolute z-score reaches the threshold.

    Inputs: annual calculation bundle, thresholds, and ID generator.
    Outputs: statistical anomaly records.
    Assumptions: at least three numeric observations are required per metric.
    """

    trends = bundle.monthly_trends
    if trends.empty:
        return []

    anomalies: list[Anomaly] = []
    source_file = Path(bundle.monthly_trends_path or "").name
    periods = trends["period"].astype(str)
    for column, label in STATISTICAL_METRICS.items():
        if column not in trends.columns:
            continue
        numeric = pd.to_numeric(trends[column], errors="coerce")
        if numeric.notna().sum() < 3:
            continue
        mean = float(numeric.mean(skipna=True))
        standard_deviation = float(numeric.std(skipna=True, ddof=0))
        if standard_deviation == 0:
            # A constant metric contains no statistical outlier by definition.
            continue
        z_scores = calculate_z_scores(numeric)
        flagged = z_scores.abs() >= thresholds.statistical_z_score_threshold
        for index in z_scores.index[flagged]:
            z_score = float(z_scores.loc[index])
            raw_value = float(numeric.loc[index])
            anomalies.append(
                Anomaly(
                    anomaly_id=generator.next_id(),
                    title=f"Statistical outlier: {label}",
                    description=(
                        "Monthly value is unusually far from its annual mean."
                    ),
                    metric=f"{column}_z_score",
                    observed_value=z_score,
                    threshold_value=thresholds.statistical_z_score_threshold,
                    severity=severity_for_z_score(z_score),
                    period=periods.loc[index],
                    source_file=source_file,
                    evidence=(
                        f"Raw value {raw_value:.6g}; mean {mean:.6g}; "
                        f"standard deviation {standard_deviation:.6g}; "
                        f"z-score {z_score:.3f}."
                    ),
                    recommended_next_check=(
                        f"Validate the {label.lower()} source and investigate its drivers."
                    ),
                    detection_method="statistical",
                    rule_id=f"Z_SCORE_{column.upper()}",
                )
            )
    return anomalies
