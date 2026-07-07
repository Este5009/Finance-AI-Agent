"""Deterministic severity assignment for financial anomalies."""

from __future__ import annotations


def severity_for_upper_threshold(
    observed_percent: float,
    threshold_percent: float,
) -> str:
    """Prioritize a percentage that exceeds an upper limit.

    Inputs: observed and threshold values in percentage points.
    Outputs: medium, high, or critical.
    Assumptions: up to two points is slight; over twenty points is extreme.
    """

    excess = observed_percent - threshold_percent
    if excess > 20:
        return "critical"
    if excess > 2:
        return "high"
    return "medium"


def severity_for_lower_threshold(
    observed_percent: float,
    threshold_percent: float,
) -> str:
    """Prioritize a percentage that falls below a minimum.

    Inputs: observed and threshold values in percentage points.
    Outputs: medium, high, or critical.
    Assumptions: a shortfall over ten points is critical.
    """

    shortfall = threshold_percent - observed_percent
    if shortfall > 10:
        return "critical"
    if shortfall > 2:
        return "high"
    return "medium"


def severity_for_threshold_multiple(
    observed_value: float,
    threshold_value: float,
) -> str:
    """Prioritize a positive value by its multiple of a review threshold.

    Inputs: observed and positive threshold values.
    Outputs: medium, high, or critical.
    Assumptions: over twice the threshold is critical.
    """

    if threshold_value <= 0:
        return "medium"
    multiple = observed_value / threshold_value
    if multiple > 2:
        return "critical"
    if multiple > 1.25:
        return "high"
    return "medium"


def severity_for_negative_value(
    observed_value: float,
    scale_value: float | None,
) -> str:
    """Prioritize a negative operating or cash result relative to scale.

    Inputs: negative observed value and positive revenue/inflow scale when available.
    Outputs: medium, high, or critical.
    Assumptions: losses above 10% of scale are critical.
    """

    if observed_value >= 0:
        return "medium"
    if scale_value is None or scale_value <= 0:
        return "high"
    magnitude_ratio = abs(observed_value) / scale_value
    if magnitude_ratio >= 0.10:
        return "critical"
    if magnitude_ratio >= 0.03:
        return "high"
    return "medium"


def severity_for_z_score(z_score: float) -> str:
    """Prioritize a statistical outlier by absolute z-score.

    Inputs: signed z-score.
    Outputs: medium, high, or critical.
    Assumptions: three sigma is high and four sigma is critical.
    """

    absolute_score = abs(z_score)
    if absolute_score >= 4:
        return "critical"
    if absolute_score >= 3:
        return "high"
    return "medium"
