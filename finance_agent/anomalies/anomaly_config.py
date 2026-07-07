"""Configurable deterministic thresholds for Step 4 anomaly detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AnomalyThresholds:
    """Financial rule and statistical thresholds expressed in percentage points."""

    payroll_percent_max: float = 42.0
    department_overspend_flag_percent: float = 12.0
    department_budget_target_range_percent: float = 8.0
    tuition_collection_min_percent: float = 94.0
    overdue_payment_max_percent: float = 6.0
    vendor_payment_review_threshold: float = 50_000.0
    month_over_month_expense_increase_percent: float = 10.0
    month_over_month_revenue_drop_percent: float = 10.0
    low_cash_flow_threshold: float = 0.0
    statistical_z_score_threshold: float = 2.0

    def to_dict(self) -> dict[str, float]:
        """Serialize thresholds for anomaly report auditability.

        Inputs: this immutable configuration.
        Outputs: field/value dictionary.
        Assumptions: percentages remain human-readable percentage points.
        """

        return asdict(self)
