"""Tests for deterministic rule, trend, statistical, and severity detection."""

from pathlib import Path
from typing import Any

import pandas as pd

from finance_agent.anomalies.anomaly_config import AnomalyThresholds
from finance_agent.anomalies.anomaly_loader import CalculationOutputBundle
from finance_agent.anomalies.anomaly_models import AnomalyIdGenerator
from finance_agent.anomalies.anomaly_rules import detect_rule_based_anomalies
from finance_agent.anomalies.anomaly_severity import (
    severity_for_negative_value,
    severity_for_upper_threshold,
)
from finance_agent.anomalies.anomaly_statistics import (
    calculate_z_scores,
    detect_statistical_anomalies,
)
from finance_agent.anomalies.anomaly_trends import detect_trend_anomalies


def _bundle(
    *,
    finance_overrides: dict[str, Any] | None = None,
    department_summary: pd.DataFrame | None = None,
    category_summary: pd.DataFrame | None = None,
    monthly_trends: pd.DataFrame | None = None,
) -> CalculationOutputBundle:
    """Build a complete calculation-output fixture with safe baseline metrics.

    Inputs: optional finance, department, category, and trend overrides.
    Outputs: CalculationOutputBundle for isolated anomaly tests.
    Assumptions: baseline values do not breach default thresholds.
    """

    finance = {
        "total_revenue": 1000.0,
        "total_expenses": 800.0,
        "net_operating_result": 200.0,
        "payroll_percentage_of_revenue": 0.40,
        "student_payments": {
            "amount_due": 1000.0,
            "amount_paid": 1000.0,
            "collection_rate": 1.0,
            "invoice_count": 100,
            "overdue_invoice_count": 0,
            "overdue_payment_percentage": 0.0,
        },
        "cash_flow": {
            "net_cash_flow": 100.0,
            "cash_inflows": 1000.0,
            "ending_cash": 500.0,
        },
        "vendor_payments": {
            "maximum_payment_amount": 10_000.0,
        },
    }
    if finance_overrides:
        finance.update(finance_overrides)

    return CalculationOutputBundle(
        period_slug="test",
        finance_summary_path=str(Path("finance_summary_test.json")),
        kpi_summary_path=str(Path("kpi_summary_test.csv")),
        department_summary_path=str(Path("department_summary_test.csv")),
        category_summary_path=str(Path("category_summary_test.csv")),
        monthly_trends_path=(
            str(Path("monthly_trends_test.csv"))
            if monthly_trends is not None
            else None
        ),
        finance_document={
            "report_period": "Test Period",
            "finance_summary": finance,
            "calculation_warnings": [],
        },
        kpi_summary=pd.DataFrame(
            {
                "metric": ["total_revenue"],
                "availability": ["available"],
            }
        ),
        department_summary=(
            department_summary
            if department_summary is not None
            else pd.DataFrame()
        ),
        category_summary=(
            category_summary
            if category_summary is not None
            else pd.DataFrame()
        ),
        monthly_trends=(
            monthly_trends
            if monthly_trends is not None
            else pd.DataFrame()
        ),
    )


def _rule_ids(bundle: CalculationOutputBundle) -> set[str]:
    """Run default rule detection and return emitted rule identifiers.

    Inputs: calculation bundle.
    Outputs: set of rule IDs.
    Assumptions: tests care about detection presence rather than generated IDs.
    """

    anomalies = detect_rule_based_anomalies(
        bundle,
        AnomalyThresholds(),
        AnomalyIdGenerator("TEST"),
    )
    return {anomaly.rule_id for anomaly in anomalies}


def test_payroll_threshold_detection() -> None:
    """Verify payroll/revenue above 42% is flagged."""

    bundle = _bundle(
        finance_overrides={"payroll_percentage_of_revenue": 0.50}
    )

    assert "PAYROLL_RATIO_MAX" in _rule_ids(bundle)


def test_system_threshold_provenance_is_not_institutional_policy() -> None:
    """Verify default detector thresholds are labeled as system references."""

    anomalies = detect_rule_based_anomalies(
        _bundle(finance_overrides={"payroll_percentage_of_revenue": 0.50}),
        AnomalyThresholds(),
        AnomalyIdGenerator("TEST"),
    )
    anomaly = next(item for item in anomalies if item.rule_id == "PAYROLL_RATIO_MAX")
    payload = anomaly.to_dict()

    assert payload["finding_type"] == "system_review_rule"
    assert payload["reference_origin"] == "system-derived/default"
    assert payload["is_institutional_reference"] is False
    assert "No corresponde a una meta, límite o política institucional" in payload["reference_notice_es"]
    assert "system analytical reference" in payload["description"]


def test_collection_rate_detection() -> None:
    """Verify student collection below 94% is flagged."""

    student = {
        "amount_due": 1000.0,
        "amount_paid": 900.0,
        "collection_rate": 0.90,
        "invoice_count": 100,
        "overdue_invoice_count": 0,
        "overdue_payment_percentage": 0.0,
    }

    assert "TUITION_COLLECTION_MIN" in _rule_ids(
        _bundle(finance_overrides={"student_payments": student})
    )


def test_overdue_payment_detection() -> None:
    """Verify overdue invoice percentage above 6% is flagged."""

    student = {
        "amount_due": 1000.0,
        "amount_paid": 1000.0,
        "collection_rate": 1.0,
        "invoice_count": 100,
        "overdue_invoice_count": 10,
        "overdue_payment_percentage": 0.10,
    }

    assert "OVERDUE_PAYMENT_MAX" in _rule_ids(
        _bundle(finance_overrides={"student_payments": student})
    )


def test_negative_operating_result_detection() -> None:
    """Verify a negative operating result is flagged."""

    bundle = _bundle(
        finance_overrides={"net_operating_result": -200.0}
    )

    assert "OPERATING_RESULT_MIN" in _rule_ids(bundle)


def test_department_overspending_detection() -> None:
    """Verify department spending above 12% budget is flagged."""

    departments = pd.DataFrame(
        {
            "department": ["Engineering"],
            "budget_expenses": [1000.0],
            "actual_expenses": [1180.0],
            "expense_variance_pct": [0.18],
        }
    )

    assert "DEPARTMENT_OVERSPEND_FLAG" in _rule_ids(
        _bundle(department_summary=departments)
    )


def test_month_over_month_trend_detection() -> None:
    """Verify an expense increase above 10% creates a trend anomaly."""

    trends = pd.DataFrame(
        {
            "period": ["2026-01", "2026-02"],
            "actual_expenses": [100.0, 120.0],
        }
    )
    anomalies = detect_trend_anomalies(
        _bundle(monthly_trends=trends),
        AnomalyThresholds(),
        AnomalyIdGenerator("TEST"),
    )

    assert any(
        anomaly.rule_id == "MOM_EXPENSE_INCREASE"
        for anomaly in anomalies
    )


def test_z_score_anomaly_detection() -> None:
    """Verify a two-sigma monthly value is statistically flagged."""

    trends = pd.DataFrame(
        {
            "period": [f"2026-{month:02d}" for month in range(1, 6)],
            "actual_revenue": [1.0, 1.0, 1.0, 1.0, 10.0],
        }
    )
    anomalies = detect_statistical_anomalies(
        _bundle(monthly_trends=trends),
        AnomalyThresholds(),
        AnomalyIdGenerator("TEST"),
    )

    assert len(anomalies) == 1
    assert anomalies[0].period == "2026-05"
    payload = anomalies[0].to_dict()
    assert payload["finding_type"] == "statistical_anomaly"
    assert payload["reference_origin"] == "historical/statistical"
    assert payload["is_institutional_reference"] is False
    assert "Referencia histórica/estadística" in payload["reference_notice_es"]


def test_potential_duplicate_uses_transaction_evidence_not_threshold() -> None:
    """Verify duplicate findings rely on transaction matches, not value thresholds."""

    vendor = {
        "maximum_payment_amount": 1000.0,
        "duplicate_candidates": [
            {
                "vendor": "MedSupply",
                "invoice": "INV-101",
                "amount": 1000.0,
                "date": "2026-07-15",
            }
        ],
    }
    anomalies = detect_rule_based_anomalies(
        _bundle(finance_overrides={"vendor_payments": vendor}),
        AnomalyThresholds(vendor_payment_review_threshold=50_000.0),
        AnomalyIdGenerator("TEST"),
    )
    duplicate = next(item for item in anomalies if item.rule_id == "VENDOR_POTENTIAL_DUPLICATE")
    payload = duplicate.to_dict()

    assert payload["finding_type"] == "potential_duplicate"
    assert payload["threshold_value"] is None
    assert payload["reference_type"] == "transaction_match"
    assert payload["reference_origin"] == "none"
    assert "MedSupply" in payload["supporting_evidence"]
    assert "fraud" not in payload["description"].lower()


def test_large_vendor_payment_is_review_rule_not_duplicate() -> None:
    """Verify high-value vendor review threshold is not labeled duplicate evidence."""

    anomalies = detect_rule_based_anomalies(
        _bundle(finance_overrides={"vendor_payments": {"maximum_payment_amount": 75_000.0}}),
        AnomalyThresholds(vendor_payment_review_threshold=50_000.0),
        AnomalyIdGenerator("TEST"),
    )
    vendor = next(item for item in anomalies if item.rule_id == "VENDOR_PAYMENT_REVIEW")
    payload = vendor.to_dict()

    assert payload["finding_type"] == "system_review_rule"
    assert payload["reference_origin"] == "system-derived/default"
    assert payload["threshold_value"] == 50_000.0
    assert payload["rule_id"] != "VENDOR_POTENTIAL_DUPLICATE"


def test_data_quality_finding_has_no_system_threshold_notice() -> None:
    """Verify unavailable KPIs are data-quality findings without threshold language."""

    bundle = _bundle()
    bundle = CalculationOutputBundle(
        **{
            **bundle.__dict__,
            "kpi_summary": pd.DataFrame({"metric": ["collection_rate"], "availability": ["unavailable"]}),
        }
    )
    anomalies = detect_rule_based_anomalies(bundle, AnomalyThresholds(), AnomalyIdGenerator("TEST"))
    finding = next(item for item in anomalies if item.rule_id == "METRIC_UNAVAILABLE")
    payload = finding.to_dict()

    assert payload["finding_type"] == "data_quality_finding"
    assert payload["reference_origin"] == "none"
    assert payload["reference_notice_es"] == ""


def test_zero_standard_deviation_handling() -> None:
    """Verify constant values return zero z-scores and no anomaly."""

    values = pd.Series([5.0, 5.0, 5.0, 5.0])
    trends = pd.DataFrame(
        {
            "period": ["2026-01", "2026-02", "2026-03", "2026-04"],
            "actual_revenue": values,
        }
    )

    assert calculate_z_scores(values).eq(0).all()
    assert detect_statistical_anomalies(
        _bundle(monthly_trends=trends),
        AnomalyThresholds(),
        AnomalyIdGenerator("TEST"),
    ) == []


def test_severity_assignment_is_deterministic() -> None:
    """Verify slight, material, and severe breaches receive stable priorities."""

    assert severity_for_upper_threshold(43.0, 42.0) == "medium"
    assert severity_for_upper_threshold(50.0, 42.0) == "high"
    assert severity_for_negative_value(-150.0, 1000.0) == "critical"
