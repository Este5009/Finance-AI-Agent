"""Tests for deterministic Step 3 financial calculations."""

from typing import Any

import pandas as pd
import pytest

from finance_agent.calculations.calculation_loader import (
    LoadedIntermediateModel,
    LoadedIntermediateTable,
)
from finance_agent.calculations.finance_calculations import (
    calculate_budget_vs_actual,
    calculate_net_operating_result,
    calculate_payroll_percentage,
    calculate_total_expenses,
    calculate_total_revenue,
)
from finance_agent.calculations.finance_engine import run_finance_calculations
from finance_agent.calculations.financial_health_ratios import (
    calculate_financial_health_ratios,
    classify_financial_health_ratio,
)


def _table(
    detected_type: str,
    data: dict[str, list[Any]],
    *,
    table_id: str | None = None,
) -> LoadedIntermediateTable:
    """Build a normalized loaded-table fixture.

    Inputs: detected type, column data, and optional table identifier.
    Outputs: LoadedIntermediateTable for isolated calculation tests.
    Assumptions: fixture data has already passed Step 2 normalization.
    """

    return LoadedIntermediateTable(
        table_id=table_id or f"test__{detected_type.lower()}",
        detected_type=detected_type,
        source_workbook="monthly.xlsx",
        sheet=detected_type,
        confidence=0.99,
        csv_path="fixture.csv",
        dataframe=pd.DataFrame(data),
        metadata={},
    )


def test_total_revenue_calculation() -> None:
    """Verify actual revenue is summed across all selected rows."""

    tables = [
        _table("Revenue", {"actual_revenue": [1000, 2500]}),
        _table("Revenue", {"actual_revenue": [500]}, table_id="revenue_2"),
    ]

    assert calculate_total_revenue(tables) == 4000


def test_total_expenses_calculation() -> None:
    """Verify actual expenses are summed deterministically."""

    tables = [_table("Expenses", {"actual_expense": [600, 900]})]

    assert calculate_total_expenses(tables) == 1500


def test_net_operating_result_calculation() -> None:
    """Verify operating result is revenue minus expenses."""

    assert calculate_net_operating_result(4000, 1500) == 2500


def test_budget_variance_calculation() -> None:
    """Verify aggregate budget dollar and percentage variances."""

    tables = [
        _table(
            "Budget_vs_Actual",
            {
                "budget_revenue": [1000, 2000],
                "actual_revenue": [900, 2100],
                "budget_expense": [500, 700],
                "actual_expense": [550, 770],
            },
        )
    ]

    result = calculate_budget_vs_actual(tables)

    assert result is not None
    assert result["revenue_variance"] == 0
    assert result["revenue_variance_pct"] == 0
    assert result["expense_variance"] == 120
    assert result["expense_variance_pct"] == pytest.approx(0.10)
    assert result["net_variance"] == -120


def test_payroll_percentage_calculation() -> None:
    """Verify payroll percentage uses total revenue as denominator."""

    assert calculate_payroll_percentage(420, 1000) == pytest.approx(0.42)


def test_missing_table_handling_returns_warnings() -> None:
    """Verify a missing table yields unavailable metrics instead of crashing."""

    warnings: list[str] = []

    total = calculate_total_revenue([], warnings)

    assert total is None
    assert warnings
    assert "Revenue" in warnings[0]


def test_engine_marks_missing_metrics_unavailable() -> None:
    """Verify orchestration completes when only a Revenue table exists."""

    revenue = _table(
        "Revenue",
        {
            "department": ["Engineering"],
            "actual_revenue": [1000],
        },
    )
    model = LoadedIntermediateModel(
        model_path="financial_document_model.json",
        model_version="2.0",
        source_workbooks=["monthly.xlsx"],
        tables=(revenue,),
        manifest={},
    )

    result = run_finance_calculations(
        model,
        source_workbook="monthly.xlsx",
        report_period="June 2026",
    )

    assert result.finance_summary["total_revenue"] == 1000
    assert result.finance_summary["total_expenses"] is None
    assert result.calculation_warnings
    expense_kpi = result.kpi_summary.loc[
        result.kpi_summary["metric"] == "total_expenses"
    ].iloc[0]
    assert expense_kpi["availability"] == "unavailable"


def test_financial_health_ratios_calculate_from_wide_position_table() -> None:
    """Verify management ratios are deterministic and provenance-backed."""

    position = _table(
        "Financial_Position",
        {
            "current_assets": [1_800_000],
            "current_liabilities": [1_200_000],
            "cash_and_equivalents": [720_000],
            "total_assets": [10_000_000],
            "total_liabilities": [4_800_000],
            "ebitda": [660_000],
            "net_income": [1_000_000],
        },
    )

    ratios = {
        row["metric"]: row
        for row in calculate_financial_health_ratios([position], {"total_revenue": 2_000_000})
    }

    assert ratios["current_ratio"]["value"] == pytest.approx(1.5)
    assert ratios["cash_ratio"]["classification"] == "Bueno"
    assert ratios["total_debt_ratio"]["classification"] == "Bueno"
    assert ratios["ebitda_margin"]["value"] == pytest.approx(0.33)
    assert ratios["net_margin"]["classification"] == "Bueno"
    assert ratios["return_on_assets"]["classification"] == "Aceptable"
    assert ratios["current_ratio"]["reference_origin"] == "configurable_internal_management_threshold"
    assert ratios["current_ratio"]["is_regulatory_limit"] is False


def test_financial_health_ratios_remain_unavailable_when_inputs_missing() -> None:
    """Verify missing source values are not fabricated for ratios."""

    ratios = {
        row["metric"]: row
        for row in calculate_financial_health_ratios([], {"total_revenue": 2_000_000})
    }

    assert ratios["current_ratio"]["availability"] == "unavailable"
    assert set(ratios["current_ratio"]["missing_inputs"]) == {"current_assets", "current_liabilities"}
    assert ratios["ebitda_margin"]["missing_inputs"] == ["ebitda"]


def test_financial_health_ratio_threshold_bands() -> None:
    """Verify the canonical management-threshold bands."""

    assert classify_financial_health_ratio("current_ratio", 0.99) == "Crítico"
    assert classify_financial_health_ratio("current_ratio", 1.20) == "Aceptable"
    assert classify_financial_health_ratio("current_ratio", 1.50) == "Bueno"
    assert classify_financial_health_ratio("total_debt_ratio", 0.61) == "Crítico"
