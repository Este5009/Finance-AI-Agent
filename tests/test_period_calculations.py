"""Tests for monthly, annual, custom-period, and trend calculations."""

from datetime import date
from typing import Any

import pandas as pd

from finance_agent.calculation_loader import (
    LoadedIntermediateModel,
    LoadedIntermediateTable,
)
from finance_agent.finance_engine import run_finance_calculations
from finance_agent.periods import PeriodScope, filter_table_for_period


def _table(
    detected_type: str,
    data: dict[str, list[Any]],
    *,
    source: str = "annual.xlsx",
) -> LoadedIntermediateTable:
    """Build a period-aware normalized table fixture.

    Inputs: detected type, column data, and source workbook.
    Outputs: LoadedIntermediateTable.
    Assumptions: fixture values already satisfy the intermediate-model contract.
    """

    return LoadedIntermediateTable(
        table_id=f"{source}__{detected_type}",
        detected_type=detected_type,
        source_workbook=source,
        sheet=detected_type,
        confidence=0.99,
        csv_path="fixture.csv",
        dataframe=pd.DataFrame(data),
        metadata={},
    )


def _period_model() -> LoadedIntermediateModel:
    """Create a compact annual model containing three monthly finance tables.

    Inputs: none.
    Outputs: loaded model with January, February, and March records.
    Assumptions: missing nonessential table types may produce expected warnings.
    """

    periods = ["2026-01-01", "2026-02-01", "2026-03-01"]
    tables = (
        _table(
            "Revenue",
            {
                "period": periods,
                "department": ["Engineering"] * 3,
                "budget_revenue": [110, 210, 310],
                "actual_revenue": [100, 200, 300],
            },
        ),
        _table(
            "Expenses",
            {
                "period": periods,
                "department": ["Engineering"] * 3,
                "expense_category": ["Payroll"] * 3,
                "budget_expense": [45, 85, 125],
                "actual_expense": [40, 80, 120],
            },
        ),
        _table(
            "Payroll",
            {
                "period": periods,
                "total_payroll": [20, 40, 60],
            },
        ),
    )
    return LoadedIntermediateModel(
        model_path="financial_document_model.json",
        model_version="2.0",
        source_workbooks=["annual.xlsx"],
        tables=tables,
        manifest={},
    )


def test_monthly_calculation_filters_to_selected_month() -> None:
    """Verify a monthly scope calculates only rows inside that month."""

    result = run_finance_calculations(
        _period_model(),
        source_workbook="annual.xlsx",
        report_period="February 2026",
        period_scope=PeriodScope.monthly(2026, 2),
    )

    assert result.finance_summary["total_revenue"] == 200
    assert result.finance_summary["total_expenses"] == 80
    assert result.finance_summary["net_operating_result"] == 120
    assert result.finance_summary["payroll_total"] == 40


def test_annual_calculation_uses_full_year_scope() -> None:
    """Verify an annual scope aggregates all rows in the selected year."""

    result = run_finance_calculations(
        _period_model(),
        source_workbook="annual.xlsx",
        report_period="2026",
        period_scope=PeriodScope.annual(2026),
    )

    assert result.finance_summary["total_revenue"] == 600
    assert result.finance_summary["total_expenses"] == 240
    assert result.finance_summary["net_operating_result"] == 360
    assert result.finance_summary["payroll_total"] == 120


def test_custom_period_filtering_uses_inclusive_dates() -> None:
    """Verify arbitrary inclusive date ranges filter normalized rows."""

    revenue_table = _period_model().tables[0]
    scope = PeriodScope.custom(
        date(2026, 2, 1),
        date(2026, 3, 1),
        label="February through March",
    )

    filtered = filter_table_for_period(revenue_table, scope)

    assert filtered is not None
    assert filtered.dataframe["actual_revenue"].tolist() == [200, 300]


def test_monthly_trend_generation_returns_ordered_year() -> None:
    """Verify annual calculations produce a 12-row monthly trend table."""

    result = run_finance_calculations(
        _period_model(),
        source_workbook="annual.xlsx",
        report_period="2026",
        period_scope=PeriodScope.annual(2026),
        monthly_trend_year=2026,
    )

    trends = result.monthly_trends
    assert len(trends.index) == 12
    assert trends["period"].tolist()[:3] == ["2026-01", "2026-02", "2026-03"]
    assert trends.loc[0, "actual_revenue"] == 100
    assert trends.loc[1, "actual_expenses"] == 80
    assert trends.loc[2, "net_operating_result"] == 180
    assert trends["actual_revenue"].sum() == 600
