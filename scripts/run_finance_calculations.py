"""Run deterministic Step 3 calculations for the June 2026 report."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution begins in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.calculation_loader import load_intermediate_model  # noqa: E402
from finance_agent.finance_engine import (  # noqa: E402
    FinanceCalculationResult,
    run_finance_calculations,
    save_finance_calculation_outputs,
)
from finance_agent.periods import PeriodScope  # noqa: E402


INTERMEDIATE_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "intermediate"
    / "financial_document_model.json"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "calculations"

@dataclass(frozen=True)
class CalculationRun:
    """Configuration for one deterministic reporting-scope calculation."""

    source_workbook: str
    period_scope: PeriodScope
    period_slug: str
    monthly_trend_year: int | None = None


CALCULATION_RUNS = (
    CalculationRun(
        source_workbook="monthly_financial_report_june_2026.xlsx",
        period_scope=PeriodScope.monthly(2026, 6),
        period_slug="june_2026",
    ),
    CalculationRun(
        source_workbook="annual_financial_report_2026.xlsx",
        period_scope=PeriodScope.annual(2026),
        period_slug="2026",
        monthly_trend_year=2026,
    ),
)


def _display_value(value: object, *, percentage: bool = False) -> str:
    """Format one console summary value without changing stored precision.

    Inputs: calculation value and whether it is a decimal percentage.
    Outputs: readable string or 'Unavailable'.
    Assumptions: currency values are displayed in USD for this synthetic report.
    """

    if value is None:
        return "Unavailable"
    if percentage:
        return f"{float(value):.1%}"
    return f"${float(value):,.0f}"


def _print_result_summary(result: FinanceCalculationResult) -> None:
    """Print one monthly or annual calculation summary.

    Inputs: completed finance calculation result.
    Outputs: concise console metrics and warnings.
    Assumptions: ratios are stored as decimals and displayed as percentages.
    """

    finance = result.finance_summary
    student = finance.get("student_payments") or {}
    print(f"\nReporting period: {result.report_period}")
    print(f"Source scope: {result.source_workbook}")
    print(f"Total revenue: {_display_value(finance.get('total_revenue'))}")
    print(f"Total expenses: {_display_value(finance.get('total_expenses'))}")
    print(
        "Net operating result: "
        f"{_display_value(finance.get('net_operating_result'))}"
    )
    print(
        "Payroll as % of revenue: "
        f"{_display_value(finance.get('payroll_percentage_of_revenue'), percentage=True)}"
    )
    print(
        "Student collection rate: "
        f"{_display_value(student.get('collection_rate'), percentage=True)}"
    )
    print(f"Calculation warnings: {len(result.calculation_warnings)}")
    for warning in result.calculation_warnings:
        print(f"  - {warning}")


def main() -> None:
    """Load the model and calculate both June and annual 2026 outputs.

    Inputs: fixed Step 2 model and monthly/annual provenance configurations.
    Outputs: monthly and annual summaries plus the annual monthly-trend CSV.
    Assumptions: no raw workbook access is allowed in this script or engine.
    """

    print("Finance AI Agent - Step 3 Deterministic Finance Calculations")
    model = load_intermediate_model(INTERMEDIATE_MODEL)
    for run in CALCULATION_RUNS:
        result = run_finance_calculations(
            model,
            source_workbook=run.source_workbook,
            report_period=run.period_scope.label,
            period_scope=run.period_scope,
            monthly_trend_year=run.monthly_trend_year,
        )
        paths = save_finance_calculation_outputs(
            result,
            OUTPUT_DIRECTORY,
            period_slug=run.period_slug,
        )
        _print_result_summary(result)
        print("Calculation outputs saved:")
        for output_path in paths.values():
            print(f"  - {output_path.resolve()}")


if __name__ == "__main__":
    main()
