"""Orchestration and output serialization for deterministic finance calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from finance_agent.calculation_loader import LoadedIntermediateModel
from finance_agent.finance_calculations import (
    calculate_budget_expenses_by_category,
    calculate_budget_expenses_by_department,
    calculate_budget_revenue_by_department,
    calculate_budget_vs_actual,
    calculate_cash_flow_summary,
    calculate_monthly_trends,
    calculate_expenses_by_category,
    calculate_expenses_by_department,
    calculate_net_operating_result,
    calculate_payroll_percentage,
    calculate_payroll_total,
    calculate_revenue_by_department,
    calculate_scholarship_totals,
    calculate_student_payment_metrics,
    calculate_total_expenses,
    calculate_total_revenue,
    calculate_vendor_payment_totals,
)
from finance_agent.periods import PeriodScope, filter_selected_tables_for_period
from finance_agent.table_selection import select_financial_tables


@dataclass
class FinanceCalculationResult:
    """Structured Step 3 result for downstream deterministic and AI modules."""

    report_period: str
    period_scope: PeriodScope | None
    source_workbook: str
    intermediate_model_path: str
    finance_summary: dict[str, Any]
    kpi_summary: pd.DataFrame
    department_summary: pd.DataFrame
    category_summary: pd.DataFrame
    monthly_trends: pd.DataFrame
    calculation_warnings: list[str]


def _merge_department_summaries(
    budget_revenue_by_department: pd.DataFrame,
    revenue_by_department: pd.DataFrame,
    budget_expenses_by_department: pd.DataFrame,
    expenses_by_department: pd.DataFrame,
) -> pd.DataFrame:
    """Combine revenue and expense aggregates into department operating results.

    Inputs: department budget/actual revenue and expense DataFrames.
    Outputs: department summary with budgets, variances, and net result.
    Assumptions: absent values remain unavailable rather than being treated as zero.
    """

    columns = [
        "department",
        "budget_revenue",
        "actual_revenue",
        "revenue_variance",
        "revenue_variance_pct",
        "budget_expenses",
        "actual_expenses",
        "expense_variance",
        "expense_variance_pct",
        "net_operating_result",
    ]
    if revenue_by_department.empty and expenses_by_department.empty:
        return pd.DataFrame(columns=columns)

    frames = [
        budget_revenue_by_department,
        revenue_by_department,
        budget_expenses_by_department,
        expenses_by_department,
    ]
    nonempty_frames = [frame for frame in frames if not frame.empty]
    merged = nonempty_frames[0]
    for frame in nonempty_frames[1:]:
        merged = merged.merge(frame, on="department", how="outer")
    for column in [
        "budget_revenue",
        "actual_revenue",
        "budget_expenses",
        "actual_expenses",
    ]:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged["revenue_variance"] = (
        merged["actual_revenue"] - merged["budget_revenue"]
    )
    merged["revenue_variance_pct"] = (
        merged["revenue_variance"]
        / merged["budget_revenue"].where(merged["budget_revenue"] != 0)
    )
    merged["expense_variance"] = (
        merged["actual_expenses"] - merged["budget_expenses"]
    )
    merged["expense_variance_pct"] = (
        merged["expense_variance"]
        / merged["budget_expenses"].where(merged["budget_expenses"] != 0)
    )
    merged["net_operating_result"] = (
        merged["actual_revenue"] - merged["actual_expenses"]
    )
    return merged[columns].sort_values("department").reset_index(drop=True)


def _build_category_summary(
    budget_expenses_by_category: pd.DataFrame,
    expenses_by_category: pd.DataFrame,
) -> pd.DataFrame:
    """Convert expense-category totals to the stable category output schema.

    Inputs: budget and actual expense category aggregations.
    Outputs: category budget, actual, variance, and percentage DataFrame.
    Assumptions: Step 3 currently requires expense categories, not a unified taxonomy.
    """

    if expenses_by_category.empty and budget_expenses_by_category.empty:
        return pd.DataFrame(
            columns=[
                "category_type",
                "category",
                "budget_amount",
                "actual_amount",
                "variance",
                "variance_pct",
            ]
        )
    budget = budget_expenses_by_category.rename(
        columns={
            "expense_category": "category",
            "budget_expenses": "budget_amount",
        }
    )
    actual = expenses_by_category.rename(
        columns={
            "expense_category": "category",
            "actual_expenses": "actual_amount",
        }
    )
    summary = budget.merge(actual, on="category", how="outer")
    summary["variance"] = summary["actual_amount"] - summary["budget_amount"]
    summary["variance_pct"] = (
        summary["variance"]
        / summary["budget_amount"].where(summary["budget_amount"] != 0)
    )
    summary.insert(0, "category_type", "expense")
    return summary[
        [
            "category_type",
            "category",
            "budget_amount",
            "actual_amount",
            "variance",
            "variance_pct",
        ]
    ]


def _kpi_row(
    metric: str,
    value: float | int | None,
    unit: str,
    source: str,
) -> dict[str, Any]:
    """Build one stable KPI output record.

    Inputs: metric name, value, unit, and source description.
    Outputs: dictionary with availability status.
    Assumptions: None means unavailable; zero remains a valid calculated result.
    """

    return {
        "metric": metric,
        "value": value,
        "unit": unit,
        "availability": "available" if value is not None else "unavailable",
        "source": source,
    }


def _build_kpi_summary(
    finance_summary: dict[str, Any],
) -> pd.DataFrame:
    """Create the tabular KPI summary from calculated finance values.

    Inputs: complete finance summary dictionary.
    Outputs: stable KPI DataFrame for CSV serialization.
    Assumptions: nested calculation sections may be unavailable.
    """

    budget = finance_summary.get("budget_vs_actual") or {}
    student = finance_summary.get("student_payments") or {}
    vendor = finance_summary.get("vendor_payments") or {}
    scholarships = finance_summary.get("scholarships") or {}
    cash_flow = finance_summary.get("cash_flow") or {}
    rows = [
        _kpi_row(
            "total_revenue",
            finance_summary.get("total_revenue"),
            "USD",
            "Revenue",
        ),
        _kpi_row(
            "total_expenses",
            finance_summary.get("total_expenses"),
            "USD",
            "Expenses",
        ),
        _kpi_row(
            "net_operating_result",
            finance_summary.get("net_operating_result"),
            "USD",
            "Revenue - Expenses",
        ),
        _kpi_row(
            "revenue_budget_variance",
            budget.get("revenue_variance"),
            "USD",
            "Budget_vs_Actual",
        ),
        _kpi_row(
            "revenue_budget_variance_pct",
            budget.get("revenue_variance_pct"),
            "ratio",
            "Budget_vs_Actual",
        ),
        _kpi_row(
            "expense_budget_variance",
            budget.get("expense_variance"),
            "USD",
            "Budget_vs_Actual",
        ),
        _kpi_row(
            "expense_budget_variance_pct",
            budget.get("expense_variance_pct"),
            "ratio",
            "Budget_vs_Actual",
        ),
        _kpi_row(
            "payroll_total",
            finance_summary.get("payroll_total"),
            "USD",
            "Payroll",
        ),
        _kpi_row(
            "payroll_percentage_of_revenue",
            finance_summary.get("payroll_percentage_of_revenue"),
            "ratio",
            "Payroll / Revenue",
        ),
        _kpi_row(
            "student_payment_collection_rate",
            student.get("collection_rate"),
            "ratio",
            "Student_Payments",
        ),
        _kpi_row(
            "overdue_payment_percentage",
            student.get("overdue_payment_percentage"),
            "ratio",
            "Student_Payments",
        ),
        _kpi_row(
            "vendor_payment_total",
            vendor.get("total_amount"),
            "USD",
            "Vendor_Payments",
        ),
        _kpi_row(
            "scholarship_awarded_total",
            scholarships.get("awarded"),
            "USD",
            "Scholarships",
        ),
        _kpi_row(
            "net_cash_flow",
            cash_flow.get("net_cash_flow"),
            "USD",
            "Cash_Flow",
        ),
        _kpi_row(
            "ending_cash",
            cash_flow.get("ending_cash"),
            "USD",
            "Cash_Flow",
        ),
    ]
    return pd.DataFrame(rows)


def run_finance_calculations(
    model: LoadedIntermediateModel,
    *,
    source_workbook: str,
    report_period: str,
    period_scope: PeriodScope | None = None,
    monthly_trend_year: int | None = None,
) -> FinanceCalculationResult:
    """Run all Step 3 calculations for one manifest reporting scope.

    Inputs: loaded model, source workbook, report label, optional period scope,
        and optional year for monthly trend generation.
    Outputs: structured summaries, KPI table, grouped tables, and warnings.
    Assumptions: source_workbook prevents monthly and annual scopes from mixing.
    """

    warnings: list[str] = []
    selected = select_financial_tables(
        model,
        source_workbook=source_workbook,
    )
    if period_scope is not None:
        selected = filter_selected_tables_for_period(
            selected,
            period_scope,
            warnings,
        )

    total_revenue = calculate_total_revenue(selected["Revenue"], warnings)
    total_expenses = calculate_total_expenses(selected["Expenses"], warnings)
    net_operating_result = calculate_net_operating_result(
        total_revenue,
        total_expenses,
        warnings,
    )
    revenue_by_department = calculate_revenue_by_department(
        selected["Revenue"],
        warnings,
    )
    budget_revenue_by_department = calculate_budget_revenue_by_department(
        selected["Revenue"],
        warnings,
    )
    expenses_by_department = calculate_expenses_by_department(
        selected["Expenses"],
        warnings,
    )
    budget_expenses_by_department = calculate_budget_expenses_by_department(
        selected["Expenses"],
        warnings,
    )
    expenses_by_category = calculate_expenses_by_category(
        selected["Expenses"],
        warnings,
    )
    budget_expenses_by_category = calculate_budget_expenses_by_category(
        selected["Expenses"],
        warnings,
    )
    budget_vs_actual = calculate_budget_vs_actual(
        selected["Budget_vs_Actual"],
        warnings,
    )
    payroll_total = calculate_payroll_total(selected["Payroll"], warnings)
    payroll_percentage = calculate_payroll_percentage(
        payroll_total,
        total_revenue,
        warnings,
    )
    student_payments = calculate_student_payment_metrics(
        selected["Student_Payments"],
        warnings,
    )
    vendor_payments = calculate_vendor_payment_totals(
        selected["Vendor_Payments"],
        warnings,
    )
    scholarships = calculate_scholarship_totals(
        selected["Scholarships"],
        warnings,
    )
    cash_flow = calculate_cash_flow_summary(
        selected["Cash_Flow"],
        warnings,
    )
    monthly_trends = (
        calculate_monthly_trends(
            selected,
            year=monthly_trend_year,
            warnings=warnings,
        )
        if monthly_trend_year is not None
        else pd.DataFrame()
    )

    finance_summary = {
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "net_operating_result": net_operating_result,
        "budget_vs_actual": budget_vs_actual,
        "payroll_total": payroll_total,
        "payroll_percentage_of_revenue": payroll_percentage,
        "student_payments": student_payments,
        "vendor_payments": vendor_payments,
        "scholarships": scholarships,
        "cash_flow": cash_flow,
    }
    return FinanceCalculationResult(
        report_period=report_period,
        period_scope=period_scope,
        source_workbook=source_workbook,
        intermediate_model_path=model.model_path,
        finance_summary=finance_summary,
        kpi_summary=_build_kpi_summary(finance_summary),
        department_summary=_merge_department_summaries(
            budget_revenue_by_department,
            revenue_by_department,
            budget_expenses_by_department,
            expenses_by_department,
        ),
        category_summary=_build_category_summary(
            budget_expenses_by_category,
            expenses_by_category,
        ),
        monthly_trends=monthly_trends,
        calculation_warnings=warnings,
    )


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a calculation DataFrame to strict JSON-compatible records.

    Inputs: calculation DataFrame.
    Outputs: records with missing values converted to JSON null.
    Assumptions: date fields are not currently part of summary outputs.
    """

    return json.loads(dataframe.to_json(orient="records"))


def save_finance_calculation_outputs(
    result: FinanceCalculationResult,
    output_directory: str | Path,
    *,
    period_slug: str,
) -> dict[str, Path]:
    """Save Step 3 JSON and CSV outputs using stable filenames.

    Inputs: calculation result, output directory, and filename-safe period slug.
    Outputs: dictionary of generated artifact paths.
    Assumptions: callers supply a slug such as june_2026.
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "finance_summary": output_dir / f"finance_summary_{period_slug}.json",
        "kpi_summary": output_dir / f"kpi_summary_{period_slug}.csv",
        "department_summary": output_dir / f"department_summary_{period_slug}.csv",
        "category_summary": output_dir / f"category_summary_{period_slug}.csv",
    }
    if not result.monthly_trends.empty:
        paths["monthly_trends"] = (
            output_dir / f"monthly_trends_{period_slug}.csv"
        )

    result.kpi_summary.to_csv(paths["kpi_summary"], index=False, encoding="utf-8")
    result.department_summary.to_csv(
        paths["department_summary"],
        index=False,
        encoding="utf-8",
    )
    result.category_summary.to_csv(
        paths["category_summary"],
        index=False,
        encoding="utf-8",
    )
    if "monthly_trends" in paths:
        result.monthly_trends.to_csv(
            paths["monthly_trends"],
            index=False,
            encoding="utf-8",
        )
    period_scope_payload = (
        {
            "mode": result.period_scope.mode,
            "label": result.period_scope.label,
            "start_date": result.period_scope.start_date.isoformat(),
            "end_date": result.period_scope.end_date.isoformat(),
        }
        if result.period_scope is not None
        else None
    )
    json_payload = {
        "report_period": result.report_period,
        "period_scope": period_scope_payload,
        "source_workbook": result.source_workbook,
        "intermediate_model": result.intermediate_model_path,
        "finance_summary": result.finance_summary,
        "kpi_summary": _dataframe_records(result.kpi_summary),
        "department_summary": _dataframe_records(result.department_summary),
        "category_summary": _dataframe_records(result.category_summary),
        "monthly_trends": _dataframe_records(result.monthly_trends),
        "calculation_warnings": result.calculation_warnings,
    }
    paths["finance_summary"].write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return paths
