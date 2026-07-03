"""Deterministic finance calculations over normalized intermediate tables."""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from typing import Any

import pandas as pd

from finance_agent.calculation_loader import LoadedIntermediateTable
from finance_agent.periods import DATE_COLUMN_CANDIDATES, month_number_from_value
from finance_agent.table_selection import append_missing_table_warning


def _add_warning(warnings: list[str] | None, message: str) -> None:
    """Append one warning without duplicates.

    Inputs: optional warning list and warning message.
    Outputs: the supplied list is updated when present.
    Assumptions: calculations may be used independently without collecting warnings.
    """

    if warnings is not None and message not in warnings:
        warnings.append(message)


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    """Find the first supported normalized column in a DataFrame.

    Inputs: DataFrame and ordered candidate names.
    Outputs: matching column name or None.
    Assumptions: candidate order expresses deterministic preference.
    """

    return next((column for column in candidates if column in dataframe.columns), None)


def _numeric_values(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Convert a normalized metric column to numeric values.

    Inputs: DataFrame and existing column name.
    Outputs: numeric Series with invalid values represented as NaN.
    Assumptions: invalid source values must never be silently treated as text math.
    """

    return pd.to_numeric(dataframe[column], errors="coerce")


def _sum_metric(
    tables: Sequence[LoadedIntermediateTable],
    column_candidates: Sequence[str],
    *,
    table_type: str,
    metric_name: str,
    warnings: list[str] | None = None,
) -> float | None:
    """Sum one metric across one or more normalized logical tables.

    Inputs: tables, ordered columns, labels, and optional warning collector.
    Outputs: deterministic numeric total or None when unavailable.
    Assumptions: multiple same-type tables represent additive logical partitions.
    """

    table_list = list(tables)
    if not table_list:
        if warnings is not None:
            append_missing_table_warning(table_list, table_type, warnings)
        return None

    total = 0.0
    usable_tables = 0
    for table in table_list:
        column = _first_existing_column(table.dataframe, column_candidates)
        if column is None:
            _add_warning(
                warnings,
                f"Metric unavailable in table '{table.table_id}': "
                f"none of columns {list(column_candidates)} exist for {metric_name}.",
            )
            continue
        numeric = _numeric_values(table.dataframe, column)
        if numeric.notna().sum() == 0 and len(table.dataframe.index) > 0:
            _add_warning(
                warnings,
                f"Metric unavailable in table '{table.table_id}': "
                f"column '{column}' contains no numeric values.",
            )
            continue
        total += float(numeric.sum(skipna=True))
        usable_tables += 1
    return total if usable_tables else None


def calculate_total_revenue(
    revenue_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> float | None:
    """Calculate total actual revenue.

    Inputs: selected Revenue tables and optional warning collector.
    Outputs: sum of actual revenue or None.
    Assumptions: actual_revenue is preferred over generic revenue/amount columns.
    """

    return _sum_metric(
        revenue_tables,
        ("actual_revenue", "revenue", "actual_amount", "amount"),
        table_type="Revenue",
        metric_name="total revenue",
        warnings=warnings,
    )


def calculate_total_expenses(
    expense_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> float | None:
    """Calculate total actual operating expenses.

    Inputs: selected Expenses tables and optional warning collector.
    Outputs: sum of actual expenses or None.
    Assumptions: scholarships and capital cash outflows are not operating expenses.
    """

    return _sum_metric(
        expense_tables,
        ("actual_expense", "expenses", "actual_amount", "amount"),
        table_type="Expenses",
        metric_name="total expenses",
        warnings=warnings,
    )


def calculate_net_operating_result(
    total_revenue: float | None,
    total_expenses: float | None,
    warnings: list[str] | None = None,
) -> float | None:
    """Calculate operating revenue less operating expenses.

    Inputs: total revenue, total expenses, and optional warning collector.
    Outputs: net operating result or None when either input is unavailable.
    Assumptions: both totals use the same selected reporting scope.
    """

    if total_revenue is None or total_expenses is None:
        _add_warning(
            warnings,
            "Net operating result unavailable because revenue or expenses are missing.",
        )
        return None
    return float(total_revenue - total_expenses)


def _group_metric(
    tables: Sequence[LoadedIntermediateTable],
    *,
    group_candidates: Sequence[str],
    value_candidates: Sequence[str],
    output_group_name: str,
    output_value_name: str,
    table_type: str,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate a normalized metric by one dimension.

    Inputs: tables, column candidates, output labels, type, and warnings.
    Outputs: grouped DataFrame, possibly empty but with stable columns.
    Assumptions: groups from multiple logical tables are additive.
    """

    table_list = list(tables)
    empty_result = pd.DataFrame(columns=[output_group_name, output_value_name])
    if not table_list:
        if warnings is not None:
            append_missing_table_warning(table_list, table_type, warnings)
        return empty_result

    parts: list[pd.DataFrame] = []
    for table in table_list:
        group_column = _first_existing_column(table.dataframe, group_candidates)
        value_column = _first_existing_column(table.dataframe, value_candidates)
        if group_column is None or value_column is None:
            _add_warning(
                warnings,
                f"Grouped metric unavailable in table '{table.table_id}': "
                "required dimension or value column is missing.",
            )
            continue
        part = pd.DataFrame(
            {
                output_group_name: table.dataframe[group_column],
                output_value_name: _numeric_values(table.dataframe, value_column),
            }
        ).dropna(subset=[output_group_name])
        parts.append(part)

    if not parts:
        return empty_result
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(output_group_name, as_index=False, dropna=False)[
            output_value_name
        ]
        .sum()
        .sort_values(output_group_name)
        .reset_index(drop=True)
    )


def calculate_revenue_by_department(
    revenue_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate actual revenue by department.

    Inputs: selected Revenue tables and optional warnings.
    Outputs: department and actual_revenue DataFrame.
    Assumptions: department is the canonical organizational dimension.
    """

    return _group_metric(
        revenue_tables,
        group_candidates=("department",),
        value_candidates=("actual_revenue", "revenue", "actual_amount", "amount"),
        output_group_name="department",
        output_value_name="actual_revenue",
        table_type="Revenue",
        warnings=warnings,
    )


def calculate_budget_revenue_by_department(
    revenue_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate budget revenue by department.

    Inputs: selected Revenue tables and optional warnings.
    Outputs: department and budget_revenue DataFrame.
    Assumptions: budget_revenue is additive across detailed revenue rows.
    """

    return _group_metric(
        revenue_tables,
        group_candidates=("department",),
        value_candidates=("budget_revenue", "budget_amount", "budget"),
        output_group_name="department",
        output_value_name="budget_revenue",
        table_type="Revenue",
        warnings=warnings,
    )


def calculate_expenses_by_department(
    expense_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate actual operating expenses by department.

    Inputs: selected Expenses tables and optional warnings.
    Outputs: department and actual_expenses DataFrame.
    Assumptions: every selected table uses compatible currency units.
    """

    return _group_metric(
        expense_tables,
        group_candidates=("department",),
        value_candidates=("actual_expense", "expenses", "actual_amount", "amount"),
        output_group_name="department",
        output_value_name="actual_expenses",
        table_type="Expenses",
        warnings=warnings,
    )


def calculate_budget_expenses_by_department(
    expense_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate budget operating expenses by department.

    Inputs: selected Expenses tables and optional warnings.
    Outputs: department and budget_expenses DataFrame.
    Assumptions: budget expense rows use compatible currency units.
    """

    return _group_metric(
        expense_tables,
        group_candidates=("department",),
        value_candidates=("budget_expense", "budget_amount", "budget"),
        output_group_name="department",
        output_value_name="budget_expenses",
        table_type="Expenses",
        warnings=warnings,
    )


def calculate_expenses_by_category(
    expense_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate actual operating expenses by expense category.

    Inputs: selected Expenses tables and optional warnings.
    Outputs: expense_category and actual_expenses DataFrame.
    Assumptions: category labels are already normalized by Step 2.
    """

    return _group_metric(
        expense_tables,
        group_candidates=("expense_category", "category"),
        value_candidates=("actual_expense", "expenses", "actual_amount", "amount"),
        output_group_name="expense_category",
        output_value_name="actual_expenses",
        table_type="Expenses",
        warnings=warnings,
    )


def calculate_budget_expenses_by_category(
    expense_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate budget operating expenses by expense category.

    Inputs: selected Expenses tables and optional warnings.
    Outputs: expense_category and budget_expenses DataFrame.
    Assumptions: category labels were normalized by Step 2.
    """

    return _group_metric(
        expense_tables,
        group_candidates=("expense_category", "category"),
        value_candidates=("budget_expense", "budget_amount", "budget"),
        output_group_name="expense_category",
        output_value_name="budget_expenses",
        table_type="Expenses",
        warnings=warnings,
    )


def calculate_budget_variance_percentage(
    actual: float | None,
    budget: float | None,
    *,
    metric_name: str = "budget variance",
    warnings: list[str] | None = None,
) -> float | None:
    """Calculate (actual - budget) divided by budget.

    Inputs: actual, budget, optional metric label, and warning collector.
    Outputs: decimal variance percentage or None for missing/zero budget.
    Assumptions: positive percentages mean actual is above budget.
    """

    if actual is None or budget is None:
        _add_warning(warnings, f"{metric_name} percentage unavailable: missing value.")
        return None
    if budget == 0:
        _add_warning(warnings, f"{metric_name} percentage unavailable: budget is zero.")
        return None
    return float((actual - budget) / budget)


def calculate_budget_vs_actual(
    budget_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> dict[str, float | None] | None:
    """Calculate aggregate revenue, expense, and net budget variances.

    Inputs: selected Budget_vs_Actual tables and optional warnings.
    Outputs: totals, dollar variances, and variance percentages, or None.
    Assumptions: budget variance percentages use aggregate totals, not averages.
    """

    table_list = list(budget_tables)
    if not table_list:
        if warnings is not None:
            append_missing_table_warning(table_list, "Budget_vs_Actual", warnings)
        return None

    revenue_budget = _sum_metric(
        table_list,
        ("budget_revenue",),
        table_type="Budget_vs_Actual",
        metric_name="budget revenue",
        warnings=warnings,
    )
    revenue_actual = _sum_metric(
        table_list,
        ("actual_revenue",),
        table_type="Budget_vs_Actual",
        metric_name="actual revenue",
        warnings=warnings,
    )
    expense_budget = _sum_metric(
        table_list,
        ("budget_expense",),
        table_type="Budget_vs_Actual",
        metric_name="budget expense",
        warnings=warnings,
    )
    expense_actual = _sum_metric(
        table_list,
        ("actual_expense",),
        table_type="Budget_vs_Actual",
        metric_name="actual expense",
        warnings=warnings,
    )

    revenue_variance = (
        revenue_actual - revenue_budget
        if revenue_actual is not None and revenue_budget is not None
        else None
    )
    expense_variance = (
        expense_actual - expense_budget
        if expense_actual is not None and expense_budget is not None
        else None
    )
    net_budget = (
        revenue_budget - expense_budget
        if revenue_budget is not None and expense_budget is not None
        else None
    )
    net_actual = (
        revenue_actual - expense_actual
        if revenue_actual is not None and expense_actual is not None
        else None
    )
    return {
        "revenue_budget": revenue_budget,
        "revenue_actual": revenue_actual,
        "revenue_variance": revenue_variance,
        "revenue_variance_pct": calculate_budget_variance_percentage(
            revenue_actual,
            revenue_budget,
            metric_name="Revenue budget variance",
            warnings=warnings,
        ),
        "expense_budget": expense_budget,
        "expense_actual": expense_actual,
        "expense_variance": expense_variance,
        "expense_variance_pct": calculate_budget_variance_percentage(
            expense_actual,
            expense_budget,
            metric_name="Expense budget variance",
            warnings=warnings,
        ),
        "net_budget": net_budget,
        "net_actual": net_actual,
        "net_variance": (
            net_actual - net_budget
            if net_actual is not None and net_budget is not None
            else None
        ),
    }


def calculate_payroll_total(
    payroll_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> float | None:
    """Calculate total payroll cost.

    Inputs: selected Payroll tables and optional warnings.
    Outputs: payroll total or None.
    Assumptions: total_payroll already includes salary, benefits, and overtime.
    """

    return _sum_metric(
        payroll_tables,
        ("total_payroll", "payroll", "actual_amount", "amount"),
        table_type="Payroll",
        metric_name="payroll total",
        warnings=warnings,
    )


def calculate_payroll_percentage(
    payroll_total: float | None,
    total_revenue: float | None,
    warnings: list[str] | None = None,
) -> float | None:
    """Calculate payroll as a percentage of total revenue.

    Inputs: payroll total, revenue total, and optional warnings.
    Outputs: decimal payroll/revenue ratio or None.
    Assumptions: both values use the same reporting period and currency.
    """

    if payroll_total is None or total_revenue is None:
        _add_warning(
            warnings,
            "Payroll percentage unavailable because payroll or revenue is missing.",
        )
        return None
    if total_revenue == 0:
        _add_warning(warnings, "Payroll percentage unavailable because revenue is zero.")
        return None
    return float(payroll_total / total_revenue)


def calculate_student_payment_metrics(
    student_payment_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> dict[str, float | int | None] | None:
    """Calculate billed, collected, outstanding, collection, and overdue metrics.

    Inputs: selected Student_Payments tables and optional warnings.
    Outputs: payment totals and decimal collection/overdue rates, or None.
    Assumptions: overdue percentage is invoice count, not outstanding-dollar share.
    """

    table_list = list(student_payment_tables)
    if not table_list:
        if warnings is not None:
            append_missing_table_warning(table_list, "Student_Payments", warnings)
        return None

    billed_total = 0.0
    paid_total = 0.0
    outstanding_total = 0.0
    invoice_count = 0
    overdue_count = 0
    usable_tables = 0

    for table in table_list:
        dataframe = table.dataframe
        if "amount_due" not in dataframe.columns or "amount_paid" not in dataframe.columns:
            _add_warning(
                warnings,
                f"Student payment metrics unavailable in table '{table.table_id}': "
                "amount_due or amount_paid is missing.",
            )
            continue
        due = _numeric_values(dataframe, "amount_due")
        paid = _numeric_values(dataframe, "amount_paid")
        outstanding = (
            _numeric_values(dataframe, "outstanding")
            if "outstanding" in dataframe.columns
            else due - paid
        )
        overdue_mask = pd.Series(False, index=dataframe.index)
        if "status" in dataframe.columns:
            overdue_mask |= (
                dataframe["status"]
                .fillna("")
                .astype(str)
                .str.contains("overdue", case=False, regex=False)
            )
        if "days_overdue" in dataframe.columns:
            overdue_mask |= _numeric_values(dataframe, "days_overdue").fillna(0) > 0

        billed_total += float(due.sum(skipna=True))
        paid_total += float(paid.sum(skipna=True))
        outstanding_total += float(outstanding.sum(skipna=True))
        invoice_count += int(len(dataframe.index))
        overdue_count += int(overdue_mask.sum())
        usable_tables += 1

    if not usable_tables:
        return None
    collection_rate = (
        paid_total / billed_total
        if billed_total != 0
        else None
    )
    overdue_percentage = (
        overdue_count / invoice_count
        if invoice_count != 0
        else None
    )
    if collection_rate is None:
        _add_warning(warnings, "Collection rate unavailable because amount due is zero.")
    if overdue_percentage is None:
        _add_warning(warnings, "Overdue percentage unavailable because no invoices exist.")
    return {
        "amount_due": billed_total,
        "amount_paid": paid_total,
        "outstanding": outstanding_total,
        "collection_rate": collection_rate,
        "invoice_count": invoice_count,
        "overdue_invoice_count": overdue_count,
        "overdue_payment_percentage": overdue_percentage,
    }


def calculate_vendor_payment_totals(
    vendor_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> dict[str, float | int] | None:
    """Calculate vendor payment count and total value.

    Inputs: selected Vendor_Payments tables and optional warnings.
    Outputs: payment_count and total_amount, or None.
    Assumptions: each row represents one payment transaction.
    """

    table_list = list(vendor_tables)
    total = _sum_metric(
        table_list,
        ("amount", "payment", "actual_amount"),
        table_type="Vendor_Payments",
        metric_name="vendor payment total",
        warnings=warnings,
    )
    if total is None:
        return None
    maximum_values = [
        _numeric_values(table.dataframe, column).max(skipna=True)
        for table in table_list
        if (
            column := _first_existing_column(
                table.dataframe,
                ("amount", "payment", "actual_amount"),
            )
        )
        is not None
    ]
    maximum_payment = (
        float(max(value for value in maximum_values if pd.notna(value)))
        if any(pd.notna(value) for value in maximum_values)
        else None
    )
    return {
        "payment_count": sum(len(table.dataframe.index) for table in table_list),
        "total_amount": total,
        "maximum_payment_amount": maximum_payment,
    }


def calculate_scholarship_totals(
    scholarship_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> dict[str, float | int | None] | None:
    """Calculate scholarship allocation, awards, balances, and utilization.

    Inputs: selected Scholarships tables and optional warnings.
    Outputs: aggregate scholarship metrics, or None.
    Assumptions: awarded divided by allocated is the utilization rate.
    """

    table_list = list(scholarship_tables)
    if not table_list:
        if warnings is not None:
            append_missing_table_warning(table_list, "Scholarships", warnings)
        return None

    allocated = _sum_metric(
        table_list,
        ("allocated", "budget_amount"),
        table_type="Scholarships",
        metric_name="scholarship allocation",
        warnings=warnings,
    )
    awarded = _sum_metric(
        table_list,
        ("awarded", "actual_amount", "amount"),
        table_type="Scholarships",
        metric_name="scholarships awarded",
        warnings=warnings,
    )
    remaining = _sum_metric(
        table_list,
        ("remaining", "outstanding"),
        table_type="Scholarships",
        metric_name="scholarship balance",
        warnings=warnings,
    )
    recipients = _sum_metric(
        table_list,
        ("recipients", "count"),
        table_type="Scholarships",
        metric_name="scholarship recipients",
        warnings=warnings,
    )
    utilization = (
        awarded / allocated
        if awarded is not None and allocated not in {None, 0}
        else None
    )
    return {
        "allocated": allocated,
        "awarded": awarded,
        "remaining": remaining,
        "recipients": int(recipients) if recipients is not None else None,
        "utilization_rate": utilization,
    }


def _first_numeric_value(dataframe: pd.DataFrame, column: str) -> float | None:
    """Return the first usable numeric value from a column.

    Inputs: DataFrame and existing column name.
    Outputs: first number or None.
    Assumptions: row order represents chronological source order.
    """

    values = _numeric_values(dataframe, column).dropna()
    return float(values.iloc[0]) if not values.empty else None


def _last_numeric_value(dataframe: pd.DataFrame, column: str) -> float | None:
    """Return the last usable numeric value from a column.

    Inputs: DataFrame and existing column name.
    Outputs: last number or None.
    Assumptions: row order represents chronological source order.
    """

    values = _numeric_values(dataframe, column).dropna()
    return float(values.iloc[-1]) if not values.empty else None


def calculate_cash_flow_summary(
    cash_flow_tables: Sequence[LoadedIntermediateTable],
    warnings: list[str] | None = None,
) -> dict[str, float | None] | None:
    """Calculate aggregate cash movements and ending liquidity.

    Inputs: selected Cash_Flow tables and optional warnings.
    Outputs: beginning/ending cash, inflows, outflows, and net cash flow.
    Assumptions: rows are chronological within the selected reporting scope.
    """

    table_list = list(cash_flow_tables)
    if not table_list:
        if warnings is not None:
            append_missing_table_warning(table_list, "Cash_Flow", warnings)
        return None

    combined = pd.concat(
        [table.dataframe for table in table_list],
        ignore_index=True,
    )
    required = {
        "beginning_cash",
        "actual_cash_inflows",
        "actual_operating_outflows",
        "actual_scholarships",
        "actual_capital_outflows",
        "actual_net_cash_flow",
        "actual_ending_cash",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        _add_warning(
            warnings,
            f"Cash flow summary incomplete because columns are missing: {missing}.",
        )

    def sum_if_present(column: str) -> float | None:
        """Sum a cash column only when present.

        Inputs: normalized cash column name.
        Outputs: numeric total or None.
        Assumptions: this helper is local to the validated combined cash table.
        """

        return (
            float(_numeric_values(combined, column).sum(skipna=True))
            if column in combined.columns
            else None
        )

    return {
        "beginning_cash": (
            _first_numeric_value(combined, "beginning_cash")
            if "beginning_cash" in combined.columns
            else None
        ),
        "cash_inflows": sum_if_present("actual_cash_inflows"),
        "operating_outflows": sum_if_present("actual_operating_outflows"),
        "scholarship_outflows": sum_if_present("actual_scholarships"),
        "capital_outflows": sum_if_present("actual_capital_outflows"),
        "net_cash_flow": sum_if_present("actual_net_cash_flow"),
        "ending_cash": (
            _last_numeric_value(combined, "actual_ending_cash")
            if "actual_ending_cash" in combined.columns
            else None
        ),
        "ending_cash_variance": (
            _last_numeric_value(combined, "ending_cash_variance")
            if "ending_cash_variance" in combined.columns
            else None
        ),
    }


def _tables_with_month_number(
    tables: Sequence[LoadedIntermediateTable],
    *,
    year: int,
    table_type: str,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Combine tables and attach a calendar month number to every usable row.

    Inputs: normalized tables, target year, table type, and optional warnings.
    Outputs: combined DataFrame containing an internal _month_number column.
    Assumptions: date columns are preferred; month labels belong to target year.
    """

    parts: list[pd.DataFrame] = []
    for table in tables:
        dataframe = table.dataframe.copy()
        month_numbers: pd.Series | None = None
        for column in DATE_COLUMN_CANDIDATES:
            if column not in dataframe.columns:
                continue
            dates = pd.to_datetime(dataframe[column], errors="coerce")
            if dates.notna().any():
                dataframe = dataframe.loc[dates.dt.year == year].copy()
                month_numbers = dates.loc[dataframe.index].dt.month
                break
        if month_numbers is None and "month" in dataframe.columns:
            month_numbers = dataframe["month"].map(month_number_from_value)

        if month_numbers is None or not month_numbers.notna().any():
            _add_warning(
                warnings,
                f"Monthly trends skipped table '{table.table_id}': "
                "no usable date or month column exists.",
            )
            continue
        dataframe["_month_number"] = month_numbers.astype("Int64")
        dataframe = dataframe.dropna(subset=["_month_number"])
        parts.append(dataframe)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _monthly_sum(
    dataframe: pd.DataFrame,
    column_candidates: Sequence[str],
    output_name: str,
) -> pd.DataFrame:
    """Aggregate one available normalized value column by month.

    Inputs: month-tagged DataFrame, ordered column candidates, and output label.
    Outputs: _month_number/value DataFrame, possibly empty.
    Assumptions: the first supported candidate is the authoritative metric.
    """

    if dataframe.empty:
        return pd.DataFrame(columns=["_month_number", output_name])
    column = _first_existing_column(dataframe, column_candidates)
    if column is None:
        return pd.DataFrame(columns=["_month_number", output_name])
    values = pd.DataFrame(
        {
            "_month_number": dataframe["_month_number"],
            output_name: _numeric_values(dataframe, column),
        }
    )
    return values.groupby("_month_number", as_index=False)[output_name].sum()


def _merge_monthly_metric(
    base: pd.DataFrame,
    metric: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge one monthly metric onto the canonical 12-month axis.

    Inputs: base monthly axis and one metric DataFrame.
    Outputs: merged DataFrame.
    Assumptions: metric has at most one row per month.
    """

    return base.merge(metric, on="_month_number", how="left")


def calculate_monthly_trends(
    selected_tables: dict[str, list[LoadedIntermediateTable]],
    *,
    year: int,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate a January-to-December KPI trend table.

    Inputs: selected annual-scope tables, year, and optional warning collector.
    Outputs: 12 ordered rows containing monthly revenue, expense, payroll,
        collections, vendor, scholarship, and cash metrics.
    Assumptions: additive metrics use zero for months with no transactions;
        ratios and ending cash remain unavailable when their inputs are absent.
    """

    base = pd.DataFrame({"_month_number": range(1, 13)})
    base["period"] = base["_month_number"].map(
        lambda month: f"{year}-{month:02d}"
    )
    base["month"] = base["_month_number"].map(
        lambda month: calendar.month_name[month]
    )

    revenue = _tables_with_month_number(
        selected_tables.get("Revenue", []),
        year=year,
        table_type="Revenue",
        warnings=warnings,
    )
    expenses = _tables_with_month_number(
        selected_tables.get("Expenses", []),
        year=year,
        table_type="Expenses",
        warnings=warnings,
    )
    payroll = _tables_with_month_number(
        selected_tables.get("Payroll", []),
        year=year,
        table_type="Payroll",
        warnings=warnings,
    )
    students = _tables_with_month_number(
        selected_tables.get("Student_Payments", []),
        year=year,
        table_type="Student_Payments",
        warnings=warnings,
    )
    vendors = _tables_with_month_number(
        selected_tables.get("Vendor_Payments", []),
        year=year,
        table_type="Vendor_Payments",
        warnings=warnings,
    )
    scholarships = _tables_with_month_number(
        selected_tables.get("Scholarships", []),
        year=year,
        table_type="Scholarships",
        warnings=warnings,
    )
    cash = _tables_with_month_number(
        selected_tables.get("Cash_Flow", []),
        year=year,
        table_type="Cash_Flow",
        warnings=warnings,
    )

    monthly_metrics = [
        _monthly_sum(revenue, ("budget_revenue",), "budget_revenue"),
        _monthly_sum(revenue, ("actual_revenue", "revenue"), "actual_revenue"),
        _monthly_sum(expenses, ("budget_expense",), "budget_expenses"),
        _monthly_sum(
            expenses,
            ("actual_expense", "expenses"),
            "actual_expenses",
        ),
        _monthly_sum(payroll, ("total_payroll", "payroll"), "payroll_total"),
        _monthly_sum(students, ("amount_due",), "student_amount_due"),
        _monthly_sum(students, ("amount_paid",), "student_amount_paid"),
        _monthly_sum(students, ("outstanding",), "student_outstanding"),
        _monthly_sum(vendors, ("amount",), "vendor_payment_total"),
        _monthly_sum(scholarships, ("allocated",), "scholarship_allocated"),
        _monthly_sum(scholarships, ("awarded",), "scholarship_awarded"),
        _monthly_sum(cash, ("actual_cash_inflows",), "cash_inflows"),
        _monthly_sum(
            cash,
            ("actual_operating_outflows",),
            "operating_outflows",
        ),
        _monthly_sum(cash, ("actual_net_cash_flow",), "net_cash_flow"),
    ]
    trend = base
    for metric in monthly_metrics:
        trend = _merge_monthly_metric(trend, metric)

    if students.empty:
        overdue = pd.DataFrame(
            columns=["_month_number", "invoice_count", "overdue_invoice_count"]
        )
    else:
        overdue_mask = pd.Series(False, index=students.index)
        if "status" in students.columns:
            overdue_mask |= (
                students["status"]
                .fillna("")
                .astype(str)
                .str.contains("overdue", case=False, regex=False)
            )
        if "days_overdue" in students.columns:
            overdue_mask |= _numeric_values(students, "days_overdue").fillna(0) > 0
        overdue_rows = pd.DataFrame(
            {
                "_month_number": students["_month_number"],
                "invoice_count": 1,
                "overdue_invoice_count": overdue_mask.astype(int),
            }
        )
        overdue = overdue_rows.groupby("_month_number", as_index=False)[
            ["invoice_count", "overdue_invoice_count"]
        ].sum()
    trend = _merge_monthly_metric(trend, overdue)

    if cash.empty or "actual_ending_cash" not in cash.columns:
        ending_cash = pd.DataFrame(columns=["_month_number", "ending_cash"])
    else:
        ending_values = pd.DataFrame(
            {
                "_month_number": cash["_month_number"],
                "ending_cash": _numeric_values(cash, "actual_ending_cash"),
            }
        )
        ending_cash = ending_values.groupby("_month_number", as_index=False)[
            "ending_cash"
        ].last()
    trend = _merge_monthly_metric(trend, ending_cash)

    additive_columns = [
        column
        for column in trend.columns
        if column
        not in {
            "_month_number",
            "period",
            "month",
            "ending_cash",
        }
    ]
    trend[additive_columns] = trend[additive_columns].fillna(0.0)
    trend["net_operating_result"] = (
        trend["actual_revenue"] - trend["actual_expenses"]
    )
    trend["revenue_variance"] = (
        trend["actual_revenue"] - trend["budget_revenue"]
    )
    trend["expense_variance"] = (
        trend["actual_expenses"] - trend["budget_expenses"]
    )
    trend["revenue_variance_pct"] = (
        trend["revenue_variance"] / trend["budget_revenue"].replace(0, pd.NA)
    )
    trend["expense_variance_pct"] = (
        trend["expense_variance"] / trend["budget_expenses"].replace(0, pd.NA)
    )
    trend["payroll_percentage_of_revenue"] = (
        trend["payroll_total"] / trend["actual_revenue"].replace(0, pd.NA)
    )
    trend["student_collection_rate"] = (
        trend["student_amount_paid"]
        / trend["student_amount_due"].replace(0, pd.NA)
    )
    trend["overdue_payment_percentage"] = (
        trend["overdue_invoice_count"]
        / trend["invoice_count"].replace(0, pd.NA)
    )

    ordered_columns = [
        "period",
        "month",
        "budget_revenue",
        "actual_revenue",
        "revenue_variance",
        "revenue_variance_pct",
        "budget_expenses",
        "actual_expenses",
        "expense_variance",
        "expense_variance_pct",
        "net_operating_result",
        "payroll_total",
        "payroll_percentage_of_revenue",
        "student_amount_due",
        "student_amount_paid",
        "student_outstanding",
        "student_collection_rate",
        "invoice_count",
        "overdue_invoice_count",
        "overdue_payment_percentage",
        "vendor_payment_total",
        "scholarship_allocated",
        "scholarship_awarded",
        "cash_inflows",
        "operating_outflows",
        "net_cash_flow",
        "ending_cash",
    ]
    return trend[ordered_columns].reset_index(drop=True)
