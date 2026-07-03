"""Period scopes and row filtering for normalized financial tables."""

from __future__ import annotations

import calendar
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

import pandas as pd

from finance_agent.calculation_loader import LoadedIntermediateTable


DATE_COLUMN_CANDIDATES = (
    "period",
    "billing_period",
    "payment_date",
    "date",
    "due_date",
)


@dataclass(frozen=True)
class PeriodScope:
    """Explicit monthly, annual, or custom calculation period."""

    mode: Literal["monthly", "annual", "custom"]
    label: str
    start_date: date
    end_date: date

    @classmethod
    def monthly(cls, year: int, month: int, label: str | None = None) -> "PeriodScope":
        """Create a calendar-month calculation scope.

        Inputs: year, month number, and optional display label.
        Outputs: inclusive first-to-last-day monthly scope.
        Assumptions: calendar months, not institution-specific fiscal periods.
        """

        if month < 1 or month > 12:
            raise ValueError("month must be between 1 and 12")
        last_day = calendar.monthrange(year, month)[1]
        return cls(
            mode="monthly",
            label=label or f"{calendar.month_name[month]} {year}",
            start_date=date(year, month, 1),
            end_date=date(year, month, last_day),
        )

    @classmethod
    def annual(cls, year: int, label: str | None = None) -> "PeriodScope":
        """Create a calendar-year calculation scope.

        Inputs: year and optional display label.
        Outputs: inclusive January-through-December annual scope.
        Assumptions: the project currently uses a calendar fiscal year.
        """

        return cls(
            mode="annual",
            label=label or str(year),
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
        )

    @classmethod
    def custom(
        cls,
        start_date: date,
        end_date: date,
        label: str | None = None,
    ) -> "PeriodScope":
        """Create an arbitrary inclusive date-range scope.

        Inputs: start date, end date, and optional display label.
        Outputs: validated custom scope.
        Assumptions: normalized rows contain a date or recognizable month.
        """

        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        return cls(
            mode="custom",
            label=label or f"{start_date.isoformat()} to {end_date.isoformat()}",
            start_date=start_date,
            end_date=end_date,
        )


def month_number_from_value(value: object) -> int | None:
    """Convert an English month name or numeric value to a month number.

    Inputs: normalized table month value.
    Outputs: month number from 1 to 12, or None.
    Assumptions: Step 2 preserves English month labels in the synthetic inputs.
    """

    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and 1 <= int(value) <= 12:
        return int(value)
    text = str(value).strip().lower()
    month_lookup = {
        name.lower(): number
        for number, name in enumerate(calendar.month_name)
        if name
    }
    month_lookup.update(
        {
            name.lower(): number
            for number, name in enumerate(calendar.month_abbr)
            if name
        }
    )
    return month_lookup.get(text)


def _allowed_months(scope: PeriodScope) -> set[int]:
    """Return calendar month numbers touched by a period scope.

    Inputs: validated period scope.
    Outputs: set of month numbers.
    Assumptions: month-only fallback is safe within one source reporting year.
    """

    month_numbers: set[int] = set()
    cursor = pd.Timestamp(scope.start_date).to_period("M")
    end_period = pd.Timestamp(scope.end_date).to_period("M")
    while cursor <= end_period:
        month_numbers.add(cursor.month)
        cursor += 1
    return month_numbers


def filter_table_for_period(
    table: LoadedIntermediateTable,
    scope: PeriodScope,
    warnings: list[str] | None = None,
) -> LoadedIntermediateTable | None:
    """Filter one normalized table to an inclusive period.

    Inputs: loaded table, period scope, and optional warning collector.
    Outputs: copied table with filtered DataFrame, or None when no period field exists.
    Assumptions: a recognized date column is stronger evidence than a month label.
    """

    dataframe = table.dataframe
    for column in DATE_COLUMN_CANDIDATES:
        if column not in dataframe.columns:
            continue
        parsed_dates = pd.to_datetime(dataframe[column], errors="coerce")
        if parsed_dates.notna().any():
            mask = parsed_dates.between(
                pd.Timestamp(scope.start_date),
                pd.Timestamp(scope.end_date),
                inclusive="both",
            )
            return replace(
                table,
                dataframe=dataframe.loc[mask].reset_index(drop=True),
            )

    if "month" in dataframe.columns:
        month_numbers = dataframe["month"].map(month_number_from_value)
        if month_numbers.notna().any():
            mask = month_numbers.isin(_allowed_months(scope))
            return replace(
                table,
                dataframe=dataframe.loc[mask].reset_index(drop=True),
            )

    if warnings is not None:
        warning = (
            f"Table '{table.table_id}' was excluded from {scope.label}: "
            "no usable period, date, or month column exists."
        )
        if warning not in warnings:
            warnings.append(warning)
    return None


def filter_selected_tables_for_period(
    selected_tables: dict[str, list[LoadedIntermediateTable]],
    scope: PeriodScope,
    warnings: list[str] | None = None,
) -> dict[str, list[LoadedIntermediateTable]]:
    """Filter all calculation input tables while preserving type keys.

    Inputs: selected table dictionary, period scope, and warnings.
    Outputs: same dictionary structure with row-filtered table copies.
    Assumptions: Department_Summary is not a primary calculation input.
    """

    filtered: dict[str, list[LoadedIntermediateTable]] = {}
    for table_type, tables in selected_tables.items():
        if table_type == "Department_Summary":
            # The engine derives department totals from detailed Revenue and
            # Expenses tables, so this static summary is retained but unused.
            filtered[table_type] = list(tables)
            continue
        filtered[table_type] = [
            filtered_table
            for table in tables
            if (
                filtered_table := filter_table_for_period(
                    table,
                    scope,
                    warnings,
                )
            )
            is not None
        ]
    return filtered
