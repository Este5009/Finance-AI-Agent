"""Helpers for selecting calculation inputs by detected table type."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from finance_agent.calculations.calculation_loader import (
    LoadedIntermediateModel,
    LoadedIntermediateTable,
)
from finance_agent.ingestion.schema import clean_column_name


SUPPORTED_FINANCIAL_TABLE_TYPES = (
    "Revenue",
    "Expenses",
    "Budget_vs_Actual",
    "Payroll",
    "Student_Payments",
    "Cash_Flow",
    "Vendor_Payments",
    "Scholarships",
    "Department_Summary",
)


def _normalized_type(table_type: str) -> str:
    """Normalize a requested or detected table type for comparison.

    Inputs: table type label.
    Outputs: lowercase snake_case identifier.
    Assumptions: type matching should be case and spacing insensitive.
    """

    return clean_column_name(table_type)


def _source_matches(table_source: str, source_workbook: str | None) -> bool:
    """Check whether a table belongs to an optional source-workbook scope.

    Inputs: manifest source path and optional requested source path/name.
    Outputs: True when no filter exists or filenames/paths match.
    Assumptions: filenames uniquely identify reporting scopes in one model.
    """

    if source_workbook is None:
        return True
    requested = Path(source_workbook)
    table_path = Path(table_source)
    if requested.name.lower() == table_path.name.lower():
        return True
    return str(requested).lower() == str(table_path).lower()


def find_tables_by_type(
    model: LoadedIntermediateModel,
    detected_type: str,
    *,
    source_workbook: str | None = None,
) -> list[LoadedIntermediateTable]:
    """Find all model tables matching a detected type and optional source.

    Inputs: loaded model, requested type, and optional workbook scope.
    Outputs: matching tables in manifest order; an empty list is valid.
    Assumptions: multiple logical tables of one type may need concatenation.
    """

    requested_type = _normalized_type(detected_type)
    return [
        table
        for table in model.tables
        if _normalized_type(table.detected_type) == requested_type
        and _source_matches(table.source_workbook, source_workbook)
    ]


def select_financial_tables(
    model: LoadedIntermediateModel,
    *,
    source_workbook: str | None = None,
    table_types: Iterable[str] = SUPPORTED_FINANCIAL_TABLE_TYPES,
) -> dict[str, list[LoadedIntermediateTable]]:
    """Select every requested financial table type.

    Inputs: loaded model, optional workbook scope, and requested type labels.
    Outputs: dictionary keyed by canonical requested type.
    Assumptions: missing types map to empty lists rather than raising exceptions.
    """

    return {
        table_type: find_tables_by_type(
            model,
            table_type,
            source_workbook=source_workbook,
        )
        for table_type in table_types
    }


def append_missing_table_warning(
    tables: list[LoadedIntermediateTable],
    table_type: str,
    warnings: list[str],
) -> None:
    """Add a clear warning when a requested table type is unavailable.

    Inputs: selected tables, requested type, and mutable warning list.
    Outputs: warning list is updated only when no tables were selected.
    Assumptions: duplicate warnings are not useful to downstream consumers.
    """

    if not tables:
        warning = (
            f"Metric unavailable: no '{table_type}' table exists "
            "for the selected reporting scope."
        )
        if warning not in warnings:
            warnings.append(warning)
