"""Column-level normalization for detected financial tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from finance_agent.understanding.models import ColumnMapping, DetectedRawTable
from finance_agent.ingestion.schema import COLUMN_ALIASES, clean_column_name


COLUMN_CONFIDENCE_THRESHOLD = 0.75

COMPOSITE_SEMANTIC_TOKENS = {
    "actual",
    "allocated",
    "amount",
    "approval",
    "awarded",
    "beginning",
    "benefits",
    "budget",
    "capital",
    "cash",
    "category",
    "count",
    "date",
    "department",
    "due",
    "ending",
    "expense",
    "expenses",
    "flow",
    "headcount",
    "inflows",
    "invoice",
    "method",
    "net",
    "operating",
    "outflows",
    "outstanding",
    "overtime",
    "paid",
    "payment",
    "payroll",
    "pct",
    "recipients",
    "remaining",
    "revenue",
    "salary",
    "scholarship",
    "scholarships",
    "student",
    "total",
    "variance",
    "vendor",
}


@dataclass
class NormalizedTableData:
    """Normalized columns, confidence mappings, and cleaned row data."""

    original_columns: list[str]
    normalized_columns: list[str]
    column_mappings: list[ColumnMapping]
    dataframe: pd.DataFrame


def _column_normalization(column_name: object) -> tuple[str, float, bool]:
    """Normalize one column and assign deterministic mapping confidence.

    Inputs: original column label.
    Outputs: normalized name, confidence, and whether an alias matched.
    Assumptions: unknown but readable labels are preserved with moderate confidence.
    """

    cleaned = clean_column_name(column_name)
    if not cleaned:
        return "", 0.10, False

    normalized_aliases = {
        clean_column_name(source): clean_column_name(target)
        for source, target in COLUMN_ALIASES.items()
    }
    if cleaned in normalized_aliases:
        normalized = normalized_aliases[cleaned]
        # A translated alias is slightly stronger evidence than a canonical
        # pass-through because the mapping rule explicitly resolved ambiguity.
        confidence = 0.98 if normalized != cleaned else 0.95
        return normalized, confidence, True
    tokens = set(cleaned.split("_"))
    if tokens and tokens <= COMPOSITE_SEMANTIC_TOKENS:
        # Compound labels such as budget_operating_outflows are unambiguous
        # compositions of known financial concepts even without a dedicated alias.
        return cleaned, 0.85, False
    return cleaned, 0.55, False


def _make_unique_names(names: list[str]) -> list[str]:
    """Make normalized names unique without losing their semantic base.

    Inputs: normalized column-name candidates.
    Outputs: unique names with stable numeric suffixes.
    Assumptions: duplicate semantics must be retained for future interpretation.
    """

    unique_names: list[str] = []
    occurrences: dict[str, int] = {}
    for position, name in enumerate(names, start=1):
        base = name or f"unnamed_column_{position}"
        occurrences[base] = occurrences.get(base, 0) + 1
        count = occurrences[base]
        unique_names.append(base if count == 1 else f"{base}_{count}")
    return unique_names


def _clean_cell_value(value: Any) -> Any:
    """Apply conservative cell cleaning without financial transformation.

    Inputs: one table cell value.
    Outputs: trimmed text, pandas missing value, or original typed value.
    Assumptions: number/date coercion belongs to later schema and finance stages.
    """

    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else pd.NA
    return value


def normalize_detected_table(table: DetectedRawTable) -> NormalizedTableData:
    """Normalize columns and clean cells in a detected raw table.

    Inputs: one logical table from document understanding.
    Outputs: cleaned DataFrame and per-column confidence mappings.
    Assumptions: row values are preserved; only whitespace-only rows are removed.
    """

    preliminary: list[str] = []
    confidences: list[float] = []
    alias_matches: list[bool] = []
    for original_column in table.original_columns:
        normalized, confidence, matched_alias = _column_normalization(original_column)
        preliminary.append(normalized)
        confidences.append(confidence)
        alias_matches.append(matched_alias)

    normalized_columns = _make_unique_names(preliminary)
    mappings = [
        ColumnMapping(
            original_name=original,
            normalized_name=normalized,
            confidence=confidence,
            matched_alias=matched_alias,
            requires_interpretation=confidence < COLUMN_CONFIDENCE_THRESHOLD,
        )
        for original, normalized, confidence, matched_alias in zip(
            table.original_columns,
            normalized_columns,
            confidences,
            alias_matches,
            strict=True,
        )
    ]

    cleaned = table.dataframe.copy()
    cleaned.columns = normalized_columns
    for column in cleaned.columns:
        # Column-wise map avoids coercing the entire mixed-type DataFrame to text.
        cleaned[column] = cleaned[column].map(_clean_cell_value)
    cleaned = cleaned.dropna(how="all").reset_index(drop=True)

    return NormalizedTableData(
        original_columns=list(table.original_columns),
        normalized_columns=normalized_columns,
        column_mappings=mappings,
        dataframe=cleaned,
    )
