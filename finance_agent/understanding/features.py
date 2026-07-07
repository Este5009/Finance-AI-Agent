"""Financial dimension and metric extraction from normalized tables."""

from __future__ import annotations

import re

import pandas as pd

from finance_agent.understanding.models import ExtractedFeature
from finance_agent.understanding.normalization import NormalizedTableData


DIMENSION_SIGNALS = {
    "approval",
    "category",
    "date",
    "day",
    "department",
    "id",
    "invoice",
    "method",
    "month",
    "name",
    "period",
    "program",
    "scholarship_type",
    "status",
    "student",
    "student_year",
    "type",
    "vendor",
    "year",
}

METRIC_SIGNALS = {
    "actual",
    "allocated",
    "amount",
    "awarded",
    "balance",
    "benefits",
    "budget",
    "cash",
    "count",
    "expense",
    "expenses",
    "flow",
    "headcount",
    "inflows",
    "outflows",
    "outstanding",
    "overtime",
    "paid",
    "payroll",
    "pct",
    "rate",
    "recipients",
    "remaining",
    "revenue",
    "salary",
    "total",
    "variance",
}


def _base_semantic_name(column_name: str) -> str:
    """Remove only duplicate-name suffixes from a normalized column.

    Inputs: normalized unique column name.
    Outputs: semantic base name.
    Assumptions: suffixes such as _2 are produced by deterministic de-duplication.
    """

    return re.sub(r"_\d+$", "", column_name)


def _column_role(column_name: str, series: pd.Series) -> tuple[str, float]:
    """Infer whether a column is a dimension or metric.

    Inputs: normalized column name and its cleaned pandas Series.
    Outputs: role and semantic-role confidence.
    Assumptions: names outweigh dtype because identifiers can be numeric.
    """

    semantic_name = _base_semantic_name(column_name)
    tokens = set(semantic_name.split("_"))
    metric_match = bool(tokens & METRIC_SIGNALS)
    dimension_match = bool(tokens & DIMENSION_SIGNALS)

    if dimension_match:
        # Explicit descriptors such as revenue_category and scholarship_type
        # remain dimensions even though another token names a financial domain.
        return "dimension", 0.96
    if metric_match:
        return "metric", 0.96
    if pd.api.types.is_numeric_dtype(series):
        return "metric", 0.65
    if pd.api.types.is_datetime64_any_dtype(series):
        return "dimension", 0.85
    return "dimension", 0.55


def extract_financial_features(
    normalized_table: NormalizedTableData,
) -> tuple[list[ExtractedFeature], list[ExtractedFeature]]:
    """Extract financial dimensions and metrics without forcing a final schema.

    Inputs: normalized table data and column-confidence mappings.
    Outputs: dimension and metric feature lists with confidence.
    Assumptions: every retained column remains useful even when confidence is low.
    """

    dimensions: list[ExtractedFeature] = []
    metrics: list[ExtractedFeature] = []
    mapping_by_name = {
        mapping.normalized_name: mapping
        for mapping in normalized_table.column_mappings
    }

    for column_name in normalized_table.normalized_columns:
        role, role_confidence = _column_role(
            column_name,
            normalized_table.dataframe[column_name],
        )
        mapping = mapping_by_name[column_name]
        feature = ExtractedFeature(
            original_column=mapping.original_name,
            normalized_column=column_name,
            semantic_name=_base_semantic_name(column_name),
            role=role,
            # Both the name mapping and role inference must be trustworthy.
            confidence=round(min(mapping.confidence, role_confidence), 4),
        )
        if role == "metric":
            metrics.append(feature)
        else:
            dimensions.append(feature)

    return dimensions, metrics
