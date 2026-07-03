"""Tests for classification, normalization confidence, and feature extraction."""

import pandas as pd

from finance_agent.classification import classify_table
from finance_agent.features import extract_financial_features
from finance_agent.models import DetectedRawTable
from finance_agent.normalization import normalize_detected_table


def _detected_table(
    *,
    sheet_name: str,
    columns: list[str],
    rows: list[list[object]],
) -> DetectedRawTable:
    """Build a detected-table fixture.

    Inputs: sheet name, original columns, and row values.
    Outputs: DetectedRawTable suitable for downstream unit tests.
    Assumptions: fixture coordinates are representative rather than workbook-derived.
    """

    return DetectedRawTable(
        table_index=1,
        sheet_name=sheet_name,
        header_row=3,
        start_row=3,
        end_row=3 + len(rows),
        start_column=1,
        end_column=len(columns),
        header_confidence=0.90,
        title=None,
        original_columns=columns,
        dataframe=pd.DataFrame(rows, columns=columns),
    )


def test_mixed_language_normalization_has_column_confidence() -> None:
    """Verify mixed Spanish/English headers map with explicit confidence."""

    table = _detected_table(
        sheet_name="Ingresos",
        columns=["Departamento", "Month", "Ingreso Presupuestado", "Actual Revenue"],
        rows=[["Engineering", "June", 1000, 950]],
    )

    normalized = normalize_detected_table(table)

    assert normalized.normalized_columns == [
        "department",
        "month",
        "budget_revenue",
        "actual_revenue",
    ]
    assert all(mapping.confidence >= 0.95 for mapping in normalized.column_mappings)
    assert not any(
        mapping.requires_interpretation
        for mapping in normalized.column_mappings
    )


def test_classifies_revenue_table_with_confidence() -> None:
    """Verify sheet and column evidence classify revenue automatically."""

    table = _detected_table(
        sheet_name="Revenue",
        columns=[
            "Department",
            "Revenue Category",
            "Budget Revenue",
            "Actual Revenue",
        ],
        rows=[["Engineering", "Tuition", 1000, 950]],
    )
    normalized = normalize_detected_table(table)

    classification = classify_table(table, normalized)

    assert classification.detected_type == "Revenue"
    assert classification.confidence >= 0.90
    assert classification.requires_interpretation is False


def test_uncertain_table_is_preserved_as_unknown() -> None:
    """Verify weak ambiguous evidence is classified as Unknown."""

    table = _detected_table(
        sheet_name="Other Data",
        columns=["Reference", "Comment"],
        rows=[["A", "Review later"]],
    )
    normalized = normalize_detected_table(table)

    classification = classify_table(table, normalized)

    assert classification.detected_type == "Unknown"
    assert classification.requires_interpretation is True


def test_feature_extraction_separates_dimensions_and_metrics() -> None:
    """Verify financial dimensions and values are extracted without final modeling."""

    table = _detected_table(
        sheet_name="Payroll",
        columns=["Departamento", "Mes", "Salario Base", "Beneficios", "Horas Extra"],
        rows=[["Health Sciences", "June", 1000, 180, 50]],
    )
    normalized = normalize_detected_table(table)

    dimensions, metrics = extract_financial_features(normalized)

    assert {feature.semantic_name for feature in dimensions} == {"department", "month"}
    assert {feature.semantic_name for feature in metrics} == {
        "base_salary",
        "benefits",
        "overtime",
    }


def test_financial_category_is_extracted_as_dimension() -> None:
    """Verify a category descriptor takes precedence over its revenue token."""

    table = _detected_table(
        sheet_name="Revenue",
        columns=["Revenue Category", "Actual Revenue"],
        rows=[["Tuition", 1000]],
    )
    normalized = normalize_detected_table(table)

    dimensions, metrics = extract_financial_features(normalized)

    assert [feature.semantic_name for feature in dimensions] == ["revenue_category"]
    assert [feature.semantic_name for feature in metrics] == ["actual_revenue"]
