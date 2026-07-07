"""Tests for loading and selecting intermediate calculation inputs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from finance_agent.calculations.calculation_loader import (
    IntermediateModelLoadError,
    load_intermediate_model,
)
from finance_agent.calculations.table_selection import find_tables_by_type


def _write_model_fixture(
    tmp_path: Path,
    *,
    include_csv: bool = True,
) -> Path:
    """Write a minimal intermediate manifest and optional referenced CSV.

    Inputs: temporary directory and whether to create the CSV.
    Outputs: path to the fixture financial_document_model.json.
    Assumptions: fixture mirrors the Step 2 manifest fields consumed by Step 3.
    """

    normalized_dir = tmp_path / "normalized_tables"
    normalized_dir.mkdir()
    csv_path = normalized_dir / "revenue.csv"
    if include_csv:
        pd.DataFrame(
            {
                "department": ["Engineering"],
                "actual_revenue": [1000],
            }
        ).to_csv(csv_path, index=False)

    manifest = {
        "model_version": "2.0",
        "source_workbooks": ["monthly.xlsx"],
        "tables": [
            {
                "table_id": "monthly__revenue__table_01",
                "detected_type": "Revenue",
                "source_workbook": "monthly.xlsx",
                "sheet": "Revenue",
                "confidence": 0.99,
                "row_count": 1,
                "normalized_columns": ["department", "actual_revenue"],
                "normalized_table_file": "normalized_tables/revenue.csv",
            }
        ],
    }
    model_path = tmp_path / "financial_document_model.json"
    model_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path


def test_intermediate_model_loader_reads_referenced_csv(tmp_path: Path) -> None:
    """Verify the loader returns structured metadata and DataFrames."""

    model = load_intermediate_model(_write_model_fixture(tmp_path))

    assert model.model_version == "2.0"
    assert len(model.tables) == 1
    assert model.tables[0].detected_type == "Revenue"
    assert model.tables[0].dataframe["actual_revenue"].sum() == 1000


def test_intermediate_model_loader_rejects_missing_csv(tmp_path: Path) -> None:
    """Verify a broken CSV reference raises a clear model error."""

    model_path = _write_model_fixture(tmp_path, include_csv=False)

    with pytest.raises(
        IntermediateModelLoadError,
        match="Referenced normalized CSV does not exist",
    ):
        load_intermediate_model(model_path)


def test_table_selection_matches_type_and_source(tmp_path: Path) -> None:
    """Verify table selection is type/case tolerant and source scoped."""

    model = load_intermediate_model(_write_model_fixture(tmp_path))

    selected = find_tables_by_type(
        model,
        "revenue",
        source_workbook="monthly.xlsx",
    )
    missing_scope = find_tables_by_type(
        model,
        "Revenue",
        source_workbook="annual.xlsx",
    )

    assert len(selected) == 1
    assert missing_scope == []
