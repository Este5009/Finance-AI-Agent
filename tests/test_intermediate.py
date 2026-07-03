"""Tests for intermediate model assembly and output serialization."""

import json
from pathlib import Path

import pandas as pd

from finance_agent.intermediate import (
    build_financial_document_model,
    save_intermediate_outputs,
)


def test_pipeline_preserves_multiple_tables_and_writes_one_csv_each(
    tmp_path: Path,
) -> None:
    """Verify the model bridges a multi-table workbook to JSON and CSV outputs."""

    workbook_path = tmp_path / "mixed_finance.xlsx"
    worksheet_rows = [
        ["Revenue Table", None, None],
        ["Department", "Revenue Category", "Actual Revenue"],
        ["Engineering", "Tuition", 1000],
        ["Business", "Tuition", 800],
        [None, None, None],
        ["Vendor", "Invoice Number", "Amount"],
        ["Supplier A", "INV-1", 400],
        ["Supplier B", "INV-2", 500],
    ]
    pd.DataFrame(worksheet_rows).to_excel(
        workbook_path,
        sheet_name="Finance Data",
        index=False,
        header=False,
    )

    model = build_financial_document_model([workbook_path])
    paths = save_intermediate_outputs(model, tmp_path / "outputs")

    assert len(model.tables) == 2
    assert {table.detected_type for table in model.tables} == {
        "Revenue",
        "Vendor_Payments",
    }
    csv_files = list(paths["normalized_tables"].glob("*.csv"))
    assert len(csv_files) == 2

    model_json = json.loads(
        paths["financial_document_model"].read_text(encoding="utf-8")
    )
    assert model_json["table_count"] == 2
    assert all(table["normalized_table_file"] for table in model_json["tables"])
    assert paths["feature_summary"].exists()
