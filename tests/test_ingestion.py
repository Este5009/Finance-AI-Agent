"""Tests for deterministic document ingestion and inspection."""

from pathlib import Path

import pandas as pd
import pytest

from finance_agent.ingestion.ingestion import extract_goals_pdf, inspect_sheet, load_excel_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_GOALS_PDF = PROJECT_ROOT / "data" / "synthetic" / "financial_goals_2026.pdf"


def test_load_excel_workbook_reads_all_sheets(tmp_path: Path) -> None:
    """Verify loading preserves sheet names, metadata, and DataFrames."""

    workbook_path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame({"Amount": [100, 200]}).to_excel(
            writer, sheet_name="Revenue Original", index=False
        )
        pd.DataFrame({"Vendor": ["A"]}).to_excel(
            writer, sheet_name="Vendor Payments", index=False
        )

    result = load_excel_workbook(workbook_path)

    assert result.sheet_names == ["Revenue Original", "Vendor Payments"]
    assert result.row_counts == {"Revenue Original": 2, "Vendor Payments": 1}
    assert result.column_names["Revenue Original"] == ["Amount"]
    assert isinstance(result.dataframes["Vendor Payments"], pd.DataFrame)


def test_load_excel_workbook_reports_missing_file(tmp_path: Path) -> None:
    """Verify a missing workbook produces a clear file error."""

    with pytest.raises(FileNotFoundError, match="Input file does not exist"):
        load_excel_workbook(tmp_path / "missing.xlsx")


def test_inspect_sheet_returns_required_summary() -> None:
    """Verify inspection includes structure, samples, dtypes, and missing values."""

    dataframe = pd.DataFrame(
        {"Department": ["Engineering", "Business"], "Amount": [100.0, None]}
    )
    summary = inspect_sheet("Expenses", dataframe)

    assert summary["sheet_name"] == "Expenses"
    assert summary["row_count"] == 2
    assert summary["column_count"] == 2
    assert summary["column_names"] == ["Department", "Amount"]
    assert summary["inferred_data_types"]["Amount"] == "float64"
    assert summary["sample_rows"][0]["Department"] == "Engineering"
    assert summary["missing_value_counts"] == {"Department": 0, "Amount": 1}


@pytest.mark.skipif(not SYNTHETIC_GOALS_PDF.exists(), reason="Synthetic goals PDF is unavailable")
def test_extract_goals_pdf_returns_text_and_metadata() -> None:
    """Verify goals PDF extraction returns useful text and page metadata."""

    result = extract_goals_pdf(SYNTHETIC_GOALS_PDF)

    assert result.metadata["page_count"] >= 1
    assert result.metadata["file_size_bytes"] > 0
    assert "Financial Goals 2026" in result.raw_text
    assert "Payroll" in result.raw_text
