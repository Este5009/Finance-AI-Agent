"""Run Step 1 ingestion against the project's synthetic documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct execution starts imports from scripts/, so expose the project root
    # without introducing package-installation work in this foundation phase.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.ingestion.ingestion import inspect_workbook, load_excel_workbook  # noqa: E402


MONTHLY_REPORT = PROJECT_ROOT / "data" / "synthetic" / "monthly_financial_report_june_2026.xlsx"
ANNUAL_REPORT = PROJECT_ROOT / "data" / "synthetic" / "annual_financial_report_2026.xlsx"
INSPECTION_DIR = PROJECT_ROOT / "outputs" / "inspection"
SYNTHETIC_HEADER_ROW = 4


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a JSON-compatible dictionary with readable formatting.

    Inputs: dictionary and destination path.
    Outputs: one UTF-8 JSON file.
    Assumptions: DataFrames were converted to inspection records first.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_workbook_summary(label: str, inspection: dict[str, Any]) -> None:
    """Print a compact workbook inspection summary.

    Inputs: report label and workbook inspection dictionary.
    Outputs: human-readable console lines.
    Assumptions: full samples and missing counts belong in JSON.
    """

    print(f"\n{label}")
    print(f"  Path: {inspection['workbook_path']}")
    print(f"  Sheets: {inspection['sheet_count']}")
    for sheet in inspection["sheets"]:
        print(
            f"  - {sheet['sheet_name']}: "
            f"{sheet['row_count']} rows x {sheet['column_count']} columns"
        )
        print(f"    Columns: {', '.join(sheet['column_names'])}")


def main() -> None:
    """Load synthetic documents and save Step 1 inspection outputs.

    Inputs: fixed project-relative synthetic document paths.
    Outputs: workbook JSON inspections and console summaries.
    Assumptions: generated workbooks use row 5 as their table header.
    """

    print("Finance AI Agent - Step 1 Document Ingestion")
    monthly = load_excel_workbook(MONTHLY_REPORT, header_row=SYNTHETIC_HEADER_ROW)
    annual = load_excel_workbook(ANNUAL_REPORT, header_row=SYNTHETIC_HEADER_ROW)
    monthly_inspection = inspect_workbook(monthly)
    annual_inspection = inspect_workbook(annual)

    print_workbook_summary("Monthly workbook", monthly_inspection)
    print_workbook_summary("Annual workbook", annual_inspection)

    outputs = [
        (INSPECTION_DIR / "monthly_workbook_inspection.json", monthly_inspection),
        (INSPECTION_DIR / "annual_workbook_inspection.json", annual_inspection),
    ]
    for output_path, inspection in outputs:
        save_json(inspection, output_path)

    print("\nInspection outputs saved:")
    for output_path, _ in outputs:
        print(f"  - {output_path.resolve()}")


if __name__ == "__main__":
    main()
