"""Run Step 2 workbook understanding and intermediate-model generation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution begins in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.intermediate import (  # noqa: E402
    build_financial_document_model,
    save_intermediate_outputs,
)


MONTHLY_REPORT = PROJECT_ROOT / "data" / "synthetic" / "monthly_financial_report_june_2026.xlsx"
ANNUAL_REPORT = PROJECT_ROOT / "data" / "synthetic" / "annual_financial_report_2026.xlsx"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "intermediate"


def main() -> None:
    """Build and save the Step 2 intermediate financial document model.

    Inputs: monthly and annual synthetic workbook paths.
    Outputs: model JSON, feature summary JSON, and one normalized CSV per table.
    Assumptions: these workbooks are test inputs, not a fixed schema contract.
    """

    print("Finance AI Agent - Step 2 Document Understanding")
    model = build_financial_document_model([MONTHLY_REPORT, ANNUAL_REPORT])
    output_paths = save_intermediate_outputs(model, OUTPUT_DIRECTORY)

    print(f"Source workbooks: {len(model.source_workbooks)}")
    print(f"Worksheets inspected: {len(model.sheet_analyses)}")
    print(f"Logical tables detected: {len(model.tables)}")
    print(
        "Tables requiring future interpretation: "
        f"{sum(table.requires_future_interpretation for table in model.tables)}"
    )
    print("\nDetected tables:")
    for table in model.tables:
        review_label = "review" if table.requires_future_interpretation else "automatic"
        print(
            f"  - {table.table_id}: {table.detected_type} "
            f"(confidence={table.confidence:.2f}, {review_label}, "
            f"rows={len(table.cleaned_dataframe)})"
        )

    print("\nIntermediate outputs saved:")
    print(f"  - {output_paths['financial_document_model']}")
    print(f"  - {output_paths['feature_summary']}")
    print(f"  - {output_paths['normalized_tables']}")


if __name__ == "__main__":
    main()
