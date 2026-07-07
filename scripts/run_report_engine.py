"""Generate renderer-agnostic report model JSON artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution starts in scripts/, so expose the package root.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.reporting import (  # noqa: E402
    build_report_model,
    load_report_inputs,
    save_report_model,
)


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "report"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for report model generation.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default project/output paths follow current pipeline layout.
    """

    parser = argparse.ArgumentParser(
        description="Build renderer-agnostic report model artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    return parser


def _build_and_save(period_slug: str, output_path: Path) -> tuple[Path, int]:
    """Build and save one period report model.

    Inputs: period slug and destination path.
    Outputs: written path and generated section count.
    Assumptions: processed upstream outputs already exist.
    """

    inputs = load_report_inputs(PROJECT_ROOT, period_slug)
    model = build_report_model(inputs)
    path = save_report_model(model, output_path)
    return path, len(model.sections)


def main() -> None:
    """Generate June and annual renderer-agnostic report models.

    Inputs: existing processed pipeline outputs.
    Outputs: two JSON report model artifacts under outputs/report.
    Assumptions: no PDF, HTML, Streamlit, email, or PowerPoint is generated.
    """

    args = build_argument_parser().parse_args()
    june_path, june_sections = _build_and_save(
        "june_2026",
        args.output_dir / "report_model_june_2026.json",
    )
    annual_path, annual_sections = _build_and_save(
        "2026",
        args.output_dir / "report_model_2026.json",
    )

    print("Finance AI Agent - Step 10A Reporting Engine")
    print(f"June 2026 sections generated: {june_sections}")
    print(f"Annual 2026 sections generated: {annual_sections}")
    print("Outputs saved:")
    print(f"  - {june_path}")
    print(f"  - {annual_path}")


if __name__ == "__main__":
    main()
