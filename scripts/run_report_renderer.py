"""Render human-readable HTML and PDF reports from report model JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution starts in scripts/, so expose the package root.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.reporting import (  # noqa: E402
    load_report_model,
    report_strategy_warnings,
    render_report_pdf,
    save_report_html,
    validate_strategy_available,
)


REPORT_DIRECTORY = PROJECT_ROOT / "outputs" / "report"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for report rendering.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default report model paths follow Step 10A output names.
    """

    parser = argparse.ArgumentParser(
        description="Render Spanish HTML and PDF financial reports from report models."
    )
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIRECTORY)
    parser.add_argument(
        "--allow-missing-strategy",
        action="store_true",
        help="Render a draft report even when accepted strategic analysis is unavailable.",
    )
    return parser


def _render_one(
    report_dir: Path,
    period_slug: str,
    *,
    allow_missing_strategy: bool,
) -> tuple[Path, Path, tuple[str, ...]]:
    """Render one report model into HTML and PDF files.

    Inputs: report directory, period slug, and strategy override flag.
    Outputs: written HTML path, PDF path, and strategy warnings.
    Assumptions: report model JSON already exists and is renderer-agnostic.
    """

    model_name = "report_model_june_2026.json" if period_slug == "june_2026" else "report_model_2026.json"
    output_stem = "financial_report_june_2026" if period_slug == "june_2026" else "financial_report_2026"
    report_model = load_report_model(report_dir / model_name)
    warnings = tuple(report_strategy_warnings(report_model))
    if warnings:
        print(f"WARNING: {period_slug} report model is missing accepted strategic analysis:")
        for warning in warnings:
            print(f"  - {warning}")
    if not allow_missing_strategy:
        # Final reports should not silently omit strategy. Use the explicit CLI
        # flag only for draft rendering when Step 9 is intentionally unavailable.
        validate_strategy_available(report_model)

    # Both renderers consume the same model so presentation stays separate from
    # business logic and no financial values are recalculated here.
    html_path = save_report_html(report_model, report_dir / f"{output_stem}.html")
    pdf_path = render_report_pdf(report_model, report_dir / f"{output_stem}.pdf")
    return html_path, pdf_path, warnings


def main() -> None:
    """Render June and annual reports from existing report model artifacts.

    Inputs: report model JSON files under outputs/report by default.
    Outputs: HTML and PDF report files under outputs/report.
    Assumptions: this stage does not modify calculations, anomalies, evidence, or analysis.
    """

    args = build_argument_parser().parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    june_html, june_pdf, june_warnings = _render_one(
        args.report_dir,
        "june_2026",
        allow_missing_strategy=args.allow_missing_strategy,
    )
    annual_html, annual_pdf, annual_warnings = _render_one(
        args.report_dir,
        "2026",
        allow_missing_strategy=args.allow_missing_strategy,
    )
    generated = [june_html, june_pdf, annual_html, annual_pdf]

    print("Finance AI Agent - Step 10B Report Renderer")
    print(f"Reports rendered: {len(generated)}")
    print(f"Strategy warnings: {len(june_warnings) + len(annual_warnings)}")
    print("Outputs saved:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
