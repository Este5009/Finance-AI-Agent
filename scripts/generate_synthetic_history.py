"""CLI for generating deterministic synthetic university financial histories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from finance_agent.synthetic_history import SyntheticHistoryConfig, generate_synthetic_history, validate_generated_history


def build_parser() -> argparse.ArgumentParser:
    """Build the synthetic history CLI argument parser.

    Inputs:
        None.
    Outputs:
        Configured ``argparse.ArgumentParser``.
    Assumptions:
        The default scenario is the Phase 12A recovery-year dataset.
    """

    parser = argparse.ArgumentParser(description="Generate a synthetic university financial history.")
    parser.add_argument("--year", type=int, default=2026, help="Year to generate.")
    parser.add_argument("--scenario", default="recovery", help="Scenario name. Phase 12A supports 'recovery'.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic_history"), help="Base output directory.")
    parser.add_argument("--department", action="append", dest="departments", help="Department to include. May be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing scenario output folder.")
    return parser


def main() -> int:
    """Run the synthetic history generator from the command line.

    Inputs:
        Command-line arguments.
    Outputs:
        Process exit code.
    Assumptions:
        Validation runs immediately after generation and reports reconciliation status.
    """

    args = build_parser().parse_args()
    config = SyntheticHistoryConfig(
        year=args.year,
        scenario=args.scenario,
        seed=args.seed,
        departments=tuple(args.departments) if args.departments else SyntheticHistoryConfig().departments,
        output_directory=args.output_dir,
        overwrite=args.overwrite,
    )
    generated = generate_synthetic_history(config)
    validation = validate_generated_history(generated.root_directory)
    print(f"Generated scenario: {generated.root_directory}")
    print(f"Reports: {len(generated.report_paths)}")
    print(f"Goals PDFs: {len(generated.goals_paths)}")
    print(f"Manifest: {generated.manifest_path}")
    print(f"Validation: {'passed' if validation.is_valid else 'failed'}")
    if validation.errors:
        print("Errors:")
        for error in validation.errors:
            print(f"- {error}")
    return 0 if validation.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
