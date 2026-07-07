"""Run Step 8 retrieval over validated investigation execution queues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution starts in scripts/, so expose the package root.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.retrieval_engine import (  # noqa: E402
    build_retrieval_summary,
    execute_retrieval_queue,
    load_execution_queue,
    load_retrieval_context,
    save_json_artifact,
)
from finance_agent.retrieval_registry import create_default_registry  # noqa: E402


PLAN_DIRECTORY = PROJECT_ROOT / "outputs" / "plans"
EVIDENCE_DIRECTORY = PROJECT_ROOT / "outputs" / "evidence"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for Step 8 retrieval execution.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default paths follow Step 7 output naming.
    """

    parser = argparse.ArgumentParser(
        description="Execute validated retrieval queues and build evidence packages."
    )
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=PLAN_DIRECTORY,
        help="Directory containing Step 7 execution queue JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVIDENCE_DIRECTORY,
        help="Directory where evidence packages will be written.",
    )
    return parser


def _print_package_summary(label: str, package: dict[str, object]) -> None:
    """Print counters for one completed evidence package.

    Inputs: display label and package document.
    Outputs: console summary.
    Assumptions: package summary follows retrieval_engine schema.
    """

    summary = package["summary"]
    assert isinstance(summary, dict)
    print(f"\n{label}")
    print(f"  Tasks executed: {summary['tasks_executed']}")
    print(f"  Successful retrievals: {summary['successful_retrievals']}")
    print(f"  Failed retrievals: {summary['failed_retrievals']}")
    print(f"  Unavailable evidence: {summary['unavailable_evidence']}")


def main() -> None:
    """Execute June and annual retrieval queues and save evidence artifacts.

    Inputs: Step 7 execution queues and processed local outputs.
    Outputs: two evidence packages and one retrieval summary JSON file.
    Assumptions: no strategic analysis, database access, or LLM calls occur.
    """

    args = build_argument_parser().parse_args()
    context = load_retrieval_context(PROJECT_ROOT)
    registry = create_default_registry()

    june_queue = load_execution_queue(args.plans_dir / "execution_queue_june_2026.json")
    annual_queue = load_execution_queue(args.plans_dir / "execution_queue_2026.json")

    june_package = execute_retrieval_queue(june_queue, context, registry)
    annual_package = execute_retrieval_queue(annual_queue, context, registry)
    retrieval_summary = build_retrieval_summary((june_package, annual_package))

    paths = [
        save_json_artifact(
            june_package,
            args.output_dir / "evidence_package_june_2026.json",
        ),
        save_json_artifact(
            annual_package,
            args.output_dir / "evidence_package_2026.json",
        ),
        save_json_artifact(
            retrieval_summary,
            args.output_dir / "retrieval_summary_2026.json",
        ),
    ]

    print("Finance AI Agent - Step 8 Retrieval Layer")
    print(f"Tasks executed: {retrieval_summary['tasks_executed']}")
    print(f"Successful retrievals: {retrieval_summary['successful_retrievals']}")
    print(f"Failed retrievals: {retrieval_summary['failed_retrievals']}")
    print(f"Unavailable evidence: {retrieval_summary['unavailable_evidence']}")
    _print_package_summary("June 2026", june_package)
    _print_package_summary("Annual 2026", annual_package)
    print("\nOutputs saved:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
