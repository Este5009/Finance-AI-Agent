"""Run optional Step 5 Ollama structure interpretation on the Step 2 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution starts in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.llm.ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
)
from finance_agent.understanding.structure_fallback import (  # noqa: E402
    enrich_intermediate_model,
    load_intermediate_model_json,
    save_enriched_model,
)


DEFAULT_INPUT = (
    PROJECT_ROOT / "outputs" / "intermediate" / "financial_document_model.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "intermediate"
    / "financial_document_model_enriched.json"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for model paths and local Ollama configuration.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: defaults follow the project's standard local output layout.
    """

    parser = argparse.ArgumentParser(
        description="Enrich uncertain table structure with optional local Ollama."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser


def main() -> None:
    """Load, safely enrich, save, and summarize the intermediate model.

    Inputs: optional command-line paths, endpoint, model, and timeout.
    Outputs: enriched JSON artifact and console review counts.
    Assumptions: Ollama is optional; unavailability is a successful fail-safe run.
    """

    args = build_argument_parser().parse_args()
    model = load_intermediate_model_json(args.input)
    client = OllamaClient(
        endpoint=args.endpoint,
        model=args.model,
        timeout_seconds=args.timeout,
    )
    enriched, summary = enrich_intermediate_model(model, client)
    output_path = save_enriched_model(enriched, args.output)

    print("Finance AI Agent - Step 5 Ollama Structure Fallback")
    print(f"Ollama available: {'yes' if summary.ollama_available else 'no'}")
    print(f"Items reviewed: {summary.items_reviewed}")
    print(f"Uncertain columns detected: {summary.uncertain_columns}")
    print(f"Suggestions accepted: {summary.accepted}")
    print(f"Suggestions rejected: {summary.rejected}")
    print(f"Items requiring human review: {summary.requiring_human_review}")
    print(f"Enriched model saved: {output_path}")


if __name__ == "__main__":
    main()
