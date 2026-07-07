"""Run the full existing Finance AI Agent pipeline in order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution starts in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.llm.ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
)
from finance_agent.orchestration import (  # noqa: E402
    PipelineConfig,
    run_full_pipeline,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for the full pipeline orchestrator.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default paths preserve the existing repository layout.
    """

    parser = argparse.ArgumentParser(
        description="Run the existing Finance AI Agent pipeline end to end."
    )
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout", type=float, default=180.0)
    parser.add_argument("--stage-timeout", type=float, default=420.0)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pipeline" / "pipeline_run_summary.json",
        help="Where to write the structured pipeline run summary.",
    )
    return parser


def _save_summary(result: dict[str, object], output_path: Path) -> Path:
    """Save the structured pipeline summary.

    Inputs: JSON-compatible pipeline result and destination path.
    Outputs: resolved output path.
    Assumptions: parent directories may be created.
    """

    path = output_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Run the full pipeline and print a concise status summary.

    Inputs: CLI configuration for Ollama and stage timeouts.
    Outputs: existing stage artifacts plus a pipeline summary JSON.
    Assumptions: stage scripts retain their existing output paths.
    """

    args = build_argument_parser().parse_args()
    config = PipelineConfig.from_project_root(
        PROJECT_ROOT,
        python_executable=sys.executable,
        ollama_endpoint=args.endpoint,
        ollama_model=args.model,
        ollama_timeout_seconds=args.ollama_timeout,
        stage_timeout_seconds=args.stage_timeout,
    )
    result = run_full_pipeline(config)
    summary_path = _save_summary(result.to_dict(), args.summary_output)

    print("Finance AI Agent - Full Pipeline Orchestrator")
    print(f"Pipeline success: {'yes' if result.success else 'no'}")
    print(f"Stages requested: {result.runtime_summary.stages_requested}")
    print(f"Stages run: {result.runtime_summary.stages_run}")
    print(f"Stages succeeded: {result.runtime_summary.stages_succeeded}")
    print(f"Stages failed: {result.runtime_summary.stages_failed}")
    print(f"Stages skipped: {result.runtime_summary.stages_skipped}")
    print(f"Outputs generated: {len(result.output_files)}")
    print(f"Warnings: {len(result.warnings)}")
    print("\nStage results:")
    for stage in result.stages:
        status = "skipped" if stage.skipped else ("ok" if stage.success else "failed")
        print(f"  - {stage.stage_name}: {status} ({stage.runtime_seconds:.2f}s)")
        for warning in stage.warnings[:2]:
            print(f"    warning: {warning}")
        if stage.error:
            print(f"    error: {stage.error}")
    print(f"\nPipeline summary saved: {summary_path}")

    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
