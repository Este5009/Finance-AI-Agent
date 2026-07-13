"""Run the pipeline from the generic one-report input workflow."""

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
)
from finance_agent.orchestration import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    PipelineConfig,
    build_pipeline_input_model,
    run_pipeline_for_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for one-report pipeline input.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: report language defaults to Spanish for user-facing outputs.
    """

    parser = argparse.ArgumentParser(
        description="Run Finance AI Agent from one report and one goals document."
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--goals", required=True, type=Path)
    parser.add_argument("--period-override", default=None)
    parser.add_argument("--language", default="es")
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Single Ollama model used by every LLM stage.",
    )
    parser.add_argument(
        "--structure-model",
        default=None,
        help="Experimental override for structure fallback only.",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Experimental override for investigation planner only.",
    )
    parser.add_argument(
        "--analysis-model",
        default=None,
        help="Experimental override for strategic analysis only.",
    )
    parser.add_argument("--ollama-timeout", type=float, default=180.0)
    parser.add_argument("--stage-timeout", type=float, default=420.0)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pipeline" / "pipeline_for_report_summary.json",
    )
    return parser


def _save_summary(data: dict[str, object], output_path: Path) -> Path:
    """Save the generic pipeline run summary.

    Inputs: JSON-compatible summary data and output path.
    Outputs: resolved written path.
    Assumptions: parent directories may be created.
    """

    path = output_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Run the compatibility pipeline from generic one-report input.

    Inputs: CLI report/goals paths, optional period override, and language.
    Outputs: existing pipeline artifacts plus generic input summary metadata.
    Assumptions: current full execution supports the existing synthetic demo files.
    """

    args = build_argument_parser().parse_args()
    input_model = build_pipeline_input_model(
        financial_report_path=args.report,
        goals_document_path=args.goals,
        period_override=args.period_override,
        report_language=args.language,
    )
    config = PipelineConfig.from_project_root(
        PROJECT_ROOT,
        python_executable=sys.executable,
        ollama_endpoint=args.endpoint,
        ollama_model=args.model,
        structure_ollama_model=args.structure_model,
        planner_ollama_model=args.planner_model,
        analysis_ollama_model=args.analysis_model,
        ollama_timeout_seconds=args.ollama_timeout,
        stage_timeout_seconds=args.stage_timeout,
        input_model=input_model,
    )

    print("Finance AI Agent - Generic Report Pipeline")
    print(f"Financial report: {input_model.financial_report_path}")
    print(f"Goals document: {input_model.goals_document_path}")
    print(f"Detected period: {input_model.detected_period.label}")
    print(f"Detected period type: {input_model.detected_period.period_type}")
    print(f"Detection confidence: {input_model.detected_period.confidence:.2f}")
    print(f"Effective period: {input_model.effective_period_label}")
    print(f"Report language: {input_model.report_language}")
    print(f"Ollama models: {config.effective_ollama_models()}")
    for evidence in input_model.detected_period.evidence:
        print(f"  evidence: {evidence}")

    try:
        result = run_pipeline_for_report(input_model, config)
    except NotImplementedError as exc:
        print(f"Pipeline execution blocked: {exc}")
        raise SystemExit(2) from exc
    except ValueError as exc:
        print(f"Pipeline input invalid: {exc}")
        raise SystemExit(2) from exc

    summary_path = _save_summary(result.to_dict(), args.summary_output)
    print(f"Pipeline success: {'yes' if result.success else 'no'}")
    print(f"Stages run: {result.runtime_summary.stages_run}")
    print(f"Stages succeeded: {result.runtime_summary.stages_succeeded}")
    print(f"Stages failed: {result.runtime_summary.stages_failed}")
    print(f"Generated outputs: {len(result.output_files)}")
    print(f"Summary saved: {summary_path}")
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
