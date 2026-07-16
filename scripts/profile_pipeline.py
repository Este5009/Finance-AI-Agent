"""Run an uncached generic pipeline profile and save timing telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.llm.ollama_client import DEFAULT_OLLAMA_ENDPOINT  # noqa: E402
from finance_agent.orchestration import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    PipelineConfig,
    build_pipeline_input_model,
    run_pipeline_for_report,
)
from finance_agent.orchestration.profiling import build_pipeline_profile  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for pipeline profiling.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: defaults profile the current synthetic monthly report.
    """

    parser = argparse.ArgumentParser(description="Profile one uncached pipeline run.")
    synthetic = PROJECT_ROOT / "data" / "synthetic"
    parser.add_argument(
        "--report",
        type=Path,
        default=synthetic / "monthly_financial_report_june_2026.xlsx",
    )
    parser.add_argument(
        "--goals",
        type=Path,
        default=synthetic / "financial_goals_2026.pdf",
    )
    parser.add_argument("--period-override", default="2026-06")
    parser.add_argument("--language", default="es")
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout", type=float, default=180.0)
    parser.add_argument("--stage-timeout", type=float, default=420.0)
    parser.add_argument("--max-planner-anomalies", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "profiling" / "pipeline_profile.json",
    )
    return parser


def save_profile(profile: dict[str, object], output_path: Path) -> Path:
    """Save one pipeline profile document.

    Inputs: JSON-compatible profile and destination path.
    Outputs: resolved written path.
    Assumptions: parent folders may be created.
    """

    path = output_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Run the uncached pipeline and persist profiling telemetry.

    Inputs: CLI arguments.
    Outputs: outputs/profiling/pipeline_profile.json by default.
    Assumptions: profiling should bypass cache to measure first-run latency.
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
        ollama_timeout_seconds=args.ollama_timeout,
        stage_timeout_seconds=args.stage_timeout,
        input_model=input_model,
        enable_cache=False,
        max_planner_anomalies=args.max_planner_anomalies,
        compact_context=True,
        deduplicate_context=True,
    )
    result = run_pipeline_for_report(input_model, config)
    profile = build_pipeline_profile(result)
    output_path = save_profile(profile, args.output)
    print(f"Pipeline success: {'yes' if result.success else 'no'}")
    print(f"Total runtime: {result.runtime_summary.total_runtime_seconds:.2f}s")
    print(f"Total context characters: {profile['total_context_characters']}")
    print("Bottlenecks:")
    for item in profile["bottleneck_ranking"][:5]:
        print(f"  {item['stage_name']}: {item['runtime_seconds']:.2f}s")
    print(f"Profile saved: {output_path}")
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
