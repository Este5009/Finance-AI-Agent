"""Run Step 9 Ollama strategic financial analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution begins in scripts/, so expose the package root.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.llm.ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
)
from finance_agent.analysis.analysis_models import StrategicAnalysisResult  # noqa: E402
from finance_agent.analysis.strategic_analysis import (  # noqa: E402
    build_analysis_summary,
    load_json_artifact,
    save_json_artifact,
)
from finance_agent.reasoning.reasoning_pipeline import create_modular_strategic_analysis  # noqa: E402


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "analysis"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for local Ollama strategic analysis.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default endpoint/model follow project guidance.
    """

    parser = argparse.ArgumentParser(
        description="Generate validated Ollama strategic financial analysis."
    )
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Backward-compatible alias for Ollama read/inference timeout.",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=None)
    parser.add_argument("--stage-timeout", type=float, default=900.0)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--warm-up", action="store_true")
    return parser


def _print_result(label: str, result: StrategicAnalysisResult) -> None:
    """Print validation and recommendation status for one analysis.

    Inputs: display label and strategic-analysis result.
    Outputs: concise console status.
    Assumptions: confidence is available only for accepted analyses.
    """

    document = result.analysis_document
    analysis = document["analysis"]
    confidence = analysis.get("confidence") if isinstance(analysis, dict) else None
    print(f"\n{label}")
    print(f"  Ollama available: {'yes' if document['ollama_available'] else 'no'}")
    print(f"  Validation status: {document['validation_status']}")
    print(f"  Analysis generated: {'yes' if result.accepted else 'no'}")
    print(f"  Confidence score: {confidence}")
    print(f"  Recommendations generated: {document['recommendation_count']}")
    for error in result.validation_errors[:3]:
        print(f"  Validation error: {error}")


def _save_reasoning_artifacts(result: StrategicAnalysisResult, period_slug: str) -> list[Path]:
    """Save modular reasoning stage outputs for one analysis.

    Inputs: strategic-analysis result and period slug.
    Outputs: paths written for financial, historical, strategic, and state dumps.
    Assumptions: rejected runs still write auditable empty/partial artifacts.
    """

    state = result.analysis_document.get("reasoning_state", {})
    state = state if isinstance(state, dict) else {}
    outputs = state.get("reasoning_outputs", {})
    outputs = outputs if isinstance(outputs, dict) else {}
    paths = [
        save_json_artifact(
            {
                "period_slug": period_slug,
                "stage_id": "financial_performance",
                "reasoning_output": outputs.get("financial_performance", {}),
            },
            OUTPUT_DIRECTORY / f"financial_reasoning_{period_slug}.json",
        ),
        save_json_artifact(
            {
                "period_slug": period_slug,
                "stage_id": "historical_operational",
                "reasoning_output": outputs.get("historical_operational", {}),
            },
            OUTPUT_DIRECTORY / f"historical_reasoning_{period_slug}.json",
        ),
        save_json_artifact(
            {
                "period_slug": period_slug,
                "stage_id": "strategic_synthesis",
                "reasoning_output": outputs.get("strategic_synthesis", {}),
            },
            OUTPUT_DIRECTORY / f"strategic_reasoning_{period_slug}.json",
        ),
        save_json_artifact(
            state,
            OUTPUT_DIRECTORY / f"reasoning_state_{period_slug}.json",
        ),
    ]
    return paths


def main() -> None:
    """Generate June and annual strategic analysis artifacts.

    Inputs: processed evidence packages, finance summaries, anomalies, and risks.
    Outputs: two strategic analysis JSON files plus a cross-scope summary.
    Assumptions: no raw reports, database, PDF, UI, or email actions are touched.
    """

    args = build_argument_parser().parse_args()
    client = OllamaClient(
        endpoint=args.endpoint,
        model=args.model,
        timeout_seconds=args.read_timeout if args.read_timeout is not None else args.timeout,
        connect_timeout_seconds=args.connect_timeout,
        read_timeout_seconds=args.read_timeout if args.read_timeout is not None else args.timeout,
        keep_alive=args.keep_alive,
    )
    if args.warm_up:
        print("Warming Ollama model...")
        client.warm_up()

    evidence_june = load_json_artifact(
        PROJECT_ROOT / "outputs" / "evidence" / "evidence_package_june_2026.json"
    )
    evidence_annual = load_json_artifact(
        PROJECT_ROOT / "outputs" / "evidence" / "evidence_package_2026.json"
    )
    finance_june = load_json_artifact(
        PROJECT_ROOT / "outputs" / "calculations" / "finance_summary_june_2026.json"
    )
    finance_annual = load_json_artifact(
        PROJECT_ROOT / "outputs" / "calculations" / "finance_summary_2026.json"
    )
    anomaly_june = load_json_artifact(
        PROJECT_ROOT / "outputs" / "anomalies" / "anomaly_report_june_2026.json"
    )
    anomaly_annual = load_json_artifact(
        PROJECT_ROOT / "outputs" / "anomalies" / "anomaly_report_2026.json"
    )
    risk_summary = load_json_artifact(
        PROJECT_ROOT / "outputs" / "anomalies" / "risk_summary_2026.json"
    )

    june_result = create_modular_strategic_analysis(
        client=client,
        evidence_package=evidence_june,
        finance_summary=finance_june,
        anomaly_report=anomaly_june,
        risk_summary=risk_summary,
        period_slug="june_2026",
        stage_timeout_seconds=args.stage_timeout,
    )
    annual_result = create_modular_strategic_analysis(
        client=client,
        evidence_package=evidence_annual,
        finance_summary=finance_annual,
        anomaly_report=anomaly_annual,
        risk_summary=risk_summary,
        period_slug="2026",
        stage_timeout_seconds=args.stage_timeout,
    )
    summary = build_analysis_summary(
        (june_result.analysis_document, annual_result.analysis_document)
    )

    paths = [
        save_json_artifact(
            june_result.analysis_document,
            OUTPUT_DIRECTORY / "strategic_analysis_june_2026.json",
        ),
        save_json_artifact(
            annual_result.analysis_document,
            OUTPUT_DIRECTORY / "strategic_analysis_2026.json",
        ),
        save_json_artifact(
            summary,
            OUTPUT_DIRECTORY / "analysis_summary_2026.json",
        ),
    ]
    paths.extend(_save_reasoning_artifacts(june_result, "june_2026"))
    paths.extend(_save_reasoning_artifacts(annual_result, "2026"))

    results = [june_result, annual_result]
    print("Finance AI Agent - Step 9 Ollama Strategic Analysis")
    print(
        "Ollama availability: "
        f"{'yes' if any(result.analysis_document['ollama_available'] for result in results) else 'no'}"
    )
    print(f"Analyses generated: {summary['analyses_generated']}")
    print(f"Analyses rejected/unavailable: {summary['analyses_rejected']}")
    _print_result("June 2026", june_result)
    _print_result("Annual 2026", annual_result)
    print("\nOutputs saved:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
