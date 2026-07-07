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
from finance_agent.analysis.strategic_analysis import (  # noqa: E402
    StrategicAnalysisResult,
    build_analysis_summary,
    create_strategic_analysis,
    load_json_artifact,
    save_json_artifact,
)


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
    parser.add_argument("--timeout", type=float, default=180.0)
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
        timeout_seconds=args.timeout,
    )

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

    june_result = create_strategic_analysis(
        client=client,
        evidence_package=evidence_june,
        finance_summary=finance_june,
        anomaly_report=anomaly_june,
        risk_summary=risk_summary,
        period_slug="june_2026",
    )
    annual_result = create_strategic_analysis(
        client=client,
        evidence_package=evidence_annual,
        finance_summary=finance_annual,
        anomaly_report=anomaly_annual,
        risk_summary=risk_summary,
        period_slug="2026",
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
