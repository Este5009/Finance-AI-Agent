"""Run primary Ollama investigation planning with deterministic fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution begins in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.investigation_planner import (  # noqa: E402
    build_investigation_plan,
)
from finance_agent.ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
)
from finance_agent.ollama_planner import (  # noqa: E402
    OllamaPlannerResult,
    create_ollama_investigation_plan,
    save_json_artifact,
)
from finance_agent.planner_loader import (  # noqa: E402
    PlannerInputBundle,
    load_planner_inputs,
)
from finance_agent.planner_models import InvestigationPlan  # noqa: E402


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "plans"


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for local Ollama configuration.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default endpoint/model follow project guidance.
    """

    parser = argparse.ArgumentParser(
        description="Generate validated Ollama investigation plans and queues."
    )
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def _build_baseline_plans(
    inputs: PlannerInputBundle,
) -> tuple[InvestigationPlan, InvestigationPlan]:
    """Build trusted Step 6 baseline plans for validation and fallback.

    Inputs: loaded processed planner artifacts.
    Outputs: June and annual deterministic plans.
    Assumptions: baseline planning performs no LLM or retrieval tool calls.
    """

    annual_anomalies = inputs.anomaly_report_annual.get("anomalies", [])
    june = build_investigation_plan(
        finance_document=inputs.finance_summary_june,
        anomaly_report=inputs.anomaly_report_june,
        monthly_trends=inputs.monthly_trends,
        recurrence_anomalies=annual_anomalies,
        enriched_model=inputs.enriched_model,
        risk_summary=inputs.risk_summary_annual,
        period_slug="june_2026",
        source_files=inputs.source_files,
    )
    annual = build_investigation_plan(
        finance_document=inputs.finance_summary_annual,
        anomaly_report=inputs.anomaly_report_annual,
        monthly_trends=inputs.monthly_trends,
        recurrence_anomalies=annual_anomalies,
        enriched_model=inputs.enriched_model,
        risk_summary=inputs.risk_summary_annual,
        period_slug="2026",
        source_files=inputs.source_files,
    )
    return june, annual


def _print_result(label: str, result: OllamaPlannerResult) -> None:
    """Print acceptance and fallback status for one scope.

    Inputs: scope label and completed planner result.
    Outputs: concise console status.
    Assumptions: plan document includes auditable validation metadata.
    """

    plan = result.plan_document
    print(f"\n{label}")
    print(f"  Ollama available: {'yes' if plan['ollama_available'] else 'no'}")
    print(f"  Validation status: {plan['validation_status']}")
    print(f"  Plan accepted: {'yes' if result.ollama_plan_accepted else 'no'}")
    print(f"  Deterministic fallback used: {'yes' if result.fallback_used else 'no'}")
    print(f"  Investigation steps: {plan['total_steps']}")
    print(f"  Safely deduplicated calls: {plan['deduplicated_tool_calls']}")
    for error in result.validation_errors[:3]:
        print(f"  Validation error: {error}")


def main() -> None:
    """Generate, validate, queue, and save June and annual plans.

    Inputs: standard processed outputs and optional Ollama CLI configuration.
    Outputs: two auditable plans and two pending execution queues.
    Assumptions: queue generation never executes the described interfaces.
    """

    args = build_argument_parser().parse_args()
    inputs = load_planner_inputs(PROJECT_ROOT)
    june_baseline, annual_baseline = _build_baseline_plans(inputs)
    client = OllamaClient(
        endpoint=args.endpoint,
        model=args.model,
        timeout_seconds=args.timeout,
    )

    june_result = create_ollama_investigation_plan(
        client=client,
        finance_document=inputs.finance_summary_june,
        anomaly_report=inputs.anomaly_report_june,
        risk_summary=inputs.risk_summary_annual,
        enriched_model=inputs.enriched_model,
        baseline_plan=june_baseline,
        period_slug="june_2026",
    )
    annual_result = create_ollama_investigation_plan(
        client=client,
        finance_document=inputs.finance_summary_annual,
        anomaly_report=inputs.anomaly_report_annual,
        risk_summary=inputs.risk_summary_annual,
        enriched_model=inputs.enriched_model,
        baseline_plan=annual_baseline,
        period_slug="2026",
    )

    paths = [
        save_json_artifact(
            june_result.plan_document,
            OUTPUT_DIRECTORY / "ollama_plan_june_2026.json",
        ),
        save_json_artifact(
            annual_result.plan_document,
            OUTPUT_DIRECTORY / "ollama_plan_2026.json",
        ),
        save_json_artifact(
            june_result.execution_queue,
            OUTPUT_DIRECTORY / "execution_queue_june_2026.json",
        ),
        save_json_artifact(
            annual_result.execution_queue,
            OUTPUT_DIRECTORY / "execution_queue_2026.json",
        ),
    ]

    results = [june_result, annual_result]
    print("Finance AI Agent - Step 7 Ollama Investigation Planner")
    print(
        "Accepted Ollama plans: "
        f"{sum(result.ollama_plan_accepted for result in results)}"
    )
    print(
        "Rejected/unavailable Ollama plans: "
        f"{sum(not result.ollama_plan_accepted for result in results)}"
    )
    _print_result("June 2026", june_result)
    _print_result("Annual 2026", annual_result)
    print("\nOutputs saved:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
