"""Generate deterministic June and annual investigation plans from prior outputs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution begins in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.investigation_planner import (  # noqa: E402
    build_investigation_plan,
    build_planner_summary,
    save_investigation_plan,
    save_planner_summary,
)
from finance_agent.planner_loader import load_planner_inputs  # noqa: E402
from finance_agent.planner_models import (  # noqa: E402
    InvestigationPlan,
    PriorityLevel,
)


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "plans"


def _print_plan_summary(plan_name: str, plan: InvestigationPlan) -> None:
    """Print task counts and top questions for one generated plan.

    Inputs: readable plan name and InvestigationPlan-compatible object.
    Outputs: console summary.
    Assumptions: plan exposes tasks with priority and question fields.
    """

    print(f"\n{plan_name}")
    print(f"Total tasks: {len(plan.tasks)}")
    for priority in PriorityLevel:
        count = sum(task.priority == priority for task in plan.tasks)
        print(f"  {priority.value.title()}: {count}")
    print("Top 5 investigation questions:")
    for task in plan.tasks[:5]:
        print(f"  - [{task.priority.value}] {task.question_to_answer}")


def main() -> None:
    """Load processed outputs, generate plans, save them, and print summaries.

    Inputs: standard Step 2-5 artifacts under the project outputs directory.
    Outputs: two plan JSON files, aggregate planner summary, and console counts.
    Assumptions: evidence requests are definitions only and no tools are executed.
    """

    inputs = load_planner_inputs(PROJECT_ROOT)
    annual_anomalies = inputs.anomaly_report_annual.get("anomalies", [])

    june_plan = build_investigation_plan(
        finance_document=inputs.finance_summary_june,
        anomaly_report=inputs.anomaly_report_june,
        monthly_trends=inputs.monthly_trends,
        recurrence_anomalies=annual_anomalies,
        enriched_model=inputs.enriched_model,
        risk_summary=inputs.risk_summary_annual,
        period_slug="june_2026",
        source_files=inputs.source_files,
    )
    annual_plan = build_investigation_plan(
        finance_document=inputs.finance_summary_annual,
        anomaly_report=inputs.anomaly_report_annual,
        monthly_trends=inputs.monthly_trends,
        recurrence_anomalies=annual_anomalies,
        enriched_model=inputs.enriched_model,
        risk_summary=inputs.risk_summary_annual,
        period_slug="2026",
        source_files=inputs.source_files,
    )

    june_path = save_investigation_plan(
        june_plan,
        OUTPUT_DIRECTORY / "investigation_plan_june_2026.json",
    )
    annual_path = save_investigation_plan(
        annual_plan,
        OUTPUT_DIRECTORY / "investigation_plan_2026.json",
    )
    summary = build_planner_summary(
        [june_plan, annual_plan],
        [june_path.name, annual_path.name],
    )
    summary_path = save_planner_summary(
        summary,
        OUTPUT_DIRECTORY / "planner_summary_2026.json",
    )

    print("Finance AI Agent - Step 6 Investigation Planner")
    _print_plan_summary("June 2026 plan", june_plan)
    _print_plan_summary("Annual 2026 plan", annual_plan)
    print("\nOutputs saved:")
    print(f"  - {june_path}")
    print(f"  - {annual_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
