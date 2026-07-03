"""Run deterministic Step 4 anomaly detection for June and annual 2026."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct script execution starts in scripts/, so expose the project package.
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.anomaly_config import AnomalyThresholds  # noqa: E402
from finance_agent.anomaly_engine import (  # noqa: E402
    AnomalyReport,
    run_anomaly_detection,
    save_anomaly_report,
    save_risk_summary,
)
from finance_agent.anomaly_loader import load_calculation_outputs  # noqa: E402


CALCULATIONS_DIRECTORY = PROJECT_ROOT / "outputs" / "calculations"
ANOMALY_DIRECTORY = PROJECT_ROOT / "outputs" / "anomalies"


def _print_report_summary(report: AnomalyReport) -> None:
    """Print anomaly totals by severity for one report.

    Inputs: completed anomaly report.
    Outputs: concise console summary.
    Assumptions: reports contain standard severity labels.
    """

    counts = report.to_dict()["anomalies_by_severity"]
    print(f"\nReporting period: {report.report_period}")
    print(f"Total anomalies: {len(report.anomalies)}")
    for severity in ("critical", "high", "medium", "low"):
        print(f"  {severity.title()}: {counts[severity]}")


def main() -> None:
    """Load calculations, detect anomalies, and save all Step 4 outputs.

    Inputs: existing June and annual calculation artifacts.
    Outputs: two JSON/CSV reports, annual risk summary, and console counts.
    Assumptions: only the annual calculation bundle contains monthly trends.
    """

    print("Finance AI Agent - Step 4 Deterministic Anomaly Detection")
    thresholds = AnomalyThresholds()
    monthly_bundle = load_calculation_outputs(
        CALCULATIONS_DIRECTORY,
        "june_2026",
    )
    annual_bundle = load_calculation_outputs(
        CALCULATIONS_DIRECTORY,
        "2026",
        include_monthly_trends=True,
    )

    monthly_report = run_anomaly_detection(
        monthly_bundle,
        thresholds=thresholds,
        anomaly_id_prefix="ANOM-JUNE-2026",
    )
    annual_report = run_anomaly_detection(
        annual_bundle,
        thresholds=thresholds,
        include_trends=True,
        include_statistics=True,
        anomaly_id_prefix="ANOM-2026",
    )

    monthly_paths = save_anomaly_report(monthly_report, ANOMALY_DIRECTORY)
    annual_paths = save_anomaly_report(annual_report, ANOMALY_DIRECTORY)
    risk_path = save_risk_summary(
        annual_report,
        ANOMALY_DIRECTORY / "risk_summary_2026.json",
    )

    _print_report_summary(monthly_report)
    _print_report_summary(annual_report)
    print("\nAnomaly outputs saved:")
    for path in [
        *monthly_paths.values(),
        *annual_paths.values(),
        risk_path,
    ]:
        print(f"  - {path.resolve()}")


if __name__ == "__main__":
    main()
