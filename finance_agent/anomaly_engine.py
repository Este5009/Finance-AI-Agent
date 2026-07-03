"""Orchestration and serialization for deterministic anomaly detection."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from finance_agent.anomaly_config import AnomalyThresholds
from finance_agent.anomaly_loader import CalculationOutputBundle
from finance_agent.anomaly_models import (
    Anomaly,
    AnomalyIdGenerator,
    severity_counts,
    sort_anomalies,
)
from finance_agent.anomaly_rules import detect_rule_based_anomalies
from finance_agent.anomaly_statistics import detect_statistical_anomalies
from finance_agent.anomaly_trends import detect_trend_anomalies


@dataclass(frozen=True)
class AnomalyReport:
    """Complete anomaly report for one calculation scope."""

    report_period: str
    period_slug: str
    thresholds: AnomalyThresholds
    source_files: tuple[str, ...]
    anomalies: tuple[Anomaly, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize report metadata, counts, thresholds, and anomalies.

        Inputs: this report.
        Outputs: JSON-compatible anomaly report dictionary.
        Assumptions: anomalies have already been prioritized.
        """

        method_counts = Counter(
            anomaly.detection_method for anomaly in self.anomalies
        )
        return {
            "report_period": self.report_period,
            "period_slug": self.period_slug,
            "thresholds": self.thresholds.to_dict(),
            "source_files": list(self.source_files),
            "total_anomalies": len(self.anomalies),
            "anomalies_by_severity": severity_counts(list(self.anomalies)),
            "anomalies_by_detection_method": dict(sorted(method_counts.items())),
            "anomalies": [anomaly.to_dict() for anomaly in self.anomalies],
        }


def _bundle_source_files(bundle: CalculationOutputBundle) -> tuple[str, ...]:
    """Collect the calculation artifacts used by one anomaly report.

    Inputs: loaded calculation bundle.
    Outputs: ordered source filename tuple.
    Assumptions: absolute paths are unnecessary inside portable report metadata.
    """

    paths = [
        bundle.finance_summary_path,
        bundle.kpi_summary_path,
        bundle.department_summary_path,
        bundle.category_summary_path,
        bundle.monthly_trends_path,
    ]
    return tuple(Path(path).name for path in paths if path)


def run_anomaly_detection(
    bundle: CalculationOutputBundle,
    *,
    thresholds: AnomalyThresholds | None = None,
    include_trends: bool = False,
    include_statistics: bool = False,
    anomaly_id_prefix: str,
) -> AnomalyReport:
    """Run configured rule, trend, and statistical detection.

    Inputs: calculation bundle, optional thresholds, detector switches, and ID prefix.
    Outputs: prioritized immutable AnomalyReport.
    Assumptions: trend/statistical detection requires monthly_trends in the bundle.
    """

    active_thresholds = thresholds or AnomalyThresholds()
    generator = AnomalyIdGenerator(anomaly_id_prefix)
    anomalies = detect_rule_based_anomalies(
        bundle,
        active_thresholds,
        generator,
    )
    if include_trends:
        anomalies.extend(
            detect_trend_anomalies(
                bundle,
                active_thresholds,
                generator,
            )
        )
    if include_statistics:
        anomalies.extend(
            detect_statistical_anomalies(
                bundle,
                active_thresholds,
                generator,
            )
        )
    return AnomalyReport(
        report_period=bundle.report_period,
        period_slug=bundle.period_slug,
        thresholds=active_thresholds,
        source_files=_bundle_source_files(bundle),
        anomalies=tuple(sort_anomalies(anomalies)),
    )


def save_anomaly_report(
    report: AnomalyReport,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save one anomaly report as JSON and CSV.

    Inputs: anomaly report and destination directory.
    Outputs: generated JSON and CSV paths.
    Assumptions: report period_slug is safe for filenames.
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"anomaly_report_{report.period_slug}.json"
    csv_path = output_dir / f"anomaly_report_{report.period_slug}.csv"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    columns = list(Anomaly.__dataclass_fields__)
    dataframe = pd.DataFrame(
        [anomaly.to_dict() for anomaly in report.anomalies],
        columns=columns,
    )
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    return {"json": json_path, "csv": csv_path}


def build_risk_summary(report: AnomalyReport) -> dict[str, Any]:
    """Build a compact annual risk prioritization summary.

    Inputs: annual anomaly report.
    Outputs: severity counts, top risks, rule coverage, and source metadata.
    Assumptions: report anomalies are already ordered by severity.
    """

    anomalies = list(report.anomalies)
    high_priority = [
        anomaly
        for anomaly in anomalies
        if anomaly.severity in {"critical", "high"}
    ]
    return {
        "report_period": report.report_period,
        "total_anomalies": len(anomalies),
        "anomalies_by_severity": severity_counts(anomalies),
        "high_priority_count": len(high_priority),
        "thresholds": report.thresholds.to_dict(),
        "source_files": list(report.source_files),
        "top_risks": [
            {
                "anomaly_id": anomaly.anomaly_id,
                "title": anomaly.title,
                "severity": anomaly.severity,
                "period": anomaly.period,
                "metric": anomaly.metric,
                "recommended_next_check": anomaly.recommended_next_check,
            }
            for anomaly in anomalies[:10]
        ],
        "detection_scope": {
            "rule_based": True,
            "trend_based": True,
            "statistical_z_score": True,
            "llm_used": False,
        },
    }


def save_risk_summary(
    report: AnomalyReport,
    output_path: str | Path,
) -> Path:
    """Save the annual risk summary JSON.

    Inputs: annual anomaly report and destination path.
    Outputs: written Path.
    Assumptions: parent directory may not yet exist.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_risk_summary(report),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path
