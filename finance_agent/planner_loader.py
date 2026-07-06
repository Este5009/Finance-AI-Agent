"""Validated loading of prior pipeline outputs for investigation planning."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PlannerInputError(RuntimeError):
    """Raised when a required prior-stage output cannot be loaded."""


@dataclass(frozen=True)
class PlannerInputBundle:
    """All processed artifacts available to the deterministic planner."""

    finance_summary_june: dict[str, Any]
    finance_summary_annual: dict[str, Any]
    monthly_trends: tuple[dict[str, Any], ...]
    anomaly_report_june: dict[str, Any]
    anomaly_report_annual: dict[str, Any]
    risk_summary_annual: dict[str, Any]
    enriched_model: dict[str, Any]
    source_files: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON output as an object.

    Inputs: path to a prior-stage JSON artifact.
    Outputs: parsed dictionary.
    Assumptions: planner inputs must have JSON objects at their roots.
    """

    if not path.is_file():
        raise PlannerInputError(f"Required planner input does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerInputError(f"Could not read planner input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlannerInputError(f"Planner JSON root must be an object: {path}")
    return value


def _read_csv_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a required CSV output without numeric interpretation.

    Inputs: path to a processed CSV artifact.
    Outputs: ordered row dictionaries.
    Assumptions: planner scoring converts only the fields it needs.
    """

    if not path.is_file():
        raise PlannerInputError(f"Required planner input does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise PlannerInputError(f"Could not read planner input {path}: {exc}") from exc


def load_planner_inputs(project_root: str | Path) -> PlannerInputBundle:
    """Load only Step 2-5 outputs required by the Step 6/7 planners.

    Inputs: project root containing outputs/calculations, anomalies, and intermediate.
    Outputs: validated bundle of parsed processed artifacts.
    Assumptions: this loader never opens raw Excel, PDF, or normalized table CSV files.
    """

    root = Path(project_root).resolve()
    paths = {
        "finance_summary_june": (
            root / "outputs" / "calculations" / "finance_summary_june_2026.json"
        ),
        "finance_summary_annual": (
            root / "outputs" / "calculations" / "finance_summary_2026.json"
        ),
        "monthly_trends": (
            root / "outputs" / "calculations" / "monthly_trends_2026.csv"
        ),
        "anomaly_report_june": (
            root / "outputs" / "anomalies" / "anomaly_report_june_2026.json"
        ),
        "anomaly_report_annual": (
            root / "outputs" / "anomalies" / "anomaly_report_2026.json"
        ),
        "risk_summary_annual": (
            root / "outputs" / "anomalies" / "risk_summary_2026.json"
        ),
        "enriched_model": (
            root
            / "outputs"
            / "intermediate"
            / "financial_document_model_enriched.json"
        ),
    }
    return PlannerInputBundle(
        finance_summary_june=_read_json(paths["finance_summary_june"]),
        finance_summary_annual=_read_json(paths["finance_summary_annual"]),
        monthly_trends=_read_csv_records(paths["monthly_trends"]),
        anomaly_report_june=_read_json(paths["anomaly_report_june"]),
        anomaly_report_annual=_read_json(paths["anomaly_report_annual"]),
        risk_summary_annual=_read_json(paths["risk_summary_annual"]),
        enriched_model=_read_json(paths["enriched_model"]),
        source_files=tuple(path.name for path in paths.values()),
    )
