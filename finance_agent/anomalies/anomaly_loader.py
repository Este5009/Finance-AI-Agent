"""Validated loader for Step 3 calculation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class CalculationOutputLoadError(RuntimeError):
    """Raised when required calculation artifacts are missing or unreadable."""


@dataclass(frozen=True)
class CalculationOutputBundle:
    """Finance summary and CSV evidence for one reporting scope."""

    period_slug: str
    finance_summary_path: str
    kpi_summary_path: str
    department_summary_path: str
    category_summary_path: str
    monthly_trends_path: str | None
    finance_document: dict[str, Any]
    kpi_summary: pd.DataFrame
    department_summary: pd.DataFrame
    category_summary: pd.DataFrame
    monthly_trends: pd.DataFrame

    @property
    def report_period(self) -> str:
        """Return the calculation report-period label.

        Inputs: loaded finance document.
        Outputs: report period string.
        Assumptions: Step 3 always emits report_period.
        """

        return str(self.finance_document.get("report_period") or self.period_slug)

    @property
    def finance_summary(self) -> dict[str, Any]:
        """Return the nested calculated finance summary.

        Inputs: loaded finance document.
        Outputs: finance_summary dictionary.
        Assumptions: missing summary becomes an empty dictionary.
        """

        value = self.finance_document.get("finance_summary")
        return value if isinstance(value, dict) else {}


def _required_path(path: Path) -> Path:
    """Validate a required calculation artifact path.

    Inputs: expected file path.
    Outputs: same path when it exists as a file.
    Assumptions: anomaly detection must fail clearly on incomplete calculations.
    """

    if not path.exists() or not path.is_file():
        raise CalculationOutputLoadError(
            f"Required calculation output does not exist: {path}"
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    """Read one calculation JSON artifact.

    Inputs: validated JSON path.
    Outputs: parsed dictionary.
    Assumptions: finance summary JSON root must be an object.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalculationOutputLoadError(
            f"Unable to read calculation JSON '{path}': {error}"
        ) from error
    if not isinstance(value, dict):
        raise CalculationOutputLoadError(
            f"Calculation JSON root must be an object: {path}"
        )
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one calculation CSV artifact with a clear error.

    Inputs: validated CSV path.
    Outputs: pandas DataFrame.
    Assumptions: calculation CSV headers define the detector input contract.
    """

    try:
        return pd.read_csv(path)
    except Exception as error:
        raise CalculationOutputLoadError(
            f"Unable to read calculation CSV '{path}': {error}"
        ) from error


def load_calculation_outputs(
    calculations_directory: str | Path,
    period_slug: str,
    *,
    include_monthly_trends: bool = False,
) -> CalculationOutputBundle:
    """Load all calculation outputs required for one anomaly report.

    Inputs: calculations directory, period slug, and trend requirement.
    Outputs: structured JSON/CSV calculation evidence.
    Assumptions: annual reports use period slug 2026 and include monthly trends.
    """

    directory = Path(calculations_directory).resolve()
    finance_path = _required_path(directory / f"finance_summary_{period_slug}.json")
    kpi_path = _required_path(directory / f"kpi_summary_{period_slug}.csv")
    department_path = _required_path(
        directory / f"department_summary_{period_slug}.csv"
    )
    category_path = _required_path(
        directory / f"category_summary_{period_slug}.csv"
    )
    trends_path = directory / f"monthly_trends_{period_slug}.csv"
    if include_monthly_trends:
        _required_path(trends_path)

    return CalculationOutputBundle(
        period_slug=period_slug,
        finance_summary_path=str(finance_path),
        kpi_summary_path=str(kpi_path),
        department_summary_path=str(department_path),
        category_summary_path=str(category_path),
        monthly_trends_path=(
            str(trends_path) if include_monthly_trends else None
        ),
        finance_document=_read_json(finance_path),
        kpi_summary=_read_csv(kpi_path),
        department_summary=_read_csv(department_path),
        category_summary=_read_csv(category_path),
        monthly_trends=(
            _read_csv(trends_path)
            if include_monthly_trends
            else pd.DataFrame()
        ),
    )
