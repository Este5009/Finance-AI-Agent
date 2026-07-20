"""Data models for synthetic university financial history generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


PeriodType = Literal["monthly"]


@dataclass(frozen=True)
class SyntheticHistoryConfig:
    """Configuration for generating a deterministic synthetic financial history.

    Inputs:
        year: Fiscal/calendar year to generate.
        scenario: Named scenario to use. Phase 12A supports ``recovery``.
        seed: Random seed for deterministic minor row-level variation.
        departments: University departments included in each monthly workbook.
        output_directory: Base directory where scenario folders are created.
        overwrite: Whether an existing scenario directory may be replaced in-place.
    Outputs:
        A validated configuration consumed by the generator.
    Assumptions:
        The generator writes one workbook and one goals PDF per month.
    """

    year: int = 2026
    scenario: str = "recovery"
    seed: int = 42
    departments: tuple[str, ...] = (
        "Engineering",
        "Business",
        "Health Sciences",
        "Arts & Humanities",
        "Student Services",
        "Administration",
    )
    output_directory: Path = Path("data/synthetic_history")
    overwrite: bool = False

    @property
    def scenario_slug(self) -> str:
        """Return the stable folder slug for this scenario and year."""

        return f"{self.scenario}_{self.year}"


@dataclass(frozen=True)
class MonthlyScenarioPoint:
    """Scenario controls and expectations for one generated month.

    Inputs:
        month: Calendar month number.
        narrative_es: Spanish period narrative for goals and manifest output.
        revenue_factor: Multiplier applied to baseline monthly revenue.
        payroll_ratio: Expected payroll divided by actual revenue.
        collection_rate: Expected student collection rate.
        net_cash_flow: Actual net cash flow encoded in the cash-flow sheet.
        health_sciences_overtime_factor: Multiplier for Health Sciences overtime.
        vendor_anomaly: Whether recurring vendor anomaly rows should be encoded.
        recommendation_milestone: Whether goals introduce the overtime recommendation.
        policy_action_es: Spanish user-facing action/milestone text.
    Outputs:
        A compact deterministic scenario point.
    Assumptions:
        These values are scenario-level controls, not generated financial rows.
    """

    month: int
    narrative_es: str
    revenue_factor: float
    payroll_ratio: float
    collection_rate: float
    net_cash_flow: float
    health_sciences_overtime_factor: float = 1.0
    vendor_anomaly: bool = False
    recommendation_milestone: bool = False
    policy_action_es: str = ""

    @property
    def period(self) -> str:
        """Return the canonical period slug for this scenario point."""

        return f"{self.month:02d}"


@dataclass
class MonthlyFinancialData:
    """Generated rows and totals for one monthly workbook.

    Inputs:
        period_slug: Stable period identifier such as ``2026_06``.
        rows_by_sheet: Workbook rows keyed by sheet name.
        totals: Reconciled monthly totals used by validation and the manifest.
        expected_anomalies: Scenario-driven anomaly IDs expected in this month.
    Outputs:
        In-memory monthly financial data that can be written to artifacts.
    Assumptions:
        Rows contain only JSON-serializable scalar values plus date objects for Excel.
    """

    period_slug: str
    month_name: str
    rows_by_sheet: dict[str, list[dict[str, Any]]]
    totals: dict[str, float]
    expected_anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratedHistory:
    """References to generated synthetic history artifacts.

    Inputs:
        root_directory: Scenario root directory.
        report_paths: Monthly financial report workbook paths.
        goals_paths: Monthly goals PDF paths.
        manifest_path: JSON manifest path.
        manifest: Manifest dictionary written to disk.
    Outputs:
        A lightweight summary returned by the generator and CLI.
    Assumptions:
        Paths are filesystem paths, not database artifact references.
    """

    root_directory: Path
    report_paths: list[Path]
    goals_paths: list[Path]
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass
class SyntheticHistoryValidationResult:
    """Validation result for generated synthetic history artifacts.

    Inputs:
        is_valid: Whether all required validation checks passed.
        errors: Blocking problems found while validating files and reconciliations.
        warnings: Non-blocking observations.
        reconciliations: Numeric reconciliation summary for reporting.
    Outputs:
        A structured result suitable for tests and CLI summaries.
    Assumptions:
        Validation reads generated artifacts only and never runs the finance pipeline.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reconciliations: dict[str, Any] = field(default_factory=dict)
