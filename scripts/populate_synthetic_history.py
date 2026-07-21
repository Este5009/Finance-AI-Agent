"""Populate an isolated SQLite memory database from synthetic history periods."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.memory.database import connect_database, initialize_database  # noqa: E402
from finance_agent.memory.repository import MemoryRepository  # noqa: E402
from finance_agent.memory.run_storage import persist_pipeline_run  # noqa: E402
from finance_agent.orchestration import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    PipelineConfig,
    PipelineInputModel,
    PipelineRunResult,
    build_pipeline_input_model,
    run_pipeline_for_report,
)


PipelineRunner = Callable[[PipelineInputModel, PipelineConfig], PipelineRunResult]


@dataclass(frozen=True)
class SyntheticPeriodInput:
    """Paired synthetic report and goals document for one period.

    Inputs:
        period_slug: Canonical period slug such as ``2026_06``.
        report_path: Monthly financial report workbook.
        goals_path: Matching monthly goals PDF.
    Outputs:
        A stable period pair for chronological population.
    Assumptions:
        Generated Phase 12A filenames end in ``YYYY_MM``.
    """

    period_slug: str
    report_path: Path
    goals_path: Path


def discover_synthetic_period_inputs(history_root: str | Path) -> list[SyntheticPeriodInput]:
    """Discover and pair generated monthly reports with goals documents.

    Inputs:
        history_root: Root directory such as ``data/synthetic_history/recovery_2026``.
    Outputs:
        Chronologically ordered period inputs.
    Assumptions:
        Report files are named ``university_financial_report_YYYY_MM.xlsx`` and
        goals files are named ``financial_goals_YYYY_MM.pdf``.
    """

    root = Path(history_root)
    reports_dir = root / "reports"
    goals_dir = root / "goals"
    pairs: list[SyntheticPeriodInput] = []
    for report in sorted(reports_dir.glob("university_financial_report_*.xlsx")):
        period_slug = _period_slug_from_report(report)
        goals = goals_dir / f"financial_goals_{period_slug}.pdf"
        if not goals.is_file():
            raise FileNotFoundError(f"Missing goals document for {period_slug}: {goals}")
        pairs.append(SyntheticPeriodInput(period_slug, report, goals))
    return sorted(pairs, key=lambda item: item.period_slug)


def populate_synthetic_history(
    *,
    history_root: str | Path = PROJECT_ROOT / "data" / "synthetic_history" / "recovery_2026",
    database_path: str | Path = PROJECT_ROOT / "data" / "memory" / "recovery_2026_memory.db",
    output_directory: str | Path = PROJECT_ROOT / "outputs" / "history_population",
    project_root: str | Path = PROJECT_ROOT,
    language: str = "es",
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_timeout_seconds: float = 180.0,
    stage_timeout_seconds: float = 420.0,
    verify_idempotency: bool = True,
    resume_missing: bool = False,
    runner: PipelineRunner = run_pipeline_for_report,
) -> dict[str, Any]:
    """Run the generic pipeline over all synthetic periods and validate memory.

    Inputs:
        history_root: Generated Phase 12A scenario root.
        database_path: Dedicated SQLite database for this historical test run.
        output_directory: Directory for population and validation JSON reports.
        project_root: Repository root.
        language: Report language passed through the generic pipeline.
        model: Single Ollama model for all LLM stages.
        ollama_timeout_seconds: Ollama timeout passed to the orchestrator.
        stage_timeout_seconds: Per-stage timeout passed to the orchestrator.
        verify_idempotency: Whether to rerun all periods once after the first pass.
        resume_missing: Whether to skip periods already stored in the target DB.
        runner: Pipeline function, injectable for tests.
    Outputs:
        Population summary dictionary written to disk.
    Assumptions:
        The production memory database is not modified because every config points
        at the supplied dedicated database path.
    """

    started = time.perf_counter()
    history = Path(history_root)
    output = Path(output_directory)
    database = Path(database_path)
    output.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(database)

    periods = discover_synthetic_period_inputs(history)
    if resume_missing:
        stored_periods = set(_load_observed_memory_patterns(database)["periods"])
        periods_to_run = [period for period in periods if period.period_slug not in stored_periods]
    else:
        periods_to_run = periods
    first_pass = _run_population_pass(
        periods=periods_to_run,
        database_path=database,
        project_root=Path(project_root),
        language=language,
        model=model,
        ollama_timeout_seconds=ollama_timeout_seconds,
        stage_timeout_seconds=stage_timeout_seconds,
        runner=runner,
        pass_name="initial",
        checkpoint_directory=output,
        checkpoint_history_root=history,
    )
    counts_after_first = MemoryRepository(database).table_counts()

    second_pass: list[dict[str, Any]] = []
    idempotent = None
    counts_after_second = counts_after_first
    if verify_idempotency:
        second_pass = _run_population_pass(
            periods=periods,
            database_path=database,
            project_root=Path(project_root),
            language=language,
            model=model,
            ollama_timeout_seconds=ollama_timeout_seconds,
            stage_timeout_seconds=stage_timeout_seconds,
            runner=runner,
            pass_name="idempotency",
            checkpoint_directory=output,
            checkpoint_history_root=history,
        )
        counts_after_second = MemoryRepository(database).table_counts()
        idempotent = _idempotency_verified(
            database,
            counts_after_first,
            counts_after_second,
            second_pass,
        )

    validation_report = validate_population_against_manifest(database, history / "scenario_manifest.json")
    summary = {
        "history_root": str(history),
        "database_path": str(database),
        "output_directory": str(output),
        "periods_discovered": len(periods),
        "periods_run": len(periods_to_run),
        "resume_missing": resume_missing,
        "successful_periods": [row["period_slug"] for row in first_pass if row["stored"]],
        "failed_periods": [row for row in first_pass if not row["stored"]],
        "initial_pass": first_pass,
        "idempotency_pass": second_pass,
        "idempotency_verified": idempotent,
        "table_counts": counts_after_second,
        "validation": validation_report,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(output / "population_summary.json", summary)
    _write_json(output / "validation_report.json", validation_report)
    return summary


def validate_population_against_manifest(database_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Compare stored historical memory records against the scenario manifest.

    Inputs:
        database_path: SQLite memory database to validate.
        manifest_path: Phase 12A scenario manifest.
    Outputs:
        JSON-compatible validation report.
    Assumptions:
        Validation is read-only and uses compact stored KPIs/anomalies/recommendations.
    """

    database = Path(database_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    observed = _load_observed_memory_patterns(database)
    expected = {
        "periods": sorted(manifest.get("monthly_totals", {}).keys()),
        "payroll_trend": manifest.get("monthly_payroll_ratio_trend", {}),
        "collection_trend": manifest.get("collection_rate_trend", {}),
        "health_sciences_overspending_periods": manifest.get("health_sciences_overspending_periods", []),
        "recurring_vendor_anomaly_periods": manifest.get("recurring_vendor_anomaly_periods", []),
        "recommendation_milestone": manifest.get("recommendation_milestone", {}),
        "cash_flow_recovery_periods": manifest.get("cash_flow_recovery_periods", []),
    }
    checks = {
        "all_periods_stored": observed["periods"] == expected["periods"],
        "payroll_trend_matches": _numeric_series_matches(observed["payroll_trend"], expected["payroll_trend"], tolerance=0.025),
        "collection_trend_matches": _numeric_series_matches(observed["collection_trend"], expected["collection_trend"], tolerance=0.025),
        "health_sciences_overspending_matches": observed["health_sciences_overspending_periods"]
        == expected["health_sciences_overspending_periods"],
        "recurring_vendor_anomaly_matches": observed["recurring_vendor_anomaly_periods"]
        == expected["recurring_vendor_anomaly_periods"],
        "recommendation_timeline_matches": expected["recommendation_milestone"].get("period") in observed["recommendation_periods"],
        "cash_flow_recovery_matches": observed["cash_flow_recovery_periods"] == expected["cash_flow_recovery_periods"],
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "expected": expected,
        "observed": observed,
    }


def _run_population_pass(
    *,
    periods: list[SyntheticPeriodInput],
    database_path: Path,
    project_root: Path,
    language: str,
    model: str,
    ollama_timeout_seconds: float,
    stage_timeout_seconds: float,
    runner: PipelineRunner,
    pass_name: str,
    checkpoint_directory: Path | None = None,
    checkpoint_history_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run one chronological population pass.

    Inputs:
        periods: Period report/goals pairs.
        database_path: Dedicated SQLite database.
        project_root: Repository root.
        language: Report language.
        model: Single Ollama model.
        ollama_timeout_seconds: Ollama timeout.
        stage_timeout_seconds: Stage timeout.
        runner: Pipeline runner function.
        pass_name: Diagnostic pass label.
        checkpoint_directory: Optional directory for incremental progress JSON.
        checkpoint_history_root: Optional history root for checkpoint metadata.
    Outputs:
        Per-period status rows.
    Assumptions:
        Recoverable per-period failures are recorded and do not stop the pass.
    """

    rows: list[dict[str, Any]] = []
    for period in periods:
        period_started = time.perf_counter()
        try:
            input_model = build_pipeline_input_model(
                financial_report_path=period.report_path,
                goals_document_path=period.goals_path,
                period_override=period.period_slug.replace("_", "-"),
                report_language=language,
            )
            config = PipelineConfig.from_project_root(
                project_root,
                python_executable=sys.executable,
                input_model=input_model,
                ollama_model=model,
                ollama_timeout_seconds=ollama_timeout_seconds,
                stage_timeout_seconds=stage_timeout_seconds,
                enable_cache=False,
                memory_database_path=database_path,
            )
            result = runner(input_model, config)
            storage = persist_pipeline_run(result, period_slug=period.period_slug, database_path=database_path)
            rows.append(
                {
                    "pass": pass_name,
                    "period_slug": period.period_slug,
                    "pipeline_success": bool(result.success),
                    "stored": bool(storage.stored),
                    "storage_reason": storage.reason,
                    "updated_existing": bool(storage.updated_existing),
                    "warnings": list(result.warnings),
                    "runtime_seconds": round(time.perf_counter() - period_started, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001 - population continues after period failures.
            rows.append(
                {
                    "pass": pass_name,
                    "period_slug": period.period_slug,
                    "pipeline_success": False,
                    "stored": False,
                    "storage_reason": str(exc),
                    "updated_existing": False,
                    "warnings": [],
                    "runtime_seconds": round(time.perf_counter() - period_started, 3),
                }
            )
        if checkpoint_directory is not None:
            # Long Ollama-backed population runs can exceed a shell timeout. A
            # small checkpoint after every period preserves what was learned.
            _write_checkpoint(
                checkpoint_directory,
                database_path,
                checkpoint_history_root,
                pass_name,
                rows,
            )
    return rows


def _idempotency_verified(
    database_path: Path,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    rerun_rows: list[dict[str, Any]],
) -> bool:
    """Return whether rerunning reused existing period records without duplicates.

    Inputs:
        database_path: SQLite memory database.
        counts_before: Table counts before the rerun.
        counts_after: Table counts after the rerun.
        rerun_rows: Per-period rows from the idempotency pass.
    Outputs:
        True when pipeline run identities are stable and duplicate periods do not exist.
    Assumptions:
        Child rows may be replaced because live Ollama strategy text can vary; the
        idempotency contract is that reprocessing updates/reuses runs rather than
        creating duplicate pipeline_runs.
    """

    if counts_before.get("pipeline_runs") != counts_after.get("pipeline_runs"):
        return False
    if not rerun_rows or not all(row.get("stored") and row.get("updated_existing") for row in rerun_rows):
        return False
    with connect_database(database_path) as connection:
        duplicate_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT period
                    FROM pipeline_runs
                    GROUP BY period
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
    return duplicate_count == 0


def _write_checkpoint(
    output_directory: Path,
    database_path: Path,
    history_root: Path | None,
    pass_name: str,
    rows: list[dict[str, Any]],
) -> None:
    """Write an incremental population checkpoint.

    Inputs:
        output_directory: Directory for checkpoint JSON.
        database_path: Target memory database.
        history_root: Optional synthetic history root.
        pass_name: Active population pass.
        rows: Per-period rows completed so far.
    Outputs:
        ``population_summary.json`` checkpoint.
    Assumptions:
        Checkpoints are diagnostic and may be overwritten by the final summary.
    """

    checkpoint = {
        "status": "in_progress",
        "pass": pass_name,
        "history_root": str(history_root) if history_root else None,
        "database_path": str(database_path),
        "completed_rows": rows,
        "successful_periods": [row["period_slug"] for row in rows if row["stored"]],
        "failed_periods": [row for row in rows if not row["stored"]],
        "table_counts": MemoryRepository(database_path).table_counts(),
    }
    _write_json(output_directory / "population_summary.json", checkpoint)


def _load_observed_memory_patterns(database_path: Path) -> dict[str, Any]:
    """Load compact observed trends and anomaly periods from SQLite memory.

    Inputs:
        database_path: SQLite database path.
    Outputs:
        Observed pattern dictionary.
    Assumptions:
        This function is read-only and does not expose SQL to Ollama.
    """

    with connect_database(database_path) as connection:
        periods = [
            str(row["period"])
            for row in connection.execute("SELECT period FROM pipeline_runs ORDER BY period").fetchall()
        ]
        payroll = _metric_series(connection, "payroll_percentage_of_revenue")
        collection = _metric_series(connection, "student_payment_collection_rate")
        cash_flow = _metric_series(connection, "net_cash_flow")
        anomaly_rows = connection.execute(
            """
            SELECT runs.period, child.department, child.type, child.metric, child.description
            FROM anomalies AS child
            JOIN pipeline_runs AS runs ON child.run_id = runs.run_id
            ORDER BY runs.period
            """
        ).fetchall()
        recommendation_rows = connection.execute(
            """
            SELECT runs.period, child.department, child.action
            FROM recommendations AS child
            JOIN pipeline_runs AS runs ON child.run_id = runs.run_id
            ORDER BY runs.period
            """
        ).fetchall()
        artifact_rows = connection.execute(
            """
            SELECT runs.period, child.artifact_type, child.path
            FROM artifacts AS child
            JOIN pipeline_runs AS runs ON child.run_id = runs.run_id
            WHERE child.artifact_type IN ('anomaly_report', 'normalized_table')
            ORDER BY runs.period
            """
        ).fetchall()
    artifact_anomalies = _load_anomaly_artifact_rows(artifact_rows)
    embedded_anomalies = _load_embedded_anomaly_rows(artifact_rows)
    hs_periods = sorted(
        {
            str(row["period"])
            for row in anomaly_rows
            if _is_health_sciences_overspend(row)
        }
        | {
            period
            for period, anomalies in artifact_anomalies.items()
            if any(_is_health_sciences_overspend_dict(item) for item in anomalies)
        }
        | {
            period
            for period, anomalies in embedded_anomalies.items()
            if any(_is_health_sciences_overspend_dict(item) for item in anomalies)
        }
    )
    vendor_periods = sorted(
        {
            str(row["period"])
            for row in anomaly_rows
            if _is_vendor_anomaly(row)
        }
        | {
            period
            for period, anomalies in artifact_anomalies.items()
            if any(_is_vendor_anomaly_dict(item) for item in anomalies)
        }
        | {
            period
            for period, anomalies in embedded_anomalies.items()
            if any(_is_vendor_anomaly_dict(item) for item in anomalies)
        }
    )
    recommendation_periods = sorted(
        {
            str(row["period"])
            for row in recommendation_rows
            if "overtime" in str(row["action"]).casefold()
            or "tiempo extra" in str(row["action"]).casefold()
            or "Health Sciences" in str(row["department"] or row["action"])
        }
    )
    cash_recovery = sorted(period for period, value in cash_flow.items() if value > 0 and period >= "2026_09")
    return {
        "periods": periods,
        "payroll_trend": payroll,
        "collection_trend": collection,
        "health_sciences_overspending_periods": hs_periods,
        "recurring_vendor_anomaly_periods": vendor_periods,
        "recommendation_periods": recommendation_periods,
        "cash_flow_recovery_periods": cash_recovery,
    }


def _load_anomaly_artifact_rows(artifact_rows: list[sqlite3.Row]) -> dict[str, list[dict[str, Any]]]:
    """Load anomaly rows from artifact references stored in memory.

    Inputs:
        artifact_rows: Joined artifact rows pointing at anomaly report JSON files.
    Outputs:
        Period-to-anomaly-list mapping.
    Assumptions:
        Artifact references are part of stored historical access and may carry
        richer context than compact anomaly rows.
    """

    anomalies_by_period: dict[str, list[dict[str, Any]]] = {}
    for row in artifact_rows:
        if str(row["artifact_type"]) != "anomaly_report":
            continue
        path = Path(str(row["path"]))
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        anomalies = document.get("anomalies", [])
        if isinstance(anomalies, list):
            anomalies_by_period[str(row["period"])] = [
                item for item in anomalies if isinstance(item, dict)
            ]
    return anomalies_by_period


def _load_embedded_anomaly_rows(artifact_rows: list[sqlite3.Row]) -> dict[str, list[dict[str, Any]]]:
    """Load scenario-embedded anomaly rows from normalized table artifacts.

    Inputs:
        artifact_rows: Joined artifact rows from memory storage.
    Outputs:
        Period-to-embedded-anomaly mapping.
    Assumptions:
        Phase 12A stores expected scenario signals in the Anomalies_Embedded
        workbook sheet, which is preserved as a normalized table artifact.
    """

    anomalies_by_period: dict[str, list[dict[str, Any]]] = {}
    for row in artifact_rows:
        path = Path(str(row["path"]))
        if str(row["artifact_type"]) != "normalized_table" or "anomalies_embedded" not in path.name.casefold():
            continue
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            anomalies_by_period[str(row["period"])] = [dict(item) for item in csv.DictReader(handle)]
    return anomalies_by_period


def _metric_series(connection: sqlite3.Connection, metric: str) -> dict[str, float]:
    """Read one KPI metric series from memory.

    Inputs:
        connection: SQLite connection.
        metric: KPI metric name.
    Outputs:
        Period-to-value mapping.
    Assumptions:
        Metric names are generated by deterministic Python calculations.
    """

    rows = connection.execute(
        """
        SELECT runs.period, child.value
        FROM kpis AS child
        JOIN pipeline_runs AS runs ON child.run_id = runs.run_id
        WHERE child.metric = ?
        ORDER BY runs.period
        """,
        (metric,),
    ).fetchall()
    return {str(row["period"]): float(row["value"]) for row in rows if row["value"] is not None}


def _is_health_sciences_overspend(row: sqlite3.Row) -> bool:
    """Return whether an anomaly row represents Health Sciences overspending.

    Inputs:
        row: Joined anomaly row.
    Outputs:
        True for the expected scenario anomaly family.
    Assumptions:
        Either department, description, metric, or type may carry the signal.
    """

    text = " ".join(
        str(row[key] or "")
        for key in ("department", "type", "metric", "description")
    ).casefold()
    return "health sciences" in text and ("overspend" in text or "payroll" in text or "variance" in text)


def _is_health_sciences_overspend_dict(item: dict[str, Any]) -> bool:
    """Return whether an anomaly dictionary describes Health Sciences overspending.

    Inputs:
        item: Anomaly dictionary loaded from an artifact reference.
    Outputs:
        True when title/evidence/description identify the expected pattern.
    Assumptions:
        Artifact anomaly reports are trusted processed outputs, not raw Excel.
    """

    text = " ".join(
        str(item.get(key) or "")
        for key in (
            "department",
            "Department",
            "title",
            "Title",
            "rule_id",
            "Rule_ID",
            "anomaly_type",
            "Anomaly_Type",
            "metric",
            "Metric",
            "description",
            "Description",
            "evidence",
            "Evidence",
        )
    ).casefold()
    return "health sciences" in text and ("overspend" in text or "payroll" in text or "variance" in text)


def _is_vendor_anomaly(row: sqlite3.Row) -> bool:
    """Return whether an anomaly row represents the recurring vendor anomaly.

    Inputs:
        row: Joined anomaly row.
    Outputs:
        True for vendor-review or duplicate-payment anomaly rows.
    Assumptions:
        Real and mocked runs may use rule IDs or natural-language descriptions.
    """

    text = " ".join(str(row[key] or "") for key in ("type", "metric", "description")).casefold()
    return "vendor" in text or "medsupply" in text or "duplicate" in text


def _is_vendor_anomaly_dict(item: dict[str, Any]) -> bool:
    """Return whether an anomaly dictionary describes the vendor anomaly.

    Inputs:
        item: Anomaly dictionary loaded from an artifact reference.
    Outputs:
        True for vendor-review or duplicate-payment anomaly rows.
    Assumptions:
        Rule IDs and evidence text may differ slightly across pipeline versions.
    """

    text = " ".join(
        str(item.get(key) or "")
        for key in (
            "title",
            "Title",
            "rule_id",
            "Rule_ID",
            "anomaly_type",
            "Anomaly_Type",
            "metric",
            "Metric",
            "description",
            "Description",
            "evidence",
            "Evidence",
        )
    ).casefold()
    return "vendor" in text or "medsupply" in text or "duplicate" in text


def _numeric_series_matches(observed: dict[str, float], expected: dict[str, Any], *, tolerance: float) -> bool:
    """Compare two numeric period series with a tolerance.

    Inputs:
        observed: Observed period-to-value mapping.
        expected: Expected period-to-value mapping.
        tolerance: Maximum allowed absolute difference.
    Outputs:
        True when all expected periods exist and match within tolerance.
    Assumptions:
        Pipeline calculations may round slightly differently from generated manifest values.
    """

    if set(observed) != set(expected):
        return False
    return all(abs(float(observed[period]) - float(expected[period])) <= tolerance for period in expected)


def _period_slug_from_report(path: Path) -> str:
    """Extract a period slug from a generated report path.

    Inputs:
        path: Generated report workbook path.
    Outputs:
        Period slug such as ``2026_06``.
    Assumptions:
        Filename ends with ``YYYY_MM.xlsx``.
    """

    parts = path.stem.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON output for population diagnostics.

    Inputs:
        path: Target JSON path.
        data: JSON-compatible dictionary.
    Outputs:
        None.
    Assumptions:
        Parent directories may be created.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the historical population CLI parser.

    Inputs:
        None.
    Outputs:
        Argument parser.
    Assumptions:
        Defaults target the Phase 12A recovery scenario and isolated memory DB.
    """

    parser = argparse.ArgumentParser(description="Populate memory from the synthetic recovery history.")
    parser.add_argument("--history-root", type=Path, default=PROJECT_ROOT / "data" / "synthetic_history" / "recovery_2026")
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "memory" / "recovery_2026_memory.db")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "history_population")
    parser.add_argument("--language", default="es")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout", type=float, default=180.0)
    parser.add_argument("--stage-timeout", type=float, default=420.0)
    parser.add_argument("--skip-idempotency-rerun", action="store_true")
    parser.add_argument(
        "--resume-missing",
        action="store_true",
        help="Skip periods already stored in the target recovery database.",
    )
    return parser


def main() -> int:
    """Run historical population from CLI arguments.

    Inputs:
        Command-line arguments.
    Outputs:
        Process exit code.
    Assumptions:
        Validation failure exits non-zero after writing diagnostic reports.
    """

    args = build_parser().parse_args()
    summary = populate_synthetic_history(
        history_root=args.history_root,
        database_path=args.database,
        output_directory=args.output_dir,
        language=args.language,
        model=args.model,
        ollama_timeout_seconds=args.ollama_timeout,
        stage_timeout_seconds=args.stage_timeout,
        verify_idempotency=not args.skip_idempotency_rerun,
        resume_missing=args.resume_missing,
    )
    print(f"History root: {summary['history_root']}")
    print(f"Database: {summary['database_path']}")
    print(f"Successful periods: {len(summary['successful_periods'])}")
    print(f"Failed periods: {len(summary['failed_periods'])}")
    print(f"Idempotency verified: {summary['idempotency_verified']}")
    print(f"Validation: {'passed' if summary['validation']['valid'] else 'failed'}")
    print(f"Table counts: {summary['table_counts']}")
    return 0 if summary["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
