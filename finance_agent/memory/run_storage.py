"""Persist completed strategy-backed pipeline runs into SQLite memory storage."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from finance_agent.memory.models import (
    DEFAULT_MEMORY_DB_PATH,
    AnomalyRecord,
    ArtifactRecord,
    GoalRecord,
    KpiRecord,
    MemoryFactRecord,
    RecommendationRecord,
    StorageResult,
    StoredPipelineRun,
)
from finance_agent.memory.repository import MemoryRepository
from finance_agent.orchestration.pipeline_models import PipelineRunResult
from finance_agent.reporting.report_quality import validate_report_artifacts


def _file_checksum(path: Path) -> str | None:
    """Return a SHA-256 checksum for an existing file.

    Inputs: artifact path.
    Outputs: hex digest or None when the file is missing.
    Assumptions: large files are streamed; contents are not stored in SQLite.
    """

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    """Return a SHA-256 hash for a deterministic text key.

    Inputs: text value.
    Outputs: hex digest.
    Assumptions: input is already normalized by caller.
    """

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Inputs: JSON path.
    Outputs: decoded dictionary or empty dictionary if unavailable.
    Assumptions: storage skips missing optional artifacts rather than failing.
    """

    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> float | None:
    """Convert a scalar to float when possible.

    Inputs: arbitrary scalar.
    Outputs: float or None.
    Assumptions: booleans are not financial numeric values.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _expected_artifact_paths(result: PipelineRunResult, period_slug: str) -> dict[str, Path]:
    """Return canonical artifact paths for a period-slugged run.

    Inputs: pipeline result and period slug.
    Outputs: mapping of artifact label to path.
    Assumptions: current generic output paths remain period-slugged.
    """

    outputs = result.config.output_directory
    return {
        "finance_summary": outputs / "calculations" / f"finance_summary_{period_slug}.json",
        "kpi_summary": outputs / "calculations" / f"kpi_summary_{period_slug}.csv",
        "department_summary": outputs / "calculations" / f"department_summary_{period_slug}.csv",
        "category_summary": outputs / "calculations" / f"category_summary_{period_slug}.csv",
        "anomaly_report": outputs / "anomalies" / f"anomaly_report_{period_slug}.json",
        "risk_summary": outputs / "anomalies" / f"risk_summary_{period_slug}.json",
        "evidence_package": outputs / "evidence" / f"evidence_package_{period_slug}.json",
        "retrieval_summary": outputs / "evidence" / f"retrieval_summary_{period_slug}.json",
        "strategic_analysis": outputs / "analysis" / f"strategic_analysis_{period_slug}.json",
        "report_model": outputs / "report" / f"report_model_{period_slug}.json",
        "report_html": outputs / "report" / f"financial_report_{period_slug}.html",
        "report_pdf": outputs / "report" / f"financial_report_{period_slug}.pdf",
        "ollama_plan": outputs / "plans" / f"ollama_plan_{period_slug}.json",
        "execution_queue": outputs / "plans" / f"execution_queue_{period_slug}.json",
        "investigation_plan": outputs / "plans" / f"investigation_plan_{period_slug}.json",
        "enriched_model": outputs
        / "intermediate"
        / period_slug
        / "financial_document_model_enriched.json",
        "goals_text": outputs / "inspection" / f"goals_text_{period_slug}.txt",
        "workbook_inspection": outputs / "inspection" / f"workbook_inspection_{period_slug}.json",
    }


def _artifact_type(path: Path, label: str | None = None) -> str:
    """Infer an artifact type from a path or supplied label.

    Inputs: artifact path and optional canonical label.
    Outputs: stable artifact type string.
    Assumptions: artifact type is for search/filtering, not validation.
    """

    if label:
        return label
    name = path.name.casefold()
    if name.endswith(".pdf"):
        return "report_pdf"
    if name.endswith(".html"):
        return "report_html"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".json"):
        return "json"
    return "artifact"


def _collect_artifacts(
    result: PipelineRunResult,
    period_slug: str,
) -> tuple[ArtifactRecord, ...]:
    """Collect artifact references and checksums without storing blobs.

    Inputs: pipeline result and period slug.
    Outputs: artifact records.
    Assumptions: missing optional artifacts are ignored; required gating happens earlier.
    """

    records: dict[tuple[str, str], ArtifactRecord] = {}
    expected = _expected_artifact_paths(result, period_slug)
    for label, path in expected.items():
        if path.is_file():
            resolved = str(path.resolve())
            records[(label, resolved)] = ArtifactRecord(label, resolved, _file_checksum(path))
    for path_text in result.output_files:
        path = Path(path_text)
        if path.is_file():
            resolved = str(path.resolve())
            kind = _artifact_type(path)
            records[(kind, resolved)] = ArtifactRecord(kind, resolved, _file_checksum(path))

    normalized_dir = result.config.output_directory / "intermediate" / period_slug / "normalized_tables"
    if normalized_dir.is_dir():
        for path in normalized_dir.glob("*.csv"):
            resolved = str(path.resolve())
            records[("normalized_table", resolved)] = ArtifactRecord(
                "normalized_table",
                resolved,
                _file_checksum(path),
            )
    return tuple(records.values())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows into dictionaries.

    Inputs: CSV path.
    Outputs: list of row dictionaries.
    Assumptions: missing files simply produce no rows.
    """

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _extract_kpis(
    period_slug: str,
    finance_summary: dict[str, Any],
    kpi_csv_path: Path,
) -> tuple[KpiRecord, ...]:
    """Extract KPI rows from processed KPI artifacts.

    Inputs: period slug, finance summary, and KPI CSV path.
    Outputs: KPI records.
    Assumptions: KPI values are already calculated by Python.
    """

    records: list[KpiRecord] = []
    csv_rows = _read_csv_rows(kpi_csv_path)
    source_rows: Iterable[dict[str, Any]] = csv_rows
    if not csv_rows:
        source_rows = [
            row for row in finance_summary.get("kpi_summary", []) if isinstance(row, dict)
        ]
    for row in source_rows:
        metric = row.get("metric") or row.get("name")
        if not metric:
            continue
        records.append(
            KpiRecord(
                period=str(row.get("period") or period_slug),
                department=str(row.get("department") or "") or None,
                metric=str(metric),
                value=_to_float(row.get("value")),
                unit=str(row.get("unit") or "") or None,
                status=str(row.get("status") or row.get("availability") or "") or None,
            )
        )
    return tuple(records)


def _extract_anomalies(anomaly_report: dict[str, Any]) -> tuple[AnomalyRecord, ...]:
    """Extract anomaly rows from deterministic anomaly output.

    Inputs: anomaly report JSON.
    Outputs: anomaly records.
    Assumptions: missing anomaly IDs are skipped because they cannot be tracked.
    """

    rows: list[AnomalyRecord] = []
    for index, item in enumerate(anomaly_report.get("anomalies", []), start=1):
        if not isinstance(item, dict):
            continue
        anomaly_id = str(item.get("anomaly_id") or f"ANOMALY-{index:03d}")
        values = {
            "observed_value": item.get("observed_value"),
            "threshold_value": item.get("threshold_value"),
            "expected_value": item.get("expected_value"),
            "variance": item.get("variance"),
        }
        rows.append(
            AnomalyRecord(
                anomaly_id=anomaly_id,
                period=str(item.get("period") or anomaly_report.get("report_period") or "") or None,
                department=str(item.get("department") or "") or None,
                type=str(item.get("rule_id") or item.get("type") or item.get("title") or "") or None,
                severity=str(item.get("severity") or "") or None,
                metric=str(item.get("metric") or "") or None,
                values_json=json.dumps(values, sort_keys=True, ensure_ascii=False),
                description=str(item.get("description") or item.get("evidence") or item.get("title") or "")[:1000],
            )
        )
    return tuple(rows)


def _extract_recommendations(
    analysis_document: dict[str, Any],
) -> tuple[RecommendationRecord, ...]:
    """Extract recommendations from accepted strategic analysis.

    Inputs: strategic analysis document.
    Outputs: recommendation records.
    Assumptions: analysis validation already enforced recommendation shape.
    """

    analysis = analysis_document.get("analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    rows: list[RecommendationRecord] = []
    for index, item in enumerate(analysis.get("recommendations", []), start=1):
        if not isinstance(item, dict) or not item.get("action"):
            continue
        priority = str(item.get("priority") or "")
        rows.append(
            RecommendationRecord(
                recommendation_id=f"REC-{index:03d}",
                priority=priority or None,
                department=str(item.get("department") or "") or None,
                action=str(item.get("action")),
                expected_impact=str(item.get("expected_impact") or "") or None,
                follow_up_required=priority.casefold() in {"critical", "high"},
            )
        )
    return tuple(rows)


def _nested_metric(finance_summary: dict[str, Any], *path: str) -> Any:
    """Read a nested metric from the finance summary.

    Inputs: finance summary and key path.
    Outputs: nested value or None.
    Assumptions: missing processed values should not block storage.
    """

    value: Any = finance_summary.get("finance_summary", finance_summary)
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _extract_goals(
    finance_summary: dict[str, Any],
    goals_text_path: Path | None = None,
) -> tuple[GoalRecord, ...]:
    """Extract goal progress rows when processed outputs expose targets.

    Inputs: finance summary.
    Outputs: goal records.
    Assumptions: current synthetic outputs may not expose goal targets yet.
    """

    rows: list[GoalRecord] = []
    candidates = finance_summary.get("goal_progress", [])
    candidates = candidates if isinstance(candidates, list) else []
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict) or not item.get("metric"):
            continue
        rows.append(
            GoalRecord(
                goal_id=str(item.get("goal_id") or f"GOAL-{index:03d}"),
                metric=str(item["metric"]),
                target=_to_float(item.get("target")),
                actual=_to_float(item.get("actual")),
                unit=str(item.get("unit") or "") or None,
                progress_status=str(item.get("status") or item.get("progress_status") or "") or None,
            )
        )
    text = (
        goals_text_path.read_text(encoding="utf-8")
        if goals_text_path is not None and goals_text_path.is_file()
        else ""
    )
    lowered = text.casefold()
    # These deterministic rows mirror measurable goals already present in the
    # processed goals text. Actual values come only from Python-calculated output.
    if "annual revenue" in lowered and "$26.0 million" in lowered:
        rows.append(
            GoalRecord(
                goal_id="GOAL-ANNUAL-REVENUE",
                metric="total_revenue",
                target=26_000_000.0,
                actual=_to_float(_nested_metric(finance_summary, "total_revenue")),
                unit="USD",
                progress_status=None,
            )
        )
    if "payroll cost" in lowered and "42%" in lowered:
        rows.append(
            GoalRecord(
                goal_id="GOAL-PAYROLL-RATIO",
                metric="payroll_percentage_of_revenue",
                target=0.42,
                actual=_to_float(
                    _nested_metric(finance_summary, "payroll_percentage_of_revenue")
                ),
                unit="ratio",
                progress_status=None,
            )
        )
    if "tuition collection" in lowered and "94%" in lowered:
        rows.append(
            GoalRecord(
                goal_id="GOAL-COLLECTION-RATE",
                metric="collection_rate",
                target=0.94,
                actual=_to_float(
                    _nested_metric(finance_summary, "student_payments", "collection_rate")
                ),
                unit="ratio",
                progress_status=None,
            )
        )
    if "expense control" in lowered:
        rows.append(
            GoalRecord(
                goal_id="GOAL-EXPENSE-VARIANCE",
                metric="expense_variance",
                target=0.0,
                actual=_to_float(
                    _nested_metric(finance_summary, "budget_vs_actual", "expense_variance")
                ),
                unit="USD",
                progress_status=None,
            )
        )
    deduplicated = {goal.goal_id: goal for goal in rows}
    return tuple(deduplicated.values())


def _extract_memory_facts(
    analysis_document: dict[str, Any],
    evidence_package: dict[str, Any],
    anomalies: tuple[AnomalyRecord, ...],
) -> tuple[MemoryFactRecord, ...]:
    """Build compact memory facts from analysis, evidence, and anomaly summaries.

    Inputs: accepted analysis, evidence package, and anomaly records.
    Outputs: compact memory facts.
    Assumptions: facts are short retrieval aids, not full historical data.
    """

    analysis = analysis_document.get("analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    confidence = _to_float(analysis.get("confidence"))
    facts: list[MemoryFactRecord] = []
    for category, field_name in (
        ("key_finding", "key_findings"),
        ("root_cause", "root_causes"),
        ("strategic_priority", "strategic_priorities"),
        ("unresolved_issue", "missing_information"),
    ):
        values = analysis.get(field_name, [])
        values = values if isinstance(values, list) else []
        for index, value in enumerate(values, start=1):
            if str(value).strip():
                facts.append(
                    MemoryFactRecord(
                        category=category,
                        subject=f"{field_name}_{index}",
                        fact=str(value)[:1000],
                        confidence=confidence,
                    )
                )
    for anomaly in anomalies:
        if str(anomaly.severity or "").casefold() in {"critical", "high"}:
            facts.append(
                MemoryFactRecord(
                    category="recurring_risk",
                    subject=anomaly.metric or anomaly.anomaly_id,
                    fact=f"{anomaly.severity} anomaly: {anomaly.description or anomaly.type}",
                    confidence=confidence,
                )
            )
    for package in evidence_package.get("evidence_packages", [])[:8]:
        if not isinstance(package, dict):
            continue
        summary = str(package.get("evidence_summary") or "").strip()
        if summary:
            facts.append(
                MemoryFactRecord(
                    category="evidence_summary",
                    subject=str(package.get("task_id") or "evidence"),
                    fact=summary[:1000],
                    confidence=_to_float(package.get("confidence")) or confidence,
                )
            )
    deduplicated: dict[tuple[str, str, str], MemoryFactRecord] = {}
    for fact in facts:
        # Evidence and analysis can repeat the same compact sentence across
        # sections. Store it once so the memory index stays concise and the
        # database uniqueness rule remains a guardrail rather than a failure.
        deduplicated[(fact.category, fact.subject, fact.fact)] = fact
    return tuple(deduplicated.values())


def _quality_gate(
    result: PipelineRunResult,
    period_slug: str,
) -> tuple[bool, str | None, dict[str, Path], dict[str, Any]]:
    """Validate whether a pipeline result is eligible for memory storage.

    Inputs: pipeline result and period slug.
    Outputs: eligible flag, skip reason, expected paths, and analysis document.
    Assumptions: only accepted strategy and valid reports should be stored.
    """

    paths = _expected_artifact_paths(result, period_slug)
    if not result.success:
        return False, "pipeline result was not successful", paths, {}
    analysis = _load_json(paths["strategic_analysis"])
    if analysis.get("validation_status") != "accepted":
        return False, "strategic analysis was not accepted", paths, analysis
    quality = validate_report_artifacts(
        paths["report_model"],
        html_path=paths["report_html"],
        pdf_path=paths["report_pdf"],
    )
    if not quality.is_valid:
        return False, "report artifacts failed quality validation", paths, analysis
    return True, None, paths, analysis


def build_stored_pipeline_run(
    *,
    result: PipelineRunResult,
    period_slug: str,
) -> StoredPipelineRun:
    """Build a repository payload from existing processed artifacts.

    Inputs: pipeline result and period slug.
    Outputs: StoredPipelineRun payload.
    Assumptions: caller already passed quality gating.
    """

    input_model = result.config.input_model
    if input_model is None:
        raise ValueError("Pipeline result must include input_model for storage.")
    paths = _expected_artifact_paths(result, period_slug)
    finance_summary = _load_json(paths["finance_summary"])
    anomaly_report = _load_json(paths["anomaly_report"])
    evidence_package = _load_json(paths["evidence_package"])
    analysis_document = _load_json(paths["strategic_analysis"])
    report_hash = _file_checksum(input_model.financial_report_path) or ""
    goals_hash = _file_checksum(input_model.goals_document_path) or ""
    configuration_json = json.dumps(
        result.config.to_dict(),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    idempotency_payload = json.dumps(
        {
            "report_hash": report_hash,
            "goals_hash": goals_hash,
            "period": period_slug,
            "configuration": configuration_json,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    idempotency_key = _hash_text(idempotency_payload)
    run_id = f"RUN-{idempotency_key[:24]}"
    analysis = analysis_document.get("analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    artifacts = _collect_artifacts(result, period_slug)
    kpis = _extract_kpis(period_slug, finance_summary, paths["kpi_summary"])
    anomalies = _extract_anomalies(anomaly_report)
    return StoredPipelineRun(
        run_id=run_id,
        idempotency_key=idempotency_key,
        period=period_slug,
        period_type=input_model.period_type,
        started_at_utc=None,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        report_hash=report_hash,
        goals_hash=goals_hash,
        report_path=str(input_model.financial_report_path.resolve()),
        goals_path=str(input_model.goals_document_path.resolve()),
        language=input_model.report_language,
        model=result.config.ollama_model,
        confidence=_to_float(analysis.get("confidence")),
        cache_hit=result.cache_hit,
        cache_key=result.cache_key,
        status="completed",
        artifact_directory=str(result.config.output_directory.resolve()),
        configuration_json=configuration_json,
        artifacts=artifacts,
        kpis=kpis,
        anomalies=anomalies,
        recommendations=_extract_recommendations(analysis_document),
        goals=_extract_goals(finance_summary, paths.get("goals_text")),
        memory_facts=_extract_memory_facts(analysis_document, evidence_package, anomalies),
    )


def persist_pipeline_run(
    result: PipelineRunResult,
    *,
    period_slug: str,
    database_path: str | Path = DEFAULT_MEMORY_DB_PATH,
) -> StorageResult:
    """Persist a successful strategy-backed pipeline run if eligible.

    Inputs: pipeline result, period slug, and database path.
    Outputs: storage result or skip reason.
    Assumptions: storage never sends memory to Ollama and never stores file blobs.
    """

    database = Path(database_path)
    eligible, reason, _paths, _analysis = _quality_gate(result, period_slug)
    repository = MemoryRepository(database)
    if not eligible:
        return StorageResult(
            stored=False,
            run_id=None,
            database_path=repository.database_path,
            idempotency_key=None,
            table_counts=repository.table_counts(),
            reason=reason,
        )
    payload = build_stored_pipeline_run(result=result, period_slug=period_slug)
    return repository.save_pipeline_run(payload)


def _build_result_for_existing_artifacts(
    *,
    project_root: Path,
    report_path: Path,
    goals_path: Path,
    period_slug: str,
    language: str,
    database_path: Path,
) -> PipelineRunResult:
    """Build a minimal PipelineRunResult for already-generated artifacts.

    Inputs: project root, source files, period slug, language, and DB path.
    Outputs: PipelineRunResult compatible with persist_pipeline_run.
    Assumptions: this helper is for storage backfill, not pipeline execution.
    """

    from finance_agent.orchestration import (  # Local import avoids circular import.
        PipelineConfig,
        PipelineStageResult,
        RuntimeSummary,
        build_pipeline_input_model,
    )

    input_model = build_pipeline_input_model(
        financial_report_path=report_path,
        goals_document_path=goals_path,
        period_override=period_slug.replace("_", "-"),
        report_language=language,
    )
    config = PipelineConfig.from_project_root(
        project_root,
        python_executable="python",
        input_model=input_model,
        memory_database_path=database_path,
    )
    expected = _expected_artifact_paths(
        PipelineRunResult(
            success=True,
            stages=(),
            output_files=(),
            warnings=(),
            runtime_summary=RuntimeSummary(0, 0, 0, 0, 0, 0),
            config=config,
        ),
        period_slug,
    )
    output_files = tuple(str(path) for path in expected.values() if path.is_file())
    stage = PipelineStageResult(
        stage_name="memory_backfill",
        display_name="Memory backfill",
        critical=False,
        success=True,
        skipped=False,
        output_files=output_files,
        warnings=(),
        error=None,
        runtime_seconds=0.0,
    )
    return PipelineRunResult(
        success=True,
        stages=(stage,),
        output_files=output_files,
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=0.0,
            stages_requested=1,
            stages_run=1,
            stages_succeeded=1,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=config,
    )


def main() -> None:
    """Backfill memory storage from existing accepted artifacts.

    Inputs: CLI arguments for report, goals, period slug, and database.
    Outputs: storage result summary.
    Assumptions: artifacts already exist under outputs/ for the period slug.
    """

    import argparse

    project_root = Path(__file__).resolve().parents[2]
    synthetic = project_root / "data" / "synthetic"
    parser = argparse.ArgumentParser(description="Store existing pipeline artifacts in memory DB.")
    parser.add_argument("--period-slug", default="2026_06")
    parser.add_argument(
        "--report",
        type=Path,
        default=synthetic / "monthly_financial_report_june_2026.xlsx",
    )
    parser.add_argument(
        "--goals",
        type=Path,
        default=synthetic / "financial_goals_2026.pdf",
    )
    parser.add_argument("--language", default="es")
    parser.add_argument(
        "--database",
        type=Path,
        default=project_root / "data" / "memory" / "finance_memory.db",
    )
    args = parser.parse_args()
    result = _build_result_for_existing_artifacts(
        project_root=project_root,
        report_path=args.report,
        goals_path=args.goals,
        period_slug=args.period_slug,
        language=args.language,
        database_path=args.database,
    )
    storage = persist_pipeline_run(
        result,
        period_slug=args.period_slug,
        database_path=args.database,
    )
    print(json.dumps(storage.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
