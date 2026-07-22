"""Run the existing Finance AI Agent stages in dependency order."""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from finance_agent.agent.investigation_planner import build_investigation_plan, save_investigation_plan
from finance_agent.agent.ollama_planner import (
    create_ollama_investigation_plan,
    save_json_artifact as save_plan_json_artifact,
)
from finance_agent.analysis.strategic_analysis import save_json_artifact as save_analysis_json_artifact
from finance_agent.anomalies.anomaly_config import AnomalyThresholds
from finance_agent.anomalies.anomaly_engine import (
    build_risk_summary,
    run_anomaly_detection,
    save_anomaly_report,
    save_risk_summary,
)
from finance_agent.anomalies.anomaly_loader import CalculationOutputBundle
from finance_agent.calculations.calculation_loader import load_intermediate_model
from finance_agent.calculations.finance_engine import (
    FinanceCalculationResult,
    run_finance_calculations,
    save_finance_calculation_outputs,
)
from finance_agent.calculations.periods import PeriodScope
from finance_agent.ingestion.ingestion import extract_goals_pdf, inspect_workbook, load_excel_workbook
from finance_agent.ingestion.schema import clean_column_name
from finance_agent.llm.ollama_client import OllamaClient
from finance_agent.memory.context_builder import (
    HistoricalContextCache,
    build_historical_context,
    save_historical_context,
)
from finance_agent.orchestration.pipeline_models import (
    DetectedPeriod,
    PipelineConfig,
    PipelineInputModel,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
)
from finance_agent.reporting.report_engine import ReportInputBundle, build_report_model, save_report_model
from finance_agent.reporting.report_quality import validate_report_artifacts
from finance_agent.reporting.renderers import render_report_pdf, save_report_html
from finance_agent.reasoning.reasoning_pipeline import create_modular_strategic_analysis
from finance_agent.retrieval.retrieval_engine import (
    RetrievalContext,
    build_retrieval_summary,
    execute_retrieval_queue,
    save_json_artifact as save_retrieval_json_artifact,
)
from finance_agent.understanding.intermediate import build_financial_document_model, save_intermediate_outputs
from finance_agent.understanding.structure_fallback import (
    detect_low_confidence_items,
    enrich_intermediate_model,
    preserve_deterministic_enrichment,
    save_enriched_model,
)


StageExecutor = Callable[["PipelineStage", PipelineConfig], PipelineStageResult]


@dataclass(frozen=True)
class PipelineStage:
    """Definition for one existing script-backed pipeline stage.

    Inputs: stage name, script path, criticality, expected outputs, and extra args.
    Outputs: immutable stage definition consumed by the orchestrator.
    Assumptions: scripts preserve existing behavior and output locations.
    """

    name: str
    display_name: str
    script_name: str
    critical: bool
    expected_outputs: tuple[Path, ...]
    ollama_dependent: bool = False


def _outputs(*parts: str) -> Path:
    """Create a relative output path for stage definitions.

    Inputs: path components under outputs/.
    Outputs: relative Path.
    Assumptions: all generated artifacts remain under outputs/.
    """

    return Path("outputs", *parts)


def build_default_stages() -> tuple[PipelineStage, ...]:
    """Build the canonical stage order for the current pipeline.

    Inputs: none.
    Outputs: ordered stage definitions from ingestion through strategic analysis.
    Assumptions: no PDF/UI/database/email/forecasting stages are included.
    """

    return (
        PipelineStage(
            "ingestion",
            "Document ingestion",
            "run_ingestion.py",
            True,
            (
                _outputs("inspection", "monthly_workbook_inspection.json"),
                _outputs("inspection", "annual_workbook_inspection.json"),
                _outputs("inspection", "goals_text_2026.txt"),
            ),
        ),
        PipelineStage(
            "document_understanding",
            "Document understanding",
            "run_document_understanding.py",
            True,
            (
                _outputs("intermediate", "financial_document_model.json"),
                _outputs("intermediate", "feature_summary.json"),
            ),
        ),
        PipelineStage(
            "finance_calculations",
            "Finance calculations",
            "run_finance_calculations.py",
            True,
            (
                _outputs("calculations", "finance_summary_june_2026.json"),
                _outputs("calculations", "finance_summary_2026.json"),
                _outputs("calculations", "monthly_trends_2026.csv"),
            ),
        ),
        PipelineStage(
            "anomaly_detection",
            "Anomaly detection",
            "run_anomaly_detection.py",
            True,
            (
                _outputs("anomalies", "anomaly_report_june_2026.json"),
                _outputs("anomalies", "anomaly_report_2026.json"),
                _outputs("anomalies", "risk_summary_2026.json"),
            ),
        ),
        PipelineStage(
            "ollama_structure_fallback",
            "Ollama structure fallback",
            "run_ollama_structure_fallback.py",
            False,
            (_outputs("intermediate", "financial_document_model_enriched.json"),),
            True,
        ),
        PipelineStage(
            "ollama_investigation_planner",
            "Ollama investigation planner",
            "run_ollama_planner.py",
            True,
            (
                _outputs("plans", "ollama_plan_june_2026.json"),
                _outputs("plans", "ollama_plan_2026.json"),
                _outputs("plans", "execution_queue_june_2026.json"),
                _outputs("plans", "execution_queue_2026.json"),
            ),
            True,
        ),
        PipelineStage(
            "retrieval_layer",
            "Retrieval layer",
            "run_retrieval_layer.py",
            True,
            (
                _outputs("evidence", "evidence_package_june_2026.json"),
                _outputs("evidence", "evidence_package_2026.json"),
                _outputs("evidence", "retrieval_summary_2026.json"),
            ),
        ),
        PipelineStage(
            "strategic_analysis",
            "Strategic analysis",
            "run_strategic_analysis.py",
            False,
            (
                _outputs("analysis", "strategic_analysis_june_2026.json"),
                _outputs("analysis", "strategic_analysis_2026.json"),
                _outputs("analysis", "analysis_summary_2026.json"),
            ),
            True,
        ),
    )


def _tail_text(text: str, *, limit: int = 1600) -> str:
    """Return a bounded diagnostic tail from process output.

    Inputs: raw output text and character limit.
    Outputs: tail text within limit.
    Assumptions: full stdout/stderr remains available in terminal logs if needed.
    """

    return text[-limit:] if len(text) > limit else text


def _stage_command(stage: PipelineStage, config: PipelineConfig) -> list[str]:
    """Build the subprocess command for one stage.

    Inputs: stage definition and pipeline configuration.
    Outputs: command list suitable for subprocess.run.
    Assumptions: Ollama-dependent scripts accept endpoint/model/timeout arguments.
    """

    command = [
        config.python_executable,
        str(config.project_root / "scripts" / stage.script_name),
    ]
    if stage.ollama_dependent:
        command.extend(
            [
                "--endpoint",
                config.ollama_endpoint,
                "--model",
                config.model_for_stage(stage.name),
                "--timeout",
                str(config.ollama_timeout_seconds),
            ]
        )
    return command


def _existing_outputs(
    stage: PipelineStage,
    config: PipelineConfig,
) -> tuple[str, ...]:
    """Collect expected output files that exist after a stage run.

    Inputs: stage definition and config.
    Outputs: string paths for existing expected artifacts.
    Assumptions: missing expected files are warnings, not hidden successes.
    """

    paths: list[str] = []
    for relative_path in stage.expected_outputs:
        path = config.project_root / relative_path
        if path.exists():
            paths.append(str(path))
    return tuple(paths)


def _output_warnings(
    stage: PipelineStage,
    config: PipelineConfig,
) -> tuple[str, ...]:
    """Create warnings for expected stage outputs that are missing.

    Inputs: stage definition and config.
    Outputs: warning messages.
    Assumptions: subprocess success plus missing outputs still deserves attention.
    """

    warnings: list[str] = []
    for relative_path in stage.expected_outputs:
        path = config.project_root / relative_path
        if not path.exists():
            warnings.append(f"Expected output not found: {path}")
    return tuple(warnings)


def run_stage_subprocess(
    stage: PipelineStage,
    config: PipelineConfig,
) -> PipelineStageResult:
    """Run one stage by invoking its existing CLI script.

    Inputs: stage definition and pipeline configuration.
    Outputs: stage result with status, outputs, warnings, and diagnostics.
    Assumptions: existing scripts implement the stage's business logic.
    """

    started = time.perf_counter()
    command = _stage_command(stage, config)
    try:
        completed = subprocess.run(
            command,
            cwd=config.project_root,
            capture_output=True,
            text=True,
            timeout=config.stage_timeout_seconds,
            check=False,
        )
        runtime = time.perf_counter() - started
        warnings = list(_output_warnings(stage, config))
        if stage.ollama_dependent and "Ollama available: no" in completed.stdout:
            warnings.append("Ollama unavailable; stage used its fail-safe behavior.")
        error = None if completed.returncode == 0 else _tail_text(completed.stderr)
        return PipelineStageResult(
            stage_name=stage.name,
            display_name=stage.display_name,
            critical=stage.critical,
            success=completed.returncode == 0,
            skipped=False,
            output_files=_existing_outputs(stage, config),
            warnings=tuple(warnings),
            error=error,
            runtime_seconds=runtime,
            return_code=completed.returncode,
            stdout_tail=_tail_text(completed.stdout),
            stderr_tail=_tail_text(completed.stderr),
        )
    except (subprocess.SubprocessError, OSError, TimeoutError) as exc:
        runtime = time.perf_counter() - started
        return PipelineStageResult(
            stage_name=stage.name,
            display_name=stage.display_name,
            critical=stage.critical,
            success=False,
            skipped=False,
            output_files=_existing_outputs(stage, config),
            warnings=(),
            error=str(exc),
            runtime_seconds=runtime,
        )


def _skipped_stage_result(stage: PipelineStage) -> PipelineStageResult:
    """Create a skipped result for stages after a critical failure.

    Inputs: stage definition.
    Outputs: skipped stage result.
    Assumptions: skipped stages do not run and have no outputs collected.
    """

    return PipelineStageResult(
        stage_name=stage.name,
        display_name=stage.display_name,
        critical=stage.critical,
        success=False,
        skipped=True,
        output_files=(),
        warnings=("Skipped because an earlier critical stage failed.",),
        error=None,
        runtime_seconds=0.0,
    )


def _json_write(data: dict[str, Any], output_path: Path) -> Path:
    """Write one generic orchestration artifact as JSON.

    Inputs: JSON-compatible data and output path.
    Outputs: written path.
    Assumptions: artifact writes happen only after a stage succeeds.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _stage_result(
    *,
    name: str,
    display: str,
    critical: bool,
    started: float,
    outputs: tuple[Path, ...],
    warnings: tuple[str, ...] = (),
    error: str | None = None,
    telemetry: dict[str, Any] | None = None,
) -> PipelineStageResult:
    """Create one object-pipeline stage result.

    Inputs: stage metadata, start time, outputs, warnings, and optional error.
    Outputs: PipelineStageResult.
    Assumptions: exceptions are converted by the caller into failed stage results.
    """

    return PipelineStageResult(
        stage_name=name,
        display_name=display,
        critical=critical,
        success=error is None,
        skipped=False,
        output_files=tuple(str(path) for path in outputs if path.exists()),
        warnings=warnings,
        error=error,
        runtime_seconds=time.perf_counter() - started,
        return_code=0 if error is None else 1,
        telemetry=telemetry or {},
    )


def _skipped_object_stage_result(
    *,
    name: str,
    display: str,
    critical: bool,
    started: float,
    outputs: tuple[Path, ...] = (),
    warnings: tuple[str, ...] = (),
    telemetry: dict[str, Any] | None = None,
) -> PipelineStageResult:
    """Create a successful skipped result for object-pipeline optimizations.

    Inputs: stage metadata, start time, outputs, and explanatory warnings.
    Outputs: PipelineStageResult marked skipped and successful.
    Assumptions: skipped optimization stages preserve valid downstream artifacts.
    """

    return PipelineStageResult(
        stage_name=name,
        display_name=display,
        critical=critical,
        success=True,
        skipped=True,
        output_files=tuple(str(path) for path in outputs if path.exists()),
        warnings=warnings,
        error=None,
        runtime_seconds=time.perf_counter() - started,
        return_code=0,
        telemetry=telemetry or {},
    )


def _hash_file(path: Path) -> str:
    """Hash one input file in chunks for cache identity.

    Inputs: file path.
    Outputs: SHA-256 hex digest.
    Assumptions: cache keys must be content-based, not mtime-based.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pipeline_cache_key(input_model: PipelineInputModel, config: PipelineConfig) -> str:
    """Build a stable cache key from inputs and model/runtime settings.

    Inputs: generic input model and pipeline config.
    Outputs: SHA-256 cache key.
    Assumptions: identical file contents plus settings can reuse validated outputs.
    """

    payload = {
        "financial_report_sha256": _hash_file(input_model.financial_report_path),
        "goals_document_sha256": _hash_file(input_model.goals_document_path),
        "period_override": input_model.period_override,
        "effective_period_label": input_model.effective_period_label,
        "report_language": input_model.report_language,
        "ollama_endpoint": config.ollama_endpoint,
        "ollama_model": config.ollama_model,
        "effective_ollama_models": config.effective_ollama_models(),
        "structure_thresholds": {
            "table": config.structure_fallback_table_threshold,
            "column": config.structure_fallback_column_threshold,
        },
        "runtime_optimization": {
            "max_planner_anomalies": config.max_planner_anomalies,
            "compact_context": config.compact_context,
            "deduplicate_context": config.deduplicate_context,
        },
        "reasoning_enabled_stages": [
            "ollama_investigation_planner",
            "strategic_analysis",
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_manifest_path(config: PipelineConfig, cache_key: str) -> Path:
    """Return the manifest path for one pipeline cache key.

    Inputs: pipeline config and cache key.
    Outputs: path under outputs/cache.
    Assumptions: cache metadata lives with reproducibility artifacts.
    """

    return config.output_directory / "cache" / f"{cache_key}.json"


def _expected_current_artifacts(config: PipelineConfig, period_slug: str) -> dict[str, Path]:
    """Return quality-critical artifacts expected for one period slug.

    Inputs: pipeline config and period slug.
    Outputs: named artifact paths.
    Assumptions: current generic output filenames are period-slugged.
    """

    outputs = config.output_directory
    return {
        "report_model": outputs / "report" / f"report_model_{period_slug}.json",
        "html": outputs / "report" / f"financial_report_{period_slug}.html",
        "pdf": outputs / "report" / f"financial_report_{period_slug}.pdf",
        "strategic_analysis": outputs / "analysis" / f"strategic_analysis_{period_slug}.json",
    }


def _structure_fallback_needed(
    model: dict[str, Any],
    config: PipelineConfig,
) -> bool:
    """Return whether structure fallback should call Ollama for this model.

    Inputs: intermediate model and pipeline config thresholds.
    Outputs: True when low-confidence tables/columns need LLM interpretation.
    Assumptions: no low-confidence items means deterministic structure is sufficient.
    """

    return bool(
        detect_low_confidence_items(
            model,
            table_threshold=config.structure_fallback_table_threshold,
            column_threshold=config.structure_fallback_column_threshold,
        )
    )


def _ollama_client_for_stage(config: PipelineConfig, stage_name: str) -> OllamaClient:
    """Create an Ollama client using the configured model for one stage.

    Inputs: pipeline config and Ollama-dependent stage name.
    Outputs: OllamaClient with shared endpoint/timeout and stage-specific model.
    Assumptions: all Ollama logic remains inside existing stage modules.
    """

    return OllamaClient(
        endpoint=config.ollama_endpoint,
        model=config.model_for_stage(stage_name),
        timeout_seconds=config.read_timeout_seconds,
        connect_timeout_seconds=config.connect_timeout_seconds,
        read_timeout_seconds=config.read_timeout_seconds,
        keep_alive=config.ollama_keep_alive,
        reasoning_enabled=stage_name
        in {"ollama_investigation_planner", "planner", "strategic_analysis", "analysis"},
    )


def _load_valid_cache(
    *,
    input_model: PipelineInputModel,
    config: PipelineConfig,
    period_slug: str,
    cache_key: str,
    pipeline_started: float,
) -> PipelineRunResult | None:
    """Return a cache-hit result when previous outputs are valid and current.

    Inputs: input/config, period slug, cache key, and run start time.
    Outputs: PipelineRunResult or None for cache miss.
    Assumptions: cached outputs are reusable only if report quality still passes.
    """

    del input_model  # The cache key already embeds the execution-relevant input fields.
    manifest_path = _cache_manifest_path(config, cache_key)
    artifacts = _expected_current_artifacts(config, period_slug)
    if not manifest_path.is_file() or not all(path.is_file() for path in artifacts.values()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis = json.loads(artifacts["strategic_analysis"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("cache_key") != cache_key:
        return None
    if analysis.get("validation_status") != "accepted":
        return None
    quality = validate_report_artifacts(
        artifacts["report_model"],
        html_path=artifacts["html"],
        pdf_path=artifacts["pdf"],
    )
    if not quality.is_valid:
        return None
    stage = _skipped_object_stage_result(
        name="pipeline_cache",
        display="Pipeline cache",
        critical=False,
        started=pipeline_started,
        outputs=tuple(artifacts.values()),
        warnings=("Cache hit; reused validated artifacts.",),
    )
    return PipelineRunResult(
        success=True,
        stages=(stage,),
        output_files=tuple(str(path) for path in artifacts.values()),
        warnings=stage.warnings,
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=time.perf_counter() - pipeline_started,
            stages_requested=1,
            stages_run=0,
            stages_succeeded=0,
            stages_failed=0,
            stages_skipped=1,
        ),
        config=config,
        cache_hit=True,
        cache_key=cache_key,
    )


def _write_cache_manifest(
    *,
    result: PipelineRunResult,
    cache_key: str,
    period_slug: str,
) -> None:
    """Persist a cache manifest for a successful quality-backed run.

    Inputs: completed result, cache key, and period slug.
    Outputs: manifest JSON written under outputs/cache.
    Assumptions: invalid strategy/report outputs must not be cached.
    """

    artifacts = _expected_current_artifacts(result.config, period_slug)
    if not result.success or not all(path.is_file() for path in artifacts.values()):
        return
    try:
        analysis = json.loads(artifacts["strategic_analysis"].read_text(encoding="utf-8"))
        quality = validate_report_artifacts(
            artifacts["report_model"],
            html_path=artifacts["html"],
            pdf_path=artifacts["pdf"],
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if analysis.get("validation_status") != "accepted" or not quality.is_valid:
        return
    manifest_path = _cache_manifest_path(result.config, cache_key)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "period_slug": period_slug,
                "output_files": list(result.output_files),
                "cached_at_epoch": time.time(),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _safe_period_slug(input_model: PipelineInputModel) -> str:
    """Build a filename-safe slug for one generic report run.

    Inputs: generic pipeline input model.
    Outputs: slug used in output artifact names.
    Assumptions: overrides are authoritative; otherwise detected metadata is used.
    """

    if input_model.period_override:
        return clean_column_name(input_model.period_override)
    detected = input_model.detected_period
    if detected.period_type == "monthly" and detected.year and detected.month:
        return f"{detected.year}_{detected.month:02d}"
    if detected.period_type == "annual" and detected.year:
        return str(detected.year)
    if detected.period_type == "quarterly" and detected.year and detected.quarter:
        return f"{detected.year}_q{detected.quarter}"
    if detected.period_type == "semester" and detected.year and detected.semester:
        return f"{detected.year}_s{detected.semester}"
    return clean_column_name(detected.label) or clean_column_name(input_model.financial_report_path.stem)


def _period_scope_from_detected(detected: DetectedPeriod, label: str) -> tuple[PeriodScope | None, int | None]:
    """Convert detected period metadata to calculation scope.

    Inputs: detected period and display label.
    Outputs: optional PeriodScope and optional monthly-trend year.
    Assumptions: quarterly and semester periods are custom date ranges.
    """

    if detected.period_type == "monthly" and detected.year and detected.month:
        return PeriodScope.monthly(detected.year, detected.month, label=label), None
    if detected.period_type == "annual" and detected.year:
        return PeriodScope.annual(detected.year, label=label), detected.year
    if detected.period_type == "quarterly" and detected.year and detected.quarter:
        start_month = (detected.quarter - 1) * 3 + 1
        end_month = start_month + 2
        end_day = 31 if end_month in {1, 3, 5, 7, 8, 10, 12} else 30
        if end_month == 2:
            end_day = 29 if detected.year % 4 == 0 else 28
        return (
            PeriodScope.custom(
                date(detected.year, start_month, 1),
                date(detected.year, end_month, end_day),
                label=label,
            ),
            None,
        )
    if detected.period_type == "semester" and detected.year and detected.semester:
        start_month = 1 if detected.semester == 1 else 7
        end_month = 6 if detected.semester == 1 else 12
        return (
            PeriodScope.custom(
                date(detected.year, start_month, 1),
                date(detected.year, end_month, 30 if end_month == 6 else 31),
                label=label,
            ),
            None,
        )
    if detected.period_type == "custom" and detected.start_date and detected.end_date:
        return (
            PeriodScope.custom(
                date.fromisoformat(detected.start_date),
                date.fromisoformat(detected.end_date),
                label=label,
            ),
            None,
        )
    return None, None


def _calculation_bundle_from_result(
    result: FinanceCalculationResult,
    paths: dict[str, Path],
    *,
    period_slug: str,
) -> CalculationOutputBundle:
    """Adapt an in-memory calculation result to anomaly detector bundle shape.

    Inputs: calculation result, saved artifact paths, and period slug.
    Outputs: CalculationOutputBundle.
    Assumptions: no file reload is needed because DataFrames and JSON are in memory.
    """

    finance_document = json.loads(paths["finance_summary"].read_text(encoding="utf-8"))
    return CalculationOutputBundle(
        period_slug=period_slug,
        finance_summary_path=str(paths["finance_summary"]),
        kpi_summary_path=str(paths["kpi_summary"]),
        department_summary_path=str(paths["department_summary"]),
        category_summary_path=str(paths["category_summary"]),
        monthly_trends_path=str(paths["monthly_trends"]) if "monthly_trends" in paths else None,
        finance_document=finance_document,
        kpi_summary=result.kpi_summary,
        department_summary=result.department_summary,
        category_summary=result.category_summary,
        monthly_trends=result.monthly_trends,
    )


def _records(dataframe: Any) -> tuple[dict[str, Any], ...]:
    """Convert a DataFrame-like object to row dictionaries.

    Inputs: pandas DataFrame.
    Outputs: tuple of JSON-compatible records.
    Assumptions: pandas handles missing-value conversion through JSON export.
    """

    if getattr(dataframe, "empty", True):
        return ()
    return tuple(json.loads(dataframe.to_json(orient="records")))


def _make_retrieval_context(
    *,
    config: PipelineConfig,
    finance_document: dict[str, Any],
    monthly_trends: tuple[dict[str, Any], ...],
    enriched_model: dict[str, Any],
    normalized_table_dir: Path,
    period_slug: str,
    source_prefix: str,
    finance_summary_source: str | None = None,
) -> RetrievalContext:
    """Build a retrieval context for a generic single-report run.

    Inputs: processed objects, normalized table directory, period slug, prefix, and summary artifact.
    Outputs: RetrievalContext compatible with existing retrieval functions.
    Assumptions: single-report runs reuse the same document for monthly/annual slots.
    """

    return RetrievalContext(
        project_root=config.project_root,
        finance_summary_june=finance_document,
        finance_summary_annual=finance_document,
        monthly_trends=monthly_trends,
        enriched_model=enriched_model,
        normalized_table_dir=normalized_table_dir,
        scope_prefix_by_period={
            period_slug: source_prefix,
            "2026": source_prefix,
            "june_2026": source_prefix,
        },
        finance_summary_by_period={
            period_slug: finance_document,
        },
        finance_summary_source_by_period={
            period_slug: finance_summary_source
            or f"outputs/calculations/finance_summary_{period_slug}.json",
        },
    )


def _finalize_pipeline_result(
    *,
    config: PipelineConfig,
    stages: list[PipelineStageResult],
    started: float,
    cache_key: str | None = None,
) -> PipelineRunResult:
    """Build the final structured result for object-based execution.

    Inputs: config, accumulated stages, and pipeline start time.
    Outputs: PipelineRunResult.
    Assumptions: object pipeline runs no skipped stages after handled non-critical failures.
    """

    outputs = tuple(dict.fromkeys(path for stage in stages for path in stage.output_files))
    warnings = tuple(warning for stage in stages for warning in stage.warnings)
    runtime = RuntimeSummary(
        total_runtime_seconds=time.perf_counter() - started,
        stages_requested=len(stages),
        stages_run=len(stages),
        stages_succeeded=sum(stage.success for stage in stages),
        stages_failed=sum(not stage.success for stage in stages),
        stages_skipped=0,
    )
    return PipelineRunResult(
        success=not any(stage.critical and not stage.success for stage in stages),
        stages=tuple(stages),
        output_files=outputs,
        warnings=warnings,
        runtime_summary=runtime,
        config=config,
        cache_hit=False,
        cache_key=cache_key,
    )


def run_full_pipeline(
    config: PipelineConfig,
    *,
    stages: tuple[PipelineStage, ...] | None = None,
    stage_executor: StageExecutor = run_stage_subprocess,
) -> PipelineRunResult:
    """Run all existing pipeline stages in dependency order.

    Inputs: pipeline configuration, optional stage list, and stage executor.
    Outputs: structured PipelineRunResult.
    Assumptions: critical failures stop later stages; non-critical failures are captured.
    """

    requested_stages = stages or build_default_stages()
    started = time.perf_counter()
    results: list[PipelineStageResult] = []
    stop_after_critical_failure = False
    for stage in requested_stages:
        if stop_after_critical_failure:
            results.append(_skipped_stage_result(stage))
            continue
        result = stage_executor(stage, config)
        results.append(result)
        # Non-critical failures are retained as warnings in the run result. A
        # critical stage failure stops the dependency chain to avoid stale outputs.
        if stage.critical and not result.success:
            stop_after_critical_failure = True

    total_runtime = time.perf_counter() - started
    output_files = tuple(
        dict.fromkeys(
            output_file
            for result in results
            for output_file in result.output_files
        )
    )
    warnings = tuple(
        warning
        for result in results
        for warning in result.warnings
    )
    runtime_summary = RuntimeSummary(
        total_runtime_seconds=total_runtime,
        stages_requested=len(requested_stages),
        stages_run=sum(not result.skipped for result in results),
        stages_succeeded=sum(result.success for result in results),
        stages_failed=sum(
            not result.success and not result.skipped for result in results
        ),
        stages_skipped=sum(result.skipped for result in results),
    )
    success = not any(
        result.critical and not result.success
        for result in results
        if not result.skipped
    )
    return PipelineRunResult(
        success=success,
        stages=tuple(results),
        output_files=output_files,
        warnings=warnings,
        runtime_summary=runtime_summary,
        config=config,
    )


def _is_legacy_synthetic_input(input_model: PipelineInputModel, config: PipelineConfig) -> bool:
    """Return whether one generic input belongs to the current synthetic demo set.

    Inputs: generic input model and pipeline config.
    Outputs: True when the existing two-workbook stage scripts can safely run.
    Assumptions: current scripts still process the stable synthetic monthly/annual pair.
    """

    report = input_model.financial_report_path.resolve()
    goals = input_model.goals_document_path.resolve()
    return (
        report in {config.monthly_workbook.resolve(), config.annual_workbook.resolve()}
        and goals == config.goals_pdf.resolve()
    )


def run_pipeline_for_report(
    input_model: PipelineInputModel,
    config: PipelineConfig,
    *,
    stages: tuple[PipelineStage, ...] | None = None,
    stage_executor: StageExecutor = run_stage_subprocess,
) -> PipelineRunResult:
    """Run the object-based pipeline from the generic one-report input contract.

    Inputs: generic input model, base config, optional stages, and executor.
    Outputs: structured PipelineRunResult.
    Assumptions: compatibility arguments are retained; this path owns pipeline state.
    """

    input_model.validate_for_execution()
    config_with_input = PipelineConfig(
        project_root=config.project_root,
        python_executable=config.python_executable,
        data_directory=config.data_directory,
        output_directory=config.output_directory,
        monthly_workbook=config.monthly_workbook,
        annual_workbook=config.annual_workbook,
        goals_pdf=config.goals_pdf,
        ollama_endpoint=config.ollama_endpoint,
        ollama_model=config.ollama_model,
        structure_ollama_model=config.structure_ollama_model,
        planner_ollama_model=config.planner_ollama_model,
        analysis_ollama_model=config.analysis_ollama_model,
        ollama_timeout_seconds=config.ollama_timeout_seconds,
        stage_timeout_seconds=config.stage_timeout_seconds,
        input_model=input_model,
        structure_fallback_table_threshold=config.structure_fallback_table_threshold,
        structure_fallback_column_threshold=config.structure_fallback_column_threshold,
        enable_cache=config.enable_cache,
        allow_draft_report=config.allow_draft_report,
        max_planner_anomalies=config.max_planner_anomalies,
        compact_context=config.compact_context,
        deduplicate_context=config.deduplicate_context,
        enable_memory_storage=config.enable_memory_storage,
        memory_database_path=config.memory_database_path,
    )
    if stages is not None or stage_executor is not run_stage_subprocess:
        # Tests and legacy callers can still exercise the script-backed path with
        # mocks. Real generic execution below does not launch stage scripts.
        if not _is_legacy_synthetic_input(input_model, config):
            raise NotImplementedError("Mocked custom stages are supported only for synthetic compatibility inputs.")
        return run_full_pipeline(
            config_with_input,
            stages=stages,
            stage_executor=stage_executor,
        )
    return run_object_pipeline_for_report(input_model, config_with_input)


def run_object_pipeline_for_report(
    input_model: PipelineInputModel,
    config: PipelineConfig,
) -> PipelineRunResult:
    """Execute all pipeline stages with orchestrator-owned Python objects.

    Inputs: generic input model and pipeline configuration.
    Outputs: structured PipelineRunResult with disk artifacts for reproducibility.
    Assumptions: business algorithms remain in their existing modules.
    """

    input_model.validate_for_execution()
    pipeline_started = time.perf_counter()
    stages: list[PipelineStageResult] = []
    outputs = config.output_directory
    period_slug = _safe_period_slug(input_model)
    cache_key = _pipeline_cache_key(input_model, config) if config.enable_cache else None
    if cache_key:
        cached_result = _load_valid_cache(
            input_model=input_model,
            config=config,
            period_slug=period_slug,
            cache_key=cache_key,
            pipeline_started=pipeline_started,
        )
        if cached_result is not None:
            if config.enable_memory_storage:
                try:
                    from finance_agent.memory.run_storage import persist_pipeline_run

                    persist_pipeline_run(
                        cached_result,
                        period_slug=period_slug,
                        database_path=config.memory_database_path
                        or config.project_root / "data" / "memory" / "finance_memory.db",
                    )
                except Exception as exc:  # noqa: BLE001 - cache reuse remains valid.
                    print(f"Memory storage skipped after cache hit: {exc}")
            return cached_result
    report_label = input_model.effective_period_label
    report_prefix = clean_column_name(input_model.financial_report_path.stem)
    source_workbook = str(input_model.financial_report_path.resolve())

    try:
        started = time.perf_counter()
        workbook = load_excel_workbook(input_model.financial_report_path, header_row=4)
        inspection = inspect_workbook(workbook)
        goals = extract_goals_pdf(input_model.goals_document_path)
        inspection_dir = outputs / "inspection"
        inspection_path = _json_write(inspection, inspection_dir / f"workbook_inspection_{period_slug}.json")
        goals_path = inspection_dir / f"goals_text_{period_slug}.txt"
        goals_path.parent.mkdir(parents=True, exist_ok=True)
        goals_path.write_text(goals.raw_text, encoding="utf-8")
        stages.append(
            _stage_result(
                name="ingestion",
                display="Document ingestion",
                critical=True,
                started=started,
                outputs=(inspection_path, goals_path),
            )
        )

        started = time.perf_counter()
        intermediate_dir = outputs / "intermediate" / period_slug
        model = build_financial_document_model([input_model.financial_report_path])
        intermediate_paths = save_intermediate_outputs(model, intermediate_dir)
        model_path = intermediate_paths["financial_document_model"]
        loaded_model = load_intermediate_model(model_path)
        stages.append(
            _stage_result(
                name="document_understanding",
                display="Document understanding",
                critical=True,
                started=started,
                outputs=(
                    intermediate_paths["financial_document_model"],
                    intermediate_paths["feature_summary"],
                ),
            )
        )

        started = time.perf_counter()
        if _structure_fallback_needed(model.to_dict(), config):
            structure_client = _ollama_client_for_stage(
                config,
                "ollama_structure_fallback",
            )
            enriched_model, fallback_summary = enrich_intermediate_model(
                model.to_dict(),
                structure_client,
                table_threshold=config.structure_fallback_table_threshold,
                column_threshold=config.structure_fallback_column_threshold,
            )
            skipped_structure_fallback = False
        else:
            enriched_model = preserve_deterministic_enrichment(model.to_dict())
            fallback_summary = None
            skipped_structure_fallback = True
        enriched_path = save_enriched_model(
            enriched_model,
            intermediate_dir / "financial_document_model_enriched.json",
        )
        if skipped_structure_fallback:
            stages.append(
                _skipped_object_stage_result(
                    name="ollama_structure_fallback",
                    display="Ollama structure fallback",
                    critical=False,
                    started=started,
                    outputs=(enriched_path,),
                    warnings=("Skipped; deterministic structure was high-confidence.",),
                    telemetry={
                        "python_preprocessing_time_seconds": time.perf_counter() - started,
                        "skipped_reason": "high_confidence_deterministic_structure",
                    },
                )
            )
        else:
            stages.append(
                _stage_result(
                    name="ollama_structure_fallback",
                    display="Ollama structure fallback",
                    critical=False,
                    started=started,
                    outputs=(enriched_path,),
                    warnings=(
                        ()
                        if fallback_summary and fallback_summary.ollama_available
                        else ("Ollama unavailable; deterministic structure was preserved.",)
                    ),
                    telemetry=fallback_summary.telemetry if fallback_summary else {},
                )
            )

        started = time.perf_counter()
        scope, monthly_trend_year = _period_scope_from_detected(
            input_model.detected_period,
            report_label,
        )
        calculation = run_finance_calculations(
            loaded_model,
            source_workbook=source_workbook,
            report_period=report_label,
            period_scope=scope,
            monthly_trend_year=monthly_trend_year,
        )
        calculation_paths = save_finance_calculation_outputs(
            calculation,
            outputs / "calculations",
            period_slug=period_slug,
        )
        finance_document = json.loads(calculation_paths["finance_summary"].read_text(encoding="utf-8"))
        calculation_bundle = _calculation_bundle_from_result(
            calculation,
            calculation_paths,
            period_slug=period_slug,
        )
        stages.append(
            _stage_result(
                name="finance_calculations",
                display="Finance calculations",
                critical=True,
                started=started,
                outputs=tuple(calculation_paths.values()),
                warnings=tuple(calculation.calculation_warnings),
            )
        )

        started = time.perf_counter()
        anomaly_report = run_anomaly_detection(
            calculation_bundle,
            thresholds=AnomalyThresholds(),
            include_trends=not calculation.monthly_trends.empty,
            include_statistics=not calculation.monthly_trends.empty,
            anomaly_id_prefix=f"ANOM-{period_slug.upper().replace('_', '-')}",
        )
        anomaly_paths = save_anomaly_report(anomaly_report, outputs / "anomalies")
        risk_summary = build_risk_summary(anomaly_report)
        risk_path = save_risk_summary(
            anomaly_report,
            outputs / "anomalies" / f"risk_summary_{period_slug}.json",
        )
        anomaly_document = anomaly_report.to_dict()
        stages.append(
            _stage_result(
                name="anomaly_detection",
                display="Anomaly detection",
                critical=True,
                started=started,
                outputs=(*anomaly_paths.values(), risk_path),
            )
        )

        started = time.perf_counter()
        trend_records = _records(calculation.monthly_trends)
        history_cache = HistoricalContextCache()
        history_dir = outputs / "history_reasoning"
        planner_history = build_historical_context(
            current_period=period_slug,
            finance_summary=finance_document,
            anomaly_report=anomaly_document,
            database_path=config.memory_database_path
            or config.project_root / "data" / "memory" / "finance_memory.db",
            purpose="planner",
            cache=history_cache,
        )
        planner_context_path = save_historical_context(
            planner_history.context,
            history_dir / "planner_context.json",
        )
        baseline_plan = build_investigation_plan(
            finance_document=finance_document,
            anomaly_report=anomaly_document,
            monthly_trends=trend_records,
            recurrence_anomalies=anomaly_document.get("anomalies", []),
            enriched_model=enriched_model,
            risk_summary=risk_summary,
            period_slug=period_slug,
            source_files=(Path(source_workbook).name,),
        )
        plan_dir = outputs / "plans"
        baseline_path = save_investigation_plan(
            baseline_plan,
            plan_dir / f"investigation_plan_{period_slug}.json",
        )
        planner_client = _ollama_client_for_stage(
            config,
            "ollama_investigation_planner",
        )
        planner_result = create_ollama_investigation_plan(
            client=planner_client,
            finance_document=finance_document,
            anomaly_report=anomaly_document,
            risk_summary=risk_summary,
            enriched_model=enriched_model,
            baseline_plan=baseline_plan,
            period_slug=period_slug,
            max_anomalies=config.max_planner_anomalies,
            compact_context=config.compact_context,
            deduplicate_context=config.deduplicate_context,
            historical_context=planner_history.context,
        )
        ollama_plan_path = save_plan_json_artifact(
            planner_result.plan_document,
            plan_dir / f"ollama_plan_{period_slug}.json",
        )
        queue_path = save_plan_json_artifact(
            planner_result.execution_queue,
            plan_dir / f"execution_queue_{period_slug}.json",
        )
        stages.append(
            _stage_result(
                name="ollama_investigation_planner",
                display="Ollama investigation planner",
                critical=True,
                started=started,
                outputs=(planner_context_path, baseline_path, ollama_plan_path, queue_path),
                warnings=tuple(planner_result.validation_errors) if planner_result.fallback_used else (),
                telemetry={
                    **(planner_result.telemetry or {}),
                    "historical_context": planner_history.telemetry,
                },
            )
        )

        started = time.perf_counter()
        retrieval_context = _make_retrieval_context(
            config=config,
            finance_document=finance_document,
            monthly_trends=trend_records,
            enriched_model=enriched_model,
            normalized_table_dir=Path(intermediate_paths["normalized_tables"]),
            period_slug=period_slug,
            source_prefix=report_prefix,
            finance_summary_source=str(calculation_paths["finance_summary"]),
        )
        evidence_package = execute_retrieval_queue(
            planner_result.execution_queue,
            retrieval_context,
        )
        evidence_dir = outputs / "evidence"
        evidence_path = save_retrieval_json_artifact(
            evidence_package,
            evidence_dir / f"evidence_package_{period_slug}.json",
        )
        retrieval_summary = build_retrieval_summary((evidence_package,))
        retrieval_summary_path = save_retrieval_json_artifact(
            retrieval_summary,
            evidence_dir / f"retrieval_summary_{period_slug}.json",
        )
        stages.append(
            _stage_result(
                name="retrieval_layer",
                display="Retrieval layer",
                critical=True,
                started=started,
                outputs=(evidence_path, retrieval_summary_path),
            )
        )

        started = time.perf_counter()
        analysis_client = _ollama_client_for_stage(
            config,
            "strategic_analysis",
        )
        strategic_history = build_historical_context(
            current_period=period_slug,
            finance_summary=finance_document,
            anomaly_report=anomaly_document,
            evidence_package=evidence_package,
            database_path=config.memory_database_path
            or config.project_root / "data" / "memory" / "finance_memory.db",
            purpose="strategic_analysis",
            cache=history_cache,
        )
        strategic_context_path = save_historical_context(
            strategic_history.context,
            history_dir / "strategic_context.json",
        )
        analysis_result = create_modular_strategic_analysis(
            client=analysis_client,
            evidence_package=evidence_package,
            finance_summary=finance_document,
            anomaly_report=anomaly_document,
            risk_summary=risk_summary,
            period_slug=period_slug,
            compact_context=config.compact_context,
            deduplicate_context=config.deduplicate_context,
            historical_context=strategic_history.context,
            stage_timeout_seconds=config.stage_timeout_seconds,
        )
        analysis_dir = outputs / "analysis"
        analysis_path = save_analysis_json_artifact(
            analysis_result.analysis_document,
            analysis_dir / f"strategic_analysis_{period_slug}.json",
        )
        reasoning_state = analysis_result.analysis_document.get("reasoning_state", {})
        reasoning_outputs = (
            reasoning_state.get("reasoning_outputs", {})
            if isinstance(reasoning_state, dict)
            else {}
        )
        financial_reasoning_path = save_analysis_json_artifact(
            {
                "period_slug": period_slug,
                "stage_id": "financial_performance",
                "reasoning_output": reasoning_outputs.get("financial_performance", {}),
            },
            analysis_dir / f"financial_reasoning_{period_slug}.json",
        )
        historical_reasoning_path = save_analysis_json_artifact(
            {
                "period_slug": period_slug,
                "stage_id": "historical_operational",
                "reasoning_output": reasoning_outputs.get("historical_operational", {}),
            },
            analysis_dir / f"historical_reasoning_{period_slug}.json",
        )
        strategic_reasoning_path = save_analysis_json_artifact(
            {
                "period_slug": period_slug,
                "stage_id": "strategic_synthesis",
                "reasoning_output": reasoning_outputs.get("strategic_synthesis", {}),
            },
            analysis_dir / f"strategic_reasoning_{period_slug}.json",
        )
        reasoning_state_path = save_analysis_json_artifact(
            reasoning_state if isinstance(reasoning_state, dict) else {},
            analysis_dir / f"reasoning_state_{period_slug}.json",
        )
        stages.append(
            _stage_result(
                name="strategic_analysis",
                display="Strategic analysis",
                critical=False,
                started=started,
                outputs=(
                    strategic_context_path,
                    financial_reasoning_path,
                    historical_reasoning_path,
                    strategic_reasoning_path,
                    reasoning_state_path,
                    analysis_path,
                ),
                warnings=tuple(analysis_result.validation_errors) if not analysis_result.accepted else (),
                telemetry={
                    **(analysis_result.telemetry or {}),
                    "historical_context": strategic_history.telemetry,
                },
            )
        )

        started = time.perf_counter()
        report_dir = outputs / "report"
        if not analysis_result.accepted and not config.allow_draft_report:
            stages.append(
                _skipped_object_stage_result(
                    name="report_generation",
                    display="Report model and renderers",
                    critical=False,
                    started=started,
                    warnings=(
                        "Skipped final report rendering because strategic analysis "
                        "was unavailable and draft reports are disabled.",
                    ),
                )
            )
        else:
            report_inputs = ReportInputBundle(
                period_slug=period_slug,
                finance_summary=finance_document,
                kpi_summary=tuple(json.loads(calculation.kpi_summary.to_json(orient="records"))),
                anomaly_report=anomaly_document,
                evidence_package=evidence_package,
                strategic_analysis=analysis_result.analysis_document,
                source_files=(
                    str(calculation_paths["finance_summary"]),
                    str(calculation_paths["kpi_summary"]),
                    str(anomaly_paths["json"]),
                    str(evidence_path),
                    str(analysis_path),
                ),
            )
            report_model = build_report_model(report_inputs)
            report_model_path = save_report_model(
                report_model,
                report_dir / f"report_model_{period_slug}.json",
            )
            html_path = save_report_html(
                report_model.to_dict(),
                report_dir / f"financial_report_{period_slug}.html",
            )
            pdf_path = render_report_pdf(
                report_model.to_dict(),
                report_dir / f"financial_report_{period_slug}.pdf",
            )
            stages.append(
                _stage_result(
                    name="report_generation",
                    display="Report model and renderers",
                    critical=False,
                    started=started,
                    outputs=(report_model_path, html_path, pdf_path),
                    warnings=(
                        ("Strategic analysis was unavailable; report rendered as draft.",)
                        if not analysis_result.accepted
                        else ()
                    ),
                )
            )
    except Exception as exc:  # noqa: BLE001 - produce structured failure for UI.
        stages.append(
            _stage_result(
                name="pipeline_error",
                display="Pipeline error",
                critical=True,
                started=time.perf_counter(),
                outputs=(),
                error=str(exc),
            )
        )

    result = _finalize_pipeline_result(
        config=config,
        stages=stages,
        started=pipeline_started,
        cache_key=cache_key,
    )
    if cache_key:
        _write_cache_manifest(
            result=result,
            cache_key=cache_key,
            period_slug=period_slug,
        )
    if result.success and config.enable_memory_storage:
        try:
            from finance_agent.memory.run_storage import persist_pipeline_run

            persist_pipeline_run(
                result,
                period_slug=period_slug,
                database_path=config.memory_database_path
                or config.project_root / "data" / "memory" / "finance_memory.db",
            )
        except Exception as exc:  # noqa: BLE001 - storage must not corrupt outputs.
            # Historical storage is post-run persistence. The report artifacts
            # remain valid even if SQLite is unavailable, so preserve existing
            # pipeline behavior and surface the error through a warning-like print.
            print(f"Memory storage skipped after pipeline run: {exc}")
    return result
