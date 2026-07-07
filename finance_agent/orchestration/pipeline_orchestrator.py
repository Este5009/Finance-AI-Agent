"""Run the existing Finance AI Agent stages in dependency order."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from finance_agent.orchestration.pipeline_models import (
    PipelineConfig,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
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
                config.ollama_model,
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
