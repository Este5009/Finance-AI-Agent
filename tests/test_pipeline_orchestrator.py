"""Tests for the full pipeline orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from finance_agent.orchestration.pipeline_models import PipelineConfig, PipelineStageResult
from finance_agent.orchestration.pipeline_orchestrator import (
    PipelineStage,
    build_default_stages,
    run_full_pipeline,
)


def _config(project_root: Path) -> PipelineConfig:
    """Build a test pipeline config rooted at a temporary directory.

    Inputs: temporary project root.
    Outputs: PipelineConfig with deterministic executable and paths.
    Assumptions: stage execution is mocked in these tests.
    """

    return PipelineConfig.from_project_root(
        project_root,
        python_executable=sys.executable,
        ollama_endpoint="http://localhost:9",
        ollama_timeout_seconds=1.0,
        stage_timeout_seconds=5.0,
    )


def _fake_success(stage: PipelineStage, config: PipelineConfig) -> PipelineStageResult:
    """Return a successful mocked stage result.

    Inputs: stage definition and config.
    Outputs: successful stage result with expected output paths.
    Assumptions: no subprocess is launched.
    """

    return PipelineStageResult(
        stage_name=stage.name,
        display_name=stage.display_name,
        critical=stage.critical,
        success=True,
        skipped=False,
        output_files=tuple(
            str(config.project_root / relative_path)
            for relative_path in stage.expected_outputs
        ),
        warnings=(),
        error=None,
        runtime_seconds=0.01,
        return_code=0,
    )


def test_orchestrator_stage_ordering() -> None:
    """Verify the canonical stage order matches the requested pipeline."""

    assert [stage.name for stage in build_default_stages()] == [
        "ingestion",
        "document_understanding",
        "finance_calculations",
        "anomaly_detection",
        "ollama_structure_fallback",
        "ollama_investigation_planner",
        "retrieval_layer",
        "strategic_analysis",
    ]


def test_successful_full_pipeline_result_uses_current_synthetic_layout(
    tmp_path: Path,
) -> None:
    """Verify a successful run returns stage, output, and runtime summaries."""

    config = _config(tmp_path)

    result = run_full_pipeline(config, stage_executor=_fake_success)

    assert result.success is True
    assert result.runtime_summary.stages_requested == 8
    assert result.runtime_summary.stages_succeeded == 8
    assert result.runtime_summary.stages_failed == 0
    assert any("finance_summary_2026.json" in path for path in result.output_files)
    assert any("monthly_financial_report_june_2026.xlsx" in str(config.monthly_workbook) for _ in [0])


def test_critical_failure_stops_later_stages(tmp_path: Path) -> None:
    """Verify a failed critical stage skips remaining dependency stages."""

    config = _config(tmp_path)
    calls: list[str] = []

    def failing_executor(
        stage: PipelineStage,
        config: PipelineConfig,
    ) -> PipelineStageResult:
        """Fail finance calculations and succeed earlier stages."""

        calls.append(stage.name)
        if stage.name == "finance_calculations":
            return PipelineStageResult(
                stage_name=stage.name,
                display_name=stage.display_name,
                critical=stage.critical,
                success=False,
                skipped=False,
                output_files=(),
                warnings=(),
                error="calculation input missing",
                runtime_seconds=0.01,
                return_code=1,
            )
        return _fake_success(stage, config)

    result = run_full_pipeline(config, stage_executor=failing_executor)

    assert result.success is False
    assert calls == ["ingestion", "document_understanding", "finance_calculations"]
    assert result.runtime_summary.stages_skipped == 5
    assert result.stages[3].skipped is True


def test_noncritical_ollama_fallback_behavior_continues(tmp_path: Path) -> None:
    """Verify non-critical Ollama failure is captured and later stages continue."""

    config = _config(tmp_path)
    calls: list[str] = []

    def executor(stage: PipelineStage, config: PipelineConfig) -> PipelineStageResult:
        """Fail only the non-critical structure fallback stage."""

        calls.append(stage.name)
        if stage.name == "ollama_structure_fallback":
            return PipelineStageResult(
                stage_name=stage.name,
                display_name=stage.display_name,
                critical=stage.critical,
                success=False,
                skipped=False,
                output_files=(),
                warnings=("Ollama unavailable; fallback preserved deterministic model.",),
                error="Ollama unavailable",
                runtime_seconds=0.01,
                return_code=1,
            )
        return _fake_success(stage, config)

    result = run_full_pipeline(config, stage_executor=executor)

    assert result.success is True
    assert calls == [stage.name for stage in build_default_stages()]
    assert result.runtime_summary.stages_failed == 1
    assert any("Ollama unavailable" in warning for warning in result.warnings)


def test_output_summary_structure(tmp_path: Path) -> None:
    """Verify serialized result exposes stable top-level summary fields."""

    result = run_full_pipeline(_config(tmp_path), stage_executor=_fake_success)

    data: dict[str, Any] = result.to_dict()

    assert set(data) == {
        "success",
        "stages",
        "output_files",
        "warnings",
        "runtime_summary",
        "config",
    }
    assert data["runtime_summary"]["stages_run"] == 8
    assert data["stages"][0]["stage_name"] == "ingestion"
