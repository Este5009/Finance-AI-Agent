"""Tests for generic one-report pipeline input workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from finance_agent.orchestration import (
    PipelineConfig,
    build_pipeline_input_model,
    detect_period,
    run_object_pipeline_for_report,
    run_pipeline_for_report,
)
from finance_agent.orchestration.pipeline_models import DetectedPeriod, PipelineInputModel
from finance_agent.orchestration import pipeline_orchestrator
from finance_agent.orchestration.pipeline_orchestrator import PipelineStage
from finance_agent.orchestration.pipeline_models import PipelineStageResult


def _touch(path: Path) -> Path:
    """Create a small placeholder input file.

    Inputs: destination path.
    Outputs: same path.
    Assumptions: detection tests use filenames, not workbook contents.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    return path


def _config(project_root: Path) -> PipelineConfig:
    """Build a test config with synthetic input paths.

    Inputs: temporary project root.
    Outputs: PipelineConfig.
    Assumptions: stage execution is mocked.
    """

    data = project_root / "data" / "synthetic"
    _touch(data / "monthly_financial_report_june_2026.xlsx")
    _touch(data / "annual_financial_report_2026.xlsx")
    _touch(data / "financial_goals_2026.pdf")
    return PipelineConfig.from_project_root(
        project_root,
        python_executable=sys.executable,
        ollama_endpoint="http://localhost:9",
        stage_timeout_seconds=5.0,
    )


def _fake_success(stage: PipelineStage, config: PipelineConfig) -> PipelineStageResult:
    """Return a successful mocked stage result.

    Inputs: stage and config.
    Outputs: successful PipelineStageResult.
    Assumptions: no subprocess is launched.
    """

    return PipelineStageResult(
        stage_name=stage.name,
        display_name=stage.display_name,
        critical=stage.critical,
        success=True,
        skipped=False,
        output_files=tuple(str(config.project_root / path) for path in stage.expected_outputs),
        warnings=(),
        error=None,
        runtime_seconds=0.01,
        return_code=0,
    )


def test_generic_input_model_validation_requires_override_for_unknown(tmp_path: Path) -> None:
    """Verify unknown periods are constructible but not execution-ready."""

    report = _touch(tmp_path / "finance_report.xlsx")
    goals = _touch(tmp_path / "goals.txt")
    input_model = PipelineInputModel(
        financial_report_path=report,
        goals_document_path=goals,
        detected_period=DetectedPeriod("unknown", "Unknown period", 0.1),
        period_type="unknown",
        report_language="es",
    )

    assert input_model.requires_period_override is True
    with pytest.raises(ValueError, match="period_override is required"):
        input_model.validate_for_execution()


def test_monthly_period_detection_from_filename(tmp_path: Path) -> None:
    """Verify monthly reports are detected from filename evidence."""

    detected = detect_period(tmp_path / "monthly_financial_report_june_2026.xlsx")

    assert detected.period_type == "monthly"
    assert detected.year == 2026
    assert detected.month == 6
    assert detected.confidence >= 0.65


def test_annual_period_detection_from_filename(tmp_path: Path) -> None:
    """Verify annual reports are detected from filename evidence."""

    detected = detect_period(tmp_path / "annual_financial_report_2026.xlsx")

    assert detected.period_type == "annual"
    assert detected.year == 2026
    assert detected.label == "2026"


def test_unknown_low_confidence_period_requires_override(tmp_path: Path) -> None:
    """Verify weak evidence is marked unknown and override-required."""

    detected = detect_period(tmp_path / "finance_upload.xlsx")
    model = build_pipeline_input_model(
        financial_report_path=tmp_path / "finance_upload.xlsx",
        goals_document_path=tmp_path / "goals.txt",
        report_language="es",
    )

    assert detected.period_type == "unknown"
    assert model.requires_period_override is True


def test_orchestrator_accepts_generic_synthetic_input(tmp_path: Path) -> None:
    """Verify generic input can run through compatibility orchestration."""

    config = _config(tmp_path)
    input_model = build_pipeline_input_model(
        financial_report_path=config.monthly_workbook,
        goals_document_path=config.goals_pdf,
        report_language="es",
    )

    result = run_pipeline_for_report(
        input_model,
        config,
        stage_executor=_fake_success,
    )

    assert result.success is True
    assert result.config.input_model is input_model
    assert result.config.input_model.report_language == "es"


def test_object_pipeline_entrypoint_is_exported() -> None:
    """Verify the object-based generic pipeline entry point is importable."""

    assert callable(run_object_pipeline_for_report)


def test_generic_non_synthetic_input_routes_to_object_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify arbitrary inputs are no longer rejected as non-synthetic."""

    config = _config(tmp_path)
    report = _touch(tmp_path / "uploads" / "school_report_july_2026.xlsx")
    goals = _touch(tmp_path / "uploads" / "goals_july_2026.txt")
    input_model = build_pipeline_input_model(
        financial_report_path=report,
        goals_document_path=goals,
        period_override="July 2026",
        report_language="es",
    )
    called: dict[str, bool] = {"object_pipeline": False}

    def fake_object_pipeline(model: PipelineInputModel, pipeline_config: PipelineConfig):
        """Return a mocked result and prove the object path was selected."""

        called["object_pipeline"] = True
        return run_full_pipeline_for_test(pipeline_config)

    monkeypatch.setattr(
        pipeline_orchestrator,
        "run_object_pipeline_for_report",
        fake_object_pipeline,
    )

    result = pipeline_orchestrator.run_pipeline_for_report(input_model, config)

    assert called["object_pipeline"] is True
    assert result.success is True


def run_full_pipeline_for_test(config: PipelineConfig):
    """Build a minimal successful PipelineRunResult for route-selection tests."""

    from finance_agent.orchestration.pipeline_models import PipelineRunResult, RuntimeSummary

    stage = PipelineStageResult(
        stage_name="object_pipeline",
        display_name="Object pipeline",
        critical=True,
        success=True,
        skipped=False,
        output_files=(),
        warnings=(),
        error=None,
        runtime_seconds=0.0,
    )
    return PipelineRunResult(
        success=True,
        stages=(stage,),
        output_files=(),
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


def test_backward_compatible_old_workflow_still_works(tmp_path: Path) -> None:
    """Verify default config preserves monthly and annual synthetic paths."""

    config = _config(tmp_path)

    assert config.monthly_workbook.name == "monthly_financial_report_june_2026.xlsx"
    assert config.annual_workbook.name == "annual_financial_report_2026.xlsx"
    assert config.goals_pdf.name == "financial_goals_2026.pdf"
