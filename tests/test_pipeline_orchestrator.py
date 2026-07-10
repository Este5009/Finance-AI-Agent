"""Tests for the full pipeline orchestrator."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas

from finance_agent.orchestration.pipeline_models import (
    DetectedPeriod,
    PipelineConfig,
    PipelineInputModel,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
)
from finance_agent.orchestration.pipeline_orchestrator import (
    PipelineStage,
    _cache_manifest_path,
    _load_valid_cache,
    _pipeline_cache_key,
    _ollama_client_for_stage,
    _stage_command,
    _structure_fallback_needed,
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


def _input_model(tmp_path: Path) -> PipelineInputModel:
    """Build an execution-ready generic input fixture.

    Inputs: temporary directory.
    Outputs: PipelineInputModel with existing report and goals paths.
    Assumptions: file contents are enough for cache-key tests.
    """

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"report")
    goals.write_bytes(b"goals")
    return PipelineInputModel(
        financial_report_path=report,
        goals_document_path=goals,
        detected_period=DetectedPeriod(
            period_type="monthly",
            label="2026-06",
            confidence=0.9,
            year=2026,
            month=6,
        ),
        period_type="monthly",
        period_override="2026-06",
        report_language="es",
    )


def _valid_report_artifacts(config: PipelineConfig, period_slug: str) -> None:
    """Write minimal strategy-backed report/cache artifacts.

    Inputs: config and period slug.
    Outputs: valid report model, HTML, PDF, analysis, and cache manifest.
    Assumptions: cache tests validate orchestration metadata, not renderer layout.
    """

    report_dir = config.output_directory / "report"
    analysis_dir = config.output_directory / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        {
            "section_id": section_id,
            "title": section_id,
            "content": {},
            "source_references": [],
            "warnings": [],
        }
        for section_id in (
            "cover",
            "executive_summary",
            "financial_health_overview",
            "kpi_overview",
            "revenue_analysis",
            "expense_analysis",
            "department_analysis",
            "anomaly_summary",
            "investigation_evidence",
            "strategic_recommendations",
            "missing_information",
            "appendix",
        )
    ]
    for section in sections:
        if section["section_id"] == "executive_summary":
            section["content"] = {
                "analysis_status": "accepted",
                "summary": "Accepted strategy summary.",
            }
        if section["section_id"] == "strategic_recommendations":
            section["content"] = {
                "recommendations": [{"action": "Act."}],
                "root_causes": [],
                "strategic_priorities": [],
            }
    model = {
        "report_id": "REPORT-MODEL-TEST",
        "period_slug": period_slug,
        "report_period": "2026-06",
        "generated_at_utc": "2026-07-09T00:00:00+00:00",
        "language": "es",
        "section_count": len(sections),
        "sections": sections,
        "source_references": [],
    }
    (report_dir / f"report_model_{period_slug}.json").write_text(
        json.dumps(model),
        encoding="utf-8",
    )
    (report_dir / f"financial_report_{period_slug}.html").write_text(
        "<html>Accepted strategy summary.</html>",
        encoding="utf-8",
    )
    pdf_path = report_dir / f"financial_report_{period_slug}.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "Accepted strategy summary.")
    pdf.save()
    (analysis_dir / f"strategic_analysis_{period_slug}.json").write_text(
        json.dumps({"validation_status": "accepted"}),
        encoding="utf-8",
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
        "cache_hit",
        "cache_key",
    }
    assert data["runtime_summary"]["stages_run"] == 8
    assert data["stages"][0]["stage_name"] == "ingestion"


def test_structure_fallback_skipped_when_confidence_high(tmp_path: Path) -> None:
    """Verify high-confidence deterministic structure does not need Ollama."""

    model = {
        "tables": [
            {
                "table_id": "revenue",
                "detected_type": "Revenue",
                "confidence": 0.95,
                "column_mappings": [
                    {"original_name": "Revenue", "confidence": 0.99},
                ],
                "normalized_columns": ["actual_revenue"],
                "extracted_dimensions": [{"confidence": 0.9}],
                "extracted_metrics": [{"confidence": 0.9}],
            }
        ]
    }

    assert _structure_fallback_needed(model, _config(tmp_path)) is False


def test_structure_fallback_runs_when_uncertainty_exists(tmp_path: Path) -> None:
    """Verify Unknown or low-confidence structure still triggers Ollama fallback."""

    model = {
        "tables": [
            {
                "table_id": "unknown",
                "detected_type": "Unknown",
                "confidence": 0.2,
                "column_mappings": [
                    {"original_name": "Mystery", "confidence": 0.2},
                ],
                "normalized_columns": [],
                "extracted_dimensions": [],
                "extracted_metrics": [],
            }
        ]
    }

    assert _structure_fallback_needed(model, _config(tmp_path)) is True


def test_cache_hit_reuses_valid_outputs(tmp_path: Path) -> None:
    """Verify a valid cache manifest returns a skipped cache-hit result."""

    config = _config(tmp_path)
    input_model = _input_model(tmp_path)
    period_slug = "2026_06"
    _valid_report_artifacts(config, period_slug)
    cache_key = _pipeline_cache_key(input_model, config)
    manifest = _cache_manifest_path(config, cache_key)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"cache_key": cache_key, "period_slug": period_slug}),
        encoding="utf-8",
    )

    result = _load_valid_cache(
        input_model=input_model,
        config=config,
        period_slug=period_slug,
        cache_key=cache_key,
        pipeline_started=0.0,
    )

    assert result is not None
    assert result.cache_hit is True
    assert result.stages[0].skipped is True


def test_cache_invalid_if_strategy_unavailable(tmp_path: Path) -> None:
    """Verify cache is not reused when strategic analysis was not accepted."""

    config = _config(tmp_path)
    input_model = _input_model(tmp_path)
    period_slug = "2026_06"
    _valid_report_artifacts(config, period_slug)
    analysis_path = config.output_directory / "analysis" / f"strategic_analysis_{period_slug}.json"
    analysis_path.write_text(
        json.dumps({"validation_status": "unavailable"}),
        encoding="utf-8",
    )
    cache_key = _pipeline_cache_key(input_model, config)
    manifest = _cache_manifest_path(config, cache_key)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"cache_key": cache_key, "period_slug": period_slug}),
        encoding="utf-8",
    )

    result = _load_valid_cache(
        input_model=input_model,
        config=config,
        period_slug=period_slug,
        cache_key=cache_key,
        pipeline_started=0.0,
    )

    assert result is None


def test_timeout_style_unavailable_strategy_skips_final_report() -> None:
    """Verify rejected strategy behavior is represented as a skipped report stage."""

    result = PipelineRunResult(
        success=True,
        stages=(
            PipelineStageResult(
                stage_name="strategic_analysis",
                display_name="Strategic analysis",
                critical=False,
                success=True,
                skipped=False,
                output_files=(),
                warnings=("Could not reach Ollama before timeout.",),
                error=None,
                runtime_seconds=1.0,
            ),
            PipelineStageResult(
                stage_name="report_generation",
                display_name="Report model and renderers",
                critical=False,
                success=True,
                skipped=True,
                output_files=(),
                warnings=("Skipped final report rendering because strategic analysis was unavailable.",),
                error=None,
                runtime_seconds=0.0,
            ),
        ),
        output_files=(),
        warnings=("Could not reach Ollama before timeout.",),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=1.0,
            stages_requested=2,
            stages_run=1,
            stages_succeeded=1,
            stages_failed=0,
            stages_skipped=1,
        ),
        config=_config(Path(".")),
    )

    assert result.stages[-1].skipped is True
    assert "strategic analysis was unavailable" in result.stages[-1].warnings[0]


def test_stage_specific_model_routing_uses_expected_models(tmp_path: Path) -> None:
    """Verify each Ollama stage receives its configured model."""

    config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        ollama_model="large",
        structure_ollama_model="small-structure",
        planner_ollama_model="small-planner",
        analysis_ollama_model="large-analysis",
    )

    assert _ollama_client_for_stage(config, "ollama_structure_fallback").model == "small-structure"
    assert _ollama_client_for_stage(config, "ollama_investigation_planner").model == "small-planner"
    assert _ollama_client_for_stage(config, "strategic_analysis").model == "large-analysis"


def test_single_model_backward_compatibility_for_stage_routing(tmp_path: Path) -> None:
    """Verify unset stage-specific models fall back to the legacy single model."""

    config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        ollama_model="one-model",
        structure_ollama_model=None,
        planner_ollama_model=None,
        analysis_ollama_model=None,
    )
    stage = PipelineStage(
        name="ollama_structure_fallback",
        display_name="Ollama structure fallback",
        script_name="run_ollama_structure_fallback.py",
        critical=False,
        expected_outputs=(),
        ollama_dependent=True,
    )

    assert config.effective_ollama_models() == {
        "structure_fallback": "one-model",
        "investigation_planner": "one-model",
        "strategic_analysis": "one-model",
    }
    command = _stage_command(stage, config)
    assert command[command.index("--model") + 1] == "one-model"


def test_cache_key_separates_stage_model_combinations(tmp_path: Path) -> None:
    """Verify changing a stage-specific model changes cache identity."""

    input_model = _input_model(tmp_path)
    config_a = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        structure_ollama_model="small-a",
        planner_ollama_model="small",
        analysis_ollama_model="large",
    )
    config_b = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        structure_ollama_model="small-b",
        planner_ollama_model="small",
        analysis_ollama_model="large",
    )

    assert _pipeline_cache_key(input_model, config_a) != _pipeline_cache_key(input_model, config_b)
