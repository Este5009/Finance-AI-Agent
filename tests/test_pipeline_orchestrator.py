"""Tests for the full pipeline orchestrator."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import pytest
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
    _is_degraded_strategy_document,
    _is_ai_backed_strategy_document,
    _pipeline_cache_key,
    _ollama_client_for_stage,
    _pending_after_failure_stage_results,
    _stage_command,
    _structure_fallback_needed,
    build_default_stages,
    run_full_pipeline,
    run_pipeline_for_report,
)
from finance_agent.orchestration.profiling import build_pipeline_profile


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
    Outputs: PipelineInputModel with an existing integrated workbook path.
    Assumptions: file contents are enough for cache-key tests.
    """

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    report.write_bytes(b"report")
    return PipelineInputModel(
        workbook_path=report,
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
        json.dumps(
            {
                "validation_status": "accepted",
                "analysis_source": "ollama_modular_reasoning",
                "ai_usage": {
                    "ollama_called": True,
                    "model": "qwen3:30b-a3b",
                    "model_calls": 3,
                    "successful_responses": 3,
                    "accepted_responses": 3,
                    "rejected_responses": 0,
                    "final_report_fields_with_ai_output": ["executive_summary"],
                },
                "section_provenance": {
                    "executive_summary": {
                        "generated_by": "ollama",
                        "model": "qwen3:30b-a3b",
                        "generation_stage": "strategic_synthesis",
                        "validation_status": "validated",
                    }
                },
            }
        ),
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
    assert "telemetry" in data["stages"][0]
    assert data["config"]["max_planner_anomalies"] == 5
    assert data["config"]["compact_context"] is True
    assert data["config"]["deduplicate_context"] is True


def test_pending_stages_after_historical_context_failure_are_explicit() -> None:
    """Verify failed object-pipeline diagnostics name skipped downstream stages."""

    pending = _pending_after_failure_stage_results("historical_context")

    assert [stage.stage_name for stage in pending[:3]] == [
        "ollama_investigation_planner",
        "retrieval_layer",
        "strategic_analysis",
    ]
    assert all(stage.skipped for stage in pending)
    assert all(stage.success for stage in pending)
    assert all(stage.warnings == ("No ejecutado por fallo previo.",) for stage in pending)


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
    analysis = json.loads(
        (config.output_directory / "analysis" / f"strategic_analysis_{period_slug}.json").read_text(
            encoding="utf-8"
        )
    )
    assert _is_ai_backed_strategy_document(analysis)
    assert result.stages[0].skipped is True


def test_progress_events_represent_cache_hit(tmp_path: Path) -> None:
    """Verify run_pipeline_for_report emits validation, cache, save, and completion events."""

    config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        enable_memory_storage=False,
    )
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
    events: list[dict[str, Any]] = []

    result = run_pipeline_for_report(
        input_model,
        config,
        progress_callback=lambda event: events.append(event.to_dict()),
    )

    assert result.cache_hit is True
    assert [event["stage_id"] for event in events] == [
        "validate_documents",
        "validate_documents",
        "prepare_interpret_files",
        "analysis_completed",
    ]
    assert events[2]["status"] == "cache_hit"
    assert events[-1]["completed_steps"] == events[-1]["total_steps"]
    assert events[-1]["label"] == "Análisis completado"


def test_progress_callback_is_optional_for_pipeline_use(tmp_path: Path) -> None:
    """Verify callers can still run without a progress callback."""

    config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        enable_memory_storage=False,
    )
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

    result = run_pipeline_for_report(input_model, config)

    assert result.success is True
    assert result.cache_hit is True


def test_progress_events_mark_validation_failure(tmp_path: Path) -> None:
    """Verify failed document validation emits a failed progress event."""

    config = _config(tmp_path)
    input_model = PipelineInputModel(
        workbook_path=tmp_path / "missing.xlsx",
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
    events: list[dict[str, Any]] = []

    with pytest.raises(ValueError):
        run_pipeline_for_report(
            input_model,
            config,
            progress_callback=lambda event: events.append(event.to_dict()),
        )

    assert events[-1]["stage_id"] == "validate_documents"
    assert events[-1]["status"] == "failed"


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


def test_cache_invalid_for_degraded_strategy_in_normal_ai_mode(tmp_path: Path) -> None:
    """Verify degraded deterministic artifacts are not reused as AI successes."""

    config = _config(tmp_path)
    input_model = _input_model(tmp_path)
    period_slug = "2026_06"
    _valid_report_artifacts(config, period_slug)
    analysis_path = config.output_directory / "analysis" / f"strategic_analysis_{period_slug}.json"
    analysis_path.write_text(
        json.dumps(
            {
                "validation_status": "accepted",
                "analysis_source": "degraded_deterministic",
                "strategic_recovery": {
                    "degraded_mode": True,
                    "source_label": "Modo degradado: análisis determinístico",
                },
            }
        ),
        encoding="utf-8",
    )
    cache_key = _pipeline_cache_key(input_model, config)
    manifest = _cache_manifest_path(config, cache_key)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"cache_key": cache_key, "period_slug": period_slug}), encoding="utf-8")

    result = _load_valid_cache(
        input_model=input_model,
        config=config,
        period_slug=period_slug,
        cache_key=cache_key,
        pipeline_started=0.0,
    )

    assert result is None
    assert _is_degraded_strategy_document(json.loads(analysis_path.read_text(encoding="utf-8")))


def test_cache_invalid_for_accepted_strategy_without_ai_provenance(tmp_path: Path) -> None:
    """Verify old accepted deterministic artifacts cannot satisfy normal AI mode."""

    config = _config(tmp_path)
    input_model = _input_model(tmp_path)
    period_slug = "2026_06"
    _valid_report_artifacts(config, period_slug)
    analysis_path = config.output_directory / "analysis" / f"strategic_analysis_{period_slug}.json"
    analysis_path.write_text(
        json.dumps({"validation_status": "accepted", "analysis_source": "ollama_modular_reasoning"}),
        encoding="utf-8",
    )
    cache_key = _pipeline_cache_key(input_model, config)
    manifest = _cache_manifest_path(config, cache_key)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"cache_key": cache_key, "period_slug": period_slug}), encoding="utf-8")

    result = _load_valid_cache(
        input_model=input_model,
        config=config,
        period_slug=period_slug,
        cache_key=cache_key,
        pipeline_started=0.0,
    )

    assert result is None


def test_degraded_mode_changes_pipeline_cache_key(tmp_path: Path) -> None:
    """Verify explicit degraded strategy mode is part of cache identity."""

    input_model = _input_model(tmp_path)
    ai_config = _config(tmp_path)
    degraded_config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        strategic_ai_mode="degraded",
    )

    assert _pipeline_cache_key(input_model, ai_config) != _pipeline_cache_key(input_model, degraded_config)


def test_timeout_style_unavailable_strategy_allows_fallback_report() -> None:
    """Verify rejected strategy behavior can still produce a report stage."""

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
                skipped=False,
                output_files=("outputs/report/financial_report_2026_05.html",),
                warnings=("Strategic analysis was unavailable or rejected; deterministic report rendered without validated recommendations.",),
                error=None,
                runtime_seconds=0.2,
            ),
        ),
        output_files=(),
        warnings=("Could not reach Ollama before timeout.",),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=1.0,
            stages_requested=2,
            stages_run=2,
            stages_succeeded=2,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=_config(Path(".")),
    )

    assert result.stages[-1].skipped is False
    assert "deterministic report rendered" in result.stages[-1].warnings[0]


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
    assert _ollama_client_for_stage(config, "ollama_structure_fallback").reasoning_enabled is False
    assert _ollama_client_for_stage(config, "ollama_investigation_planner").reasoning_enabled is False
    assert _ollama_client_for_stage(config, "strategic_analysis").reasoning_enabled is False


def test_default_model_routing_uses_balanced_model(tmp_path: Path) -> None:
    """Verify the supported default uses the SLA-qualified balanced model."""

    config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
    )

    assert config.effective_ollama_models() == {
        "structure_fallback": "qwen3:8b",
        "investigation_planner": "qwen3:8b",
        "strategic_analysis": "qwen3:8b",
    }


def test_runtime_optimization_settings_change_cache_key(tmp_path: Path) -> None:
    """Verify prompt-shaping settings are part of cache identity."""

    input_model = _input_model(tmp_path)
    default_config = _config(tmp_path)
    changed_config = PipelineConfig.from_project_root(
        tmp_path,
        python_executable=sys.executable,
        ollama_endpoint="http://localhost:9",
        max_planner_anomalies=3,
    )

    assert _pipeline_cache_key(input_model, default_config) != _pipeline_cache_key(
        input_model,
        changed_config,
    )


def test_pipeline_config_rejects_revision_confirmation_keyword(tmp_path: Path) -> None:
    """Verify revision confirmation is not a PipelineConfig runtime setting."""

    with pytest.raises(TypeError):
        PipelineConfig.from_project_root(
            tmp_path,
            python_executable=sys.executable,
            source_revision_confirmed=True,
        )


def test_monthly_cache_key_stable_for_same_inputs(tmp_path: Path) -> None:
    """Verify identical monthly inputs and config keep the same fingerprint."""

    input_model = _input_model(tmp_path)
    config = _config(tmp_path)

    assert _pipeline_cache_key(input_model, config) == _pipeline_cache_key(input_model, config)


def test_pipeline_version_changes_cache_key(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify pipeline/schema version participates in cache identity."""

    from finance_agent.orchestration import pipeline_orchestrator

    input_model = _input_model(tmp_path)
    config = _config(tmp_path)
    original = _pipeline_cache_key(input_model, config)

    monkeypatch.setattr(pipeline_orchestrator, "PIPELINE_SCHEMA_VERSION", "test-version-change")

    assert pipeline_orchestrator._pipeline_cache_key(input_model, config) != original


def test_pipeline_profile_summarizes_stage_telemetry(tmp_path: Path) -> None:
    """Verify profiler output includes stage timings and context telemetry."""

    config = _config(tmp_path)
    stage = PipelineStageResult(
        stage_name="strategic_analysis",
        display_name="Strategic analysis",
        critical=False,
        success=True,
        skipped=False,
        output_files=(),
        warnings=(),
        error=None,
        runtime_seconds=2.5,
        telemetry={
            "context_characters": 1200,
            "context_token_estimate": 300,
            "generation_time_seconds": 1.2,
        },
    )
    result = PipelineRunResult(
        success=True,
        stages=(stage,),
        output_files=(),
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=2.5,
            stages_requested=1,
            stages_run=1,
            stages_succeeded=1,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=config,
    )

    profile = build_pipeline_profile(result)

    assert profile["total_context_characters"] == 1200
    assert profile["total_token_estimate"] == 300
    assert profile["bottleneck_ranking"][0]["stage_name"] == "strategic_analysis"


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
