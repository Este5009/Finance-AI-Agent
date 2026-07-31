from __future__ import annotations

from pathlib import Path
from typing import Any

from finance_agent.orchestration import PipelineConfig, PipelineInputModel
from finance_agent.orchestration.pipeline_models import (
    DetectedPeriod,
    PipelineProgressEvent,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
)
from finance_agent.ui import streamlit_app
from finance_agent.ui.streamlit_app import (
    StreamlitRunSettings,
    build_input_model_from_uploads,
    run_analysis_from_files,
    save_uploaded_file,
)


class FakeUpload:
    """Small UploadedFile stand-in for Streamlit UI tests."""

    def __init__(self, name: str, payload: bytes) -> None:
        """Create a fake upload with a name and byte payload."""

        self.name = name
        self._payload = payload

    def getbuffer(self) -> memoryview:
        """Return the fake upload bytes as Streamlit would."""

        return memoryview(self._payload)


def _pipeline_result(config: PipelineConfig) -> PipelineRunResult:
    """Build a minimal successful pipeline result fixture.

    Inputs: pipeline config from the UI helper.
    Outputs: successful PipelineRunResult.
    Assumptions: artifact rendering is tested elsewhere.
    """

    stage = PipelineStageResult(
        stage_name="ingestion",
        display_name="Document ingestion",
        critical=True,
        success=True,
        skipped=False,
        output_files=(),
        warnings=(),
        error=None,
        runtime_seconds=0.1,
    )
    return PipelineRunResult(
        success=True,
        stages=(stage,),
        output_files=(),
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=0.1,
            stages_requested=1,
            stages_run=1,
            stages_succeeded=1,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=config,
    )


def test_streamlit_ui_imports_without_streamlit_dependency() -> None:
    """Verify the UI module imports without importing the Streamlit package."""

    assert callable(streamlit_app.main)
    assert callable(streamlit_app.run_analysis_from_files)


def test_uploaded_files_are_saved_safely(tmp_path: Path) -> None:
    """Verify uploaded filenames are sanitized before writing to disk."""

    upload = FakeUpload("../June Report.xlsx", b"demo")

    saved = save_uploaded_file(upload, tmp_path)

    assert saved.parent == tmp_path
    assert saved.name == "June_Report.xlsx"
    assert saved.read_bytes() == b"demo"


def test_pdf_goals_upload_keeps_pdf_suffix(tmp_path: Path) -> None:
    """Verify uploaded PDF goals are saved with a .pdf suffix."""

    upload = FakeUpload("financial goals 2026.pdf", b"%PDF-1.4")

    saved = save_uploaded_file(upload, tmp_path)

    assert saved.suffix == ".pdf"
    assert saved.name == "financial_goals_2026.pdf"
    assert saved.read_bytes().startswith(b"%PDF")


def test_goals_uploader_accepts_pdf_docx_xlsx_and_xls() -> None:
    """Verify the Streamlit goals upload contract includes all supported types."""

    assert streamlit_app.GOALS_UPLOAD_TYPES == ("pdf", "docx", "xlsx", "xls")


def test_build_input_model_from_uploads_uses_generic_contract(tmp_path: Path) -> None:
    """Verify the UI delegates period detection to the shared input builder."""

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"placeholder")
    goals.write_bytes(b"placeholder")

    input_model = build_input_model_from_uploads(
        financial_report_path=report,
        goals_document_path=goals,
        settings=StreamlitRunSettings(report_language="es", period_override="2026-06"),
    )

    assert input_model.financial_report_path == report
    assert input_model.goals_document_path == goals
    assert input_model.report_language == "es"
    assert input_model.period_override == "2026-06"


def test_run_analysis_from_files_invokes_pipeline_runner(tmp_path: Path) -> None:
    """Verify the UI helper calls run_pipeline_for_report-compatible runners."""

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"placeholder")
    goals.write_bytes(b"placeholder")
    captured: dict[str, Any] = {}

    def fake_runner(
        input_model: PipelineInputModel,
        config: PipelineConfig,
    ) -> PipelineRunResult:
        """Capture the orchestrator inputs and return a successful result."""

        captured["input_model"] = input_model
        captured["config"] = config
        return _pipeline_result(config)

    result = run_analysis_from_files(
        financial_report_path=report,
        goals_document_path=goals,
        settings=StreamlitRunSettings(
            report_language="es",
            period_override="2026-06",
            ollama_endpoint="http://localhost:11434",
            ollama_model="qwen3:30b-a3b",
            ollama_timeout_seconds=12,
            stage_timeout_seconds=34,
            max_planner_anomalies=3,
            compact_context=True,
            deduplicate_context=False,
            enable_cache=False,
            enable_memory_storage=False,
        ),
        runner=fake_runner,
    )

    assert result.success is True
    assert captured["input_model"].period_override == "2026-06"
    assert captured["input_model"].source_revision_confirmed is False
    assert captured["config"].ollama_timeout_seconds == 12
    assert captured["config"].stage_timeout_seconds == 34
    assert captured["config"].max_planner_anomalies == 3
    assert captured["config"].deduplicate_context is False
    assert captured["config"].enable_cache is False
    assert captured["config"].enable_memory_storage is False
    assert captured["config"].input_model is captured["input_model"]
    assert not hasattr(captured["config"], "source_revision_confirmed")
    assert captured["config"].effective_ollama_models() == {
        "structure_fallback": "qwen3:30b-a3b",
        "investigation_planner": "qwen3:30b-a3b",
        "strategic_analysis": "qwen3:30b-a3b",
    }


def test_run_analysis_from_files_passes_progress_callback_when_supported(tmp_path: Path) -> None:
    """Verify the UI helper forwards progress callbacks to compatible runners."""

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"placeholder")
    goals.write_bytes(b"placeholder")
    received_events: list[PipelineProgressEvent] = []

    def fake_runner(
        input_model: PipelineInputModel,
        config: PipelineConfig,
        *,
        progress_callback: Any = None,
    ) -> PipelineRunResult:
        """Emit one progress event through the supplied callback."""

        assert input_model.period_override == "2026-06"
        if progress_callback is not None:
            progress_callback(
                PipelineProgressEvent(
                    stage_id="analysis_completed",
                    label="Análisis completado",
                    detail="Listo.",
                    completed_steps=9,
                    total_steps=9,
                    status="completed",
                )
            )
        return _pipeline_result(config)

    result = run_analysis_from_files(
        financial_report_path=report,
        goals_document_path=goals,
        settings=StreamlitRunSettings(report_language="es", period_override="2026-06"),
        runner=fake_runner,
        progress_callback=received_events.append,
    )

    assert result.success is True
    assert received_events[0].stage_id == "analysis_completed"
    assert received_events[0].status == "completed"


def test_progress_event_merge_preserves_rerun_snapshot() -> None:
    """Verify progress state can be stored and reused across Streamlit reruns."""

    events = streamlit_app._initial_progress_events()
    updated = streamlit_app._merge_progress_event(
        events,
        PipelineProgressEvent(
            stage_id="query_history",
            label="Consultando el historial",
            detail="Historial recuperado.",
            completed_steps=5,
            total_steps=9,
            status="completed",
        ),
    )

    assert len(updated) == len(events)
    assert updated[4]["stage_id"] == "query_history"
    assert updated[4]["status"] == "completed"
    assert streamlit_app._latest_active_progress(updated)["stage_id"] == "query_history"


def test_failure_progress_snapshot_marks_analysis_completed_failed() -> None:
    """Verify failure events produce a recoverable progress snapshot."""

    failed = PipelineProgressEvent(
        stage_id="analysis_completed",
        label="Análisis completado",
        detail="No se pudo iniciar el análisis.",
        completed_steps=0,
        total_steps=9,
        status="failed",
    )
    updated = streamlit_app._merge_progress_event(
        streamlit_app._initial_progress_events(),
        failed,
    )

    latest = streamlit_app._latest_active_progress(updated)

    assert latest["stage_id"] == "analysis_completed"
    assert latest["status"] == "failed"


def test_run_analysis_from_files_passes_revision_confirmation_on_input_model(tmp_path: Path) -> None:
    """Verify revision confirmation is per-submission input, not PipelineConfig."""

    report = tmp_path / "monthly_financial_report_may_2026.xlsx"
    goals = tmp_path / "financial_goals_2026_05.pdf"
    report.write_bytes(b"may financial report")
    goals.write_bytes(b"%PDF may goals")
    captured: dict[str, Any] = {}

    def fake_runner(input_model: PipelineInputModel, config: PipelineConfig) -> PipelineRunResult:
        """Capture per-submission revision confirmation."""

        captured["input_model"] = input_model
        captured["config"] = config
        return _pipeline_result(config)

    run_analysis_from_files(
        financial_report_path=report,
        goals_document_path=goals,
        settings=StreamlitRunSettings(
            report_language="es",
            period_override="2026-05",
            source_revision_confirmed=True,
        ),
        runner=fake_runner,
    )

    assert captured["input_model"].source_revision_confirmed is True
    assert not hasattr(captured["config"], "source_revision_confirmed")


def test_streamlit_preflight_classification_does_not_raise_attribute_error(tmp_path: Path) -> None:
    """Verify Streamlit calls the canonical MemoryRepository preflight API."""

    classification = streamlit_app._classify_upload_for_period(
        uploaded_file=FakeUpload("may_report.xlsx", b"may report bytes"),
        document_type="financial_report",
        effective_period="2026-05",
        database_path=tmp_path / "memory.db",
    )

    assert classification is not None
    assert classification["status"] == "new"
    assert classification["effective_period"] == "2026-05"


def test_ui_single_model_setting_routes_all_stages(tmp_path: Path) -> None:
    """Verify UI single-model override preserves one-model compatibility."""

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"placeholder")
    goals.write_bytes(b"placeholder")
    input_model = build_input_model_from_uploads(
        financial_report_path=report,
        goals_document_path=goals,
        settings=StreamlitRunSettings(report_language="es", period_override="2026-06"),
    )

    config = streamlit_app.build_pipeline_config(
        input_model,
        StreamlitRunSettings(
            report_language="es",
            period_override="2026-06",
            ollama_model="qwen3:30b-a3b",
        ),
    )

    assert config.effective_ollama_models() == {
        "structure_fallback": "qwen3:30b-a3b",
        "investigation_planner": "qwen3:30b-a3b",
        "strategic_analysis": "qwen3:30b-a3b",
    }
    assert config.max_planner_anomalies == 5
    assert config.compact_context is True
    assert config.deduplicate_context is True


def test_ui_experimental_stage_models_remain_available(tmp_path: Path) -> None:
    """Verify explicit experimental stage-specific model settings still route."""

    report = tmp_path / "monthly_financial_report_june_2026.xlsx"
    goals = tmp_path / "financial_goals_2026.pdf"
    report.write_bytes(b"placeholder")
    goals.write_bytes(b"placeholder")
    input_model = build_input_model_from_uploads(
        financial_report_path=report,
        goals_document_path=goals,
        settings=StreamlitRunSettings(report_language="es", period_override="2026-06"),
    )

    config = streamlit_app.build_pipeline_config(
        input_model,
        StreamlitRunSettings(
            report_language="es",
            period_override="2026-06",
            ollama_model="qwen3:30b-a3b",
            structure_ollama_model="qwen3:latest",
            planner_ollama_model="qwen3:latest",
            analysis_ollama_model="qwen3:30b-a3b",
        ),
    )

    assert config.effective_ollama_models() == {
        "structure_fallback": "qwen3:latest",
        "investigation_planner": "qwen3:latest",
        "strategic_analysis": "qwen3:30b-a3b",
    }


def test_period_override_auto_returns_none() -> None:
    """Verify Auto mode leaves period detection in charge."""

    assert streamlit_app._period_override_from_selection("Auto", "2026-06") is None
    assert streamlit_app._period_override_from_selection("Detectar automáticamente", "2026-06") is None
    assert streamlit_app._period_override_from_selection("Monthly", "2026-06") == "2026-06"


def test_supported_ui_period_options_are_monthly_only() -> None:
    """Verify unsupported frequencies are not selectable in the UI."""

    assert streamlit_app.SUPPORTED_UI_PERIOD_OPTIONS == ("Detectar automáticamente", "Mensual")
    for unsupported in ("Trimestral", "Semestral", "Anual", "Personalizado"):
        assert unsupported not in streamlit_app.SUPPORTED_UI_PERIOD_OPTIONS


def test_monthly_readiness_accepts_automatic_monthly_detection(tmp_path: Path) -> None:
    """Verify confident monthly detection enables the Streamlit monthly workflow."""

    input_model = PipelineInputModel(
        financial_report_path=tmp_path / "university_financial_report_2026_12.xlsx",
        goals_document_path=tmp_path / "financial_goals_2026_12.pdf",
        detected_period=DetectedPeriod(
            period_type="monthly",
            label="2026-12",
            confidence=0.9,
            year=2026,
            month=12,
        ),
        period_type="monthly",
        report_language="es",
    )

    ready, message = streamlit_app._monthly_readiness_message(
        input_model=input_model,
        override_mode="Detectar automáticamente",
        override_value="",
    )

    assert ready is True
    assert "Período detectado: Mensual" in message
    assert "Dic 2026" in message


def test_monthly_readiness_accepts_manual_monthly_selection() -> None:
    """Verify manual monthly mode requires a concrete month/year."""

    ready, message = streamlit_app._monthly_readiness_message(
        input_model=None,
        override_mode="Mensual",
        override_value="2027-01",
    )

    assert ready is True
    assert "2027-01" in message


def test_monthly_readiness_blocks_ambiguous_detection() -> None:
    """Verify ambiguous/non-monthly auto detection is blocked instead of guessed."""

    ready, message = streamlit_app._monthly_readiness_message(
        input_model=None,
        override_mode="Detectar automáticamente",
        override_value="",
    )

    assert ready is False
    assert "Seleccione el reporte financiero" in message


def test_monthly_readiness_blocks_invalid_manual_month() -> None:
    """Verify manual monthly mode blocks unsupported labels."""

    ready, message = streamlit_app._monthly_readiness_message(
        input_model=None,
        override_mode="Mensual",
        override_value="2027-Q1",
    )

    assert ready is False
    assert "formato 2026-12" in message


def test_file_validation_messages_are_spanish() -> None:
    """Verify upload validation returns actionable Spanish messages."""

    pending_status, pending_message = streamlit_app._file_status_message(None, ("xlsx",))
    ok_status, ok_message = streamlit_app._file_status_message(FakeUpload("reporte.xlsx", b"1234"), ("xlsx",))
    bad_status, bad_message = streamlit_app._file_status_message(FakeUpload("metas.txt", b"bad"), ("pdf",))

    assert pending_status == "pending"
    assert "Ningún archivo seleccionado" in pending_message
    assert ok_status == "ok"
    assert "Archivo listo" in ok_message
    assert bad_status == "error"
    assert "Formato no permitido" in bad_message


def test_ui_copy_contains_executive_workflow_cards_and_spanish_upload_copy() -> None:
    """Verify the UI source contains Spanish executive workflow and upload copy."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")

    assert "Cargue archivos" in source
    assert "Genere el análisis" in source
    assert "Descargue resultados" in source
    assert "Archivos compatibles" in source
    assert "Seleccionar archivo" in source
    assert "Detectar automáticamente" in source
    assert "Actualmente, el sistema procesa reportes mensuales" in source
    assert "Ningún archivo seleccionado" in source
    forbidden_native_copy = (
        "Drag" + " and drop file here",
        "Browse" + " files",
        "Limit" + " 200MB per file",
    )
    for phrase in forbidden_native_copy:
        assert phrase not in source


def test_success_registration_messages_are_spanish() -> None:
    """Verify post-run document registration outcomes use clear Spanish labels."""

    config = PipelineConfig.from_project_root(Path("."), python_executable="python")
    result = _pipeline_result(config)
    cached = PipelineRunResult(
        success=True,
        stages=result.stages,
        output_files=(),
        warnings=(),
        runtime_summary=result.runtime_summary,
        config=config,
        cache_hit=True,
    )

    assert streamlit_app._success_registration_message(cached, ()) == "Análisis reutilizado."
    assert streamlit_app._success_registration_message(result, ({"status": "revision"},)) == "Nueva versión registrada."
    assert "registrado anteriormente" in streamlit_app._success_registration_message(result, ({"status": "duplicate"},))
    assert streamlit_app._success_registration_message(result, ({"status": "new"},)) == "Nuevo período registrado."


def test_custom_ui_css_uses_accessible_card_contrast() -> None:
    """Verify custom card CSS avoids white-on-light low-contrast combinations."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")

    assert ".run-action-card" in source
    assert "color: #33485c" in source
    assert "color: #435466" in source
    assert "color: white" not in source
    assert "color: #fff" not in source


def test_ui_stage_results_display_cache_and_skipped_status() -> None:
    """Verify stage rendering exposes cache state and skipped statuses."""

    class FakeStreamlit:
        """Capture calls made by the stage renderer."""

        def __init__(self) -> None:
            """Initialize capture lists."""

            self.info_messages: list[str] = []
            self.tables: list[list[dict[str, object]]] = []

        def subheader(self, _text: str) -> None:
            """Accept subheader calls without rendering."""

        def info(self, text: str) -> None:
            """Capture informational messages."""

            self.info_messages.append(text)

        def dataframe(self, rows: list[dict[str, object]], **_kwargs: object) -> None:
            """Capture dataframe rows."""

            self.tables.append(rows)

    config = PipelineConfig.from_project_root(
        Path("."),
        python_executable="python",
    )
    result = PipelineRunResult(
        success=True,
        stages=(
            PipelineStageResult(
                stage_name="ollama_structure_fallback",
                display_name="Ollama structure fallback",
                critical=False,
                success=True,
                skipped=True,
                output_files=(),
                warnings=("Skipped; deterministic structure was high-confidence.",),
                error=None,
                runtime_seconds=0.01,
                telemetry={
                    "context_characters": 100,
                    "context_token_estimate": 25,
                    "generation_time_seconds": 0.0,
                    "json_validation_time_seconds": 0.0,
                    "python_preprocessing_time_seconds": 0.01,
                },
            ),
        ),
        output_files=(),
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=0.01,
            stages_requested=1,
            stages_run=0,
            stages_succeeded=0,
            stages_failed=0,
            stages_skipped=1,
        ),
        config=config,
        cache_hit=True,
    )
    fake_st = FakeStreamlit()

    streamlit_app._render_stage_results(fake_st, result)

    assert fake_st.info_messages == ["Reutilización: se reutilizó un análisis existente."]
    assert fake_st.tables[0][0]["Estado"] == "Omitido"
    assert fake_st.tables[-1][0]["Tamaño contexto"] == 100
