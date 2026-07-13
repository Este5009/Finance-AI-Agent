from __future__ import annotations

from pathlib import Path
from typing import Any

from finance_agent.orchestration import PipelineConfig, PipelineInputModel
from finance_agent.orchestration.pipeline_models import (
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
        ),
        runner=fake_runner,
    )

    assert result.success is True
    assert captured["input_model"].period_override == "2026-06"
    assert captured["config"].ollama_timeout_seconds == 12
    assert captured["config"].stage_timeout_seconds == 34
    assert captured["config"].input_model is captured["input_model"]
    assert captured["config"].effective_ollama_models() == {
        "structure_fallback": "qwen3:30b-a3b",
        "investigation_planner": "qwen3:30b-a3b",
        "strategic_analysis": "qwen3:30b-a3b",
    }


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
    assert streamlit_app._period_override_from_selection("Monthly", "2026-06") == "2026-06"


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

    assert fake_st.info_messages == ["Pipeline cache: hit"]
    assert fake_st.tables[0][0]["Status"] == "Skipped"
