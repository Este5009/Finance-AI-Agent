from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

import pytest

from finance_agent.orchestration import (
    PipelineConfig,
    PipelineInputModel,
    PipelineProgressCallback,
    PipelineProgressEvent as PublicPipelineProgressEvent,
)
from finance_agent.orchestration.pipeline_models import (
    DetectedPeriod,
    PipelineProgressEvent,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
)
from finance_agent.ui import streamlit_app
from finance_agent.reporting import ReportInputBundle, build_report_model
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


class FakeStreamlitRenderer:
    """Reusable Streamlit stand-in for result-tab rendering tests."""

    def __init__(self) -> None:
        """Initialize capture lists for visible UI calls."""

        self.markdown_calls: list[str] = []
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.error_messages: list[str] = []
        self.code_blocks: list[str] = []
        self.tables: list[Any] = []
        self.downloads: list[dict[str, str]] = []
        self.tab_labels: list[str] = []
        self.vega_lite_charts: list[dict[str, Any]] = []

    def __enter__(self) -> "FakeStreamlitRenderer":
        """Return this object for ``with`` blocks."""

        return self

    def __exit__(self, *_args: object) -> bool:
        """Do not suppress exceptions unless the UI guard catches them."""

        return False

    def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
        """Capture markdown/HTML output."""

        self.markdown_calls.append(text)

    def info(self, text: str) -> None:
        """Capture informational messages."""

        self.info_messages.append(text)

    def success(self, text: str) -> None:
        """Capture success messages."""

        self.success_messages.append(text)

    def warning(self, text: str) -> None:
        """Capture warning messages."""

        self.warning_messages.append(text)

    def error(self, text: str) -> None:
        """Capture error messages."""

        self.error_messages.append(text)

    def code(self, text: str) -> None:
        """Capture technical traceback blocks."""

        self.code_blocks.append(text)

    def caption(self, text: str) -> None:
        """Capture caption text as markdown-like output."""

        self.markdown_calls.append(text)

    def subheader(self, text: str) -> None:
        """Capture subheaders."""

        self.markdown_calls.append(text)

    def dataframe(self, rows: Any, **_kwargs: object) -> None:
        """Capture dataframe payloads."""

        self.tables.append(rows)

    def vega_lite_chart(self, spec: dict[str, Any], *, use_container_width: bool = False) -> None:
        """Capture Streamlit Vega-Lite chart specs."""

        self.vega_lite_charts.append({"spec": spec, "use_container_width": use_container_width})

    def columns(self, count: int) -> list["FakeStreamlitRenderer"]:
        """Return context managers for column layouts."""

        return [self for _ in range(count)]

    def tabs(self, labels: list[str]) -> list["FakeStreamlitRenderer"]:
        """Return context managers for tab layouts."""

        self.tab_labels = list(labels)
        return [self for _ in labels]

    def expander(self, _label: str, *, expanded: bool = False) -> "FakeStreamlitRenderer":
        """Return this object for collapsed diagnostic blocks."""

        return self

    def download_button(self, *, label: str, data: bytes, file_name: str, mime: str) -> None:
        """Capture download button metadata."""

        self.downloads.append({"label": label, "file_name": file_name, "mime": mime})

    def progress(self, *_args: object, **_kwargs: object) -> None:
        """Accept progress calls without rendering."""


def _july_report_model() -> dict[str, Any]:
    """Load the existing July report model used for UI presentation validation."""

    paths = {
        "finance": Path("outputs/calculations/finance_summary_2026_07.json"),
        "kpis": Path("outputs/calculations/kpi_summary_2026_07.csv"),
        "anomalies": Path("outputs/anomalies/anomaly_report_2026_07.json"),
        "evidence": Path("outputs/evidence/evidence_package_2026_07.json"),
        "analysis": Path("outputs/analysis/strategic_analysis_2026_07.json"),
    }
    if not all(path.is_file() for path in paths.values()):
        fallback = Path("outputs/report/report_model_2026_07.json")
        if not fallback.is_file():
            pytest.skip("July report artifacts are not available in this checkout.")
        return streamlit_app._load_json(fallback)
    with paths["kpis"].open("r", encoding="utf-8-sig", newline="") as handle:
        kpis = tuple(dict(row) for row in csv.DictReader(handle))
    return build_report_model(
        ReportInputBundle(
            period_slug="2026_07",
            finance_summary=json.loads(paths["finance"].read_text(encoding="utf-8")),
            kpi_summary=kpis,
            anomaly_report=json.loads(paths["anomalies"].read_text(encoding="utf-8")),
            evidence_package=json.loads(paths["evidence"].read_text(encoding="utf-8")),
            strategic_analysis=json.loads(paths["analysis"].read_text(encoding="utf-8")),
            source_files=tuple(str(path) for path in paths.values()),
        )
    ).to_dict()


def _october_report_model() -> dict[str, Any]:
    """Load the existing October report model for presentation-only validation."""

    path = Path("outputs/report/report_model_2026_10.json")
    if not path.is_file():
        pytest.skip("October report model is not available in this checkout.")
    return streamlit_app._load_json(path)


def _july_result() -> PipelineRunResult:
    """Build a pipeline result pointing at existing July presentation artifacts."""

    config = PipelineConfig.from_project_root(Path("."), python_executable="python")
    outputs = (
        "outputs/report/report_model_2026_07.json",
        "outputs/report/financial_report_2026_07.html",
        "outputs/report/financial_report_2026_07.pdf",
        "outputs/analysis/strategic_analysis_2026_07.json",
        "outputs/evidence/evidence_package_2026_07.json",
    )
    missing = [path for path in outputs if not Path(path).is_file()]
    if missing:
        pytest.skip(f"July artifacts are not available: {missing}")
    return PipelineRunResult(
        success=True,
        stages=(),
        output_files=outputs,
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=1.0,
            stages_requested=0,
            stages_run=0,
            stages_succeeded=0,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=config,
    )


def _october_result() -> PipelineRunResult:
    """Build a pipeline result pointing at existing October presentation artifacts."""

    config = PipelineConfig.from_project_root(Path("."), python_executable="python")
    outputs = (
        "outputs/report/report_model_2026_10.json",
        "outputs/report/financial_report_2026_10.html",
        "outputs/report/financial_report_2026_10.pdf",
        "outputs/analysis/strategic_analysis_2026_10.json",
        "outputs/evidence/evidence_package_2026_10.json",
    )
    missing = [path for path in outputs if not Path(path).is_file()]
    if missing:
        pytest.skip(f"October artifacts are not available: {missing}")
    return PipelineRunResult(
        success=True,
        stages=(),
        output_files=outputs,
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=1.0,
            stages_requested=0,
            stages_run=0,
            stages_succeeded=0,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=config,
    )


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


def test_streamlit_progress_symbols_use_public_orchestration_api() -> None:
    """Verify Streamlit-visible progress symbols import from the public package path."""

    assert PublicPipelineProgressEvent is PipelineProgressEvent
    assert PipelineProgressCallback is not None


def test_stage_and_period_labels_are_spanish_for_normal_ui() -> None:
    """Verify internal English labels are localized before display."""

    historical_stage = PipelineStageResult(
        stage_name="historical_context",
        display_name="Historical context",
        critical=False,
        success=True,
        skipped=False,
        output_files=(),
        warnings=(),
        error=None,
        runtime_seconds=0.1,
    )
    renderer_stage = PipelineStageResult(
        stage_name="report_model_and_renderers",
        display_name="Report model and renderers",
        critical=False,
        success=True,
        skipped=False,
        output_files=(),
        warnings=(),
        error=None,
        runtime_seconds=0.1,
    )

    assert streamlit_app._stage_display_name(historical_stage) == "Contexto histórico"
    assert streamlit_app._stage_display_name(renderer_stage) == "Generación del reporte"
    assert streamlit_app._period_type_label("monthly") == "Mensual"


def test_financial_health_card_does_not_render_previous_value_as_delta() -> None:
    """Verify KPI cards keep previous value and change as separate rows."""

    class FakeStreamlit:
        """Capture markdown rendered by the KPI card helper."""

        def __init__(self) -> None:
            """Create a markdown capture list."""

            self.markdown_calls: list[tuple[str, bool]] = []

        def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
            """Capture markdown text and HTML flag."""

            self.markdown_calls.append((text, unsafe_allow_html))

        def metric(self, *_args: object, **_kwargs: object) -> None:
            """Fail if a financial card tries to use Streamlit delta rendering."""

            raise AssertionError("KPI comparison cards must not call st.metric")

    fake_st = FakeStreamlit()
    streamlit_app._render_financial_health_card(
        fake_st,
        {
            "label": "Ingresos",
            "value": "$1,992,060",
            "badge": "Atención",
            "description": "Ingresos totales del periodo.",
            "comparison_rows": [
                {"label": "Periodo anterior", "value": "$2,005,584"},
                {"label": "Variación respecto al periodo anterior", "value": "-$13,524 (-0.7%)"},
            ],
        },
    )
    html = fake_st.markdown_calls[0][0]

    assert "Período anterior" in html
    assert "$2,005,584" in html
    assert "Variación respecto al período anterior" in html
    assert "-$13,524 (-0.7%)" in html


def test_executive_summary_renders_without_markdown_math() -> None:
    """Verify currency prose is escaped instead of passed through st.write/LaTeX."""

    class FakeStreamlit:
        """Capture escaped summary rendering."""

        def __init__(self) -> None:
            """Create a markdown capture list."""

            self.markdown_calls: list[tuple[str, bool]] = []

        def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
            """Capture rendered paragraph."""

            self.markdown_calls.append((text, unsafe_allow_html))

    fake_st = FakeStreamlit()
    streamlit_app._render_safe_text_block(
        fake_st,
        "Resultado operativo de $-374,000 USD, con ingresos de $1,992,060 USD.",
    )

    html, unsafe = fake_st.markdown_calls[0]
    assert unsafe is True
    assert "$-374,000 USD, con ingresos" in html
    assert "st.write" not in html


def test_kpi_card_css_has_light_and_dark_contrast_rules() -> None:
    """Verify custom KPI card colors define readable light/dark foregrounds."""

    class FakeStreamlit:
        """Capture injected CSS."""

        def __init__(self) -> None:
            """Create a markdown capture list."""

            self.markdown_calls: list[str] = []

        def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
            """Capture CSS markdown."""

            self.markdown_calls.append(text)

    fake_st = FakeStreamlit()
    streamlit_app._apply_page_styles(fake_st)
    css = "\n".join(fake_st.markdown_calls)

    assert ".ui-kpi-card" in css
    assert "@media (prefers-color-scheme: dark)" in css
    assert "--fa-surface: #ffffff" in css
    assert "--fa-surface: #172033" in css
    assert "background: var(--fa-surface)" in css
    assert "color: var(--fa-text)" in css
    assert "color: #f5f7fa" in css or "color: #f8fafc" in css


def test_status_badge_renders_label_not_metadata_repr() -> None:
    """Verify status metadata dictionaries never render as Python repr strings."""

    html = streamlit_app._status_badge_html(
        {"label": "Informativo", "class": "neutral", "icon": "ℹ"}
    )

    assert "Informativo" in html
    assert "ℹ" in html
    assert "{'label'" not in html
    assert "'class'" not in html


def test_financial_health_card_handles_badge_metadata_without_raw_dict() -> None:
    """Verify KPI cards render badge metadata as a styled chip."""

    class FakeStreamlit:
        """Capture KPI card HTML."""

        def __init__(self) -> None:
            """Initialize markdown capture list."""

            self.markdown_calls: list[str] = []

        def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
            """Capture HTML generated by the card renderer."""

            self.markdown_calls.append(text)

    fake_st = FakeStreamlit()
    streamlit_app._render_financial_health_card(
        fake_st,
        {
            "label": "Resultado operativo",
            "value": "-$192,500",
            "badge": {"label": "Crítico", "class": "critical", "icon": "⚠"},
            "description": "Resultado neto del periodo.",
            "comparison_rows": [
                {"label": "Periodo anterior", "value": "-$374,000"},
                {"label": "Variación respecto al periodo anterior", "value": "$181,500"},
            ],
        },
    )
    html = fake_st.markdown_calls[0]

    assert "Crítico" in html
    assert "⚠" in html
    assert "{'label'" not in html
    assert "Resultado operativo" in html


def test_artifact_paths_restore_downloads_from_period_slug_siblings(tmp_path: Path) -> None:
    """Verify PDF/HTML downloads survive when output_files lists only the model."""

    output_dir = tmp_path / "outputs"
    report_dir = output_dir / "report"
    analysis_dir = output_dir / "analysis"
    evidence_dir = output_dir / "evidence"
    report_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    report_model = report_dir / "report_model_2026_07.json"
    html = report_dir / "financial_report_2026_07.html"
    pdf = report_dir / "financial_report_2026_07.pdf"
    strategic = analysis_dir / "strategic_analysis_2026_07.json"
    evidence = evidence_dir / "evidence_package_2026_07.json"
    for path in (report_model, html, pdf, strategic, evidence):
        path.write_text("{}", encoding="utf-8")
    config = PipelineConfig.from_project_root(tmp_path, python_executable="python")
    result = PipelineRunResult(
        success=True,
        stages=(),
        output_files=(str(report_model),),
        warnings=(),
        runtime_summary=RuntimeSummary(
            total_runtime_seconds=1.0,
            stages_requested=0,
            stages_run=0,
            stages_succeeded=0,
            stages_failed=0,
            stages_skipped=0,
        ),
        config=config,
    )

    artifacts = streamlit_app._artifact_paths(result)

    assert artifacts["PDF"] == pdf
    assert artifacts["HTML"] == html
    assert artifacts["Report model JSON"] == report_model
    assert artifacts["Strategic analysis JSON"] == strategic
    assert artifacts["Evidence package JSON"] == evidence


def test_render_download_card_uses_default_badge_without_name_error(tmp_path: Path) -> None:
    """Verify download cards do not reference undefined presentation variables."""

    path = tmp_path / "financial_report_2026_07.pdf"
    path.write_bytes(b"%PDF demo")
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_download_card(fake_st, "PDF", path, "application/pdf")

    assert fake_st.downloads == [
        {"label": "Descargar PDF", "file_name": path.name, "mime": "application/pdf"}
    ]
    assert not fake_st.error_messages


def test_render_downloads_tab_has_no_name_error_for_july_artifacts() -> None:
    """Verify the downloads tab renders existing July files without NameError."""

    result = _july_result()
    artifacts = streamlit_app._artifact_paths(result)
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_downloads_tab(fake_st, artifacts, _july_report_model())

    downloaded = {item["file_name"] for item in fake_st.downloads}
    assert "financial_report_2026_07.pdf" in downloaded
    assert "financial_report_2026_07.html" in downloaded
    assert not fake_st.error_messages


def test_each_results_tab_renders_from_july_report_model() -> None:
    """Verify every results tab can render independently from the July model."""

    report_model = _july_report_model()
    result = _july_result()
    artifacts = streamlit_app._artifact_paths(result)
    renderers = (
        lambda st: streamlit_app._render_overview_tab(st, report_model, result),
        lambda st: streamlit_app._render_kpi_tab(st, report_model),
        lambda st: streamlit_app._render_anomaly_tab(st, report_model),
        lambda st: streamlit_app._render_analysis_tab(st, report_model),
        lambda st: streamlit_app._render_recommendations_tab(st, report_model),
        lambda st: streamlit_app._render_downloads_tab(st, artifacts, report_model),
    )

    for renderer in renderers:
        fake_st = FakeStreamlitRenderer()
        renderer(fake_st)
        assert not fake_st.error_messages


def test_july_anomaly_tab_renders_deterministic_detail_cards() -> None:
    """Verify severity counts with anomaly rows render concrete anomaly details."""

    report_model = _july_report_model()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_anomaly_tab(fake_st, report_model)

    visible = "\n".join(fake_st.markdown_calls)
    assert "Anomalías detectadas" in visible
    assert "Indicador afectado" in visible
    assert "Valor observado" in visible
    assert "Referencia" in visible
    assert "Pagos estudiantiles vencidos por encima del límite" in visible or "Flujo de caja bajo o negativo" in visible
    assert "Overdue student payments above limit" not in visible
    assert "Negative or low cash flow" not in visible
    assert "No hay anomalías relevantes para mostrar." not in fake_st.info_messages


def test_july_analysis_tab_contains_deterministic_fallback_text() -> None:
    """Verify rejected strategy does not empty executive analysis sections."""

    report_model = _july_report_model()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)

    visible = "\n".join(fake_st.markdown_calls)
    assert "Situación financiera actual" in visible
    assert "Cambios frente al período anterior" in visible
    assert "Presiones y riesgos" in visible
    assert "Resultados por departamento" in visible
    assert "Tendencias históricas" in visible
    assert "Acciones para la gestión" in visible
    assert "Resultado operativo" in visible


def test_july_recommendations_tab_shows_fallback_follow_up_and_missing_state() -> None:
    """Verify recommendations tab never leaves headings without visible content."""

    report_model = _july_report_model()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_recommendations_tab(fake_st, report_model)

    visible = "\n".join(fake_st.markdown_calls)
    assert "Recomendaciones estratégicas no validadas" in visible
    assert "Hallazgos verificados que requieren atención" in visible
    assert "Seguimiento verificado de recomendaciones previas" in visible
    assert "Emitida en" in visible
    assert "Jun 2026" in visible
    assert fake_st.success_messages == ["No se reporta información faltante relevante."]


def test_download_tab_presentation_error_does_not_break_other_result_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify one tab failure is isolated to that tab."""

    result = _july_result()
    fake_st = FakeStreamlitRenderer()

    def fail_downloads(*_args: object, **_kwargs: object) -> None:
        """Raise a presentation-only error for the downloads tab."""

        raise NameError("simulated downloads presentation bug")

    monkeypatch.setattr(streamlit_app, "_render_downloads_tab", fail_downloads)

    streamlit_app._render_results(fake_st, result)

    visible_text = "\n".join(fake_st.markdown_calls)
    assert fake_st.tab_labels == ["Resumen", "KPIs", "Anomalías", "Análisis", "Recomendaciones", "Descargas"]
    assert "Resumen ejecutivo" in visible_text
    assert "Indicadores principales" in visible_text
    assert "Anomalías del periodo" in visible_text
    assert "Situación financiera actual" in visible_text
    assert "Acciones para la gestión" in visible_text
    assert "Recomendaciones estratégicas actuales" in visible_text
    assert len(fake_st.error_messages) == 1
    assert "No se pudo mostrar la pestaña Descargas" in fake_st.error_messages[0]
    assert fake_st.code_blocks and "simulated downloads presentation bug" in fake_st.code_blocks[0]


def test_results_header_uses_responsive_grid_not_rigid_five_columns() -> None:
    """Verify summary cards avoid the narrow fixed-column layout that wrapped letters."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")
    start = source.index("def _render_results_header")
    end = source.index("def _render_attention_summary", start)
    header_body = source[start:end]

    assert "_render_responsive_card_grid" in header_body
    assert "st.columns(5)" not in header_body
    assert "ui-responsive-grid" in source
    assert "word-break: normal" in source
    assert "hyphens: none" in source


def test_october_anomaly_labels_are_spanish_in_presentation_view() -> None:
    """Verify deterministic anomaly names/descriptions are localized for users."""

    report_model = _october_report_model()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_anomaly_tab(fake_st, report_model)

    visible = "\n".join(fake_st.markdown_calls)
    assert "Nómina sobre ingresos por encima del umbral" in visible
    assert "Cobranza de matrícula por debajo de la meta" in visible
    assert "Payroll exceeds revenue threshold" not in visible
    assert "Tuition collection below target" not in visible
    assert "Collection rate is" not in visible


def test_october_analysis_tab_uses_grouped_executive_sections() -> None:
    """Verify the analysis tab answers management questions through grouped sections."""

    report_model = _october_report_model()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)

    visible = "\n".join(fake_st.markdown_calls)
    for heading in (
        "Situación financiera actual",
        "Cambios frente al período anterior",
        "Presiones y riesgos",
        "Resultados por departamento",
        "Tendencias históricas",
        "Acciones para la gestión",
    ):
        assert heading in visible
    assert "Ver tabla completa por departamento" in visible or fake_st.tables


def test_analysis_tab_renders_canonical_historical_trend_cards_and_charts() -> None:
    """Verify Streamlit shows historical trend cards/charts from the report view."""

    report_model = _july_report_model()
    report_model["period_slug"] = "2026_08"
    report_model["report_period"] = "2026_08"
    report_model["sections"] = [
        section for section in report_model["sections"] if section["section_id"] != "historical_trends"
    ]
    for section in report_model["sections"]:
        if section["section_id"] == "financial_health_overview":
            section["content"]["collection_rate"] = 0.9
            section["content"]["payroll_percentage_of_revenue"] = 0.49
    report_model["sections"].append(
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "collection_rate",
                        "metric": "Tasa de cobranza",
                        "unit": "ratio",
                        "direction": "improving",
                        "points": [
                            {"period": "2026_06", "value": 0.84},
                            {"period": "2026_07", "value": 0.85},
                        ],
                    },
                    {
                        "metric_id": "payroll_percentage_of_revenue",
                        "metric": "Nómina / ingresos",
                        "unit": "ratio",
                        "direction": "improving",
                        "points": [
                            {"period": "2026_06", "value": 0.53},
                            {"period": "2026_07", "value": 0.52},
                        ],
                    },
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_08.json"],
            "warnings": [],
        }
    )
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)
    html = "\n".join(fake_st.markdown_calls)

    assert len(fake_st.vega_lite_charts) == 2
    assert "Último valor" in html
    chart_labels = {
        row["period_label"]
        for chart in fake_st.vega_lite_charts
        for row in chart["spec"]["data"]["values"]
    }
    assert "Ago 2026" in chart_labels
    assert "get_metric_history" not in html


def test_august_analysis_tab_renders_all_historical_charts() -> None:
    """Verify existing August artifact renders all seven rolling-window charts."""

    path = Path("outputs/report/report_model_2026_08.json")
    if not path.is_file():
        pytest.skip("August report model artifact is not available.")
    report_model = streamlit_app._load_json(path)
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)
    html = "\n".join(fake_st.markdown_calls)

    assert len(fake_st.vega_lite_charts) == 7
    chart_labels = {
        row["period_label"]
        for chart in fake_st.vega_lite_charts
        for row in chart["spec"]["data"]["values"]
    }
    for label in ("Mar 2026", "Abr 2026", "May 2026", "Jun 2026", "Jul 2026", "Ago 2026"):
        assert label in chart_labels
    assert "Historial insuficiente" not in html


def test_september_analysis_tab_preserves_july_and_august_points() -> None:
    """Verify September trend charts are not collapsed to June and September only."""

    report_model = _july_report_model()
    report_model["period_slug"] = "2026_09"
    report_model["report_period"] = "2026_09"
    report_model["sections"] = [
        section for section in report_model["sections"] if section["section_id"] != "historical_trends"
    ]
    for section in report_model["sections"]:
        if section["section_id"] == "financial_health_overview":
            section["content"]["collection_rate"] = 0.92
    report_model["sections"].append(
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "collection_rate",
                        "metric": "Tasa de cobranza",
                        "unit": "ratio",
                        "direction": "improving",
                        "points": [
                            {"period": "2026_06", "value": 0.84},
                            {"period": "2026_07", "value": 0.85},
                            {"period": "2026_08", "value": 0.90},
                        ],
                    }
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_09.json"],
            "warnings": [],
        }
    )
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)

    chart_labels = [
        row["period_label"]
        for chart in fake_st.vega_lite_charts
        for row in chart["spec"]["data"]["values"]
    ]
    assert "Jun 2026" in chart_labels
    assert "Jul 2026" in chart_labels
    assert "Ago 2026" in chart_labels
    assert "Sep 2026" in chart_labels
    assert fake_st.vega_lite_charts


def test_streamlit_chart_input_preserves_exact_september_revenue_and_expense_points() -> None:
    """Verify Streamlit receives exact Apr-Sep chart points."""

    report_model = _july_report_model()
    report_model["period_slug"] = "2026_09"
    report_model["report_period"] = "2026_09"
    report_model["sections"] = [
        section for section in report_model["sections"] if section["section_id"] != "historical_trends"
    ]
    for section in report_model["sections"]:
        if section["section_id"] == "financial_health_overview":
            section["content"].update({"total_revenue": 2_123_856.0, "total_expenses": 2_096_356.0})
    report_model["sections"].append(
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "total_revenue",
                        "metric": "Ingresos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_018_940.0},
                            {"period": "2026_05", "value": 2_005_584.0},
                            {"period": "2026_06", "value": 1_992_060.0},
                            {"period": "2026_07", "value": 2_021_376.0},
                            {"period": "2026_08", "value": 2_072_448.0},
                        ],
                    },
                    {
                        "metric_id": "total_expenses",
                        "metric": "Gastos",
                        "unit": "USD",
                        "points": [
                            {"period": "2026_04", "value": 2_084_940.0},
                            {"period": "2026_05", "value": 2_126_584.0},
                            {"period": "2026_06", "value": 2_366_060.0},
                            {"period": "2026_07", "value": 2_213_876.0},
                            {"period": "2026_08", "value": 2_138_448.0},
                        ],
                    },
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_09.json"],
            "warnings": [],
        }
    )
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)

    specs_by_metric = {
        chart["spec"]["data"]["values"][0]["metric_id"]: chart["spec"]
        for chart in fake_st.vega_lite_charts
        if chart["spec"].get("data", {}).get("values")
    }
    captured = {
        metric: [
            (str(row.get("period")), float(row.get("value")))
            for row in specs_by_metric[metric]["data"]["values"]
        ]
        for metric in ("total_revenue", "total_expenses")
    }

    assert captured["total_revenue"] == [
        ("2026_04", 2_018_940.0),
        ("2026_05", 2_005_584.0),
        ("2026_06", 1_992_060.0),
        ("2026_07", 2_021_376.0),
        ("2026_08", 2_072_448.0),
        ("2026_09", 2_123_856.0),
    ]
    assert captured["total_expenses"] == [
        ("2026_04", 2_084_940.0),
        ("2026_05", 2_126_584.0),
        ("2026_06", 2_366_060.0),
        ("2026_07", 2_213_876.0),
        ("2026_08", 2_138_448.0),
        ("2026_09", 2_096_356.0),
    ]
    assert len(captured["total_revenue"]) == 6
    assert len(captured["total_expenses"]) == 6
    for metric in ("total_revenue", "total_expenses"):
        spec = specs_by_metric[metric]
        assert spec["mark"]["point"]
        assert "transform" not in spec
        assert "aggregate" not in json.dumps(spec)
        assert spec["encoding"]["x"]["axis"]["values"] == [
            "Abr 2026",
            "May 2026",
            "Jun 2026",
            "Jul 2026",
            "Ago 2026",
            "Sep 2026",
        ]


def test_september_artifact_streamlit_specs_keep_all_monthly_points() -> None:
    """Verify the real September artifact reaches Streamlit as six-point charts."""

    path = Path("outputs/report/report_model_2026_09.json")
    if not path.is_file():
        pytest.skip("September report model artifact is not available.")
    report_model = streamlit_app._load_json(path)
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)

    specs_by_metric = {
        chart["spec"]["data"]["values"][0]["metric_id"]: chart["spec"]
        for chart in fake_st.vega_lite_charts
        if chart["spec"].get("data", {}).get("values")
    }
    assert len(fake_st.vega_lite_charts) == 7
    assert [
        (row["period_label"], float(row["value"]))
        for row in specs_by_metric["total_revenue"]["data"]["values"]
    ] == [
        ("Abr 2026", 2_018_940.0),
        ("May 2026", 2_005_584.0),
        ("Jun 2026", 1_992_060.0),
        ("Jul 2026", 2_021_376.0),
        ("Ago 2026", 2_072_448.0),
        ("Sep 2026", 2_123_856.0),
    ]
    assert [
        (row["period_label"], float(row["value"]))
        for row in specs_by_metric["total_expenses"]["data"]["values"]
    ] == [
        ("Abr 2026", 2_084_940.0),
        ("May 2026", 2_126_584.0),
        ("Jun 2026", 2_366_060.0),
        ("Jul 2026", 2_213_876.0),
        ("Ago 2026", 2_138_448.0),
        ("Sep 2026", 2_096_356.0),
    ]
    for spec in specs_by_metric.values():
        rows = spec["data"]["values"]
        assert len(rows) == 6
        assert any(row["period_label"] == "Jul 2026" for row in rows)
        assert any(row["period_label"] == "Ago 2026" for row in rows)
        assert spec["mark"]["point"]
        assert "transform" not in spec
        assert "aggregate" not in json.dumps(spec)


def test_analysis_tab_uses_one_point_history_fallback_card() -> None:
    """Verify Streamlit avoids empty charts when only one trend point exists."""

    report_model = _july_report_model()
    report_model["period_slug"] = "2026_06"
    report_model["report_period"] = "2026_06"
    report_model["sections"] = [
        section for section in report_model["sections"] if section["section_id"] != "historical_trends"
    ]
    report_model["sections"].append(
        {
            "section_id": "historical_trends",
            "title": "Historical Trends",
            "content": {
                "trend_series": [
                    {
                        "metric_id": "collection_rate",
                        "metric": "Tasa de cobranza",
                        "unit": "ratio",
                        "direction": "stable",
                        "points": [{"period": "2026_06", "value": 0.84}],
                    }
                ]
            },
            "source_references": ["outputs/analysis/strategic_analysis_2026_06.json"],
            "warnings": [],
        }
    )
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_analysis_tab(fake_st, report_model)
    html = "\n".join(fake_st.markdown_calls)

    assert "Historial insuficiente" in html
    assert "ui-trend-chart" not in html


def test_semantic_badges_show_variety_in_results_dashboard() -> None:
    """Verify cards use meaningful executive badges instead of all Informativo."""

    report_model = _october_report_model()
    result = _october_result()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_results_header(
        fake_st,
        report_model=report_model,
        result=result,
        artifacts=streamlit_app._artifact_paths(result),
    )
    streamlit_app._render_attention_summary(fake_st, report_model)
    streamlit_app._render_analysis_tab(fake_st, report_model)

    visible = "\n".join(fake_st.markdown_calls)
    assert "Verificado" in visible
    assert "Favorable" in visible or "Estable" in visible
    assert "Riesgo" in visible or "Requiere atención" in visible or "Advertencia" in visible
    assert visible.count("Informativo") < 4


def test_october_results_do_not_expose_raw_dicts_or_internal_ids() -> None:
    """Verify normal result tabs do not show raw objects, tools, or canonical IDs."""

    report_model = _october_report_model()
    result = _october_result()
    fake_st = FakeStreamlitRenderer()

    streamlit_app._render_results(fake_st, result)

    visible = "\n".join(fake_st.markdown_calls + fake_st.info_messages + fake_st.success_messages + fake_st.warning_messages)
    forbidden = ("{'", "total_revenue", "payroll_percentage_of_revenue", "get_metric_history", "Report model and renderers", "Historical context", "monthly")
    for text in forbidden:
        assert text not in visible
    assert fake_st.downloads
    assert not fake_st.error_messages


def test_custom_card_css_uses_theme_safe_light_and_dark_colors() -> None:
    """Verify responsive cards retain contrast in light and dark themes."""

    class FakeStreamlit:
        """Capture CSS emitted by the app."""

        def __init__(self) -> None:
            """Create a markdown capture list."""

            self.markdown_calls: list[str] = []

        def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
            """Capture CSS markdown."""

            self.markdown_calls.append(text)

    fake_st = FakeStreamlit()
    streamlit_app._apply_page_styles(fake_st)
    css = "\n".join(fake_st.markdown_calls)

    assert ".ui-responsive-grid" in css
    assert "var(--fa-surface)" in css
    assert "var(--fa-text)" in css
    assert "border-radius: 18px" in css
    assert "0 6px 18px" in css
    assert "@media (prefers-color-scheme: dark)" in css
    assert ".ui-status-risk" in css
    assert ".ui-status-verified" in css


def test_no_stale_undefined_download_helper_variables_remain() -> None:
    """Verify stale redesign variables are not referenced in download cards."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")
    start = source.index("def _render_download_card")
    end = source.index("def _stage_status", start)
    body = source[start:end]

    assert "badge or variant" not in body
    assert "row_html" not in body
    assert "card_class" not in body
    assert "tone" not in body


def test_report_model_sections_are_mapped_to_streamlit_tabs() -> None:
    """Verify July-style deterministic report sections have UI representation."""

    section_ids = [
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
        "historical_summary",
        "historical_trends",
        "recommendation_follow_up",
        "longitudinal_risk_assessment",
        "missing_information",
        "appendix",
    ]
    model = {"sections": [{"section_id": section_id, "title": section_id} for section_id in section_ids]}

    rows = streamlit_app._ui_section_consistency_rows(
        model,
        html_available=True,
        pdf_available=True,
    )

    assert all(row["present_in_streamlit"] for row in rows)
    assert {row["streamlit_tab"] for row in rows} >= {
        "Resumen",
        "KPIs",
        "Anomalías",
        "Análisis",
        "Recomendaciones",
        "Descargas",
    }


def test_results_tabs_include_analysis_and_downloads() -> None:
    """Verify the normal results layout exposes the full report dashboard."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")

    assert '["Resumen", "KPIs", "Anomalías", "Análisis", "Recomendaciones", "Descargas"]' in source
    assert "_render_analysis_tab" in source
    assert "Evidence package JSON" in source


def test_visible_ui_source_does_not_intentionally_render_raw_objects() -> None:
    """Verify custom components use safe display helpers for dict/list values."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")

    assert "escape(str(card.get(\"badge\")" not in source
    assert "_safe_display_text" in source
    assert "_status_badge_html" in source


def test_friendly_stage_error_distinguishes_historical_context_failure() -> None:
    """Verify memory-filter errors do not appear as strategic validation failures."""

    message = streamlit_app._friendly_stage_error("department contains unsupported characters")

    assert "historial" in message
    assert "estratÃ©gico" not in message


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
    assert "--fa-text: #172033" in source
    assert "--fa-muted: #526273" in source
    assert "--fa-text: #f5f7fa" in source
    assert "--fa-muted: #c5d0dc" in source
    assert "color: white" not in source
    assert "color: #fff" not in source


def test_card_variants_use_accent_borders_not_full_solid_alert_backgrounds() -> None:
    """Verify semantic cards stay restrained and reserve solid colors for chips."""

    source = Path(streamlit_app.__file__).read_text(encoding="utf-8")

    assert ".ui-card--negative { border-left-color: var(--fa-negative); background: var(--fa-surface); }" in source
    assert ".ui-card--positive { border-left-color: var(--fa-positive); background: var(--fa-surface); }" in source
    assert "background: #b42318" not in source
    assert "background: #1b7f4a" not in source


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

    assert "Modo de resultado" in fake_st.info_messages[0]
    assert "reutiliz" in fake_st.info_messages[0]
    assert "Tiempo total" in fake_st.info_messages[0]
    assert fake_st.tables[0][0]["Estado"] == "Omitido"
    assert fake_st.tables[-1][0]["Tamaño contexto"] == 100
