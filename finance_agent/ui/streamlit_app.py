"""Streamlit v1 interface for running the Finance AI Agent pipeline."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from finance_agent.llm.ollama_client import DEFAULT_OLLAMA_ENDPOINT
from finance_agent.orchestration import (
    DEFAULT_OLLAMA_MODEL,
    EXPERIMENTAL_FAST_OLLAMA_MODEL,
    PipelineConfig,
    PipelineInputModel,
    PipelineRunResult,
    build_pipeline_input_model,
    run_pipeline_for_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "ui_uploads"
FINANCIAL_REPORT_UPLOAD_TYPES = ("xlsx", "xls", "csv")
GOALS_UPLOAD_TYPES = ("pdf", "docx", "xlsx", "xls")


class UploadedFileLike(Protocol):
    """Protocol for Streamlit uploaded files used by this thin UI layer."""

    name: str

    def getbuffer(self) -> memoryview:
        """Return the uploaded file bytes.

        Inputs: none.
        Outputs: memoryview containing uploaded file content.
        Assumptions: Streamlit's UploadedFile implements this API.
        """


@dataclass(frozen=True)
class StreamlitRunSettings:
    """User-configurable settings for one Streamlit-triggered pipeline run.

    Inputs: report language, optional period override, and Ollama/runtime settings.
    Outputs: immutable settings used to construct PipelineConfig.
    Assumptions: UI validation keeps values in a practical range before running.
    """

    report_language: str = "es"
    period_override: str | None = None
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    structure_ollama_model: str | None = None
    planner_ollama_model: str | None = None
    analysis_ollama_model: str | None = None
    ollama_timeout_seconds: float = 180.0
    stage_timeout_seconds: float = 420.0


PipelineRunner = Callable[[PipelineInputModel, PipelineConfig], PipelineRunResult]


def _safe_upload_name(filename: str) -> str:
    """Return a filesystem-safe upload filename.

    Inputs: original uploaded filename.
    Outputs: sanitized filename preserving the extension when possible.
    Assumptions: this protects the local output folder from path traversal.
    """

    name = Path(filename).name.strip() or "uploaded_file"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_uploaded_file(uploaded_file: UploadedFileLike, destination_dir: Path) -> Path:
    """Persist one Streamlit uploaded file for orchestrator consumption.

    Inputs: Streamlit UploadedFile-like object and destination directory.
    Outputs: path to the written local copy.
    Assumptions: the orchestrator expects paths, so uploads must be materialized.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / _safe_upload_name(uploaded_file.name)
    path.write_bytes(bytes(uploaded_file.getbuffer()))
    return path


def build_input_model_from_uploads(
    *,
    financial_report_path: Path,
    goals_document_path: Path,
    settings: StreamlitRunSettings,
) -> PipelineInputModel:
    """Build the generic pipeline input model from saved upload paths.

    Inputs: saved report/goals paths and UI settings.
    Outputs: PipelineInputModel produced by the shared period-detection layer.
    Assumptions: period detection and validation remain owned by orchestration.
    """

    return build_pipeline_input_model(
        financial_report_path=financial_report_path,
        goals_document_path=goals_document_path,
        period_override=settings.period_override,
        report_language=settings.report_language,
    )


def build_pipeline_config(
    input_model: PipelineInputModel,
    settings: StreamlitRunSettings,
) -> PipelineConfig:
    """Create the orchestrator configuration for one UI run.

    Inputs: generic input model and UI settings.
    Outputs: PipelineConfig using the current Python executable and repo paths.
    Assumptions: the UI preserves existing output locations under outputs/.
    """

    return PipelineConfig.from_project_root(
        PROJECT_ROOT,
        python_executable=sys.executable,
        ollama_endpoint=settings.ollama_endpoint,
        ollama_model=settings.ollama_model,
        structure_ollama_model=settings.structure_ollama_model,
        planner_ollama_model=settings.planner_ollama_model,
        analysis_ollama_model=settings.analysis_ollama_model,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
        stage_timeout_seconds=settings.stage_timeout_seconds,
        input_model=input_model,
    )


def run_analysis_from_files(
    *,
    financial_report_path: Path,
    goals_document_path: Path,
    settings: StreamlitRunSettings,
    runner: PipelineRunner = run_pipeline_for_report,
) -> PipelineRunResult:
    """Run the existing pipeline for saved upload files.

    Inputs: saved report/goals paths, UI settings, and injectable runner.
    Outputs: structured PipelineRunResult.
    Assumptions: this function is the only place the UI triggers pipeline work.
    """

    input_model = build_input_model_from_uploads(
        financial_report_path=financial_report_path,
        goals_document_path=goals_document_path,
        settings=settings,
    )
    config = build_pipeline_config(input_model, settings)
    return runner(input_model, config)


def _period_override_from_selection(selection: str, value: str) -> str | None:
    """Convert period override widgets into the orchestrator override string.

    Inputs: selected mode and optional user-entered value.
    Outputs: None for Auto, otherwise the stripped override text.
    Assumptions: pipeline period parsing/validation remains downstream.
    """

    if selection == "Auto":
        return None
    return value.strip() or None


def _load_json(path: Path | str | None) -> dict[str, Any]:
    """Read one JSON artifact if it exists.

    Inputs: optional artifact path.
    Outputs: parsed dictionary or empty dict.
    Assumptions: UI should fail softly when optional outputs are absent.
    """

    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_output(result: PipelineRunResult, suffix: str) -> Path | None:
    """Find an output file from a pipeline result by filename suffix.

    Inputs: pipeline result and expected filename suffix.
    Outputs: matching Path or None.
    Assumptions: output filenames remain stable, while roots may vary.
    """

    for output_file in result.output_files:
        path = Path(output_file)
        if path.name.endswith(suffix):
            return path
    return None


def _section_by_id(report_model: dict[str, Any], section_id: str) -> dict[str, Any]:
    """Return one section from a renderer-agnostic report model.

    Inputs: report model dictionary and section ID.
    Outputs: section dictionary or empty dict.
    Assumptions: report model schema is validated upstream by reporting code.
    """

    sections = report_model.get("sections", [])
    for section in sections if isinstance(sections, list) else []:
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return section
    return {}


def _artifact_paths(result: PipelineRunResult) -> dict[str, Path | None]:
    """Collect the downloadable artifacts produced by one pipeline run.

    Inputs: PipelineRunResult.
    Outputs: mapping from artifact label to optional path.
    Assumptions: report model naming exposes the period slug in the output name.
    """

    report_model = next(
        (Path(path) for path in result.output_files if Path(path).name.startswith("report_model_")),
        None,
    )
    period_suffix = ""
    if report_model is not None:
        period_suffix = report_model.stem.replace("report_model_", "")
    return {
        "PDF": _find_output(result, f"financial_report_{period_suffix}.pdf"),
        "HTML": _find_output(result, f"financial_report_{period_suffix}.html"),
        "Report model JSON": report_model,
        "Strategic analysis JSON": _find_output(result, f"strategic_analysis_{period_suffix}.json"),
    }


def _render_stage_results(st: Any, result: PipelineRunResult) -> None:
    """Render pipeline stage statuses from orchestrator results.

    Inputs: Streamlit module and pipeline result.
    Outputs: stage status table in the UI.
    Assumptions: stages are complete when this function is called.
    """

    rows = [
        {
            "Stage": stage.display_name,
            "Status": "Skipped" if stage.skipped else "OK" if stage.success else "Failed",
            "Critical": "Yes" if stage.critical else "No",
            "Runtime (s)": round(stage.runtime_seconds, 2),
            "Warnings": "; ".join(stage.warnings),
            "Error": stage.error or "",
        }
        for stage in result.stages
    ]
    st.subheader("Pipeline progress")
    cache_label = "hit" if result.cache_hit else "miss"
    st.info(f"Pipeline cache: {cache_label}")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    ollama_rows = [
        row
        for row in rows
        if "Ollama" in row["Stage"] or row["Stage"] == "Strategic analysis"
    ]
    if ollama_rows:
        st.subheader("Ollama stage runtimes")
        st.dataframe(ollama_rows, use_container_width=True, hide_index=True)


def _render_overview_tab(st: Any, report_model: dict[str, Any], result: PipelineRunResult) -> None:
    """Render the Overview tab from report and pipeline metadata.

    Inputs: Streamlit module, report model, and result.
    Outputs: overview content.
    Assumptions: no calculations are performed in the UI.
    """

    input_model = result.config.input_model
    detected = input_model.detected_period if input_model else None
    executive = _section_by_id(report_model, "executive_summary").get("content", {})
    health = _section_by_id(report_model, "financial_health_overview").get("content", {})
    st.markdown("### Executive summary")
    st.write(executive.get("summary") or "Executive summary unavailable.")
    if detected:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Detected period", detected.label)
        col_b.metric("Period type", detected.period_type)
        col_c.metric("Confidence", f"{detected.confidence:.0%}")
    st.markdown("### Financial health")
    st.json(health, expanded=False)


def _render_kpi_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render KPI rows from the report model.

    Inputs: Streamlit module and report model.
    Outputs: KPI table.
    Assumptions: KPIs were calculated upstream.
    """

    content = _section_by_id(report_model, "kpi_overview").get("content", {})
    st.dataframe(content.get("kpis", []), use_container_width=True, hide_index=True)


def _render_anomaly_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render anomaly summary rows from the report model.

    Inputs: Streamlit module and report model.
    Outputs: anomaly severity and top anomaly tables.
    Assumptions: anomaly detection already ran in Python.
    """

    content = _section_by_id(report_model, "anomaly_summary").get("content", {})
    st.markdown("### Severity summary")
    st.json(content.get("anomalies_by_severity", {}), expanded=False)
    st.markdown("### Top anomalies")
    st.dataframe(content.get("top_anomalies", []), use_container_width=True, hide_index=True)


def _render_recommendations_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render strategic analysis fields from the report model.

    Inputs: Streamlit module and report model.
    Outputs: root causes, priorities, recommendations, and missing info.
    Assumptions: strategic reasoning was generated and validated upstream.
    """

    recommendations = _section_by_id(report_model, "strategic_recommendations").get(
        "content",
        {},
    )
    missing = _section_by_id(report_model, "missing_information").get("content", {})
    st.markdown("### Root causes")
    for item in recommendations.get("root_causes", []):
        st.write(f"- {item}")
    st.markdown("### Strategic priorities")
    for item in recommendations.get("strategic_priorities", []):
        st.write(f"- {item}")
    st.markdown("### Recommendations")
    for item in recommendations.get("recommendations", []):
        st.write(item if isinstance(item, str) else item.get("action", item))
    missing_items = missing.get("missing_information", [])
    if missing_items:
        st.warning("Missing information remains:")
        for item in missing_items:
            st.write(f"- {item}")
    else:
        st.success("No missing information is currently reported.")


def _render_downloads_tab(st: Any, artifacts: dict[str, Path | None]) -> None:
    """Render download buttons for generated artifacts.

    Inputs: Streamlit module and artifact path mapping.
    Outputs: download buttons or availability messages.
    Assumptions: files are generated by the pipeline, not by the UI.
    """

    mime_by_label = {
        "PDF": "application/pdf",
        "HTML": "text/html",
        "Report model JSON": "application/json",
        "Strategic analysis JSON": "application/json",
    }
    for label, path in artifacts.items():
        if path and path.is_file():
            st.download_button(
                label=f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime_by_label[label],
            )
        else:
            st.info(f"{label} is not available for this run.")


def _render_results(st: Any, result: PipelineRunResult) -> None:
    """Render all result tabs for a completed pipeline run.

    Inputs: Streamlit module and pipeline result.
    Outputs: tabbed results area.
    Assumptions: report artifacts are read-only presentation data.
    """

    artifacts = _artifact_paths(result)
    report_model = _load_json(artifacts["Report model JSON"])
    overview, kpis, anomalies, recommendations, downloads = st.tabs(
        ["Overview", "KPIs", "Anomalies", "Recommendations", "Downloads"]
    )
    with overview:
        _render_overview_tab(st, report_model, result)
    with kpis:
        _render_kpi_tab(st, report_model)
    with anomalies:
        _render_anomaly_tab(st, report_model)
    with recommendations:
        _render_recommendations_tab(st, report_model)
    with downloads:
        _render_downloads_tab(st, artifacts)


def main() -> None:
    """Run the Streamlit Finance AI Agent UI.

    Inputs: user uploads and UI controls.
    Outputs: rendered app that invokes the shared pipeline once per button click.
    Assumptions: Streamlit is installed in the active Python environment.
    """

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by launch check.
        raise RuntimeError(
            "Streamlit is not installed. Install project dependencies with "
            "`pip install -r requirements.txt` before launching the UI."
        ) from exc

    st.set_page_config(
        page_title="Finance AI Agent",
        page_icon="📊",
        layout="wide",
    )
    st.title("Finance AI Agent")
    st.caption("Upload one financial report and one goals document, then run the existing pipeline.")

    with st.sidebar:
        st.header("Inputs")
        financial_report = st.file_uploader(
            "Financial report",
            type=FINANCIAL_REPORT_UPLOAD_TYPES,
        )
        goals_document = st.file_uploader(
            "Goals document",
            type=GOALS_UPLOAD_TYPES,
        )
        language = st.selectbox("Report language", options=("es", "en"), index=0)
        override_mode = st.selectbox(
            "Period override",
            options=("Auto", "Monthly", "Quarterly", "Semester", "Annual", "Custom"),
            index=0,
        )
        override_value = ""
        if override_mode != "Auto":
            override_value = st.text_input(
                "Override value",
                placeholder="Examples: 2026-06, 2026-Q2, 2026-S1, 2026",
            )
        with st.expander("Advanced settings", expanded=False):
            endpoint = st.text_input("Ollama endpoint", value=DEFAULT_OLLAMA_ENDPOINT)
            model = st.text_input(
                "Ollama model",
                value=DEFAULT_OLLAMA_MODEL,
                help="Supported default: one model for every Ollama stage.",
            )
            experimental_models = st.checkbox(
                "Experimental: use separate models per Ollama stage",
                value=False,
                help="Benchmarking showed this was slower on the current machine.",
            )
            structure_model = planner_model = analysis_model = None
            if experimental_models:
                structure_model = st.text_input(
                    "Structure fallback model",
                    value=EXPERIMENTAL_FAST_OLLAMA_MODEL,
                )
                planner_model = st.text_input(
                    "Investigation planner model",
                    value=EXPERIMENTAL_FAST_OLLAMA_MODEL,
                )
                analysis_model = st.text_input(
                    "Strategic analysis model",
                    value=DEFAULT_OLLAMA_MODEL,
                )
            ollama_timeout = st.number_input(
                "Ollama timeout seconds",
                min_value=5.0,
                max_value=900.0,
                value=180.0,
                step=5.0,
            )
            stage_timeout = st.number_input(
                "Stage timeout seconds",
                min_value=30.0,
                max_value=1800.0,
                value=420.0,
                step=30.0,
            )
        run_button = st.button("Run Analysis", type="primary", use_container_width=True)

    if not run_button:
        st.info("Upload both files and click Run Analysis to start.")
        return
    if financial_report is None or goals_document is None:
        st.error("Please upload both a financial report and a goals document.")
        return

    run_dir = UPLOAD_ROOT / time.strftime("%Y%m%d_%H%M%S")
    report_path = save_uploaded_file(financial_report, run_dir)
    goals_path = save_uploaded_file(goals_document, run_dir)
    settings = StreamlitRunSettings(
        report_language=language,
        period_override=_period_override_from_selection(override_mode, override_value),
        ollama_endpoint=endpoint,
        ollama_model=model.strip() or DEFAULT_OLLAMA_MODEL,
        structure_ollama_model=structure_model.strip() if structure_model else None,
        planner_ollama_model=planner_model.strip() if planner_model else None,
        analysis_ollama_model=analysis_model.strip() if analysis_model else None,
        ollama_timeout_seconds=float(ollama_timeout),
        stage_timeout_seconds=float(stage_timeout),
    )

    try:
        with st.spinner("Running Finance AI Agent pipeline..."):
            result = run_analysis_from_files(
                financial_report_path=report_path,
                goals_document_path=goals_path,
                settings=settings,
            )
    except Exception as exc:  # noqa: BLE001 - UI must display graceful failures.
        st.error(f"Pipeline could not start: {exc}")
        return

    if result.success:
        st.success("Pipeline completed successfully.")
    else:
        st.error("Pipeline completed with a critical failure.")
    if result.warnings:
        st.warning("Warnings: " + "; ".join(result.warnings))
    _render_stage_results(st, result)
    _render_results(st, result)


if __name__ == "__main__":
    main()
