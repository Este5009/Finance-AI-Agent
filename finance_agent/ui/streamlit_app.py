"""Streamlit v1 interface for running the Finance AI Agent pipeline."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Protocol

from finance_agent.llm.ollama_client import DEFAULT_OLLAMA_ENDPOINT
from finance_agent.memory.repository import MemoryRepository
from finance_agent.orchestration import (
    DEFAULT_OLLAMA_MODEL,
    EXPERIMENTAL_FAST_OLLAMA_MODEL,
    PipelineConfig,
    PipelineInputModel,
    PipelineProgressCallback,
    PipelineProgressEvent,
    PipelineRunResult,
    build_pipeline_input_model,
    run_pipeline_for_report,
)
from finance_agent.reporting.presentation import build_presentation_view


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "ui_uploads"
FINANCIAL_REPORT_UPLOAD_TYPES = ("xlsx", "xls", "csv")
GOALS_UPLOAD_TYPES = ("pdf", "docx", "xlsx", "xls")
SUPPORTED_UI_PERIOD_OPTIONS = ("Detectar automáticamente", "Mensual")
SPANISH_MONTHS = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dic",
}


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
    ollama_timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 600.0
    stage_timeout_seconds: float = 900.0
    ollama_keep_alive: str = "15m"
    max_planner_anomalies: int = 5
    compact_context: bool = True
    deduplicate_context: bool = True
    enable_cache: bool = True
    enable_memory_storage: bool = True
    memory_database_path: Path | None = None
    source_revision_confirmed: bool = False


PipelineRunner = Callable[..., PipelineRunResult]


PROGRESS_UI_STAGES: tuple[dict[str, str], ...] = (
    {
        "stage_id": "validate_documents",
        "label": "Validando documentos",
        "detail": "Comprobando archivos, periodo y configuración antes de ejecutar el análisis.",
    },
    {
        "stage_id": "prepare_interpret_files",
        "label": "Preparando e interpretando archivos",
        "detail": "Leyendo el reporte y normalizando la estructura financiera disponible.",
    },
    {
        "stage_id": "calculate_financial_indicators",
        "label": "Calculando indicadores financieros",
        "detail": "Calculando KPIs, presupuestos, variaciones y resultados del periodo.",
    },
    {
        "stage_id": "analyze_financial_performance",
        "label": "Analizando el desempeño financiero",
        "detail": "Detectando anomalías y preparando el plan de investigación validado.",
    },
    {
        "stage_id": "query_history",
        "label": "Consultando el historial",
        "detail": "Recuperando evidencia y contexto histórico relevante para el periodo.",
    },
    {
        "stage_id": "generate_strategic_recommendations",
        "label": "Generando recomendaciones estratégicas",
        "detail": "Ollama está procesando evidencia compacta y validada; esta etapa puede tardar varios minutos.",
    },
    {
        "stage_id": "build_executive_report",
        "label": "Construyendo el reporte ejecutivo",
        "detail": "Armando el modelo de reporte y generando los archivos HTML/PDF.",
    },
    {
        "stage_id": "save_results",
        "label": "Guardando resultados",
        "detail": "Registrando artefactos, caché y memoria histórica cuando corresponde.",
    },
    {
        "stage_id": "analysis_completed",
        "label": "Análisis completado",
        "detail": "El análisis finalizó y los reportes están listos para revisar o descargar.",
    },
)


STAGE_LABELS_ES: dict[str, str] = {
    "ingestion": "Procesando archivos",
    "document_understanding": "Entendiendo estructura del reporte",
    "finance_calculations": "Calculando indicadores",
    "anomaly_detection": "Detectando anomalías",
    "ollama_structure_fallback": "Revisando estructura dudosa",
    "ollama_investigation_planner": "Preparando investigación",
    "retrieval_layer": "Consultando historial",
    "strategic_analysis": "Generando análisis estratégico",
    "report_engine": "Creando modelo de reporte",
    "report_renderer": "Creando reporte descargable",
    "pipeline_cache": "Reutilizando análisis existente",
    "memory_storage": "Guardando memoria histórica",
}

STATUS_LABELS_ES: dict[str, str] = {
    "ok": "Completado",
    "skipped": "Omitido",
    "failed": "Requiere atención",
}


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
        source_revision_confirmed=settings.source_revision_confirmed,
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
        connect_timeout_seconds=settings.connect_timeout_seconds,
        read_timeout_seconds=settings.read_timeout_seconds,
        stage_timeout_seconds=settings.stage_timeout_seconds,
        ollama_keep_alive=settings.ollama_keep_alive,
        max_planner_anomalies=settings.max_planner_anomalies,
        compact_context=settings.compact_context,
        deduplicate_context=settings.deduplicate_context,
        enable_cache=settings.enable_cache,
        enable_memory_storage=settings.enable_memory_storage,
        memory_database_path=settings.memory_database_path,
        input_model=input_model,
    )


def run_analysis_from_files(
    *,
    financial_report_path: Path,
    goals_document_path: Path,
    settings: StreamlitRunSettings,
    runner: PipelineRunner = run_pipeline_for_report,
    progress_callback: PipelineProgressCallback | None = None,
) -> PipelineRunResult:
    """Run the existing pipeline for saved upload files.

    Inputs: saved report/goals paths, UI settings, injectable runner, and optional progress callback.
    Outputs: structured PipelineRunResult.
    Assumptions: this function is the only place the UI triggers pipeline work.
    """

    input_model = build_input_model_from_uploads(
        financial_report_path=financial_report_path,
        goals_document_path=goals_document_path,
        settings=settings,
    )
    config = build_pipeline_config(input_model, settings)
    if progress_callback is None or not _runner_accepts_progress_callback(runner):
        return runner(input_model, config)
    return runner(input_model, config, progress_callback=progress_callback)


def _runner_accepts_progress_callback(runner: PipelineRunner) -> bool:
    """Return whether an injectable runner accepts progress callbacks.

    Inputs: callable runner used by tests or the real orchestrator.
    Outputs: True when ``progress_callback`` can be passed safely.
    Assumptions: older two-argument test doubles should keep working unchanged.
    """

    try:
        parameters = signature(runner).parameters
    except (TypeError, ValueError):
        return True
    return "progress_callback" in parameters or any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _period_override_from_selection(selection: str, value: str) -> str | None:
    """Convert period override widgets into the orchestrator override string.

    Inputs: selected mode and optional user-entered value.
    Outputs: None for automatic detection, otherwise the stripped override text.
    Assumptions: pipeline period parsing/validation remains downstream.
    """

    if selection in {"Auto", "Detectar automáticamente"}:
        return None
    return value.strip() or None


def _is_monthly_period_text(value: str) -> bool:
    """Return whether text is a supported monthly override.

    Inputs: user-entered period text.
    Outputs: True for YYYY-MM or YYYY_MM with month 01..12.
    Assumptions: Streamlit supports monthly execution only for now.
    """

    match = re.match(r"^20\d{2}[-_](0[1-9]|1[0-2])$", str(value or "").strip())
    return bool(match)


def _spanish_month_label(year: int | None, month: int | None) -> str:
    """Return a compact Spanish month/year label.

    Inputs: optional year and month number.
    Outputs: label like ``Dic 2026`` or empty string.
    Assumptions: callers show raw period text only when parsed metadata is absent.
    """

    if not year or not month or month not in SPANISH_MONTHS:
        return ""
    return f"{SPANISH_MONTHS[month]} {year}"


def _monthly_readiness_message(
    *,
    input_model: PipelineInputModel | None,
    override_mode: str,
    override_value: str,
) -> tuple[bool, str]:
    """Determine whether the UI can run the currently supported monthly path.

    Inputs: optional detected input model and period selector state.
    Outputs: ready flag and Spanish explanation.
    Assumptions: quarterly/semester/annual/custom comparisons are intentionally disabled.
    """

    if override_mode == "Mensual":
        if not _is_monthly_period_text(override_value):
            return False, "Indique el mes y año en formato 2026-12 para ejecutar un reporte mensual."
        return True, f"Período seleccionado: Mensual — {override_value.replace('_', '-')}"
    if input_model is None:
        return False, "Seleccione el reporte financiero y el documento de metas para continuar."
    detected = input_model.detected_period
    if detected.period_type == "monthly" and not detected.requires_override:
        label = _spanish_month_label(detected.year, detected.month) or detected.label
        return True, f"Período detectado: Mensual — {label}"
    return (
        False,
        "No se detectó un período mensual con suficiente confianza. Seleccione “Mensual” e indique el mes y año.",
    )


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


def _stage_display_name(stage: Any) -> str:
    """Return a Spanish stage name for administrators.

    Inputs: pipeline stage result.
    Outputs: Spanish display label.
    Assumptions: unknown stage names are sanitized but not exposed as tool names.
    """

    key = str(getattr(stage, "stage_name", "") or "")
    if key in STAGE_LABELS_ES:
        return STAGE_LABELS_ES[key]
    raw = str(getattr(stage, "display_name", "") or key).replace("_", " ")
    return raw.capitalize()


def _stage_status(stage: Any) -> tuple[str, str]:
    """Return Spanish status text and severity for a stage.

    Inputs: pipeline stage result.
    Outputs: label and CSS/status key.
    Assumptions: skipped successful stages are expected runtime behavior.
    """

    if getattr(stage, "skipped", False):
        return STATUS_LABELS_ES["skipped"], "skipped"
    if getattr(stage, "success", False):
        return STATUS_LABELS_ES["ok"], "ok"
    return STATUS_LABELS_ES["failed"], "failed"


def _is_allowed_extension(uploaded_file: UploadedFileLike | None, allowed: tuple[str, ...]) -> bool:
    """Validate an uploaded file extension.

    Inputs: uploaded file and allowed suffixes.
    Outputs: True when the file is absent or allowed.
    Assumptions: Streamlit also filters extensions, but explicit messaging helps users.
    """

    if uploaded_file is None:
        return True
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    return suffix in allowed


def _file_status_message(uploaded_file: UploadedFileLike | None, allowed: tuple[str, ...]) -> tuple[str, str]:
    """Return a Spanish validation message for one uploaded file.

    Inputs: uploaded file and allowed extensions.
    Outputs: status kind and message.
    Assumptions: byte-level validation remains with the pipeline.
    """

    if uploaded_file is None:
        return "pending", "Ningún archivo seleccionado."
    if not _is_allowed_extension(uploaded_file, allowed):
        allowed_text = ", ".join(f".{item}" for item in allowed)
        return "error", f"Formato no permitido. Use: {allowed_text}."
    size = len(uploaded_file.getbuffer())
    return "ok", f"Archivo listo: {uploaded_file.name} ({size / 1024:.1f} KB)."


def _classify_upload_for_period(
    *,
    uploaded_file: UploadedFileLike | None,
    document_type: str,
    effective_period: str | None,
    database_path: Path | None,
) -> dict[str, Any] | None:
    """Classify an uploaded file against the persistent document registry.

    Inputs: upload, document type, effective period, and SQLite path.
    Outputs: classification dictionary or None when classification is unavailable.
    Assumptions: this is a read-only preflight used before pipeline execution.
    """

    if uploaded_file is None or not effective_period or database_path is None:
        return None
    repository = MemoryRepository(database_path)
    classification = repository.classify_source_document(
        document_type=document_type,
        raw_bytes=bytes(uploaded_file.getbuffer()),
        effective_period=effective_period,
    )
    return classification.to_dict()


def _render_upload_registry_status(st: Any, label: str, classification: dict[str, Any] | None) -> None:
    """Render Spanish document-registry status for one upload.

    Inputs: Streamlit module, display label, and optional classification.
    Outputs: visible status message.
    Assumptions: classification happens before analysis and registration after success.
    """

    if not classification:
        return
    status = classification.get("status")
    message = str(classification.get("message") or "")
    if status == "duplicate":
        st.info(f"{label}: {message}")
    elif status == "revision":
        st.warning(f"{label}: {message}")
    else:
        st.success(f"{label}: Archivo nuevo.")


def _success_registration_message(
    result: PipelineRunResult,
    classifications: tuple[dict[str, Any] | None, ...],
) -> str:
    """Return the Spanish post-run registration status.

    Inputs: pipeline result and preflight document classifications.
    Outputs: concise success message for administrators.
    Assumptions: accepted persistence happens after successful quality-backed runs.
    """

    if result.cache_hit:
        return "Análisis reutilizado."
    statuses = {str(item.get("status")) for item in classifications if item}
    if "revision" in statuses:
        return "Nueva versión registrada."
    if "duplicate" in statuses:
        return "Este archivo ya fue registrado anteriormente. Se reutilizará el registro existente."
    return "Nuevo período registrado."


def _read_text_artifact(path: Path | None, limit: int = 2000) -> str:
    """Read a small preview from a text artifact.

    Inputs: optional path and character limit.
    Outputs: preview text or empty string.
    Assumptions: previews are informational; downloads provide full files.
    """

    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _apply_page_styles(st: Any) -> None:
    """Apply lightweight dashboard styling to the Streamlit app.

    Inputs: Streamlit module.
    Outputs: injected CSS.
    Assumptions: styling is presentation-only and does not affect pipeline logic.
    """

    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.1rem; max-width: 1180px; }
        div[data-testid="stMetric"] {
            background: #fbfdff;
            border: 1px solid #d8e1ea;
            border-radius: 14px;
            padding: 14px 16px;
        }
        .finance-card {
            background: #fbfdff;
            border: 1px solid #d8e1ea;
            border-radius: 16px;
            padding: 16px 18px;
            margin: 10px 0 16px;
        }
        .step-card {
            min-height: 112px;
            background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
            border: 1px solid #d8e1ea;
            border-left: 5px solid #245b89;
            border-radius: 14px;
            padding: 16px 18px;
            margin: 4px 0 8px;
            box-shadow: 0 1px 2px rgba(23, 43, 77, 0.05);
        }
        .step-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 999px;
            background: #e7f0f8;
            margin-right: 8px;
        }
        .step-title {
            color: #12314d;
            font-size: 1.02rem;
            font-weight: 700;
        }
        .step-description {
            color: #435466;
            line-height: 1.38;
            margin-top: 8px;
            font-size: 0.94rem;
        }
        .compat-card {
            background: #f8fbff;
            border: 1px solid #d8e1ea;
            border-radius: 14px;
            padding: 14px 16px;
            margin: 2px 0 14px;
            color: #33485c;
        }
        .run-action-card {
            background: #f5f9fd;
            border: 1px solid #d8e1ea;
            border-radius: 16px;
            padding: 14px 16px 10px;
            margin-top: 8px;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
        }
        div[data-testid="stFileUploaderDropzone"] small {
            display: none;
        }
        div[data-testid="stFileUploaderDropzone"] button {
            font-size: 0;
        }
        div[data-testid="stFileUploaderDropzone"] button::after {
            content: "Seleccionar archivo";
            font-size: 0.9rem;
        }
        div[data-testid="stButton"] button:disabled {
            background-color: #e4e9ef !important;
            color: #526273 !important;
            border: 1px solid #b8c4d0 !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_file_validation(st: Any, label: str, uploaded_file: UploadedFileLike | None, allowed: tuple[str, ...]) -> None:
    """Render file validation feedback.

    Inputs: Streamlit module, label, uploaded file, and allowed extensions.
    Outputs: status message.
    Assumptions: deep file validation remains with ingestion.
    """

    status, message = _file_status_message(uploaded_file, allowed)
    if status == "ok":
        st.success(f"{label}: {message}")
    elif status == "error":
        st.error(f"{label}: {message}")
    else:
        st.info(f"{label}: {message}")


def _render_workflow_intro(st: Any) -> None:
    """Render a short guided workflow explanation.

    Inputs: Streamlit module.
    Outputs: visible step cards.
    Assumptions: stage wording should be understandable without architecture context.
    """

    st.markdown("### Cómo funciona")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(
        """
        <div class='step-card'>
          <div><span class='step-icon'>📄</span><span class='step-title'>Cargue archivos</span></div>
          <div class='step-description'>Suba el reporte financiero y el documento de metas del mismo periodo.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_b.markdown(
        """
        <div class='step-card'>
          <div><span class='step-icon'>📊</span><span class='step-title'>Genere el análisis</span></div>
          <div class='step-description'>El sistema calcula indicadores, consulta historial e interpreta evidencia validada.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_c.markdown(
        """
        <div class='step-card'>
          <div><span class='step-icon'>📥</span><span class='step-title'>Descargue resultados</span></div>
          <div class='step-description'>Revise el resumen ejecutivo y descargue el PDF, HTML o JSON de auditoría.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _initial_progress_events() -> list[dict[str, Any]]:
    """Return the default progress timeline used before callbacks arrive.

    Inputs: none.
    Outputs: list of event-like dictionaries with pending statuses.
    Assumptions: the UI owns display state; the orchestrator emits real updates.
    """

    return [
        {
            "stage_id": stage["stage_id"],
            "label": stage["label"],
            "detail": stage["detail"],
            "completed_steps": 0,
            "total_steps": len(PROGRESS_UI_STAGES),
            "status": "pending",
            "elapsed_seconds": 0.0,
        }
        for stage in PROGRESS_UI_STAGES
    ]


def _progress_status_icon(status: str) -> str:
    """Return a compact icon for a progress status.

    Inputs: progress status.
    Outputs: visual indicator string.
    Assumptions: icons are presentation-only and do not encode business state.
    """

    return {
        "completed": "✅",
        "cache_hit": "♻️",
        "skipped": "⏭️",
        "failed": "⚠️",
        "running": "🔄",
        "pending": "▫️",
    }.get(status, "▫️")


def _format_elapsed_seconds(seconds: float) -> str:
    """Format elapsed seconds for the progress panel.

    Inputs: elapsed seconds.
    Outputs: Spanish duration string.
    Assumptions: precision should be useful for administrators, not forensic.
    """

    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"{minutes} min {remainder:02d} s"
    return f"{remainder} s"


def _merge_progress_event(
    events: list[dict[str, Any]],
    event: PipelineProgressEvent,
) -> list[dict[str, Any]]:
    """Merge one orchestrator event into the persisted UI progress state.

    Inputs: current event snapshots and a new structured event.
    Outputs: updated list preserving the configured stage order.
    Assumptions: stage IDs are stable internal identifiers; labels remain Spanish.
    """

    by_id = {item["stage_id"]: dict(item) for item in events}
    by_id[event.stage_id] = event.to_dict()
    return [by_id.get(stage["stage_id"], {**stage, "status": "pending"}) for stage in PROGRESS_UI_STAGES]


def _latest_active_progress(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the most relevant progress event to display as current.

    Inputs: progress events.
    Outputs: the running/failed/latest completed event.
    Assumptions: the latest non-pending event best represents user-facing state.
    """

    for event in reversed(events):
        if event.get("status") in {"running", "failed", "cache_hit", "completed", "skipped"}:
            return event
    return events[0] if events else {}


def _render_progress_panel(st: Any, placeholder: Any | None = None) -> None:
    """Render the live analysis progress panel.

    Inputs: Streamlit module and optional placeholder for in-place updates.
    Outputs: progress bar, current stage, elapsed time, and checklist.
    Assumptions: this is UI-only; percentages reflect completed major stages.
    """

    events = st.session_state.get("finance_ai_progress_events") or _initial_progress_events()
    current = _latest_active_progress(events)
    total_steps = int(current.get("total_steps") or len(PROGRESS_UI_STAGES))
    completed_steps = int(current.get("completed_steps") or 0)
    percent = int((completed_steps / max(total_steps, 1)) * 100)
    started_at = st.session_state.get("finance_ai_progress_started_at")
    completed_at = st.session_state.get("finance_ai_progress_completed_at")
    if completed_at and started_at:
        elapsed = float(completed_at) - float(started_at)
    elif started_at:
        elapsed = time.time() - float(started_at)
    else:
        elapsed = float(current.get("elapsed_seconds") or 0.0)

    target = placeholder.container() if placeholder is not None else st.container()
    with target:
        st.markdown("### Progreso del análisis")
        st.progress(percent, text=f"{current.get('label', 'Preparando análisis')} — {percent}%")
        status = str(current.get("status") or "running")
        if status == "failed":
            st.error(f"{current.get('label')}: {current.get('detail')}")
        elif status == "cache_hit":
            st.info(f"{current.get('label')}: {current.get('detail')}")
        elif status == "completed" and current.get("stage_id") == "analysis_completed":
            st.success(f"Análisis completado en {_format_elapsed_seconds(elapsed)}.")
        else:
            st.info(f"{current.get('label')}: {current.get('detail')}")
        st.caption(f"Tiempo transcurrido: {_format_elapsed_seconds(elapsed)}")
        rows = []
        for event in events:
            rows.append(
                {
                    "Estado": _progress_status_icon(str(event.get("status") or "pending")),
                    "Etapa": event.get("label"),
                    "Detalle": event.get("detail"),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


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

    rows = []
    for stage in result.stages:
        status, _ = _stage_status(stage)
        rows.append(
            {
                "Paso": _stage_display_name(stage),
                "Estado": status,
                "Tipo": "Obligatorio" if stage.critical else "Complementario",
                "Tiempo (s)": round(stage.runtime_seconds, 2),
                "Avisos": _friendly_stage_warning(stage.warnings),
                "Error accionable": _friendly_stage_error(stage.error),
            }
        )
    st.subheader("Progreso del análisis")
    cache_label = "se reutilizó un análisis existente" if result.cache_hit else "análisis nuevo"
    st.info(f"Reutilización: {cache_label}.")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    ollama_rows = [
        row
        for row in rows
        if "análisis" in row["Paso"].lower() or "investigación" in row["Paso"].lower()
    ]
    if ollama_rows:
        st.subheader("Tiempos de interpretación estratégica")
        st.dataframe(ollama_rows, use_container_width=True, hide_index=True)
    telemetry_rows = [
        {
            "Paso": _stage_display_name(stage),
            "Carga modelo (s)": round(
                float(stage.telemetry.get("model_load_time_seconds", 0.0)),
                2,
            ),
            "Lectura contexto (s)": round(
                float(stage.telemetry.get("prompt_evaluation_time_seconds", 0.0)),
                2,
            ),
            "Generación (s)": round(
                float(stage.telemetry.get("generation_time_seconds", 0.0)),
                2,
            ),
            "Validación (s)": round(
                float(stage.telemetry.get("json_validation_time_seconds", 0.0)),
                4,
            ),
            "Preparación Python (s)": round(
                float(stage.telemetry.get("python_preprocessing_time_seconds", 0.0)),
                4,
            ),
            "Tamaño contexto": stage.telemetry.get("context_characters", ""),
            "Tokens estimados": stage.telemetry.get("context_token_estimate", ""),
        }
        for stage in result.stages
        if stage.telemetry
    ]
    if telemetry_rows:
        st.subheader("Diagnóstico avanzado de Ollama")
        st.dataframe(telemetry_rows, use_container_width=True, hide_index=True)
        validation_rows = [
            {
                "Paso": _stage_display_name(stage),
                "Detalle técnico": "; ".join(stage.warnings)
                or str(stage.error or ""),
            }
            for stage in result.stages
            if stage.warnings or stage.error
        ]
        if validation_rows and hasattr(st, "expander"):
            with st.expander("Detalles técnicos de validación", expanded=False):
                st.dataframe(validation_rows, use_container_width=True, hide_index=True)


def _friendly_stage_warning(warnings: tuple[str, ...]) -> str:
    """Return a concise non-technical warning for normal UI tables.

    Inputs: raw stage warnings.
    Outputs: Spanish user-facing warning.
    Assumptions: raw validator details remain available in advanced diagnostics.
    """

    text = " ".join(str(warning) for warning in warnings)
    if not text:
        return ""
    lowered = text.casefold()
    if "unsupported" in lowered or "validation" in lowered or "strategic analysis" in lowered:
        return "Algunas afirmaciones estratégicas fueron ajustadas o excluidas por falta de evidencia suficiente."
    return text


def _friendly_stage_error(error: str | None) -> str:
    """Return a concise non-technical error for normal UI tables.

    Inputs: raw error string.
    Outputs: Spanish user-facing error.
    Assumptions: advanced diagnostics retain raw details.
    """

    if not error:
        return ""
    lowered = str(error).casefold()
    if "unsupported" in lowered or "validation" in lowered:
        return "No se pudo validar parte del análisis estratégico con la evidencia disponible."
    return str(error)


def _render_overview_tab(st: Any, report_model: dict[str, Any], result: PipelineRunResult) -> None:
    """Render the Overview tab from report and pipeline metadata.

    Inputs: Streamlit module, report model, and result.
    Outputs: overview content.
    Assumptions: no calculations are performed in the UI.
    """

    input_model = result.config.input_model
    detected = input_model.detected_period if input_model else None
    view = build_presentation_view(report_model) if report_model else {}
    executive = view.get("executive_summary", {}) if isinstance(view, dict) else {}
    health_cards = view.get("financial_health", {}).get("cards", []) if isinstance(view, dict) else []
    st.markdown("### Resumen ejecutivo")
    st.write(executive.get("summary") or "El resumen ejecutivo no está disponible.")
    if detected:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Periodo detectado", detected.label)
        col_b.metric("Tipo de periodo", detected.period_type)
        col_c.metric("Confianza", f"{detected.confidence:.0%}")
    st.markdown("### Salud financiera")
    if health_cards:
        columns = st.columns(min(4, len(health_cards)))
        for index, card in enumerate(health_cards[:8]):
            with columns[index % len(columns)]:
                st.metric(
                    card.get("label", "Indicador"),
                    card.get("value", "No disponible"),
                    card.get("comparison_rows", [{}])[0].get("value", "") if card.get("comparison_rows") else None,
                )
                st.caption(card.get("description", ""))
    else:
        st.info("No hay indicadores financieros para mostrar en esta vista.")


def _render_kpi_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render KPI rows from the report model.

    Inputs: Streamlit module and report model.
    Outputs: KPI table.
    Assumptions: KPIs were calculated upstream.
    """

    view = build_presentation_view(report_model) if report_model else {}
    rows = [
        {
            "Indicador": item.get("indicator"),
            "Valor": item.get("value"),
            "Estado": item.get("status"),
            "Lectura": item.get("description"),
        }
        for item in view.get("kpis", [])
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No hay KPIs disponibles para este periodo.")


def _render_anomaly_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render anomaly summary rows from the report model.

    Inputs: Streamlit module and report model.
    Outputs: anomaly severity and top anomaly tables.
    Assumptions: anomaly detection already ran in Python.
    """

    view = build_presentation_view(report_model) if report_model else {}
    anomalies = view.get("anomalies", {})
    st.markdown("### Anomalías del periodo")
    if anomalies.get("current_period_status"):
        st.success(anomalies["current_period_status"])
    rows = [
        {
            "Anomalía": item.get("title"),
            "Severidad": item.get("severity"),
            "Evidencia": item.get("evidence"),
        }
        for item in anomalies.get("top", [])
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    elif not anomalies.get("current_period_status"):
        st.info("No hay anomalías relevantes para mostrar.")


def _render_recommendations_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render strategic analysis fields from the report model.

    Inputs: Streamlit module and report model.
    Outputs: root causes, priorities, recommendations, and missing info.
    Assumptions: strategic reasoning was generated and validated upstream.
    """

    view = build_presentation_view(report_model) if report_model else {}
    recommendations = view.get("recommendations", {})
    missing_items = view.get("missing_information", [])
    st.markdown("### Prioridades estratégicas")
    for item in recommendations.get("priorities", []):
        st.write(f"- {item}")
    st.markdown("### Recomendaciones")
    cards = recommendations.get("cards", [])
    if cards:
        for item in cards:
            with st.container(border=True):
                st.markdown(f"**{item.get('action', 'Recomendación')}**")
                st.write(f"Prioridad: {item.get('priority', 'No disponible')}")
                st.write(f"Impacto esperado: {item.get('expected_impact', 'No disponible')}")
                st.caption(f"Responsable sugerido: {item.get('owner', 'Por asignar')}")
    else:
        st.info("No hay recomendaciones estratégicas disponibles.")
    if missing_items:
        st.warning("Información pendiente:")
        for item in missing_items:
            st.write(f"- {item}")
    else:
        st.success("No se reporta información faltante relevante.")


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
            label_es = {
                "PDF": "Descargar PDF",
                "HTML": "Descargar HTML",
                "Report model JSON": "Descargar modelo JSON",
                "Strategic analysis JSON": "Descargar análisis JSON",
            }.get(label, f"Descargar {label}")
            st.download_button(
                label=label_es,
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime_by_label[label],
            )
        else:
            st.info(f"{label} no está disponible para esta ejecución.")


def _render_results(st: Any, result: PipelineRunResult) -> None:
    """Render all result tabs for a completed pipeline run.

    Inputs: Streamlit module and pipeline result.
    Outputs: tabbed results area.
    Assumptions: report artifacts are read-only presentation data.
    """

    artifacts = _artifact_paths(result)
    report_model = _load_json(artifacts["Report model JSON"])
    overview, kpis, anomalies, recommendations, downloads = st.tabs(
        ["Resumen", "KPIs", "Anomalías", "Recomendaciones", "Descargas"]
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


def _render_streamlit_app(st: Any) -> None:
    """Render the administrator-friendly Streamlit workflow.

    Inputs: imported Streamlit module.
    Outputs: interactive UI that delegates execution to the orchestrator.
    Assumptions: all business logic remains outside this UI layer.
    """

    st.set_page_config(
        page_title="Analista Financiero Universitario",
        page_icon="📊",
        layout="wide",
    )
    _apply_page_styles(st)
    st.title("Analista financiero universitario")
    st.caption(
        "Cargue un reporte financiero y un documento de metas. La aplicación calcula indicadores, "
        "consulta historial disponible y genera un reporte ejecutivo descargable."
    )
    _render_workflow_intro(st)

    st.session_state.setdefault("finance_ai_result", None)
    st.session_state.setdefault("finance_ai_running", False)
    st.session_state.setdefault("finance_ai_error", "")
    st.session_state.setdefault("finance_ai_progress_events", _initial_progress_events())
    st.session_state.setdefault("finance_ai_progress_started_at", None)
    st.session_state.setdefault("finance_ai_progress_completed_at", None)

    with st.sidebar:
        st.header("Configuración")
        language_label = st.selectbox("Idioma del reporte", options=("Español", "Inglés"), index=0)
        language = {"Español": "es", "Inglés": "en"}[language_label]
        override_mode = st.selectbox(
            "Periodo",
            options=SUPPORTED_UI_PERIOD_OPTIONS,
            index=0,
            help="Use la detección automática salvo que el sistema no pueda identificar el periodo correctamente.",
        )
        st.caption(
            "Actualmente, el sistema procesa reportes mensuales. Los períodos trimestrales, "
            "semestrales, anuales y personalizados se habilitarán después de implementar "
            "comparaciones compatibles."
        )
        override_value = ""
        if override_mode != "Detectar automáticamente":
            override_value = st.text_input(
                "Mes y año",
                placeholder="Ejemplo: 2026-12",
                help="Use el formato AAAA-MM.",
            )
        with st.expander("Opciones avanzadas", expanded=False):
            st.caption("Use estas opciones solo si administra Ollama, reutilización o memoria histórica.")
            endpoint = st.text_input("Servidor Ollama", value=DEFAULT_OLLAMA_ENDPOINT)
            model = st.text_input(
                "Modelo Ollama",
                value=DEFAULT_OLLAMA_MODEL,
                help="Configuración recomendada: un solo modelo para todas las etapas con LLM.",
            )
            enable_cache = st.checkbox(
                "Reutilizar análisis idénticos",
                value=True,
                help="Acelera ejecuciones repetidas cuando los archivos y opciones no cambiaron.",
            )
            enable_memory_storage = st.checkbox(
                "Guardar en memoria histórica",
                value=True,
                help="Guarda solo ejecuciones completas con análisis estratégico aceptado.",
            )
            memory_database = st.text_input(
                "Base de memoria histórica",
                value=str(PROJECT_ROOT / "data" / "memory" / "finance_memory.db"),
            )
            st.markdown("**Formatos de salida**")
            st.checkbox("PDF ejecutivo", value=True, disabled=True)
            st.checkbox("HTML ejecutivo", value=True, disabled=True)
            st.checkbox("JSON de auditoría", value=True, disabled=True)
            experimental_models = st.checkbox(
                "Experimental: modelos distintos por etapa",
                value=False,
                help="No recomendado por defecto; benchmarks locales fueron más lentos.",
            )
            structure_model = planner_model = analysis_model = None
            if experimental_models:
                structure_model = st.text_input("Modelo para estructura", value=EXPERIMENTAL_FAST_OLLAMA_MODEL)
                planner_model = st.text_input("Modelo para investigación", value=EXPERIMENTAL_FAST_OLLAMA_MODEL)
                analysis_model = st.text_input("Modelo para análisis estratégico", value=DEFAULT_OLLAMA_MODEL)
            connect_timeout = st.number_input("Tiempo máximo de conexión a Ollama (s)", 5.0, 120.0, 10.0, 5.0)
            read_timeout = st.number_input("Tiempo máximo de respuesta de Ollama (s)", 30.0, 1800.0, 600.0, 30.0)
            stage_timeout = st.number_input("Tiempo máximo por etapa (s)", 30.0, 2400.0, 900.0, 30.0)
            keep_alive = st.text_input("Mantener modelo cargado", value="15m")
            max_planner_anomalies = st.number_input("Máximo de anomalías enviadas al planificador", 1, 20, 5, 1)
            compact_context = st.checkbox("Compactar contexto para Ollama", value=True)
            deduplicate_context = st.checkbox("Eliminar evidencia duplicada", value=True)
        if st.button("Nuevo análisis / limpiar resultados", use_container_width=True):
            st.session_state["finance_ai_result"] = None
            st.session_state["finance_ai_error"] = ""
            st.session_state["finance_ai_progress_events"] = _initial_progress_events()
            st.session_state["finance_ai_progress_started_at"] = None
            st.session_state["finance_ai_progress_completed_at"] = None
            st.rerun()

    st.markdown("### 1. Cargue los documentos")
    st.markdown(
        """
        <div class='compat-card'>
          <b>Archivos compatibles:</b>
          reporte financiero en Excel o CSV (.xlsx, .xls, .csv) y documento de metas en PDF,
          Word o Excel (.pdf, .docx, .xlsx, .xls). El periodo se detecta automáticamente
          a partir del nombre del archivo, fechas y contenido disponible.
        </div>
        """,
        unsafe_allow_html=True,
    )
    upload_col_a, upload_col_b = st.columns(2)
    with upload_col_a:
        financial_report = st.file_uploader(
            "Reporte financiero",
            type=FINANCIAL_REPORT_UPLOAD_TYPES,
            help="Formatos admitidos: .xlsx, .xls, .csv.",
        )
        _render_file_validation(st, "Reporte financiero", financial_report, FINANCIAL_REPORT_UPLOAD_TYPES)
    with upload_col_b:
        goals_document = st.file_uploader(
            "Documento de metas",
            type=GOALS_UPLOAD_TYPES,
            help="Formatos admitidos: .pdf, .docx, .xlsx, .xls.",
        )
        _render_file_validation(st, "Documento de metas", goals_document, GOALS_UPLOAD_TYPES)

    files_ready = (
        financial_report is not None
        and goals_document is not None
        and _is_allowed_extension(financial_report, FINANCIAL_REPORT_UPLOAD_TYPES)
        and _is_allowed_extension(goals_document, GOALS_UPLOAD_TYPES)
    )
    preflight_input: PipelineInputModel | None = None
    report_classification: dict[str, Any] | None = None
    goals_classification: dict[str, Any] | None = None
    preflight_error = ""
    memory_database_path = Path(memory_database).expanduser() if memory_database.strip() else None
    if files_ready:
        # The UI writes temporary preflight copies so the shared deterministic
        # period-detection layer can classify uploads before pipeline execution.
        preflight_dir = UPLOAD_ROOT / "_preflight"
        preflight_report_path = save_uploaded_file(financial_report, preflight_dir)
        preflight_goals_path = save_uploaded_file(goals_document, preflight_dir)
        period_mode_map = {
            "Detectar automáticamente": "Auto",
            "Mensual": "Monthly",
        }
        preflight_settings = StreamlitRunSettings(
            report_language=language,
            period_override=_period_override_from_selection(
                period_mode_map.get(override_mode, "Auto"),
                override_value,
            ),
            memory_database_path=memory_database_path,
        )
        try:
            preflight_input = build_input_model_from_uploads(
                financial_report_path=preflight_report_path,
                goals_document_path=preflight_goals_path,
                settings=preflight_settings,
            )
            effective_period = preflight_input.effective_period_label
            report_classification = _classify_upload_for_period(
                uploaded_file=financial_report,
                document_type="financial_report",
                effective_period=effective_period,
                database_path=memory_database_path,
            )
            goals_classification = _classify_upload_for_period(
                uploaded_file=goals_document,
                document_type="goals_document",
                effective_period=effective_period,
                database_path=memory_database_path,
            )
        except Exception as exc:  # noqa: BLE001 - preflight should inform, not crash.
            preflight_error = f"No se pudo validar el registro del archivo antes del análisis: {exc}"
            st.error(preflight_error)
    _render_upload_registry_status(st, "Reporte financiero", report_classification)
    _render_upload_registry_status(st, "Documento de metas", goals_classification)
    monthly_ready, monthly_message = _monthly_readiness_message(
        input_model=preflight_input,
        override_mode=override_mode,
        override_value=override_value,
    )
    if files_ready and monthly_ready and not preflight_error:
        st.success(monthly_message)
    elif files_ready and not preflight_error:
        st.warning(monthly_message)
    revision_required = any(
        item and item.get("requires_revision_confirmation")
        for item in (report_classification, goals_classification)
    )
    revision_confirmed = False
    if revision_required:
        revision_confirmed = st.checkbox(
            "Confirmo registrar una nueva versión para este período si el análisis es aceptado.",
            value=False,
        )
    st.markdown("### 2. Genere el análisis financiero")
    if files_ready and monthly_ready and not preflight_error and (not revision_required or revision_confirmed):
        st.markdown(
            """
            <div class='run-action-card'>
              <b>Listo para analizar.</b><br/>
              Los documentos están completos. Genere el reporte ejecutivo con un solo clic.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif revision_required:
        st.warning("Confirme el registro de la nueva versión antes de ejecutar el análisis.")
    elif preflight_error:
        st.info("Corrija el problema de validación del registro antes de ejecutar el análisis.")
    elif files_ready and not monthly_ready:
        st.info("El análisis se habilitará cuando el período mensual esté confirmado.")
    else:
        st.info("Seleccione el reporte financiero y el documento de metas para continuar.")
    run_button = st.button(
        "Generar análisis financiero",
        type="primary",
        use_container_width=True,
        disabled=(
            not files_ready
            or bool(preflight_error)
            or not monthly_ready
            or (revision_required and not revision_confirmed)
            or bool(st.session_state.get("finance_ai_running"))
        ),
    )

    if not run_button:
        result = st.session_state.get("finance_ai_result")
        if result is not None:
            _render_progress_panel(st)
            _render_stage_results(st, result)
            _render_results(st, result)
        elif st.session_state.get("finance_ai_error"):
            _render_progress_panel(st)
            st.error(st.session_state["finance_ai_error"])
        return

    run_dir = UPLOAD_ROOT / time.strftime("%Y%m%d_%H%M%S")
    report_path = save_uploaded_file(financial_report, run_dir)
    goals_path = save_uploaded_file(goals_document, run_dir)
    period_mode_map = {
        "Detectar automáticamente": "Auto",
        "Mensual": "Monthly",
    }
    settings = StreamlitRunSettings(
        report_language=language,
        period_override=_period_override_from_selection(period_mode_map.get(override_mode, "Auto"), override_value),
        ollama_endpoint=endpoint,
        ollama_model=model.strip() or DEFAULT_OLLAMA_MODEL,
        structure_ollama_model=structure_model.strip() if structure_model else None,
        planner_ollama_model=planner_model.strip() if planner_model else None,
        analysis_ollama_model=analysis_model.strip() if analysis_model else None,
        ollama_timeout_seconds=float(read_timeout),
        connect_timeout_seconds=float(connect_timeout),
        read_timeout_seconds=float(read_timeout),
        stage_timeout_seconds=float(stage_timeout),
        ollama_keep_alive=keep_alive.strip() or "15m",
        max_planner_anomalies=int(max_planner_anomalies),
        compact_context=bool(compact_context),
        deduplicate_context=bool(deduplicate_context),
        enable_cache=bool(enable_cache),
        enable_memory_storage=bool(enable_memory_storage),
        memory_database_path=memory_database_path,
        source_revision_confirmed=revision_confirmed,
    )

    try:
        st.session_state["finance_ai_running"] = True
        st.session_state["finance_ai_error"] = ""
        st.session_state["finance_ai_progress_events"] = _initial_progress_events()
        st.session_state["finance_ai_progress_started_at"] = time.time()
        st.session_state["finance_ai_progress_completed_at"] = None
        progress_placeholder = st.empty()
        _render_progress_panel(st, progress_placeholder)

        def handle_progress(event: PipelineProgressEvent) -> None:
            """Refresh the Streamlit progress panel from an orchestrator event.

            Inputs: structured progress event.
            Outputs: updated session-state snapshot and rendered progress area.
            Assumptions: the callback is synchronous and does not run business logic.
            """

            st.session_state["finance_ai_progress_events"] = _merge_progress_event(
                st.session_state.get("finance_ai_progress_events") or _initial_progress_events(),
                event,
            )
            if event.stage_id == "analysis_completed" and event.status in {"completed", "failed"}:
                st.session_state["finance_ai_progress_completed_at"] = time.time()
            _render_progress_panel(st, progress_placeholder)

        result = run_analysis_from_files(
            financial_report_path=report_path,
            goals_document_path=goals_path,
            settings=settings,
            progress_callback=handle_progress,
        )
        st.session_state["finance_ai_result"] = result
        st.session_state["finance_ai_error"] = ""
    except Exception as exc:  # noqa: BLE001 - UI must display graceful failures.
        st.session_state["finance_ai_error"] = f"No se pudo iniciar el análisis: {exc}"
        st.session_state["finance_ai_progress_completed_at"] = time.time()
        failed_event = PipelineProgressEvent(
            stage_id="analysis_completed",
            label="Análisis completado",
            detail=st.session_state["finance_ai_error"],
            completed_steps=0,
            total_steps=len(PROGRESS_UI_STAGES),
            status="failed",
        )
        st.session_state["finance_ai_progress_events"] = _merge_progress_event(
            st.session_state.get("finance_ai_progress_events") or _initial_progress_events(),
            failed_event,
        )
        _render_progress_panel(st)
        st.error(st.session_state["finance_ai_error"])
        return
    finally:
        st.session_state["finance_ai_running"] = False

    if result.success:
        st.success(
            f"{_success_registration_message(result, (report_classification, goals_classification))} "
            "Revise el resumen y descargue los reportes."
        )
    else:
        st.error("El análisis terminó con una falla crítica. Revise el paso marcado como 'Requiere atención'.")
    if result.warnings:
        st.warning("Avisos: " + _friendly_stage_warning(result.warnings))
    _render_stage_results(st, result)
    _render_results(st, result)


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

    _render_streamlit_app(st)


if __name__ == "__main__":
    main()
