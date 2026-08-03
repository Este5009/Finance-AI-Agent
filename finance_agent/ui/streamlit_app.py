"""Streamlit v1 interface for running the Finance AI Agent pipeline."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from html import escape
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

STAGE_DISPLAY_LABELS_ES: dict[str, str] = {
    "Document ingestion": "Procesamiento de archivos",
    "Document understanding": "Interpretación de documentos",
    "Finance calculations": "Cálculo de indicadores financieros",
    "Anomaly detection": "Detección de anomalías",
    "Ollama structure fallback": "Revisión de estructura",
    "Ollama investigation planner": "Plan de investigación",
    "Retrieval layer": "Recuperación de evidencia",
    "Historical context": "Contexto histórico",
    "Strategic analysis": "Análisis estratégico",
    "Report model and renderers": "Generación del reporte",
    "Memory storage": "Guardado de resultados",
    "Pipeline error": "Error del pipeline",
}

PERIOD_TYPE_LABELS_ES: dict[str, str] = {
    "monthly": "Mensual",
    "quarterly": "Trimestral",
    "semester": "Semestral",
    "annual": "Anual",
    "custom": "Personalizado",
    "unknown": "No determinado",
}

VARIANT_LABELS_ES: dict[str, str] = {
    "positive": "Favorable",
    "negative": "Requiere atención",
    "warning": "Advertencia",
    "neutral": "Informativo",
    "info": "Informativo",
    "critical": "Crítica",
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
    "good": "Favorable",
    "amber": "Advertencia",
    "red": "Crítica",
}

UI_SECTION_TAB_BY_ID: dict[str, str] = {
    "cover": "Resumen",
    "executive_summary": "Resumen",
    "financial_health_overview": "Resumen",
    "kpi_overview": "KPIs",
    "revenue_analysis": "Análisis",
    "expense_analysis": "Análisis",
    "department_analysis": "Análisis",
    "anomaly_summary": "Anomalías",
    "investigation_evidence": "Análisis",
    "historical_summary": "Análisis",
    "historical_trends": "Análisis",
    "recommendation_follow_up": "Análisis",
    "longitudinal_risk_assessment": "Análisis",
    "strategic_recommendations": "Recomendaciones",
    "missing_information": "Recomendaciones",
    "appendix": "Descargas",
}

KPI_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Desempeño financiero",
        ("total_revenue", "total_expenses", "net_operating_result"),
    ),
    ("Liquidez", ("net_cash_flow", "ending_cash")),
    ("Eficiencia operativa", ("payroll_percentage_of_revenue", "collection_rate")),
)

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
    return STAGE_DISPLAY_LABELS_ES.get(raw, raw.capitalize())


def _period_type_label(period_type: str | None) -> str:
    """Return a Spanish label for a detected period type.

    Inputs: canonical period type from the pipeline.
    Outputs: administrator-facing Spanish label.
    Assumptions: internal period identifiers remain English.
    """

    key = str(period_type or "unknown").strip().lower()
    return PERIOD_TYPE_LABELS_ES.get(key, key.capitalize() if key else "No determinado")


def _render_safe_text_block(st: Any, text: str, *, css_class: str = "safe-text-block") -> None:
    """Render user-facing prose without Markdown math side effects.

    Inputs: Streamlit module and already-validated prose.
    Outputs: escaped HTML paragraph.
    Assumptions: escaping keeps Ollama/deterministic text from becoming HTML or
    LaTeX; this is presentation-only and performs no narrative rewriting.
    """

    st.markdown(
        f"<div class='{css_class}'>{escape(text or '')}</div>",
        unsafe_allow_html=True,
    )


def _comparison_value(card: dict[str, Any], label: str) -> str:
    """Return one deterministic KPI comparison row value by Spanish label.

    Inputs: presentation KPI card and expected row label.
    Outputs: formatted value or an empty string.
    Assumptions: rows are built upstream from report-model comparisons.
    """

    for row in card.get("comparison_rows", []) or []:
        if isinstance(row, dict) and row.get("label") == label:
            return str(row.get("value") or "")
    return ""


def _render_financial_health_card(st: Any, card: dict[str, Any]) -> None:
    """Render one deterministic KPI card without misusing Streamlit deltas.

    Inputs: Streamlit module and presentation card.
    Outputs: theme-safe HTML card.
    Assumptions: previous values and changes are displayed as separate facts so
    a previous-period value is never shown as an up/down delta.
    """

    previous = _comparison_value(card, "Periodo anterior")
    prior_change = _comparison_value(card, "Variación respecto al periodo anterior")
    budget = _comparison_value(card, "Presupuesto / meta")
    budget_change = _comparison_value(card, "Variación respecto al presupuesto")
    rows = []
    if previous:
        rows.append(("Período anterior", previous))
    if prior_change:
        rows.append(("Variación respecto al período anterior", prior_change))
    if budget:
        rows.append(("Presupuesto / meta", budget))
    if budget_change:
        rows.append(("Variación respecto al presupuesto", budget_change))
    row_html = "".join(
        "<div class='ui-kpi-row'>"
        f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
        "</div>"
        for label, value in rows
    )
    badge_html = _status_badge_html(card.get("badge"))
    variant = _variant_from_card(card)
    st.markdown(
        """
        <div class="ui-kpi-card ui-card--{variant}">
            <div class="ui-kpi-label">{label}</div>
            <div class="ui-kpi-value">{value}</div>
            {badge}
            <div class="ui-kpi-description">{description}</div>
            {rows}
        </div>
        """.format(
            variant=escape(variant),
            label=escape(str(card.get("label") or "Indicador")),
            value=escape(str(card.get("value") or "No disponible")),
            badge=badge_html,
            description=escape(str(card.get("description") or "")),
            rows=row_html,
        ),
        unsafe_allow_html=True,
    )


def _safe_display_text(value: Any) -> str:
    """Return user-facing text without exposing Python object representations.

    Inputs: any presentation value.
    Outputs: safe string for visible UI.
    Assumptions: dictionaries from presentation metadata expose a ``label`` and
    optional ``icon`` field; lists are joined only when their items are scalar.
    """

    if value is None:
        return ""
    if isinstance(value, dict):
        label = value.get("label") or value.get("title") or value.get("name") or ""
        icon = value.get("icon") or ""
        return f"{icon} {label}".strip()
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_safe_display_text(item) for item in value if _safe_display_text(item))
    return str(value)


def _badge_parts(value: Any) -> tuple[str, str, str]:
    """Return badge icon, label, and visual class from metadata or text.

    Inputs: status metadata dictionary or scalar label.
    Outputs: icon, Spanish label, and CSS class.
    Assumptions: no raw dictionary keys should ever be rendered.
    """

    if isinstance(value, dict):
        label = str(value.get("label") or "Informativo")
        klass = str(value.get("class") or "neutral").lower()
        icon = str(value.get("icon") or "")
    else:
        raw = str(value or "Informativo")
        key = raw.strip().lower()
        label = VARIANT_LABELS_ES.get(key, raw)
        klass = key if key else "neutral"
        icon = ""
    return icon, label, klass


def _status_badge_html(value: Any, *, fallback_class: str = "neutral") -> str:
    """Build a status badge HTML fragment from safe presentation metadata.

    Inputs: badge metadata or status text.
    Outputs: escaped HTML badge.
    Assumptions: CSS classes are sanitized to alphanumeric/hyphen variants.
    """

    icon, label, klass = _badge_parts(value)
    klass = re.sub(r"[^a-z0-9_-]+", "-", klass.lower()).strip("-") or fallback_class
    return (
        f"<span class='ui-status-badge ui-status-{escape(klass)}'>"
        f"{escape((icon + ' ') if icon else '')}{escape(label)}</span>"
    )


def _variant_from_card(card: dict[str, Any]) -> str:
    """Return a visual variant for one KPI card.

    Inputs: presentation card.
    Outputs: one of positive, negative, warning, neutral.
    Assumptions: card status is deterministic presentation metadata.
    """

    badge = card.get("badge")
    _, _, klass = _badge_parts(badge if badge else card.get("status"))
    klass = klass.lower()
    if klass in {"good", "positive", "achieved"}:
        return "positive"
    if klass in {"critical", "red", "bad", "negative"}:
        return "negative"
    if klass in {"amber", "warning", "medium", "high"}:
        return "warning"
    return "neutral"


def _card_variant_from_text(*values: Any) -> str:
    """Choose a card variant from severity/status text.

    Inputs: visible status or severity values.
    Outputs: CSS variant string.
    Assumptions: this maps presentation labels only; it does not infer finance.
    """

    text = " ".join(_safe_display_text(value).casefold() for value in values)
    if any(word in text for word in ("crítica", "critica", "alta", "riesgo", "negativo")):
        return "negative"
    if any(word in text for word in ("media", "advertencia", "pendiente", "seguimiento", "parcial")):
        return "warning"
    if any(word in text for word in ("favorable", "resuelto", "alcanzado", "positivo")):
        return "positive"
    return "neutral"


def _render_section_card(
    st: Any,
    *,
    title: str,
    body: str = "",
    variant: str = "neutral",
    badge: Any = None,
    rows: list[tuple[str, Any]] | None = None,
) -> None:
    """Render one reusable executive dashboard card.

    Inputs: Streamlit module, title, body, variant, optional badge and rows.
    Outputs: escaped HTML card.
    Assumptions: callers pass already-sanitized/report-model values; this helper
    prevents raw dictionaries/lists from leaking into normal UI.
    """

    row_html = "".join(
        "<div class='ui-card-row'>"
        f"<span>{escape(label)}</span><strong>{escape(_safe_display_text(value) or 'No disponible')}</strong>"
        "</div>"
        for label, value in (rows or [])
        if _safe_display_text(value)
    )


def _analysis_mode_label(report_model: dict[str, Any]) -> tuple[str, str]:
    """Return the user-facing mode label for one report model.

    Inputs: report model.
    Outputs: Spanish mode label and semantic variant.
    Assumptions: analysis status is metadata; this does not change report facts.
    """

    executive = _section_by_id(report_model, "executive_summary").get("content", {})
    status = str(executive.get("analysis_status") or "").lower()
    warnings = _section_by_id(report_model, "executive_summary").get("warnings", [])
    recommendations = build_presentation_view(report_model).get("recommendations", {}) if report_model else {}
    if status == "accepted" and recommendations.get("cards"):
        return "Análisis estratégico validado", "positive"
    if status == "sanitized" or warnings:
        return "Análisis estratégico ajustado", "warning"
    return "Reporte determinístico", "info"


def _report_status_label(result: PipelineRunResult, artifacts: dict[str, Path | None]) -> tuple[str, str]:
    """Return a concise report generation status for the result header.

    Inputs: pipeline result and artifact map.
    Outputs: Spanish status label and semantic variant.
    Assumptions: HTML/PDF file existence means deterministic rendering completed.
    """

    if not result.success:
        return "Ejecución con errores", "negative"
    if artifacts.get("PDF") and artifacts["PDF"].is_file() and artifacts.get("HTML") and artifacts["HTML"].is_file():
        return "Reporte listo para descargar", "positive"
    return "Reporte parcialmente disponible", "warning"


def _count_anomalies(view: dict[str, Any], severity_name: str) -> int:
    """Count anomalies with one Spanish severity label.

    Inputs: presentation view and severity label.
    Outputs: integer count.
    Assumptions: anomaly counts were produced by deterministic anomaly detection.
    """

    rows = view.get("anomalies", {}).get("severity_rows", []) if isinstance(view, dict) else []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and str(row.get("severity", "")).casefold() == severity_name.casefold():
            return int(row.get("count") or 0)
    return 0


def _render_results_header(
    st: Any,
    *,
    report_model: dict[str, Any],
    result: PipelineRunResult,
    artifacts: dict[str, Path | None],
) -> None:
    """Render the completed-results executive header.

    Inputs: Streamlit module, report model, pipeline result, and artifacts.
    Outputs: compact header cards.
    Assumptions: all values are metadata from pipeline/report artifacts.
    """

    view = build_presentation_view(report_model) if report_model else {}
    mode_label, mode_variant = _analysis_mode_label(report_model)
    status_label, status_variant = _report_status_label(result, artifacts)
    st.markdown("## Resultado del análisis financiero")
    st.caption("Vista ejecutiva generada desde el modelo de reporte determinístico.")
    columns = st.columns(5)
    header_cards = (
        ("Periodo", view.get("period") or "No disponible", "info", "Informativo"),
        ("Modo de análisis", mode_label, mode_variant, mode_label),
        ("Datos verificados", "Cifras procesadas por Python", "positive", "Verificado"),
        ("Tiempo total", _format_elapsed_seconds(result.runtime_summary.total_runtime_seconds), "info", "Informativo"),
        ("Estado del reporte", status_label, status_variant, status_label),
    )
    for column, (title, body, variant, badge) in zip(columns, header_cards):
        with column:
            _render_section_card(st, title=title, body=body, variant=variant, badge=badge)


def _render_attention_summary(st: Any, report_model: dict[str, Any]) -> None:
    """Render a compact attention summary for executives.

    Inputs: Streamlit module and report model.
    Outputs: anomaly/risk/strategy status cards.
    Assumptions: counts and statuses are read-only report-model/presentation data.
    """

    view = build_presentation_view(report_model) if report_model else {}
    historical = view.get("historical", {}) if isinstance(view, dict) else {}
    recommendations = view.get("recommendations", {}) if isinstance(view, dict) else {}
    critical = _count_anomalies(view, "Crítica")
    high = _count_anomalies(view, "Alta")
    recurring_risks = len(historical.get("recurring_risks", []) or []) if isinstance(historical, dict) else 0
    follow_up = len(historical.get("recommendation_follow_up", []) or []) if isinstance(historical, dict) else 0
    strategy_available = bool(recommendations.get("cards")) if isinstance(recommendations, dict) else False
    st.markdown("### Resumen de atención")
    columns = st.columns(4)
    cards = (
        (
            "Anomalías críticas",
            f"{critical}",
            "negative" if critical else "positive",
            "Crítica" if critical else "Favorable",
            [("Alta severidad", high)],
        ),
        (
            "Riesgos recurrentes",
            f"{recurring_risks}",
            "warning" if recurring_risks else "info",
            "En seguimiento" if recurring_risks else "Informativo",
            [],
        ),
        (
            "Recomendaciones previas",
            f"{follow_up}",
            "info",
            "Seguimiento",
            [],
        ),
        (
            "Enriquecimiento estratégico",
            "Disponible" if strategy_available else "No validado",
            "positive" if strategy_available else "warning",
            "Validado" if strategy_available else "Advertencia",
            [],
        ),
    )
    for column, (title, body, variant, badge, rows) in zip(columns, cards):
        with column:
            _render_section_card(st, title=title, body=body, variant=variant, badge=badge, rows=rows)


def _render_grouped_kpi_cards(st: Any, cards: list[dict[str, Any]]) -> None:
    """Render KPI cards grouped by executive meaning.

    Inputs: Streamlit module and presentation KPI cards.
    Outputs: grouped KPI card layout.
    Assumptions: cards are built from deterministic report-model comparisons.
    """

    by_id = {str(card.get("id")): card for card in cards if isinstance(card, dict)}
    rendered: set[str] = set()
    for group_title, metric_ids in KPI_GROUPS:
        group_cards = [by_id[metric_id] for metric_id in metric_ids if metric_id in by_id]
        if not group_cards:
            continue
        st.markdown(f"#### {group_title}")
        columns = st.columns(min(3, len(group_cards)))
        for index, card in enumerate(group_cards):
            rendered.add(str(card.get("id")))
            with columns[index % len(columns)]:
                _render_financial_health_card(st, card)
    remaining = [card for card in cards if str(card.get("id")) not in rendered]
    if remaining:
        st.markdown("#### Otros indicadores")
        columns = st.columns(min(3, len(remaining)))
        for index, card in enumerate(remaining):
            with columns[index % len(columns)]:
                _render_financial_health_card(st, card)


def _render_download_card(st: Any, label: str, path: Path, mime: str) -> None:
    """Render one prominent download card and button.

    Inputs: Streamlit module, Spanish label, path, and MIME type.
    Outputs: card with download button.
    Assumptions: caller verified the file exists.
    """

    descriptions = {
        "PDF": "Reporte ejecutivo listo para compartir.",
        "HTML": "Versión navegable del reporte.",
        "modelo JSON del reporte": "Estructura completa usada por la interfaz y renderizadores.",
        "análisis estratégico JSON": "Salida estratégica validada o diagnóstico de rechazo.",
        "evidencia JSON": "Evidencia determinística consultada por el pipeline.",
    }
    _render_section_card(
        st,
        title=label,
        body=descriptions.get(label, "Artefacto generado por el pipeline."),
        variant="info",
        badge="Disponible",
        rows=[("Archivo", path.name)],
    )
    st.download_button(
        label=f"Descargar {label}",
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
    )
    badge_html = _status_badge_html(badge or variant)
    st.markdown(
        """
        <div class="ui-dashboard-card ui-card--{variant}">
            <div class="ui-card-header">
                <div class="ui-card-title">{title}</div>
                {badge}
            </div>
            <div class="ui-card-body">{body}</div>
            {rows}
        </div>
        """.format(
            variant=escape(variant),
            title=escape(title),
            badge=badge_html,
            body=escape(body),
            rows=row_html,
        ),
        unsafe_allow_html=True,
    )


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
        :root {
            --fa-bg: #f5f7fa;
            --fa-surface: #ffffff;
            --fa-surface-elevated: #fbfdff;
            --fa-border: #d8e1ea;
            --fa-text: #172033;
            --fa-text-strong: #10243a;
            --fa-muted: #526273;
            --fa-positive: #1b7f4a;
            --fa-negative: #b42318;
            --fa-warning: #b7791f;
            --fa-info: #39739d;
            --fa-neutral: #697586;
        }
        .main .block-container { padding-top: 1.0rem; max-width: 1220px; }
        .stApp { background: var(--fa-bg); }
        div[data-testid="stMetric"] {
            background: var(--fa-surface-elevated);
            border: 1px solid var(--fa-border);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .finance-card {
            background: var(--fa-surface-elevated);
            border: 1px solid var(--fa-border);
            border-radius: 16px;
            padding: 16px 18px;
            margin: 10px 0 16px;
        }
        .step-card {
            min-height: 112px;
            background: linear-gradient(180deg, var(--fa-surface) 0%, var(--fa-surface-elevated) 100%);
            border: 1px solid var(--fa-border);
            border-left: 5px solid var(--fa-info);
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
            color: var(--fa-text-strong);
            font-size: 1.02rem;
            font-weight: 700;
        }
        .step-description {
            color: var(--fa-muted);
            line-height: 1.38;
            margin-top: 8px;
            font-size: 0.94rem;
        }
        .compat-card {
            background: var(--fa-surface-elevated);
            border: 1px solid var(--fa-border);
            border-radius: 14px;
            padding: 14px 16px;
            margin: 2px 0 14px;
            color: var(--fa-text);
        }
        .run-action-card {
            background: var(--fa-surface-elevated);
            border: 1px solid var(--fa-border);
            border-radius: 16px;
            padding: 14px 16px 10px;
            margin-top: 8px;
        }
        .safe-text-block {
            color: var(--fa-text);
            line-height: 1.58;
            margin: 0.2rem 0 0.8rem;
        }
        .verified-data-note {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #0f5132;
            background: #dff4e8;
            border: 1px solid #9ed6b5;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.86rem;
            font-weight: 650;
            margin: 0.2rem 0 0.9rem;
        }
        .ui-kpi-card {
            background: var(--fa-surface);
            color: var(--fa-text);
            border: 1px solid var(--fa-border);
            border-left: 5px solid var(--fa-info);
            border-radius: 16px;
            padding: 16px 17px;
            margin: 0.25rem 0 1rem;
            box-shadow: 0 1px 3px rgba(23, 43, 77, 0.06);
        }
        .ui-dashboard-card {
            background: var(--fa-surface);
            color: var(--fa-text);
            border: 1px solid var(--fa-border);
            border-left: 5px solid var(--fa-info);
            border-radius: 16px;
            padding: 16px 18px;
            margin: 0.35rem 0 1rem;
            box-shadow: 0 1px 3px rgba(23, 43, 77, 0.06);
        }
        .ui-card--positive { border-left-color: var(--fa-positive); background: var(--fa-surface); }
        .ui-card--negative { border-left-color: var(--fa-negative); background: var(--fa-surface); }
        .ui-card--warning { border-left-color: var(--fa-warning); background: var(--fa-surface); }
        .ui-card--neutral, .ui-card--info { border-left-color: var(--fa-info); background: var(--fa-surface); }
        .ui-card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.45rem;
        }
        .ui-card-title {
            color: var(--fa-text-strong);
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.25;
        }
        .ui-card-body {
            color: var(--fa-muted);
            line-height: 1.45;
            font-size: 0.92rem;
            margin: 0.3rem 0 0.55rem;
        }
        .ui-card-row {
            display: flex;
            justify-content: space-between;
            gap: 0.85rem;
            border-top: 1px solid #e6edf3;
            padding-top: 0.48rem;
            margin-top: 0.48rem;
            color: var(--fa-muted);
            font-size: 0.85rem;
        }
        .ui-card-row strong {
            color: var(--fa-text-strong);
            text-align: right;
        }
        .ui-kpi-label {
            color: var(--fa-muted);
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            min-height: 2.2rem;
        }
        .ui-kpi-value {
            color: var(--fa-text-strong);
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.12;
            margin-top: 0.4rem;
        }
        .ui-kpi-badge,
        .ui-status-badge {
            display: inline-block;
            color: #17324d;
            background: #e8f0f8;
            border: 1px solid #c9dceb;
            border-radius: 999px;
            padding: 3px 9px;
            margin-top: 0.7rem;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .ui-status-badge { margin-top: 0; white-space: nowrap; }
        .ui-status-good,
        .ui-status-positive,
        .ui-status-achieved {
            color: #0f5132;
            background: #dff4e8;
            border-color: #9ed6b5;
        }
        .ui-status-critical,
        .ui-status-high,
        .ui-status-red,
        .ui-status-negative,
        .ui-status-bad {
            color: #842029;
            background: #fde2e1;
            border-color: #f5b5b1;
        }
        .ui-status-amber,
        .ui-status-warning,
        .ui-status-medium {
            color: #7a4b00;
            background: #fff0cc;
            border-color: #e3bd63;
        }
        .ui-status-neutral,
        .ui-status-info,
        .ui-status-low {
            color: #17324d;
            background: #e8f0f8;
            border-color: #c9dceb;
        }
        .ui-kpi-description {
            color: var(--fa-muted);
            font-size: 0.86rem;
            line-height: 1.35;
            margin: 0.65rem 0;
        }
        .ui-kpi-row {
            display: flex;
            justify-content: space-between;
            gap: 0.85rem;
            border-top: 1px solid #e6edf3;
            padding-top: 0.5rem;
            margin-top: 0.5rem;
            color: var(--fa-muted);
            font-size: 0.82rem;
        }
        .ui-kpi-row strong {
            color: var(--fa-text-strong);
            text-align: right;
            white-space: nowrap;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --fa-bg: #0f1724;
                --fa-surface: #172033;
                --fa-surface-elevated: #1b2638;
                --fa-border: #35465a;
                --fa-text: #f5f7fa;
                --fa-text-strong: #f8fafc;
                --fa-muted: #c5d0dc;
                --fa-positive: #3ab777;
                --fa-negative: #f07067;
                --fa-warning: #e2ad43;
                --fa-info: #6ca7d8;
                --fa-neutral: #a8b3c0;
            }
            .safe-text-block { color: #f5f7fa; }
            .verified-data-note {
                color: #d8f8e7;
                background: #123925;
                border-color: #2b6f49;
            }
            .ui-kpi-card {
                background: #172033;
                color: #f5f7fa;
                border-color: #35465a;
                box-shadow: none;
            }
            .ui-dashboard-card {
                background: #172033;
                color: #f5f7fa;
                border-color: #35465a;
                box-shadow: none;
            }
            .ui-card--positive { background: var(--fa-surface); border-left-color: var(--fa-positive); }
            .ui-card--negative { background: var(--fa-surface); border-left-color: var(--fa-negative); }
            .ui-card--warning { background: var(--fa-surface); border-left-color: var(--fa-warning); }
            .ui-card--neutral, .ui-card--info { background: #172033; border-left-color: #6ca7d8; }
            .ui-kpi-label,
            .ui-kpi-description,
            .ui-kpi-row,
            .ui-card-body,
            .ui-card-row { color: #c5d0dc; }
            .ui-kpi-value,
            .ui-kpi-row strong,
            .ui-card-title,
            .ui-card-row strong { color: #f8fafc; }
            .ui-kpi-badge,
            .ui-status-badge {
                color: #dcecff;
                background: #243c58;
                border-color: #3b5878;
            }
            .ui-status-good,
            .ui-status-positive,
            .ui-status-achieved {
                color: #d8f8e7;
                background: #123925;
                border-color: #2b6f49;
            }
            .ui-status-critical,
            .ui-status-high,
            .ui-status-red,
            .ui-status-negative,
            .ui-status-bad {
                color: #ffd9d7;
                background: #4a1d1d;
                border-color: #8d3430;
            }
            .ui-status-amber,
            .ui-status-warning,
            .ui-status-medium {
                color: #ffe8ad;
                background: #493410;
                border-color: #8b641e;
            }
            .ui-status-neutral,
            .ui-status-info,
            .ui-status-low {
                color: #dcecff;
                background: #243c58;
                border-color: #3b5878;
            }
            .ui-kpi-row,
            .ui-card-row { border-top-color: #33465a; }
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
        @media (prefers-color-scheme: dark) {
            div[data-testid="stButton"] button:disabled {
                background-color: #243247 !important;
                color: #c5d0dc !important;
                border: 1px solid #43546a !important;
            }
        }
        @media (max-width: 760px) {
            .ui-card-header,
            .ui-card-row,
            .ui-kpi-row {
                flex-direction: column;
                align-items: flex-start;
            }
            .ui-card-row strong,
            .ui-kpi-row strong { text-align: left; white-space: normal; }
            .ui-kpi-value { font-size: 1.45rem; }
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


def _ui_section_consistency_rows(
    report_model: dict[str, Any],
    *,
    html_available: bool = False,
    pdf_available: bool = False,
) -> list[dict[str, Any]]:
    """Build a deterministic UI/report section coverage audit.

    Inputs: report model and optional rendered-artifact availability flags.
    Outputs: rows showing whether each report section has a Streamlit route.
    Assumptions: PDF text extraction is intentionally out of scope here; file
    existence represents renderer availability for UI download validation.
    """

    sections = report_model.get("sections", [])
    rows: list[dict[str, Any]] = []
    for section in sections if isinstance(sections, list) else []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id") or "")
        tab = UI_SECTION_TAB_BY_ID.get(section_id)
        rows.append(
            {
                "section_id": section_id,
                "section_name": section.get("title") or section_id,
                "present_in_report_model": True,
                "present_in_html": bool(html_available),
                "present_in_pdf": bool(pdf_available),
                "present_in_streamlit": tab is not None,
                "streamlit_tab": tab or "Solo descarga",
                "expected_behavior": (
                    f"Visible en la pestaña {tab}" if tab else "Disponible en descargas o artefactos técnicos"
                ),
                "pass": tab is not None or section_id == "appendix",
            }
        )
    return rows


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
    if report_model is None:
        report_model = _report_model_from_period(result)
    period_suffix = ""
    if report_model is not None:
        period_suffix = report_model.stem.replace("report_model_", "")
    return {
        "PDF": _find_output(result, f"financial_report_{period_suffix}.pdf")
        or _sibling_artifact(report_model, f"financial_report_{period_suffix}.pdf"),
        "HTML": _find_output(result, f"financial_report_{period_suffix}.html")
        or _sibling_artifact(report_model, f"financial_report_{period_suffix}.html"),
        "Report model JSON": report_model,
        "Strategic analysis JSON": _find_output(result, f"strategic_analysis_{period_suffix}.json")
        or _analysis_artifact_from_period(result, period_suffix),
        "Evidence package JSON": _find_output(result, f"evidence_package_{period_suffix}.json")
        or _evidence_artifact_from_period(result, period_suffix),
    }


def _period_slug_from_result(result: PipelineRunResult) -> str:
    """Infer the period slug associated with a pipeline result.

    Inputs: pipeline result.
    Outputs: period slug such as ``2026_07`` or an empty string.
    Assumptions: output filenames are preferred because they are generated by
    the pipeline; input-model labels are fallback metadata.
    """

    for output_file in result.output_files:
        path = Path(output_file)
        for prefix, suffix in (
            ("report_model_", ".json"),
            ("financial_report_", ".pdf"),
            ("financial_report_", ".html"),
            ("strategic_analysis_", ".json"),
        ):
            if path.name.startswith(prefix) and path.name.endswith(suffix):
                return path.stem.replace(prefix, "")
    input_model = getattr(result.config, "input_model", None)
    label = str(getattr(input_model, "effective_period_label", "") or "")
    match = re.search(r"(20\d{2})[-_](0[1-9]|1[0-2])", label)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return ""


def _report_model_from_period(result: PipelineRunResult) -> Path | None:
    """Find an existing report model by the result period slug.

    Inputs: pipeline result.
    Outputs: report-model path or None.
    Assumptions: this filesystem check restores downloads when renderer outputs
    exist but were not listed in ``PipelineRunResult.output_files``.
    """

    slug = _period_slug_from_result(result)
    if not slug:
        return None
    candidate = result.config.output_directory / "report" / f"report_model_{slug}.json"
    return candidate if candidate.is_file() else None


def _sibling_artifact(report_model: Path | None, filename: str) -> Path | None:
    """Return a sibling artifact when it exists.

    Inputs: report-model path and expected filename.
    Outputs: sibling path or None.
    Assumptions: report renderers save HTML/PDF beside the model.
    """

    if report_model is None:
        return None
    candidate = report_model.parent / filename
    return candidate if candidate.is_file() else None


def _analysis_artifact_from_period(result: PipelineRunResult, period_suffix: str) -> Path | None:
    """Find the strategic-analysis artifact for one period.

    Inputs: result and period suffix.
    Outputs: strategic-analysis path or None.
    Assumptions: the artifact is useful for diagnostics even when validation failed.
    """

    if not period_suffix:
        return None
    candidate = result.config.output_directory / "analysis" / f"strategic_analysis_{period_suffix}.json"
    return candidate if candidate.is_file() else None


def _evidence_artifact_from_period(result: PipelineRunResult, period_suffix: str) -> Path | None:
    """Find the evidence package artifact for one period.

    Inputs: result and period suffix.
    Outputs: evidence-package path or None.
    Assumptions: deterministic evidence remains downloadable regardless of strategy status.
    """

    if not period_suffix:
        return None
    candidate = result.config.output_directory / "evidence" / f"evidence_package_{period_suffix}.json"
    return candidate if candidate.is_file() else None


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
    st.subheader("Resultado de ejecución")
    cache_label = "se reutilizó un análisis existente" if result.cache_hit else "análisis nuevo"
    st.info(
        f"Modo de resultado: {cache_label}. Tiempo total: "
        f"{_format_elapsed_seconds(result.runtime_summary.total_runtime_seconds)}."
    )
    if result.success and hasattr(st, "expander"):
        with st.expander("Ver detalle de etapas ejecutadas", expanded=False):
            st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    for stage in (stage for stage in result.stages if not stage.success):
        st.error(f"{_stage_display_name(stage)}: {_friendly_stage_error(stage.error)}")
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
                "Traceback": str(stage.telemetry.get("traceback") or ""),
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
    if "department contains unsupported characters" in lowered:
        return "No se pudo consultar el historial de un departamento con el nombre recibido."
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
    _render_safe_text_block(
        st,
        executive.get("summary") or "El resumen ejecutivo no está disponible.",
        css_class="safe-text-block",
    )
    st.markdown(
        "<div class='verified-data-note'>✓ Cifras verificadas con datos procesados</div>",
        unsafe_allow_html=True,
    )
    if detected:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            _render_section_card(
                st,
                title="Periodo detectado",
                body=str(detected.label),
                variant="info",
                badge="Informativo",
            )
        with col_b:
            _render_section_card(
                st,
                title="Tipo de periodo",
                body=_period_type_label(detected.period_type),
                variant="info",
                badge="Informativo",
            )
        with col_c:
            _render_section_card(
                st,
                title="Confianza de detección del periodo",
                body=f"{detected.confidence:.0%}",
                variant="info",
                badge="Determinístico",
            )
    st.markdown("### Salud financiera")
    if health_cards:
        _render_grouped_kpi_cards(st, health_cards[:8])
        _render_attention_summary(st, report_model)
    else:
        st.info("No hay indicadores financieros para mostrar en esta vista.")


def _render_kpi_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render KPI rows from the report model.

    Inputs: Streamlit module and report model.
    Outputs: KPI table.
    Assumptions: KPIs were calculated upstream.
    """

    view = build_presentation_view(report_model) if report_model else {}
    cards = view.get("financial_health", {}).get("cards", []) if isinstance(view, dict) else []
    if cards:
        st.markdown("### Indicadores principales")
        _render_grouped_kpi_cards(st, cards[:8])
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
        with st.expander("Ver tabla completa de KPIs", expanded=False):
            st.dataframe(rows, use_container_width=True, hide_index=True)
        health_section = _section_by_id(report_model, "financial_health_overview")
        comparisons = health_section.get("content", {}).get("kpi_comparisons", {})
        if isinstance(comparisons, dict) and comparisons.get("items"):
            with st.expander("Proveniencia y cálculo determinístico", expanded=False):
                st.dataframe(
                    [
                        {
                            "Indicador": item.get("metric"),
                            "Fuente": item.get("provenance", {}).get("source_artifact"),
                            "Valor actual": item.get("current_value"),
                            "Comparación": item.get("previous_value"),
                            "Cambio calculado": item.get("absolute_change")
                            if item.get("unit") != "ratio"
                            else item.get("percentage_point_change"),
                            "Método": item.get("provenance", {}).get("calculation_method"),
                        }
                        for item in comparisons.get("items", {}).values()
                        if isinstance(item, dict)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("No hay KPIs disponibles para este periodo.")


def _render_anomaly_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render current-period anomalies from the canonical report model.

    Inputs: Streamlit module and report model.
    Outputs: anomaly severity cards/tables for the current period.
    Assumptions: anomaly detection already ran in Python; this function does not
    recalculate or ask Ollama for anything.
    """

    view = build_presentation_view(report_model) if report_model else {}
    anomalies = view.get("anomalies", {}) if isinstance(view, dict) else {}
    st.markdown("### Anomalías del periodo")
    if anomalies.get("current_period_status"):
        _render_section_card(
            st,
            title="Sin anomalías nuevas",
            body=str(anomalies["current_period_status"]),
            variant="positive",
            badge="Favorable",
        )
    severity_rows = anomalies.get("severity_rows", []) if isinstance(anomalies, dict) else []
    if severity_rows:
        st.markdown("#### Resumen por severidad")
        st.dataframe(
            [
                {"Severidad": row.get("severity"), "Cantidad": row.get("count")}
                for row in severity_rows
                if isinstance(row, dict)
            ],
            use_container_width=True,
            hide_index=True,
        )
    top_rows = anomalies.get("top_rows", []) if isinstance(anomalies, dict) else []
    if top_rows:
        st.markdown("#### Anomalías detectadas")
        columns = st.columns(2)
        for index, item in enumerate(top_rows):
            if not isinstance(item, dict):
                continue
            with columns[index % 2]:
                _render_section_card(
                    st,
                    title=_safe_display_text(item.get("title") or "Anomalía detectada"),
                    body=_safe_display_text(item.get("evidence")),
                    variant=_card_variant_from_text(item.get("severity"), item.get("severity_class")),
                    badge=item.get("severity"),
                    rows=[
                        ("Severidad", item.get("severity")),
                        ("Recurrencia", item.get("recurrence")),
                        ("Períodos afectados", item.get("period_chips")),
                    ],
                )
    elif not anomalies.get("current_period_status"):
        st.info("No hay anomalías relevantes para mostrar.")
    if anomalies.get("distinction_note"):
        _render_section_card(
            st,
            title="Lectura de riesgos históricos",
            body=str(anomalies["distinction_note"]),
            variant="info",
            badge="Informativo",
        )


def _render_analysis_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render deterministic analysis sections from the report model.

    Inputs: Streamlit module and report model.
    Outputs: historical, departmental, revenue/expense, evidence, and follow-up views.
    Assumptions: this function only presents report-model data; it performs no
    calculations and never calls Ollama.
    """

    view = build_presentation_view(report_model) if report_model else {}
    revenue_expense = view.get("revenue_expense", {}) if isinstance(view, dict) else {}
    departments = view.get("departments", []) if isinstance(view, dict) else []
    historical = view.get("historical", {}) if isinstance(view, dict) else {}
    evidence = view.get("evidence", []) if isinstance(view, dict) else []

    st.markdown("### Análisis determinístico del reporte")
    rows = revenue_expense.get("rows", []) if isinstance(revenue_expense, dict) else []
    if rows:
        st.markdown("#### Ingresos, gastos y presupuesto")
        summary_columns = st.columns(3)
        for index, row in enumerate(rows[:9]):
            if not isinstance(row, dict):
                continue
            with summary_columns[index % 3]:
                _render_section_card(
                    st,
                    title=_safe_display_text(row.get("metric")),
                    body=_safe_display_text(row.get("description")),
                    variant="neutral",
                    badge="Informativo",
                    rows=[("Valor", row.get("value"))],
                )
        if revenue_expense.get("chart_insight"):
            _render_section_card(
                st,
                title="Conclusión ejecutiva",
                body=_safe_display_text(revenue_expense.get("chart_insight")),
                variant="info",
                badge="Informativo",
            )
        if revenue_expense.get("budget_chart_insight"):
            _render_section_card(
                st,
                title="Comparación contra presupuesto",
                body=_safe_display_text(revenue_expense.get("budget_chart_insight")),
                variant="warning",
                badge="Advertencia",
            )

    if departments:
        st.markdown("#### Resultados por departamento")
        st.dataframe(
            [
                {
                    "Departamento": row.get("department"),
                    "Ingresos": row.get("revenue"),
                    "Gastos": row.get("expenses"),
                    "Resultado": row.get("result"),
                    "Presión": row.get("rank_badge") or "",
                }
                for row in departments
                if isinstance(row, dict)
            ],
            use_container_width=True,
            hide_index=True,
        )

    if historical.get("available"):
        st.markdown("#### Contexto histórico")
        narrative = historical.get("narrative", []) or []
        for item in narrative[:3] if isinstance(narrative, list) else []:
            _render_section_card(
                st,
                title="Lectura histórica",
                body=_safe_display_text(item),
                variant="info",
                badge="Informativo",
            )
        trends = historical.get("trends", []) or []
        if trends:
            trend_rows = []
            for trend in trends:
                if isinstance(trend, dict):
                    points = trend.get("points", []) or []
                    trend_rows.append(
                        {
                            "Indicador": trend.get("metric"),
                            "Dirección": trend.get("direction"),
                            "Períodos": len(points),
                            "Primer valor": points[0].get("value") if points and isinstance(points[0], dict) else "",
                            "Último valor": points[-1].get("value") if points and isinstance(points[-1], dict) else "",
                        }
                    )
            st.dataframe(trend_rows, use_container_width=True, hide_index=True)
        risks = historical.get("recurring_risks", []) or []
        if risks:
            st.markdown("#### Riesgos históricos recurrentes")
            risk_columns = st.columns(2)
            for index, risk in enumerate(risks):
                if not isinstance(risk, dict):
                    continue
                with risk_columns[index % 2]:
                    _render_section_card(
                        st,
                        title=_safe_display_text(risk.get("risk") or "Riesgo recurrente"),
                        body=_safe_display_text(risk.get("why_it_matters") or risk.get("status_reason")),
                        variant="warning",
                        badge=risk.get("status") or "En seguimiento",
                        rows=[
                            ("Departamento", risk.get("department")),
                            ("Frecuencia", risk.get("frequency") or risk.get("occurrences")),
                            ("Períodos afectados", risk.get("affected_periods") or risk.get("periods")),
                        ],
                    )
        follow_up = historical.get("recommendation_follow_up", []) or []
        if follow_up:
            st.markdown("#### Seguimiento de recomendaciones emitidas anteriormente")
            if historical.get("recommendation_intro"):
                st.caption(_safe_display_text(historical.get("recommendation_intro")))
            if historical.get("recommendation_summary"):
                _render_section_card(
                    st,
                    title="Estado del seguimiento",
                    body=_safe_display_text(historical.get("recommendation_summary")),
                    variant="info",
                    badge="Informativo",
                )
            columns = st.columns(2)
            for index, item in enumerate(follow_up):
                if not isinstance(item, dict):
                    continue
                with columns[index % 2]:
                    _render_section_card(
                        st,
                        title=_safe_display_text(item.get("recommendation") or "Recomendación previa"),
                        body=_safe_display_text(item.get("status_reason") or item.get("current_evidence")),
                        variant=_card_variant_from_text(item.get("progress")),
                        badge=item.get("progress") or "En seguimiento",
                        rows=[
                            ("Emitida en", item.get("origin_period")),
                            ("Objetivo original", item.get("objective")),
                            ("Evidencia actual", item.get("current_evidence")),
                            ("Siguiente acción", item.get("next_action")),
                        ],
                    )
    else:
        st.info("No hay contexto histórico suficiente para este periodo.")

    if evidence:
        with st.expander("Evidencia determinística consultada", expanded=False):
            st.dataframe(
                [
                    {
                        "Prioridad": item.get("priority"),
                        "Evidencia": item.get("evidence"),
                        "Registros": item.get("records"),
                        "Resumen": item.get("summary"),
                    }
                    for item in evidence
                    if isinstance(item, dict)
                ],
                use_container_width=True,
                hide_index=True,
            )


def _render_recommendations_tab(st: Any, report_model: dict[str, Any]) -> None:
    """Render strategic recommendations and deterministic follow-up.

    Inputs: Streamlit module and report model.
    Outputs: validated strategic recommendation cards or a clear fallback plus
    previous recommendation follow-up when available.
    Assumptions: strategic recommendations may be unavailable without invalidating
    deterministic analysis sections.
    """

    view = build_presentation_view(report_model) if report_model else {}
    recommendations = view.get("recommendations", {}) if isinstance(view, dict) else {}
    historical = view.get("historical", {}) if isinstance(view, dict) else {}
    missing_items = view.get("missing_information", []) if isinstance(view, dict) else []
    st.markdown("### Recomendaciones estratégicas actuales")
    priorities = recommendations.get("priorities", []) or []
    if priorities:
        st.markdown("#### Prioridades validadas")
        for item in priorities:
            _render_section_card(
                st,
                title="Prioridad estratégica",
                body=_safe_display_text(item),
                variant="warning",
                badge="Alta prioridad",
            )
    cards = recommendations.get("cards", []) or []
    if cards:
        columns = st.columns(2)
        for index, item in enumerate(cards):
            if not isinstance(item, dict):
                continue
            with columns[index % 2]:
                _render_section_card(
                    st,
                    title=_safe_display_text(item.get("action") or "Recomendación"),
                    body=_safe_display_text(item.get("rationale")),
                    variant=_card_variant_from_text(item.get("priority")),
                    badge=item.get("priority") or "Media",
                    rows=[
                        ("Impacto esperado", item.get("expected_impact")),
                        ("Responsable sugerido", item.get("owner")),
                        ("Estado", item.get("status")),
                    ],
                )
    else:
        _render_section_card(
            st,
            title="Recomendaciones estratégicas no validadas",
            body=(
                "Se generó el reporte financiero con información determinística validada. "
                "Las recomendaciones estratégicas no se incluyeron porque no superaron la validación de evidencia."
            ),
            variant="warning",
            badge="Advertencia",
        )

    follow_up = historical.get("recommendation_follow_up", []) if isinstance(historical, dict) else []
    if follow_up:
        st.markdown("### Seguimiento determinístico de recomendaciones previas")
        if historical.get("recommendation_summary"):
            _render_section_card(
                st,
                title="Lectura del seguimiento",
                body=_safe_display_text(historical.get("recommendation_summary")),
                variant="info",
                badge="Informativo",
            )
        columns = st.columns(2)
        for index, item in enumerate(follow_up):
            if not isinstance(item, dict):
                continue
            with columns[index % 2]:
                _render_section_card(
                    st,
                    title=_safe_display_text(item.get("recommendation") or "Recomendación previa"),
                    body=_safe_display_text(item.get("status_reason") or item.get("current_evidence")),
                    variant=_card_variant_from_text(item.get("progress")),
                    badge=item.get("progress") or "En seguimiento",
                    rows=[
                        ("Emitida en", item.get("origin_period")),
                        ("Objetivo original", item.get("objective")),
                        ("Evidencia actual", item.get("current_evidence")),
                    ],
                )

    if missing_items:
        st.markdown("### Información pendiente")
        for item in missing_items:
            _render_section_card(
                st,
                title="Información requerida",
                body=_safe_display_text(item),
                variant="warning",
                badge="Pendiente",
            )
    else:
        st.success("No se reporta información faltante relevante.")


def _render_downloads_tab(
    st: Any,
    artifacts: dict[str, Path | None],
    report_model: dict[str, Any] | None = None,
) -> None:
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
        "Evidence package JSON": "application/json",
    }
    label_by_artifact = {
        "PDF": "PDF",
        "HTML": "HTML",
        "Report model JSON": "modelo JSON del reporte",
        "Strategic analysis JSON": "análisis estratégico JSON",
        "Evidence package JSON": "evidencia JSON",
    }
    st.markdown("### Archivos disponibles")
    columns = st.columns(2)
    for index, (label, path) in enumerate(artifacts.items()):
        display_label = label_by_artifact.get(label, label)
        with columns[index % 2]:
            if path and path.is_file():
                _render_download_card(st, display_label, path, mime_by_label.get(label, "application/octet-stream"))
            else:
                _render_section_card(
                    st,
                    title=display_label,
                    body=f"El archivo {display_label} no está disponible para esta ejecución.",
                    variant="warning",
                    badge="No disponible",
                )
    if report_model and hasattr(st, "expander"):
        with st.expander("Auditoría de cobertura del reporte", expanded=False):
            st.dataframe(
                _ui_section_consistency_rows(
                    report_model,
                    html_available=bool(artifacts.get("HTML") and artifacts["HTML"].is_file()),
                    pdf_available=bool(artifacts.get("PDF") and artifacts["PDF"].is_file()),
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_results(st: Any, result: PipelineRunResult) -> None:
    """Render all result tabs for a completed pipeline run.

    Inputs: Streamlit module and pipeline result.
    Outputs: tabbed results area.
    Assumptions: report artifacts are read-only presentation data.
    """

    artifacts = _artifact_paths(result)
    report_model = _load_json(artifacts["Report model JSON"])
    _render_results_header(st, report_model=report_model, result=result, artifacts=artifacts)
    overview, kpis, anomalies, analysis, recommendations, downloads = st.tabs(
        ["Resumen", "KPIs", "Anomalías", "Análisis", "Recomendaciones", "Descargas"]
    )
    with overview:
        _render_overview_tab(st, report_model, result)
    with kpis:
        _render_kpi_tab(st, report_model)
    with anomalies:
        _render_anomaly_tab(st, report_model)
    with analysis:
        _render_analysis_tab(st, report_model)
    with recommendations:
        _render_recommendations_tab(st, report_model)
    with downloads:
        _render_downloads_tab(st, artifacts, report_model)


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
