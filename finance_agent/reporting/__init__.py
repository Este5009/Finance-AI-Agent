"""Renderer-agnostic reporting model generation."""

from finance_agent.reporting.report_engine import (
    ReportInputBundle,
    build_report_model,
    load_report_inputs,
    rebuild_report_artifacts_from_processed_outputs,
    refresh_strategic_historical_context,
    report_model_needs_historical_refresh,
    save_report_model,
    validate_report_model,
)
from finance_agent.reporting.report_models import (
    REQUIRED_SECTION_IDS,
    ReportModel,
    ReportSection,
)
from finance_agent.reporting.report_quality import (
    ReportQualityResult,
    require_report_quality,
    validate_report_artifacts,
    validate_report_model_quality,
)
from finance_agent.reporting.renderers import (
    load_report_model,
    report_strategy_warnings,
    render_report_html,
    render_report_pdf,
    save_report_html,
    validate_strategy_available,
)

__all__ = [
    "REQUIRED_SECTION_IDS",
    "ReportInputBundle",
    "ReportModel",
    "ReportQualityResult",
    "ReportSection",
    "build_report_model",
    "load_report_model",
    "load_report_inputs",
    "rebuild_report_artifacts_from_processed_outputs",
    "render_report_html",
    "render_report_pdf",
    "refresh_strategic_historical_context",
    "report_model_needs_historical_refresh",
    "report_strategy_warnings",
    "save_report_model",
    "save_report_html",
    "require_report_quality",
    "validate_report_artifacts",
    "validate_report_model_quality",
    "validate_strategy_available",
    "validate_report_model",
]
