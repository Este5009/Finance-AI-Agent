"""Renderer-agnostic reporting model generation."""

from finance_agent.reporting.report_engine import (
    ReportInputBundle,
    build_report_model,
    load_report_inputs,
    save_report_model,
    validate_report_model,
)
from finance_agent.reporting.report_models import (
    REQUIRED_SECTION_IDS,
    ReportModel,
    ReportSection,
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
    "ReportSection",
    "build_report_model",
    "load_report_model",
    "load_report_inputs",
    "render_report_html",
    "render_report_pdf",
    "report_strategy_warnings",
    "save_report_model",
    "save_report_html",
    "validate_strategy_available",
    "validate_report_model",
]
