"""Deterministic foundations for the Finance AI Agent."""

from finance_agent.ingestion import (
    GoalsPdfResult,
    PdfIngestionError,
    WorkbookIngestionError,
    WorkbookIngestionResult,
    extract_goals_pdf,
    inspect_sheet,
    inspect_workbook,
    load_excel_workbook,
    load_raw_excel_workbook,
)
from finance_agent.calculation_loader import (
    IntermediateModelLoadError,
    LoadedIntermediateModel,
    LoadedIntermediateTable,
    load_intermediate_model,
)
from finance_agent.anomaly_config import AnomalyThresholds
from finance_agent.anomaly_engine import (
    AnomalyReport,
    run_anomaly_detection,
    save_anomaly_report,
    save_risk_summary,
)
from finance_agent.anomaly_loader import (
    CalculationOutputBundle,
    CalculationOutputLoadError,
    load_calculation_outputs,
)
from finance_agent.finance_engine import (
    FinanceCalculationResult,
    run_finance_calculations,
    save_finance_calculation_outputs,
)
from finance_agent.periods import PeriodScope, filter_table_for_period
from finance_agent.ollama_client import OllamaClient, OllamaError
from finance_agent.intermediate import (
    build_feature_summary,
    build_financial_document_model,
    save_intermediate_outputs,
)
from finance_agent.structure_fallback import (
    FallbackSummary,
    detect_low_confidence_items,
    enrich_intermediate_model,
    save_enriched_model,
)
from finance_agent.schema import COLUMN_ALIASES, clean_column_name, map_column_alias, normalize_column_names

__all__ = [
    "COLUMN_ALIASES",
    "AnomalyReport",
    "AnomalyThresholds",
    "CalculationOutputBundle",
    "CalculationOutputLoadError",
    "FinanceCalculationResult",
    "GoalsPdfResult",
    "IntermediateModelLoadError",
    "LoadedIntermediateModel",
    "LoadedIntermediateTable",
    "OllamaClient",
    "OllamaError",
    "PeriodScope",
    "PdfIngestionError",
    "WorkbookIngestionError",
    "WorkbookIngestionResult",
    "clean_column_name",
    "extract_goals_pdf",
    "inspect_sheet",
    "inspect_workbook",
    "load_excel_workbook",
    "load_calculation_outputs",
    "load_intermediate_model",
    "load_raw_excel_workbook",
    "map_column_alias",
    "normalize_column_names",
    "build_feature_summary",
    "build_financial_document_model",
    "detect_low_confidence_items",
    "enrich_intermediate_model",
    "FallbackSummary",
    "save_intermediate_outputs",
    "save_enriched_model",
    "run_finance_calculations",
    "run_anomaly_detection",
    "save_anomaly_report",
    "save_finance_calculation_outputs",
    "save_risk_summary",
    "filter_table_for_period",
]
