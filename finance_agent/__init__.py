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
from finance_agent.finance_engine import (
    FinanceCalculationResult,
    run_finance_calculations,
    save_finance_calculation_outputs,
)
from finance_agent.periods import PeriodScope, filter_table_for_period
from finance_agent.intermediate import (
    build_feature_summary,
    build_financial_document_model,
    save_intermediate_outputs,
)
from finance_agent.schema import COLUMN_ALIASES, clean_column_name, map_column_alias, normalize_column_names

__all__ = [
    "COLUMN_ALIASES",
    "FinanceCalculationResult",
    "GoalsPdfResult",
    "IntermediateModelLoadError",
    "LoadedIntermediateModel",
    "LoadedIntermediateTable",
    "PeriodScope",
    "PdfIngestionError",
    "WorkbookIngestionError",
    "WorkbookIngestionResult",
    "clean_column_name",
    "extract_goals_pdf",
    "inspect_sheet",
    "inspect_workbook",
    "load_excel_workbook",
    "load_intermediate_model",
    "load_raw_excel_workbook",
    "map_column_alias",
    "normalize_column_names",
    "build_feature_summary",
    "build_financial_document_model",
    "save_intermediate_outputs",
    "run_finance_calculations",
    "save_finance_calculation_outputs",
    "filter_table_for_period",
]
