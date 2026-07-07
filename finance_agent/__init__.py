"""Deterministic foundations for the Finance AI Agent."""

from finance_agent.ingestion.ingestion import (
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
from finance_agent.calculations.calculation_loader import (
    IntermediateModelLoadError,
    LoadedIntermediateModel,
    LoadedIntermediateTable,
    load_intermediate_model,
)
from finance_agent.anomalies.anomaly_config import AnomalyThresholds
from finance_agent.anomalies.anomaly_engine import (
    AnomalyReport,
    run_anomaly_detection,
    save_anomaly_report,
    save_risk_summary,
)
from finance_agent.anomalies.anomaly_loader import (
    CalculationOutputBundle,
    CalculationOutputLoadError,
    load_calculation_outputs,
)
from finance_agent.analysis.analysis_models import (
    AnalysisRunSummary,
    AnalysisValidationResult,
    StrategicAnalysisResult,
)
from finance_agent.calculations.finance_engine import (
    FinanceCalculationResult,
    run_finance_calculations,
    save_finance_calculation_outputs,
)
from finance_agent.calculations.periods import PeriodScope, filter_table_for_period
from finance_agent.llm.ollama_client import OllamaClient, OllamaError
from finance_agent.agent.ollama_planner import (
    OllamaPlannerResult,
    build_execution_queue,
    build_ollama_planner_prompt,
    create_ollama_investigation_plan,
)
from finance_agent.understanding.intermediate import (
    build_feature_summary,
    build_financial_document_model,
    save_intermediate_outputs,
)
from finance_agent.agent.investigation_planner import (
    build_investigation_plan,
    save_investigation_plan,
)
from finance_agent.agent.planner_loader import (
    PlannerInputBundle,
    PlannerInputError,
    load_planner_inputs,
)
from finance_agent.agent.planner_models import (
    EvidenceRequest,
    InvestigationPlan,
    InvestigationTask,
    PriorityLevel,
)
from finance_agent.agent.planner_validation import (
    PlanValidationResult,
    validate_ollama_plan_response,
)
from finance_agent.retrieval.retrieval_engine import (
    RetrievalInputError,
    build_retrieval_summary,
    execute_retrieval_queue,
    load_execution_queue,
    load_retrieval_context,
)
from finance_agent.retrieval.retrieval_models import (
    EvidencePackage,
    RetrievalRequest,
    RetrievalResult,
    RetrievalRunSummary,
)
from finance_agent.retrieval.retrieval_registry import (
    RetrievalRegistry,
    create_default_registry,
)
from finance_agent.analysis.strategic_analysis import (
    build_analysis_summary,
    build_strategic_analysis_prompt,
    create_strategic_analysis,
    validate_strategic_analysis_response,
)
from finance_agent.understanding.structure_fallback import (
    FallbackSummary,
    detect_low_confidence_items,
    enrich_intermediate_model,
    save_enriched_model,
)
from finance_agent.ingestion.schema import COLUMN_ALIASES, clean_column_name, map_column_alias, normalize_column_names

__all__ = [
    "COLUMN_ALIASES",
    "AnomalyReport",
    "AnomalyThresholds",
    "AnalysisRunSummary",
    "AnalysisValidationResult",
    "CalculationOutputBundle",
    "CalculationOutputLoadError",
    "FinanceCalculationResult",
    "GoalsPdfResult",
    "IntermediateModelLoadError",
    "InvestigationPlan",
    "InvestigationTask",
    "LoadedIntermediateModel",
    "LoadedIntermediateTable",
    "OllamaClient",
    "OllamaError",
    "OllamaPlannerResult",
    "PeriodScope",
    "PlannerInputBundle",
    "PlannerInputError",
    "PlanValidationResult",
    "PriorityLevel",
    "PdfIngestionError",
    "EvidencePackage",
    "RetrievalInputError",
    "RetrievalRegistry",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalRunSummary",
    "StrategicAnalysisResult",
    "WorkbookIngestionError",
    "WorkbookIngestionResult",
    "build_retrieval_summary",
    "build_analysis_summary",
    "build_strategic_analysis_prompt",
    "clean_column_name",
    "create_default_registry",
    "create_strategic_analysis",
    "extract_goals_pdf",
    "execute_retrieval_queue",
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
    "build_execution_queue",
    "build_investigation_plan",
    "build_ollama_planner_prompt",
    "create_ollama_investigation_plan",
    "detect_low_confidence_items",
    "enrich_intermediate_model",
    "FallbackSummary",
    "save_intermediate_outputs",
    "save_enriched_model",
    "save_investigation_plan",
    "run_finance_calculations",
    "run_anomaly_detection",
    "save_anomaly_report",
    "save_finance_calculation_outputs",
    "save_risk_summary",
    "filter_table_for_period",
    "load_planner_inputs",
    "load_execution_queue",
    "load_retrieval_context",
    "validate_strategic_analysis_response",
    "validate_ollama_plan_response",
    "EvidenceRequest",
]
