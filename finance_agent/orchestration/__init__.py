"""Pipeline orchestration entry points for the Finance AI Agent."""

from finance_agent.orchestration.pipeline_models import (
    DEFAULT_OLLAMA_MODEL,
    EXPERIMENTAL_FAST_OLLAMA_MODEL,
    DetectedPeriod,
    PipelineConfig,
    PipelineInputModel,
    PipelineRunResult,
    PipelineStageResult,
    RuntimeSummary,
)
from finance_agent.orchestration.pipeline_orchestrator import (
    build_default_stages,
    run_full_pipeline,
    run_object_pipeline_for_report,
    run_pipeline_for_report,
)
from finance_agent.orchestration.period_detection import (
    build_pipeline_input_model,
    detect_period,
)

__all__ = [
    "DetectedPeriod",
    "DEFAULT_OLLAMA_MODEL",
    "EXPERIMENTAL_FAST_OLLAMA_MODEL",
    "PipelineConfig",
    "PipelineInputModel",
    "PipelineRunResult",
    "PipelineStageResult",
    "RuntimeSummary",
    "build_pipeline_input_model",
    "build_default_stages",
    "detect_period",
    "run_full_pipeline",
    "run_object_pipeline_for_report",
    "run_pipeline_for_report",
]
