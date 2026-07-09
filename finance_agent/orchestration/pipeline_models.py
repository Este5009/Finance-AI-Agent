"""Structured models for full pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_PERIOD_TYPES = frozenset(
    {"monthly", "quarterly", "semester", "annual", "custom", "unknown"}
)


@dataclass(frozen=True)
class DetectedPeriod:
    """Detected reporting period metadata for one user-supplied report.

    Inputs: period type, label, confidence, optional dates, and evidence.
    Outputs: serializable period-detection result.
    Assumptions: low-confidence detection must not drive calculations without override.
    """

    period_type: str
    label: str
    confidence: float
    evidence: tuple[str, ...] = ()
    year: int | None = None
    month: int | None = None
    quarter: int | None = None
    semester: int | None = None
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        """Validate basic detected-period fields after dataclass creation.

        Inputs: this detected period.
        Outputs: None; raises ValueError for invalid fields.
        Assumptions: date strings are ISO-like display values from detection.
        """

        if self.period_type not in SUPPORTED_PERIOD_TYPES:
            raise ValueError(f"Unsupported period_type: {self.period_type}")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("Detected period confidence must be between 0 and 1")

    @property
    def requires_override(self) -> bool:
        """Return whether the detection is too uncertain for final execution.

        Inputs: this detected period.
        Outputs: True for unknown or low-confidence detections.
        Assumptions: 0.65 is the minimum confidence for unattended period use.
        """

        return self.period_type == "unknown" or self.confidence < 0.65

    def to_dict(self) -> dict[str, Any]:
        """Serialize detected period metadata.

        Inputs: this detected period.
        Outputs: JSON-compatible dictionary.
        Assumptions: tuples are rendered as lists for JSON.
        """

        data = asdict(self)
        data["evidence"] = list(self.evidence)
        data["requires_override"] = self.requires_override
        return data


@dataclass(frozen=True)
class PipelineInputModel:
    """Generic user-facing input contract for one report pipeline run.

    Inputs: financial report path, goals path, detected/override period, and language.
    Outputs: serializable model for orchestrator, CLI, and future UI use.
    Assumptions: one financial report and one goals document describe the same period.
    """

    financial_report_path: Path
    goals_document_path: Path
    detected_period: DetectedPeriod
    period_type: str
    period_override: str | None = None
    report_language: str = "es"

    def __post_init__(self) -> None:
        """Validate generic input model fields.

        Inputs: this input model.
        Outputs: None; raises ValueError for invalid configuration.
        Assumptions: callers decide whether missing files are fatal before execution.
        """

        if self.period_type not in SUPPORTED_PERIOD_TYPES:
            raise ValueError(f"Unsupported period_type: {self.period_type}")
        if not str(self.report_language).strip():
            raise ValueError("report_language must be non-empty")
        if self.period_type != self.detected_period.period_type and not self.period_override:
            raise ValueError("period_type must match detected_period unless an override is supplied")

    @property
    def requires_period_override(self) -> bool:
        """Return whether execution needs a user-supplied period override.

        Inputs: this input model.
        Outputs: True when detection is uncertain and no override was supplied.
        Assumptions: UIs should display this state instead of guessing.
        """

        return self.detected_period.requires_override and not self.period_override

    def validate_for_execution(self) -> None:
        """Validate that the generic input can be used for pipeline execution.

        Inputs: this input model.
        Outputs: None; raises ValueError for missing files or missing override.
        Assumptions: detection may be displayed before it is execution-ready.
        """

        if self.requires_period_override:
            raise ValueError("period_override is required for unknown or low-confidence periods")
        if not self.financial_report_path.is_file():
            raise ValueError(f"Financial report does not exist: {self.financial_report_path}")
        if not self.goals_document_path.is_file():
            raise ValueError(f"Goals document does not exist: {self.goals_document_path}")

    @property
    def effective_period_label(self) -> str:
        """Return the override label or the detected label.

        Inputs: this input model.
        Outputs: effective reporting-period label.
        Assumptions: user overrides are authoritative display metadata.
        """

        return self.period_override or self.detected_period.label

    def to_dict(self) -> dict[str, Any]:
        """Serialize the input model for pipeline summaries.

        Inputs: this input model.
        Outputs: JSON-compatible dictionary with string paths.
        Assumptions: paths may point anywhere the caller has permission to read.
        """

        return {
            "financial_report_path": str(self.financial_report_path),
            "goals_document_path": str(self.goals_document_path),
            "detected_period": self.detected_period.to_dict(),
            "period_type": self.period_type,
            "period_override": self.period_override,
            "requires_period_override": self.requires_period_override,
            "effective_period_label": self.effective_period_label,
            "report_language": self.report_language,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for running the existing pipeline entry points.

    Inputs: project paths, Python executable, and optional Ollama configuration.
    Outputs: immutable configuration passed to the orchestrator.
    Assumptions: default paths preserve the repository's current output layout.
    """

    project_root: Path
    python_executable: str
    data_directory: Path
    output_directory: Path
    monthly_workbook: Path
    annual_workbook: Path
    goals_pdf: Path
    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "qwen3:30b-a3b"
    ollama_timeout_seconds: float = 180.0
    stage_timeout_seconds: float = 420.0
    input_model: PipelineInputModel | None = None
    structure_fallback_table_threshold: float = 0.75
    structure_fallback_column_threshold: float = 0.70
    enable_cache: bool = True
    allow_draft_report: bool = False

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        *,
        python_executable: str,
        ollama_endpoint: str = "http://localhost:11434",
        ollama_model: str = "qwen3:30b-a3b",
        ollama_timeout_seconds: float = 180.0,
        stage_timeout_seconds: float = 420.0,
        input_model: PipelineInputModel | None = None,
        structure_fallback_table_threshold: float = 0.75,
        structure_fallback_column_threshold: float = 0.70,
        enable_cache: bool = True,
        allow_draft_report: bool = False,
    ) -> "PipelineConfig":
        """Build a default configuration from the repository root.

        Inputs: project root, Python executable, and optional Ollama settings.
        Outputs: PipelineConfig with standard synthetic input and output paths.
        Assumptions: current stage scripts use the standard repository layout.
        """

        root = Path(project_root).resolve()
        data_directory = root / "data" / "synthetic"
        return cls(
            project_root=root,
            python_executable=python_executable,
            data_directory=data_directory,
            output_directory=root / "outputs",
            monthly_workbook=data_directory / "monthly_financial_report_june_2026.xlsx",
            annual_workbook=data_directory / "annual_financial_report_2026.xlsx",
            goals_pdf=data_directory / "financial_goals_2026.pdf",
            ollama_endpoint=ollama_endpoint,
            ollama_model=ollama_model,
            ollama_timeout_seconds=ollama_timeout_seconds,
            stage_timeout_seconds=stage_timeout_seconds,
            input_model=input_model,
            structure_fallback_table_threshold=structure_fallback_table_threshold,
            structure_fallback_column_threshold=structure_fallback_column_threshold,
            enable_cache=enable_cache,
            allow_draft_report=allow_draft_report,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration for audit output.

        Inputs: this configuration.
        Outputs: JSON-compatible dictionary.
        Assumptions: paths are rendered as strings for portability.
        """

        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        data["input_model"] = self.input_model.to_dict() if self.input_model else None
        return data


@dataclass(frozen=True)
class PipelineStageResult:
    """Result of one orchestrated pipeline stage.

    Inputs: stage metadata, status, outputs, warnings, and runtime.
    Outputs: serializable stage result.
    Assumptions: stdout/stderr snippets are diagnostic and not source-of-truth.
    """

    stage_name: str
    display_name: str
    critical: bool
    success: bool
    skipped: bool
    output_files: tuple[str, ...]
    warnings: tuple[str, ...]
    error: str | None
    runtime_seconds: float
    return_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize one stage result.

        Inputs: this stage result.
        Outputs: JSON-compatible dictionary.
        Assumptions: output paths are already string paths.
        """

        return {
            "stage_name": self.stage_name,
            "display_name": self.display_name,
            "critical": self.critical,
            "success": self.success,
            "skipped": self.skipped,
            "output_files": list(self.output_files),
            "warnings": list(self.warnings),
            "error": self.error,
            "runtime_seconds": self.runtime_seconds,
            "return_code": self.return_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass(frozen=True)
class RuntimeSummary:
    """Aggregate runtime and status counters for one pipeline run.

    Inputs: completed stage results and total elapsed time.
    Outputs: serializable summary fields.
    Assumptions: skipped stages are counted separately from failures.
    """

    total_runtime_seconds: float
    stages_requested: int
    stages_run: int
    stages_succeeded: int
    stages_failed: int
    stages_skipped: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime summary.

        Inputs: this runtime summary.
        Outputs: JSON-compatible dictionary.
        Assumptions: counters are non-negative.
        """

        return asdict(self)


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result returned by the full pipeline orchestrator.

    Inputs: configuration, stage results, output files, and runtime summary.
    Outputs: auditable run result for CLI or future API use.
    Assumptions: success means no critical stage failed.
    """

    success: bool
    stages: tuple[PipelineStageResult, ...]
    output_files: tuple[str, ...]
    warnings: tuple[str, ...]
    runtime_summary: RuntimeSummary
    config: PipelineConfig = field(repr=False)
    cache_hit: bool = False
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline result.

        Inputs: this run result.
        Outputs: JSON-compatible result document.
        Assumptions: output paths remain under the configured output directory.
        """

        return {
            "success": self.success,
            "stages": [stage.to_dict() for stage in self.stages],
            "output_files": list(self.output_files),
            "warnings": list(self.warnings),
            "runtime_summary": self.runtime_summary.to_dict(),
            "config": self.config.to_dict(),
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
        }
