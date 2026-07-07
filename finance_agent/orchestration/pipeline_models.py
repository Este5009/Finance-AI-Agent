"""Structured models for full pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
        }
