"""Quality and freshness validation for rendered financial reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


MISSING_STRATEGY_PLACEHOLDERS = (
    "Strategic analysis was unavailable",
    "No hay recomendaciones estratégicas generadas",
    "No hay recomendaciones estratÃ©gicas generadas",
)


@dataclass(frozen=True)
class ReportQualityResult:
    """Report quality validation result.

    Inputs: error messages, warnings, and recommendation count.
    Outputs: immutable validation result for tests and CLIs.
    Assumptions: errors block current/final report publishing.
    """

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    recommendation_count: int


def _section(report_model: dict[str, Any], section_id: str) -> dict[str, Any]:
    """Find a section in a report model by ID.

    Inputs: report model dictionary and section ID.
    Outputs: section dictionary or empty dictionary.
    Assumptions: missing sections are validation errors elsewhere.
    """

    for section in report_model.get("sections", []):
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return section
    return {}


def _contains_placeholder(text: str) -> list[str]:
    """Find missing-strategy placeholders in report text.

    Inputs: report text.
    Outputs: list of placeholders found.
    Assumptions: these strings are never acceptable in current strategy-backed reports.
    """

    return [placeholder for placeholder in MISSING_STRATEGY_PLACEHOLDERS if placeholder in text]


def _read_pdf_text(path: Path) -> str:
    """Extract text from a PDF for quality checks.

    Inputs: PDF path.
    Outputs: extracted page text.
    Assumptions: text checks complement visual review but do not replace it.
    """

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def validate_report_model_quality(report_model: dict[str, Any]) -> ReportQualityResult:
    """Validate report model strategy quality.

    Inputs: renderer-agnostic report model dictionary.
    Outputs: ReportQualityResult with blocking errors and warnings.
    Assumptions: current reports require accepted strategic analysis and recommendations.
    """

    errors: list[str] = []
    warnings: list[str] = []
    executive = _section(report_model, "executive_summary")
    recommendations = _section(report_model, "strategic_recommendations")
    executive_content = executive.get("content", {}) if isinstance(executive, dict) else {}
    recommendation_content = (
        recommendations.get("content", {}) if isinstance(recommendations, dict) else {}
    )
    if executive_content.get("analysis_status") != "accepted":
        errors.append("Strategic analysis is not accepted.")
    summary = str(executive_content.get("summary", "")).strip()
    if not summary:
        errors.append("Executive summary is missing.")
    recs = recommendation_content.get("recommendations", [])
    if not isinstance(recs, list) or not recs:
        errors.append("Strategic recommendations are missing.")
        rec_count = 0
    else:
        rec_count = len(recs)
    model_text = json.dumps(report_model, ensure_ascii=False)
    placeholders = _contains_placeholder(model_text)
    if placeholders:
        errors.append(f"Missing-strategy placeholder text found: {placeholders}")
    for section in report_model.get("sections", []):
        if isinstance(section, dict):
            warnings.extend(str(warning) for warning in section.get("warnings", []))
    return ReportQualityResult(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        recommendation_count=rec_count,
    )


def _validate_freshness(model_path: Path, artifact_paths: tuple[Path, ...]) -> list[str]:
    """Validate that rendered artifacts and source references are not stale.

    Inputs: report model path and related artifact paths.
    Outputs: blocking freshness errors.
    Assumptions: artifact modification time should be at least source/model time.
    """

    errors: list[str] = []
    model_mtime = model_path.stat().st_mtime
    model = json.loads(model_path.read_text(encoding="utf-8"))
    for source in model.get("source_references", []):
        source_path = Path(source)
        if source_path.is_file() and source_path.stat().st_mtime > model_mtime + 1:
            errors.append(f"Report model is older than source reference: {source_path}")
    for artifact in artifact_paths:
        if artifact.is_file() and artifact.stat().st_mtime + 1 < model_mtime:
            errors.append(f"Rendered artifact is older than report model: {artifact}")
    return errors


def validate_report_artifacts(
    model_path: str | Path,
    *,
    html_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> ReportQualityResult:
    """Validate a report model and optional rendered HTML/PDF artifacts.

    Inputs: model path plus optional HTML and PDF paths.
    Outputs: ReportQualityResult; invalid when strategy is missing or stale.
    Assumptions: callers raise or stop rendering when result.is_valid is False.
    """

    model_file = Path(model_path)
    report_model = json.loads(model_file.read_text(encoding="utf-8"))
    result = validate_report_model_quality(report_model)
    errors = list(result.errors)
    warnings = list(result.warnings)
    rendered_paths = tuple(
        Path(path)
        for path in (html_path, pdf_path)
        if path is not None
    )
    errors.extend(_validate_freshness(model_file, rendered_paths))
    if html_path is not None:
        html_file = Path(html_path)
        if not html_file.is_file():
            errors.append(f"HTML report does not exist: {html_file}")
        else:
            placeholders = _contains_placeholder(html_file.read_text(encoding="utf-8"))
            if placeholders:
                errors.append(f"HTML contains missing-strategy placeholders: {placeholders}")
    if pdf_path is not None:
        pdf_file = Path(pdf_path)
        if not pdf_file.is_file():
            errors.append(f"PDF report does not exist: {pdf_file}")
        else:
            placeholders = _contains_placeholder(_read_pdf_text(pdf_file))
            if placeholders:
                errors.append(f"PDF contains missing-strategy placeholders: {placeholders}")
    return ReportQualityResult(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        recommendation_count=result.recommendation_count,
    )


def require_report_quality(
    model_path: str | Path,
    *,
    html_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> ReportQualityResult:
    """Require a report model and rendered artifacts to pass quality checks.

    Inputs: model path plus optional rendered artifact paths.
    Outputs: successful ReportQualityResult.
    Assumptions: ValueError is appropriate for CLI/rendering failures.
    """

    result = validate_report_artifacts(
        model_path,
        html_path=html_path,
        pdf_path=pdf_path,
    )
    if not result.is_valid:
        raise ValueError("; ".join(result.errors))
    return result
