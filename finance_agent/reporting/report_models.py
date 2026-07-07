"""Structured report models for renderer-agnostic report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_SECTION_IDS: tuple[str, ...] = (
    "cover",
    "executive_summary",
    "financial_health_overview",
    "kpi_overview",
    "revenue_analysis",
    "expense_analysis",
    "department_analysis",
    "anomaly_summary",
    "investigation_evidence",
    "strategic_recommendations",
    "missing_information",
    "appendix",
)


@dataclass(frozen=True)
class ReportSection:
    """One renderer-agnostic report section.

    Inputs: stable section ID, title, content payload, sources, and warnings.
    Outputs: serializable section used by future renderers.
    Assumptions: content is presentation-ready data, not renderer-specific markup.
    """

    section_id: str
    title: str
    content: dict[str, Any]
    source_references: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize one report section.

        Inputs: this report section.
        Outputs: JSON-compatible dictionary.
        Assumptions: content values are already JSON-compatible.
        """

        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "source_references": list(self.source_references),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ReportModel:
    """Complete renderer-agnostic financial report representation.

    Inputs: report identity, period metadata, ordered sections, and sources.
    Outputs: serializable model for future PDF/HTML/UI/PPT renderers.
    Assumptions: renderers decide layout; this model carries structured content only.
    """

    report_id: str
    period_slug: str
    report_period: str
    renderer_contract_version: str
    sections: tuple[ReportSection, ...]
    source_references: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report model.

        Inputs: this report model.
        Outputs: JSON-compatible report document.
        Assumptions: section ordering is meaningful for downstream renderers.
        """

        return {
            "report_id": self.report_id,
            "period_slug": self.period_slug,
            "report_period": self.report_period,
            "renderer_contract_version": self.renderer_contract_version,
            "section_count": len(self.sections),
            "sections": [section.to_dict() for section in self.sections],
            "source_references": list(self.source_references),
        }
