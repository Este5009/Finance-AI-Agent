"""Tests for financial-health ratios and organizational-unit budget presentation."""

from __future__ import annotations

from finance_agent.reporting.presentation import build_presentation_view
from finance_agent.reasoning.reasoning_pipeline import build_executive_evidence_package


def _section(section_id: str, content: dict[str, object]) -> dict[str, object]:
    """Build a minimal report-model section fixture.

    Inputs: section ID and content dictionary.
    Outputs: section dictionary compatible with ``build_presentation_view``.
    Assumptions: source references are irrelevant for these presentation tests.
    """

    return {"section_id": section_id, "title": section_id, "content": content, "source_references": [], "warnings": []}


def _report_model() -> dict[str, object]:
    """Build a compact report model containing new feature payloads.

    Inputs: none.
    Outputs: renderer-agnostic report model fixture.
    Assumptions: values are deterministic outputs from calculation/report stages.
    """

    return {
        "report_id": "test",
        "period_slug": "2026_09",
        "report_period": "2026_09",
        "sections": [
            _section("executive_summary", {"summary": "Resumen.", "key_findings": [], "root_causes": [], "confidence": 0.9}),
            _section(
                "financial_health_overview",
                {
                    "total_revenue": 2_000_000,
                    "total_expenses": 1_800_000,
                    "net_operating_result": 200_000,
                    "net_cash_flow": 50_000,
                    "ending_cash": 1_000_000,
                    "financial_health_ratios": [
                        {
                            "metric": "current_ratio",
                            "display_label": "Razón corriente",
                            "value": 1.62,
                            "unit": "ratio_number",
                            "availability": "available",
                            "classification": "Bueno",
                            "reference_range": "Crítico < 1.00; Aceptable 1.00–1.49; Bueno ≥ 1.50",
                            "reference_origin_es": "Referencia interna configurable de gestión",
                            "is_regulatory_limit": False,
                            "formula": "Activo corriente / Pasivo corriente",
                            "interpretation": "Mide liquidez corriente.",
                            "missing_inputs": [],
                        }
                    ],
                    "kpi_comparisons": {"items": {}},
                },
            ),
            _section("kpi_overview", {"kpis": []}),
            _section("goal_budget_performance", {"items": [], "overall_score": None}),
            _section(
                "department_analysis",
                {
                    "department_summary": [
                        {
                            "department": "Engineering",
                            "budget_revenue": 500_000,
                            "actual_revenue": 540_000,
                            "budget_expenses": 420_000,
                            "actual_expenses": 430_000,
                            "expense_variance": 10_000,
                            "expense_variance_pct": 0.0238,
                            "net_operating_result": 110_000,
                        },
                        {
                            "department": "Business",
                            "budget_revenue": 460_000,
                            "actual_revenue": 440_000,
                            "budget_expenses": 390_000,
                            "actual_expenses": 410_000,
                            "expense_variance": 20_000,
                            "expense_variance_pct": 0.0513,
                            "net_operating_result": 30_000,
                        },
                    ]
                },
            ),
        ],
    }


def test_kpi_ratio_cards_show_management_reference_not_regulatory_limit() -> None:
    """Verify KPI dashboard ratio cards expose canonical threshold provenance."""

    view = build_presentation_view(_report_model())
    cards = view["financial_health_ratios"]

    assert cards[0]["label"] == "Razón corriente"
    assert cards[0]["value"] == "1.62"
    assert cards[0]["classification"] == "Bueno"
    assert cards[0]["reference_origin"] == "Referencia interna configurable de gestión"
    assert cards[0]["is_regulatory_limit"] is False


def test_organizational_unit_budget_view_uses_grouped_actual_budget_rows() -> None:
    """Verify unit budget charts compare actual and budget without stacking."""

    view = build_presentation_view(_report_model())
    units = view["goal_budget"]["organizational_units"]

    assert units["available"] is True
    assert len(units["rows"]) == 2
    revenue_group = next(group for group in units["chart_groups"] if group["metric_key"] == "revenue")
    assert revenue_group["encoding"] == "grouped_actual_reference"
    assert {row["series"] for row in revenue_group["rows"]} == {"Real", "Presupuesto"}
    assert any(row["metric"] == "Ingeniería" and row["value"] == 540_000 for row in revenue_group["rows"])
    assert any(row["metric"] == "Ingeniería" and row["value"] == 500_000 for row in revenue_group["rows"])


def test_executive_evidence_package_includes_material_unit_budget_fields() -> None:
    """Verify AI evidence receives material unit findings, not raw workbook rows."""

    package = build_executive_evidence_package(
        {},
        period_slug="2026_09",
        finance_summary={
            "finance_summary": {
                "total_expenses": 1_800_000,
            },
            "department_summary": [
                {
                    "department": "Engineering",
                    "actual_revenue": 540_000,
                    "budget_revenue": 500_000,
                    "revenue_variance": 40_000,
                    "revenue_variance_pct": 0.08,
                    "actual_expenses": 430_000,
                    "budget_expenses": 420_000,
                    "expense_variance": 10_000,
                    "expense_variance_pct": 0.0238,
                    "net_operating_result": 110_000,
                }
            ],
        },
        anomaly_report={},
        historical_context={},
    )

    driver = package["department_drivers"][0]
    assert driver["department"] == "Engineering"
    assert driver["actual_revenue"] == 540_000
    assert driver["budget_revenue"] == 500_000
    assert driver["net_contribution"] == 110_000
