"""Tests for deterministic goal and budget performance scoring."""

from __future__ import annotations

import math

from finance_agent.reporting.goal_performance import build_goal_performance


def _finance_summary() -> dict[str, object]:
    """Return a compact September-like finance summary for goal tests.

    Inputs: none.
    Outputs: finance summary dictionary with deterministic actuals and budgets.
    Assumptions: values are already calculated by Python finance stages.
    """

    return {
        "total_revenue": 2_123_856.0,
        "total_expenses": 2_096_356.0,
        "net_operating_result": 27_500.0,
        "payroll_percentage_of_revenue": 0.46,
        "student_payments": {"collection_rate": 0.92},
        "cash_flow": {"net_cash_flow": 60_000.0},
        "budget_vs_actual": {
            "revenue_budget": 2_167_200.0,
            "expense_budget": 2_015_496.0,
            "net_budget": 151_704.0,
        },
    }


def _items_by_metric(result: dict[str, object]) -> dict[str, dict[str, object]]:
    """Index goal performance items by canonical metric id.

    Inputs: goal performance result.
    Outputs: mapping from metric id to item.
    Assumptions: each configured goal has a unique metric id.
    """

    return {item["metric_id"]: item for item in result["items"]}  # type: ignore[index]


def test_budget_performance_uses_direction_aware_variances() -> None:
    """Revenue and expenses classify gaps according to metric direction."""

    result = build_goal_performance(
        _finance_summary(),
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )
    items = _items_by_metric(result)

    revenue = items["total_revenue"]
    assert revenue["actual_value"] == 2_123_856.0
    assert revenue["target_value"] == 2_167_200.0
    assert revenue["absolute_gap"] == -43_344.0
    assert revenue["budget_classification"] == "Desfavorable"
    assert revenue["status"] == "Cerca de cumplir"

    expenses = items["total_expenses"]
    assert expenses["actual_value"] == 2_096_356.0
    assert expenses["target_value"] == 2_015_496.0
    assert expenses["absolute_gap"] == 80_860.0
    assert expenses["budget_classification"] == "Desfavorable"
    assert expenses["status"] == "Cerca de cumplir"


def test_ratio_goals_use_percentage_point_gaps() -> None:
    """Ratio goals expose percentage-point gaps without asking the LLM."""

    result = build_goal_performance(
        _finance_summary(),
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )
    items = _items_by_metric(result)

    payroll = items["payroll_percentage_of_revenue"]
    assert payroll["target_value"] == 0.42
    assert math.isclose(payroll["absolute_gap"], 0.04)
    assert math.isclose(payroll["percentage_point_gap"], 0.04)
    assert payroll["budget_classification"] == "Desfavorable"

    collection = items["collection_rate"]
    assert collection["target_value"] == 0.94
    assert math.isclose(collection["absolute_gap"], -0.02)
    assert math.isclose(collection["percentage_point_gap"], -0.02)


def test_overall_goal_score_uses_equal_weighting_of_valid_goals() -> None:
    """Overall score is a deterministic equal-weight average of valid goals."""

    result = build_goal_performance(
        _finance_summary(),
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )

    items = [item for item in result["items"] if item["achievement_score"] is not None]
    expected = sum(float(item["achievement_score"]) for item in items) / len(items)
    assert result["valid_goal_count"] == len(items)
    assert math.isclose(float(result["overall_score"]), expected)
    assert result["weighting_method"] == "equal_weight_valid_goals"


def test_missing_budget_target_is_excluded_but_reported() -> None:
    """Missing deterministic targets are transparent and excluded from scoring."""

    finance = _finance_summary()
    finance["budget_vs_actual"] = {"expense_budget": 2_015_496.0}

    result = build_goal_performance(
        finance,
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )
    revenue = _items_by_metric(result)["total_revenue"]

    assert revenue["status"] == "Sin datos"
    assert revenue["achievement_score"] is None
    assert revenue["confidence"] == "low"
    assert revenue["reconciliation_state"] == "missing_data"


def test_zero_threshold_edge_cases_are_safe_and_deterministic() -> None:
    """Zero targets do not produce divide-by-zero or LLM-filled values."""

    positive = _finance_summary()
    positive["cash_flow"] = {"net_cash_flow": 1.0}
    positive_result = build_goal_performance(
        positive,
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )
    assert _items_by_metric(positive_result)["net_cash_flow"]["achievement_score"] == 100.0
    assert _items_by_metric(positive_result)["net_cash_flow"]["status"] == "Cumplida"

    negative = _finance_summary()
    negative["cash_flow"] = {"net_cash_flow": -1.0}
    negative_result = build_goal_performance(
        negative,
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )
    assert _items_by_metric(negative_result)["net_cash_flow"]["achievement_score"] == 0.0
    assert _items_by_metric(negative_result)["net_cash_flow"]["status"] == "Crítica"


def test_source_provenance_is_retained_for_audit() -> None:
    """Every deterministic goal row carries source provenance for diagnostics."""

    result = build_goal_performance(
        _finance_summary(),
        period="2026_09",
        current_source="outputs/calculations/finance_summary_2026_09.json",
    )

    for item in result["items"]:
        actual = item["source_provenance_actual"]
        target = item["source_provenance_target"]
        assert actual["artifact"] == "outputs/calculations/finance_summary_2026_09.json"
        assert actual["path"]
        assert target["artifact"]
