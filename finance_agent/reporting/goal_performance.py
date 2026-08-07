"""Deterministic goal and budget performance models.

This module is intentionally free of LLM calls. It converts processed finance
outputs into transparent actual-versus-target comparisons that can be reused by
report models, renderers, Streamlit, and historical storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GOAL_STATUS_ORDER: dict[str, int] = {
    "Crítica": 4,
    "En riesgo": 3,
    "Cerca de cumplir": 2,
    "Cumplida": 1,
    "Sin datos": 0,
}


@dataclass(frozen=True)
class GoalRule:
    """Canonical rule describing how one goal should be evaluated.

    Inputs: stable metric identifier, display label, direction, unit, and source
    field names.
    Outputs: immutable rule consumed by deterministic scoring.
    Assumptions: direction is explicit and never inferred from arithmetic sign.
    """

    metric_id: str
    display_label: str
    direction: str
    unit: str
    actual_path: tuple[str, ...]
    target_path: tuple[str, ...]
    goal_id: str
    calculation_method: str


GOAL_RULES: tuple[GoalRule, ...] = (
    GoalRule(
        "total_revenue",
        "Ingresos totales",
        "higher_is_better",
        "USD",
        ("total_revenue",),
        ("budget_vs_actual", "revenue_budget"),
        "GOAL-REVENUE-BUDGET",
        "actual_revenue compared with approved revenue budget; higher is favorable",
    ),
    GoalRule(
        "total_expenses",
        "Gastos totales",
        "lower_is_better",
        "USD",
        ("total_expenses",),
        ("budget_vs_actual", "expense_budget"),
        "GOAL-EXPENSE-BUDGET",
        "actual_expenses compared with approved expense budget; lower is favorable",
    ),
    GoalRule(
        "net_operating_result",
        "Resultado operativo",
        "minimum_threshold",
        "USD",
        ("net_operating_result",),
        ("budget_vs_actual", "net_budget"),
        "GOAL-OPERATING-RESULT",
        "net operating result compared with net budget or minimum positive result",
    ),
    GoalRule(
        "payroll_percentage_of_revenue",
        "Nómina / ingresos",
        "maximum_threshold",
        "ratio",
        ("payroll_percentage_of_revenue",),
        ("goal_thresholds", "payroll_percentage_of_revenue"),
        "GOAL-PAYROLL-RATIO",
        "payroll ratio compared with the configured maximum threshold",
    ),
    GoalRule(
        "collection_rate",
        "Tasa de cobranza",
        "minimum_threshold",
        "ratio",
        ("student_payments", "collection_rate"),
        ("goal_thresholds", "collection_rate"),
        "GOAL-COLLECTION-RATE",
        "collection rate compared with the configured minimum threshold",
    ),
    GoalRule(
        "net_cash_flow",
        "Flujo neto de caja",
        "minimum_threshold",
        "USD",
        ("cash_flow", "net_cash_flow"),
        ("goal_thresholds", "net_cash_flow"),
        "GOAL-CASH-FLOW",
        "net cash flow compared with the minimum zero-cash-flow target",
    ),
)


DEFAULT_THRESHOLDS: dict[str, float] = {
    "payroll_percentage_of_revenue": 0.42,
    "collection_rate": 0.94,
    "net_cash_flow": 0.0,
}


def number_value(value: Any) -> float | None:
    """Coerce a value to float when possible.

    Inputs: arbitrary processed value.
    Outputs: float or None.
    Assumptions: callers use None to represent unavailable evidence.
    """

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Read a nested value from a processed finance payload.

    Inputs: dictionary and key path.
    Outputs: value or None.
    Assumptions: non-dict intermediates mean the value is absent.
    """

    current: Any = payload
    for key in path:
        if key == "goal_thresholds":
            return DEFAULT_THRESHOLDS.get(path[-1])
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _is_satisfied(actual: float, target: float, direction: str) -> bool:
    """Return whether a goal condition is satisfied.

    Inputs: actual value, target value, and explicit direction.
    Outputs: boolean result.
    Assumptions: direction strings are validated by rule construction.
    """

    if direction in {"higher_is_better", "minimum_threshold"}:
        return actual >= target
    if direction in {"lower_is_better", "maximum_threshold"}:
        return actual <= target
    if direction == "exact_target":
        return actual == target
    if direction == "target_range":
        return actual == target
    raise ValueError(f"Unsupported goal direction: {direction}")


def _score(actual: float | None, target: float | None, direction: str) -> tuple[float | None, float | None]:
    """Calculate raw and visible goal score.

    Inputs: actual value, target value, and direction.
    Outputs: raw score and visible score clamped to 0-100.
    Assumptions:
        - satisfied targets score at least 100;
        - adverse gaps reduce score proportionally;
        - zero targets avoid division by zero.
    """

    if actual is None or target is None:
        return None, None
    if direction in {"higher_is_better", "minimum_threshold"}:
        if target == 0:
            raw = 100.0 if actual >= target else 0.0
        else:
            raw = (actual / target) * 100.0
    elif direction in {"lower_is_better", "maximum_threshold"}:
        if actual == 0:
            raw = 100.0 if actual <= target else 0.0
        elif target == 0:
            raw = 100.0 if actual <= 0 else 0.0
        else:
            raw = (target / actual) * 100.0
    elif direction == "exact_target":
        if target == 0:
            raw = 100.0 if actual == 0 else 0.0
        else:
            raw = max(0.0, 100.0 - abs((actual - target) / target) * 100.0)
    else:
        raw = 0.0
    return raw, max(0.0, min(100.0, raw))


def _status(actual: float | None, target: float | None, direction: str, visible_score: float | None) -> str:
    """Classify one goal using the project status bands.

    Inputs: actual, target, direction, and visible score.
    Outputs: Spanish status label.
    Assumptions: missing values are excluded from overall scoring.
    """

    if actual is None or target is None or visible_score is None:
        return "Sin datos"
    if _is_satisfied(actual, target, direction):
        return "Cumplida"
    if visible_score >= 90.0:
        return "Cerca de cumplir"
    if visible_score >= 75.0:
        return "En riesgo"
    return "Crítica"


def _gap_fields(actual: float | None, target: float | None, unit: str) -> dict[str, float | None]:
    """Calculate deterministic gap fields.

    Inputs: actual, target, and unit.
    Outputs: absolute, relative, and percentage-point gaps.
    Assumptions: ratio units use percentage-point gaps; relative gaps need a
    non-zero target.
    """

    if actual is None or target is None:
        return {"absolute_gap": None, "relative_gap": None, "percentage_point_gap": None}
    absolute = actual - target
    relative = absolute / abs(target) if target else None
    pp_gap = absolute if unit == "ratio" else None
    return {
        "absolute_gap": absolute,
        "relative_gap": relative,
        "percentage_point_gap": pp_gap,
    }


def _budget_classification(metric_id: str, gap: float | None) -> str:
    """Classify budget variance directionally.

    Inputs: metric identifier and actual-minus-budget gap.
    Outputs: Spanish favorable/unfavorable label.
    Assumptions: revenue/result positive gaps are favorable, expense positive
    gaps are unfavorable.
    """

    if gap is None:
        return "Sin datos"
    if metric_id in {"total_revenue", "net_operating_result", "collection_rate", "net_cash_flow"}:
        return "Favorable" if gap >= 0 else "Desfavorable"
    if metric_id in {"total_expenses", "payroll_percentage_of_revenue"}:
        return "Favorable" if gap <= 0 else "Desfavorable"
    return "Favorable" if gap >= 0 else "Desfavorable"


def build_goal_performance(
    finance: dict[str, Any],
    *,
    period: str,
    current_source: str,
    historical_directions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build deterministic goal and budget performance payload.

    Inputs: processed finance summary, period slug/label, source artifact, and
    optional historical direction by metric.
    Outputs: renderer-agnostic goal-performance dictionary.
    Assumptions: all calculations use processed Python outputs only.
    """

    historical_directions = historical_directions or {}
    items: list[dict[str, Any]] = []
    for rule in GOAL_RULES:
        actual = number_value(_nested(finance, rule.actual_path))
        target = number_value(_nested(finance, rule.target_path))
        if rule.metric_id == "net_operating_result" and target is None:
            target = 0.0
        raw_score, visible_score = _score(actual, target, rule.direction)
        gaps = _gap_fields(actual, target, rule.unit)
        status = _status(actual, target, rule.direction, visible_score)
        # Every actual-vs-target row needs a directional business reading:
        # below-budget expenses are favorable, above-threshold ratios are not.
        budget_classification = _budget_classification(rule.metric_id, gaps["absolute_gap"])
        items.append(
            {
                "goal_id": rule.goal_id,
                "metric_id": rule.metric_id,
                "display_label": rule.display_label,
                "actual_value": actual,
                "target_value": target,
                "comparison_operator": rule.direction,
                "direction": rule.direction,
                "unit": rule.unit,
                **gaps,
                "achievement_score": visible_score,
                "raw_achievement_score": raw_score,
                "overachievement": max(0.0, (raw_score or 0.0) - 100.0) if raw_score is not None else None,
                "status": status,
                "period": period,
                "department": None,
                "category": None,
                "historical_direction": historical_directions.get(rule.metric_id),
                "source_provenance_actual": {
                    "artifact": current_source,
                    "path": ".".join(rule.actual_path),
                    "source_metric_id": rule.metric_id,
                },
                "source_provenance_target": {
                    "artifact": current_source,
                    "path": ".".join(rule.target_path),
                    "source_metric_id": rule.metric_id,
                },
                "calculation_method": rule.calculation_method,
                "confidence": "high" if actual is not None and target is not None else "low",
                "reconciliation_state": "reconciled" if actual is not None and target is not None else "missing_data",
                "budget_classification": budget_classification,
            }
        )
    valid = [item for item in items if item.get("achievement_score") is not None]
    overall_score = (
        sum(float(item["achievement_score"]) for item in valid) / len(valid)
        if valid
        else None
    )
    counts = {
        "valid_goal_count": len(valid),
        "met_goal_count": sum(1 for item in valid if item["status"] == "Cumplida"),
        "near_goal_count": sum(1 for item in valid if item["status"] == "Cerca de cumplir"),
        "risk_goal_count": sum(1 for item in valid if item["status"] == "En riesgo"),
        "critical_goal_count": sum(1 for item in valid if item["status"] == "Crítica"),
        "excluded_goal_count": len(items) - len(valid),
    }
    strongest = max(valid, key=lambda item: float(item.get("achievement_score") or 0.0), default=None)
    weakest = max(valid, key=lambda item: GOAL_STATUS_ORDER.get(str(item.get("status")), 0), default=None)
    contribution = [
        {
            "goal_id": item["goal_id"],
            "metric_id": item["metric_id"],
            "weight": (1.0 / len(valid)) if valid else None,
            "weighted_score": (float(item["achievement_score"]) / len(valid)) if valid else None,
        }
        for item in valid
    ]
    return {
        "overall_score": overall_score,
        **counts,
        "weighting_method": "equal_weight_valid_goals",
        "weighted_contributions": contribution,
        "items": items,
        "budget_items": [
            item
            for item in items
            if item.get("budget_classification") is not None
        ],
        "strongest_goal": strongest,
        "highest_priority_gap": weakest,
        "technical_details": {
            "scoring_formula": (
                "higher/minimum: actual ÷ target × 100; lower/maximum: target ÷ actual × 100; "
                "visible score clamped to 0-100; satisfied goals report status Cumplida."
            ),
            "status_bands": "Cumplida if condition satisfied; Cerca 90-99.9; En riesgo 75-89.9; Crítica below 75; Sin datos when excluded.",
            "directions": sorted({rule.direction for rule in GOAL_RULES}),
        },
        "deterministic_conclusion": _deterministic_conclusion(overall_score, counts, strongest, weakest),
    }


def _deterministic_conclusion(
    overall_score: float | None,
    counts: dict[str, int],
    strongest: dict[str, Any] | None,
    weakest: dict[str, Any] | None,
) -> str:
    """Build a concise deterministic Spanish conclusion.

    Inputs: score, count summary, strongest and weakest goal records.
    Outputs: user-facing deterministic conclusion.
    Assumptions: statement uses only fields from computed goal records.
    """

    if overall_score is None:
        return "No hay metas con datos suficientes para calcular un cumplimiento consolidado."
    strongest_label = strongest.get("display_label") if strongest else "una meta disponible"
    weakest_label = weakest.get("display_label") if weakest else "sin brechas críticas"
    return (
        f"El cumplimiento consolidado es {overall_score:.1f}/100, con "
        f"{counts['met_goal_count']} de {counts['valid_goal_count']} metas cumplidas. "
        f"El mejor desempeño corresponde a {strongest_label}; la mayor atención requerida corresponde a {weakest_label}."
    )
