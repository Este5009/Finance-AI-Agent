"""Deterministic context-ranking and telemetry helpers for LLM stages."""

from __future__ import annotations

import json
from typing import Any, Iterable


SEVERITY_SCORES = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def estimate_tokens_from_text(text: str) -> int:
    """Estimate prompt tokens from character count.

    Inputs: prompt or JSON text.
    Outputs: rough token estimate using a conservative four-character heuristic.
    Assumptions: this is profiler telemetry, not billing or model accounting.
    """

    return max(1, len(text) // 4) if text else 0


def compact_json_size(value: Any) -> int:
    """Return the byte-like character size of compact JSON context.

    Inputs: JSON-compatible value.
    Outputs: compact serialized character count.
    Assumptions: callers pass processed pipeline dictionaries, not raw files.
    """

    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def normalize_severity(value: Any) -> str:
    """Normalize an anomaly severity value for deterministic scoring.

    Inputs: arbitrary anomaly severity value.
    Outputs: lowercase severity label or an empty string.
    Assumptions: unknown labels sort below known severities.
    """

    return str(value or "").strip().casefold()


def anomaly_financial_impact(anomaly: dict[str, Any]) -> float:
    """Extract a deterministic impact score from common anomaly fields.

    Inputs: one processed anomaly dictionary.
    Outputs: largest absolute numeric impact-like value found.
    Assumptions: field names vary across anomaly rules, so this is ranking-only.
    """

    candidate_fields = (
        "financial_impact",
        "impact",
        "amount",
        "variance",
        "variance_amount",
        "observed_value",
        "value",
    )
    impacts: list[float] = []
    for field_name in candidate_fields:
        value = anomaly.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            impacts.append(abs(float(value)))
    return max(impacts) if impacts else 0.0


def anomaly_rank_key(anomaly: dict[str, Any], index: int = 0) -> tuple[Any, ...]:
    """Build the sort key used to prioritize anomalies before LLM planning.

    Inputs: anomaly dictionary and original index.
    Outputs: tuple ordered from most to least important.
    Assumptions: stable original order is retained as the final tie-breaker.
    """

    severity = normalize_severity(anomaly.get("severity"))
    repeated = bool(
        anomaly.get("repeated")
        or anomaly.get("is_repeated")
        or anomaly.get("repeated_issue")
        or anomaly.get("repeat_count", 0)
    )
    goal_violation = bool(
        anomaly.get("goal_violation")
        or anomaly.get("violates_goal")
        or anomaly.get("goal_breach")
    )
    operational = bool(
        anomaly.get("operational_importance")
        or anomaly.get("operationally_important")
    )
    return (
        SEVERITY_SCORES.get(severity, 0),
        anomaly_financial_impact(anomaly),
        int(repeated),
        int(goal_violation),
        int(operational),
        -index,
    )


def rank_anomalies(
    anomalies: Iterable[dict[str, Any]],
    *,
    allowed_severities: set[str] | None = None,
    max_count: int | None = None,
) -> list[dict[str, Any]]:
    """Rank processed anomalies deterministically for compact LLM context.

    Inputs: anomaly dictionaries, optional severity filter, and maximum count.
    Outputs: ranked anomaly dictionaries.
    Assumptions: filtering affects LLM context only; Python still keeps all data.
    """

    normalized_allowed = (
        {severity.casefold() for severity in allowed_severities}
        if allowed_severities
        else None
    )
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for index, anomaly in enumerate(anomalies):
        if not isinstance(anomaly, dict):
            continue
        severity = normalize_severity(anomaly.get("severity"))
        if normalized_allowed is not None and severity not in normalized_allowed:
            continue
        candidates.append((anomaly_rank_key(anomaly, index), anomaly))
    ranked = [anomaly for _, anomaly in sorted(candidates, key=lambda item: item[0], reverse=True)]
    return ranked[:max_count] if max_count is not None else ranked


def deduplicate_dicts(
    items: Iterable[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Remove duplicate dictionaries while preserving first occurrence.

    Inputs: dictionaries and the fields that define equivalence.
    Outputs: deduplicated list.
    Assumptions: this removes repeated context only; it does not merge facts.
    """

    seen: set[tuple[str, ...]] = set()
    deduplicated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = tuple(str(item.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def merge_telemetry(*parts: dict[str, Any] | None) -> dict[str, Any]:
    """Merge telemetry dictionaries using additive numeric semantics.

    Inputs: optional telemetry fragments.
    Outputs: one telemetry dictionary.
    Assumptions: repeated numeric fields represent independent stage durations.
    """

    merged: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = float(merged.get(key, 0.0)) + float(value)
            else:
                merged[key] = value
    return merged
