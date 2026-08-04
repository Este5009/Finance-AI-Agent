"""Read-only historical memory retrieval tools backed by SQLite."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

from finance_agent.memory.models import DEFAULT_MEMORY_DB_PATH
from finance_agent.memory.repository import MemoryRepository
from finance_agent.memory.retrieval_models import (
    HistoricalMetricPoint,
    HistoricalPeriod,
    MemoryToolResult,
)
from finance_agent.retrieval.retrieval_models import RetrievalResult


VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.% -]{1,120}$")
VALID_PERIOD_RE = re.compile(
    r"^(20\d{2})(?:[-_](0[1-9]|1[0-2]|Q[1-4]|S[1-2]))?$"
)
MAX_TEXT_VALUE_LENGTH = 240
METRIC_ALIASES = {
    "collection_rate": ("collection_rate", "student_payment_collection_rate"),
    "student_payment_collection_rate": ("student_payment_collection_rate", "collection_rate"),
}
FINANCE_SUMMARY_METRIC_PATHS = {
    "total_revenue": ("finance_summary", "total_revenue"),
    "total_expenses": ("finance_summary", "total_expenses"),
    "net_operating_result": ("finance_summary", "net_operating_result"),
    "net_cash_flow": ("finance_summary", "cash_flow", "net_cash_flow"),
    "ending_cash": ("finance_summary", "cash_flow", "ending_cash"),
    "payroll_percentage_of_revenue": ("finance_summary", "payroll_percentage_of_revenue"),
    "collection_rate": ("finance_summary", "student_payments", "collection_rate"),
    "student_payment_collection_rate": ("finance_summary", "student_payments", "collection_rate"),
}


def _repository(database_path: str | Path | None = None) -> MemoryRepository:
    """Create a repository for read-only retrieval functions.

    Inputs: optional database path.
    Outputs: MemoryRepository.
    Assumptions: repository initialization is safe and idempotent.
    """

    return MemoryRepository(database_path or DEFAULT_MEMORY_DB_PATH)


def period_sort_key(period: str) -> tuple[int, int, str]:
    """Sort monthly, quarterly, semester, annual, and custom period IDs.

    Inputs: period identifier such as 2026_06, 2026-Q2, 2026-S1, or 2026.
    Outputs: sortable tuple.
    Assumptions: unknown/custom suffixes sort after annual within the same year.
    """

    normalized = str(period).replace("-", "_").upper()
    match = re.match(r"^(20\d{2})(?:_(.+))?$", normalized)
    if not match:
        return (9999, 99, normalized)
    year = int(match.group(1))
    suffix = match.group(2)
    if suffix is None:
        return (year, 12, normalized)
    if suffix.isdigit():
        return (year, int(suffix), normalized)
    if suffix.startswith("Q") and suffix[1:].isdigit():
        return (year, int(suffix[1:]) * 3, normalized)
    if suffix.startswith("S") and suffix[1:].isdigit():
        return (year, int(suffix[1:]) * 6, normalized)
    return (year, 99, normalized)


def _validate_period(value: str, field_name: str = "period") -> str:
    """Validate and normalize a period identifier.

    Inputs: user-supplied period and field name.
    Outputs: underscore-normalized period.
    Assumptions: custom labels must still begin with a four-digit year.
    """

    period = str(value or "").strip().replace("-", "_")
    if not period or not re.match(r"^20\d{2}(?:_[A-Za-z0-9]+)?$", period):
        raise ValueError(f"{field_name} must be a valid period identifier")
    return period


def _validate_limit(value: Any, field_name: str, *, minimum: int = 1, maximum: int = 120) -> int:
    """Validate an integer limit/window argument.

    Inputs: user-supplied value, field name, and bounds.
    Outputs: integer value.
    Assumptions: small bounded windows keep retrieval compact.
    """

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return parsed


def _validate_identifier(value: str | None, field_name: str) -> str | None:
    """Validate an internal canonical identifier filter.

    Inputs: optional text value and field name.
    Outputs: stripped value or None.
    Assumptions: this is for internal metric/artifact/status identifiers, not
    business names from uploaded data. SQL still uses bound parameters.
    """

    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    if not VALID_IDENTIFIER_RE.match(stripped):
        raise ValueError(f"{field_name} contains unsupported characters")
    return stripped


def _validate_text_value(
    value: str | None,
    field_name: str,
    *,
    required: bool = False,
    maximum_length: int = MAX_TEXT_VALUE_LENGTH,
) -> str | None:
    """Normalize and validate a bound SQL text value.

    Inputs: raw value, field name, required flag, and maximum length.
    Outputs: Unicode-normalized text value or None.
    Assumptions: returned values are used only as SQLite bound parameters, never
    as table/column names or SQL snippets.
    """

    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    normalized = unicodedata.normalize("NFC", str(value))
    if any(char == "\x00" or unicodedata.category(char) in {"Cc", "Cf"} for char in normalized):
        raise ValueError(f"{field_name} contains control characters")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if len(normalized) > maximum_length:
        raise ValueError(f"{field_name} is too long")
    return normalized


def _validate_detail_level(value: str) -> str:
    """Validate a detail-level argument.

    Inputs: requested detail level.
    Outputs: normalized detail level.
    Assumptions: only compact summary or full structured records are supported.
    """

    level = str(value or "summary").strip().casefold()
    if level not in {"summary", "full"}:
        raise ValueError("detail_level must be 'summary' or 'full'")
    return level


def _row_dict(row: Any) -> dict[str, Any]:
    """Convert a sqlite row into a dictionary.

    Inputs: sqlite row.
    Outputs: dictionary.
    Assumptions: row values are JSON-compatible SQLite scalars.
    """

    return dict(row)


def _period_rows(
    repository: MemoryRepository,
    *,
    limit: int,
    before_period: str | None = None,
    include_current: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Fetch and chronologically sort period rows.

    Inputs: repository, limit, before period, and inclusivity.
    Outputs: newest window in chronological order.
    Assumptions: filtering is conservative and excludes future periods.
    """

    rows = [_row_dict(row) for row in repository.fetch_periods()]
    rows = [row for row in rows if str(row.get("status") or "").lower() == "completed"]
    if before_period is not None:
        before = _validate_period(before_period, "before_period")
        rows = [
            row
            for row in rows
            if period_sort_key(row["period"]) < period_sort_key(before)
            or (include_current and period_sort_key(row["period"]) == period_sort_key(before))
        ]
    rows = _latest_row_per_period(rows)
    rows = sorted(rows, key=lambda row: period_sort_key(row["period"]))
    return tuple(rows[-limit:])


def _latest_row_per_period(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one completed run row per period, preferring the latest update.

    Inputs: pipeline run rows from SQLite.
    Outputs: de-duplicated rows keyed by period.
    Assumptions: repeated completed rows for a period represent reprocessing or
    revision history; read-only retrieval should use the newest accepted row.
    """

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        period = str(row.get("period") or "")
        if not period:
            continue
        current = latest.get(period)
        row_stamp = str(row.get("updated_at_utc") or row.get("completed_at_utc") or "")
        current_stamp = str(current.get("updated_at_utc") or current.get("completed_at_utc") or "") if current else ""
        if current is None or row_stamp >= current_stamp:
            latest[period] = row
    return list(latest.values())


def _unavailable(tool_name: str, message: str) -> MemoryToolResult:
    """Build an explicit unavailable retrieval result.

    Inputs: tool name and unavailable message.
    Outputs: MemoryToolResult with success false.
    Assumptions: missing history should not crash callers.
    """

    return MemoryToolResult(
        tool_name=tool_name,
        success=False,
        data={"summary": message, "record_count": 0, "records": []},
        unavailable_data=(message,),
        confidence=0.0,
    )


def _period_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Build a compact period summary from pipeline_runs.

    Inputs: pipeline run row.
    Outputs: summary dictionary.
    Assumptions: run metadata is already trusted storage output.
    """

    return HistoricalPeriod(
        run_id=str(row["run_id"]),
        period=str(row["period"]),
        period_type=str(row["period_type"]),
        completed_at_utc=str(row["completed_at_utc"]),
        status=str(row["status"]),
        confidence=row.get("confidence"),
    ).to_dict()


def _selected_periods(
    repository: MemoryRepository,
    *,
    periods: int,
    before_period: str | None = None,
) -> tuple[str, ...]:
    """Return a bounded historical period window.

    Inputs: repository, period count, and optional before-period.
    Outputs: period identifiers in chronological order.
    Assumptions: absent before-period means latest stored periods.
    """

    rows = _period_rows(repository, limit=periods, before_period=before_period)
    return tuple(str(row["period"]) for row in rows)


def get_previous_period(
    current_period: str,
    *,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve the immediately previous stored period.

    Inputs: current period and optional database path.
    Outputs: structured result with previous period metadata or unavailable status.
    Assumptions: future periods relative to current_period are excluded.
    """

    current = _validate_period(current_period, "current_period")
    repository = _repository(database_path)
    rows = _period_rows(repository, limit=1, before_period=current)
    if not rows:
        return _unavailable("get_previous_period", f"No period found before {current}")
    record = _period_summary(rows[-1])
    return MemoryToolResult(
        "get_previous_period",
        True,
        {"summary": f"Previous period is {record['period']}.", "record": record},
        confidence=0.95,
    )


def get_period_history(
    limit: int,
    before_period: str | None = None,
    *,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve compact metadata for recent historical periods.

    Inputs: limit, optional exclusive before-period, and database path.
    Outputs: period summaries sorted chronologically.
    Assumptions: this returns metadata only, not full period records.
    """

    parsed_limit = _validate_limit(limit, "limit")
    before = _validate_period(before_period, "before_period") if before_period else None
    repository = _repository(database_path)
    rows = _period_rows(repository, limit=parsed_limit, before_period=before)
    records = [_period_summary(row) for row in rows]
    if not records:
        return _unavailable("get_period_history", "No historical periods found")
    return MemoryToolResult(
        "get_period_history",
        True,
        {
            "summary": f"Retrieved {len(records)} historical period(s).",
            "record_count": len(records),
            "records": records,
        },
        confidence=0.95,
    )


def get_metric_history(
    metric: str,
    periods: int,
    department: str | None = None,
    *,
    before_period: str | None = None,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve KPI history for one metric and optional department.

    Inputs: metric name, period count, optional department/before-period/database.
    Outputs: metric points sorted chronologically.
    Assumptions: only stored KPI rows are queried; no recalculation occurs.
    """

    metric_name = _validate_identifier(metric, "metric")
    if metric_name is None:
        raise ValueError("metric is required")
    dept = _validate_text_value(department, "department")
    period_count = _validate_limit(periods, "periods")
    repository = _repository(database_path)
    selected = _selected_periods(repository, periods=period_count, before_period=before_period)
    metric_names = METRIC_ALIASES.get(metric_name, (metric_name,))
    metric_placeholders = ",".join("?" for _ in metric_names)
    where = f"child.metric IN ({metric_placeholders})"
    params: list[object] = list(metric_names)
    if dept is not None:
        where += " AND COALESCE(child.department, '') = ?"
        params.append(dept)
    rows = repository.fetch_rows_for_periods(
        "kpis",
        selected,
        extra_where=where,
        params=tuple(params),
    )
    points = _deduplicate_metric_points(
        [
            HistoricalMetricPoint(
                period=str(row["run_period"]),
                department=row["department"],
                metric=str(row["metric"]),
                value=row["value"],
                unit=row["unit"],
                status=row["status"],
            ).to_dict()
            for row in sorted(
                rows,
                key=lambda item: (
                    period_sort_key(str(item["run_period"])),
                    str(item["updated_at_utc"] or ""),
                ),
            )
        ]
    )
    points = _augment_metric_points_from_finance_artifacts(
        repository,
        selected_periods=selected,
        metric_name=metric_name,
        existing_points=points,
        department=dept,
    )
    if not points:
        return _unavailable(
            "get_metric_history",
            f"No metric history found for {metric_name}",
        )
    return MemoryToolResult(
        "get_metric_history",
        True,
        {
            "summary": f"Retrieved {len(points)} {metric_name} point(s).",
            "metric": metric_name,
            "department": dept,
            "record_count": len(points),
            "records": points,
        },
        confidence=0.95,
    )


def _deduplicate_metric_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one KPI point per period and department.

    Inputs: metric points sorted by period/update time.
    Outputs: chronologically ordered de-duplicated points.
    Assumptions: later rows for the same period/department supersede earlier
    rows in read-only historical views.
    """

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for point in points:
        key = (str(point.get("period") or ""), str(point.get("department") or ""))
        latest[key] = point
    return sorted(latest.values(), key=lambda item: period_sort_key(str(item.get("period") or "")))


def _augment_metric_points_from_finance_artifacts(
    repository: MemoryRepository,
    *,
    selected_periods: tuple[str, ...],
    metric_name: str,
    existing_points: list[dict[str, Any]],
    department: str | None,
) -> list[dict[str, Any]]:
    """Fill missing scalar KPI history from stored finance-summary artifacts.

    Inputs: repository, selected periods, metric name, existing KPI points, and
    optional department filter.
    Outputs: existing points plus artifact-derived points for missing periods.
    Assumptions: artifact references point to processed JSON summaries; this
    does not read raw Excel/PDF files or recalculate any value.
    """

    if department is not None or metric_name not in FINANCE_SUMMARY_METRIC_PATHS:
        return existing_points
    existing_periods = {str(point.get("period") or "") for point in existing_points}
    missing_periods = tuple(period for period in selected_periods if period not in existing_periods)
    if not missing_periods:
        return existing_points
    rows = repository.fetch_rows_for_periods(
        "artifacts",
        missing_periods,
        extra_where="child.artifact_type = ?",
        params=("finance_summary",),
    )
    artifact_points: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            period_sort_key(str(item["run_period"])),
            str(item["updated_at_utc"] or ""),
        ),
    ):
        period = str(row["run_period"])
        if period in existing_periods:
            continue
        value = _value_from_finance_summary_artifact(row["path"], metric_name)
        if value is None:
            continue
        artifact_points.append(
            HistoricalMetricPoint(
                period=period,
                department=None,
                metric=metric_name,
                value=value,
                unit="ratio" if metric_name in {"payroll_percentage_of_revenue", "collection_rate", "student_payment_collection_rate"} else "USD",
                status="available",
            ).to_dict()
        )
        existing_periods.add(period)
    return _deduplicate_metric_points([*existing_points, *artifact_points])


def _value_from_finance_summary_artifact(path_value: Any, metric_name: str) -> float | None:
    """Read one approved scalar value from a processed finance-summary JSON.

    Inputs: artifact path and metric name.
    Outputs: numeric value or None.
    Assumptions: only allowlisted JSON paths are used; malformed artifacts are
    treated as unavailable evidence.
    """

    path = Path(str(path_value))
    if not path.is_absolute():
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    current: Any = payload
    for key in FINANCE_SUMMARY_METRIC_PATHS[metric_name]:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def get_department_history(
    department: str,
    periods: int,
    detail_level: str = "summary",
    *,
    before_period: str | None = None,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve department-scoped memory history.

    Inputs: department, period count, detail level, optional before-period/database.
    Outputs: summary counts or full related records.
    Assumptions: detail_level='full' remains structured and bounded by periods.
    """

    dept = _validate_text_value(department, "department", required=True)
    level = _validate_detail_level(detail_level)
    period_count = _validate_limit(periods, "periods")
    repository = _repository(database_path)
    selected = _selected_periods(repository, periods=period_count, before_period=before_period)
    kpis = [_row_dict(row) for row in repository.fetch_rows_for_periods("kpis", selected, extra_where="child.department = ?", params=(dept,))]
    anomalies = [_row_dict(row) for row in repository.fetch_rows_for_periods("anomalies", selected, extra_where="child.department = ?", params=(dept,))]
    recommendations = [_row_dict(row) for row in repository.fetch_rows_for_periods("recommendations", selected, extra_where="child.department = ?", params=(dept,))]
    records = {"kpis": kpis, "anomalies": anomalies, "recommendations": recommendations}
    total = sum(len(value) for value in records.values())
    if total == 0:
        return _unavailable("get_department_history", f"No history found for {dept}")
    data: dict[str, Any] = {
        "summary": f"Retrieved department history for {dept}.",
        "department": dept,
        "periods": list(selected),
        "counts": {key: len(value) for key, value in records.items()},
    }
    if level == "full":
        data["records"] = records
    return MemoryToolResult("get_department_history", True, data, confidence=0.95)


def get_repeated_anomalies(
    periods: int,
    department: str | None = None,
    min_occurrences: int = 2,
    *,
    before_period: str | None = None,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve anomaly types/metrics recurring across stored periods.

    Inputs: period count, optional department, occurrence threshold, and DB path.
    Outputs: repeated anomaly summaries.
    Assumptions: recurrence groups by department, type, and metric.
    """

    period_count = _validate_limit(periods, "periods")
    minimum = _validate_limit(min_occurrences, "min_occurrences", minimum=2, maximum=24)
    dept = _validate_text_value(department, "department")
    repository = _repository(database_path)
    selected = _selected_periods(repository, periods=period_count, before_period=before_period)
    where = ""
    params: tuple[object, ...] = ()
    if dept is not None:
        where = "child.department = ?"
        params = (dept,)
    rows = [_row_dict(row) for row in repository.fetch_rows_for_periods("anomalies", selected, extra_where=where, params=params)]
    groups: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("department"), row.get("type"), row.get("metric"))
        groups.setdefault(key, []).append(row)
    repeated = []
    for (group_dept, anomaly_type, metric), group_rows in groups.items():
        periods_seen = sorted({str(row["run_period"]) for row in group_rows}, key=period_sort_key)
        if len(periods_seen) >= minimum:
            repeated.append(
                {
                    "department": group_dept,
                    "type": anomaly_type,
                    "metric": metric,
                    "occurrences": len(periods_seen),
                    "periods": periods_seen,
                    "latest_severity": group_rows[-1].get("severity"),
                }
            )
    repeated = sorted(repeated, key=lambda item: (-item["occurrences"], str(item["metric"])))
    if not repeated:
        return _unavailable("get_repeated_anomalies", "No repeated anomalies found")
    return MemoryToolResult(
        "get_repeated_anomalies",
        True,
        {
            "summary": f"Retrieved {len(repeated)} repeated anomaly group(s).",
            "record_count": len(repeated),
            "records": repeated,
        },
        confidence=0.95,
    )


def get_previous_recommendations(
    periods: int,
    department: str | None = None,
    status: str | None = None,
    *,
    before_period: str | None = None,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve previous recommendations with optional filters.

    Inputs: period count, department/status filters, before-period, and DB path.
    Outputs: recommendation rows sorted chronologically.
    Assumptions: recommendations are Python-validated model outputs from stored runs.
    """

    period_count = _validate_limit(periods, "periods")
    dept = _validate_text_value(department, "department")
    status_filter = _validate_text_value(status, "status")
    repository = _repository(database_path)
    selected = _selected_periods(repository, periods=period_count, before_period=before_period)
    predicates: list[str] = []
    params: list[object] = []
    if dept is not None:
        predicates.append("child.department = ?")
        params.append(dept)
    if status_filter is not None:
        predicates.append("child.status = ?")
        params.append(status_filter)
    rows = [
        _row_dict(row)
        for row in repository.fetch_rows_for_periods(
            "recommendations",
            selected,
            extra_where=" AND ".join(predicates),
            params=tuple(params),
        )
    ]
    if not rows:
        return _unavailable("get_previous_recommendations", "No previous recommendations found")
    return MemoryToolResult(
        "get_previous_recommendations",
        True,
        {
            "summary": f"Retrieved {len(rows)} previous recommendation(s).",
            "record_count": len(rows),
            "records": rows,
        },
        confidence=0.95,
    )


def get_goal_progress(
    metric: str | None = None,
    periods: int = 6,
    *,
    before_period: str | None = None,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve stored goal progress rows.

    Inputs: optional metric, period count, before-period, and DB path.
    Outputs: goal records sorted chronologically.
    Assumptions: targets/actuals are stored facts, not recalculated here.
    """

    metric_filter = _validate_identifier(metric, "metric")
    period_count = _validate_limit(periods, "periods")
    repository = _repository(database_path)
    selected = _selected_periods(repository, periods=period_count, before_period=before_period)
    where = "child.metric = ?" if metric_filter else ""
    params = (metric_filter,) if metric_filter else ()
    rows = [_row_dict(row) for row in repository.fetch_rows_for_periods("goals", selected, extra_where=where, params=params)]
    if not rows:
        return _unavailable("get_goal_progress", "No goal progress found")
    return MemoryToolResult(
        "get_goal_progress",
        True,
        {"summary": f"Retrieved {len(rows)} goal progress row(s).", "record_count": len(rows), "records": rows},
        confidence=0.95,
    )


def get_memory_facts(
    category: str | None = None,
    subject: str | None = None,
    periods: int = 6,
    *,
    before_period: str | None = None,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve compact memory facts with optional filters.

    Inputs: category, subject, periods, before-period, and DB path.
    Outputs: compact fact records.
    Assumptions: facts are concise retrieval aids from accepted runs.
    """

    category_filter = _validate_text_value(category, "category")
    subject_filter = _validate_text_value(subject, "subject")
    period_count = _validate_limit(periods, "periods")
    repository = _repository(database_path)
    selected = _selected_periods(repository, periods=period_count, before_period=before_period)
    predicates: list[str] = []
    params: list[object] = []
    if category_filter:
        predicates.append("child.category = ?")
        params.append(category_filter)
    if subject_filter:
        predicates.append("child.subject = ?")
        params.append(subject_filter)
    rows = [
        _row_dict(row)
        for row in repository.fetch_rows_for_periods(
            "memory_facts",
            selected,
            extra_where=" AND ".join(predicates),
            params=tuple(params),
        )
    ]
    if not rows:
        return _unavailable("get_memory_facts", "No memory facts found")
    return MemoryToolResult(
        "get_memory_facts",
        True,
        {"summary": f"Retrieved {len(rows)} memory fact(s).", "record_count": len(rows), "records": rows},
        confidence=0.95,
    )


def get_artifact_references(
    period: str,
    artifact_type: str | None = None,
    *,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve artifact references and checksums for one stored period.

    Inputs: period, optional artifact type, and DB path.
    Outputs: artifact path/checksum records.
    Assumptions: artifact files remain on disk and are not duplicated in SQLite.
    """

    normalized_period = _validate_period(period)
    artifact_filter = _validate_identifier(artifact_type, "artifact_type")
    repository = _repository(database_path)
    where = "child.artifact_type = ?" if artifact_filter else ""
    params = (artifact_filter,) if artifact_filter else ()
    rows = [_row_dict(row) for row in repository.fetch_rows_for_periods("artifacts", (normalized_period,), extra_where=where, params=params)]
    if not rows:
        return _unavailable("get_artifact_references", f"No artifact references found for {normalized_period}")
    return MemoryToolResult(
        "get_artifact_references",
        True,
        {"summary": f"Retrieved {len(rows)} artifact reference(s).", "record_count": len(rows), "records": rows},
        source_references=tuple(str(row.get("path")) for row in rows if row.get("path")),
        confidence=0.95,
    )


def get_full_period_record(
    period: str,
    *,
    database_path: str | Path | None = None,
) -> MemoryToolResult:
    """Retrieve full structured memory record for one period.

    Inputs: period and optional database path.
    Outputs: run metadata plus KPIs, anomalies, recommendations, goals, facts, artifacts.
    Assumptions: full means complete SQLite memory rows, not raw file contents.
    """

    normalized_period = _validate_period(period)
    repository = _repository(database_path)
    run = repository.fetch_period_run(normalized_period)
    if run is None:
        return _unavailable("get_full_period_record", f"No full record found for {normalized_period}")
    data = {"run": _period_summary(_row_dict(run))}
    for table in ("kpis", "anomalies", "recommendations", "goals", "memory_facts", "artifacts"):
        data[table] = [
            _row_dict(row)
            for row in repository.fetch_rows_for_periods(table, (normalized_period,))
        ]
    data["summary"] = f"Retrieved full period record for {normalized_period}."
    return MemoryToolResult(
        "get_full_period_record",
        True,
        data,
        source_references=tuple(str(row.get("path")) for row in data["artifacts"] if row.get("path")),
        confidence=0.95,
    )


def memory_result_to_retrieval_result(result: MemoryToolResult) -> RetrievalResult:
    """Convert a memory tool result into the existing retrieval package shape.

    Inputs: MemoryToolResult.
    Outputs: RetrievalResult.
    Assumptions: this adapter lets Step 8 registry expose tools without using them yet.
    """

    return RetrievalResult(
        retrieval_name=result.tool_name,
        success=result.success,
        data=result.data,
        source_references=result.source_references,
        warnings=result.warnings,
        unavailable_data=result.unavailable_data,
        confidence=result.confidence,
    )


def registry_adapter(tool_name: str, arguments: dict[str, Any]) -> MemoryToolResult:
    """Dispatch a registry-style memory retrieval call.

    Inputs: public tool name and validated-ish argument dictionary.
    Outputs: MemoryToolResult.
    Assumptions: argument validation remains inside each tool function.
    """

    database_path = arguments.get("database_path")
    if tool_name == "get_previous_period":
        return get_previous_period(arguments.get("current_period", ""), database_path=database_path)
    if tool_name == "get_period_history":
        return get_period_history(arguments.get("limit", 6), arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_metric_history":
        return get_metric_history(arguments.get("metric", ""), arguments.get("periods", 6), department=arguments.get("department"), before_period=arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_memory_department_history":
        return get_department_history(arguments.get("department", ""), arguments.get("periods", 6), detail_level=arguments.get("detail_level", "summary"), before_period=arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_repeated_anomalies":
        return get_repeated_anomalies(arguments.get("periods", 6), department=arguments.get("department"), min_occurrences=arguments.get("min_occurrences", 2), before_period=arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_previous_recommendations":
        return get_previous_recommendations(arguments.get("periods", 6), department=arguments.get("department"), status=arguments.get("status"), before_period=arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_goal_progress":
        return get_goal_progress(arguments.get("metric"), arguments.get("periods", 6), before_period=arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_memory_facts":
        return get_memory_facts(arguments.get("category"), arguments.get("subject"), arguments.get("periods", 6), before_period=arguments.get("before_period"), database_path=database_path)
    if tool_name == "get_full_period_record":
        return get_full_period_record(arguments.get("period", ""), database_path=database_path)
    if tool_name == "get_artifact_references":
        return get_artifact_references(arguments.get("period", ""), arguments.get("artifact_type"), database_path=database_path)
    raise ValueError(f"Unknown memory retrieval tool: {tool_name}")


def to_json(data: MemoryToolResult) -> str:
    """Serialize a memory retrieval result for CLIs.

    Inputs: MemoryToolResult.
    Outputs: pretty JSON string.
    Assumptions: helper keeps query_memory.py small.
    """

    return json.dumps(data.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
