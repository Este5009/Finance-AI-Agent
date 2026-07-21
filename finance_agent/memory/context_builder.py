"""Build compact historical context for planner and strategic analysis stages."""

from __future__ import annotations

import json
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from finance_agent.memory.retrieval import (
    get_department_history,
    get_goal_progress,
    get_memory_facts,
    get_metric_history,
    get_previous_recommendations,
    get_repeated_anomalies,
    period_sort_key,
)
from finance_agent.memory.repository import MemoryRepository
from finance_agent.memory.retrieval_models import MemoryToolResult


MemoryRetriever = Callable[..., MemoryToolResult]


@dataclass
class HistoricalContextCache:
    """Small in-run cache for deterministic historical retrieval calls.

    Inputs:
        values: Optional preloaded cache dictionary.
    Outputs:
        Mutable cache object used by the context builder.
    Assumptions:
        Cache scope is one pipeline run; it is not persisted to SQLite.
    """

    values: dict[str, MemoryToolResult] = field(default_factory=dict)

    def get_or_call(self, key: str, callback: Callable[[], MemoryToolResult]) -> tuple[MemoryToolResult, bool]:
        """Return a cached value or execute the retrieval callback.

        Inputs:
            key: Stable call key.
            callback: Function that performs the read-only retrieval call.
        Outputs:
            Result plus a boolean indicating whether the value came from cache.
        Assumptions:
            Retrieval functions are deterministic for a fixed database state.
        """

        if key in self.values:
            return self.values[key], True
        result = callback()
        self.values[key] = result
        return result, False


@dataclass(frozen=True)
class HistoricalContextResult:
    """Historical context plus retrieval telemetry.

    Inputs:
        context: Compact JSON-compatible context.
        telemetry: Query count, cache hits, latency, and context size.
    Outputs:
        Structured result consumed by planner/analysis integrations.
    Assumptions:
        ``context`` contains summaries, not full historical reports.
    """

    context: dict[str, Any]
    telemetry: dict[str, Any]


def build_historical_context(
    *,
    current_period: str,
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any] | None = None,
    database_path: str | Path | None = None,
    purpose: str = "planner",
    periods: int = 12,
    cache: HistoricalContextCache | None = None,
    retrievers: dict[str, MemoryRetriever] | None = None,
) -> HistoricalContextResult:
    """Build relevant compact historical context for one current report.

    Inputs:
        current_period: Current period slug; historical queries exclude this period.
        finance_summary: Current processed finance summary.
        anomaly_report: Current processed anomaly report.
        evidence_package: Optional current evidence package for strategic analysis.
        database_path: SQLite memory database path.
        purpose: ``planner`` or ``strategic_analysis``.
        periods: Maximum historical periods to inspect.
        cache: Optional in-run retrieval cache.
        retrievers: Optional retrieval overrides for tests.
    Outputs:
        HistoricalContextResult with compact context and telemetry.
    Assumptions:
        Only relevant bounded memory retrieval tools are called; full reports are
        never loaded or sent to Ollama.
    """

    started = time.perf_counter()
    active_cache = cache or HistoricalContextCache()
    active_retrievers = _retrievers(retrievers)
    signals = _detect_current_signals(finance_summary, anomaly_report, evidence_package)
    calls = _planned_calls(
        signals,
        current_period=current_period,
        periods=periods,
        database_path=database_path,
    )

    results: list[dict[str, Any]] = []
    query_count = 0
    cache_hits = 0
    for call in calls:
        key = _call_key(call)

        def callback(call: dict[str, Any] = call) -> MemoryToolResult:
            """Execute one planned retrieval call.

            Inputs: planned call captured from the loop.
            Outputs: MemoryToolResult.
            Assumptions: retriever names were selected from a local allowlist.
            """

            return active_retrievers[call["tool_name"]](**call["arguments"])

        result, hit = active_cache.get_or_call(key, callback)
        if hit:
            cache_hits += 1
        else:
            query_count += 1
        results.append(_compact_result(call, result))

    artifact_patterns = _load_artifact_anomaly_patterns(
        database_path,
        current_period=current_period,
        periods=periods,
    )
    context = _assemble_context(
        current_period=current_period,
        purpose=purpose,
        signals=signals,
        retrieval_results=results,
        artifact_patterns=artifact_patterns,
    )
    telemetry = {
        "historical_context_available": any(item["success"] for item in results),
        "planned_retrievals": len(calls),
        "database_queries": query_count,
        "cache_hits": cache_hits,
        "artifact_reference_reads": artifact_patterns.get("artifact_reference_reads", 0),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "context_characters": len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))),
    }
    return HistoricalContextResult(context=context, telemetry=telemetry)


def _retrievers(overrides: dict[str, MemoryRetriever] | None) -> dict[str, MemoryRetriever]:
    """Return retrieval function mapping with optional test overrides.

    Inputs:
        overrides: Optional custom retriever functions.
    Outputs:
        Tool-name to callable mapping.
    Assumptions:
        Defaults are read-only SQLite memory tools.
    """

    defaults: dict[str, MemoryRetriever] = {
        "get_metric_history": get_metric_history,
        "get_department_history": get_department_history,
        "get_repeated_anomalies": get_repeated_anomalies,
        "get_previous_recommendations": get_previous_recommendations,
        "get_goal_progress": get_goal_progress,
        "get_memory_facts": get_memory_facts,
    }
    if overrides:
        defaults.update(overrides)
    return defaults


def _finance_metrics(finance_summary: dict[str, Any]) -> dict[str, Any]:
    """Extract current scalar finance metrics from processed summary.

    Inputs:
        finance_summary: Current finance-summary document.
    Outputs:
        Metric dictionary.
    Assumptions:
        Calculated values may be nested under ``finance_summary``.
    """

    finance = finance_summary.get("finance_summary", finance_summary)
    finance = finance if isinstance(finance, dict) else {}
    payments = finance.get("student_payments", {})
    payments = payments if isinstance(payments, dict) else {}
    cash = finance.get("cash_flow", {})
    cash = cash if isinstance(cash, dict) else {}
    return {
        "payroll_percentage_of_revenue": finance.get("payroll_percentage_of_revenue"),
        "student_payment_collection_rate": payments.get("collection_rate")
        or finance.get("student_payment_collection_rate"),
        "net_cash_flow": cash.get("net_cash_flow") or finance.get("net_cash_flow"),
    }


def _detect_current_signals(
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any] | None,
) -> dict[str, Any]:
    """Detect which historical retrievals are relevant to current state.

    Inputs:
        finance_summary: Current finance summary.
        anomaly_report: Current anomaly report.
        evidence_package: Optional evidence package.
    Outputs:
        Deterministic signal dictionary used to plan retrievals.
    Assumptions:
        Signals are conservative; irrelevant history is not queried.
    """

    metrics = _finance_metrics(finance_summary)
    anomalies = anomaly_report.get("anomalies", [])
    anomalies = anomalies if isinstance(anomalies, list) else []
    text = " ".join(
        " ".join(str(item.get(key) or "") for key in ("title", "description", "metric", "rule_id", "evidence"))
        for item in anomalies
        if isinstance(item, dict)
    ).casefold()
    departments = sorted(
        {
            department
            for department in _departments_from_anomalies(anomalies)
            if department
        }
    )
    evidence_text = json.dumps(evidence_package or {}, ensure_ascii=False).casefold()[:10_000]
    return {
        "current_metrics": metrics,
        "anomaly_count": len(anomalies),
        "payroll_anomaly": "payroll" in text or _as_float(metrics.get("payroll_percentage_of_revenue")) > 0.42,
        "collection_anomaly": "collection" in text or "tuition" in text or _as_float(metrics.get("student_payment_collection_rate")) < 0.93,
        "cashflow_anomaly": "cash flow" in text or "cash_flow" in text or _as_float(metrics.get("net_cash_flow")) < 0,
        "vendor_anomaly": "vendor" in text or "medsupply" in text or "duplicate" in text,
        "department_overspending": "department" in text or "overspend" in text or bool(departments),
        "goal_deviation": any(term in text for term in ("target", "goal", "threshold", "below", "exceeds")),
        "recommendation_relevant": bool(anomalies) or "recommendation" in evidence_text,
        "departments": departments[:4],
    }


def _departments_from_anomalies(anomalies: list[Any]) -> set[str]:
    """Infer departments referenced by current anomaly evidence.

    Inputs:
        anomalies: Current anomaly dictionaries.
    Outputs:
        Department names.
    Assumptions:
        Department may appear in explicit fields or title/evidence text.
    """

    known = {
        "Health Sciences",
        "Engineering",
        "Business",
        "Arts & Humanities",
        "Student Services",
        "Administration",
    }
    found: set[str] = set()
    for item in anomalies:
        if not isinstance(item, dict):
            continue
        explicit = item.get("department")
        if isinstance(explicit, str) and explicit.strip():
            found.add(explicit.strip())
        text = " ".join(str(item.get(key) or "") for key in ("title", "description", "evidence"))
        lowered = text.casefold()
        for department in known:
            if department.casefold() in lowered:
                found.add(department)
    return found


def _planned_calls(
    signals: dict[str, Any],
    *,
    current_period: str,
    periods: int,
    database_path: str | Path | None,
) -> list[dict[str, Any]]:
    """Plan relevant historical retrieval calls.

    Inputs:
        signals: Current-state signal dictionary.
        current_period: Current period slug.
        periods: Historical window size.
        database_path: SQLite memory database path.
    Outputs:
        Ordered unique retrieval call descriptors.
    Assumptions:
        Retrieval order is deterministic for stable prompt context.
    """

    database_arg = {"before_period": current_period, "database_path": database_path}
    calls: list[dict[str, Any]] = []
    current_metrics = signals.get("current_metrics", {})
    if signals["payroll_anomaly"] or current_metrics.get("payroll_percentage_of_revenue") is not None:
        calls.append(_call("get_metric_history", metric="payroll_percentage_of_revenue", periods=periods, **database_arg))
    if signals["collection_anomaly"] or current_metrics.get("student_payment_collection_rate") is not None:
        calls.append(_call("get_metric_history", metric="student_payment_collection_rate", periods=periods, **database_arg))
    if signals["cashflow_anomaly"] or current_metrics.get("net_cash_flow") is not None:
        calls.append(_call("get_metric_history", metric="net_cash_flow", periods=periods, **database_arg))
    if signals["department_overspending"]:
        for department in signals.get("departments", []) or ["Health Sciences"]:
            calls.append(_call("get_department_history", department=department, periods=periods, detail_level="summary", **database_arg))
    calls.append(_call("get_repeated_anomalies", periods=periods, min_occurrences=2, **database_arg))
    calls.append(_call("get_previous_recommendations", periods=periods, **database_arg))
    calls.append(_call("get_goal_progress", periods=periods, **database_arg))
    # Compact facts are useful for longitudinal context but bounded and category-free.
    calls.append(_call("get_memory_facts", periods=periods, **database_arg))
    return _deduplicate_calls(calls)


def _call(tool_name: str, **arguments: Any) -> dict[str, Any]:
    """Create a planned retrieval call descriptor.

    Inputs:
        tool_name: Memory retrieval tool name.
        arguments: Tool arguments.
    Outputs:
        Call descriptor.
    Assumptions:
        ``database_path`` is filled in by ``_compact_result`` caller before execution.
    """

    return {"tool_name": tool_name, "arguments": arguments}


def _deduplicate_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate retrieval calls while preserving order.

    Inputs:
        calls: Planned calls.
    Outputs:
        Unique planned calls.
    Assumptions:
        Equivalent calls have identical JSON-serialized tool/arguments.
    """

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for call in calls:
        key = _call_key(call)
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def _call_key(call: dict[str, Any]) -> str:
    """Build a stable cache/deduplication key for a call.

    Inputs:
        call: Planned retrieval call descriptor.
    Outputs:
        Stable JSON key.
    Assumptions:
        Arguments are JSON-compatible.
    """

    return json.dumps(call, sort_keys=True, default=str, separators=(",", ":"))


def _compact_result(call: dict[str, Any], result: MemoryToolResult) -> dict[str, Any]:
    """Compact one historical retrieval result for LLM context.

    Inputs:
        call: Planned call descriptor.
        result: Memory retrieval result.
    Outputs:
        Bounded result dictionary.
    Assumptions:
        Full historical artifact access remains available through source references.
    """

    data = result.data if isinstance(result.data, dict) else {}
    records = data.get("records", [])
    if isinstance(records, list):
        compact_records = records[-8:]
    elif isinstance(records, dict):
        compact_records = {key: value[-8:] if isinstance(value, list) else value for key, value in records.items()}
    else:
        compact_records = []
    return {
        "tool_name": call["tool_name"],
        "arguments": {
            key: value
            for key, value in call["arguments"].items()
            if key != "database_path" and value is not None
        },
        "success": result.success,
        "summary": data.get("summary"),
        "record_count": data.get("record_count") or data.get("counts"),
        "metric": data.get("metric"),
        "department": data.get("department"),
        "records": compact_records,
        "unavailable_data": list(result.unavailable_data),
        "warnings": list(result.warnings),
        "confidence": result.confidence,
    }


def _assemble_context(
    *,
    current_period: str,
    purpose: str,
    signals: dict[str, Any],
    retrieval_results: list[dict[str, Any]],
    artifact_patterns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble compact historical context from retrieval results.

    Inputs:
        current period, purpose, detected signals, and compact retrieval results.
    Outputs:
        JSON-compatible context.
    Assumptions:
        Results are already bounded and sorted by deterministic call order.
    """

    available = [item for item in retrieval_results if item["success"]]
    derived = _derive_context_from_retrievals(retrieval_results)
    if artifact_patterns:
        derived["artifact_anomaly_patterns"] = artifact_patterns.get("patterns", [])
    return {
        "current_period": current_period,
        "purpose": purpose,
        "history_policy": {
            "compact_only": True,
            "excludes_current_period": True,
            "no_full_reports": True,
        },
        "detected_signals": signals,
        "retrievals": retrieval_results,
        "derived_context": derived,
        "summary": {
            "available_retrievals": len(available),
            "unavailable_retrievals": len(retrieval_results) - len(available),
            "topics": sorted(
                {
                    item["tool_name"]
                    for item in available
                }
            ),
        },
    }


def _derive_context_from_retrievals(retrieval_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive compact trend and follow-up summaries from retrieval results.

    Inputs:
        retrieval_results: Compact memory retrieval outputs.
    Outputs:
        Derived context dictionary.
    Assumptions:
        Derivations summarize stored facts; they do not recalculate finance outputs.
    """

    metric_trends: dict[str, dict[str, Any]] = {}
    previous_recommendations: list[dict[str, Any]] = []
    for item in retrieval_results:
        if item.get("tool_name") == "get_metric_history" and item.get("success"):
            records = item.get("records", [])
            records = records if isinstance(records, list) else []
            metric = item.get("metric") or (records[0].get("metric") if records and isinstance(records[0], dict) else None)
            values = [
                float(record["value"])
                for record in records
                if isinstance(record, dict) and record.get("value") is not None
            ]
            if metric and values:
                metric_trends[str(metric)] = {
                    "periods": [record.get("period") for record in records if isinstance(record, dict)],
                    "first_value": values[0],
                    "latest_value": values[-1],
                    "direction": _trend_direction(values),
                }
        if item.get("tool_name") == "get_previous_recommendations" and item.get("success"):
            records = item.get("records", [])
            if isinstance(records, list):
                previous_recommendations.extend(record for record in records if isinstance(record, dict))
    goal_progress_proxy = _goal_progress_from_trends(metric_trends)
    return {
        "kpi_trends": metric_trends,
        "goal_progress": goal_progress_proxy,
        "previous_recommendation_count": len(previous_recommendations),
        "recommendation_effectiveness": _recommendation_effectiveness(metric_trends, previous_recommendations),
    }


def _trend_direction(values: list[float]) -> str:
    """Return simple deterministic direction for a metric series.

    Inputs:
        values: Ordered metric values.
    Outputs:
        ``improving``, ``worsening``, or ``stable``.
    Assumptions:
        Direction is qualitative context, not a replacement for calculations.
    """

    if len(values) < 2:
        return "stable"
    if values[-1] > values[0]:
        return "improving"
    if values[-1] < values[0]:
        return "improving" if max(values) == values[0] else "worsening"
    return "stable"


def _goal_progress_from_trends(metric_trends: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build deterministic goal-progress proxies from KPI trends.

    Inputs:
        metric_trends: Metric trend summaries.
    Outputs:
        Goal-progress proxy records.
    Assumptions:
        Targets reflect existing project thresholds used by anomaly rules.
    """

    targets = {
        "payroll_percentage_of_revenue": {"target": 0.42, "direction": "at_or_below"},
        "student_payment_collection_rate": {"target": 0.94, "direction": "at_or_above"},
        "net_cash_flow": {"target": 0.0, "direction": "at_or_above"},
    }
    rows: list[dict[str, Any]] = []
    for metric, trend in metric_trends.items():
        target = targets.get(metric)
        latest = trend.get("latest_value")
        if not target or latest is None:
            continue
        achieved = latest <= target["target"] if target["direction"] == "at_or_below" else latest >= target["target"]
        rows.append(
            {
                "metric": metric,
                "target": target["target"],
                "latest_value": latest,
                "status": "met" if achieved else "not_met",
                "trend_direction": trend.get("direction"),
            }
        )
    return rows


def _recommendation_effectiveness(
    metric_trends: dict[str, dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link prior recommendations to later KPI movement.

    Inputs:
        metric_trends: Derived KPI trend summaries.
        recommendations: Previous recommendation rows.
    Outputs:
        Compact effectiveness evidence.
    Assumptions:
        This is evidence packaging only; the LLM draws the final conclusion.
    """

    evidence: list[dict[str, Any]] = []
    text = " ".join(str(item.get("action", "")) for item in recommendations).casefold()
    if "payroll" in text or "overtime" in text:
        evidence.append(
            {
                "topic": "payroll_overtime",
                "recommendation_count": sum(1 for item in recommendations if "payroll" in str(item.get("action", "")).casefold() or "overtime" in str(item.get("action", "")).casefold()),
                "related_trend": metric_trends.get("payroll_percentage_of_revenue"),
            }
        )
    if "collection" in text or "cobranza" in text or "payment" in text:
        evidence.append(
            {
                "topic": "collections",
                "recommendation_count": sum(1 for item in recommendations if any(term in str(item.get("action", "")).casefold() for term in ("collection", "payment", "cobranza"))),
                "related_trend": metric_trends.get("student_payment_collection_rate"),
            }
        )
    if "vendor" in text:
        evidence.append(
            {
                "topic": "vendor_controls",
                "recommendation_count": sum(1 for item in recommendations if "vendor" in str(item.get("action", "")).casefold()),
                "related_trend": None,
            }
        )
    return evidence


def _load_artifact_anomaly_patterns(
    database_path: str | Path | None,
    *,
    current_period: str,
    periods: int,
) -> dict[str, Any]:
    """Load compact repeated anomaly patterns from stored processed artifacts.

    Inputs:
        database_path: Memory database path.
        current_period: Period to exclude.
        periods: Historical window.
    Outputs:
        Compact repeated anomaly patterns plus read count.
    Assumptions:
        Only normalized Anomalies_Embedded CSV artifacts are read; full reports are not.
    """

    if database_path is None:
        return {"patterns": [], "artifact_reference_reads": 0}
    if not Path(database_path).is_file():
        return {"patterns": [], "artifact_reference_reads": 0}
    repository = MemoryRepository(database_path)
    period_rows = [dict(row) for row in repository.fetch_periods()]
    selected = [
        str(row["period"])
        for row in sorted(period_rows, key=lambda row: period_sort_key(str(row["period"])))
        if period_sort_key(str(row["period"])) < period_sort_key(current_period)
    ][-periods:]
    artifact_rows = repository.fetch_rows_for_periods("artifacts", tuple(selected), extra_where="child.artifact_type = ?", params=("normalized_table",))
    groups: dict[tuple[str, str], set[str]] = {}
    reads = 0
    for row in artifact_rows:
        path = Path(str(row["path"]))
        if "anomalies_embedded" not in path.name.casefold() or not path.is_file():
            continue
        reads += 1
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                department = str(item.get("department") or item.get("Department") or "")
                anomaly_type = str(item.get("anomaly_type") or item.get("Anomaly_Type") or "")
                period = str(item.get("detected_period") or item.get("Detected_Period") or row["run_period"])
                if department or anomaly_type:
                    groups.setdefault((department, anomaly_type), set()).add(period)
    patterns = [
        {"department": department, "anomaly_type": anomaly_type, "periods": sorted(periods_seen, key=period_sort_key), "occurrences": len(periods_seen)}
        for (department, anomaly_type), periods_seen in groups.items()
        if len(periods_seen) >= 2
    ]
    patterns = sorted(patterns, key=lambda item: (-item["occurrences"], item["department"], item["anomaly_type"]))
    return {"patterns": patterns[:8], "artifact_reference_reads": reads}


def _as_float(value: Any) -> float:
    """Convert a scalar to float with a safe default.

    Inputs:
        value: Any scalar.
    Outputs:
        Float value or zero.
    Assumptions:
        Failed conversion should not trigger a false positive by itself.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def save_historical_context(context: dict[str, Any], output_path: str | Path) -> Path:
    """Save a historical context artifact for audit and validation.

    Inputs:
        context: JSON-compatible historical context.
        output_path: Target JSON path.
    Outputs:
        Resolved path written.
    Assumptions:
        Parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
