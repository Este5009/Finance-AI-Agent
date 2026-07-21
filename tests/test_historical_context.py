"""Tests for Phase 13 historical reasoning context integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finance_agent.agent.investigation_planner import build_investigation_plan
from finance_agent.agent.ollama_planner import build_ollama_planner_prompt
from finance_agent.analysis.strategic_analysis import build_strategic_analysis_prompt
from finance_agent.memory.context_builder import (
    HistoricalContextCache,
    build_historical_context,
)
from finance_agent.memory.retrieval_models import MemoryToolResult


def _finance_summary() -> dict[str, Any]:
    """Return a current finance summary with payroll, collection, and cash issues.

    Inputs: none.
    Outputs: finance summary dictionary.
    Assumptions: values are already calculated by Python.
    """

    return {
        "report_period": "2026-12",
        "finance_summary": {
            "payroll_percentage_of_revenue": 0.40,
            "student_payments": {"collection_rate": 0.96},
            "cash_flow": {"net_cash_flow": 450_000},
        },
        "kpi_summary": [
            {"metric": "payroll_percentage_of_revenue", "value": 0.40, "unit": "ratio"},
            {"metric": "student_payment_collection_rate", "value": 0.96, "unit": "ratio"},
        ],
    }


def _anomaly_report() -> dict[str, Any]:
    """Return current anomalies that should trigger relevant history calls.

    Inputs: none.
    Outputs: anomaly report dictionary.
    Assumptions: anomaly IDs are deterministic test values.
    """

    return {
        "report_period": "2026-12",
        "total_anomalies": 2,
        "anomalies_by_severity": {"high": 2},
        "anomalies": [
            {
                "anomaly_id": "A1",
                "title": "Health Sciences payroll trend remains above target",
                "metric": "payroll_percentage_of_revenue",
                "severity": "high",
                "evidence": "Health Sciences overtime remains relevant.",
                "rule_id": "PAYROLL_RATIO_MAX",
            },
            {
                "anomaly_id": "A2",
                "title": "Recurring vendor anomaly",
                "metric": "maximum_vendor_payment",
                "severity": "high",
                "evidence": "MedSupply vendor anomaly recurred.",
                "rule_id": "VENDOR_PAYMENT_REVIEW",
            },
        ],
    }


def _result(tool_name: str, records: list[dict[str, Any]] | None = None, success: bool = True) -> MemoryToolResult:
    """Build one fake memory retrieval result.

    Inputs: tool name, records, and success flag.
    Outputs: MemoryToolResult.
    Assumptions: tests need compact result shapes only.
    """

    if not success:
        return MemoryToolResult(
            tool_name,
            False,
            {"summary": "No history", "record_count": 0, "records": []},
            unavailable_data=("No history",),
            confidence=0.0,
        )
    rows = records or [{"period": "2026_11", "metric": "payroll_percentage_of_revenue", "value": 0.41}]
    return MemoryToolResult(
        tool_name,
        True,
        {"summary": f"{tool_name} ok", "record_count": len(rows), "records": rows},
        confidence=0.95,
    )


def _fake_retrievers(calls: list[str]) -> dict[str, Any]:
    """Return fake retrievers that record deterministic call order.

    Inputs: mutable call log.
    Outputs: retriever mapping.
    Assumptions: fake retrievers ignore database state.
    """

    def metric(**kwargs: Any) -> MemoryToolResult:
        """Return fake metric history."""

        calls.append(f"get_metric_history:{kwargs.get('metric')}")
        return _result("get_metric_history", [{"period": "2026_10", "metric": kwargs.get("metric"), "value": 0.43}])

    def department(**kwargs: Any) -> MemoryToolResult:
        """Return fake department history."""

        calls.append(f"get_department_history:{kwargs.get('department')}")
        return _result("get_department_history", [{"period": "2026_11", "department": kwargs.get("department")}])

    def repeated(**kwargs: Any) -> MemoryToolResult:
        """Return fake repeated anomaly history."""

        calls.append("get_repeated_anomalies")
        return _result("get_repeated_anomalies", [{"metric": "maximum_vendor_payment", "periods": ["2026_07", "2026_08"]}])

    def previous_recommendations(**kwargs: Any) -> MemoryToolResult:
        """Return fake previous recommendation history."""

        calls.append("get_previous_recommendations")
        return _result("get_previous_recommendations", [{"period": "2026_05", "action": "Reduce overtime"}])

    def goals(**kwargs: Any) -> MemoryToolResult:
        """Return fake goal progress history."""

        calls.append("get_goal_progress")
        return _result("get_goal_progress", [{"period": "2026_11", "metric": "payroll_percentage_of_revenue"}])

    def facts(**kwargs: Any) -> MemoryToolResult:
        """Return fake memory facts."""

        calls.append("get_memory_facts")
        return _result("get_memory_facts", [{"period": "2026_11", "fact": "Payroll stabilized"}])

    return {
        "get_metric_history": metric,
        "get_department_history": department,
        "get_repeated_anomalies": repeated,
        "get_previous_recommendations": previous_recommendations,
        "get_goal_progress": goals,
        "get_memory_facts": facts,
    }


def test_context_builder_selects_relevant_history_and_order() -> None:
    """Verify builder selects relevant compact history in deterministic order."""

    calls: list[str] = []
    result = build_historical_context(
        current_period="2026_12",
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        database_path=Path("memory.db"),
        retrievers=_fake_retrievers(calls),
    )

    assert calls[:3] == [
        "get_metric_history:payroll_percentage_of_revenue",
        "get_metric_history:student_payment_collection_rate",
        "get_metric_history:net_cash_flow",
    ]
    assert "get_department_history:Health Sciences" in calls
    assert result.context["summary"]["available_retrievals"] == len(calls)
    assert result.telemetry["database_queries"] == len(calls)
    assert result.context["history_policy"]["no_full_reports"] is True


def test_context_caching_and_retrieval_deduplication() -> None:
    """Verify repeated builder calls reuse cached retrievals."""

    calls: list[str] = []
    cache = HistoricalContextCache()
    kwargs = {
        "current_period": "2026_12",
        "finance_summary": _finance_summary(),
        "anomaly_report": _anomaly_report(),
        "database_path": Path("memory.db"),
        "retrievers": _fake_retrievers(calls),
        "cache": cache,
    }

    first = build_historical_context(**kwargs)
    second = build_historical_context(**kwargs)

    assert first.telemetry["database_queries"] > 0
    assert second.telemetry["database_queries"] == 0
    assert second.telemetry["cache_hits"] == first.telemetry["planned_retrievals"]
    assert len(calls) == first.telemetry["planned_retrievals"]


def test_empty_history_fallback_is_explicit() -> None:
    """Verify missing historical data produces explicit unavailable context."""

    calls: list[str] = []

    def unavailable(**kwargs: Any) -> MemoryToolResult:
        """Return unavailable history for every requested tool."""

        calls.append("called")
        return _result("tool", success=False)

    result = build_historical_context(
        current_period="2026_12",
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        retrievers={name: unavailable for name in _fake_retrievers([])},
    )

    assert calls
    assert result.context["summary"]["available_retrievals"] == 0
    assert result.telemetry["historical_context_available"] is False
    assert all(item["unavailable_data"] for item in result.context["retrievals"])


def test_planner_prompt_includes_historical_context() -> None:
    """Verify planner prompt receives historical context alongside current state."""

    baseline = build_investigation_plan(
        finance_document=_finance_summary(),
        anomaly_report=_anomaly_report(),
        monthly_trends=[],
        recurrence_anomalies=_anomaly_report()["anomalies"],
        enriched_model={"tables": []},
        risk_summary={},
        period_slug="2026_12",
        source_files=("report.xlsx",),
    )

    prompt = build_ollama_planner_prompt(
        finance_document=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary={},
        enriched_model={"tables": []},
        baseline_plan=baseline,
        period_slug="2026_12",
        historical_context={"summary": {"available_retrievals": 3}, "retrievals": []},
    )

    assert "historical_context" in prompt
    assert "available_retrievals" in prompt


def test_strategic_analysis_prompt_includes_historical_context() -> None:
    """Verify strategic analysis prompt includes historical comparison instructions."""

    prompt = build_strategic_analysis_prompt(
        evidence_package={"evidence_packages": []},
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary={},
        period_slug="2026_12",
        historical_context={
            "retrievals": [
                {"tool_name": "get_metric_history", "records": [{"period": "2026_11", "value": 0.41}]}
            ]
        },
    )

    payload = prompt.split("STRATEGIC_ANALYSIS_CONTEXT:\n", 1)[1].split("\n\nINSTRUCTIONS", 1)[0]
    assert json.loads(payload)["historical_context"]["retrievals"][0]["tool_name"] == "get_metric_history"
    assert "historical_context" in prompt
