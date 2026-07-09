"""Tests for Step 8 retrieval execution and evidence packaging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from finance_agent.retrieval.retrieval_engine import (
    RetrievalContext,
    build_retrieval_summary,
    execute_retrieval_queue,
    load_retrieval_context,
    retrieve_department_history,
    retrieve_financial_report,
    retrieve_payroll_history,
    retrieve_transactions,
    retrieve_vendor_history,
    save_json_artifact,
)
from finance_agent.retrieval.retrieval_models import RetrievalResult
from finance_agent.retrieval.retrieval_registry import RetrievalRegistry, create_default_registry


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one JSON fixture.

    Inputs: output path and JSON-compatible data.
    Outputs: fixture file on disk.
    Assumptions: parent directories may not exist yet.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one CSV fixture from dictionaries.

    Inputs: output path and ordered row dictionaries.
    Outputs: fixture file on disk.
    Assumptions: at least one row is supplied for field names.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_processed_outputs(root: Path) -> None:
    """Create a minimal processed-output tree for retrieval tests.

    Inputs: temporary project root.
    Outputs: JSON and CSV files matching the local retrieval implementation.
    Assumptions: tests need only the tables directly exercised.
    """

    _write_json(
        root / "outputs" / "calculations" / "finance_summary_june_2026.json",
        {"report_period": "June 2026", "finance_summary": {"total_revenue": 100}},
    )
    _write_json(
        root / "outputs" / "calculations" / "finance_summary_2026.json",
        {"report_period": "2026", "finance_summary": {"total_revenue": 1200}},
    )
    _write_csv(
        root / "outputs" / "calculations" / "monthly_trends_2026.csv",
        [
            {
                "period": "2026-06",
                "month": "June",
                "net_operating_result": "-10",
            }
        ],
    )
    _write_json(
        root
        / "outputs"
        / "intermediate"
        / "financial_document_model_enriched.json",
        {
            "tables": [
                {
                    "table_id": "unknown_table",
                    "final_table_type": "Unknown",
                    "requires_human_review": True,
                    "final_confidence": 0.2,
                }
            ]
        },
    )
    table_dir = root / "outputs" / "intermediate" / "normalized_tables"
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__department_summary__table_01.csv",
        [
            {
                "period": "2026-06-01",
                "department": "Engineering",
                "actual_expenses": "500",
            }
        ],
    )
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__payroll__table_01.csv",
        [
            {
                "period": "2026-06-01",
                "month": "June",
                "department": "Engineering",
                "payroll_budget": "250",
                "total_payroll": "300",
                "variance": "50",
            }
        ],
    )
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__student_payments__table_01.csv",
        [
            {
                "invoice_id": "INV-1",
                "billing_period": "2026-06-01",
                "status": "Overdue - Partial",
                "amount_due": "100",
            }
        ],
    )
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__vendor_payments__table_01.csv",
        [
            {
                "payment_id": "PAY-1",
                "payment_date": "2026-06-01",
                "month": "June",
                "department": "Engineering",
                "vendor": "Acme",
                "amount": "900",
                "high_value_flag": "No",
                "potential_duplicate": "No",
            },
            {
                "payment_id": "PAY-2",
                "payment_date": "2026-06-02",
                "month": "June",
                "department": "Engineering",
                "vendor": "Risk Vendor",
                "amount": "90000",
                "high_value_flag": "Yes",
                "potential_duplicate": "No",
            }
        ],
    )
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__expenses__table_01.csv",
        [
            {
                "period": "2026-06-01",
                "department": "Engineering",
                "expense_category": "Equipment",
                "actual_expense": "700",
            }
        ],
    )
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__budget_vs_actual__table_01.csv",
        [
            {
                "period": "2026-06-01",
                "month": "June",
                "department": "Engineering",
                "budget_expense": "400",
                "actual_expense": "500",
                "expense_variance": "100",
            }
        ],
    )
    _write_csv(
        table_dir / "monthly_financial_report_june_2026__cash_flow__table_01.csv",
        [{"period": "2026-06-01", "net_cash_flow": "-50"}],
    )
    _write_csv(
        table_dir / "annual_financial_report_2026__department_summary__table_01.csv",
        [
            {
                "period": "2026-06-01",
                "department": "Engineering",
                "actual_expenses": "500",
            }
        ],
    )
    _write_csv(
        table_dir / "annual_financial_report_2026__payroll__table_01.csv",
        [
            {
                "period": "2026-06-01",
                "month": "June",
                "department": "Engineering",
                "total_payroll": "300",
                "payroll_budget": "250",
                "variance": "50",
            },
            {
                "period": "2026-05-01",
                "month": "May",
                "department": "Engineering",
                "total_payroll": "280",
                "payroll_budget": "250",
                "variance": "30",
            },
            {
                "period": "2026-12-01",
                "month": "December",
                "department": "Engineering",
                "total_payroll": "999",
                "payroll_budget": "250",
                "variance": "749",
            }
        ],
    )


def _queue() -> dict[str, Any]:
    """Build a small validated-queue fixture.

    Inputs: none.
    Outputs: Step 7-like execution queue.
    Assumptions: queue items use validated public tool names.
    """

    return {
        "queue_id": "QUEUE-JUNE-2026",
        "period_slug": "june_2026",
        "source_plan_id": "OLLAMA-PLAN-JUNE-2026",
        "items": [
            {
                "execution_id": "EXEC-001",
                "step_id": "STEP-001",
                "anomaly_id": "ANOM-001",
                "priority": "high",
                "question": "Which Engineering records exist?",
                "tool_name": "get_department_history",
                "arguments": {"department": "Engineering", "months": 1},
                "expected_output": "Department rows.",
            },
            {
                "execution_id": "EXEC-002",
                "step_id": "STEP-002",
                "anomaly_id": None,
                "priority": "medium",
                "question": "Which overdue invoices exist?",
                "tool_name": "get_transactions",
                "arguments": {
                    "filters": {"status": "overdue", "period": "june_2026"}
                },
                "expected_output": "Overdue rows.",
            },
        ],
    }


def test_default_retrieval_registry_contains_required_interfaces() -> None:
    """Verify Step 8 and Step 7 retrieval names are registered."""

    registry = create_default_registry()

    assert "department_history" in registry.names()
    assert "student_payment_history" in registry.names()
    assert "cashflow_history" in registry.names()
    assert "get_full_report" in registry.names()
    assert "get_transactions" in registry.names()


def test_execution_queue_execution_builds_evidence_packages(tmp_path: Path) -> None:
    """Verify queue items execute sequentially into evidence packages."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)

    package = execute_retrieval_queue(_queue(), context)

    assert package["summary"]["tasks_executed"] == 2
    assert package["summary"]["successful_retrievals"] == 2
    assert len(package["evidence_packages"]) == 2
    first = package["evidence_packages"][0]
    assert first["task_id"] == "STEP-001"
    assert first["retrieved_evidence"]["data"]["record_count"] >= 4
    assert "payroll" in first["retrieved_evidence"]["data"]["source_tables"]
    assert "recommend" not in first["evidence_summary"].casefold()


def test_missing_data_is_packaged_as_unavailable(tmp_path: Path) -> None:
    """Verify absent matching rows do not abort queue execution."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)
    queue = _queue()
    queue["items"][0]["arguments"]["department"] = "Medicine"

    package = execute_retrieval_queue(queue, context)

    assert package["summary"]["tasks_executed"] == 2
    assert package["summary"]["failed_retrievals"] == 1
    assert package["summary"]["unavailable_evidence"] == 1
    assert package["evidence_packages"][0]["retrieved_evidence"]["unavailable_data"]


def test_retrieval_failure_handling_continues_queue(tmp_path: Path) -> None:
    """Verify a registry exception becomes a failed package and later items run."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)
    registry = RetrievalRegistry()

    def failing_retrieval(
        context: RetrievalContext,
        arguments: dict[str, Any],
    ) -> RetrievalResult:
        """Raise an implementation failure for test coverage."""

        raise RuntimeError("backend unavailable")

    registry.register("get_department_history", failing_retrieval, "Failure fixture.")
    registry.register("get_transactions", retrieve_department_history, "Wrong but safe.")

    package = execute_retrieval_queue(_queue(), context, registry)

    assert package["summary"]["tasks_executed"] == 2
    assert package["summary"]["failed_retrievals"] >= 1
    assert "backend unavailable" in package["evidence_packages"][0]["retrieval_warnings"][0]


def test_financial_report_and_summary_schema_are_valid(tmp_path: Path) -> None:
    """Verify full-report retrieval and cross-scope summary schema."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)
    queue = {
        **_queue(),
        "items": [
            {
                **_queue()["items"][0],
                "tool_name": "get_full_report",
                "arguments": {"period": "june_2026"},
            }
        ],
    }

    package = execute_retrieval_queue(queue, context)
    summary = build_retrieval_summary((package,))

    assert package["tools_executed"] is True
    assert package["evidence_packages"][0]["retrieved_evidence"]["success"] is True
    assert summary["summary_id"] == "RETRIEVAL-SUMMARY-2026"
    assert summary["tasks_executed"] == 1


def test_save_json_artifact_writes_output_schema(tmp_path: Path) -> None:
    """Verify retrieval artifacts can be saved as strict JSON."""

    data = {
        "package_id": "EVIDENCE-TEST",
        "period_slug": "test",
        "summary": {"tasks_executed": 0},
        "evidence_packages": [],
    }

    path = save_json_artifact(data, tmp_path / "evidence" / "package.json")

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["package_id"] == "EVIDENCE-TEST"


def test_department_history_retrieves_across_multiple_normalized_tables(
    tmp_path: Path,
) -> None:
    """Verify department evidence includes payroll, expenses, vendors, and budget rows."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)

    result = retrieve_department_history(
        context,
        {"department": "Engineering", "months": 1, "_queue_period_slug": "june_2026"},
    )

    assert result.success is True
    assert result.data["record_count"] >= 4
    assert set(result.data["source_tables"]) >= {
        "budget_vs_actual",
        "department_summary",
        "expenses",
        "payroll",
        "vendor_payments",
    }


def test_payroll_history_includes_department_level_breakdown(tmp_path: Path) -> None:
    """Verify payroll retrieval returns department, amount, budget, and variance."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)

    result = retrieve_payroll_history(
        context,
        {"department": "Engineering", "months": 6, "_queue_period_slug": "june_2026"},
    )

    assert result.success is True
    assert result.data["payroll_breakdown"]
    assert result.data["payroll_breakdown"][-1]["department"] == "Engineering"
    assert result.data["payroll_breakdown"][-1]["payroll_amount"] == "300"
    assert result.data["payroll_breakdown"][-1]["payroll_budget"] == "250"
    assert result.data["payroll_breakdown"][-1]["variance"] == "50"


def test_vendor_placeholder_resolves_to_high_risk_vendor(tmp_path: Path) -> None:
    """Verify placeholder vendor requests map to actual flagged vendor rows."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)

    result = retrieve_vendor_history(
        context,
        {"vendor": "flagged_vendor", "months": 1, "_queue_period_slug": "june_2026"},
    )

    assert result.success is True
    assert result.data["resolved_vendors"] == ["Risk Vendor"]
    assert result.data["records"][0]["vendor"] == "Risk Vendor"


def test_unknown_category_does_not_create_false_unavailable_evidence(
    tmp_path: Path,
) -> None:
    """Verify placeholder category filters return available processed evidence."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)

    result = retrieve_transactions(
        context,
        {
            "filters": {"expense_category": "unknown", "period": "june_2026"},
            "_queue_period_slug": "june_2026",
        },
    )

    assert result.success is True
    assert result.unavailable_data == ()
    assert result.data["record_count"] >= 1


def test_payroll_question_periods_override_latest_annual_month(tmp_path: Path) -> None:
    """Verify merged annual payroll questions retrieve requested months, not December."""

    _make_processed_outputs(tmp_path)
    context = load_retrieval_context(tmp_path)

    result = retrieve_payroll_history(
        context,
        {
            "department": "Engineering",
            "months": 1,
            "_queue_period_slug": "2026",
            "_question": "What payroll components caused risk in 2026-06?",
        },
    )

    assert result.success is True
    periods = {row["period"] for row in result.data["payroll_breakdown"]}
    assert periods == {"2026-06-01"}


def test_generic_period_financial_report_uses_period_slugged_artifact(
    tmp_path: Path,
) -> None:
    """Verify generic report retrieval preserves the period-slugged summary source."""

    _make_processed_outputs(tmp_path)
    loaded = load_retrieval_context(tmp_path)
    context = RetrievalContext(
        project_root=loaded.project_root,
        finance_summary_june=loaded.finance_summary_june,
        finance_summary_annual=loaded.finance_summary_annual,
        monthly_trends=loaded.monthly_trends,
        enriched_model=loaded.enriched_model,
        normalized_table_dir=loaded.normalized_table_dir,
        scope_prefix_by_period={
            "2026_06": "monthly_financial_report_june_2026",
        },
        finance_summary_by_period={
            "2026_06": {
                "report_period": "2026-06",
                "finance_summary": {"total_revenue": 100},
            }
        },
        finance_summary_source_by_period={
            "2026_06": "outputs/calculations/finance_summary_2026_06.json",
        },
    )

    result = retrieve_financial_report(
        context,
        {"period": "2026-06", "_queue_period_slug": "2026_06"},
    )

    assert result.success is True
    assert result.data["report"]["report_period"] == "2026-06"
    assert result.source_references == (
        "outputs/calculations/finance_summary_2026_06.json",
    )
