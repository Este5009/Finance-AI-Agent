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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
                "department": "Engineering",
                "total_payroll": "300",
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
                "vendor": "Acme",
                "amount": "900",
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
                "department": "Engineering",
                "total_payroll": "300",
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
    assert first["retrieved_evidence"]["data"]["record_count"] == 1
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
