"""Execute validated investigation queues and build evidence packages."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finance_agent.retrieval_models import (
    EvidencePackage,
    RetrievalRequest,
    RetrievalResult,
    RetrievalRunSummary,
)
from finance_agent.retrieval_registry import RetrievalRegistry, create_default_registry


class RetrievalInputError(RuntimeError):
    """Raised when retrieval inputs cannot be loaded from processed outputs."""


@dataclass(frozen=True)
class RetrievalContext:
    """Local processed-output context used by retrieval functions.

    Inputs: project root and already parsed processed artifacts.
    Outputs: helper methods for local retrieval implementations.
    Assumptions: future database/API contexts can keep the same retrieval signatures.
    """

    project_root: Path
    finance_summary_june: dict[str, Any]
    finance_summary_annual: dict[str, Any]
    monthly_trends: tuple[dict[str, Any], ...]
    enriched_model: dict[str, Any]
    normalized_table_dir: Path

    def normalized_records(self, table_type: str, period_slug: str) -> tuple[dict[str, Any], ...]:
        """Read normalized processed rows for one table type and reporting scope.

        Inputs: normalized table type and period slug.
        Outputs: row dictionaries from the matching processed CSV.
        Assumptions: CSV names retain workbook scope and table type.
        """

        prefix = _scope_prefix(period_slug)
        return _read_matching_normalized_csv(
            self.normalized_table_dir,
            prefix,
            table_type,
        )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON object from disk.

    Inputs: JSON artifact path.
    Outputs: parsed dictionary.
    Assumptions: retrieval consumes processed outputs only.
    """

    if not path.is_file():
        raise RetrievalInputError(f"Required retrieval input does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalInputError(f"Could not read retrieval input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RetrievalInputError(f"Retrieval JSON root must be an object: {path}")
    return value


def _read_csv_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Read one processed CSV into row dictionaries.

    Inputs: CSV artifact path.
    Outputs: ordered tuple of rows.
    Assumptions: values remain strings unless specific filters cast them.
    """

    if not path.is_file():
        raise RetrievalInputError(f"Required retrieval input does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise RetrievalInputError(f"Could not read retrieval input {path}: {exc}") from exc


def _read_matching_normalized_csv(
    table_dir: Path,
    scope_prefix: str,
    table_type: str,
) -> tuple[dict[str, Any], ...]:
    """Read the normalized CSV matching a scope and table type.

    Inputs: normalized-table directory, scope prefix, and table type.
    Outputs: processed row dictionaries.
    Assumptions: one normalized CSV exists per detected table in current fixtures.
    """

    pattern = f"{scope_prefix}__{table_type}__*.csv"
    matches = sorted(table_dir.glob(pattern))
    if not matches:
        raise RetrievalInputError(
            f"No normalized table found for {scope_prefix}/{table_type}"
        )
    return _read_csv_records(matches[0])


def load_retrieval_context(project_root: str | Path) -> RetrievalContext:
    """Load processed artifacts required by the local retrieval implementation.

    Inputs: project root containing outputs/calculations and outputs/intermediate.
    Outputs: RetrievalContext for registry functions.
    Assumptions: this loader never opens raw Excel, PDF, or DOCX files.
    """

    root = Path(project_root).resolve()
    return RetrievalContext(
        project_root=root,
        finance_summary_june=_read_json(
            root / "outputs" / "calculations" / "finance_summary_june_2026.json"
        ),
        finance_summary_annual=_read_json(
            root / "outputs" / "calculations" / "finance_summary_2026.json"
        ),
        monthly_trends=_read_csv_records(
            root / "outputs" / "calculations" / "monthly_trends_2026.csv"
        ),
        enriched_model=_read_json(
            root
            / "outputs"
            / "intermediate"
            / "financial_document_model_enriched.json"
        ),
        normalized_table_dir=(
            root / "outputs" / "intermediate" / "normalized_tables"
        ),
    )


def _scope_prefix(period_slug: str) -> str:
    """Map a report period slug to the normalized table filename prefix.

    Inputs: queue period slug.
    Outputs: normalized table prefix.
    Assumptions: current processed artifacts include June monthly and annual 2026.
    """

    return (
        "annual_financial_report_2026"
        if period_slug == "2026"
        else "monthly_financial_report_june_2026"
    )


def _period_slug_from_arguments(arguments: dict[str, Any], default: str = "2026") -> str:
    """Infer the most specific local scope from retrieval arguments.

    Inputs: tool arguments and default queue period.
    Outputs: period slug used for processed artifact lookup.
    Assumptions: 2026-06 and june_2026 both map to the June monthly artifact.
    """

    filters = arguments.get("filters", {})
    filters = filters if isinstance(filters, dict) else {}
    period = str(arguments.get("period") or filters.get("period") or default)
    if period in {"june_2026", "2026-06", "June 2026"}:
        return "june_2026"
    return "2026"


def _matches_text(value: Any, expected: str) -> bool:
    """Case-insensitively compare a processed field with a requested value.

    Inputs: processed value and requested string.
    Outputs: True when trimmed values match.
    Assumptions: exact local matching is safer than fuzzy matching at retrieval time.
    """

    return str(value).strip().casefold() == expected.strip().casefold()


def _contains_text(value: Any, expected: str) -> bool:
    """Case-insensitively check whether a processed field contains text.

    Inputs: processed value and requested string.
    Outputs: True when expected text appears in value.
    Assumptions: status labels may contain prefixes such as Overdue - Partial.
    """

    return expected.strip().casefold() in str(value).strip().casefold()


def _number(value: Any) -> float | None:
    """Convert a processed scalar to float when possible.

    Inputs: processed string or numeric scalar.
    Outputs: float or None.
    Assumptions: commas are formatting only in processed CSV exports.
    """

    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _latest_rows(
    rows: tuple[dict[str, Any], ...],
    months: int,
) -> tuple[dict[str, Any], ...]:
    """Keep rows within the latest requested month window.

    Inputs: processed rows and requested month count.
    Outputs: rows whose period is in the last N distinct periods.
    Assumptions: periods are ISO-like strings and sort chronologically.
    """

    periods = sorted(
        {str(row.get("period", ""))[:7] for row in rows if row.get("period")}
    )
    if not periods:
        return rows
    allowed = set(periods[-months:])
    return tuple(row for row in rows if str(row.get("period", ""))[:7] in allowed)


def _row_count_summary(rows: tuple[dict[str, Any], ...], label: str) -> str:
    """Create a neutral availability summary for retrieved rows.

    Inputs: retrieved rows and evidence label.
    Outputs: one-sentence summary with no financial interpretation.
    Assumptions: later analysis layers will reason over the evidence.
    """

    return f"Retrieved {len(rows)} processed {label} record(s)."


def _bounded_rows(rows: tuple[dict[str, Any], ...], limit: int = 200) -> list[dict[str, Any]]:
    """Limit evidence rows to a safe package size.

    Inputs: processed rows and maximum count.
    Outputs: list of at most limit rows.
    Assumptions: retrieval packages should stay compact for later LLM use.
    """

    return list(rows[:limit])


def _result(
    *,
    retrieval_name: str,
    rows: tuple[dict[str, Any], ...],
    source: str,
    label: str,
    warnings: tuple[str, ...] = (),
    unavailable: tuple[str, ...] = (),
    extra_data: dict[str, Any] | None = None,
) -> RetrievalResult:
    """Build a standard row-oriented retrieval result.

    Inputs: retrieval metadata, rows, sources, warnings, and optional extra data.
    Outputs: RetrievalResult.
    Assumptions: no recommendations or root-cause conclusions are added here.
    """

    data = {
        "summary": _row_count_summary(rows, label),
        "record_count": len(rows),
        "records": _bounded_rows(rows),
    }
    if extra_data:
        data.update(extra_data)
    return RetrievalResult(
        retrieval_name=retrieval_name,
        success=not unavailable,
        data=data,
        source_references=(source,),
        warnings=warnings,
        unavailable_data=unavailable,
        confidence=0.95 if rows else 0.55,
    )


def retrieve_department_history(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve processed department summary rows.

    Inputs: retrieval context and arguments with department/months.
    Outputs: department history result.
    Assumptions: local implementation reads normalized department summary CSVs.
    """

    department = str(arguments.get("department", "")).strip()
    months = int(arguments.get("months", 12))
    period_slug = "2026" if months > 1 else "june_2026"
    rows = context.normalized_records("department_summary", period_slug)
    matches = tuple(row for row in rows if _matches_text(row.get("department"), department))
    matches = _latest_rows(matches, months)
    unavailable = () if matches else (f"No department history found for {department}",)
    return _result(
        retrieval_name="department_history",
        rows=matches,
        source=f"normalized_tables/{_scope_prefix(period_slug)}__department_summary__table_01.csv",
        label="department history",
        unavailable=unavailable,
    )


def retrieve_payroll_history(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve processed payroll rows.

    Inputs: retrieval context and arguments with department/months.
    Outputs: payroll history result.
    Assumptions: department='all' returns institution-wide processed payroll rows.
    """

    department = str(arguments.get("department", "all")).strip()
    months = int(arguments.get("months", 12))
    period_slug = "2026" if months > 1 else "june_2026"
    rows = context.normalized_records("payroll", period_slug)
    if department.casefold() != "all":
        rows = tuple(row for row in rows if _matches_text(row.get("department"), department))
    rows = _latest_rows(rows, months)
    unavailable = () if rows else (f"No payroll history found for {department}",)
    return _result(
        retrieval_name="payroll_history",
        rows=rows,
        source=f"normalized_tables/{_scope_prefix(period_slug)}__payroll__table_01.csv",
        label="payroll history",
        unavailable=unavailable,
    )


def retrieve_vendor_history(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve processed vendor payment rows.

    Inputs: retrieval context and arguments with vendor/months.
    Outputs: vendor history result.
    Assumptions: vendor matching is exact to avoid accidental cross-vendor evidence.
    """

    vendor = str(arguments.get("vendor", "")).strip()
    months = int(arguments.get("months", 12))
    period_slug = "2026" if months > 1 else "june_2026"
    rows = context.normalized_records("vendor_payments", period_slug)
    matches = tuple(row for row in rows if _matches_text(row.get("vendor"), vendor))
    matches = _latest_rows(matches, months)
    unavailable = () if matches else (f"No vendor history found for {vendor}",)
    return _result(
        retrieval_name="vendor_history",
        rows=matches,
        source=f"normalized_tables/{_scope_prefix(period_slug)}__vendor_payments__table_01.csv",
        label="vendor payment history",
        unavailable=unavailable,
    )


def retrieve_student_payment_history(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve processed student payment rows.

    Inputs: retrieval context and optional period/month arguments.
    Outputs: student payment history result.
    Assumptions: local retrieval is table-backed and does not infer collection causes.
    """

    period_slug = _period_slug_from_arguments(arguments)
    rows = context.normalized_records("student_payments", period_slug)
    filters = arguments.get("filters", {}) if isinstance(arguments.get("filters"), dict) else {}
    if "status" in filters:
        rows = tuple(row for row in rows if _contains_text(row.get("status"), str(filters["status"])))
    unavailable = () if rows else ("No student payment records matched the request.",)
    return _result(
        retrieval_name="student_payment_history",
        rows=rows,
        source=f"normalized_tables/{_scope_prefix(period_slug)}__student_payments__table_01.csv",
        label="student payment history",
        unavailable=unavailable,
    )


def retrieve_cashflow_history(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve processed cash-flow rows.

    Inputs: retrieval context and optional period/month arguments.
    Outputs: cash-flow history result.
    Assumptions: cash-flow retrieval provides evidence, not cash recommendations.
    """

    period_slug = _period_slug_from_arguments(arguments)
    rows = context.normalized_records("cash_flow", period_slug)
    unavailable = () if rows else ("No cash-flow records matched the request.",)
    return _result(
        retrieval_name="cashflow_history",
        rows=rows,
        source=f"normalized_tables/{_scope_prefix(period_slug)}__cash_flow__table_01.csv",
        label="cash-flow history",
        unavailable=unavailable,
    )


def _transaction_candidate_tables(filters: dict[str, Any]) -> tuple[str, ...]:
    """Choose processed tables likely to satisfy transaction filters.

    Inputs: validated transaction filters.
    Outputs: normalized table type names.
    Assumptions: local processed outputs split transaction-like evidence by domain.
    """

    if filters.get("type") == "student_payment" or "status" in filters:
        return ("student_payments",)
    if "vendor" in filters:
        return ("vendor_payments",)
    if "expense_category" in filters or "department" in filters:
        return ("expenses", "vendor_payments")
    return ("student_payments", "vendor_payments", "expenses")


def _filter_transaction_rows(
    rows: tuple[dict[str, Any], ...],
    filters: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Apply safe deterministic filters to processed transaction rows.

    Inputs: candidate rows and validated filters.
    Outputs: matching processed rows.
    Assumptions: filters are field comparisons only, never executable code.
    """

    filtered = rows
    for key in ("department", "vendor", "expense_category"):
        if key in filters:
            filtered = tuple(row for row in filtered if _matches_text(row.get(key), str(filters[key])))
    if "status" in filters:
        filtered = tuple(row for row in filtered if _contains_text(row.get("status"), str(filters["status"])))
    if "minimum_amount" in filters:
        minimum = float(filters["minimum_amount"])
        amount_keys = ("amount", "amount_due", "amount_paid", "actual_expense")
        filtered = tuple(
            row
            for row in filtered
            if any((_number(row.get(key)) or 0.0) >= minimum for key in amount_keys)
        )
    limit = filters.get("limit")
    if isinstance(limit, int):
        filtered = filtered[:limit]
    return filtered


def retrieve_transactions(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve processed transaction rows matching bounded filters.

    Inputs: retrieval context and arguments with a filters object.
    Outputs: transaction evidence result.
    Assumptions: local retrieval searches normalized processed tables only.
    """

    filters = arguments.get("filters", {})
    filters = filters if isinstance(filters, dict) else {}
    period_slug = _period_slug_from_arguments({"filters": filters})
    rows_by_table: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for table_type in _transaction_candidate_tables(filters):
        try:
            table_rows = context.normalized_records(table_type, period_slug)
        except RetrievalInputError as exc:
            warnings.append(str(exc))
            continue
        matched = _filter_transaction_rows(table_rows, filters)
        if matched:
            rows_by_table[table_type] = list(matched)
    flattened = tuple(
        {**row, "_source_table": table_type}
        for table_type, rows in rows_by_table.items()
        for row in rows
    )
    unavailable = () if flattened else ("No processed transactions matched the filters.",)
    return _result(
        retrieval_name="transactions",
        rows=flattened,
        source=f"normalized_tables/{_scope_prefix(period_slug)}__transaction_tables",
        label="transaction",
        warnings=tuple(warnings),
        unavailable=unavailable,
        extra_data={"filters_applied": filters, "matched_tables": sorted(rows_by_table)},
    )


def retrieve_previous_cycle_memory(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve compact previous-cycle memory and unresolved data-quality notes.

    Inputs: retrieval context and empty arguments.
    Outputs: memory placeholder/data-quality result.
    Assumptions: no history database exists yet, so unavailable memory is explicit.
    """

    unresolved = [
        {
            "table_id": table.get("table_id"),
            "final_table_type": table.get("final_table_type"),
            "requires_human_review": table.get("requires_human_review"),
            "final_confidence": table.get("final_confidence"),
        }
        for table in context.enriched_model.get("tables", [])
        if isinstance(table, dict)
        and (
            table.get("requires_human_review")
            or table.get("final_table_type") == "Unknown"
        )
    ]
    return RetrievalResult(
        retrieval_name="previous_cycle_memory",
        success=True,
        data={
            "summary": (
                "Previous-cycle memory database is not implemented; returned "
                "current unresolved data-quality notes instead."
            ),
            "unresolved_table_count": len(unresolved),
            "unresolved_tables": unresolved[:50],
        },
        source_references=("outputs/intermediate/financial_document_model_enriched.json",),
        warnings=("Historical memory database is not available in Step 8.",),
        unavailable_data=("previous_cycle_memory_database",),
        confidence=0.6,
    )


def _monthly_trend_for_period(
    trends: tuple[dict[str, Any], ...],
    period: str,
) -> dict[str, Any] | None:
    """Find a processed monthly trend row for a period label.

    Inputs: monthly trend rows and period string.
    Outputs: matching row or None.
    Assumptions: trends use YYYY-MM period labels.
    """

    normalized = "2026-06" if period == "june_2026" else period
    for row in trends:
        if str(row.get("period")) == normalized:
            return row
    return None


def retrieve_financial_report(
    context: RetrievalContext,
    arguments: dict[str, Any],
) -> RetrievalResult:
    """Retrieve one processed financial report summary.

    Inputs: retrieval context and arguments with period.
    Outputs: processed report evidence.
    Assumptions: only processed JSON summaries and monthly trends are returned.
    """

    period = str(arguments.get("period", "2026"))
    if period in {"june_2026", "2026-06", "June 2026"}:
        return RetrievalResult(
            retrieval_name="financial_report",
            success=True,
            data={
                "summary": "Retrieved processed June 2026 finance summary.",
                "report": context.finance_summary_june,
            },
            source_references=("outputs/calculations/finance_summary_june_2026.json",),
            confidence=0.98,
        )
    if period == "2026":
        return RetrievalResult(
            retrieval_name="financial_report",
            success=True,
            data={
                "summary": "Retrieved processed annual 2026 finance summary.",
                "report": context.finance_summary_annual,
            },
            source_references=("outputs/calculations/finance_summary_2026.json",),
            confidence=0.98,
        )
    trend_row = _monthly_trend_for_period(context.monthly_trends, period)
    if trend_row is None:
        return RetrievalResult(
            retrieval_name="financial_report",
            success=False,
            data={"summary": "No processed report or trend row matched the period."},
            source_references=("outputs/calculations/monthly_trends_2026.csv",),
            unavailable_data=(f"financial_report:{period}",),
            confidence=0.2,
        )
    return RetrievalResult(
        retrieval_name="financial_report",
        success=True,
        data={
            "summary": f"Retrieved processed monthly trend row for {period}.",
            "monthly_trend": trend_row,
        },
        source_references=("outputs/calculations/monthly_trends_2026.csv",),
        warnings=("Detailed monthly finance summary is unavailable for this period.",),
        unavailable_data=(f"detailed_finance_summary:{period}",),
        confidence=0.7,
    )


def _request_from_queue_item(item: dict[str, Any]) -> RetrievalRequest:
    """Convert one execution queue item into a typed retrieval request.

    Inputs: Step 7 queue item.
    Outputs: RetrievalRequest.
    Assumptions: queue item shape was produced by validated planner code.
    """

    return RetrievalRequest(
        execution_id=str(item.get("execution_id", "")),
        task_id=str(item.get("step_id", "")),
        anomaly_id=item.get("anomaly_id"),
        question=str(item.get("question", "")),
        priority=str(item.get("priority", "")),
        tool_name=str(item.get("tool_name", "")),
        arguments=dict(item.get("arguments", {})),
        expected_output=str(item.get("expected_output", "")),
    )


def _evidence_summary(result: RetrievalResult) -> str:
    """Create a non-analytical summary for one retrieval result.

    Inputs: retrieval result.
    Outputs: short availability sentence.
    Assumptions: summaries must not contain recommendations or causal reasoning.
    """

    base = str(result.data.get("summary", "Retrieved processed evidence."))
    if result.unavailable_data:
        return f"{base} Unavailable: {', '.join(result.unavailable_data)}."
    return base


def _package_for_result(
    request: RetrievalRequest,
    result: RetrievalResult,
) -> EvidencePackage:
    """Build one investigation evidence package from a retrieval result.

    Inputs: original request and retrieval result.
    Outputs: task-level EvidencePackage.
    Assumptions: failures are packaged rather than raised to keep queue execution going.
    """

    return EvidencePackage(
        task_id=request.task_id,
        execution_id=request.execution_id,
        anomaly_id=request.anomaly_id,
        priority=request.priority,
        investigation_question=request.question,
        retrieved_evidence=result,
        evidence_summary=_evidence_summary(result),
        source_references=result.source_references,
        retrieval_warnings=result.warnings,
        confidence=result.confidence,
    )


def execute_retrieval_queue(
    queue: dict[str, Any],
    context: RetrievalContext,
    registry: RetrievalRegistry | None = None,
) -> dict[str, Any]:
    """Execute all retrieval requests in a validated execution queue.

    Inputs: Step 7 execution queue, retrieval context, and optional registry.
    Outputs: evidence package document with per-task evidence and counters.
    Assumptions: retrieval runs sequentially and continues after individual failures.
    """

    active_registry = registry or create_default_registry()
    packages: list[EvidencePackage] = []
    for item in queue.get("items", []):
        request = _request_from_queue_item(item)
        try:
            tool = active_registry.get(request.tool_name)
            result = tool.function(context, request.arguments)
        except Exception as exc:  # noqa: BLE001 - queue execution must not abort.
            # A retrieval failure is evidence metadata for the next layer, not a
            # reason to stop executing the remaining validated queue.
            result = RetrievalResult(
                retrieval_name=request.tool_name,
                success=False,
                data={"summary": "Retrieval failed before evidence was returned."},
                source_references=(),
                warnings=(str(exc),),
                unavailable_data=(request.tool_name,),
                confidence=0.0,
            )
        packages.append(_package_for_result(request, result))

    success_count = sum(package.retrieved_evidence.success for package in packages)
    unavailable_count = sum(
        bool(package.retrieved_evidence.unavailable_data)
        for package in packages
    )
    summary = RetrievalRunSummary(
        package_id=f"EVIDENCE-{str(queue.get('period_slug', '')).upper().replace('_', '-')}",
        period_slug=str(queue.get("period_slug", "")),
        tasks_executed=len(packages),
        successful_retrievals=success_count,
        failed_retrievals=len(packages) - success_count,
        unavailable_evidence=unavailable_count,
    )
    return {
        "package_id": summary.package_id,
        "period_slug": summary.period_slug,
        "source_queue_id": queue.get("queue_id"),
        "source_plan_id": queue.get("source_plan_id"),
        "retrieval_status": "completed",
        "tools_executed": True,
        "summary": summary.to_dict(),
        "evidence_packages": [package.to_dict() for package in packages],
    }


def build_retrieval_summary(packages: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Build a cross-scope retrieval summary artifact.

    Inputs: completed evidence package documents.
    Outputs: compact JSON-compatible summary.
    Assumptions: individual package summaries already contain accurate counters.
    """

    scope_summaries = [package["summary"] for package in packages]
    return {
        "summary_id": "RETRIEVAL-SUMMARY-2026",
        "total_scopes": len(scope_summaries),
        "tasks_executed": sum(item["tasks_executed"] for item in scope_summaries),
        "successful_retrievals": sum(item["successful_retrievals"] for item in scope_summaries),
        "failed_retrievals": sum(item["failed_retrievals"] for item in scope_summaries),
        "unavailable_evidence": sum(item["unavailable_evidence"] for item in scope_summaries),
        "scopes": scope_summaries,
    }


def load_execution_queue(path: str | Path) -> dict[str, Any]:
    """Load one validated Step 7 execution queue.

    Inputs: execution queue JSON path.
    Outputs: parsed queue object.
    Assumptions: queue roots are JSON objects.
    """

    return _read_json(Path(path))


def save_json_artifact(data: dict[str, Any], output_path: str | Path) -> Path:
    """Save one retrieval artifact as readable JSON.

    Inputs: JSON-compatible data and output path.
    Outputs: resolved path written to disk.
    Assumptions: parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
