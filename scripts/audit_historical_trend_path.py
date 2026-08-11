"""Audit the end-to-end historical trend path without launching Streamlit.

This diagnostic follows the same artifact and presentation helpers used by the
Streamlit results page. It writes a compact JSON file under ``outputs/debug`` so
chart regressions can be traced from SQLite/processed artifacts to the final
Vega-Lite payload passed to Streamlit.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.memory.context_builder import build_historical_context  # noqa: E402
from finance_agent.orchestration import PipelineConfig  # noqa: E402
from finance_agent.orchestration.pipeline_models import PIPELINE_SCHEMA_VERSION  # noqa: E402
from finance_agent.reporting import (  # noqa: E402
    build_report_model,
    load_report_inputs,
    save_report_html,
    save_report_model,
)
from finance_agent.reporting.renderers import render_report_pdf  # noqa: E402
from finance_agent.ui import streamlit_app  # noqa: E402


DEFAULT_METRICS = (
    "total_revenue", "total_expenses", "net_operating_result",
    "payroll_percentage_of_revenue", "collection_rate", "net_cash_flow", "ending_cash",
)


class CaptureStreamlit:
    """Small Streamlit stand-in that captures final chart payloads.

    Inputs: none.
    Outputs: captured Vega-Lite specs and markdown calls.
    Assumptions: this mirrors the subset of Streamlit used by ``_render_results``.
    """

    def __init__(self) -> None:
        """Initialize capture collections."""

        self.vega_lite_charts: list[dict[str, Any]] = []
        self.markdown_calls: list[str] = []
        self.tab_labels: list[str] = []
        self.errors: list[str] = []
        self.tables: list[Any] = []
        self.downloads: list[dict[str, Any]] = []

    def __enter__(self) -> "CaptureStreamlit":
        """Return this object for context-manager tabs/expanders."""

        return self

    def __exit__(self, *_args: object) -> bool:
        """Do not suppress exceptions."""

        return False

    def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
        """Capture markdown text."""

        del unsafe_allow_html
        self.markdown_calls.append(text)

    def caption(self, text: str) -> None:
        """Capture captions as markdown-like text."""

        self.markdown_calls.append(text)

    def subheader(self, text: str) -> None:
        """Capture subheaders."""

        self.markdown_calls.append(text)

    def info(self, text: str) -> None:
        """Capture info messages."""

        self.markdown_calls.append(text)

    def success(self, text: str) -> None:
        """Capture success messages."""

        self.markdown_calls.append(text)

    def warning(self, text: str) -> None:
        """Capture warning messages."""

        self.markdown_calls.append(text)

    def error(self, text: str) -> None:
        """Capture error messages."""

        self.errors.append(text)

    def code(self, text: str) -> None:
        """Capture code/traceback blocks."""

        self.markdown_calls.append(text)

    def dataframe(self, rows: Any, **_kwargs: object) -> None:
        """Capture dataframe rows."""

        self.tables.append(rows)

    def tabs(self, labels: list[str]) -> list["CaptureStreamlit"]:
        """Return tab context managers."""

        self.tab_labels = list(labels)
        return [self for _ in labels]

    def expander(self, _label: str, *, expanded: bool = False) -> "CaptureStreamlit":
        """Return this object for expander context."""

        del expanded
        return self

    def download_button(self, **kwargs: object) -> None:
        """Capture download metadata."""

        self.downloads.append(dict(kwargs))

    def vega_lite_chart(self, spec: dict[str, Any], *, use_container_width: bool = False) -> None:
        """Capture the exact chart spec passed to Streamlit."""

        self.vega_lite_charts.append({"spec": spec, "use_container_width": use_container_width})


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object if present.

    Inputs: JSON path.
    Outputs: parsed dictionary or empty dictionary.
    Assumptions: missing optional artifacts are reported in the audit.
    """

    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _metric_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return period/value pairs from heterogeneous records.

    Inputs: records from DB, context, model, presentation, or chart specs.
    Outputs: compact period/value dictionaries.
    Assumptions: the caller has already selected one metric.
    """

    pairs: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        period = record.get("period") or record.get("period_label")
        value = record.get("value")
        if period is not None and value is not None:
            pairs.append({"period": str(period), "value": float(value)})
    return pairs


def _db_metric_history(database_path: Path, metric: str, *, before_period: str) -> list[dict[str, Any]]:
    """Read stored KPI history directly from SQLite for audit purposes.

    Inputs: DB path, metric, and exclusive current period.
    Outputs: chronologically ordered period/value records.
    Assumptions: this is read-only diagnostics; production retrieval remains in
    ``finance_agent.memory``.
    """

    if not database_path.is_file():
        return []
    aliases = (metric, "student_payment_collection_rate") if metric == "collection_rate" else (metric,)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"""
            WITH latest_runs AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY period ORDER BY updated_at_utc DESC, completed_at_utc DESC, run_id DESC
                ) AS revision_rank
                FROM pipeline_runs
                WHERE status = 'completed' AND period < ?
            )
            SELECT pr.period, k.metric, k.value
            FROM kpis k
            JOIN latest_runs pr ON pr.run_id = k.run_id AND pr.revision_rank = 1
            WHERE k.metric IN ({','.join('?' for _ in aliases)})
            ORDER BY pr.period
            """,
            (before_period, *aliases),
        ).fetchall()
    return [{"period": period, "metric": found_metric, "value": value} for period, found_metric, value in rows]


def _processed_metric_history(project_root: Path, metric: str, *, current_period: str) -> list[dict[str, Any]]:
    """Read historical metric values from processed monthly finance summaries.

    Inputs: project root, metric, and current monthly period.
    Outputs: prior period/value rows.
    Assumptions: these artifacts are deterministic calculation outputs.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "outputs" / "calculations").glob("finance_summary_20??_??.json")):
        period = path.stem.replace("finance_summary_", "")
        if period >= current_period:
            continue
        document = _read_json(path)
        finance = document.get("finance_summary", {})
        finance = finance if isinstance(finance, dict) else {}
        payments = finance.get("student_payments", {}) if isinstance(finance.get("student_payments"), dict) else {}
        cash_flow = finance.get("cash_flow", {}) if isinstance(finance.get("cash_flow"), dict) else {}
        value = {
            "total_revenue": finance.get("total_revenue"),
            "total_expenses": finance.get("total_expenses"),
            "net_operating_result": finance.get("net_operating_result"),
            "payroll_percentage_of_revenue": finance.get("payroll_percentage_of_revenue"),
            "collection_rate": payments.get("collection_rate"),
            "net_cash_flow": cash_flow.get("net_cash_flow"),
            "ending_cash": cash_flow.get("ending_cash"),
        }.get(metric)
        if value is not None:
            rows.append({"period": period, "metric": metric, "value": float(value), "source": str(path)})
    return rows


def _context_metric_history(context: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    """Extract metric records from compact historical context.

    Inputs: historical context and metric ID.
    Outputs: period/value rows.
    Assumptions: aliases are normalized for collection-rate compatibility.
    """

    aliases = {metric}
    if metric == "collection_rate":
        aliases.add("student_payment_collection_rate")
    for retrieval in context.get("retrievals", []) if isinstance(context, dict) else []:
        if not isinstance(retrieval, dict) or retrieval.get("tool_name") != "get_metric_history":
            continue
        found = str(retrieval.get("metric") or retrieval.get("arguments", {}).get("metric") or "")
        if found in aliases:
            return _metric_pairs(retrieval.get("records", []))
    return []


def _report_model_metric_history(report_model: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    """Extract one trend series from the report model.

    Inputs: report model and metric ID.
    Outputs: period/value rows.
    Assumptions: report model stores canonical trend series under
    ``historical_trends``.
    """

    for section in report_model.get("sections", []) if isinstance(report_model, dict) else []:
        if isinstance(section, dict) and section.get("section_id") == "historical_trends":
            for series in section.get("content", {}).get("trend_series", []):
                if isinstance(series, dict) and series.get("metric_id") == metric:
                    return _metric_pairs(series.get("points", []))
    return []


def _presentation_metric_history(report_model: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    """Extract one trend series after presentation adaptation.

    Inputs: report model and metric ID.
    Outputs: period/value rows.
    Assumptions: presentation adapter is the source shared by renderers/UI.
    """

    view = streamlit_app.build_presentation_view(report_model)
    for series in view.get("historical", {}).get("trends", []):
        if isinstance(series, dict) and series.get("metric_id") == metric:
            return _metric_pairs(series.get("points", []))
    return []


def _streamlit_chart_payload(report_model: dict[str, Any], metric: str) -> dict[str, Any]:
    """Capture final Streamlit chart rows/spec for one metric.

    Inputs: report model and metric ID.
    Outputs: chart rows plus spec metadata.
    Assumptions: this uses the same helper as the actual UI.
    """

    view = streamlit_app.build_presentation_view(report_model)
    for series in view.get("historical", {}).get("trends", []):
        if isinstance(series, dict) and series.get("metric_id") == metric:
            rows = streamlit_app._trend_chart_rows(series)
            spec = streamlit_app._trend_chart_spec(series)
            return {
                "rows": _metric_pairs(rows),
                "vega_data": _metric_pairs(spec.get("data", {}).get("values", [])),
                "mark": spec.get("mark"),
                "x_axis_values": spec.get("encoding", {}).get("x", {}).get("axis", {}).get("values"),
                "has_aggregate": "aggregate" in json.dumps(spec),
                "has_transform": "transform" in spec,
            }
    return {"rows": [], "vega_data": []}


def _first_divergence(
    stages: dict[str, list[dict[str, Any]]],
    *,
    expected_prior: list[dict[str, Any]],
    expected_full: list[dict[str, Any]],
) -> str:
    """Find the first canonical stage whose period/value pairs differ.

    Inputs: stage mapping and expected rows.
    Outputs: stage name or empty string when all match.
    Assumptions: expected rows represent the required rolling-window contract.
    """

    prior_pairs = [(row["period"], round(float(row["value"]), 6)) for row in expected_prior]
    full_pairs = [(row["period"], round(float(row["value"]), 6)) for row in expected_full]
    for stage, rows in stages.items():
        pairs = [(row["period"], round(float(row["value"]), 6)) for row in rows]
        expected_pairs = prior_pairs if stage in {"sqlite_memory", "history_query", "historical_context_raw", "report_engine_input"} else full_pairs
        if pairs != expected_pairs:
            return stage
    return ""


def build_audit(period: str, *, project_root: Path, memory_db: Path) -> dict[str, Any]:
    """Build the historical trend audit payload.

    Inputs: period slug, project root, and configured memory DB.
    Outputs: JSON-compatible audit dictionary.
    Assumptions: September processed outputs already exist for this diagnostic.
    """

    config = PipelineConfig.from_project_root(
        project_root,
        python_executable=sys.executable,
        memory_database_path=memory_db,
    )
    finance = _read_json(project_root / "outputs" / "calculations" / f"finance_summary_{period}.json")
    anomalies = _read_json(project_root / "outputs" / "anomalies" / f"anomaly_report_{period}.json")
    evidence = _read_json(project_root / "outputs" / "evidence" / f"evidence_package_{period}.json")
    history = build_historical_context(
        current_period=period,
        finance_summary=finance,
        anomaly_report=anomalies,
        evidence_package=evidence,
        database_path=memory_db,
        purpose="report_model",
    ).context
    report_inputs = load_report_inputs(project_root, period, memory_database_path=memory_db)
    fresh_model = build_report_model(report_inputs).to_dict()
    model_path = project_root / "outputs" / "report" / f"report_model_{period}.json"
    capture = CaptureStreamlit()
    streamlit_app._render_results(capture, _fake_result_for_period(config, period))
    # The real results path may refresh and replace a stale historical artifact.
    # Read it afterwards so the audit reflects what the packaged UI selected.
    disk_model = _read_json(model_path)
    actual_model_path = streamlit_app._artifact_paths(_fake_result_for_period(config, period)).get("Report model JSON")
    metrics: dict[str, Any] = {}
    for metric in DEFAULT_METRICS:
        expected_prior = _metric_pairs(_db_metric_history(memory_db, metric, before_period=period))[-5:]
        current = _report_model_metric_history(fresh_model, metric)[-1:]
        expected_full = [*expected_prior, *current] if current else list(expected_prior)
        stages = {
            "sqlite_memory": expected_prior,
            "history_query": _context_metric_history(history, metric),
            "historical_context_raw": _context_metric_history(history, metric),
            "report_engine_input": _context_metric_history(report_inputs.strategic_analysis.get("historical_context", {}), metric),
            "report_model_disk": _report_model_metric_history(disk_model, metric),
            "report_model_fresh": _report_model_metric_history(fresh_model, metric),
            "presentation": _presentation_metric_history(fresh_model, metric),
            "streamlit_rows": _streamlit_chart_payload(fresh_model, metric).get("rows", []),
            "vega_data": _streamlit_chart_payload(fresh_model, metric).get("vega_data", []),
        }
        metrics[metric] = {
            "expected_prior": expected_prior,
            "expected_full": expected_full,
            "processed_artifacts_supplement": _metric_pairs(_processed_metric_history(project_root, metric, current_period=period)),
            "stages": stages,
            "first_divergence_from_expected": _first_divergence(
                stages, expected_prior=expected_prior, expected_full=expected_full,
            ),
            "streamlit_spec": _streamlit_chart_payload(fresh_model, metric),
        }
    return {
        "period": period,
        "project_root": str(project_root),
        "memory_db": str(memory_db),
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "report_model_path_selected_by_ui": str(actual_model_path) if actual_model_path else None,
        "report_model_path_mtime": model_path.stat().st_mtime if model_path.is_file() else None,
        "streamlit_tabs": capture.tab_labels,
        "streamlit_errors": capture.errors,
        "metrics": metrics,
    }


def _fake_result_for_period(config: PipelineConfig, period: str) -> Any:
    """Build a minimal PipelineRunResult-like object for artifact resolution.

    Inputs: pipeline config and period slug.
    Outputs: PipelineRunResult instance.
    Assumptions: diagnostics only need output paths and runtime metadata.
    """

    from finance_agent.orchestration.pipeline_models import PipelineRunResult, RuntimeSummary

    outputs = (
        str(config.output_directory / "report" / f"report_model_{period}.json"),
        str(config.output_directory / "report" / f"financial_report_{period}.html"),
        str(config.output_directory / "report" / f"financial_report_{period}.pdf"),
    )
    return PipelineRunResult(
        success=True,
        stages=(),
        output_files=outputs,
        warnings=(),
        runtime_summary=RuntimeSummary(0, 0, 0, 0, 0, 0),
        config=config,
    )


def main() -> None:
    """Run the historical trend audit and write JSON output.

    Inputs: CLI period/database options.
    Outputs: ``outputs/debug/historical_trend_audit_<period>.json``.
    Assumptions: this command is bounded and never starts local services.
    """

    parser = argparse.ArgumentParser(description="Audit historical trend chart data path.")
    parser.add_argument("--period", default="2026_09")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--memory-db", type=Path, default=PROJECT_ROOT / "data" / "memory" / "finance_memory.db")
    args = parser.parse_args()

    audit = build_audit(args.period, project_root=args.project_root, memory_db=args.memory_db)
    output_path = args.project_root / "outputs" / "debug" / f"historical_trend_audit_{args.period}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"audit_path": str(output_path), "period": args.period}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
