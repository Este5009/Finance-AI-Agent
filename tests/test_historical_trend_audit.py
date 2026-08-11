"""Final-layer regression coverage for the packaged historical chart path."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.audit_historical_trend_path import build_audit
from scripts.import_synthetic_history import import_synthetic_history


def test_runtime_db_reaches_final_streamlit_vega_rows_without_interpolation(tmp_path: Path) -> None:
    """Apr-Aug accepted DB history plus Sep reaches the actual results chart spec."""

    current_artifacts = (
        "calculations/finance_summary_2026_09.json",
        "calculations/kpi_summary_2026_09.csv",
        "anomalies/anomaly_report_2026_09.json",
        "evidence/evidence_package_2026_09.json",
        "analysis/strategic_analysis_2026_09.json",
        "report/report_model_2026_09.json",
        "report/financial_report_2026_09.html",
        "report/financial_report_2026_09.pdf",
    )
    missing = [name for name in current_artifacts if not (Path("outputs") / name).is_file()]
    if missing:
        pytest.skip(f"September audit artifacts unavailable: {missing}")
    for name in current_artifacts:
        destination = tmp_path / "outputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("outputs") / name, destination)
    database = tmp_path / "database" / "finance_memory.db"
    import_synthetic_history(
        Path("data/memory/recovery_2026_memory.db"), database,
        periods=("2026_04", "2026_05", "2026_06", "2026_07", "2026_08"),
    )

    audit = build_audit("2026_09", project_root=tmp_path, memory_db=database)
    expected_labels = ["Abr 2026", "May 2026", "Jun 2026", "Jul 2026", "Ago 2026", "Sep 2026"]
    expected_periods = ["2026_04", "2026_05", "2026_06", "2026_07", "2026_08", "2026_09"]
    for metric in ("total_revenue", "total_expenses", "net_operating_result"):
        metric_audit = audit["metrics"][metric]
        rows = metric_audit["streamlit_spec"]["vega_data"]
        assert [row["period"] for row in rows] == expected_periods
        assert metric_audit["streamlit_spec"]["x_axis_values"] == expected_labels
        assert len(rows) == len({row["period"] for row in rows}) == 6
        assert metric_audit["streamlit_spec"]["has_aggregate"] is False
        assert metric_audit["streamlit_spec"]["has_transform"] is False
        assert metric_audit["first_divergence_from_expected"] == ""

    # The audit uses the same app-data output resolution contract as packaged Streamlit.
    selected = Path(audit["report_model_path_selected_by_ui"])
    assert selected == tmp_path / "outputs" / "report" / "report_model_2026_09.json"
