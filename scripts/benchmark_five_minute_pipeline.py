"""Run a reproducible end-to-end five-minute-SLA benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_agent.orchestration import PipelineConfig, build_pipeline_input_model, run_pipeline_for_report  # noqa: E402


def main() -> None:
    """Execute one isolated integrated-workbook run and print timing telemetry.

    Inputs: workbook/model/output CLI options.
    Outputs: complete pipeline artifacts and machine-readable performance trace.
    Assumptions: Ollama is already installed; this command never downloads a model.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:30b-a3b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-db", type=Path, required=True)
    parser.add_argument("--period", default=None)
    parser.add_argument("--degraded", action="store_true", help="Profile deterministic/report stages without Ollama inference")
    args = parser.parse_args()
    input_model = build_pipeline_input_model(workbook_path=args.workbook, period=args.period, report_language="es")
    config = PipelineConfig.from_project_root(
        ROOT, python_executable=sys.executable, ollama_model=args.model,
        analysis_ollama_model=args.model, read_timeout_seconds=180,
        stage_timeout_seconds=180, ollama_keep_alive="15m", input_model=input_model,
        enable_cache=False, enable_memory_storage=True,
        memory_database_path=args.memory_db, output_directory=args.output_dir,
        strategic_ai_mode="degraded" if args.degraded else "ai",
    )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    result = run_pipeline_for_report(input_model, config)
    trace = args.output_dir / "performance" / f"performance_trace_{input_model.effective_period_label.replace('-', '_')}.json"
    if not trace.is_file():
        trace.parent.mkdir(parents=True, exist_ok=True)
        history_stage = next((stage for stage in result.stages if stage.stage_name == "historical_context"), None)
        strategic_stage = next((stage for stage in result.stages if stage.stage_name == "strategic_analysis"), None)
        strategic_calls = sum(
            int(item.get("ollama_call_count", 0) or 0)
            for item in ((strategic_stage.telemetry if strategic_stage else {}).get("stage_telemetry", []))
            if isinstance(item, dict)
        )
        trace.write_text(json.dumps({
            "trace_version": "1.0", "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": time.monotonic() - started,
            "runtime_budget_seconds": 300,
            "operation_counts": {
                "workbook_reads": 1,
                "canonical_model_reloads": 0,
                "sqlite_history_queries": int((history_stage.telemetry if history_stage else {}).get("database_queries", 0) or 0),
                "ollama_primary_and_repair_calls": strategic_calls,
                "persisted_output_files": len(result.output_files),
            },
            "stages": [{
                "stage_name": stage.stage_name, "duration_seconds": stage.runtime_seconds,
                "success": stage.success, "telemetry": stage.telemetry,
            } for stage in result.stages],
            "privacy": "No raw financial contents are included.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TOTAL {result.runtime_summary.total_runtime_seconds:.1f} s")
    for stage in result.stages:
        print(f"{stage.display_name:32} {stage.runtime_seconds:8.1f} s")
    print(f"SUCCESS {result.success}")
    print(f"TRACE {trace}")
    print(json.dumps({"success": result.success, "trace": str(trace)}, ensure_ascii=False))
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
