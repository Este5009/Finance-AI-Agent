"""Benchmark stage-specific Ollama model routing on the synthetic report."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.orchestration import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    EXPERIMENTAL_FAST_OLLAMA_MODEL,
    PipelineConfig,
    build_pipeline_input_model,
    run_pipeline_for_report,
)
from finance_agent.reporting.report_quality import validate_report_artifacts  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact if it exists.

    Inputs: artifact path.
    Outputs: parsed object or empty dictionary.
    Assumptions: missing benchmark artifacts should become quality warnings.
    """

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _summarize_run(label: str, result: Any, period_slug: str) -> dict[str, Any]:
    """Collect benchmark metrics from one completed pipeline run.

    Inputs: benchmark label, PipelineRunResult, and period slug.
    Outputs: JSON-compatible summary metrics.
    Assumptions: each run writes the standard period-slugged artifacts.
    """

    outputs = PROJECT_ROOT / "outputs"
    enriched = _read_json(
        outputs / "intermediate" / period_slug / "financial_document_model_enriched.json"
    )
    planner = _read_json(outputs / "plans" / f"ollama_plan_{period_slug}.json")
    analysis = _read_json(outputs / "analysis" / f"strategic_analysis_{period_slug}.json")
    report_model = outputs / "report" / f"report_model_{period_slug}.json"
    html = outputs / "report" / f"financial_report_{period_slug}.html"
    pdf = outputs / "report" / f"financial_report_{period_slug}.pdf"
    quality = (
        validate_report_artifacts(report_model, html_path=html, pdf_path=pdf)
        if report_model.is_file()
        else None
    )
    return {
        "label": label,
        "success": result.success,
        "cache_hit": result.cache_hit,
        "total_runtime_seconds": result.runtime_summary.total_runtime_seconds,
        "stage_runtimes": {
            stage.stage_name: stage.runtime_seconds for stage in result.stages
        },
        "stage_status": {
            stage.stage_name: {
                "success": stage.success,
                "skipped": stage.skipped,
                "warnings": list(stage.warnings),
            }
            for stage in result.stages
        },
        "structure": {
            "items_detected": enriched.get("enrichment", {}).get("items_detected"),
            "accepted": sum(
                bool(table.get("llm_confidence")) and not table.get("requires_human_review")
                for table in enriched.get("tables", [])
                if isinstance(table, dict)
            ),
            "rejected_or_review": sum(
                bool(table.get("requires_human_review"))
                for table in enriched.get("tables", [])
                if isinstance(table, dict)
            ),
        },
        "planner": {
            "validation_status": planner.get("validation_status"),
            "fallback_used": planner.get("fallback_used"),
            "validation_errors": planner.get("validation_errors", []),
        },
        "strategic_analysis": {
            "validation_status": analysis.get("validation_status"),
            "confidence": analysis.get("analysis", {}).get("confidence"),
            "recommendation_count": analysis.get("recommendation_count"),
        },
        "report_quality": {
            "is_valid": quality.is_valid if quality else False,
            "errors": list(quality.errors) if quality else ["report model missing"],
            "recommendation_count": quality.recommendation_count if quality else 0,
        },
    }


def _run_config(
    *,
    label: str,
    structure_model: str | None,
    planner_model: str | None,
    analysis_model: str | None,
    single_model: str,
) -> dict[str, Any]:
    """Run one benchmark configuration.

    Inputs: benchmark label and stage-specific model routing.
    Outputs: collected runtime and quality metrics.
    Assumptions: cache is disabled so every benchmark measures fresh LLM calls.
    """

    input_model = build_pipeline_input_model(
        financial_report_path=PROJECT_ROOT
        / "data"
        / "synthetic"
        / "monthly_financial_report_june_2026.xlsx",
        goals_document_path=PROJECT_ROOT / "data" / "synthetic" / "financial_goals_2026.pdf",
        period_override="2026-06",
        report_language="es",
    )
    config = PipelineConfig.from_project_root(
        PROJECT_ROOT,
        python_executable=sys.executable,
        ollama_model=single_model,
        structure_ollama_model=structure_model,
        planner_ollama_model=planner_model,
        analysis_ollama_model=analysis_model,
        enable_cache=False,
        input_model=input_model,
    )
    started = time.perf_counter()
    result = run_pipeline_for_report(input_model, config)
    metrics = _summarize_run(label, result, "2026_06")
    metrics["wall_clock_seconds"] = time.perf_counter() - started
    metrics["models"] = config.effective_ollama_models()
    return metrics


def main() -> None:
    """Run both benchmark configurations and write a comparison summary.

    Inputs: none; uses the current synthetic June report.
    Outputs: benchmark JSON under outputs/benchmarks and console summary.
    Assumptions: Ollama is running with both requested models installed.
    """

    large = DEFAULT_OLLAMA_MODEL
    small = EXPERIMENTAL_FAST_OLLAMA_MODEL
    results = [
        _run_config(
            label="all_large",
            structure_model=None,
            planner_model=None,
            analysis_model=None,
            single_model=large,
        ),
        _run_config(
            label="small_structure_planner_large_analysis",
            structure_model=small,
            planner_model=small,
            analysis_model=large,
            single_model=large,
        ),
    ]
    baseline = results[0]
    recommended = baseline["label"]
    candidate = results[1]
    if (
        candidate["report_quality"]["is_valid"]
        and candidate["strategic_analysis"]["validation_status"] == "accepted"
        and candidate["total_runtime_seconds"] <= baseline["total_runtime_seconds"]
    ):
        recommended = candidate["label"]
    summary = {
        "benchmark_id": "ollama_stage_model_benchmark",
        "results": results,
        "recommended_configuration": recommended,
        "recommendation_reason": (
            "The mixed setup is recommended only when it is faster and report "
            "quality plus strategic validation pass."
        ),
    }
    output = PROJECT_ROOT / "outputs" / "benchmarks" / "ollama_model_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    for result in results:
        print(
            f"{result['label']}: total={result['total_runtime_seconds']:.2f}s "
            f"quality={result['report_quality']['is_valid']} "
            f"analysis={result['strategic_analysis']['validation_status']}"
        )
    print(f"recommended={recommended}")
    print(f"summary={output}")


if __name__ == "__main__":
    main()
