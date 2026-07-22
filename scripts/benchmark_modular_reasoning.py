"""Benchmark Phase 14 modular reasoning on December 2026 synthetic history."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.analysis.strategic_analysis import (  # noqa: E402
    build_evidence_ledger,
    save_json_artifact,
    strategic_analysis_json_schema,
)
from finance_agent.llm.ollama_client import (  # noqa: E402
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
)
from finance_agent.memory.context_builder import build_historical_context  # noqa: E402
from finance_agent.reasoning.reasoning_models import ReasoningStageResult  # noqa: E402
from finance_agent.reasoning.reasoning_pipeline import (  # noqa: E402
    _analysis_document,
    _empty_analysis,
    _run_structured_stage,
    build_financial_performance_prompt,
    build_historical_operational_prompt,
    build_strategic_synthesis_prompt,
    validate_reasoning_stage_response,
    validate_strategic_synthesis_response,
)
from finance_agent.reasoning.reasoning_state import ReasoningState  # noqa: E402
from finance_agent.reporting.report_engine import ReportInputBundle, build_report_model, save_report_model  # noqa: E402
from finance_agent.reporting.renderers import render_report_pdf, save_report_html  # noqa: E402


PERIOD_SLUG = "2026_12"
MONOLITHIC_BASELINE_PROMPT_CHARS = 25_283


def build_argument_parser() -> argparse.ArgumentParser:
    """Create benchmark CLI options.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: defaults target the local synthetic December dataset.
    """

    parser = argparse.ArgumentParser(description="Benchmark modular reasoning for December 2026.")
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=600.0)
    parser.add_argument("--stage-timeout", type=float, default=900.0)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--warm-up", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "memory" / "recovery_2026_memory.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "profiling" / "modular_reasoning_benchmark.json",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    """Load one processed JSON artifact.

    Inputs: JSON path.
    Outputs: parsed dictionary.
    Assumptions: benchmark never reads raw Excel/PDF inputs.
    """

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return value


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load December processed inputs.

    Inputs: none.
    Outputs: finance summary, anomaly report, risk summary, evidence package.
    Assumptions: upstream pipeline artifacts already exist for December.
    """

    outputs = PROJECT_ROOT / "outputs"
    return (
        _load_json(outputs / "calculations" / f"finance_summary_{PERIOD_SLUG}.json"),
        _load_json(outputs / "anomalies" / f"anomaly_report_{PERIOD_SLUG}.json"),
        _load_json(outputs / "anomalies" / f"risk_summary_{PERIOD_SLUG}.json"),
        _load_json(outputs / "evidence" / f"evidence_package_{PERIOD_SLUG}.json"),
    )


def _checkpoint_path(stage_id: str) -> Path:
    """Return the checkpoint path for one reasoning stage.

    Inputs: stage ID.
    Outputs: checkpoint file path.
    Assumptions: checkpoints are debug/profiling artifacts only.
    """

    return PROJECT_ROOT / "outputs" / "profiling" / "modular_reasoning_checkpoints" / f"{stage_id}_{PERIOD_SLUG}.json"


def _load_checkpoint(stage_id: str) -> ReasoningStageResult | None:
    """Load an accepted stage checkpoint if present.

    Inputs: stage ID.
    Outputs: ReasoningStageResult or None.
    Assumptions: invalid/rejected checkpoints are not reused.
    """

    path = _checkpoint_path(stage_id)
    if not path.is_file():
        return None
    data = _load_json(path)
    if not data.get("accepted"):
        return None
    return ReasoningStageResult(
        stage_id=str(data["stage_id"]),
        stage_name=str(data["stage_name"]),
        accepted=True,
        payload=data.get("payload", {}),
        validation_errors=tuple(data.get("validation_errors", [])),
        telemetry=data.get("telemetry", {}),
    )


def _save_checkpoint(result: ReasoningStageResult) -> Path:
    """Save one stage checkpoint.

    Inputs: stage result.
    Outputs: written checkpoint path.
    Assumptions: rejected checkpoints are also saved for diagnostics but are not reused.
    """

    path = _checkpoint_path(result.stage_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _run_stage_with_optional_resume(
    *,
    stage_id: str,
    stage_name: str,
    prompt: str,
    client: OllamaClient,
    validator: Any,
    resume: bool,
    response_format: dict[str, Any] | str = "json",
    stage_timeout_seconds: float,
) -> ReasoningStageResult:
    """Run or reuse one reasoning stage checkpoint.

    Inputs: stage metadata, prompt/client/validator, resume flag and timeouts.
    Outputs: stage result.
    Assumptions: accepted checkpoints are immutable for a fixed input run.
    """

    if resume:
        checkpoint = _load_checkpoint(stage_id)
        if checkpoint is not None:
            copied = ReasoningStageResult(
                stage_id=checkpoint.stage_id,
                stage_name=checkpoint.stage_name,
                accepted=checkpoint.accepted,
                payload=checkpoint.payload,
                validation_errors=checkpoint.validation_errors,
                telemetry={**checkpoint.telemetry, "checkpoint_reused": True},
            )
            return copied
    result = _run_structured_stage(
        client=client,
        stage_id=stage_id,
        stage_name=stage_name,
        prompt=prompt,
        validator=validator,
        response_format=response_format,
        stage_timeout_seconds=stage_timeout_seconds,
    )
    _save_checkpoint(result)
    return result


def main() -> None:
    """Run the December modular reasoning benchmark.

    Inputs: CLI timeout/model options.
    Outputs: profiling JSON plus reasoning/report artifacts when accepted.
    Assumptions: no memory storage or pipeline cache is used by this benchmark.
    """

    args = build_argument_parser().parse_args()
    started = time.perf_counter()
    finance_summary, anomaly_report, risk_summary, evidence_package = _inputs()
    history = build_historical_context(
        current_period=PERIOD_SLUG,
        finance_summary=finance_summary,
        anomaly_report=anomaly_report,
        evidence_package=evidence_package,
        database_path=args.memory_db,
        purpose="strategic_analysis",
    )
    ledger = build_evidence_ledger(
        finance_summary=finance_summary,
        anomaly_report=anomaly_report,
        evidence_package=evidence_package,
        risk_summary=risk_summary,
        period_slug=PERIOD_SLUG,
        historical_context=history.context,
    )
    client = OllamaClient(
        endpoint=args.endpoint,
        model=args.model,
        connect_timeout_seconds=args.connect_timeout,
        read_timeout_seconds=args.read_timeout,
        timeout_seconds=args.read_timeout,
        keep_alive=args.keep_alive,
        reasoning_enabled=True,
    )
    health = client.health()
    warm_up = None
    if health["available"] and args.warm_up:
        try:
            warm_up = client.warm_up()
        except Exception as exc:  # noqa: BLE001 - benchmark reports failure.
            warm_up = {"error": str(exc), "error_category": getattr(exc, "category", "warmup_error")}

    financial_prompt = build_financial_performance_prompt(
        evidence_ledger=ledger,
        finance_summary=finance_summary,
        anomaly_report=anomaly_report,
        period_slug=PERIOD_SLUG,
    )
    financial_alone = None
    if health["available"]:
        financial_alone = _run_stage_with_optional_resume(
            stage_id="financial_performance_alone",
            stage_name="Financial Performance Reasoning Alone",
            prompt=financial_prompt,
            client=client,
            validator=lambda text: validate_reasoning_stage_response(
                text,
                stage_id="financial_performance",
                evidence_ledger=ledger,
            ),
            resume=False,
            stage_timeout_seconds=args.stage_timeout,
        )

    state = ReasoningState(period_slug=PERIOD_SLUG, evidence_ledger=ledger)
    stage_results: list[ReasoningStageResult] = []
    report_generated = False
    report_paths: list[str] = []
    if health["available"]:
        stage1 = _run_stage_with_optional_resume(
            stage_id="financial_performance",
            stage_name="Financial Performance Reasoning",
            prompt=financial_prompt,
            client=client,
            validator=lambda text: validate_reasoning_stage_response(
                text,
                stage_id="financial_performance",
                evidence_ledger=ledger,
            ),
            resume=args.resume,
            stage_timeout_seconds=args.stage_timeout,
        )
        state.add_stage_result(stage1)
        stage_results.append(stage1)
        if stage1.accepted:
            historical_prompt = build_historical_operational_prompt(
                evidence_ledger=ledger,
                historical_context=history.context,
                state=state,
                period_slug=PERIOD_SLUG,
            )
            stage2 = _run_stage_with_optional_resume(
                stage_id="historical_operational",
                stage_name="Historical & Operational Reasoning",
                prompt=historical_prompt,
                client=client,
                validator=lambda text: validate_reasoning_stage_response(
                    text,
                    stage_id="historical_operational",
                    evidence_ledger=ledger,
                ),
                resume=args.resume,
                stage_timeout_seconds=args.stage_timeout,
            )
            state.add_stage_result(stage2)
            stage_results.append(stage2)
            if stage2.accepted:
                strategic_prompt = build_strategic_synthesis_prompt(
                    state=state,
                    finance_summary=finance_summary,
                    period_slug=PERIOD_SLUG,
                )
                stage3 = _run_stage_with_optional_resume(
                    stage_id="strategic_synthesis",
                    stage_name="Strategic Synthesis",
                    prompt=strategic_prompt,
                    client=client,
                    validator=lambda text: validate_strategic_synthesis_response(
                        text,
                        finance_summary=finance_summary,
                        anomaly_report=anomaly_report,
                        evidence_package=evidence_package,
                        risk_summary=risk_summary,
                        historical_context=history.context,
                        evidence_ledger=ledger,
                    ),
                    resume=args.resume,
                    response_format=strategic_analysis_json_schema(),
                    stage_timeout_seconds=args.stage_timeout,
                )
                state.add_stage_result(stage3)
                stage_results.append(stage3)

    accepted = bool(stage_results) and all(stage.accepted for stage in stage_results) and len(stage_results) == 3
    analysis = stage_results[-1].payload if accepted else _empty_analysis()
    errors = tuple(
        error
        for stage in stage_results
        if not stage.accepted
        for error in stage.validation_errors
    )
    if not health["available"]:
        errors = (str(health["error"]),)
    document = _analysis_document(
        period_slug=PERIOD_SLUG,
        report_period=str(finance_summary.get("report_period", PERIOD_SLUG)),
        ollama_available=bool(health["available"]),
        validation_status="accepted" if accepted else ("rejected" if health["available"] else "unavailable"),
        validation_errors=() if accepted else errors,
        analysis=analysis,
        historical_context=history.context,
        evidence_ledger=ledger,
        reasoning_state=state,
    )
    analysis_dir = PROJECT_ROOT / "outputs" / "analysis"
    save_json_artifact(document, analysis_dir / f"strategic_analysis_{PERIOD_SLUG}.json")
    save_json_artifact(state.to_dict(), analysis_dir / f"reasoning_state_{PERIOD_SLUG}.json")
    outputs = state.to_dict().get("reasoning_outputs", {})
    for stage_id, stem in (
        ("financial_performance", "financial_reasoning"),
        ("historical_operational", "historical_reasoning"),
        ("strategic_synthesis", "strategic_reasoning"),
    ):
        save_json_artifact(
            {"period_slug": PERIOD_SLUG, "stage_id": stage_id, "reasoning_output": outputs.get(stage_id, {})},
            analysis_dir / f"{stem}_{PERIOD_SLUG}.json",
        )

    if accepted:
        report_model = build_report_model(
            ReportInputBundle(
                period_slug=PERIOD_SLUG,
                finance_summary=finance_summary,
                kpi_summary=(),
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                strategic_analysis=document,
                source_files=(
                    f"outputs/calculations/finance_summary_{PERIOD_SLUG}.json",
                    f"outputs/calculations/kpi_summary_{PERIOD_SLUG}.json",
                    f"outputs/anomalies/anomaly_report_{PERIOD_SLUG}.json",
                    f"outputs/evidence/evidence_package_{PERIOD_SLUG}.json",
                    f"outputs/analysis/strategic_analysis_{PERIOD_SLUG}.json",
                ),
            )
        )
        report_dir = PROJECT_ROOT / "outputs" / "report"
        report_paths = [
            str(save_report_model(report_model, report_dir / f"report_model_{PERIOD_SLUG}.json")),
            str(save_report_html(report_model.to_dict(), report_dir / f"financial_report_{PERIOD_SLUG}.html")),
            str(render_report_pdf(report_model.to_dict(), report_dir / f"financial_report_{PERIOD_SLUG}.pdf")),
        ]
        report_generated = True

    profile = {
        "benchmark_id": "modular_reasoning_2026_12",
        "period_slug": PERIOD_SLUG,
        "ollama_health": health,
        "warm_up": warm_up,
        "configuration": {
            "endpoint": args.endpoint,
            "model": args.model,
            "connect_timeout_seconds": args.connect_timeout,
            "read_timeout_seconds": args.read_timeout,
            "stage_timeout_seconds": args.stage_timeout,
            "keep_alive": args.keep_alive,
            "resume": args.resume,
        },
        "baseline": {
            "monolithic_prompt_characters": MONOLITHIC_BASELINE_PROMPT_CHARS,
        },
        "financial_stage_alone": financial_alone.to_dict() if financial_alone else None,
        "full_modular_pipeline": {
            "accepted": accepted,
            "validation_errors": list(errors),
            "stage_results": [stage.to_dict() for stage in stage_results],
            "reasoning_state_path": str(analysis_dir / f"reasoning_state_{PERIOD_SLUG}.json"),
            "strategic_analysis_path": str(analysis_dir / f"strategic_analysis_{PERIOD_SLUG}.json"),
        },
        "report_generation": {
            "generated": report_generated,
            "paths": report_paths,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "health": health,
        "accepted": accepted,
        "report_generated": report_generated,
        "output": str(args.output),
        "runtime_seconds": profile["runtime_seconds"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
