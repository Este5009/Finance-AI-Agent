"""Pipeline profiling document builders for runtime optimization."""

from __future__ import annotations

from typing import Any

from finance_agent.orchestration.pipeline_models import PipelineRunResult


def _stage_context_size(stage: dict[str, Any]) -> int:
    """Return a stage context-size estimate from telemetry.

    Inputs: serialized stage result.
    Outputs: context character count, or zero when unavailable.
    Assumptions: context size is emitted only by LLM-capable stages.
    """

    telemetry = stage.get("telemetry", {})
    if not isinstance(telemetry, dict):
        return 0
    value = telemetry.get("context_characters", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _recommendations_for_profile(profile: dict[str, Any]) -> list[str]:
    """Create deterministic optimization recommendations from profile data.

    Inputs: partially built profile.
    Outputs: short recommendation strings.
    Assumptions: recommendations describe runtime mechanics, not finance logic.
    """

    recommendations: list[str] = []
    bottlenecks = profile.get("bottleneck_ranking", [])
    if bottlenecks:
        top = bottlenecks[0]
        recommendations.append(
            f"Focus first on {top['stage_name']} ({top['runtime_seconds']:.2f}s)."
        )
    for stage in profile.get("per_stage", []):
        telemetry = stage.get("telemetry", {})
        if telemetry.get("context_characters", 0) > 20_000:
            recommendations.append(
                f"Review context compaction for {stage['stage_name']}; prompt remains large."
            )
        if telemetry.get("generation_time_seconds", 0) > 60:
            recommendations.append(
                f"{stage['stage_name']} is generation-bound; preserve reasoning but keep summaries tight."
            )
    if not recommendations:
        recommendations.append("No single profiling bottleneck exceeded configured heuristics.")
    return recommendations


def build_pipeline_profile(result: PipelineRunResult) -> dict[str, Any]:
    """Build a JSON-compatible pipeline profile from one run result.

    Inputs: completed PipelineRunResult.
    Outputs: profile document with timings, context size, telemetry, and bottlenecks.
    Assumptions: profiling observes existing outputs and does not rerun stages.
    """

    result_dict = result.to_dict()
    stages = result_dict["stages"]
    per_stage = [
        {
            "stage_name": stage["stage_name"],
            "display_name": stage["display_name"],
            "success": stage["success"],
            "skipped": stage["skipped"],
            "runtime_seconds": stage["runtime_seconds"],
            "context_characters": _stage_context_size(stage),
            "token_estimate": stage.get("telemetry", {}).get(
                "context_token_estimate",
                0,
            )
            if isinstance(stage.get("telemetry"), dict)
            else 0,
            "telemetry": stage.get("telemetry", {}),
        }
        for stage in stages
    ]
    bottlenecks = sorted(
        (
            {
                "stage_name": stage["stage_name"],
                "runtime_seconds": float(stage["runtime_seconds"]),
            }
            for stage in per_stage
            if not stage["skipped"]
        ),
        key=lambda item: item["runtime_seconds"],
        reverse=True,
    )
    profile = {
        "profile_id": "PIPELINE-PROFILE",
        "success": result.success,
        "cache_hit": result.cache_hit,
        "runtime_summary": result.runtime_summary.to_dict(),
        "runtime_config": {
            "max_planner_anomalies": result.config.max_planner_anomalies,
            "compact_context": result.config.compact_context,
            "deduplicate_context": result.config.deduplicate_context,
            "effective_ollama_models": result.config.effective_ollama_models(),
        },
        "per_stage": per_stage,
        "total_context_characters": sum(stage["context_characters"] for stage in per_stage),
        "total_token_estimate": sum(int(stage.get("token_estimate") or 0) for stage in per_stage),
        "bottleneck_ranking": bottlenecks,
    }
    profile["optimization_recommendations"] = _recommendations_for_profile(profile)
    return profile
