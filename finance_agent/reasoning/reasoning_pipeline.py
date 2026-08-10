"""Modular multi-stage Ollama reasoning pipeline."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from finance_agent.analysis.analysis_models import StrategicAnalysisResult
from finance_agent.analysis.strategic_analysis import (
    build_deterministic_strategic_synthesis,
    build_evidence_ledger,
    estimate_tokens_from_text,
    strategic_analysis_json_schema,
    validate_evidence_bound_claims,
    validate_strategic_analysis_response,
    validate_user_facing_spanish,
)
from finance_agent.llm.ollama_client import OllamaError
from finance_agent.reasoning.fact_registry import FactRegistry
from finance_agent.reasoning.reasoning_models import (
    ReasoningStageResult,
    ReasoningValidationResult,
)
from finance_agent.reasoning.reasoning_state import ReasoningState


STAGE_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "financial_performance": (
        "claims",
        "risks",
        "opportunities",
        "open_questions",
    ),
    "historical_operational": (
        "claims",
        "risks",
        "opportunities",
        "open_questions",
    ),
}

STAGE_TOP_LEVEL_ALIASES = {
    "validated_financial_claims": "claims",
    "financial_claims": "claims",
    "validated_historical_claims": "claims",
    "historical_claims": "claims",
    "trend_observations": "claims",
    "recommendation_effectiveness": "claims",
    "identified_financial_risks": "risks",
    "identified_risks": "risks",
    "persistent_risks": "risks",
    "financial_opportunities": "opportunities",
    "identified_opportunities": "opportunities",
    "questions": "open_questions",
}
STAGE_WRAPPER_KEYS = {
    "financial_reasoning",
    "historical_reasoning",
    "reasoning",
    "result",
    "output",
}
CLAIM_TYPES = {"fact", "interpretation", "hypothesis"}

FINAL_NARRATIVE_FIELDS: tuple[str, ...] = (
    "executive_summary",
    "executive_analysis",
    "key_findings",
    "root_causes",
    "financial_health_analysis",
    "kpi_analysis",
    "historical_summary",
    "historical_trend_analysis",
    "department_analysis",
    "anomaly_analysis",
    "recommendation_follow_up_analysis",
    "longitudinal_risk_analysis",
    "strategic_recommendations",
    "strategic_priorities",
    "missing_information",
    "recommendations",
    "reasoning_summary",
)


def create_modular_strategic_analysis(
    *,
    client: Any,
    evidence_package: dict[str, Any],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    risk_summary: dict[str, Any],
    period_slug: str,
    historical_context: dict[str, Any] | None = None,
    compact_context: bool = True,
    deduplicate_context: bool = True,
    stage_timeout_seconds: float | None = None,
    force_degraded_deterministic: bool = False,
    degraded_reason: tuple[str, ...] = (),
) -> StrategicAnalysisResult:
    """Run the three-stage reasoning pipeline and return Step-9-compatible output.

    Inputs: Ollama client plus processed finance, anomaly, risk, evidence, and
    optional compact historical context.
    Outputs: StrategicAnalysisResult whose document can feed existing report
    generation, memory storage, and UI download paths.
    Assumptions: Python remains the source of truth; every stage output is
    rejected unless strict JSON, Spanish prose, and evidence grounding validate.
    """

    del compact_context, deduplicate_context  # Modular prompts are compact by construction.
    started = time.perf_counter()
    available = client.is_available()
    evidence_ledger = build_evidence_ledger(
        finance_summary=finance_summary,
        anomaly_report=anomaly_report,
        evidence_package=evidence_package,
        risk_summary=risk_summary,
        period_slug=period_slug,
        historical_context=historical_context,
    )
    fact_registry = FactRegistry.from_evidence_ledger(evidence_ledger)
    state = ReasoningState(
        period_slug=period_slug,
        evidence_ledger=evidence_ledger,
        fact_registry=fact_registry.to_dict(),
    )
    telemetry: dict[str, Any] = {
        "reasoning_pipeline": "modular_multi_stage",
        "stage_count": 3,
        "monolithic_prompt_baseline_characters": None,
    }
    model_name = str(getattr(client, "model", "") or "unknown")
    if force_degraded_deterministic:
        return _degraded_result(
            finance_summary=finance_summary,
            anomaly_report=anomaly_report,
            evidence_package=evidence_package,
            period_slug=period_slug,
            historical_context=historical_context,
            evidence_ledger=evidence_ledger,
            state=state,
            telemetry=telemetry,
            started=started,
            reason=degraded_reason or ("Degraded deterministic mode was explicitly selected.",),
            ollama_available=available,
            model_name=model_name,
        )
    if not available:
        document = _analysis_document(
            period_slug=period_slug,
            report_period=str(finance_summary.get("report_period", period_slug)),
            ollama_available=False,
            validation_status="unavailable",
            validation_errors=("Ollama is unavailable.",),
            analysis=_empty_analysis(),
            historical_context=historical_context,
            evidence_ledger=evidence_ledger,
            reasoning_state=state,
            model_name=model_name,
        )
        return StrategicAnalysisResult(
            analysis_document=document,
            accepted=False,
            validation_errors=("Ollama is unavailable.",),
            telemetry={**telemetry, "total_stage_time_seconds": time.perf_counter() - started},
        )

    try:
        financial_result = _run_structured_stage(
            client=client,
            stage_id="financial_performance",
            stage_name="Financial Performance Reasoning",
            prompt=build_financial_performance_prompt(
                evidence_ledger=evidence_ledger,
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                period_slug=period_slug,
                fact_registry=fact_registry,
            ),
            validator=lambda text: validate_reasoning_stage_response(
                text,
                stage_id="financial_performance",
                evidence_ledger=evidence_ledger,
                fact_registry=fact_registry,
            ),
            response_format=reasoning_stage_json_schema("financial_performance"),
            stage_timeout_seconds=stage_timeout_seconds,
            fact_registry=fact_registry,
        )
        state.add_stage_result(financial_result)
        if not financial_result.accepted:
            return _degraded_result(
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                period_slug=period_slug,
                historical_context=historical_context,
                evidence_ledger=evidence_ledger,
                state=state,
                telemetry=telemetry,
                started=started,
                reason=financial_result.validation_errors,
                ollama_available=True,
                model_name=model_name,
            )

        historical_result = _run_structured_stage(
            client=client,
            stage_id="historical_operational",
            stage_name="Historical & Operational Reasoning",
            prompt=build_historical_operational_prompt(
                evidence_ledger=evidence_ledger,
                historical_context=historical_context,
                state=state,
                period_slug=period_slug,
                fact_registry=fact_registry,
            ),
            validator=lambda text: validate_reasoning_stage_response(
                text,
                stage_id="historical_operational",
                evidence_ledger=evidence_ledger,
                fact_registry=fact_registry,
            ),
            response_format=reasoning_stage_json_schema("historical_operational"),
            stage_timeout_seconds=stage_timeout_seconds,
            fact_registry=fact_registry,
        )
        state.add_stage_result(historical_result)
        if not historical_result.accepted:
            return _degraded_result(
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                period_slug=period_slug,
                historical_context=historical_context,
                evidence_ledger=evidence_ledger,
                state=state,
                telemetry=telemetry,
                started=started,
                reason=historical_result.validation_errors,
                ollama_available=True,
                model_name=model_name,
            )

        strategic_prompt = build_strategic_synthesis_prompt(
            state=state,
            finance_summary=finance_summary,
            period_slug=period_slug,
            fact_registry=fact_registry,
        )
        strategic_result = _run_structured_stage(
            client=client,
            stage_id="strategic_synthesis",
            stage_name="Strategic Synthesis",
            prompt=strategic_prompt,
            validator=lambda text: validate_strategic_synthesis_response(
                text,
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                risk_summary=risk_summary,
                historical_context=historical_context,
                evidence_ledger=evidence_ledger,
                fact_registry=fact_registry,
            ),
            response_format=strategic_synthesis_fact_json_schema(),
            stage_timeout_seconds=stage_timeout_seconds,
            fact_registry=fact_registry,
        )
        state.add_stage_result(strategic_result)
        if not strategic_result.accepted:
            strategic_result = _run_strategic_recovery_attempt(
                client=client,
                state=state,
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                risk_summary=risk_summary,
                historical_context=historical_context,
                evidence_ledger=evidence_ledger,
                fact_registry=fact_registry,
                period_slug=period_slug,
                previous_result=strategic_result,
                stage_timeout_seconds=stage_timeout_seconds,
                attempt_label="targeted_repair",
            )
            state.add_stage_result(strategic_result)
        if not strategic_result.accepted:
            strategic_result = _run_strategic_recovery_attempt(
                client=client,
                state=state,
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                risk_summary=risk_summary,
                historical_context=historical_context,
                evidence_ledger=evidence_ledger,
                fact_registry=fact_registry,
                period_slug=period_slug,
                previous_result=strategic_result,
                stage_timeout_seconds=stage_timeout_seconds,
                attempt_label="constrained_regeneration",
            )
            state.add_stage_result(strategic_result)
        if not strategic_result.accepted:
            strategic_result = _run_strategic_recovery_attempt(
                client=client,
                state=state,
                finance_summary=finance_summary,
                anomaly_report=anomaly_report,
                evidence_package=evidence_package,
                risk_summary=risk_summary,
                historical_context=historical_context,
                evidence_ledger=evidence_ledger,
                fact_registry=fact_registry,
                period_slug=period_slug,
                previous_result=strategic_result,
                stage_timeout_seconds=stage_timeout_seconds,
                attempt_label="final_constrained_regeneration",
            )
            state.add_stage_result(strategic_result)
    except OllamaError as exc:
        return _degraded_result(
            finance_summary=finance_summary,
            anomaly_report=anomaly_report,
            evidence_package=evidence_package,
            period_slug=period_slug,
            historical_context=historical_context,
            evidence_ledger=evidence_ledger,
            state=state,
            telemetry=telemetry,
            started=started,
            reason=(str(exc),),
            ollama_available=True,
            model_name=model_name,
        )

    accepted = state.stage_results[-1].accepted if state.stage_results else False
    analysis = state.stage_results[-1].payload if accepted else _empty_analysis()
    errors = state.stage_results[-1].validation_errors if state.stage_results else ("No reasoning stages ran.",)
    sanitized_recovery = analysis.get("_strategic_recovery") if isinstance(analysis, dict) else None
    validation_status = "accepted" if accepted else "rejected"
    validation_errors: tuple[str, ...] = () if accepted else errors
    if accepted and isinstance(sanitized_recovery, dict) and sanitized_recovery:
        validation_status = "sanitized"
        validation_errors = tuple(
            str(error)
            for error in sanitized_recovery.get("original_validation_errors", [])
        )
    document = _analysis_document(
        period_slug=period_slug,
        report_period=str(finance_summary.get("report_period", period_slug)),
        ollama_available=True,
        validation_status=validation_status,
        validation_errors=validation_errors,
        analysis=analysis,
        historical_context=historical_context,
        evidence_ledger=evidence_ledger,
        reasoning_state=state,
        model_name=model_name,
    )
    if isinstance(sanitized_recovery, dict) and sanitized_recovery:
        document["strategic_recovery"] = {
            **sanitized_recovery,
            "outcome": "ollama_repaired_validated",
            "source_label": "Análisis reparado y validado",
            "ai_mode_enabled": True,
            "degraded_mode": False,
            "attempts": _strategic_attempt_labels(state),
        }
    elif accepted:
        document["strategic_recovery"] = {
            "outcome": "ollama_validated",
            "source_label": "Análisis estratégico validado por IA",
            "ai_mode_enabled": True,
            "degraded_mode": False,
            "attempts": _strategic_attempt_labels(state),
        }
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=accepted,
        validation_errors=validation_errors,
        telemetry={
            **telemetry,
            "total_stage_time_seconds": time.perf_counter() - started,
            "stage_telemetry": [stage.telemetry for stage in state.stage_results],
            "reasoning_state_claim_count": len(state.validated_claims),
            "reasoning_state_risk_count": len(state.risks),
        },
    )


def build_financial_performance_prompt(
    *,
    evidence_ledger: dict[str, Any],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    period_slug: str,
    fact_registry: FactRegistry | None = None,
) -> str:
    """Build the Stage 1 financial-performance prompt.

    Inputs: ledger, current finance summary, anomaly report, and period slug.
    Outputs: compact strict-JSON prompt for current-performance reasoning.
    Assumptions: prompt includes only current financial facts and top anomalies.
    """

    del fact_registry
    facts = _fact_cards_for_prefixes(evidence_ledger, ("finance.", "anomaly."), limit=25)
    context = {
        "period_slug": period_slug,
        "objective": "Qué está ocurriendo financieramente en el periodo actual.",
        "facts": facts,
        "report_period": finance_summary.get("report_period"),
        "anomaly_count": anomaly_report.get("total_anomalies"),
    }
    return _stage_prompt(
        stage_name="Financial Performance Reasoning",
        schema=_financial_stage_schema_text(),
        context=context,
    )


def build_historical_operational_prompt(
    *,
    evidence_ledger: dict[str, Any],
    historical_context: dict[str, Any] | None,
    state: ReasoningState,
    period_slug: str,
    fact_registry: FactRegistry | None = None,
) -> str:
    """Build the Stage 2 historical/operational prompt.

    Inputs: ledger, compact historical context, and accepted Stage 1 state.
    Outputs: strict-JSON prompt focused on trends and persistence.
    Assumptions: only historical facts plus Stage 1 validated reasoning are sent.
    """

    del fact_registry
    facts = _fact_cards_for_prefixes(evidence_ledger, ("history.", "anomaly.", "finance."), limit=25)
    context = {
        "period_slug": period_slug,
        "objective": "Cómo evolucionaron los riesgos y avances respecto de periodos previos.",
        "historical_facts": facts,
        "validated_stage_1": {
            "claims": _llm_safe_reasoning_items(state.validated_claims[:8]),
            "risks": _llm_safe_reasoning_items(state.risks[:8]),
            "opportunities": _llm_safe_reasoning_items(state.opportunities[:6]),
            "open_questions": _llm_safe_reasoning_items(state.open_questions[:6]),
        },
        "history_summary": (historical_context or {}).get("summary", {})
        if isinstance(historical_context, dict)
        else {},
    }
    return _stage_prompt(
        stage_name="Historical & Operational Reasoning",
        schema=_historical_stage_schema_text(),
        context=context,
    )


def build_strategic_synthesis_prompt(
    *,
    state: ReasoningState,
    finance_summary: dict[str, Any],
    period_slug: str,
    fact_registry: FactRegistry | None = None,
) -> str:
    """Build the Stage 3 strategic-synthesis prompt.

    Inputs: validated reasoning state, current goals/period metadata.
    Outputs: strict-JSON prompt compatible with existing strategic-analysis schema.
    Assumptions: this prompt intentionally excludes the full evidence ledger and
    relies on validated Stage 1/2 outputs for facts and citations.
    """

    context = {
        "period_slug": period_slug,
        "report_period": finance_summary.get("report_period"),
        "objective": "Qué debe hacer la dirección universitaria.",
        "validated_reasoning_state": state.to_prompt_context(),
        "facts": _stage3_fact_cards(state, fact_registry, limit=20),
        "rules": (
            "Usa solo afirmaciones validadas en validated_reasoning_state.",
            "Usa solo los hechos adicionales incluidos en facts cuando necesites valores deterministas.",
            "No calcules ni inventes valores; si falta evidencia, decláralo como información faltante.",
        ),
    }
    return (
        _stage_prompt(
            stage_name="Strategic Synthesis",
            schema=(
                "Return strategic analysis JSON without evidence IDs. "
                "Narrative fields are plain Spanish strings. "
                "key_findings, root_causes, strategic_priorities and missing_information are lists of strings. "
                "strategic_recommendations items use priority, action, rationale, supporting_evidence, "
                "expected_impact, confidence. Never include evidence_ids or narrative_evidence."
            ),
            context=context,
        )
        + "\nStage 3 MUST NOT ask for or assume the full evidence ledger."
    )


def build_strategic_recovery_prompt(
    *,
    state: ReasoningState,
    finance_summary: dict[str, Any],
    period_slug: str,
    fact_registry: FactRegistry | None,
    validation_errors: tuple[str, ...],
    original_payload: dict[str, Any],
    attempt_label: str,
) -> str:
    """Build one bounded strategic repair/regeneration prompt.

    Inputs: accepted prior reasoning state, exact validation errors, the
    previous failed payload, and a recovery-attempt label.
    Outputs: compact Stage-3-compatible prompt.
    Assumptions: Ollama may rewrite reasoning, but Python still validates all
    numbers, Spanish prose, and grounding after the attempt.
    """

    base_context = {
        "period_slug": period_slug,
        "report_period": finance_summary.get("report_period"),
        "attempt": attempt_label,
        "validation_errors_to_fix": list(validation_errors)[:12],
        "previous_output": original_payload,
        "validated_reasoning_state": state.to_prompt_context(),
        "facts": _stage3_fact_cards(state, fact_registry, limit=16 if attempt_label == "targeted_repair" else 10),
        "rules": (
            "Conserva solamente contenido respaldado por los hechos suministrados.",
            "Elimina o suaviza afirmaciones rechazadas; no agregues cifras nuevas.",
            "Devuelve JSON estricto con el mismo esquema de Strategic Synthesis.",
            "No conviertas referencias del sistema en reglas institucionales.",
        ),
    }
    return _stage_prompt(
        stage_name=f"Strategic Synthesis Recovery - {attempt_label}",
        schema=(
            "Return strategic analysis JSON without evidence IDs. "
            "Narrative fields are plain Spanish strings. "
            "key_findings, root_causes, strategic_priorities and missing_information are lists of strings. "
            "strategic_recommendations items use priority, action, rationale, supporting_evidence, "
            "expected_impact, confidence."
        ),
        context=base_context,
    )


def _run_strategic_recovery_attempt(
    *,
    client: Any,
    state: ReasoningState,
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any],
    risk_summary: dict[str, Any],
    historical_context: dict[str, Any] | None,
    evidence_ledger: dict[str, Any],
    fact_registry: FactRegistry,
    period_slug: str,
    previous_result: ReasoningStageResult,
    stage_timeout_seconds: float | None,
    attempt_label: str,
) -> ReasoningStageResult:
    """Run one bounded Stage-3 recovery attempt.

    Inputs: client, current reasoning state, deterministic contexts, prior
    failed result, timeout budget, and attempt label.
    Outputs: validated or rejected stage result.
    Assumptions: this function never loops; callers decide the bounded sequence.
    """

    return _run_structured_stage(
        client=client,
        stage_id="strategic_synthesis",
        stage_name=f"Strategic Synthesis {attempt_label.replace('_', ' ').title()}",
        prompt=build_strategic_recovery_prompt(
            state=state,
            finance_summary=finance_summary,
            period_slug=period_slug,
            fact_registry=fact_registry,
            validation_errors=previous_result.validation_errors,
            original_payload=previous_result.payload,
            attempt_label=attempt_label,
        ),
        validator=lambda text: validate_strategic_synthesis_response(
            text,
            finance_summary=finance_summary,
            anomaly_report=anomaly_report,
            evidence_package=evidence_package,
            risk_summary=risk_summary,
            historical_context=historical_context,
            evidence_ledger=evidence_ledger,
            fact_registry=fact_registry,
        ),
        response_format=strategic_synthesis_fact_json_schema(),
        stage_timeout_seconds=stage_timeout_seconds,
        fact_registry=fact_registry,
    )


def _fact_cards_for_prefixes(
    evidence_ledger: dict[str, Any],
    prefixes: tuple[str, ...],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return compact deterministic fact cards for Ollama.

    Inputs: evidence ledger, evidence-ID prefixes, and maximum count.
    Outputs: fact cards with exact Python-formatted values but no internal IDs.
    Assumptions: Python selects relevance; Ollama reasons from these facts only.
    """

    cards: list[dict[str, Any]] = []
    for fact in evidence_ledger.get("facts", []):
        if not isinstance(fact, dict):
            continue
        evidence_id = str(fact.get("evidence_id", ""))
        if not evidence_id.startswith(prefixes):
            continue
        value = fact.get("display_value")
        raw_value = fact.get("raw_value")
        cards.append(
            {
                "metric": fact.get("metric") or fact.get("field"),
                "value": value if value not in {None, ""} else raw_value,
                "raw_value": raw_value if isinstance(raw_value, (int, float)) else None,
                "unit": fact.get("unit"),
                "period": fact.get("period"),
                "entity": fact.get("entity"),
                "category": fact.get("category"),
                "meaning": fact.get("claim") or fact.get("metric") or fact.get("field"),
            }
        )
        if len(cards) >= limit:
            break
    return cards


def _stage3_fact_cards(
    state: ReasoningState,
    registry: FactRegistry | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return a small set of synthesis facts from prior-stage referenced facts.

    Inputs: reasoning state, optional fact registry and maximum count.
    Outputs: fact cards with Python-formatted values.
    Assumptions: the active simplified runtime does not require model citations;
    this only gives Stage 3 enough deterministic context for synthesis.
    """

    if registry is None:
        return []
    wanted_evidence = {
        evidence_id
        for collection in (state.validated_claims, state.risks, state.opportunities, state.open_questions)
        for item in collection
        for evidence_id in item.get("resolved_evidence_ids", [])
        if isinstance(evidence_id, str)
    }
    cards: list[dict[str, Any]] = []
    for fact in registry.facts:
        if wanted_evidence and not set(fact.evidence_ids).intersection(wanted_evidence):
            continue
        cards.append(
            {
                "metric": fact.metric_name,
                "value": fact.display_value,
                "raw_value": fact.raw_value if isinstance(fact.raw_value, (int, float)) else None,
                "unit": fact.unit,
                "period": fact.period,
                "entity": fact.entity,
                "category": fact.source_metadata.get("category"),
                "meaning": fact.source_metadata.get("claim") or fact.metric_name,
            }
        )
        if len(cards) >= limit:
            break
    return cards


def _llm_safe_reasoning_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip Python-only metadata from prior-stage prompt context.

    Inputs: validated internal reasoning items.
    Outputs: items containing narrative, confidence/type, and stage provenance.
    Assumptions: internal evidence references remain Python-owned.
    """

    safe: list[dict[str, Any]] = []
    for item in items:
        safe.append(
            {
                key: value
                for key, value in item.items()
                if key
                in {
                    "text",
                    "confidence",
                    "claim_type",
                    "stage_id",
                }
            }
        )
    return safe


def validate_reasoning_stage_response(
    response_text: str,
    *,
    stage_id: str,
    evidence_ledger: dict[str, Any],
    fact_registry: FactRegistry | None = None,
) -> ReasoningValidationResult:
    """Validate Stage 1 or Stage 2 strict JSON output.

    Inputs: raw response text, stage ID, and full Python evidence ledger.
    Outputs: validation result with cleaned payload or errors.
    Assumptions: stage schemas are intentionally smaller than final report schema.
    """

    try:
        payload = json.loads(response_text.strip())
    except (AttributeError, json.JSONDecodeError):
        return ReasoningValidationResult(False, None, ("response is not strict JSON",))
    if not isinstance(payload, dict):
        return ReasoningValidationResult(False, None, ("response root must be an object",))
    payload, normalizations = normalize_reasoning_stage_payload(payload)
    required = set(_stage_required_fields(stage_id))
    if set(payload) != required:
        return ReasoningValidationResult(
            False,
            None,
            (
                "schema: "
                f"{stage_id} must contain exactly {sorted(required)}; received {sorted(payload)}",
            ),
        )

    del fact_registry
    errors: list[str] = []
    for field_name in STAGE_TEXT_FIELDS[stage_id]:
        _validate_reasoning_items(
            field_name,
            payload.get(field_name),
            evidence_ledger,
            errors,
        )
    if errors:
        return ReasoningValidationResult(False, None, tuple(dict.fromkeys(errors)))

    cleaned = {
        key: _clean_reasoning_items(value, field_name=key)
        for key, value in payload.items()
    }
    if normalizations:
        cleaned["_schema_normalizations"] = normalizations
    cleaned["_selected_fact_summary"] = _selected_fact_summary(evidence_ledger, stage_id)
    return ReasoningValidationResult(True, cleaned, ())


def _convert_stage3_supporting_facts_payload(
    payload: Any,
    *,
    registry: FactRegistry,
    allowed_placeholders: set[str],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Convert Stage 3 LLM fact-support output into internal analysis JSON.

    Inputs: raw parsed payload, fact registry, and Stage 3 allowed placeholders.
    Outputs: converted analysis payload, validation errors, and support audit.
    Assumptions: model output must not contain evidence IDs; Python resolves
    them exclusively from referenced facts.
    """

    errors: list[str] = []
    audit: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return {}, ["response root must be an object"], audit
    if _contains_key(payload, "evidence_ids") or _contains_key(payload, "narrative_evidence"):
        return {}, ["model output must not contain evidence_ids or narrative_evidence"], audit

    required = {
        "executive_summary",
        "key_findings",
        "root_causes",
        "financial_health_analysis",
        "kpi_analysis",
        "historical_summary",
        "historical_trend_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "strategic_recommendations",
        "strategic_priorities",
        "missing_information",
        "confidence",
        "reasoning_summary",
    }
    if set(payload) != required:
        return {}, [f"schema: strategic_synthesis must contain exactly {sorted(required)}; received {sorted(payload)}"], audit

    converted: dict[str, Any] = {"narrative_evidence": {}}
    for field_name in (
        "executive_summary",
        "financial_health_analysis",
        "kpi_analysis",
        "historical_summary",
        "historical_trend_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "reasoning_summary",
    ):
        text, evidence_ids, item_errors, item_audit = _resolve_stage3_block(
            field_name,
            payload.get(field_name),
            registry=registry,
            allowed_placeholders=allowed_placeholders,
            require_support=True,
        )
        errors.extend(item_errors)
        audit.extend(item_audit)
        converted[field_name] = {"text": text, "evidence_ids": evidence_ids}
        converted["narrative_evidence"][field_name] = evidence_ids

    for field_name in ("key_findings", "root_causes", "strategic_priorities", "missing_information"):
        values = payload.get(field_name)
        if not isinstance(values, list):
            errors.append(f"{field_name} must be a list")
            values = []
        texts: list[str] = []
        evidence_ids: set[str] = set()
        support_sets: list[tuple[str, ...]] = []
        for index, item in enumerate(values):
            text, ids, item_errors, item_audit = _resolve_stage3_block(
                f"{field_name}[{index}]",
                item,
                registry=registry,
                allowed_placeholders=allowed_placeholders,
                require_support=field_name != "missing_information",
            )
            errors.extend(item_errors)
            audit.extend(item_audit)
            if text:
                texts.append(text)
            evidence_ids.update(ids)
            if isinstance(item, dict):
                support_sets.append(tuple(sorted(str(value) for value in item.get("supporting_facts", []))))
        if len(support_sets) > 2 and len(set(support_sets)) == 1 and support_sets[0]:
            errors.append(f"{field_name} repeats identical supporting_facts across unrelated items")
        converted[field_name] = texts
        converted["narrative_evidence"][field_name] = sorted(evidence_ids)

    recommendations = payload.get("strategic_recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        errors.append("strategic_recommendations must be a non-empty list")
        recommendations = []
    converted_recommendations: list[dict[str, Any]] = []
    for index, recommendation in enumerate(recommendations):
        if not isinstance(recommendation, dict):
            errors.append(f"strategic_recommendations[{index}] must be an object")
            continue
        expected_keys = {
            "priority",
            "text",
            "rationale",
            "expected_effect",
            "measurement_plan",
            "supporting_facts",
            "confidence",
        }
        if set(recommendation) != expected_keys:
            errors.append(
                f"schema: strategic_recommendations[{index}] must contain exactly {sorted(expected_keys)}"
            )
            continue
        text = str(recommendation.get("text") or "")
        support_errors, resolved_facts = _validate_item_support(
            prefix=f"strategic_recommendations[{index}]",
            field_name="strategic_recommendations",
            item={"text": text, "supporting_facts": recommendation.get("supporting_facts", [])},
            text=" ".join(
                str(recommendation.get(part) or "")
                for part in ("text", "rationale", "expected_effect", "measurement_plan")
            ),
            registry=registry,
            allowed_placeholders=allowed_placeholders,
        )
        errors.extend(support_errors)
        resolved_ids = sorted({evidence_id for fact in resolved_facts for evidence_id in fact.evidence_ids})
        audit.append(
            {
                "path": f"strategic_recommendations[{index}]",
                "supporting_facts": list(recommendation.get("supporting_facts", [])),
                "resolved_evidence_ids": resolved_ids,
                "fact_metadata": [_support_fact_metadata(fact) for fact in resolved_facts],
            }
        )
        converted_recommendations.append(
            {
                "priority": recommendation["priority"],
                "action": recommendation["text"],
                "rationale": recommendation["rationale"],
                "supporting_evidence": ", ".join(str(item) for item in recommendation.get("supporting_facts", [])),
                "expected_impact": (
                    f"{recommendation['expected_effect']} Plan de medición: {recommendation['measurement_plan']}"
                ),
                "evidence_ids": resolved_ids,
                "confidence": recommendation["confidence"],
            }
        )
    converted["strategic_recommendations"] = converted_recommendations
    converted["confidence"] = payload.get("confidence")
    return converted, errors, audit


def _resolve_stage3_block(
    path: str,
    block: Any,
    *,
    registry: FactRegistry,
    allowed_placeholders: set[str],
    require_support: bool,
) -> tuple[str, list[str], list[str], list[dict[str, Any]]]:
    """Resolve one Stage 3 narrative block to internal evidence IDs.

    Inputs: path, raw block, registry, allowlist and support requirement.
    Outputs: text, resolved evidence IDs, errors, and audit records.
    Assumptions: evidence IDs are created only from referenced facts.
    """

    if not isinstance(block, dict):
        return "", [], [f"{path} must be an object"], []
    if set(block) != {"text", "supporting_facts"}:
        return "", [], [f"schema: {path} must contain exactly ['supporting_facts', 'text']"], []
    text = str(block.get("text") or "")
    support_errors, resolved_facts = _validate_item_support(
        prefix=path,
        field_name="claims" if require_support else "open_questions",
        item={"text": text, "supporting_facts": block.get("supporting_facts", []), "claim_type": "interpretation"},
        text=text,
        registry=registry,
        allowed_placeholders=allowed_placeholders,
    )
    resolved_ids = sorted({evidence_id for fact in resolved_facts for evidence_id in fact.evidence_ids})
    return (
        text,
        resolved_ids,
        support_errors,
        [
            {
                "path": path,
                "supporting_facts": list(block.get("supporting_facts", [])),
                "resolved_evidence_ids": resolved_ids,
                "fact_metadata": [_support_fact_metadata(fact) for fact in resolved_facts],
            }
        ],
    )


def _contains_key(value: Any, key_name: str) -> bool:
    """Return whether a nested payload contains a forbidden key.

    Inputs: JSON-like value and key name.
    Outputs: True when found anywhere in dictionaries.
    Assumptions: used to reject model-supplied internal evidence IDs.
    """

    if isinstance(value, dict):
        return key_name in value or any(_contains_key(item, key_name) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key_name) for item in value)
    return False


def _convert_stage3_plain_payload(
    response_text: str,
    evidence_ledger: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Convert simplified Stage 3 JSON into the internal analysis schema.

    Inputs: raw Stage 3 response and deterministic evidence ledger.
    Outputs: internal strategic-analysis payload and schema errors.
    Assumptions: Ollama writes only narrative/recommendation text; Python
    attaches evidence IDs from the selected ledger facts for validation/reporting.
    """

    try:
        payload = json.loads(response_text.strip())
    except (AttributeError, json.JSONDecodeError):
        return {}, ("response is not strict JSON",)
    if not isinstance(payload, dict):
        return {}, ("response root must be an object",)
    if _contains_key(payload, "evidence_ids") or _contains_key(payload, "narrative_evidence"):
        return {}, ("model output must not contain evidence_ids or narrative_evidence",)

    required = {
        "executive_summary",
        "key_findings",
        "root_causes",
        "financial_health_analysis",
        "kpi_analysis",
        "historical_summary",
        "historical_trend_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "strategic_recommendations",
        "strategic_priorities",
        "missing_information",
        "confidence",
        "reasoning_summary",
    }
    if set(payload) != required:
        return {}, (f"schema: strategic_synthesis must contain exactly {sorted(required)}; received {sorted(payload)}",)

    converted: dict[str, Any] = {"narrative_evidence": {}}
    block_fields = (
        "executive_summary",
        "financial_health_analysis",
        "kpi_analysis",
        "historical_summary",
        "historical_trend_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "reasoning_summary",
    )
    for field_name in block_fields:
        text = _plain_text(payload.get(field_name))
        evidence_ids = _section_evidence_ids(evidence_ledger, field_name)
        converted[field_name] = {"text": text, "evidence_ids": evidence_ids}
        converted["narrative_evidence"][field_name] = evidence_ids
    for field_name in ("key_findings", "root_causes", "strategic_priorities", "missing_information"):
        values = payload.get(field_name)
        if not isinstance(values, list):
            return {}, (f"{field_name} must be a list",)
        converted[field_name] = [_plain_text(item) for item in values]
        converted["narrative_evidence"][field_name] = _section_evidence_ids(evidence_ledger, field_name)
    recommendations = payload.get("strategic_recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        return {}, ("strategic_recommendations must be a non-empty list",)
    converted_recommendations: list[dict[str, Any]] = []
    for index, recommendation in enumerate(recommendations):
        if not isinstance(recommendation, dict):
            return {}, (f"strategic_recommendations[{index}] must be an object",)
        expected = {"priority", "action", "rationale", "supporting_evidence", "expected_impact", "confidence"}
        if set(recommendation) != expected:
            return {}, (f"schema: strategic_recommendations[{index}] must contain exactly {sorted(expected)}",)
        converted_recommendations.append(
            {
                "priority": recommendation["priority"],
                "action": recommendation["action"],
                "rationale": recommendation["rationale"],
                "supporting_evidence": recommendation["supporting_evidence"],
                "expected_impact": recommendation["expected_impact"],
                "evidence_ids": _section_evidence_ids(evidence_ledger, "strategic_recommendations"),
                "confidence": recommendation["confidence"],
            }
        )
    converted["strategic_recommendations"] = converted_recommendations
    converted["confidence"] = payload.get("confidence")
    return converted, ()


def _default_evidence_ids(evidence_ledger: dict[str, Any]) -> list[str]:
    """Return deterministic evidence IDs for internal validation only.

    Inputs: evidence ledger.
    Outputs: up to eight source IDs selected by Python.
    Assumptions: IDs are never exposed to the model and are only used to satisfy
    downstream report/source-reference contracts.
    """

    ids = [
        str(fact.get("evidence_id"))
        for fact in evidence_ledger.get("facts", [])
        if isinstance(fact, dict) and fact.get("evidence_id")
    ]
    return ids[:8] or ["python.processed.outputs"]


def _section_evidence_ids(evidence_ledger: dict[str, Any], section: str) -> list[str]:
    """Return Python-selected evidence IDs appropriate for a report section.

    Inputs: evidence ledger and strategic-analysis section name.
    Outputs: bounded evidence IDs whose ledger facts declare section support.
    Assumptions: source attribution is deterministic and never model-generated.
    """

    ids = [
        str(fact.get("evidence_id"))
        for fact in evidence_ledger.get("facts", [])
        if isinstance(fact, dict)
        and fact.get("evidence_id")
        and section in set(fact.get("supports", []) or [])
    ]
    if ids:
        return ids[:8]
    prefix_map = {
        "department_analysis": ("finance.department.",),
        "anomaly_analysis": ("anomaly.",),
        "missing_information": ("evidence.",),
    }
    prefixes = prefix_map.get(section)
    if prefixes:
        heuristic_ids = [
            str(fact.get("evidence_id"))
            for fact in evidence_ledger.get("facts", [])
            if isinstance(fact, dict)
            and fact.get("evidence_id")
            and str(fact.get("evidence_id")).startswith(prefixes)
        ]
        if heuristic_ids:
            return heuristic_ids[:8]
    return _default_evidence_ids(evidence_ledger)[:3]


def _plain_text(value: Any) -> str:
    """Return a string from either simplified text or legacy text block.

    Inputs: arbitrary model field value.
    Outputs: bounded string used for subsequent validation.
    Assumptions: this does not author new narrative; it only unwraps text.
    """

    if isinstance(value, dict) and "text" in value:
        return str(value.get("text") or "")
    return str(value or "")


def validate_strategic_synthesis_response(
    response_text: str,
    *,
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any],
    risk_summary: dict[str, Any],
    historical_context: dict[str, Any] | None,
    evidence_ledger: dict[str, Any],
    fact_registry: FactRegistry | None = None,
) -> ReasoningValidationResult:
    """Validate Stage 3 final synthesis against existing Step-9 guards.

    Inputs: raw response text plus processed evidence contexts and ledger.
    Outputs: reasoning validation result.
    Assumptions: strategic synthesis reuses the same strict report-analysis
    schema so downstream report generation remains unchanged.
    """

    del fact_registry
    converted, conversion_errors = _convert_stage3_plain_payload(response_text, evidence_ledger)
    if conversion_errors:
        return ReasoningValidationResult(False, None, conversion_errors)
    validation = validate_strategic_analysis_response(json.dumps(converted, ensure_ascii=False))
    if not validation.is_valid or validation.analysis is None:
        return ReasoningValidationResult(False, None, validation.errors)
    claim_errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=finance_summary,
        anomaly_report=anomaly_report,
        evidence_package=evidence_package,
        risk_summary=risk_summary,
        historical_context=historical_context,
        evidence_ledger=evidence_ledger,
    )
    if claim_errors:
        sanitized, recovery = sanitize_strategic_analysis_claims(
            validation.analysis,
            claim_errors,
            evidence_ledger=evidence_ledger,
        )
        sanitized_errors = validate_evidence_bound_claims(
            sanitized,
            finance_summary=finance_summary,
            anomaly_report=anomaly_report,
            evidence_package=evidence_package,
            risk_summary=risk_summary,
            historical_context=historical_context,
            evidence_ledger=evidence_ledger,
        )
        if not sanitized_errors:
            sanitized["_python_attached_evidence"] = True
            sanitized["_strategic_recovery"] = recovery
            return ReasoningValidationResult(True, sanitized, ())
        return ReasoningValidationResult(False, sanitized, tuple(dict.fromkeys((*claim_errors, *sanitized_errors))))
    validation.analysis["_python_attached_evidence"] = True
    return ReasoningValidationResult(True, validation.analysis, ())


def sanitize_strategic_analysis_claims(
    analysis: dict[str, Any],
    claim_errors: tuple[str, ...],
    *,
    evidence_ledger: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove or soften unsupported strategic enrichment without inventing facts.

    Inputs: schema-valid strategic analysis, claim-validation errors, and evidence ledger.
    Outputs: sanitized analysis plus recovery metadata.
    Assumptions: deterministic facts remain untouched; unsupported observed
    financial claims are removed from their field while recommendation prose may
    be made qualitative.
    """

    del evidence_ledger
    sanitized = json.loads(json.dumps(analysis, ensure_ascii=False))
    repaired_fields: set[str] = set()
    sanitized_fields: set[str] = set()
    removed_fields: set[str] = set()
    for error in claim_errors:
        field_path = error.split(" contains ", 1)[0]
        if field_path.startswith("strategic_recommendations["):
            match = re.search(r"strategic_recommendations\[(\d+)\]", field_path)
            if match:
                index = int(match.group(1))
                _sanitize_recommendation(
                    sanitized,
                    index,
                    error=error,
                    sanitized_fields=sanitized_fields,
                    repaired_fields=repaired_fields,
                )
            continue
        _remove_invalid_analysis_field(
            sanitized,
            field_path,
            error=error,
            removed_fields=removed_fields,
            sanitized_fields=sanitized_fields,
        )
    recovery = {
        "accepted_fields": _accepted_analysis_fields(sanitized, removed_fields),
        "repaired_fields": sorted(repaired_fields),
        "sanitized_fields": sorted(sanitized_fields),
        "removed_fields": sorted(removed_fields),
        "remaining_blocking_violations": [],
        "original_validation_errors": list(claim_errors),
        "status_message": (
            "Algunas afirmaciones estratégicas fueron ajustadas o excluidas "
            "por falta de evidencia suficiente."
        ),
    }
    return sanitized, recovery


def _sanitize_recommendation(
    analysis: dict[str, Any],
    index: int,
    *,
    error: str,
    sanitized_fields: set[str],
    repaired_fields: set[str],
) -> None:
    """Sanitize one invalid recommendation in place.

    Inputs: analysis document, recommendation index, error, and audit sets.
    Outputs: mutates the recommendation by removing unsupported specificity.
    Assumptions: qualitative actions are safer than unsupported budgets/targets.
    """

    recommendations = analysis.get("strategic_recommendations", [])
    if not isinstance(recommendations, list) or index >= len(recommendations):
        return
    recommendation = recommendations[index]
    if not isinstance(recommendation, dict):
        return
    for key in ("action", "rationale", "expected_impact", "supporting_evidence"):
        text = str(recommendation.get(key) or "")
        if "unsupported number" in error or "unsupported causal claim" in error:
            text = _remove_unsupported_numeric_specificity(text)
            if key in {"action", "expected_impact"}:
                text = _append_sentence_once(
                    text,
                    "Definir montos, porcentajes o metas cuantitativas después de validar la capacidad financiera.",
                )
            sanitized_fields.add(f"strategic_recommendations[{index}].{key}")
        if "unsupported causal claim" in error:
            text = _soften_causal_statement(text)
            repaired_fields.add(f"strategic_recommendations[{index}].{key}")
        recommendation[key] = text.strip() or "Requiere validación adicional con evidencia financiera suficiente."


def _remove_invalid_analysis_field(
    analysis: dict[str, Any],
    field_path: str,
    *,
    error: str,
    removed_fields: set[str],
    sanitized_fields: set[str],
) -> None:
    """Remove unsupported observed narrative from one analysis field.

    Inputs: analysis document, field path, validation error, and audit sets.
    Outputs: mutates the invalid field to a conservative empty/neutral value.
    Assumptions: unsupported KPI/revenue/period facts must not remain visible.
    """

    field_name = field_path.split("[", 1)[0]
    if field_name not in analysis:
        return
    value = analysis.get(field_name)
    if isinstance(value, list):
        match = re.search(r"\[(\d+)\]", field_path)
        if match:
            index = int(match.group(1))
            if 0 <= index < len(value):
                value.pop(index)
                removed_fields.add(field_path)
        else:
            analysis[field_name] = []
            removed_fields.add(field_name)
    elif isinstance(value, str):
        if "unsupported causal claim" in error:
            analysis[field_name] = _soften_causal_statement(value)
            sanitized_fields.add(field_name)
        else:
            analysis[field_name] = _fallback_analysis_text(field_name)
            removed_fields.add(field_name)
    # Preserve Python-attached evidence IDs even when unsupported prose is
    # removed. The remaining section stays auditable without inventing content.


def _remove_unsupported_numeric_specificity(text: str) -> str:
    """Remove numeric specificity from strategic recommendation prose.

    Inputs: recommendation text.
    Outputs: text without currency/percentage/large-number claims.
    Assumptions: this never substitutes another number.
    """

    text = re.sub(r"\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", "", text)
    text = re.sub(r"\b\d+(?:[\.,]\d+)?\s*%", "", text)
    text = re.sub(r"\b\d{4,}\b", "", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;")


def _soften_causal_statement(text: str) -> str:
    """Turn unsupported causal certainty into a clearly labeled hypothesis.

    Inputs: model-authored prose.
    Outputs: cautious Spanish wording.
    Assumptions: this only changes epistemic framing, not deterministic facts.
    """

    if not text.strip():
        return "Una posible explicación requiere validación con evidencia adicional."
    if any(marker in text.casefold() for marker in ("hipótesis", "posible", "podría", "requiere validación")):
        return text
    return f"Como hipótesis que requiere validación: {text}"


def _append_sentence_once(text: str, sentence: str) -> str:
    """Append a sentence unless it is already present.

    Inputs: base text and sentence.
    Outputs: combined text.
    Assumptions: keeps sanitization messages concise.
    """

    if sentence in text:
        return text
    return (text.rstrip(". ") + ". " + sentence).strip() if text else sentence


def _fallback_analysis_text(field_name: str) -> str:
    """Return a conservative Spanish note for a removed strategic field.

    Inputs: field name.
    Outputs: bounded Spanish fallback text.
    Assumptions: this states validation status only and does not add analysis.
    """

    labels = {
        "executive_summary": "El análisis determinístico está disponible; se omitieron afirmaciones estratégicas sin evidencia suficiente.",
        "financial_health_analysis": "La lectura financiera se limita a los indicadores determinísticos del reporte.",
        "kpi_analysis": "La lectura de KPIs se limita a los valores determinísticos calculados.",
        "historical_summary": "El resumen histórico se limita a los registros determinísticos disponibles.",
        "historical_trend_analysis": "La tendencia histórica se presenta solo con evidencia validada.",
        "department_analysis": "El análisis por departamento se limita a los datos procesados disponibles.",
        "anomaly_analysis": "La lectura de anomalías se limita a los resultados determinísticos.",
        "recommendation_follow_up_analysis": "El seguimiento de recomendaciones se limita a evidencia histórica validada.",
        "longitudinal_risk_analysis": "La evaluación longitudinal se limita a riesgos históricos validados.",
        "reasoning_summary": "Se excluyeron afirmaciones no respaldadas por evidencia suficiente.",
    }
    return labels.get(field_name, "")


def _accepted_analysis_fields(analysis: dict[str, Any], removed_fields: set[str]) -> list[str]:
    """List non-empty fields that survived strategic sanitization.

    Inputs: sanitized analysis and removed-field set.
    Outputs: sorted field names.
    Assumptions: used for audit metadata only.
    """

    accepted: list[str] = []
    for key, value in analysis.items():
        if key.startswith("_") or key in removed_fields:
            continue
        if isinstance(value, (str, list, dict)) and value:
            accepted.append(key)
        elif value is not None:
            accepted.append(key)
    return sorted(accepted)


def _run_structured_stage(
    *,
    client: Any,
    stage_id: str,
    stage_name: str,
    prompt: str,
    validator: Any,
    response_format: dict[str, Any] | str = "json",
    stage_timeout_seconds: float | None = None,
    fact_registry: FactRegistry | None = None,
) -> ReasoningStageResult:
    """Call Ollama once and validate one reasoning stage.

    Inputs: client, stage metadata, prompt, validator, and optional JSON schema.
    Outputs: stage result with prompt/runtime/validation telemetry.
    Assumptions: Phase 14 does not perform deterministic rewrite/translation.
    """

    started = time.perf_counter()
    previous_response_format = getattr(client, "response_format", None)
    if previous_response_format is not None:
        setattr(client, "response_format", response_format)
    try:
        if hasattr(client, "generate_with_metadata"):
            generation = client.generate_with_metadata(prompt)
            response = str(generation["response"])
            ollama_telemetry = dict(generation.get("telemetry", {}))
        else:
            response = client.generate(prompt)
            ollama_telemetry = {}
    except OllamaError as exc:
        telemetry = {
            "stage_id": stage_id,
            "prompt_characters": len(prompt),
            "prompt_token_estimate": estimate_tokens_from_text(prompt),
            "json_validation_time_seconds": 0.0,
            "total_stage_time_seconds": time.perf_counter() - started,
            "timeout_error_category": exc.category,
            "error_category": exc.category,
            **getattr(exc, "telemetry", {}),
        }
        return ReasoningStageResult(
            stage_id=stage_id,
            stage_name=stage_name,
            accepted=False,
            payload={},
            validation_errors=(str(exc),),
            telemetry=telemetry,
        )
    finally:
        if previous_response_format is not None:
            setattr(client, "response_format", previous_response_format)

    elapsed_after_generation = time.perf_counter() - started
    stage_timeout_exceeded_after_generation = (
        stage_timeout_seconds is not None
        and elapsed_after_generation > stage_timeout_seconds
    )
    if stage_timeout_seconds is not None and elapsed_after_generation > stage_timeout_seconds:
        # A response that has already returned is still useful: validate it
        # instead of discarding potentially salvageable strategy. The telemetry
        # records the budget overrun without weakening evidence validation.
        ollama_telemetry = {
            **ollama_telemetry,
            "stage_timeout_budget_exceeded": True,
            "stage_timeout_seconds": stage_timeout_seconds,
            "timeout_error_category": "stage_timeout",
        }

    validation_started = time.perf_counter()
    validation = validator(response)
    validation_time = time.perf_counter() - validation_started
    schema_retry_attempted = False
    if not validation.is_valid and _is_schema_only_error(validation.errors):
        schema_retry_attempted = True
        retry_prompt = build_schema_repair_prompt(
            stage_name=stage_name,
            schema=_stage_schema_text_for_id(stage_id),
            schema_errors=validation.errors,
            original_response=response,
        )
        previous_response_format = getattr(client, "response_format", None)
        if previous_response_format is not None:
            setattr(client, "response_format", response_format)
        try:
            if hasattr(client, "generate_with_metadata"):
                generation = client.generate_with_metadata(retry_prompt)
                response = str(generation["response"])
                retry_telemetry = dict(generation.get("telemetry", {}))
                ollama_telemetry = _merge_retry_telemetry(ollama_telemetry, retry_telemetry)
            else:
                response = client.generate(retry_prompt)
        except OllamaError as exc:
            validation = ReasoningValidationResult(False, None, (str(exc),))
            ollama_telemetry = {
                **ollama_telemetry,
                "schema_retry_error_category": exc.category,
            }
        finally:
            if previous_response_format is not None:
                setattr(client, "response_format", previous_response_format)
        validation_started = time.perf_counter()
        validation = validator(response)
        validation_time += time.perf_counter() - validation_started
    telemetry = {
        "stage_id": stage_id,
        "prompt_characters": len(prompt),
        "prompt_token_estimate": estimate_tokens_from_text(prompt),
        "json_validation_time_seconds": validation_time,
        "total_stage_time_seconds": time.perf_counter() - started,
        "error_category": None if validation.is_valid else "validation_rejection",
        "timeout_error_category": ollama_telemetry.get("timeout_error_category"),
        "stage_timeout_budget_exceeded": stage_timeout_exceeded_after_generation,
        "schema_retry_attempted": schema_retry_attempted,
        "placeholder_retry_attempted": False,
        **ollama_telemetry,
    }
    return ReasoningStageResult(
        stage_id=stage_id,
        stage_name=stage_name,
        accepted=validation.is_valid,
        payload=validation.payload or {
            "_validation_failed": True,
            "_raw_model_response": response,
            "_validation_errors": list(validation.errors),
        },
        validation_errors=validation.errors,
        telemetry=telemetry,
    )


def _validate_reasoning_items(
    field_name: str,
    value: Any,
    evidence_ledger: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate a list of structured stage reasoning items.

    Inputs: field name, untrusted value, ledger, and mutable error list.
    Outputs: appends field-specific validation errors.
    Assumptions: every item must include model-authored Spanish reasoning text;
    Python validates unsupported deterministic claims separately.
    """

    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return
    if len(value) > 8:
        errors.append(f"{field_name} may contain at most 8 items")
    approved_numbers = _approved_numbers_for_reasoning(evidence_ledger)
    approved_periods = set(str(period) for period in evidence_ledger.get("approved_periods", []))
    approved_entities = set(str(entity) for entity in evidence_ledger.get("approved_entities", []))
    for index, item in enumerate(value):
        prefix = f"{field_name}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        required_keys = {"text"}
        if field_name in {"claims", "risks", "opportunities"}:
            required_keys.add("confidence")
        if field_name == "claims":
            required_keys.add("claim_type")
        if set(item) != required_keys:
            errors.append(
                f"schema: {prefix} must contain exactly {sorted(required_keys)}; received {sorted(item)}"
            )
            continue
        text = item.get("text") or item.get("claim") or item.get("question") or item.get("risk")
        if not isinstance(text, str) or not text.strip() or len(text) > 1200:
            errors.append(f"{prefix}.text must be non-empty bounded text")
            continue
        if field_name in {"claims", "risks", "opportunities"}:
            confidence = item.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0 <= float(confidence) <= 1
            ):
                errors.append(f"{prefix}.confidence must be numeric between 0 and 1")
        if field_name == "claims" and item.get("claim_type") not in CLAIM_TYPES:
            errors.append(f"{prefix}.claim_type must be fact, interpretation, or hypothesis")
        language_errors = validate_user_facing_spanish({"key_findings": [text]})
        errors.extend(f"{prefix}: {error}" for error in language_errors)
        for number in _numbers_in_text(text):
            if number not in approved_numbers and number not in approved_periods:
                errors.append(f"{prefix} contains unsupported number: {number}")
        for period in set(re.findall(r"20\d{2}[-_]\d{2}|20\d{2}", text)):
            if period not in approved_periods:
                errors.append(f"{prefix} contains unsupported period: {period}")
        for entity in approved_entities:
            if entity and entity in text:
                break
        if item.get("claim_type") == "hypothesis" and _sounds_certain(text):
            errors.append(f"{prefix}.claim_type hypothesis is written as established fact")


def _validate_item_support(
    *,
    prefix: str,
    field_name: str,
    item: dict[str, Any],
    text: str,
    registry: FactRegistry,
    allowed_placeholders: set[str],
) -> tuple[list[str], list[RegisteredFact]]:
    """Validate one item's supporting facts and semantic compatibility.

    Inputs: item path, field name, model item, text, registry and stage allowset.
    Outputs: errors plus resolved facts.
    Assumptions: this function never selects substitute facts or rewrites prose.
    """

    errors: list[str] = []
    supporting_facts = item.get("supporting_facts")
    if not isinstance(supporting_facts, list):
        return ([f"{prefix}.supporting_facts must be a list"], [])
    if field_name != "open_questions" and not supporting_facts:
        errors.append(f"{prefix}.supporting_facts must be non-empty")
    if len(supporting_facts) > 5:
        errors.append(f"{prefix}.supporting_facts may contain at most 5 placeholders")
    normalized_support = [str(placeholder).strip() for placeholder in supporting_facts]
    resolved_facts, resolution_errors = registry.resolve_supporting_facts(
        normalized_support,
        allowed_placeholders=allowed_placeholders,
    )
    errors.extend(f"{prefix}: {error}" for error in resolution_errors)

    text_placeholders = sorted(PLACEHOLDER_PATTERN.findall(text))
    support_fact_ids = {
        PLACEHOLDER_PATTERN.fullmatch(placeholder).group(1)  # type: ignore[union-attr]
        for placeholder in normalized_support
        if PLACEHOLDER_PATTERN.fullmatch(placeholder)
    }
    for fact_id in text_placeholders:
        if fact_id not in support_fact_ids:
            errors.append(f"{prefix}.text uses {{{{{fact_id}}}}} missing from supporting_facts")
    if item.get("claim_type") == "fact":
        text_fact_ids = set(text_placeholders)
        for fact in resolved_facts:
            if fact.value_type not in {"entity", "period"} and fact.fact_id not in text_fact_ids:
                errors.append(f"{prefix} includes supporting fact {fact.placeholder} not used by factual claim text")

    entities = {fact.entity for fact in resolved_facts if fact.entity}
    if len(entities) > 1:
        errors.append(f"{prefix} has department/entity mismatch in supporting_facts: {sorted(entities)}")
    periods = {fact.period for fact in resolved_facts if fact.period}
    if item.get("claim_type") == "fact" and len(periods) > 1:
        errors.append(f"{prefix} has period mismatch in factual supporting_facts: {sorted(periods)}")
    metrics = {fact.metric_name for fact in resolved_facts if fact.metric_name not in {"entity", "period"}}
    if item.get("claim_type") == "fact" and len(metrics) > 2:
        errors.append(f"{prefix} has metric mismatch in factual supporting_facts: {sorted(metrics)}")
    if item.get("claim_type") == "hypothesis" and _sounds_certain(text):
        errors.append(f"{prefix}.claim_type hypothesis is written as established fact")
    return errors, resolved_facts


def _support_fact_metadata(fact: RegisteredFact) -> dict[str, Any]:
    """Return audit metadata for a resolved support fact.

    Inputs: registered fact.
    Outputs: metadata without model-generated evidence choices.
    Assumptions: this is internal/debug information, not user-facing prose.
    """

    return {
        "fact_id": fact.fact_id,
        "placeholder": fact.placeholder,
        "metric_name": fact.metric_name,
        "value_type": fact.value_type,
        "period": fact.period,
        "entity": fact.entity,
        "resolved_evidence_ids": list(fact.evidence_ids),
    }


def _selected_fact_summary(evidence_ledger: dict[str, Any], stage_id: str) -> dict[str, Any]:
    """Return audit counts for the simplified stage fact selection.

    Inputs: evidence ledger and stage ID.
    Outputs: selected fact count and category.
    Assumptions: facts themselves are not copied into every stage payload.
    """

    prefixes = {
        "financial_performance": ("finance.", "anomaly."),
        "historical_operational": ("history.", "finance.", "anomaly."),
    }.get(stage_id, ("finance.", "anomaly.", "history."))
    return {
        "stage_id": stage_id,
        "selected_fact_count": len(_fact_cards_for_prefixes(evidence_ledger, prefixes, limit=25)),
        "selection_prefixes": list(prefixes),
    }


def _allowed_placeholders_for_stage(stage_id: str, registry: FactRegistry) -> set[str]:
    """Return the stage-specific placeholder allowlist.

    Inputs: stage ID and registry.
    Outputs: allowed placeholder set.
    Assumptions: Stage 1 uses current finance/anomaly facts; Stage 2 may compare
    history with current context.
    """

    prefixes = {
        "financial_performance": ("finance.", "anomaly."),
        "historical_operational": ("history.", "finance.", "anomaly."),
        "strategic_synthesis": ("history.", "finance.", "anomaly.", "evidence."),
    }.get(stage_id, ("finance.", "anomaly.", "history.", "evidence."))
    return {
        fact.placeholder
        for fact in registry.facts
        if any(evidence_id.startswith(prefixes) for evidence_id in fact.evidence_ids)
    }


def _sounds_certain(text: str) -> bool:
    """Return whether a hypothesis is phrased as established fact.

    Inputs: Spanish text.
    Outputs: True for a small set of deterministic certainty markers.
    Assumptions: this conservative check catches obvious validator gaming only.
    """

    lowered = text.casefold()
    certain_terms = ("se debe a", "es causado por", "demuestra que", "confirma que")
    cautious_terms = ("hipótesis", "posible", "probable", "podría", "requiere validar")
    return any(term in lowered for term in certain_terms) and not any(term in lowered for term in cautious_terms)


def _clean_reasoning_items(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    """Clean validated reasoning items without altering meaning.

    Inputs: model item list.
    Outputs: normalized list with text and optional confidence/type metadata.
    Assumptions: validation has already checked structure and evidence IDs.
    """

    cleaned: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return cleaned
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("claim") or item.get("question") or item.get("risk") or "").strip()
        copied = dict(item)
        copied["text"] = text
        if field_name in {"claims", "risks", "opportunities"}:
            copied["confidence"] = float(item.get("confidence"))
        cleaned.append(copied)
    return cleaned


def normalize_reasoning_stage_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Normalize safe schema aliases without changing model-authored claims.

    Inputs: parsed model JSON object.
    Outputs: normalized payload plus audit records for each key normalization.
    Assumptions: adapter may rename fields and unwrap one obvious wrapper only;
    it never writes prose, adds evidence IDs, computes values, or deletes claims.
    """

    normalizations: list[dict[str, str]] = []
    current = dict(payload)
    if len(current) == 1:
        wrapper_key, wrapper_value = next(iter(current.items()))
        if wrapper_key in STAGE_WRAPPER_KEYS and isinstance(wrapper_value, dict):
            current = dict(wrapper_value)
            normalizations.append(
                {"kind": "unwrap", "from": wrapper_key, "to": "root"}
            )

    normalized: dict[str, Any] = {}
    for key, value in current.items():
        target_key = STAGE_TOP_LEVEL_ALIASES.get(key, key)
        if target_key != key:
            normalizations.append({"kind": "rename", "from": key, "to": target_key})
        if target_key in normalized and isinstance(normalized[target_key], list) and isinstance(value, list):
            normalized[target_key] = [*normalized[target_key], *value]
            normalizations.append({"kind": "merge", "from": key, "to": target_key})
        else:
            normalized[target_key] = value

    for list_key in ("claims", "risks", "opportunities", "open_questions"):
        items = normalized.get(list_key)
        if not isinstance(items, list):
            continue
        converted_items = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                converted_items.append(item)
                continue
            converted = dict(item)
            for alias, canonical in (("claim", "text"), ("risk", "text"), ("question", "text")):
                if alias in converted and canonical not in converted:
                    converted[canonical] = converted.pop(alias)
                    normalizations.append(
                        {
                            "kind": "rename",
                            "from": f"{list_key}[{index}].{alias}",
                            "to": f"{list_key}[{index}].{canonical}",
                        }
                    )
            if "confidence_level" in converted and "confidence" not in converted:
                converted["confidence"] = converted.pop("confidence_level")
                normalizations.append(
                    {
                        "kind": "rename",
                        "from": f"{list_key}[{index}].confidence_level",
                        "to": f"{list_key}[{index}].confidence",
                    }
                )
            converted_items.append(converted)
        normalized[list_key] = converted_items

    return normalized, normalizations


def _approved_numbers_for_reasoning(evidence_ledger: dict[str, Any]) -> set[str]:
    """Return numeric strings the model may cite in simplified reasoning.

    Inputs: Python evidence ledger containing approved numbers and fact cards.
    Outputs: normalized number variants accepted by the stage validator.
    Assumptions: variants are display/rendering equivalents of supplied facts;
    this function never derives new finance metrics or expands allowed claims.
    """

    approved = {str(number) for number in evidence_ledger.get("approved_numbers", [])}
    for fact in evidence_ledger.get("facts", []):
        if not isinstance(fact, dict):
            continue
        for value in (fact.get("display_value"), fact.get("raw_value"), fact.get("value")):
            if value in {None, ""}:
                continue
            text = str(value)
            approved.add(text)
            approved.update(_numbers_in_text(text))
            # Currency displays may be copied with or without thousands
            # separators/currency symbols. Accept only the normalized literal
            # Python already supplied, not arbitrary calculations.
            compact = re.sub(r"[^0-9.\-%]", "", text)
            if compact:
                approved.update(_numbers_in_text(compact))
        display = str(fact.get("display_value") or "")
        raw_value = fact.get("raw_value")
        if display.endswith("%") and isinstance(raw_value, (int, float)) and raw_value < 0:
            # Spanish prose often says "disminución del 0.3%" instead of
            # repeating the minus sign. Treat the absolute percent as the same
            # supplied fact only when the ledger value itself is negative.
            approved.add(display.lstrip("-"))
            approved.update(_numbers_in_text(display.lstrip("-")))
    return approved


def _numbers_in_text(text: str) -> set[str]:
    """Extract normalized explicit numeric claims from prose.

    Inputs: prose text.
    Outputs: set of number strings comparable to ledger approved values.
    Assumptions: years are handled separately as approved periods.
    """

    numbers: set[str] = set()
    pattern = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|-?\d+(?:[\.,]\d+)?%?"
    for match in re.finditer(pattern, text):
        value = match.group(0).replace(",", ".")
        if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:\.\d+)?%?", value):
            suffix = "%" if value.endswith("%") else ""
            core = value[:-1] if suffix else value
            parts = core.split(".")
            value = "".join(parts) + suffix
        if value.endswith(".0"):
            value = value[:-2]
        numbers.add(value)
    return numbers


def _prompt_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """Return one compact fact for a stage prompt.

    Inputs: full ledger fact.
    Outputs: compact fact preserving IDs, exact values, entity, period and claim.
    Assumptions: validator-only metadata is not needed by the model.
    """

    return {
        "evidence_id": fact.get("evidence_id"),
        "metric": fact.get("metric") or fact.get("field"),
        "display_value": fact.get("display_value"),
        "period": fact.get("period"),
        "entity": fact.get("entity"),
        "claim": fact.get("claim"),
        "source_reference": fact.get("source_reference"),
    }


def _stage_prompt(*, stage_name: str, schema: str, context: dict[str, Any]) -> str:
    """Build a shared stage prompt wrapper.

    Inputs: stage title, schema text, and compact JSON context.
    Outputs: strict prompt string.
    Assumptions: all user-facing prose must be generated directly in Spanish.
    """

    return (
        f"STAGE: {stage_name}\n"
        "Escribe todo texto de usuario en español profesional.\n"
        "Usa solo los hechos deterministas incluidos en CONTEXT.\n"
        "Nunca escribas evidence IDs en la respuesta.\n"
        "No calcules, no estimes, no redondees, no inventes causas ni resultados.\n"
        "Si una causa es incierta, márcala como hipótesis o información faltante.\n"
        "Devuelve JSON estricto únicamente.\n"
        f"SCHEMA:\n{schema}\n"
        "CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )

def build_schema_repair_prompt(
    *,
    stage_name: str,
    schema: str,
    schema_errors: tuple[str, ...],
    original_response: str,
) -> str:
    """Build one schema-only repair prompt for a reasoning stage.

    Inputs: stage name, exact schema text, schema errors, and original output.
    Outputs: compact prompt asking Ollama to restructure only.
    Assumptions: the retry may rename/restructure fields but must preserve
    original facts, numbers, Spanish text, evidence IDs, and meaning.
    """

    return (
        f"STAGE_SCHEMA_REPAIR: {stage_name}\n"
        "El JSON anterior fue válido, pero no cumplió la estructura requerida.\n"
        "Reestructura el MISMO contenido para cumplir exactamente el esquema.\n"
        "No agregues ni elimines afirmaciones, riesgos, oportunidades, preguntas, números ni supporting_facts.\n"
        "No cambies el significado ni escribas análisis nuevo.\n"
        "Devuelve JSON estricto únicamente.\n"
        f"SCHEMA_ERRORS:\n{json.dumps(list(schema_errors), ensure_ascii=False)}\n"
        f"REQUIRED_SCHEMA:\n{schema}\n"
        "ORIGINAL_RESPONSE:\n"
        + original_response[:20_000]
    )


def _is_schema_only_error(errors: tuple[str, ...]) -> bool:
    """Return whether validation failed only because of schema shape.

    Inputs: validation errors.
    Outputs: True when a schema-only retry is safe.
    Assumptions: evidence/language/number/entity failures must not trigger a
    restructure retry because they require substantive correction.
    """

    return bool(errors) and all(str(error).startswith("schema:") for error in errors)


def build_placeholder_repair_prompt(
    *,
    stage_name: str,
    placeholder_errors: tuple[str, ...],
    original_response: str,
    fact_registry: FactRegistry,
) -> str:
    """Build one placeholder-only repair prompt for a reasoning stage.

    Inputs: stage name, validation errors, original response and fact registry.
    Outputs: compact prompt asking Ollama to replace deterministic literals with
    approved placeholders only.
    Assumptions: retry must preserve claims, evidence IDs, priorities, numbers,
    and meaning; Python still validates again afterwards.
    """

    return (
        f"STAGE_PLACEHOLDER_REPAIR: {stage_name}\n"
        "El JSON anterior fue estructuralmente válido, pero incumplió el contrato de placeholders.\n"
        "Reescribe el MISMO JSON usando solo placeholders {{FACT_###}} para hechos deterministas.\n"
        "No agregues ni elimines afirmaciones, riesgos, oportunidades, preguntas, recomendaciones ni supporting_facts.\n"
        "No calcules, no redondees, no cambies significado y no escribas literales numéricos, periodos o entidades.\n"
        "Devuelve JSON estricto únicamente.\n"
        f"PLACEHOLDER_ERRORS:\n{json.dumps(list(placeholder_errors), ensure_ascii=False)}\n"
        "ALLOWED_FACTS:\n"
        f"{json.dumps(fact_registry.prompt_facts()[:120], ensure_ascii=False, separators=(',', ':'))}\n"
        "ORIGINAL_RESPONSE:\n"
        + original_response[:20_000]
    )


def _is_placeholder_error(errors: tuple[str, ...]) -> bool:
    """Return whether a placeholder-compliance retry is safe.

    Inputs: validation errors.
    Outputs: True when errors are limited to placeholder/literal formatting.
    Assumptions: schema, language, evidence, causal, and unsupported claim
    failures remain rejected without this repair path.
    """

    markers = (
        "placeholder",
        "unsupported numeric literal",
        "deterministic literal",
        "supporting_facts must be non-empty",
        "missing from supporting_facts",
        "unknown placeholder",
        "malformed placeholder",
    )
    return bool(errors) and all(any(marker in str(error) for marker in markers) for error in errors)


def _merge_retry_telemetry(first: dict[str, Any], retry: dict[str, Any]) -> dict[str, Any]:
    """Merge first-attempt and schema-retry telemetry.

    Inputs: two Ollama telemetry dictionaries.
    Outputs: combined telemetry preserving attempt details and total timings.
    Assumptions: numeric seconds/counts may be summed for high-level totals.
    """

    merged = dict(first)
    merged["schema_retry_telemetry"] = retry
    for key in (
        "http_elapsed_time_seconds",
        "model_load_time_seconds",
        "prompt_evaluation_time_seconds",
        "generation_time_seconds",
        "total_ollama_time_seconds",
    ):
        if isinstance(first.get(key), (int, float)) or isinstance(retry.get(key), (int, float)):
            merged[key] = float(first.get(key) or 0) + float(retry.get(key) or 0)
    for key in ("prompt_eval_count", "generation_eval_count"):
        if isinstance(first.get(key), (int, float)) or isinstance(retry.get(key), (int, float)):
            merged[key] = int(first.get(key) or 0) + int(retry.get(key) or 0)
    return merged


def _stage_schema_text_for_id(stage_id: str) -> str:
    """Return the exact text schema for one modular stage ID.

    Inputs: stage ID.
    Outputs: schema text.
    Assumptions: Stage 1 and Stage 2 share the same minimal shape.
    """

    if stage_id == "financial_performance":
        return _financial_stage_schema_text()
    if stage_id == "historical_operational":
        return _historical_stage_schema_text()
    return "Use the required strategic synthesis schema from the previous prompt."


def _financial_stage_schema_text() -> str:
    """Return the Stage 1 JSON contract.

    Inputs: none.
    Outputs: compact schema instructions.
    Assumptions: each item includes text and supporting_facts.
    """

    return (
        "Return exactly one JSON object with exactly these top-level keys and no others: "
        "claims, risks, opportunities, open_questions. "
        "claims item: {text: Spanish string, confidence: 0..1, claim_type: fact|interpretation|hypothesis}. "
        "risks/opportunities item: {text: Spanish string, confidence: 0..1}. "
        "open_questions item: {text: Spanish string}. "
        "Do not include examples, fake values, markdown, or prose outside JSON."
    )


def _historical_stage_schema_text() -> str:
    """Return the Stage 2 JSON contract.

    Inputs: none.
    Outputs: compact schema instructions.
    Assumptions: each item includes text and supporting_facts.
    """

    return (
        "Return exactly one JSON object with exactly these top-level keys and no others: "
        "claims, risks, opportunities, open_questions. "
        "claims item: {text: Spanish string, confidence: 0..1, claim_type: fact|interpretation|hypothesis}. "
        "risks/opportunities item: {text: Spanish string, confidence: 0..1}. "
        "open_questions item: {text: Spanish string}. "
        "Do not include examples, fake values, markdown, or prose outside JSON."
    )


def _stage_required_fields(stage_id: str) -> tuple[str, ...]:
    """Return required fields for a modular reasoning stage.

    Inputs: stage ID.
    Outputs: tuple of field names.
    Assumptions: only Stage 1 and Stage 2 use this smaller contract.
    """

    if stage_id in {"financial_performance", "historical_operational"}:
        return ("claims", "risks", "opportunities", "open_questions")
    raise ValueError(f"Unknown reasoning stage: {stage_id}")


def reasoning_stage_json_schema(stage_id: str, allowed_placeholders: set[str] | None = None) -> dict[str, Any]:
    """Return the provider JSON schema for Stage 1/2 reasoning.

    Inputs: stage ID.
    Outputs: JSON schema dictionary passed to Ollama's ``format`` parameter.
    Assumptions: Stage-specific semantics are prompt-driven; the shape remains
    minimal and stable across financial and historical reasoning.
    """

    del allowed_placeholders
    if stage_id not in {"financial_performance", "historical_operational"}:
        raise ValueError(f"Unknown reasoning stage schema: {stage_id}")
    text_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 1200},
        },
    }
    confidence_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "confidence"],
        "properties": {
            **text_item["properties"],
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    claim_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "confidence", "claim_type"],
        "properties": {
            **confidence_item["properties"],
            "claim_type": {"type": "string", "enum": sorted(CLAIM_TYPES)},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_stage_required_fields(stage_id)),
        "properties": {
            "claims": {"type": "array", "items": claim_item, "maxItems": 8},
            "risks": {"type": "array", "items": confidence_item, "maxItems": 8},
            "opportunities": {"type": "array", "items": confidence_item, "maxItems": 8},
            "open_questions": {"type": "array", "items": text_item, "maxItems": 8},
        },
    }


def strategic_synthesis_fact_json_schema(allowed_placeholders: set[str] | None = None) -> dict[str, Any]:
    """Return the LLM-facing Stage 3 schema using supporting facts only.

    Inputs: optional allowed placeholder set.
    Outputs: JSON schema for strategic synthesis without evidence IDs.
    Assumptions: Python converts supporting facts to internal evidence IDs.
    """

    string_schema = {"type": "string", "minLength": 1, "maxLength": 1800}
    del allowed_placeholders
    narrative_block = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": string_schema},
    }
    recommendation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["priority", "action", "rationale", "supporting_evidence", "expected_impact", "confidence"],
        "properties": {
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            "action": string_schema,
            "rationale": string_schema,
            "supporting_evidence": string_schema,
            "expected_impact": string_schema,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    block_list = {
        "type": "array",
        "minItems": 0,
        "maxItems": 8,
        "items": narrative_block,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "executive_summary",
            "key_findings",
            "root_causes",
            "financial_health_analysis",
            "kpi_analysis",
            "historical_summary",
            "historical_trend_analysis",
            "department_analysis",
            "anomaly_analysis",
            "recommendation_follow_up_analysis",
            "longitudinal_risk_analysis",
            "strategic_recommendations",
            "strategic_priorities",
            "missing_information",
            "confidence",
            "reasoning_summary",
        ],
        "properties": {
            "executive_summary": narrative_block,
            "key_findings": block_list,
            "root_causes": block_list,
            "financial_health_analysis": narrative_block,
            "kpi_analysis": narrative_block,
            "historical_summary": narrative_block,
            "historical_trend_analysis": narrative_block,
            "department_analysis": narrative_block,
            "anomaly_analysis": narrative_block,
            "recommendation_follow_up_analysis": narrative_block,
            "longitudinal_risk_analysis": narrative_block,
            "strategic_recommendations": {"type": "array", "minItems": 1, "maxItems": 8, "items": recommendation},
            "strategic_priorities": block_list,
            "missing_information": block_list,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning_summary": narrative_block,
        },
    }


def _rejected_result(
    finance_summary: dict[str, Any],
    period_slug: str,
    historical_context: dict[str, Any] | None,
    evidence_ledger: dict[str, Any],
    state: ReasoningState,
    telemetry: dict[str, Any],
) -> StrategicAnalysisResult:
    """Build a rejected result after one modular stage fails.

    Inputs: processed metadata, ledger, state, and pipeline telemetry.
    Outputs: rejected StrategicAnalysisResult.
    Assumptions: final reporting remains blocked unless every stage validates.
    """

    errors = tuple(
        error
        for stage in state.stage_results
        if not stage.accepted
        for error in stage.validation_errors
    ) or ("A modular reasoning stage failed validation.",)
    document = _analysis_document(
        period_slug=period_slug,
        report_period=str(finance_summary.get("report_period", period_slug)),
        ollama_available=True,
        validation_status="rejected",
        validation_errors=errors,
        analysis=_empty_analysis(),
        historical_context=historical_context,
        evidence_ledger=evidence_ledger,
        reasoning_state=state,
        model_name=None,
    )
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=False,
        validation_errors=errors,
        telemetry={**telemetry, "stage_telemetry": [stage.telemetry for stage in state.stage_results]},
    )


def _degraded_result(
    *,
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any],
    period_slug: str,
    historical_context: dict[str, Any] | None,
    evidence_ledger: dict[str, Any],
    state: ReasoningState,
    telemetry: dict[str, Any],
    started: float,
    reason: tuple[str, ...],
    ollama_available: bool,
    model_name: str | None = None,
) -> StrategicAnalysisResult:
    """Build an explicit degraded deterministic strategy document.

    Inputs: deterministic evidence contexts, reasoning state, telemetry, and
    degradation reason.
    Outputs: accepted StrategicAnalysisResult clearly labeled as degraded.
    Assumptions: this is an emergency fallback after explicit degraded mode,
    Ollama unavailability, or exhausted bounded AI recovery.
    """

    analysis = build_deterministic_strategic_synthesis(
        evidence_ledger=evidence_ledger,
        finance_summary=finance_summary,
        anomaly_report=anomaly_report,
        evidence_package=evidence_package,
        period_slug=period_slug,
    )
    recovery = {
        "outcome": "degraded_deterministic_fallback",
        "source_label": "Modo degradado: análisis determinístico",
        "degraded_mode": True,
        "degraded_reason": list(reason),
        "ai_mode_enabled": not any("explicitly selected" in item for item in reason),
        "attempts": _strategic_attempt_labels(state),
        "user_message": "Las capacidades de razonamiento de IA no estuvieron disponibles para este análisis.",
    }
    analysis["_strategic_recovery"] = recovery
    document = _analysis_document(
        period_slug=period_slug,
        report_period=str(finance_summary.get("report_period", period_slug)),
        ollama_available=ollama_available,
        validation_status="accepted",
        validation_errors=(),
        analysis=analysis,
        historical_context=historical_context,
        evidence_ledger=evidence_ledger,
        reasoning_state=state,
        model_name=model_name,
    )
    document["analysis_source"] = "degraded_deterministic"
    document["strategic_recovery"] = recovery
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=True,
        validation_errors=(),
        telemetry={
            **telemetry,
            "ai_mode_enabled": recovery["ai_mode_enabled"],
            "final_strategy_source": "degraded_deterministic",
            "degraded_mode": True,
            "degraded_reason": list(reason),
            "stage_telemetry": [stage.telemetry for stage in state.stage_results],
            "total_stage_time_seconds": time.perf_counter() - started,
        },
    )


def _strategic_attempt_labels(state: ReasoningState) -> list[str]:
    """Return labels for strategic-synthesis attempts captured in state.

    Inputs: current reasoning state.
    Outputs: stage names for strategic synthesis attempts in execution order.
    Assumptions: callers use this only for diagnostics.
    """

    return [
        stage.stage_name
        for stage in state.stage_results
        if stage.stage_id == "strategic_synthesis"
    ]


def _analysis_document(
    *,
    period_slug: str,
    report_period: str,
    ollama_available: bool,
    validation_status: str,
    validation_errors: tuple[str, ...],
    analysis: dict[str, Any],
    historical_context: dict[str, Any] | None,
    evidence_ledger: dict[str, Any],
    reasoning_state: ReasoningState,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Assemble a Step-9-compatible strategic-analysis document.

    Inputs: metadata, validation state, final analysis, history, ledger and
    reasoning state.
    Outputs: JSON-compatible document used by existing report generation.
    Assumptions: accepted reports consume ``analysis`` exactly as before.
    """

    recommendations = analysis.get("recommendations", analysis.get("strategic_recommendations", []))
    recommendations = recommendations if isinstance(recommendations, list) else []
    analysis_source = "ollama_modular_reasoning"
    section_provenance = _analysis_section_provenance(
        analysis=analysis,
        analysis_source=analysis_source,
        validation_status=validation_status,
        reasoning_state=reasoning_state,
        model_name=model_name,
    )
    ai_usage = _ai_usage_diagnostics(
        reasoning_state=reasoning_state,
        section_provenance=section_provenance,
        model_name=model_name,
    )
    return {
        "analysis_id": f"STRATEGIC-ANALYSIS-{period_slug.upper().replace('_', '-')}",
        "period_slug": period_slug,
        "report_period": report_period,
        "analysis_source": analysis_source,
        "ollama_available": ollama_available,
        "validation_status": validation_status,
        "analysis_generated": validation_status in {"accepted", "sanitized"},
        "validation_errors": list(validation_errors),
        "section_provenance": section_provenance,
        "ai_usage": ai_usage,
        "recommendation_count": len(recommendations),
        "historical_context_summary": (historical_context or {}).get("summary", {})
        if isinstance(historical_context, dict)
        else {},
        "historical_context": historical_context or {},
        "evidence_ledger_summary": {
            "fact_count": len(evidence_ledger.get("facts", [])),
            "approved_number_count": len(evidence_ledger.get("approved_numbers", [])),
            "approved_period_count": len(evidence_ledger.get("approved_periods", [])),
            "approved_entity_count": len(evidence_ledger.get("approved_entities", [])),
        },
        "reasoning_state": reasoning_state.to_dict(),
        "analysis": analysis,
    }


def _analysis_section_provenance(
    *,
    analysis: dict[str, Any],
    analysis_source: str,
    validation_status: str,
    reasoning_state: ReasoningState,
    model_name: str | None,
) -> dict[str, dict[str, Any]]:
    """Build explicit per-field provenance for final strategic prose.

    Inputs: final analysis payload, source/status metadata, reasoning state and
    configured model name.
    Outputs: mapping from analysis field to provenance used by report badges.
    Assumptions: provenance is metadata only; it never rewrites reasoning text.
    """

    recovery = analysis.get("_strategic_recovery")
    recovery = recovery if isinstance(recovery, dict) else {}
    removed_fields = _field_prefixes(recovery.get("removed_fields", []))
    touched_fields = _field_prefixes(
        [
            *list(recovery.get("repaired_fields", []) or []),
            *list(recovery.get("sanitized_fields", []) or []),
        ]
    )
    deterministic = "deterministic" in analysis_source.casefold() or bool(recovery.get("degraded_mode"))
    fields: dict[str, dict[str, Any]] = {}
    for field_name in FINAL_NARRATIVE_FIELDS:
        value = analysis.get(field_name)
        if not _provenance_value_present(value):
            continue
        if deterministic or field_name in removed_fields:
            fields[field_name] = {
                "generated_by": "deterministic",
                "model": None,
                "generation_stage": (
                    "deterministic_replacement_after_validation"
                    if field_name in removed_fields
                    else "deterministic_fallback"
                ),
                "validation_status": "accepted",
            }
            continue
        stage = "strategic_synthesis"
        fields[field_name] = {
            "generated_by": "ollama",
            "model": _model_from_stage_results(reasoning_state, model_name),
            "generation_stage": stage,
            "validation_status": (
                "repaired_validated"
                if field_name in touched_fields or validation_status == "sanitized"
                else "validated"
            ),
        }
    return fields


def _ai_usage_diagnostics(
    *,
    reasoning_state: ReasoningState,
    section_provenance: dict[str, dict[str, Any]],
    model_name: str | None,
) -> dict[str, Any]:
    """Summarize actual Ollama calls and final AI-authored fields.

    Inputs: completed reasoning state, field provenance and configured model.
    Outputs: diagnostics persisted in strategic-analysis JSON.
    Assumptions: a successful response means a stage has Ollama telemetry and
    returned enough text to reach Python validation, whether accepted or not.
    """

    stage_results = list(reasoning_state.stage_results)
    successful = [
        stage
        for stage in stage_results
        if stage.telemetry.get("error_category") in {None, "validation_rejection"}
    ]
    ai_fields = [
        field
        for field, provenance in section_provenance.items()
        if provenance.get("generated_by") == "ollama"
    ]
    repaired = [
        stage
        for stage in stage_results
        if stage.stage_id == "strategic_synthesis"
        and (
            "repair" in stage.stage_name.casefold()
            or "regeneration" in stage.stage_name.casefold()
            or stage.telemetry.get("schema_retry_attempted")
        )
    ]
    return {
        "ollama_called": bool(stage_results),
        "model": _model_from_stage_results(reasoning_state, model_name),
        "model_calls": len(stage_results),
        "successful_responses": len(successful),
        "accepted_responses": len([stage for stage in stage_results if stage.accepted]),
        "repaired_responses": len(repaired),
        "rejected_responses": len([stage for stage in stage_results if not stage.accepted]),
        "final_report_fields_with_ai_output": ai_fields,
    }


def _field_prefixes(values: Any) -> set[str]:
    """Return root analysis field names from recovery paths.

    Inputs: recovery path list such as ``field[0].subfield``.
    Outputs: field names for provenance decisions.
    Assumptions: malformed values are ignored as plain metadata.
    """

    roots: set[str] = set()
    for value in values if isinstance(values, list) else []:
        root = str(value).split("[", 1)[0].split(".", 1)[0].strip()
        if root:
            roots.add(root)
    return roots


def _provenance_value_present(value: Any) -> bool:
    """Return whether one final field contains visible report content.

    Inputs: arbitrary analysis field value.
    Outputs: True when the value can reach executive output.
    Assumptions: empty strings/lists/dicts carry no final provenance.
    """

    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _model_from_stage_results(state: ReasoningState, fallback_model: str | None) -> str | None:
    """Return the exact model recorded by Ollama telemetry when available.

    Inputs: reasoning state and configured model fallback.
    Outputs: model name or None for deterministic-only outputs.
    Assumptions: OllamaClient telemetry records ``model`` for every request.
    """

    for stage in reversed(state.stage_results):
        model = stage.telemetry.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return str(fallback_model).strip() if fallback_model else None


def _empty_analysis() -> dict[str, Any]:
    """Return an empty final-analysis payload for rejected modular runs.

    Inputs: none.
    Outputs: analysis-shaped dictionary.
    Assumptions: rejected analyses must not appear as report-ready content.
    """

    return {
        "executive_summary": "",
        "key_findings": [],
        "root_causes": [],
        "financial_health_analysis": "",
        "kpi_analysis": "",
        "department_analysis": "",
        "anomaly_analysis": "",
        "recommendation_follow_up_analysis": "",
        "longitudinal_risk_analysis": "",
        "strategic_recommendations": [],
        "recommendations": [],
        "strategic_priorities": [],
        "missing_information": [],
        "historical_summary": "",
        "historical_trend_analysis": "",
        "narrative_evidence": {},
        "confidence": None,
        "reasoning_summary": "",
    }
