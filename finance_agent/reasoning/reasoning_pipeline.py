"""Modular multi-stage Ollama reasoning pipeline."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from finance_agent.analysis.analysis_models import StrategicAnalysisResult
from finance_agent.analysis.strategic_analysis import (
    build_evidence_ledger,
    estimate_tokens_from_text,
    strategic_analysis_json_schema,
    validate_evidence_bound_claims,
    validate_strategic_analysis_response,
    validate_user_facing_spanish,
)
from finance_agent.llm.ollama_client import OllamaError
from finance_agent.reasoning.reasoning_models import (
    ReasoningStageResult,
    ReasoningValidationResult,
)
from finance_agent.reasoning.reasoning_state import ReasoningState


STAGE_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "financial_performance": (
        "validated_financial_claims",
        "identified_financial_risks",
        "financial_opportunities",
        "open_questions",
    ),
    "historical_operational": (
        "validated_historical_claims",
        "trend_observations",
        "persistent_risks",
        "recommendation_effectiveness",
        "open_questions",
    ),
}


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
    state = ReasoningState(period_slug=period_slug, evidence_ledger=evidence_ledger)
    telemetry: dict[str, Any] = {
        "reasoning_pipeline": "modular_multi_stage",
        "stage_count": 3,
        "monolithic_prompt_baseline_characters": None,
    }
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
            ),
            validator=lambda text: validate_reasoning_stage_response(
                text,
                stage_id="financial_performance",
                evidence_ledger=evidence_ledger,
            ),
        )
        state.add_stage_result(financial_result)
        if not financial_result.accepted:
            return _rejected_result(finance_summary, period_slug, historical_context, evidence_ledger, state, telemetry)

        historical_result = _run_structured_stage(
            client=client,
            stage_id="historical_operational",
            stage_name="Historical & Operational Reasoning",
            prompt=build_historical_operational_prompt(
                evidence_ledger=evidence_ledger,
                historical_context=historical_context,
                state=state,
                period_slug=period_slug,
            ),
            validator=lambda text: validate_reasoning_stage_response(
                text,
                stage_id="historical_operational",
                evidence_ledger=evidence_ledger,
            ),
        )
        state.add_stage_result(historical_result)
        if not historical_result.accepted:
            return _rejected_result(finance_summary, period_slug, historical_context, evidence_ledger, state, telemetry)

        strategic_prompt = build_strategic_synthesis_prompt(
            state=state,
            finance_summary=finance_summary,
            period_slug=period_slug,
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
            ),
            response_format=strategic_analysis_json_schema(),
        )
        state.add_stage_result(strategic_result)
    except OllamaError as exc:
        errors = (str(exc),)
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
        )
        return StrategicAnalysisResult(
            analysis_document=document,
            accepted=False,
            validation_errors=errors,
            telemetry={**telemetry, "total_stage_time_seconds": time.perf_counter() - started},
        )

    accepted = state.stage_results[-1].accepted if state.stage_results else False
    analysis = state.stage_results[-1].payload if accepted else _empty_analysis()
    errors = state.stage_results[-1].validation_errors if state.stage_results else ("No reasoning stages ran.",)
    document = _analysis_document(
        period_slug=period_slug,
        report_period=str(finance_summary.get("report_period", period_slug)),
        ollama_available=True,
        validation_status="accepted" if accepted else "rejected",
        validation_errors=() if accepted else errors,
        analysis=analysis,
        historical_context=historical_context,
        evidence_ledger=evidence_ledger,
        reasoning_state=state,
    )
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=accepted,
        validation_errors=() if accepted else errors,
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
) -> str:
    """Build the Stage 1 financial-performance prompt.

    Inputs: ledger, current finance summary, anomaly report, and period slug.
    Outputs: compact strict-JSON prompt for current-performance reasoning.
    Assumptions: prompt includes only current financial facts and top anomalies.
    """

    facts = [
        _prompt_fact(fact)
        for fact in evidence_ledger.get("facts", [])
        if isinstance(fact, dict)
        and fact.get("category") == "current_finding"
        and (
            str(fact.get("evidence_id", "")).startswith("finance.")
            or str(fact.get("evidence_id", "")).startswith("anomaly.")
        )
    ][:28]
    context = {
        "period_slug": period_slug,
        "objective": "Qué está ocurriendo financieramente en el periodo actual.",
        "facts": facts,
        "approved_numbers": evidence_ledger.get("approved_numbers", []),
        "approved_periods": evidence_ledger.get("approved_periods", []),
        "approved_entities": evidence_ledger.get("approved_entities", []),
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
) -> str:
    """Build the Stage 2 historical/operational prompt.

    Inputs: ledger, compact historical context, and accepted Stage 1 state.
    Outputs: strict-JSON prompt focused on trends and persistence.
    Assumptions: only historical facts plus Stage 1 validated reasoning are sent.
    """

    facts = [
        _prompt_fact(fact)
        for fact in evidence_ledger.get("facts", [])
        if isinstance(fact, dict)
        and str(fact.get("category", "")).startswith(("kpi_trend", "recurring", "prior"))
    ][:30]
    context = {
        "period_slug": period_slug,
        "objective": "Cómo evolucionaron los riesgos y avances respecto de periodos previos.",
        "historical_facts": facts,
        "validated_stage_1": {
            "claims": state.validated_claims[:8],
            "risks": state.risks[:8],
            "opportunities": state.opportunities[:6],
            "open_questions": state.open_questions[:6],
        },
        "history_summary": (historical_context or {}).get("summary", {})
        if isinstance(historical_context, dict)
        else {},
        "approved_numbers": evidence_ledger.get("approved_numbers", []),
        "approved_periods": evidence_ledger.get("approved_periods", []),
        "approved_entities": evidence_ledger.get("approved_entities", []),
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
        "allowed_evidence_ids": sorted(state.evidence_references)[:80],
        "rules": (
            "Usa solo afirmaciones validadas en validated_reasoning_state.",
            "Copia números exactamente como aparecen en esas afirmaciones.",
            "Cita evidence_ids ya presentes en las afirmaciones validadas.",
        ),
    }
    return (
        _stage_prompt(
            stage_name="Strategic Synthesis",
            schema=(
                "Return the existing strategic-analysis JSON schema: executive_summary, "
                "key_findings, root_causes, financial_health_analysis, kpi_analysis, "
                "department_analysis, anomaly_analysis, recommendation_follow_up_analysis, "
                "longitudinal_risk_analysis, strategic_recommendations, "
                "strategic_priorities, missing_information, historical_summary, "
                "historical_trend_analysis, narrative_evidence, confidence, reasoning_summary."
            ),
            context=context,
        )
        + "\nStage 3 MUST NOT ask for or assume the full evidence ledger."
    )


def validate_reasoning_stage_response(
    response_text: str,
    *,
    stage_id: str,
    evidence_ledger: dict[str, Any],
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
    required = set(_stage_required_fields(stage_id))
    if set(payload) != required:
        return ReasoningValidationResult(
            False,
            None,
            (f"{stage_id} must contain exactly {sorted(required)}; received {sorted(payload)}",),
        )

    errors: list[str] = []
    for field_name in STAGE_TEXT_FIELDS[stage_id]:
        _validate_reasoning_items(field_name, payload.get(field_name), evidence_ledger, errors)
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be numeric between 0 and 1")
    if errors:
        return ReasoningValidationResult(False, None, tuple(dict.fromkeys(errors)))

    cleaned = {
        key: _clean_reasoning_items(value)
        if key in STAGE_TEXT_FIELDS[stage_id]
        else float(value)
        if key == "confidence"
        else value
        for key, value in payload.items()
    }
    return ReasoningValidationResult(True, cleaned, ())


def validate_strategic_synthesis_response(
    response_text: str,
    *,
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any],
    risk_summary: dict[str, Any],
    historical_context: dict[str, Any] | None,
    evidence_ledger: dict[str, Any],
) -> ReasoningValidationResult:
    """Validate Stage 3 final synthesis against existing Step-9 guards.

    Inputs: raw response text plus processed evidence contexts and ledger.
    Outputs: reasoning validation result.
    Assumptions: strategic synthesis reuses the same strict report-analysis
    schema so downstream report generation remains unchanged.
    """

    validation = validate_strategic_analysis_response(response_text)
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
        return ReasoningValidationResult(False, None, claim_errors)
    return ReasoningValidationResult(True, validation.analysis, ())


def _run_structured_stage(
    *,
    client: Any,
    stage_id: str,
    stage_name: str,
    prompt: str,
    validator: Any,
    response_format: dict[str, Any] | str = "json",
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
    finally:
        if previous_response_format is not None:
            setattr(client, "response_format", previous_response_format)

    validation_started = time.perf_counter()
    validation = validator(response)
    validation_time = time.perf_counter() - validation_started
    telemetry = {
        "stage_id": stage_id,
        "prompt_characters": len(prompt),
        "prompt_token_estimate": estimate_tokens_from_text(prompt),
        "json_validation_time_seconds": validation_time,
        "total_stage_time_seconds": time.perf_counter() - started,
        **ollama_telemetry,
    }
    return ReasoningStageResult(
        stage_id=stage_id,
        stage_name=stage_name,
        accepted=validation.is_valid,
        payload=validation.payload or {},
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
    Assumptions: every item must include text and evidence_ids.
    """

    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return
    if len(value) > 8:
        errors.append(f"{field_name} may contain at most 8 items")
    known_ids = {
        str(fact.get("evidence_id"))
        for fact in evidence_ledger.get("facts", [])
        if isinstance(fact, dict) and fact.get("evidence_id")
    }
    approved_numbers = set(str(number) for number in evidence_ledger.get("approved_numbers", []))
    approved_periods = set(str(period) for period in evidence_ledger.get("approved_periods", []))
    approved_entities = set(str(entity) for entity in evidence_ledger.get("approved_entities", []))
    for index, item in enumerate(value):
        prefix = f"{field_name}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        text = item.get("text") or item.get("claim") or item.get("question") or item.get("risk")
        if not isinstance(text, str) or not text.strip() or len(text) > 1200:
            errors.append(f"{prefix}.text must be non-empty bounded text")
            continue
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
        evidence_ids = item.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            errors.append(f"{prefix}.evidence_ids must be a non-empty list")
        else:
            for evidence_id in evidence_ids:
                if str(evidence_id) not in known_ids:
                    errors.append(f"{prefix} cites unknown evidence_id: {evidence_id}")


def _clean_reasoning_items(value: Any) -> list[dict[str, Any]]:
    """Clean validated reasoning items without altering meaning.

    Inputs: model item list.
    Outputs: normalized list with text, evidence_ids, and optional metadata.
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
        copied["evidence_ids"] = [str(evidence_id).strip() for evidence_id in item.get("evidence_ids", [])]
        cleaned.append(copied)
    return cleaned


def _numbers_in_text(text: str) -> set[str]:
    """Extract normalized explicit numeric claims from prose.

    Inputs: prose text.
    Outputs: set of number strings comparable to ledger approved values.
    Assumptions: years are handled separately as approved periods.
    """

    numbers: set[str] = set()
    for match in re.finditer(r"-?\d+(?:[\.,]\d+)?%?", text):
        value = match.group(0).replace(",", ".")
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
        "Usa solo hechos, números, periodos, entidades y evidence_ids presentes en el contexto.\n"
        "No calcules, no estimes, no redondees, no inventes causas ni resultados.\n"
        "Si una causa es incierta, márcala como hipótesis y cita evidence_ids.\n"
        "Devuelve JSON estricto únicamente.\n"
        f"SCHEMA:\n{schema}\n"
        "CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )


def _financial_stage_schema_text() -> str:
    """Return the Stage 1 JSON contract.

    Inputs: none.
    Outputs: compact schema instructions.
    Assumptions: each item includes text and evidence_ids.
    """

    return (
        "Object keys exactly: validated_financial_claims, identified_financial_risks, "
        "financial_opportunities, open_questions, confidence. Each list item is "
        "{text: Spanish string, evidence_ids: [ledger IDs]}. confidence is 0..1."
    )


def _historical_stage_schema_text() -> str:
    """Return the Stage 2 JSON contract.

    Inputs: none.
    Outputs: compact schema instructions.
    Assumptions: each item includes text and evidence_ids.
    """

    return (
        "Object keys exactly: validated_historical_claims, trend_observations, "
        "persistent_risks, recommendation_effectiveness, open_questions, confidence. "
        "Each list item is {text: Spanish string, evidence_ids: [ledger IDs]}. confidence is 0..1."
    )


def _stage_required_fields(stage_id: str) -> tuple[str, ...]:
    """Return required fields for a modular reasoning stage.

    Inputs: stage ID.
    Outputs: tuple of field names.
    Assumptions: only Stage 1 and Stage 2 use this smaller contract.
    """

    if stage_id == "financial_performance":
        return (
            "validated_financial_claims",
            "identified_financial_risks",
            "financial_opportunities",
            "open_questions",
            "confidence",
        )
    if stage_id == "historical_operational":
        return (
            "validated_historical_claims",
            "trend_observations",
            "persistent_risks",
            "recommendation_effectiveness",
            "open_questions",
            "confidence",
        )
    raise ValueError(f"Unknown reasoning stage: {stage_id}")


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
    )
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=False,
        validation_errors=errors,
        telemetry={**telemetry, "stage_telemetry": [stage.telemetry for stage in state.stage_results]},
    )


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
) -> dict[str, Any]:
    """Assemble a Step-9-compatible strategic-analysis document.

    Inputs: metadata, validation state, final analysis, history, ledger and
    reasoning state.
    Outputs: JSON-compatible document used by existing report generation.
    Assumptions: accepted reports consume ``analysis`` exactly as before.
    """

    recommendations = analysis.get("recommendations", analysis.get("strategic_recommendations", []))
    recommendations = recommendations if isinstance(recommendations, list) else []
    return {
        "analysis_id": f"STRATEGIC-ANALYSIS-{period_slug.upper().replace('_', '-')}",
        "period_slug": period_slug,
        "report_period": report_period,
        "analysis_source": "ollama_modular_reasoning",
        "ollama_available": ollama_available,
        "validation_status": validation_status,
        "analysis_generated": validation_status == "accepted",
        "validation_errors": list(validation_errors),
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
