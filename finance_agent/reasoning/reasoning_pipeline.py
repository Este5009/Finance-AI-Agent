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
            response_format=reasoning_stage_json_schema("financial_performance"),
            stage_timeout_seconds=stage_timeout_seconds,
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
            response_format=reasoning_stage_json_schema("historical_operational"),
            stage_timeout_seconds=stage_timeout_seconds,
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
            stage_timeout_seconds=stage_timeout_seconds,
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

    errors: list[str] = []
    for field_name in STAGE_TEXT_FIELDS[stage_id]:
        _validate_reasoning_items(field_name, payload.get(field_name), evidence_ledger, errors)
    if errors:
        return ReasoningValidationResult(False, None, tuple(dict.fromkeys(errors)))

    cleaned = {
        key: _clean_reasoning_items(value, field_name=key)
        for key, value in payload.items()
    }
    if normalizations:
        cleaned["_schema_normalizations"] = normalizations
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
    stage_timeout_seconds: float | None = None,
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
    if stage_timeout_seconds is not None and elapsed_after_generation > stage_timeout_seconds:
        telemetry = {
            "stage_id": stage_id,
            "prompt_characters": len(prompt),
            "prompt_token_estimate": estimate_tokens_from_text(prompt),
            "json_validation_time_seconds": 0.0,
            "total_stage_time_seconds": elapsed_after_generation,
            "timeout_error_category": "stage_timeout",
            "error_category": "stage_timeout",
            **ollama_telemetry,
        }
        return ReasoningStageResult(
            stage_id=stage_id,
            stage_name=stage_name,
            accepted=False,
            payload={},
            validation_errors=(
                f"{stage_name} exceeded stage timeout of {stage_timeout_seconds:.1f}s.",
            ),
            telemetry=telemetry,
        )

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
        "schema_retry_attempted": schema_retry_attempted,
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
        required_keys = {"text", "evidence_ids"}
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
        evidence_ids = item.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            errors.append(f"{prefix}.evidence_ids must be a non-empty list")
        else:
            for evidence_id in evidence_ids:
                if str(evidence_id) not in known_ids:
                    errors.append(f"{prefix} cites unknown evidence_id: {evidence_id}")


def _clean_reasoning_items(value: Any, *, field_name: str) -> list[dict[str, Any]]:
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
        "No agregues ni elimines afirmaciones, riesgos, oportunidades, preguntas, números ni evidence_ids.\n"
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
    Assumptions: each item includes text and evidence_ids.
    """

    return (
        "Return exactly one JSON object with exactly these top-level keys and no others: "
        "claims, risks, opportunities, open_questions. "
        "claims item: {text: Spanish string, evidence_ids: [valid IDs], confidence: 0..1, claim_type: fact|interpretation|hypothesis}. "
        "risks/opportunities item: {text: Spanish string, evidence_ids: [valid IDs], confidence: 0..1}. "
        "open_questions item: {text: Spanish string, evidence_ids: [valid IDs]}. "
        "Do not include examples, fake values, markdown, or prose outside JSON."
    )


def _historical_stage_schema_text() -> str:
    """Return the Stage 2 JSON contract.

    Inputs: none.
    Outputs: compact schema instructions.
    Assumptions: each item includes text and evidence_ids.
    """

    return (
        "Return exactly one JSON object with exactly these top-level keys and no others: "
        "claims, risks, opportunities, open_questions. "
        "claims item: {text: Spanish string, evidence_ids: [valid IDs], confidence: 0..1, claim_type: fact|interpretation|hypothesis}. "
        "risks/opportunities item: {text: Spanish string, evidence_ids: [valid IDs], confidence: 0..1}. "
        "open_questions item: {text: Spanish string, evidence_ids: [valid IDs]}. "
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


def reasoning_stage_json_schema(stage_id: str) -> dict[str, Any]:
    """Return the provider JSON schema for Stage 1/2 reasoning.

    Inputs: stage ID.
    Outputs: JSON schema dictionary passed to Ollama's ``format`` parameter.
    Assumptions: Stage-specific semantics are prompt-driven; the shape remains
    minimal and stable across financial and historical reasoning.
    """

    if stage_id not in {"financial_performance", "historical_operational"}:
        raise ValueError(f"Unknown reasoning stage schema: {stage_id}")
    text_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence_ids"],
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 1200},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
            },
        },
    }
    confidence_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence_ids", "confidence"],
        "properties": {
            **text_item["properties"],
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    claim_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "evidence_ids", "confidence", "claim_type"],
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
