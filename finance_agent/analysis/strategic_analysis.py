"""Ollama-backed strategic financial analysis over processed evidence."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

from finance_agent.analysis.analysis_models import (
    AnalysisRunSummary,
    AnalysisValidationResult,
    StrategicAnalysisResult,
)
from finance_agent.common.context_optimization import (
    compact_json_size,
    deduplicate_dicts,
    estimate_tokens_from_text,
    merge_telemetry,
    rank_anomalies,
)
from finance_agent.llm.ollama_client import OllamaError


MAX_RESPONSE_CHARACTERS = 80_000
MAX_TEXT_LENGTH = 1_200
MAX_LIST_ITEMS = 8
MAX_RECOMMENDATIONS = 8
REQUIRED_ANALYSIS_FIELDS = frozenset(
    {
        "executive_summary",
        "key_findings",
        "root_causes",
        "financial_health_analysis",
        "kpi_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "strategic_recommendations",
        "strategic_priorities",
        "missing_information",
        "historical_summary",
        "historical_trend_analysis",
        "narrative_evidence",
        "confidence",
        "reasoning_summary",
    }
)
RECOMMENDATION_FIELDS = frozenset(
    {
        "priority",
        "action",
        "rationale",
        "supporting_evidence",
        "expected_impact",
        "evidence_ids",
        "confidence",
    }
)
ALLOWED_RECOMMENDATION_PRIORITIES = frozenset(
    {"critical", "high", "medium", "low"}
)
SPANISH_TEXT_FIELDS = frozenset(
    {
        "executive_summary",
        "key_findings",
        "root_causes",
        "strategic_priorities",
        "missing_information",
        "historical_summary",
        "historical_trend_analysis",
        "financial_health_analysis",
        "kpi_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "reasoning_summary",
        "strategic_recommendations.action",
        "strategic_recommendations.rationale",
        "strategic_recommendations.supporting_evidence",
        "strategic_recommendations.expected_impact",
    }
)
ALLOWED_ENGLISH_ACRONYMS = frozenset({"KPI", "USD", "EBITDA", "PDF", "HTML", "JSON", "CSV", "LLM"})
COMMON_ENGLISH_WORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "without",
        "cash",
        "flow",
        "payroll",
        "budget",
        "revenue",
        "expense",
        "expenses",
        "collection",
        "collections",
        "rate",
        "risk",
        "risks",
        "recommendation",
        "recommendations",
        "department",
        "departments",
        "evidence",
        "analysis",
        "shows",
        "indicate",
        "indicates",
        "improve",
        "improved",
        "review",
        "requires",
        "required",
        "management",
        "operational",
        "financial",
        "performance",
        "variance",
        "variances",
        "target",
        "targets",
        "root",
        "cause",
        "causes",
    }
)
COMMON_SPANISH_WORDS = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "en",
        "con",
        "sin",
        "para",
        "por",
        "que",
        "y",
        "o",
        "financiero",
        "financiera",
        "nómina",
        "ingresos",
        "gastos",
        "cobranza",
        "riesgo",
        "riesgos",
        "presupuesto",
        "presupuestario",
        "evidencia",
        "recomendación",
        "recomendaciones",
        "departamento",
        "departamentos",
        "flujo",
        "caja",
        "meta",
        "metas",
        "mejorar",
        "revisar",
        "control",
        "gestión",
        "análisis",
        "causa",
        "causas",
    }
)
GENERIC_PLACEHOLDER_PHRASES = frozenset(
    {
        "no se encontraron datos históricos disponibles",
        "sin datos suficientes para analizar",
        "se requiere más información",
        "análisis no disponible",
        "información no disponible",
        "análisis español",
        "acción concreta en español",
        "racional en español",
        "referencia breve a evidencia",
        "impacto esperado en español",
        "hallazgo específico en español",
        "causa probable en español",
        "prioridad ejecutiva en español",
        "evidencia faltante en español",
        "resumen breve en español",
    }
)


def _iter_nested_values(value: Any) -> list[Any]:
    """Flatten nested JSON-compatible values.

    Inputs: arbitrary nested context value.
    Outputs: scalar-ish values found recursively.
    Assumptions: used only for validation token extraction, not rendering.
    """

    if isinstance(value, dict):
        values: list[Any] = []
        for child in value.values():
            values.extend(_iter_nested_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_iter_nested_values(child))
        return values
    return [value]


def _normalize_claim_number(token: str) -> str:
    """Normalize a prose/context number token for comparison.

    Inputs: raw number token, possibly with decimal comma or percent sign.
    Outputs: normalized token using dot decimals.
    Assumptions: this is only for validation equality, not calculation.
    """

    text = token.strip()
    suffix = "%" if text.endswith("%") else ""
    core = text[:-1] if suffix else text
    if "," in core and "." not in core:
        parts = core.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and suffix != "%":
            core = "".join(parts)
        else:
            core = core.replace(",", ".")
    else:
        core = core.replace(",", "")
    return f"{core}{suffix}"


def _extract_supported_claim_tokens(*contexts: dict[str, Any]) -> dict[str, set[str]]:
    """Extract numbers, periods, and department names supported by evidence.

    Inputs: processed context dictionaries.
    Outputs: token sets used for conservative claim validation.
    Assumptions: supported facts must appear somewhere in processed context.
    """

    text = json.dumps(contexts, ensure_ascii=False)
    numbers = {_normalize_claim_number(match.group(0)) for match in re.finditer(r"-?\d+(?:[\.,]\d+)?%?", text)}
    # Ratios are stored as decimals in processed JSON, while executive prose may
    # cite the equivalent percentage. Allow both forms without allowing new math.
    for number in list(numbers):
        if number.endswith("%"):
            continue
        try:
            value = float(number)
        except ValueError:
            continue
        if 0 <= value <= 1:
            numbers.add(f"{value:.0%}")
            numbers.add(f"{value:.1%}")
            numbers.add(f"{value:.2%}")
    periods = set(re.findall(r"20\d{2}[-_]\d{2}|20\d{2}", text))
    departments: set[str] = set()
    for context in contexts:
        for value in _iter_nested_values(context):
            if isinstance(value, str) and value.strip() and len(value.strip()) <= 80:
                lowered = value.casefold()
                if any(term in lowered for term in ("sciences", "engineering", "business", "administration", "services", "humanities", "tecnología", "instalaciones")):
                    departments.add(value.strip())
    return {"numbers": numbers, "periods": periods, "departments": departments}


def _analysis_visible_text_by_field(analysis: dict[str, Any]) -> dict[str, str]:
    """Return user-facing narrative text grouped by field.

    Inputs: accepted/cleaned analysis payload.
    Outputs: mapping of field path to text.
    Assumptions: internal JSON keys and numeric confidence fields are excluded.
    """

    fields: dict[str, str] = {}
    for field_name in (
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
        "strategic_priorities",
        "missing_information",
        "reasoning_summary",
    ):
        value = analysis.get(field_name)
        if isinstance(value, list):
            fields[field_name] = ". ".join(str(item) for item in value)
        elif isinstance(value, str):
            fields[field_name] = value
    for index, recommendation in enumerate(analysis.get("strategic_recommendations", [])):
        if isinstance(recommendation, dict):
            fields[f"strategic_recommendations[{index}]"] = " ".join(
                str(recommendation.get(field_name, ""))
                for field_name in ("action", "rationale", "supporting_evidence", "expected_impact")
            )
    return fields


def strategic_analysis_json_schema() -> dict[str, Any]:
    """Return the strict JSON schema requested from Ollama when supported.

    Inputs: none.
    Outputs: JSON-schema-like dictionary for Ollama's format parameter.
    Assumptions: Python validation below remains authoritative even when model
    decoding is schema-constrained.
    """

    string_schema = {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_LENGTH}
    string_array = {
        "type": "array",
        "minItems": 0,
        "maxItems": MAX_LIST_ITEMS,
        "items": string_schema,
    }
    evidence_array = {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_LIST_ITEMS,
        "items": {"type": "string", "minLength": 1, "maxLength": 160},
    }
    recommendation_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(RECOMMENDATION_FIELDS),
        "properties": {
            "priority": {"type": "string", "enum": sorted(ALLOWED_RECOMMENDATION_PRIORITIES)},
            "action": string_schema,
            "rationale": string_schema,
            "supporting_evidence": string_schema,
            "expected_impact": string_schema,
            "evidence_ids": evidence_array,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    narrative_properties = {
        field: evidence_array
        for field in (
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
            "strategic_priorities",
            "missing_information",
            "reasoning_summary",
        )
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_ANALYSIS_FIELDS),
        "properties": {
            "executive_summary": string_schema,
            "key_findings": string_array,
            "root_causes": string_array,
            "financial_health_analysis": string_schema,
            "kpi_analysis": string_schema,
            "historical_summary": string_schema,
            "historical_trend_analysis": string_schema,
            "department_analysis": string_schema,
            "anomaly_analysis": string_schema,
            "recommendation_follow_up_analysis": string_schema,
            "longitudinal_risk_analysis": string_schema,
            "strategic_recommendations": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_RECOMMENDATIONS,
                "items": recommendation_schema,
            },
            "strategic_priorities": string_array,
            "missing_information": string_array,
            "narrative_evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(narrative_properties),
                "properties": narrative_properties,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning_summary": string_schema,
        },
    }


def validate_evidence_bound_claims(
    analysis: dict[str, Any],
    *,
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any],
    risk_summary: dict[str, Any],
    historical_context: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate narrative claims stay bound to supplied evidence.

    Inputs: analysis and processed evidence contexts.
    Outputs: field-specific claim validation errors.
    Assumptions: this guard checks unsupported explicit numbers, periods,
    department names, generic filler, and missing narrative evidence IDs.
    """

    supported = _extract_supported_claim_tokens(
        finance_summary,
        anomaly_report,
        evidence_package,
        risk_summary,
        historical_context or {},
    )
    errors: list[str] = []
    narrative_evidence = analysis.get("narrative_evidence", {})
    narrative_evidence = narrative_evidence if isinstance(narrative_evidence, dict) else {}
    for field_name, text in _analysis_visible_text_by_field(analysis).items():
        lowered = text.casefold()
        if any(phrase in lowered for phrase in GENERIC_PLACEHOLDER_PHRASES):
            errors.append(f"{field_name} contains generic placeholder prose")
        for number in {_normalize_claim_number(match.group(0)) for match in re.finditer(r"-?\d+(?:[\.,]\d+)?%?", text)}:
            if number not in supported["numbers"]:
                errors.append(f"{field_name} contains unsupported number: {number}")
        for period in set(re.findall(r"20\d{2}[-_]\d{2}|20\d{2}", text)):
            if period not in supported["periods"]:
                errors.append(f"{field_name} contains unsupported period: {period}")
        named_entities = set(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z&][A-Za-z&]+){0,2}\b", text))
        named_entities.update(re.findall(r"\b(?:El|La|Los|Las)\s+[A-Z][A-Za-z]+(?:\s+(?:de|del)\s+[A-Z][A-Za-z]+)+\b", text))
        for department in named_entities:
            words = department.split()
            candidate = " ".join(words[1:]) if words and words[0] in {"El", "La", "Los", "Las"} else department
            if any(word in ALLOWED_ENGLISH_ACRONYMS for word in words):
                continue
            if department in supported["departments"] or candidate in supported["departments"]:
                continue
            if department in {"Ollama", "Python", "KPI", "USD", "PDF", "HTML", "JSON", "CSV"}:
                continue
            # Avoid treating sentence-start Spanish words as departments.
            if department.casefold() in COMMON_SPANISH_WORDS:
                continue
            if department in {"El", "La", "Los", "Las", "No", "Sin"}:
                continue
            if len(candidate.split()) == 1:
                continue
            errors.append(f"{field_name} contains unsupported named entity: {candidate}")
        base = field_name.split("[")[0]
        if base != "strategic_recommendations" and base in SPANISH_TEXT_FIELDS:
            if not narrative_evidence.get(base):
                errors.append(f"{base} must retain source/evidence IDs")
    for index, recommendation in enumerate(analysis.get("strategic_recommendations", [])):
        if isinstance(recommendation, dict) and not recommendation.get("evidence_ids"):
            errors.append(f"strategic_recommendations[{index}] must retain source/evidence IDs")
    return tuple(dict.fromkeys(errors))


class StrategicAnalysisClient(Protocol):
    """Minimal client contract required by the Step 9 analysis layer."""

    def is_available(self) -> bool:
        """Return whether the local model service can be reached."""

    def generate(self, prompt: str) -> str:
        """Return one model-generated strict-JSON analysis."""


def _bounded_string(value: Any, *, maximum: int = MAX_TEXT_LENGTH) -> bool:
    """Validate a model-authored bounded non-empty string.

    Inputs: untrusted value and maximum length.
    Outputs: True when value is a non-empty string within the limit.
    Assumptions: surrounding whitespace has no semantic value.
    """

    return isinstance(value, str) and bool(value.strip()) and len(value.strip()) <= maximum


def _valid_confidence(value: Any) -> bool:
    """Validate a confidence score.

    Inputs: untrusted confidence value.
    Outputs: True for numeric values in the inclusive 0..1 range.
    Assumptions: booleans are not valid numeric confidence scores.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= float(value) <= 1
    )


def _language_tokens(text: str) -> list[str]:
    """Return alphabetic language tokens from user-facing prose.

    Inputs: user-facing text.
    Outputs: lowercase tokens excluding allowed acronyms.
    Assumptions: numeric values and common finance acronyms are language-neutral.
    """

    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text)
    return [
        token.casefold()
        for token in tokens
        if token.upper() not in ALLOWED_ENGLISH_ACRONYMS and len(token) > 2
    ]


def _looks_professional_spanish(text: str) -> bool:
    """Heuristically validate that substantial prose is Spanish.

    Inputs: one user-facing text field.
    Outputs: True when text is short/neutral or Spanish-dominant.
    Assumptions: this is a conservative guard, not a full language detector.
    """

    stripped = text.strip()
    tokens = _language_tokens(stripped)
    if len(tokens) < 4:
        return True
    english_hits = sum(1 for token in tokens if token in COMMON_ENGLISH_WORDS)
    spanish_hits = sum(1 for token in tokens if token in COMMON_SPANISH_WORDS)
    accented = bool(re.search(r"[áéíóúÁÉÍÓÚñÑ]", stripped))
    if english_hits >= 3 and english_hits > spanish_hits:
        return False
    if english_hits >= 2 and not accented and spanish_hits == 0:
        return False
    return True


def _validate_spanish_text_field(
    field_path: str,
    value: Any,
    errors: list[str],
) -> None:
    """Validate one user-facing field is Spanish.

    Inputs: field path, value, and mutable error list.
    Outputs: appends field-specific errors.
    Assumptions: internal JSON keys remain English and are not checked here.
    """

    if isinstance(value, str):
        if value.strip() and not _looks_professional_spanish(value):
            errors.append(f"{field_path} must be professional Spanish")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_spanish_text_field(f"{field_path}[{index}]", item, errors)


def validate_user_facing_spanish(analysis: dict[str, Any]) -> tuple[str, ...]:
    """Validate all model-authored user-facing analysis text is Spanish.

    Inputs: cleaned strategic analysis payload.
    Outputs: tuple of field-specific language errors.
    Assumptions: common acronyms such as KPI, USD, PDF, HTML and JSON are allowed.
    """

    errors: list[str] = []
    for field_name in (
        "executive_summary",
        "key_findings",
        "root_causes",
        "strategic_priorities",
        "missing_information",
        "historical_summary",
        "historical_trend_analysis",
        "financial_health_analysis",
        "kpi_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "reasoning_summary",
    ):
        _validate_spanish_text_field(field_name, analysis.get(field_name), errors)
    recommendations = analysis.get("strategic_recommendations", analysis.get("recommendations", []))
    if isinstance(recommendations, list):
        for index, recommendation in enumerate(recommendations):
            if not isinstance(recommendation, dict):
                continue
            for field_name in ("action", "rationale", "supporting_evidence", "expected_impact"):
                _validate_spanish_text_field(
                    f"recommendations[{index}].{field_name}",
                    recommendation.get(field_name),
                    errors,
                )
    return tuple(errors)


def _validate_string_list(
    payload: dict[str, Any],
    field_name: str,
    errors: list[str],
) -> None:
    """Validate a required bounded list of strings.

    Inputs: model payload, field name, and mutable error list.
    Outputs: appends validation errors when field is malformed.
    Assumptions: concise strings are enough for downstream reports.
    """

    value = payload.get(field_name)
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return
    if len(value) > MAX_LIST_ITEMS:
        errors.append(f"{field_name} may contain at most {MAX_LIST_ITEMS} items")
    for index, item in enumerate(value):
        if not _bounded_string(item):
            errors.append(
                f"{field_name}[{index}] must be non-empty text up to {MAX_TEXT_LENGTH} characters"
            )


def _validate_recommendations(payload: dict[str, Any], errors: list[str]) -> None:
    """Validate model-authored recommendations.

    Inputs: model payload and mutable error list.
    Outputs: appends validation errors for malformed recommendations.
    Assumptions: recommendations are actions, not financial data mutations.
    """

    recommendations = payload.get("strategic_recommendations")
    if not isinstance(recommendations, list):
        errors.append("strategic_recommendations must be a list")
        return
    if not recommendations:
        errors.append("strategic_recommendations must contain at least one item")
    if len(recommendations) > MAX_RECOMMENDATIONS:
        errors.append(
            f"strategic_recommendations may contain at most {MAX_RECOMMENDATIONS} items"
        )
    for index, recommendation in enumerate(recommendations):
        prefix = f"strategic_recommendations[{index}]"
        if not isinstance(recommendation, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(recommendation) != RECOMMENDATION_FIELDS:
            errors.append(
                f"{prefix} must contain exactly {sorted(RECOMMENDATION_FIELDS)}"
            )
            continue
        if recommendation["priority"] not in ALLOWED_RECOMMENDATION_PRIORITIES:
            errors.append(f"{prefix}.priority is not allowed")
        for field_name in (
            "action",
            "rationale",
            "supporting_evidence",
            "expected_impact",
        ):
            if not _bounded_string(recommendation[field_name]):
                errors.append(
                    f"{prefix}.{field_name} must be non-empty text up to {MAX_TEXT_LENGTH} characters"
                )
        if not isinstance(recommendation["evidence_ids"], list):
            errors.append(f"{prefix}.evidence_ids must be a list")
        else:
            for evidence_index, evidence_id in enumerate(recommendation["evidence_ids"]):
                if not _bounded_string(evidence_id, maximum=160):
                    errors.append(f"{prefix}.evidence_ids[{evidence_index}] must be bounded text")
        if not _valid_confidence(recommendation["confidence"]):
            errors.append(f"{prefix}.confidence must be between 0 and 1")


def validate_strategic_analysis_response(
    response_text: str,
    *,
    require_spanish: bool = True,
) -> AnalysisValidationResult:
    """Parse and validate strict JSON returned by Ollama.

    Inputs: raw model response text and Spanish validation flag.
    Outputs: validation result with accepted analysis or rejection errors.
    Assumptions: markdown fences/prose are invalid because strict JSON is required.
    """

    if not isinstance(response_text, str):
        return AnalysisValidationResult(False, None, ("response must be text",))
    if len(response_text) > MAX_RESPONSE_CHARACTERS:
        return AnalysisValidationResult(
            False,
            None,
            ("response exceeds maximum character count",),
        )
    try:
        payload = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return AnalysisValidationResult(False, None, ("response is not strict JSON",))
    if not isinstance(payload, dict):
        return AnalysisValidationResult(False, None, ("response root must be an object",))
    # Some local models add a harmless top-level envelope. Unwrap only this
    # exact non-semantic case, then validate the inner object normally.
    if set(payload) == {"analysis"} and isinstance(payload.get("analysis"), dict):
        payload = payload["analysis"]
    if set(payload) != REQUIRED_ANALYSIS_FIELDS:
        return AnalysisValidationResult(
            False,
            None,
            (
                "response must contain exactly "
                f"{sorted(REQUIRED_ANALYSIS_FIELDS)}; received {sorted(payload)}",
            ),
        )

    errors: list[str] = []
    if not _bounded_string(payload["executive_summary"]):
        errors.append("executive_summary must be non-empty bounded text")
    if not _bounded_string(payload["reasoning_summary"]):
        errors.append("reasoning_summary must be non-empty bounded text")
    for field_name in (
        "financial_health_analysis",
        "kpi_analysis",
        "historical_summary",
        "historical_trend_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
    ):
        if not _bounded_string(payload[field_name]):
            errors.append(f"{field_name} must be non-empty bounded text")
    for field_name in (
        "key_findings",
        "root_causes",
        "strategic_priorities",
        "missing_information",
    ):
        _validate_string_list(payload, field_name, errors)
    if not isinstance(payload["narrative_evidence"], dict):
        errors.append("narrative_evidence must be an object")
    else:
        for field_name in SPANISH_TEXT_FIELDS:
            base = field_name.split(".")[0]
            if base == "strategic_recommendations":
                continue
            evidence_ids = payload["narrative_evidence"].get(base)
            if not isinstance(evidence_ids, list):
                errors.append(f"narrative_evidence.{base} must be a list")
            else:
                for index, evidence_id in enumerate(evidence_ids):
                    if not _bounded_string(evidence_id, maximum=160):
                        errors.append(f"narrative_evidence.{base}[{index}] must be bounded text")
    _validate_recommendations(payload, errors)
    if not _valid_confidence(payload["confidence"]):
        errors.append("confidence must be numeric between 0 and 1")
    if errors:
        return AnalysisValidationResult(False, None, tuple(errors))

    # Normalize strings and confidence after validation. This cleans harmless
    # whitespace without changing model-authored financial meaning.
    cleaned = {
        "executive_summary": payload["executive_summary"].strip(),
        "key_findings": [item.strip() for item in payload["key_findings"]],
        "root_causes": [item.strip() for item in payload["root_causes"]],
        "strategic_recommendations": [
            {
                "priority": item["priority"],
                "action": item["action"].strip(),
                "rationale": item["rationale"].strip(),
                "supporting_evidence": item["supporting_evidence"].strip(),
                "expected_impact": item["expected_impact"].strip(),
                "evidence_ids": [str(evidence_id).strip() for evidence_id in item["evidence_ids"]],
                "confidence": float(item["confidence"]),
            }
            for item in payload["strategic_recommendations"]
        ],
        "strategic_priorities": [
            item.strip() for item in payload["strategic_priorities"]
        ],
        "missing_information": [item.strip() for item in payload["missing_information"]],
        "financial_health_analysis": payload["financial_health_analysis"].strip(),
        "kpi_analysis": payload["kpi_analysis"].strip(),
        "historical_summary": payload["historical_summary"].strip(),
        "historical_trend_analysis": payload["historical_trend_analysis"].strip(),
        "department_analysis": payload["department_analysis"].strip(),
        "anomaly_analysis": payload["anomaly_analysis"].strip(),
        "recommendation_follow_up_analysis": payload["recommendation_follow_up_analysis"].strip(),
        "longitudinal_risk_analysis": payload["longitudinal_risk_analysis"].strip(),
        "narrative_evidence": {
            key: [str(evidence_id).strip() for evidence_id in value]
            for key, value in payload["narrative_evidence"].items()
            if isinstance(value, list)
        },
        "confidence": float(payload["confidence"]),
        "reasoning_summary": payload["reasoning_summary"].strip(),
    }
    cleaned["recommendations"] = cleaned["strategic_recommendations"]
    language_errors = validate_user_facing_spanish(cleaned) if require_spanish else ()
    if language_errors:
        return AnalysisValidationResult(False, None, language_errors)
    return AnalysisValidationResult(True, cleaned, ())


def _evidence_supports_payroll_field(
    evidence_package: dict[str, Any],
    field_name: str,
) -> bool:
    """Check whether Step 8 evidence contains a payroll field.

    Inputs: evidence package and payroll field name.
    Outputs: True when a retrieved payroll breakdown includes the field.
    Assumptions: this is used only to remove false missing-evidence claims.
    """

    for package in evidence_package.get("evidence_packages", []):
        if not isinstance(package, dict):
            continue
        evidence = package.get("retrieved_evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        data = evidence.get("data", {})
        data = data if isinstance(data, dict) else {}
        breakdown = data.get("payroll_breakdown", [])
        if not isinstance(breakdown, list):
            continue
        for row in breakdown:
            if isinstance(row, dict) and row.get(field_name) not in {None, ""}:
                return True
    return False


def _processed_anomaly_data_exists(anomaly_report: dict[str, Any]) -> bool:
    """Check whether a processed anomaly artifact is present for the period.

    Inputs: Step 4 anomaly report document.
    Outputs: True when anomaly data exists, even if the count is zero.
    Assumptions: the presence of processed anomaly fields is authoritative enough
    to reject a model claim that anomaly data itself is missing.
    """

    return any(
        key in anomaly_report
        for key in ("total_anomalies", "anomalies", "anomalies_by_severity")
    )


def _processed_cash_flow_data_exists(finance_summary: dict[str, Any]) -> bool:
    """Check whether processed cash-flow values exist in the finance summary.

    Inputs: Step 3 finance summary document.
    Outputs: True when the cash-flow section contains at least one populated value.
    Assumptions: Python-calculated cash-flow fields are evidence, not LLM output.
    """

    finance = finance_summary.get("finance_summary", {})
    finance = finance if isinstance(finance, dict) else {}
    cash_flow = finance.get("cash_flow", {})
    if not isinstance(cash_flow, dict):
        return False
    return any(value not in {None, ""} for value in cash_flow.values())


def _evidence_has_source_keyword(
    evidence_package: dict[str, Any],
    keyword: str,
) -> bool:
    """Search source references for one processed-evidence keyword.

    Inputs: Step 8 evidence package and a lowercase keyword.
    Outputs: True when any source reference contains the keyword.
    Assumptions: source references are lightweight provenance, not business logic.
    """

    lowered_keyword = keyword.casefold()
    for package in evidence_package.get("evidence_packages", []):
        if not isinstance(package, dict):
            continue
        for source in package.get("source_references", []):
            if lowered_keyword in str(source).casefold():
                return True
        evidence = package.get("retrieved_evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        for source in evidence.get("source_references", []):
            if lowered_keyword in str(source).casefold():
                return True
    return False


def _remove_false_missing_information(
    analysis: dict[str, Any],
    evidence_package: dict[str, Any],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
) -> dict[str, Any]:
    """Remove missing-information items contradicted by retrieved evidence.

    Inputs: validated analysis, Step 8 evidence, and processed finance/anomaly artifacts.
    Outputs: cleaned analysis preserving all other model-authored content.
    Assumptions: only objective availability claims are filtered, never reasoning.
    """

    missing = analysis.get("missing_information", [])
    if not isinstance(missing, list):
        return analysis
    has_headcount = _evidence_supports_payroll_field(evidence_package, "headcount_fte")
    has_payroll_amount = _evidence_supports_payroll_field(
        evidence_package,
        "payroll_amount",
    )
    has_overtime = _evidence_supports_payroll_field(evidence_package, "overtime")
    has_benefits = _evidence_supports_payroll_field(evidence_package, "benefits")
    has_anomaly_data = _processed_anomaly_data_exists(
        anomaly_report
    ) or _evidence_has_source_keyword(evidence_package, "anomaly")
    has_cash_flow = _processed_cash_flow_data_exists(
        finance_summary
    ) or _evidence_has_source_keyword(evidence_package, "cash_flow")
    cleaned_missing: list[str] = []
    for item in missing:
        lowered = str(item).casefold()
        # Ollama can overstate "missing data" even when Python already supplied
        # authoritative processed artifacts. These guards only remove objective
        # availability claims; they do not edit findings or recommendations.
        if has_anomaly_data and "anomal" in lowered:
            continue
        cash_flow_terms = ("cash flow", "cash-flow", "cashflow", "flujo de caja", "caja")
        if has_cash_flow and any(term in lowered for term in cash_flow_terms):
            continue
        if has_headcount and ("headcount" in lowered or "plantilla" in lowered):
            continue
        if has_payroll_amount and "payroll breakdown" in lowered:
            continue
        if has_payroll_amount and "payroll amount" in lowered:
            continue
        if has_overtime and ("overtime" in lowered or "horas extra" in lowered):
            continue
        if has_benefits and ("benefit" in lowered or "beneficio" in lowered):
            continue
        cleaned_missing.append(item)
    return {**analysis, "missing_information": cleaned_missing}


def _compact_finance_summary(
    finance_document: dict[str, Any],
    *,
    deduplicate_context: bool = True,
) -> dict[str, Any]:
    """Select authoritative calculated values for analysis context.

    Inputs: processed finance summary JSON.
    Outputs: compact scalar context.
    Assumptions: Python-calculated values are source-of-truth and not modified.
    """

    finance = finance_document.get("finance_summary", {})
    finance = finance if isinstance(finance, dict) else {}
    budget = finance.get("budget_vs_actual", {})
    payments = finance.get("student_payments", {})
    cash = finance.get("cash_flow", {})
    top_departments = finance_document.get("department_summary", [])[:8]
    top_categories = finance_document.get("category_summary", [])[:10]
    top_departments = top_departments if isinstance(top_departments, list) else []
    top_categories = top_categories if isinstance(top_categories, list) else []
    if deduplicate_context:
        top_departments = deduplicate_dicts(
            top_departments,
            key_fields=("department", "period", "actual_expense", "total_payroll"),
        )
        top_categories = deduplicate_dicts(
            top_categories,
            key_fields=("expense_category", "category", "amount", "actual_expense"),
        )
    return {
        "report_period": finance_document.get("report_period"),
        "period_scope": finance_document.get("period_scope"),
        "metrics": {
            "total_revenue": finance.get("total_revenue"),
            "total_expenses": finance.get("total_expenses"),
            "net_operating_result": finance.get("net_operating_result"),
            "payroll_total": finance.get("payroll_total"),
            "payroll_percentage_of_revenue": finance.get(
                "payroll_percentage_of_revenue"
            ),
            "revenue_variance": budget.get("revenue_variance")
            if isinstance(budget, dict)
            else None,
            "expense_variance": budget.get("expense_variance")
            if isinstance(budget, dict)
            else None,
            "collection_rate": payments.get("collection_rate")
            if isinstance(payments, dict)
            else None,
            "overdue_invoice_count": payments.get("overdue_invoice_count")
            if isinstance(payments, dict)
            else None,
            "net_cash_flow": cash.get("net_cash_flow")
            if isinstance(cash, dict)
            else None,
            "ending_cash": cash.get("ending_cash") if isinstance(cash, dict) else None,
        },
        "top_departments": top_departments[:4],
        "top_categories": top_categories[:6],
        "warnings": finance_document.get("calculation_warnings", [])[:5],
    }


def _compact_anomalies(
    anomaly_report: dict[str, Any],
    *,
    max_anomalies: int = 8,
) -> dict[str, Any]:
    """Select anomaly facts needed for strategic reasoning.

    Inputs: processed anomaly report JSON.
    Outputs: bounded anomaly summary.
    Assumptions: anomaly detector values are authoritative facts.
    """

    anomalies = anomaly_report.get("anomalies", [])
    anomalies = anomalies if isinstance(anomalies, list) else []
    ranked = rank_anomalies(
        anomalies,
        allowed_severities={"critical", "high"},
        max_count=max_anomalies,
    )
    if not ranked:
        # Analysis still needs visibility when no high-severity anomalies exist.
        ranked = rank_anomalies(anomalies, max_count=max_anomalies)
    return {
        "report_period": anomaly_report.get("report_period"),
        "total_anomalies": anomaly_report.get("total_anomalies"),
        "anomalies_by_severity": anomaly_report.get("anomalies_by_severity", {}),
        "context_policy": {
            "ranked_anomalies": True,
            "preferred_severities": ["critical", "high"],
            "included_count": len(ranked),
            "available_to_python_count": len(anomalies),
        },
        "anomalies": [
            {
                "anomaly_id": item.get("anomaly_id"),
                "title": item.get("title"),
                "severity": item.get("severity"),
                "metric": item.get("metric"),
                "period": item.get("period"),
                "observed_value": item.get("observed_value"),
                "threshold_value": item.get("threshold_value"),
                "evidence": str(item.get("evidence", ""))[:220],
            }
            for item in ranked
            if isinstance(item, dict)
        ],
    }


def _compact_risk_summary(risk_summary: dict[str, Any]) -> dict[str, Any]:
    """Select annual risk facts for strategic context.

    Inputs: processed risk summary JSON.
    Outputs: bounded top-risk summary.
    Assumptions: risk summary is detector-authored, not LLM-authored.
    """

    return {
        "total_anomalies": risk_summary.get("total_anomalies"),
        "high_priority_count": risk_summary.get("high_priority_count"),
        "anomalies_by_severity": risk_summary.get("anomalies_by_severity", {}),
        "top_risks": risk_summary.get("top_risks", [])[:6],
        "thresholds": risk_summary.get("thresholds", {}),
    }


def _compact_evidence_package(
    evidence_package: dict[str, Any],
    *,
    deduplicate_context: bool = True,
) -> dict[str, Any]:
    """Compress evidence packages without copying full reports or row sets.

    Inputs: Step 8 evidence package document.
    Outputs: compact task/evidence availability summary.
    Assumptions: detailed rows remain in evidence package files for audit.
    """

    compact_items: list[dict[str, Any]] = []
    source_packages = evidence_package.get("evidence_packages", [])
    source_packages = source_packages if isinstance(source_packages, list) else []
    for package in source_packages[:12]:
        if not isinstance(package, dict):
            continue
        evidence = package.get("retrieved_evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        data = evidence.get("data", {})
        data = data if isinstance(data, dict) else {}
        # Include bounded structured evidence, not just counts. This prevents the
        # analysis model from reporting payroll/department breakdowns as missing
        # when the retrieval layer has already provided them.
        records = data.get("records", [])
        records = records if isinstance(records, list) else []
        recent_records = records[-4:]
        compact_records = [
            {
                key: value
                for key, value in record.items()
                if key
                in {
                    "period",
                    "billing_period",
                    "payment_date",
                    "month",
                    "department",
                    "expense_category",
                    "vendor",
                    "headcount_fte",
                    "base_salary",
                    "benefits",
                    "overtime",
                    "total_payroll",
                    "payroll_budget",
                    "variance",
                    "actual_expense",
                    "budget_expense",
                    "amount",
                    "status",
                    "_source_table",
                }
            }
            for record in recent_records
            if isinstance(record, dict)
        ]
        payroll_breakdown = data.get("payroll_breakdown")
        payroll_breakdown = payroll_breakdown if isinstance(payroll_breakdown, list) else []
        compact_items.append(
            {
                "task_id": package.get("task_id"),
                "priority": package.get("priority"),
                "question": str(package.get("investigation_question", ""))[:260],
                "retrieval_name": evidence.get("retrieval_name"),
                "success": evidence.get("success"),
                "evidence_summary": str(package.get("evidence_summary", ""))[:260],
                "record_count": data.get("record_count"),
                "matched_tables": data.get("matched_tables"),
                "source_tables": data.get("source_tables"),
                "counts_by_source": data.get("counts_by_source"),
                "payroll_breakdown": payroll_breakdown[-4:],
                "sample_records": compact_records,
                "warnings": evidence.get("warnings", [])[:3],
                "unavailable_data": evidence.get("unavailable_data", [])[:3],
                "confidence": evidence.get("confidence"),
                "source_references": evidence.get("source_references", [])[:3],
            }
        )
    if deduplicate_context:
        compact_items = deduplicate_dicts(
            compact_items,
            key_fields=("retrieval_name", "question", "evidence_summary"),
        )
    return {
        "package_id": evidence_package.get("package_id"),
        "period_slug": evidence_package.get("period_slug"),
        "summary": evidence_package.get("summary"),
        "evidence_items": compact_items[:8],
    }


def build_strategic_analysis_prompt(
    *,
    evidence_package: dict[str, Any],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    risk_summary: dict[str, Any],
    period_slug: str,
    compact_context: bool = True,
    deduplicate_context: bool = True,
    historical_context: dict[str, Any] | None = None,
) -> str:
    """Build a compact strict-JSON strategic-analysis prompt.

    Inputs: processed evidence, finance, anomaly, risk artifacts, period slug, and history.
    Outputs: prompt string for Ollama.
    Assumptions: no raw Excel/PDF content or full evidence row sets are sent.
    """

    del compact_context  # Strategic prompt always uses compact processed summaries.
    context = {
        "period_slug": period_slug,
        "finance_summary": _compact_finance_summary(
            finance_summary,
            deduplicate_context=deduplicate_context,
        ),
        "anomaly_report": _compact_anomalies(anomaly_report),
        "risk_summary": _compact_risk_summary(risk_summary),
        "evidence_package": _compact_evidence_package(
            evidence_package,
            deduplicate_context=deduplicate_context,
        ),
        "historical_context": historical_context or {},
        "context_policy": {
            "compact_context": True,
            "deduplicate_context": deduplicate_context,
            "no_raw_reports_or_tables": True,
            "historical_context_compact_only": True,
        },
    }
    schema_text = (
        "Required exact JSON keys and types:\n"
        "- executive_summary: Spanish string.\n"
        "- key_findings: list of 1-8 Spanish strings.\n"
        "- root_causes: list of 1-8 Spanish strings.\n"
        "- financial_health_analysis: Spanish string.\n"
        "- kpi_analysis: Spanish string.\n"
        "- historical_summary: Spanish string.\n"
        "- historical_trend_analysis: Spanish string.\n"
        "- department_analysis: Spanish string.\n"
        "- anomaly_analysis: Spanish string.\n"
        "- recommendation_follow_up_analysis: Spanish string.\n"
        "- longitudinal_risk_analysis: Spanish string.\n"
        "- strategic_recommendations: list of 1-8 objects with keys priority, action, rationale, supporting_evidence, expected_impact, evidence_ids, confidence.\n"
        "- strategic_priorities: list of 1-8 Spanish strings.\n"
        "- missing_information: list of Spanish strings, or [] only when no material evidence gap exists.\n"
        "- narrative_evidence: object mapping every narrative/list field except strategic_recommendations to a non-empty list of context path IDs.\n"
        "- confidence: number from 0 to 1.\n"
        "- reasoning_summary: Spanish string.\n"
    )
    return (
        "INSTRUCTIONS_BEFORE_CONTEXT:\n"
        "You are the strategic financial analyst stage of a Python-first finance "
        "agent. Use only the processed context supplied below. Do not calculate "
        "new financial values, alter any metric, invent source data, execute tools, "
        "or draft a report. Explain the most important financial issues, likely "
        "root causes, annual-goal implications, prioritized risks, concrete "
        "actions, missing evidence, and confidence. Use cautious language when "
        "evidence is incomplete. Do not list payroll amount, budget, variance, "
        "headcount, salary, benefits, overtime, department, or source-table "
        "breakdowns as missing when they appear in payroll_breakdown or "
        "sample_records. Every narrative field must cite its source/evidence IDs "
        "in narrative_evidence using compact IDs from the context path, and each "
        "strategic recommendation must include evidence_ids. Do not include "
        "numbers, periods, departments, vendors or claims unless they appear in "
        "the supplied context. Omit or keep concise any section whose evidence is "
        "absent rather than inventing content. Avoid generic filler such as "
        "'se requiere más información' unless it names the specific missing "
        "evidence. Return STRICT JSON only, with exactly the fields listed in "
        "REQUIRED_SCHEMA below. Do not return examples, empty strings, type "
        "labels, placeholder text, markdown, or prose outside JSON. Every value "
        "must be evidence-specific analysis from the supplied context. All user-facing string values MUST be written directly in "
        "professional Spanish for university leadership. Keep internal JSON keys, "
        "canonical metric names, IDs, and numeric evidence unchanged in English "
        "where they appear as data. The Spanish-only requirement applies to: "
        "executive_summary, key_findings, root_causes, financial_health_analysis, "
        "kpi_analysis, historical_summary, historical_trend_analysis, "
        "department_analysis, anomaly_analysis, recommendation_follow_up_analysis, "
        "longitudinal_risk_analysis, strategic_priorities, "
        "strategic_recommendations.action, strategic_recommendations.rationale, "
        "strategic_recommendations.supporting_evidence, "
        "strategic_recommendations.expected_impact, missing_information, and "
        "reasoning_summary. Common acronyms such as KPI, USD, EBITDA, PDF, HTML, "
        "JSON and CSV are allowed. Keep every string under 1200 characters. Use no more than "
        f"{MAX_LIST_ITEMS} items in each list and no more than "
        f"{MAX_RECOMMENDATIONS} recommendations. Recommendation priority must be "
        "critical, high, medium, or low. Confidence values must be 0..1.\n"
        "REQUIRED_SCHEMA:\n"
        + schema_text
        + "\nSTRATEGIC_ANALYSIS_CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nINSTRUCTIONS_FINAL_OUTPUT_REMINDER:\n"
        "Return only the final JSON analysis object with the exact required keys. "
        "Do not return any object copied from STRATEGIC_ANALYSIS_CONTEXT."
    )


def build_spanish_rewrite_prompt(
    analysis: dict[str, Any],
    *,
    language_errors: tuple[str, ...],
) -> str:
    """Build one bounded rewrite prompt for Spanish-language correction.

    Inputs: schema-valid analysis payload and Spanish validation errors.
    Outputs: compact strict-JSON rewrite prompt.
    Assumptions: Ollama must preserve numbers, priorities, evidence and meaning.
    """

    return (
        "SPANISH_REWRITE_INPUT:\n"
        + json.dumps(analysis, ensure_ascii=False, separators=(",", ":"))
        + "\n\nLANGUAGE_ERRORS:\n"
        + json.dumps(list(language_errors), ensure_ascii=False, separators=(",", ":"))
        + "\n\nINSTRUCTIONS:\n"
        "Return STRICT JSON with the exact same keys and structure as "
        "SPANISH_REWRITE_INPUT. Rewrite only user-facing prose into professional "
        "Spanish for university leadership. Preserve all numbers, priorities, "
        "confidence values, evidence references, canonical metric names, IDs, and "
        "meaning. Do not recalculate, add claims, remove claims, or change list "
        "lengths. Common acronyms KPI, USD, EBITDA, PDF, HTML, JSON and CSV are "
        "allowed."
    )


def build_evidence_repair_prompt(
    analysis: dict[str, Any],
    *,
    claim_errors: tuple[str, ...],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    evidence_package: dict[str, Any],
    risk_summary: dict[str, Any],
    historical_context: dict[str, Any] | None = None,
) -> str:
    """Build one bounded prompt to remove unsupported narrative claims.

    Inputs: schema-valid analysis, claim-validation errors, and compact evidence.
    Outputs: strict-JSON repair prompt.
    Assumptions: the model may rewrite prose but must not add claims or alter
    processed financial data.
    """

    repair_context = {
        "analysis_to_repair": analysis,
        "claim_errors": list(claim_errors),
        "allowed_evidence": {
            "finance_summary": _compact_finance_summary(finance_summary),
            "anomaly_report": _compact_anomalies(anomaly_report),
            "risk_summary": _compact_risk_summary(risk_summary),
            "evidence_package": _compact_evidence_package(evidence_package),
            "historical_context": historical_context or {},
        },
    }
    return (
        "EVIDENCE_REPAIR_TASK:\n"
        + json.dumps(repair_context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nINSTRUCTIONS:\n"
        "Return STRICT JSON with exactly the same schema as analysis_to_repair. "
        "Rewrite only the user-facing Spanish prose needed to remove claim_errors. "
        "Preserve priorities, evidence_ids, confidence values, canonical IDs, and "
        "strategic meaning. Do not calculate or introduce new numbers, periods, "
        "departments, vendors, or claims. If a numeric claim is not present in "
        "allowed_evidence, omit the number and describe the issue qualitatively. "
        "Return JSON only."
    )


def _schema_valid_but_not_spanish(response_text: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Detect whether a response only failed Spanish-language validation.

    Inputs: raw model response text.
    Outputs: schema-valid analysis payload and language errors, or None/errors.
    Assumptions: retry is allowed only when JSON/schema are otherwise valid.
    """

    schema_validation = validate_strategic_analysis_response(
        response_text,
        require_spanish=False,
    )
    if not schema_validation.is_valid or schema_validation.analysis is None:
        return None, schema_validation.errors
    language_errors = validate_user_facing_spanish(schema_validation.analysis)
    if language_errors:
        return schema_validation.analysis, language_errors
    return None, ()


def _empty_rejected_analysis() -> dict[str, Any]:
    """Create an empty analysis shape for rejected/unavailable model outputs.

    Inputs: none.
    Outputs: analysis-shaped dictionary.
    Assumptions: rejected outputs should be auditable but not treated as generated.
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


def _build_analysis_document(
    *,
    period_slug: str,
    report_period: str,
    ollama_available: bool,
    validation_status: str,
    validation_errors: tuple[str, ...],
    analysis: dict[str, Any],
    historical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one auditable strategic-analysis output document.

    Inputs: metadata, validation status, errors, accepted/empty analysis, and history.
    Outputs: JSON-compatible document.
    Assumptions: only accepted documents should feed final reporting later.
    """

    recommendations = analysis.get("recommendations", [])
    recommendations = recommendations if isinstance(recommendations, list) else []
    return {
        "analysis_id": f"STRATEGIC-ANALYSIS-{period_slug.upper().replace('_', '-')}",
        "period_slug": period_slug,
        "report_period": report_period,
        "analysis_source": "ollama",
        "ollama_available": ollama_available,
        "validation_status": validation_status,
        "analysis_generated": validation_status == "accepted",
        "validation_errors": list(validation_errors),
        "recommendation_count": len(recommendations),
        "historical_context_summary": (historical_context or {}).get("summary", {}),
        "historical_context": historical_context or {},
        "analysis": analysis,
    }


def create_strategic_analysis(
    *,
    client: StrategicAnalysisClient,
    evidence_package: dict[str, Any],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    risk_summary: dict[str, Any],
    period_slug: str,
    compact_context: bool = True,
    deduplicate_context: bool = True,
    historical_context: dict[str, Any] | None = None,
) -> StrategicAnalysisResult:
    """Generate and validate one Ollama strategic financial analysis.

    Inputs: Ollama client, processed Step 3/4/8 artifacts, and optional history.
    Outputs: accepted or rejected strategic-analysis result.
    Assumptions: invalid or unavailable model output is rejected, not repaired.
    """

    stage_started = time.perf_counter()
    preprocessing_started = time.perf_counter()
    report_period = str(finance_summary.get("report_period", period_slug))
    available = client.is_available()
    errors: tuple[str, ...] = ()
    validation: AnalysisValidationResult | None = None
    prompt = ""
    ollama_telemetry: dict[str, Any] = {}
    validation_time = 0.0
    spanish_rewrite_attempted = False
    if available:
        prompt = build_strategic_analysis_prompt(
            evidence_package=evidence_package,
            finance_summary=finance_summary,
            anomaly_report=anomaly_report,
            risk_summary=risk_summary,
            period_slug=period_slug,
            compact_context=compact_context,
            deduplicate_context=deduplicate_context,
            historical_context=historical_context,
        )
        preprocessing_time = time.perf_counter() - preprocessing_started
        previous_response_format = getattr(client, "response_format", None)
        if previous_response_format is not None:
            # Ask supporting Ollama clients for schema-constrained JSON. Python
            # validation remains the source of truth after generation.
            setattr(client, "response_format", strategic_analysis_json_schema())
        try:
            if hasattr(client, "generate_with_metadata"):
                generation = client.generate_with_metadata(prompt)  # type: ignore[attr-defined]
                response = str(generation["response"])
                ollama_telemetry = dict(generation.get("telemetry", {}))
            else:
                response = client.generate(prompt)
            validation_started = time.perf_counter()
            validation = validate_strategic_analysis_response(response)
            errors = validation.errors
            validation_time = time.perf_counter() - validation_started
            if not validation.is_valid:
                schema_analysis, language_errors = _schema_valid_but_not_spanish(response)
                if schema_analysis is not None and language_errors:
                    spanish_rewrite_attempted = True
                    rewrite_prompt = build_spanish_rewrite_prompt(
                        schema_analysis,
                        language_errors=language_errors,
                    )
                    if hasattr(client, "generate_with_metadata"):
                        generation = client.generate_with_metadata(rewrite_prompt)  # type: ignore[attr-defined]
                        response = str(generation["response"])
                        ollama_telemetry = merge_telemetry(
                            ollama_telemetry,
                            dict(generation.get("telemetry", {})),
                        )
                    else:
                        response = client.generate(rewrite_prompt)
                    validation_started = time.perf_counter()
                    validation = validate_strategic_analysis_response(response)
                    errors = validation.errors
                    validation_time += time.perf_counter() - validation_started
            if validation.is_valid and validation.analysis is not None:
                claim_errors = validate_evidence_bound_claims(
                    validation.analysis,
                    finance_summary=finance_summary,
                    anomaly_report=anomaly_report,
                    evidence_package=evidence_package,
                    risk_summary=risk_summary,
                    historical_context=historical_context,
                )
                if claim_errors:
                    repair_prompt = build_evidence_repair_prompt(
                        validation.analysis,
                        claim_errors=claim_errors,
                        finance_summary=finance_summary,
                        anomaly_report=anomaly_report,
                        evidence_package=evidence_package,
                        risk_summary=risk_summary,
                        historical_context=historical_context,
                    )
                    if hasattr(client, "generate_with_metadata"):
                        generation = client.generate_with_metadata(repair_prompt)  # type: ignore[attr-defined]
                        response = str(generation["response"])
                        ollama_telemetry = merge_telemetry(
                            ollama_telemetry,
                            dict(generation.get("telemetry", {})),
                        )
                    else:
                        response = client.generate(repair_prompt)
                    validation_started = time.perf_counter()
                    repaired_validation = validate_strategic_analysis_response(response)
                    validation_time += time.perf_counter() - validation_started
                    if repaired_validation.is_valid and repaired_validation.analysis is not None:
                        repaired_claim_errors = validate_evidence_bound_claims(
                            repaired_validation.analysis,
                            finance_summary=finance_summary,
                            anomaly_report=anomaly_report,
                            evidence_package=evidence_package,
                            risk_summary=risk_summary,
                            historical_context=historical_context,
                        )
                        if repaired_claim_errors:
                            validation = AnalysisValidationResult(False, None, repaired_claim_errors)
                            errors = repaired_claim_errors
                        else:
                            validation = repaired_validation
                            errors = ()
                    else:
                        schema_analysis, language_errors = _schema_valid_but_not_spanish(response)
                        if schema_analysis is not None and language_errors:
                            spanish_rewrite_attempted = True
                            rewrite_prompt = build_spanish_rewrite_prompt(
                                schema_analysis,
                                language_errors=language_errors,
                            )
                            if hasattr(client, "generate_with_metadata"):
                                generation = client.generate_with_metadata(rewrite_prompt)  # type: ignore[attr-defined]
                                response = str(generation["response"])
                                ollama_telemetry = merge_telemetry(
                                    ollama_telemetry,
                                    dict(generation.get("telemetry", {})),
                                )
                            else:
                                response = client.generate(rewrite_prompt)
                            validation_started = time.perf_counter()
                            validation = validate_strategic_analysis_response(response)
                            validation_time += time.perf_counter() - validation_started
                            if validation.is_valid and validation.analysis is not None:
                                claim_errors = validate_evidence_bound_claims(
                                    validation.analysis,
                                    finance_summary=finance_summary,
                                    anomaly_report=anomaly_report,
                                    evidence_package=evidence_package,
                                    risk_summary=risk_summary,
                                    historical_context=historical_context,
                                )
                                if claim_errors:
                                    validation = AnalysisValidationResult(False, None, claim_errors)
                                    errors = claim_errors
                                else:
                                    errors = ()
                            else:
                                errors = validation.errors
                        else:
                            validation = repaired_validation
                            errors = repaired_validation.errors
        except OllamaError as exc:
            errors = (str(exc),)
        finally:
            if previous_response_format is not None:
                setattr(client, "response_format", previous_response_format)
    else:
        preprocessing_time = time.perf_counter() - preprocessing_started
        errors = ("Ollama is unavailable.",)

    accepted = validation is not None and validation.is_valid
    analysis = validation.analysis if accepted and validation else _empty_rejected_analysis()
    if accepted:
        analysis = _remove_false_missing_information(
            analysis,
            evidence_package,
            finance_summary,
            anomaly_report,
        )
    document = _build_analysis_document(
        period_slug=period_slug,
        report_period=report_period,
        ollama_available=available,
        validation_status="accepted"
        if accepted
        else ("rejected" if available else "unavailable"),
        validation_errors=errors,
        analysis=analysis,
        historical_context=historical_context,
    )
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=accepted,
        validation_errors=errors,
        telemetry=merge_telemetry(
            {
                "python_preprocessing_time_seconds": preprocessing_time,
                "json_validation_time_seconds": validation_time,
                "total_stage_time_seconds": time.perf_counter() - stage_started,
                "context_characters": len(prompt),
                "context_token_estimate": estimate_tokens_from_text(prompt),
                "compact_context": compact_context,
                "deduplicate_context": deduplicate_context,
                "spanish_rewrite_attempted": spanish_rewrite_attempted,
                "compact_context_json_characters": compact_json_size(
                    {
                        "finance_summary": _compact_finance_summary(
                            finance_summary,
                            deduplicate_context=deduplicate_context,
                        ),
                        "anomaly_report": _compact_anomalies(anomaly_report),
                        "evidence_package": _compact_evidence_package(
                            evidence_package,
                            deduplicate_context=deduplicate_context,
                        ),
                        "historical_context": historical_context or {},
                    }
                ),
            },
            ollama_telemetry,
        ),
    )


def build_analysis_summary(documents: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Build a compact cross-scope strategic-analysis summary.

    Inputs: strategic-analysis documents.
    Outputs: JSON-compatible summary artifact.
    Assumptions: rejected analyses have no confidence or recommendations.
    """

    accepted = [
        document
        for document in documents
        if document.get("validation_status") == "accepted"
    ]
    confidences = [
        float(document["analysis"]["confidence"])
        for document in accepted
        if isinstance(document.get("analysis"), dict)
        and document["analysis"].get("confidence") is not None
    ]
    summary = AnalysisRunSummary(
        summary_id="ANALYSIS-SUMMARY-2026",
        analyses_requested=len(documents),
        analyses_generated=len(accepted),
        analyses_rejected=len(documents) - len(accepted),
        average_confidence=(
            sum(confidences) / len(confidences) if confidences else None
        ),
        recommendations_generated=sum(
            int(document.get("recommendation_count", 0)) for document in accepted
        ),
        scopes=tuple(
            {
                "analysis_id": document.get("analysis_id"),
                "period_slug": document.get("period_slug"),
                "validation_status": document.get("validation_status"),
                "confidence": document.get("analysis", {}).get("confidence")
                if isinstance(document.get("analysis"), dict)
                else None,
                "recommendation_count": document.get("recommendation_count"),
            }
            for document in documents
        ),
    )
    return summary.to_dict()


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load one processed JSON artifact for analysis.

    Inputs: path to a processed JSON file.
    Outputs: parsed dictionary.
    Assumptions: Step 9 never opens raw Excel/PDF inputs.
    """

    artifact_path = Path(path)
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {artifact_path}")
    return value


def save_json_artifact(data: dict[str, Any], output_path: str | Path) -> Path:
    """Save one analysis artifact as readable JSON.

    Inputs: JSON-compatible data and output path.
    Outputs: resolved written path.
    Assumptions: parent directories may be created.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
