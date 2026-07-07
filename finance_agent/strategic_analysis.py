"""Ollama-backed strategic financial analysis over processed evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from finance_agent.analysis_models import (
    AnalysisRunSummary,
    AnalysisValidationResult,
    StrategicAnalysisResult,
)
from finance_agent.ollama_client import OllamaError


MAX_RESPONSE_CHARACTERS = 80_000
MAX_TEXT_LENGTH = 1_200
MAX_LIST_ITEMS = 8
MAX_RECOMMENDATIONS = 8
REQUIRED_ANALYSIS_FIELDS = frozenset(
    {
        "executive_summary",
        "key_findings",
        "root_causes",
        "recommendations",
        "strategic_priorities",
        "missing_information",
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
        "confidence",
    }
)
ALLOWED_RECOMMENDATION_PRIORITIES = frozenset(
    {"critical", "high", "medium", "low"}
)


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

    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list):
        errors.append("recommendations must be a list")
        return
    if not recommendations:
        errors.append("recommendations must contain at least one item")
    if len(recommendations) > MAX_RECOMMENDATIONS:
        errors.append(
            f"recommendations may contain at most {MAX_RECOMMENDATIONS} items"
        )
    for index, recommendation in enumerate(recommendations):
        prefix = f"recommendations[{index}]"
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
        if not _valid_confidence(recommendation["confidence"]):
            errors.append(f"{prefix}.confidence must be between 0 and 1")


def validate_strategic_analysis_response(response_text: str) -> AnalysisValidationResult:
    """Parse and validate strict JSON returned by Ollama.

    Inputs: raw model response text.
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
        "key_findings",
        "root_causes",
        "strategic_priorities",
        "missing_information",
    ):
        _validate_string_list(payload, field_name, errors)
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
        "recommendations": [
            {
                "priority": item["priority"],
                "action": item["action"].strip(),
                "rationale": item["rationale"].strip(),
                "supporting_evidence": item["supporting_evidence"].strip(),
                "expected_impact": item["expected_impact"].strip(),
                "confidence": float(item["confidence"]),
            }
            for item in payload["recommendations"]
        ],
        "strategic_priorities": [
            item.strip() for item in payload["strategic_priorities"]
        ],
        "missing_information": [item.strip() for item in payload["missing_information"]],
        "confidence": float(payload["confidence"]),
        "reasoning_summary": payload["reasoning_summary"].strip(),
    }
    return AnalysisValidationResult(True, cleaned, ())


def _compact_finance_summary(finance_document: dict[str, Any]) -> dict[str, Any]:
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
        "top_departments": finance_document.get("department_summary", [])[:4],
        "top_categories": finance_document.get("category_summary", [])[:6],
        "warnings": finance_document.get("calculation_warnings", [])[:5],
    }


def _compact_anomalies(anomaly_report: dict[str, Any]) -> dict[str, Any]:
    """Select anomaly facts needed for strategic reasoning.

    Inputs: processed anomaly report JSON.
    Outputs: bounded anomaly summary.
    Assumptions: anomaly detector values are authoritative facts.
    """

    return {
        "report_period": anomaly_report.get("report_period"),
        "total_anomalies": anomaly_report.get("total_anomalies"),
        "anomalies_by_severity": anomaly_report.get("anomalies_by_severity", {}),
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
            for item in anomaly_report.get("anomalies", [])[:12]
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
        "top_risks": risk_summary.get("top_risks", [])[:10],
        "thresholds": risk_summary.get("thresholds", {}),
    }


def _compact_evidence_package(evidence_package: dict[str, Any]) -> dict[str, Any]:
    """Compress evidence packages without copying full reports or row sets.

    Inputs: Step 8 evidence package document.
    Outputs: compact task/evidence availability summary.
    Assumptions: detailed rows remain in evidence package files for audit.
    """

    compact_items: list[dict[str, Any]] = []
    for package in evidence_package.get("evidence_packages", [])[:12]:
        if not isinstance(package, dict):
            continue
        evidence = package.get("retrieved_evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        data = evidence.get("data", {})
        data = data if isinstance(data, dict) else {}
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
                "warnings": evidence.get("warnings", [])[:3],
                "unavailable_data": evidence.get("unavailable_data", [])[:3],
                "confidence": evidence.get("confidence"),
                "source_references": evidence.get("source_references", [])[:3],
            }
        )
    return {
        "package_id": evidence_package.get("package_id"),
        "period_slug": evidence_package.get("period_slug"),
        "summary": evidence_package.get("summary"),
        "evidence_items": compact_items,
    }


def build_strategic_analysis_prompt(
    *,
    evidence_package: dict[str, Any],
    finance_summary: dict[str, Any],
    anomaly_report: dict[str, Any],
    risk_summary: dict[str, Any],
    period_slug: str,
) -> str:
    """Build a compact strict-JSON strategic-analysis prompt.

    Inputs: processed evidence, finance, anomaly, risk artifacts, and period slug.
    Outputs: prompt string for Ollama.
    Assumptions: no raw Excel/PDF content or full evidence row sets are sent.
    """

    context = {
        "period_slug": period_slug,
        "finance_summary": _compact_finance_summary(finance_summary),
        "anomaly_report": _compact_anomalies(anomaly_report),
        "risk_summary": _compact_risk_summary(risk_summary),
        "evidence_package": _compact_evidence_package(evidence_package),
    }
    response_shape = {
        "executive_summary": "Two to four concise sentences.",
        "key_findings": ["Finding supported by processed evidence."],
        "root_causes": ["Likely cause, framed as likely when evidence is incomplete."],
        "recommendations": [
            {
                "priority": "high",
                "action": "Concrete management action.",
                "rationale": "Why this action follows from the evidence.",
                "supporting_evidence": "Evidence reference or metric.",
                "expected_impact": "Operational or financial outcome to monitor.",
                "confidence": 0.75,
            }
        ],
        "strategic_priorities": ["Priority to manage next."],
        "missing_information": ["Evidence still needed, or [] if none."],
        "confidence": 0.75,
        "reasoning_summary": "Short explanation of how evidence supports the conclusions.",
    }
    return (
        "STRATEGIC_ANALYSIS_CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nINSTRUCTIONS_AFTER_CONTEXT:\n"
        "You are the strategic financial analyst stage of a Python-first finance "
        "agent. Use only the processed context supplied above. Do not calculate "
        "new financial values, alter any metric, invent source data, execute tools, "
        "or draft a report. Explain the most important financial issues, likely "
        "root causes, annual-goal implications, prioritized risks, concrete "
        "actions, missing evidence, and confidence. Use cautious language when "
        "evidence is incomplete. Return STRICT JSON only, with exactly the fields "
        "shown below. Keep every string under 1200 characters. Use no more than "
        f"{MAX_LIST_ITEMS} items in each list and no more than "
        f"{MAX_RECOMMENDATIONS} recommendations. Recommendation priority must be "
        "critical, high, medium, or low. Confidence values must be 0..1.\n"
        "VALID_RESPONSE_SHAPE:\n"
        + json.dumps(response_shape, ensure_ascii=False, separators=(",", ":"))
    )


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
        "recommendations": [],
        "strategic_priorities": [],
        "missing_information": [],
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
) -> dict[str, Any]:
    """Assemble one auditable strategic-analysis output document.

    Inputs: metadata, validation status, errors, and accepted or empty analysis.
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
) -> StrategicAnalysisResult:
    """Generate and validate one Ollama strategic financial analysis.

    Inputs: Ollama client and processed Step 3/4/8 artifacts.
    Outputs: accepted or rejected strategic-analysis result.
    Assumptions: invalid or unavailable model output is rejected, not repaired.
    """

    report_period = str(finance_summary.get("report_period", period_slug))
    available = client.is_available()
    errors: tuple[str, ...] = ()
    validation: AnalysisValidationResult | None = None
    if available:
        prompt = build_strategic_analysis_prompt(
            evidence_package=evidence_package,
            finance_summary=finance_summary,
            anomaly_report=anomaly_report,
            risk_summary=risk_summary,
            period_slug=period_slug,
        )
        try:
            response = client.generate(prompt)
            validation = validate_strategic_analysis_response(response)
            errors = validation.errors
        except OllamaError as exc:
            errors = (str(exc),)
    else:
        errors = ("Ollama is unavailable.",)

    accepted = validation is not None and validation.is_valid
    analysis = validation.analysis if accepted and validation else _empty_rejected_analysis()
    document = _build_analysis_document(
        period_slug=period_slug,
        report_period=report_period,
        ollama_available=available,
        validation_status="accepted"
        if accepted
        else ("rejected" if available else "unavailable"),
        validation_errors=errors,
        analysis=analysis,
    )
    return StrategicAnalysisResult(
        analysis_document=document,
        accepted=accepted,
        validation_errors=errors,
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
