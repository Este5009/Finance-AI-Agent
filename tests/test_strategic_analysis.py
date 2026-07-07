"""Tests for Step 9 Ollama strategic financial analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass

from finance_agent.strategic_analysis import (
    build_strategic_analysis_prompt,
    create_strategic_analysis,
    validate_strategic_analysis_response,
)


@dataclass
class FakeAnalysisClient:
    """Configurable test double for the Step 9 Ollama client."""

    available: bool
    response: str = ""
    generate_calls: int = 0
    last_prompt: str = ""

    def is_available(self) -> bool:
        """Return configured availability.

        Inputs: fixture state.
        Outputs: availability boolean.
        Assumptions: no network request occurs in tests.
        """

        return self.available

    def generate(self, prompt: str) -> str:
        """Record prompt and return configured response.

        Inputs: strategic-analysis prompt.
        Outputs: configured model response.
        Assumptions: tests control the response bytes.
        """

        self.generate_calls += 1
        self.last_prompt = prompt
        return self.response


def _valid_analysis() -> dict[str, object]:
    """Build a valid strategic-analysis payload fixture.

    Inputs: none.
    Outputs: JSON-compatible analysis payload.
    Assumptions: values are intentionally concise for schema validation.
    """

    return {
        "executive_summary": (
            "June performance shows an operating deficit, negative cash flow, "
            "and collection pressure requiring near-term management focus."
        ),
        "key_findings": [
            "Operating result is negative based on processed finance summary.",
            "Student collections and overdue invoices are flagged by anomalies.",
        ],
        "root_causes": [
            "Expense pressure appears to exceed revenue performance.",
            "Collections are likely delayed for a subset of student invoices.",
        ],
        "recommendations": [
            {
                "priority": "high",
                "action": "Review department expense approvals for overspending categories.",
                "rationale": "Processed evidence shows department and category pressure.",
                "supporting_evidence": "Evidence package includes department and report retrievals.",
                "expected_impact": "Reduce preventable variance in the next reporting cycle.",
                "confidence": 0.78,
            },
            {
                "priority": "medium",
                "action": "Prioritize overdue student invoice follow-up.",
                "rationale": "Collection anomalies and overdue evidence indicate receivable risk.",
                "supporting_evidence": "Student payment transactions include overdue records.",
                "expected_impact": "Improve cash conversion and reduce outstanding balances.",
                "confidence": 0.74,
            },
        ],
        "strategic_priorities": [
            "Stabilize cash flow.",
            "Reduce expense variance.",
        ],
        "missing_information": ["Approval notes for flagged vendor payments."],
        "confidence": 0.76,
        "reasoning_summary": (
            "Conclusions combine processed finance metrics, anomaly severity, "
            "and retrieved evidence availability without recalculating values."
        ),
    }


def _evidence_package() -> dict[str, object]:
    """Build a compact Step 8-like evidence fixture.

    Inputs: none.
    Outputs: evidence package dictionary.
    Assumptions: embedded records should not be copied into prompts in full.
    """

    return {
        "package_id": "EVIDENCE-JUNE-2026",
        "period_slug": "june_2026",
        "summary": {
            "tasks_executed": 2,
            "successful_retrievals": 2,
            "failed_retrievals": 0,
            "unavailable_evidence": 0,
        },
        "evidence_packages": [
            {
                "task_id": "STEP-001",
                "priority": "critical",
                "investigation_question": "What caused the deficit?",
                "evidence_summary": "Retrieved 1 processed report.",
                "retrieved_evidence": {
                    "retrieval_name": "financial_report",
                    "success": True,
                    "data": {
                        "summary": "Retrieved processed report.",
                        "record_count": 1,
                        "records": [{"secret_row": "MUST_NOT_APPEAR"}],
                    },
                    "warnings": [],
                    "unavailable_data": [],
                    "source_references": ["finance_summary.json"],
                    "confidence": 0.98,
                },
            }
        ],
    }


def _finance_summary() -> dict[str, object]:
    """Build a processed finance summary fixture.

    Inputs: none.
    Outputs: Step 3-like finance summary.
    Assumptions: values are Python-calculated and model must not modify them.
    """

    return {
        "report_period": "June 2026",
        "finance_summary": {
            "total_revenue": 1000,
            "total_expenses": 1200,
            "net_operating_result": -200,
            "payroll_percentage_of_revenue": 0.52,
            "student_payments": {"collection_rate": 0.84},
            "cash_flow": {"net_cash_flow": -300, "ending_cash": 5000},
        },
        "department_summary": [{"department": "Engineering", "variance": 100}],
        "category_summary": [{"category": "Payroll", "variance": 50}],
    }


def _anomaly_report() -> dict[str, object]:
    """Build a processed anomaly report fixture.

    Inputs: none.
    Outputs: Step 4-like anomaly report.
    Assumptions: anomaly facts are source-of-truth for the prompt.
    """

    return {
        "report_period": "June 2026",
        "total_anomalies": 2,
        "anomalies_by_severity": {"critical": 1, "high": 1},
        "anomalies": [
            {
                "anomaly_id": "ANOM-1",
                "title": "Operating deficit",
                "severity": "critical",
                "metric": "net_operating_result",
                "observed_value": -200,
                "threshold_value": 0,
                "period": "2026-06",
            }
        ],
    }


def _risk_summary() -> dict[str, object]:
    """Build a processed annual risk summary fixture.

    Inputs: none.
    Outputs: Step 4-like risk summary.
    Assumptions: thresholds are included only as context.
    """

    return {
        "total_anomalies": 2,
        "high_priority_count": 2,
        "top_risks": [{"title": "Operating deficit", "severity": "critical"}],
        "thresholds": {"low_cash_flow_threshold": 0},
    }


def test_valid_json_response_is_accepted() -> None:
    """Verify a schema-compliant analysis response validates successfully."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis()))

    assert validation.is_valid is True
    assert validation.analysis is not None
    assert validation.analysis["confidence"] == 0.76
    assert len(validation.analysis["recommendations"]) == 2


def test_invalid_json_response_is_rejected() -> None:
    """Verify prose or malformed JSON is rejected."""

    validation = validate_strategic_analysis_response("Here is the analysis")

    assert validation.is_valid is False
    assert validation.errors == ("response is not strict JSON",)


def test_confidence_out_of_range_is_rejected() -> None:
    """Verify top-level confidence must remain in the 0..1 range."""

    payload = _valid_analysis()
    payload["confidence"] = 1.5

    validation = validate_strategic_analysis_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("confidence must be numeric between 0 and 1" in error for error in validation.errors)


def test_missing_required_fields_are_rejected() -> None:
    """Verify exact root schema is required."""

    payload = _valid_analysis()
    payload.pop("root_causes")

    validation = validate_strategic_analysis_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("response must contain exactly" in error for error in validation.errors)


def test_oversized_outputs_are_rejected() -> None:
    """Verify long model-authored strings cannot pass validation."""

    payload = _valid_analysis()
    payload["executive_summary"] = "X" * 1300

    validation = validate_strategic_analysis_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("executive_summary" in error for error in validation.errors)


def test_recommendation_count_limit_is_enforced() -> None:
    """Verify excessive recommendation lists are rejected."""

    payload = _valid_analysis()
    payload["recommendations"] = payload["recommendations"] * 5

    validation = validate_strategic_analysis_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("recommendations may contain at most" in error for error in validation.errors)


def test_successful_analysis_generation_uses_mocked_ollama() -> None:
    """Verify accepted mocked Ollama output becomes an analysis document."""

    client = FakeAnalysisClient(True, json.dumps(_valid_analysis()))

    result = create_strategic_analysis(
        client=client,
        evidence_package=_evidence_package(),
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert result.accepted is True
    assert result.analysis_document["validation_status"] == "accepted"
    assert result.analysis_document["recommendation_count"] == 2
    assert result.analysis_document["analysis"]["confidence"] == 0.76
    assert client.generate_calls == 1


def test_unavailable_ollama_rejects_without_generation() -> None:
    """Verify unavailable Ollama does not call generate or invent analysis."""

    client = FakeAnalysisClient(False)

    result = create_strategic_analysis(
        client=client,
        evidence_package=_evidence_package(),
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert result.accepted is False
    assert result.analysis_document["validation_status"] == "unavailable"
    assert result.analysis_document["analysis_generated"] is False
    assert client.generate_calls == 0


def test_prompt_is_compact_and_omits_full_evidence_rows() -> None:
    """Verify prompt includes summaries but not full row payloads."""

    prompt = build_strategic_analysis_prompt(
        evidence_package=_evidence_package(),
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert "STRATEGIC_ANALYSIS_CONTEXT" in prompt
    assert "Never" not in prompt
    assert "MUST_NOT_APPEAR" not in prompt
    assert "records" not in prompt
    assert "net_operating_result" in prompt
