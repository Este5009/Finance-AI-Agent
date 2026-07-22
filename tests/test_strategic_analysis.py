"""Tests for Step 9 Ollama strategic financial analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass

from finance_agent.analysis.strategic_analysis import (
    build_evidence_ledger,
    build_strategic_analysis_prompt,
    create_strategic_analysis,
    validate_strategic_analysis_response,
    validate_evidence_bound_claims,
    validate_user_facing_spanish,
)


@dataclass
class FakeAnalysisClient:
    """Configurable test double for the Step 9 Ollama client."""

    available: bool
    response: str = ""
    responses: tuple[str, ...] = ()
    generate_calls: int = 0
    last_prompt: str = ""
    response_format: object = "json"
    observed_formats: tuple[object, ...] = ()

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
        self.observed_formats = (*self.observed_formats, self.response_format)
        if self.responses:
            index = min(self.generate_calls - 1, len(self.responses) - 1)
            return self.responses[index]
        return self.response


def _valid_analysis() -> dict[str, object]:
    """Build a valid strategic-analysis payload fixture.

    Inputs: none.
    Outputs: JSON-compatible analysis payload.
    Assumptions: values are intentionally concise for schema validation.
    """

    return {
        "executive_summary": (
            "El desempeño de junio muestra déficit operativo, flujo de caja "
            "negativo y presión de cobranza que requiere atención directiva."
        ),
        "key_findings": [
            "El resultado operativo es negativo según el resumen financiero procesado.",
            "La cobranza estudiantil y las facturas vencidas aparecen como señales de riesgo.",
        ],
        "root_causes": [
            "La presión de gastos parece superar el desempeño de ingresos.",
            "La cobranza probablemente está retrasada en una parte de las facturas estudiantiles.",
        ],
        "financial_health_analysis": "La salud financiera combina déficit operativo de -200, flujo de caja de -300 y nómina equivalente a 52.0% de ingresos.",
        "kpi_analysis": "El KPI de cobranza estudiantil está en 84.0%, por debajo del nivel esperado para sostener caja.",
        "historical_summary": "El contexto histórico disponible para junio de 2026 muestra presión recurrente en caja y cobranza.",
        "historical_trend_analysis": "La tendencia histórica disponible debe revisarse contra junio de 2026 y el KPI de cobranza.",
        "department_analysis": "Engineering presenta una variación de 100 que requiere revisión con evidencia departamental.",
        "anomaly_analysis": "El reporte contiene 2 anomalías, incluyendo una crítica relacionada con resultado operativo.",
        "recommendation_follow_up_analysis": "No hay seguimiento previo suficiente para cerrar recomendaciones con la evidencia actual.",
        "longitudinal_risk_analysis": "El riesgo longitudinal principal es la persistencia de caja negativa y presión de cobranza.",
        "strategic_recommendations": [
            {
                "priority": "high",
                "action": "Revisar aprobaciones de gasto por departamento en categorías con sobrepresupuesto.",
                "rationale": "La evidencia procesada muestra presión por departamento y categoría.",
                "supporting_evidence": "El paquete de evidencia incluye recuperación departamental y del reporte.",
                "expected_impact": "Reducir variaciones prevenibles en el próximo ciclo de reporte.",
                "evidence_ids": ["evidence.step_001.summary", "finance.metric.total_expenses"],
                "confidence": 0.78,
            },
            {
                "priority": "medium",
                "action": "Priorizar seguimiento de facturas estudiantiles vencidas.",
                "rationale": "Las anomalías de cobranza y la evidencia vencida indican riesgo de cuentas por cobrar.",
                "supporting_evidence": "Las transacciones de pagos estudiantiles incluyen registros vencidos.",
                "expected_impact": "Mejorar la conversión de caja y reducir saldos pendientes.",
                "evidence_ids": ["finance.metric.collection_rate"],
                "confidence": 0.74,
            },
        ],
        "strategic_priorities": [
            "Estabilizar el flujo de caja.",
            "Reducir la variación de gastos.",
        ],
        "missing_information": ["Notas de aprobación para pagos de proveedores marcados."],
        "narrative_evidence": {
            "executive_summary": ["finance.metric.net_operating_result", "finance.metric.net_cash_flow"],
            "key_findings": ["finance.metric.net_operating_result", "anomaly.anom_1"],
            "root_causes": ["finance.metric.total_expenses", "evidence.step_001.summary"],
            "financial_health_analysis": ["finance.metric.net_operating_result", "finance.metric.payroll_percentage_of_revenue"],
            "kpi_analysis": ["finance.metric.collection_rate"],
            "historical_summary": ["finance.metric.net_cash_flow"],
            "historical_trend_analysis": ["finance.metric.collection_rate"],
            "department_analysis": ["finance.department.engineering.variance"],
            "anomaly_analysis": ["anomaly.total_count", "anomaly.anom_1"],
            "recommendation_follow_up_analysis": ["evidence.step_001.summary"],
            "longitudinal_risk_analysis": ["finance.metric.net_cash_flow"],
            "strategic_priorities": ["finance.metric.net_cash_flow", "finance.metric.total_expenses"],
            "missing_information": ["evidence.step_001.summary"],
            "reasoning_summary": ["finance.metric.net_operating_result", "anomaly.anom_1", "evidence.step_001.summary"],
        },
        "confidence": 0.76,
        "reasoning_summary": (
            "Las conclusiones combinan métricas financieras procesadas, severidad de anomalías "
            "y disponibilidad de evidencia recuperada sin recalcular valores."
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


def test_analysis_wrapper_is_safely_unwrapped() -> None:
    """Verify a harmless top-level analysis envelope is accepted."""

    validation = validate_strategic_analysis_response(
        json.dumps({"analysis": _valid_analysis()}, ensure_ascii=False)
    )

    assert validation.is_valid is True
    assert validation.analysis is not None
    assert validation.analysis["executive_summary"]


def test_section_narrative_blocks_are_normalized() -> None:
    """Verify new section output blocks become compatible string fields."""

    payload = _valid_analysis()
    payload["executive_summary"] = {
        "text": payload["executive_summary"],
        "evidence_ids": ["finance.metric.net_cash_flow"],
    }
    payload["financial_health_analysis"] = {
        "text": payload["financial_health_analysis"],
        "evidence_ids": ["finance.metric.net_operating_result"],
    }

    validation = validate_strategic_analysis_response(json.dumps(payload, ensure_ascii=False))

    assert validation.is_valid is True
    assert validation.analysis is not None
    assert isinstance(validation.analysis["executive_summary"], str)
    assert validation.analysis["narrative_evidence"]["executive_summary"] == ["finance.metric.net_cash_flow"]


def test_english_user_facing_response_is_rejected() -> None:
    """Verify schema-valid English narrative fails Spanish validation."""

    payload = _valid_analysis()
    payload["executive_summary"] = "The financial performance shows cash flow risk and requires management review."

    validation = validate_strategic_analysis_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("executive_summary" in error for error in validation.errors)


def test_spanish_language_validator_allows_common_acronyms() -> None:
    """Verify common finance/report acronyms do not trigger English rejection."""

    payload = _valid_analysis()
    payload["executive_summary"] = "El KPI principal se mantiene en USD y el PDF conserva evidencia ejecutiva."

    errors = validate_user_facing_spanish(payload)

    assert errors == ()


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
    payload["strategic_recommendations"] = payload["strategic_recommendations"] * 5

    validation = validate_strategic_analysis_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("strategic_recommendations may contain at most" in error for error in validation.errors)


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
    assert result.telemetry is not None
    assert result.telemetry["context_characters"] > 0
    assert result.telemetry["deduplicate_context"] is True
    assert result.telemetry["spanish_rewrite_attempted"] is False
    assert isinstance(client.observed_formats[0], dict)
    assert client.response_format == "json"


def test_spanish_rewrite_retry_accepts_second_response() -> None:
    """Verify one schema-valid English response is retried and accepted in Spanish."""

    english = _valid_analysis()
    english["executive_summary"] = "The financial performance shows cash flow risk and requires management review."
    spanish = _valid_analysis()
    client = FakeAnalysisClient(True, responses=(json.dumps(english), json.dumps(spanish)))

    result = create_strategic_analysis(
        client=client,
        evidence_package=_evidence_package(),
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert result.accepted is True
    assert client.generate_calls == 2
    assert "SPANISH_REWRITE_INPUT" in client.last_prompt
    assert result.telemetry is not None
    assert result.telemetry["spanish_rewrite_attempted"] is True
    assert result.analysis_document["analysis"]["confidence"] == spanish["confidence"]
    assert result.analysis_document["analysis"]["recommendations"][0]["priority"] == "high"


def test_spanish_rewrite_retry_failure_rejects_analysis() -> None:
    """Verify analysis is rejected when the bounded Spanish retry still fails."""

    english = _valid_analysis()
    english["executive_summary"] = "The financial performance shows cash flow risk and requires management review."
    client = FakeAnalysisClient(True, responses=(json.dumps(english), json.dumps(english)))

    result = create_strategic_analysis(
        client=client,
        evidence_package=_evidence_package(),
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert result.accepted is False
    assert client.generate_calls == 2
    assert result.analysis_document["validation_status"] == "rejected"
    assert any("executive_summary" in error for error in result.validation_errors)


def test_evidence_repair_retry_accepts_corrected_claims() -> None:
    """Verify one evidence-bound repair retry can remove unsupported claims."""

    invalid = _valid_analysis()
    invalid["financial_health_analysis"] = "La Escuela de Medicina tuvo una caída de 99% en 2030."
    repaired = _valid_analysis()
    client = FakeAnalysisClient(True, responses=(json.dumps(invalid), json.dumps(repaired)))

    result = create_strategic_analysis(
        client=client,
        evidence_package=_evidence_package(),
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert result.accepted is True
    assert client.generate_calls == 2
    assert "EVIDENCE_REPAIR_TASK" in client.last_prompt


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


def test_supported_payroll_missing_information_is_removed() -> None:
    """Verify false missing payroll/headcount claims are removed after validation."""

    payload = _valid_analysis()
    payload["missing_information"] = [
        "Cambios reales de plantilla durante junio",
        "Motivo del aumento de horas extra",
    ]
    evidence = _evidence_package()
    evidence["evidence_packages"][0]["retrieved_evidence"]["data"][
        "payroll_breakdown"
    ] = [
        {
            "period": "2026-06-01",
            "department": "Engineering",
            "headcount_fte": "76",
            "payroll_amount": "278696",
        }
    ]
    client = FakeAnalysisClient(True, json.dumps(payload))

    result = create_strategic_analysis(
        client=client,
        evidence_package=evidence,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    assert result.accepted is True
    assert result.analysis_document["analysis"]["missing_information"] == [
        "Motivo del aumento de horas extra"
    ]


def test_processed_anomaly_cashflow_and_payroll_missing_claims_are_removed() -> None:
    """Verify generic-period analysis cannot claim available evidence is missing."""

    payload = _valid_analysis()
    payload["missing_information"] = [
        "Validación de datos de cash flow para junio de 2026",
        "Datos de anomalías para junio de 2026",
        "Desglose detallado de overtime y benefits de Health Sciences",
        "Actas de aprobación del comité para compras marcadas",
    ]
    evidence = _evidence_package()
    evidence["period_slug"] = "2026_06"
    evidence["evidence_packages"][0]["source_references"] = [
        "outputs/calculations/finance_summary_2026_06.json",
        "outputs/anomalies/anomaly_report_2026_06.json",
    ]
    evidence["evidence_packages"][0]["retrieved_evidence"]["data"][
        "payroll_breakdown"
    ] = [
        {
            "period": "2026-06-01",
            "department": "Health Sciences",
            "benefits": "61560",
            "overtime": "18810",
            "payroll_amount": "342000",
        }
    ]
    client = FakeAnalysisClient(True, json.dumps(payload))

    result = create_strategic_analysis(
        client=client,
        evidence_package=evidence,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        risk_summary=_risk_summary(),
        period_slug="2026_06",
    )

    assert result.accepted is True
    assert result.analysis_document["analysis"]["missing_information"] == [
        "Actas de aprobación del comité para compras marcadas"
    ]


def test_evidence_bound_validation_rejects_unsupported_claims() -> None:
    """Verify unsupported numbers, periods, and departments cannot pass."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis(), ensure_ascii=False))
    assert validation.analysis is not None
    validation.analysis["financial_health_analysis"] = (
        "La Escuela de Medicina registró una caída de 99% durante 2030."
    )

    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )

    assert any("unsupported number: 99%" in error for error in errors)
    assert any("unsupported period: 2030" in error for error in errors)
    assert any("unsupported named entity: Escuela de Medicina" in error for error in errors)


def test_evidence_bound_validation_rejects_schema_placeholder_prose() -> None:
    """Verify copied schema examples are rejected as generic filler."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis(), ensure_ascii=False))
    assert validation.analysis is not None
    validation.analysis["financial_health_analysis"] = "Análisis español de salud financiera basado en KPIs principales."
    validation.analysis["strategic_recommendations"][0]["action"] = "Acción concreta en español."

    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )

    assert any("generic placeholder prose" in error for error in errors)


def test_evidence_bound_validation_accepts_spanish_number_formats() -> None:
    """Verify decimal-comma percentages and comma-thousands can match evidence."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis(), ensure_ascii=False))
    assert validation.analysis is not None
    validation.analysis["financial_health_analysis"] = (
        "La salud financiera usa 52,0% de nómina sobre ingresos y $1,200 de gastos."
    )

    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )

    assert not any("52.0%" in error or "1200" in error for error in errors)


def test_evidence_bound_validation_rejects_rounding_or_derived_values() -> None:
    """Verify only exact ledger display/raw values are allowed."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis(), ensure_ascii=False))
    assert validation.analysis is not None
    validation.analysis["financial_health_analysis"] = "La nómina representa 52% de ingresos."

    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )

    assert any("unsupported number: 52%" in error for error in errors)


def test_evidence_bound_validation_requires_valid_evidence_ids() -> None:
    """Verify all cited evidence IDs must exist in the ledger."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis(), ensure_ascii=False))
    assert validation.analysis is not None
    validation.analysis["narrative_evidence"]["kpi_analysis"] = ["fake.metric"]

    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )

    assert any("cites unknown evidence_id: fake.metric" in error for error in errors)


def test_causal_claims_require_hypothesis_language_and_evidence() -> None:
    """Verify unsupported factual causes fail while labeled hypotheses pass."""

    validation = validate_strategic_analysis_response(json.dumps(_valid_analysis(), ensure_ascii=False))
    assert validation.analysis is not None
    validation.analysis["root_causes"] = ["El déficit fue causado por mala gestión."]

    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )

    assert any("unsupported causal claim" in error for error in errors)
    validation.analysis["root_causes"] = [
        "Como hipótesis, la presión de gastos parece asociarse con el déficit observado."
    ]
    errors = validate_evidence_bound_claims(
        validation.analysis,
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
    )
    assert not any("unsupported causal claim" in error for error in errors)


def test_evidence_ledger_contains_approved_fact_structure() -> None:
    """Verify ledger facts expose exact values, IDs, period/entity, and sources."""

    ledger = build_evidence_ledger(
        finance_summary=_finance_summary(),
        anomaly_report=_anomaly_report(),
        evidence_package=_evidence_package(),
        risk_summary=_risk_summary(),
        period_slug="june_2026",
    )

    fact = next(item for item in ledger["facts"] if item["evidence_id"] == "finance.metric.collection_rate")
    assert fact["display_value"] == "84.0%"
    assert fact["raw_value"] == 0.84
    assert fact["period"] == "june_2026"
    assert fact["source_reference"].endswith("finance_summary_june_2026.json")
    assert "finance.metric.collection_rate" in ledger["evidence_ids"]


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
    assert "secret_row" not in prompt
    assert "net_operating_result" in prompt
    assert "professional Spanish" in prompt
    assert "historical_summary" in prompt


def test_strategy_prompt_deduplicates_evidence_and_ranks_anomalies() -> None:
    """Verify strategic context avoids repeated evidence/anomaly noise."""

    evidence = _evidence_package()
    evidence["evidence_packages"].append(dict(evidence["evidence_packages"][0]))
    anomalies = _anomaly_report()
    anomalies["anomalies"] = [
        {"anomaly_id": "LOW", "severity": "low", "observed_value": 10000},
        {"anomaly_id": "HIGH", "severity": "high", "observed_value": 2},
        {"anomaly_id": "CRIT", "severity": "critical", "observed_value": 1},
    ]

    prompt = build_strategic_analysis_prompt(
        evidence_package=evidence,
        finance_summary=_finance_summary(),
        anomaly_report=anomalies,
        risk_summary=_risk_summary(),
        period_slug="june_2026",
        deduplicate_context=True,
    )

    assert prompt.count("Retrieved 1 processed report.") == 1
    assert "CRIT" in prompt
    assert "HIGH" in prompt
    assert "LOW" not in prompt
