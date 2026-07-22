"""Tests for deterministic fact placeholders in modular reasoning."""

from __future__ import annotations

import json
from typing import Any

from finance_agent.reasoning.fact_registry import FactRegistry, validate_placeholders_in_payload
from finance_agent.reasoning.reasoning_pipeline import (
    _run_structured_stage,
    reasoning_stage_json_schema,
    validate_reasoning_stage_response,
)


def _ledger() -> dict[str, Any]:
    """Return a compact evidence ledger with deterministic facts.

    Inputs: none.
    Outputs: ledger dictionary containing one currency, percentage, period and
    entity-bearing fact.
    Assumptions: values are already approved by deterministic Python stages.
    """

    return {
        "facts": [
            {
                "evidence_id": "finance.metric.net_operating_result",
                "metric": "net_operating_result",
                "display_value": "$100",
                "raw_value": 100,
                "unit": "USD",
                "period": "2026_12",
                "entity": "",
                "source_reference": "outputs/calculations/finance_summary_2026_12.json",
            },
            {
                "evidence_id": "finance.metric.collection_rate",
                "metric": "collection_rate",
                "display_value": "95.0%",
                "raw_value": 0.95,
                "unit": "ratio",
                "period": "2026_12",
                "entity": "",
                "source_reference": "outputs/calculations/finance_summary_2026_12.json",
            },
            {
                "evidence_id": "anomaly.health_sciences.overtime",
                "metric": "overtime_variance",
                "display_value": "$25",
                "raw_value": 25,
                "unit": "USD",
                "period": "2026_12",
                "entity": "Health Sciences",
                "source_reference": "outputs/anomalies/anomaly_report_2026_12.json",
            },
        ],
        "approved_numbers": ["100", "95.0%", "25"],
        "approved_periods": ["2026_12"],
        "approved_entities": ["Health Sciences"],
    }


def _placeholder(evidence_id: str, metric_name: str) -> str:
    """Return a placeholder for one test fact.

    Inputs: evidence ID and metric name.
    Outputs: placeholder token.
    Assumptions: registry construction is deterministic for the fixture ledger.
    """

    registry = FactRegistry.from_evidence_ledger(_ledger())
    for fact in registry.facts:
        if evidence_id in fact.evidence_ids and fact.metric_name == metric_name:
            return fact.placeholder
    raise AssertionError(f"Missing placeholder for {evidence_id}/{metric_name}")


def _stage_payload(text: str, evidence_ids: list[str] | None = None) -> dict[str, Any]:
    """Build a minimal Stage 1 payload.

    Inputs: claim text and optional evidence IDs.
    Outputs: schema-compatible Stage 1 payload.
    Assumptions: tests vary only placeholder usage.
    """

    ids = evidence_ids or ["finance.metric.net_operating_result"]
    return {
        "claims": [{"text": text, "evidence_ids": ids, "confidence": 0.8, "claim_type": "fact"}],
        "risks": [],
        "opportunities": [],
        "open_questions": [],
    }


def test_fact_ids_are_stable_within_run() -> None:
    """Verify registry construction produces stable placeholders.

    Inputs: identical ledgers.
    Outputs: same ordered placeholders.
    Assumptions: stability is per deterministic input ledger.
    """

    first = FactRegistry.from_evidence_ledger(_ledger())
    second = FactRegistry.from_evidence_ledger(_ledger())

    assert [fact.placeholder for fact in first.facts] == [fact.placeholder for fact in second.facts]


def test_prompt_facts_hide_display_values_but_keep_numeric_metadata() -> None:
    """Verify prompts expose placeholders instead of formatted literals.

    Inputs: registry built from ledger.
    Outputs: prompt fact descriptors with numeric metadata and no display value.
    Assumptions: Python owns final formatting.
    """

    registry = FactRegistry.from_evidence_ledger(_ledger())
    prompt_fact = next(fact for fact in registry.prompt_facts() if fact["metric_name"] == "net_operating_result")

    assert prompt_fact["placeholder"].startswith("{{FACT_")
    assert prompt_fact["numeric_value"] == 100
    assert "display_value" not in prompt_fact


def test_recursive_substitution_preserves_model_prose() -> None:
    """Verify substitution only replaces placeholder tokens.

    Inputs: nested model payload.
    Outputs: substituted payload and audit entries.
    Assumptions: surrounding prose remains model-authored.
    """

    registry = FactRegistry.from_evidence_ledger(_ledger())
    placeholder = _placeholder("finance.metric.net_operating_result", "net_operating_result")
    payload = {"claims": [{"text": f"El resultado fue {placeholder}.", "evidence_ids": ["finance.metric.net_operating_result"]}]}

    substituted, audit = registry.substitute(payload)

    assert substituted["claims"][0]["text"] == "El resultado fue $100."
    assert audit[0]["display_value"] == "$100"


def test_unknown_and_malformed_placeholders_are_rejected() -> None:
    """Verify invalid placeholder tokens fail before substitution.

    Inputs: payload with unknown and malformed placeholders.
    Outputs: invalid validation result.
    Assumptions: fail-closed behavior protects report generation.
    """

    registry = FactRegistry.from_evidence_ledger(_ledger())
    payload = _stage_payload("El valor {{FACT_999}} y {FACT_001} no son válidos.")
    result = validate_placeholders_in_payload(payload, registry)

    assert not result.is_valid
    assert any("unknown placeholder" in error for error in result.errors)
    assert any("malformed placeholder" in error for error in result.errors)


def test_unsupported_numeric_literal_rejected_even_if_approved() -> None:
    """Verify model prose may not write deterministic numbers directly.

    Inputs: narrative using approved display value without placeholder.
    Outputs: validation rejection.
    Assumptions: exact approved values must still be substituted by Python.
    """

    validation = validate_reasoning_stage_response(
        json.dumps(_stage_payload("El resultado operativo fue $100."), ensure_ascii=False),
        stage_id="financial_performance",
        evidence_ledger=_ledger(),
        fact_registry=FactRegistry.from_evidence_ledger(_ledger()),
    )

    assert not validation.is_valid
    assert any("numeric" in error or "literal" in error for error in validation.errors)


def test_placeholder_must_be_supported_by_item_evidence_ids() -> None:
    """Verify placeholders cite compatible evidence IDs.

    Inputs: text using a collection placeholder but citing net result evidence.
    Outputs: validation rejection.
    Assumptions: placeholders and evidence IDs must point to the same facts.
    """

    collection = _placeholder("finance.metric.collection_rate", "collection_rate")
    validation = validate_reasoning_stage_response(
        json.dumps(_stage_payload(f"La cobranza fue {collection}."), ensure_ascii=False),
        stage_id="financial_performance",
        evidence_ledger=_ledger(),
        fact_registry=FactRegistry.from_evidence_ledger(_ledger()),
    )

    assert not validation.is_valid
    assert any("without supporting evidence_ids" in error for error in validation.errors)


class _RepairClient:
    """Fake client that returns one bad payload and one repaired payload."""

    def __init__(self) -> None:
        """Create the fake repair client.

        Inputs: none.
        Outputs: queued responses.
        Assumptions: response format is mutable like the Ollama client.
        """

        placeholder = _placeholder("finance.metric.net_operating_result", "net_operating_result")
        self.responses = [
            json.dumps(_stage_payload("El resultado operativo fue $100."), ensure_ascii=False),
            json.dumps(_stage_payload(f"El resultado operativo fue {placeholder}."), ensure_ascii=False),
        ]
        self.response_format: str | dict[str, Any] = "json"

    def generate_with_metadata(self, prompt: str) -> dict[str, Any]:
        """Return the next fake response.

        Inputs: prompt.
        Outputs: Ollama-like response envelope.
        Assumptions: exactly two calls are expected.
        """

        return {"response": self.responses.pop(0), "telemetry": {"prompt_characters": len(prompt)}}


def test_placeholder_repair_retry_preserves_claim_and_accepts() -> None:
    """Verify one placeholder repair retry can fix literal formatting only.

    Inputs: fake client with literal first response and placeholder repair.
    Outputs: accepted stage result with substitution audit.
    Assumptions: retry does not invent claims or evidence IDs.
    """

    registry = FactRegistry.from_evidence_ledger(_ledger())
    result = _run_structured_stage(
        client=_RepairClient(),
        stage_id="financial_performance",
        stage_name="Financial Performance",
        prompt="prompt",
        validator=lambda text: validate_reasoning_stage_response(
            text,
            stage_id="financial_performance",
            evidence_ledger=_ledger(),
            fact_registry=registry,
        ),
        response_format=reasoning_stage_json_schema("financial_performance"),
        fact_registry=registry,
    )

    assert result.accepted
    assert result.telemetry["placeholder_retry_attempted"] is True
    assert result.payload["_substituted_payload"]["claims"][0]["text"] == "El resultado operativo fue $100."
