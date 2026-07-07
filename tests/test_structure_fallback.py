"""Tests for optional, validated Ollama structure interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from finance_agent.understanding.structure_fallback import (
    LowConfidenceItem,
    StructureResponseError,
    build_structure_prompt,
    detect_low_confidence_items,
    enrich_intermediate_model,
    parse_and_validate_structure_response,
)


def _uncertain_table() -> dict[str, object]:
    """Build a compact uncertain table fixture.

    Inputs: none.
    Outputs: serialized intermediate table with one low-confidence mapping.
    Assumptions: the shape mirrors Step 2 model JSON.
    """

    return {
        "table_id": "workbook__datos__table_01",
        "sheet": "Datos",
        "table_title": "Cobros por cohorte",
        "detected_type": "Unknown",
        "confidence": 0.30,
        "requires_future_ollama_interpretation": True,
        "original_columns": ["Ciclo AcadÃ©mico", "Monto"],
        "normalized_columns": ["ciclo_academico", "amount"],
        "column_mappings": [
            {
                "original_name": "Ciclo AcadÃ©mico",
                "normalized_name": "ciclo_academico",
                "confidence": 0.55,
            },
            {
                "original_name": "Monto",
                "normalized_name": "amount",
                "confidence": 0.98,
            },
        ],
        "extracted_dimensions": [
            {"semantic_name": "ciclo_academico", "confidence": 0.55}
        ],
        "extracted_metrics": [{"semantic_name": "amount", "confidence": 0.96}],
        "sample_rows": [
            {"ciclo_academico": index, "amount": index * 100}
            for index in range(1, 8)
        ],
    }


def _valid_response() -> str:
    """Return a valid allowlisted Ollama response fixture.

    Inputs: none.
    Outputs: strict JSON string.
    Assumptions: Unknown remains valid when the existing table taxonomy is insufficient.
    """

    return json.dumps(
        {
            "suggested_table_type": "Revenue",
            "table_confidence": 0.87,
            "column_mappings": {
                "Ciclo AcadÃ©mico": "student_year",
                "Monto": "amount",
            },
            "dimension_fields": ["student_year"],
            "metric_fields": ["amount"],
            "reasoning_short": "The rows contain cohort labels and monetary values.",
            "requires_human_review": False,
        }
    )


@dataclass
class FakeInterpreter:
    """Test double for deterministic Ollama availability and responses."""

    available: bool
    response: str = ""
    calls: int = 0

    def is_available(self) -> bool:
        """Return configured test availability.

        Inputs: fixture state.
        Outputs: configured availability boolean.
        Assumptions: no network request is made.
        """

        return self.available

    def generate(self, prompt: str) -> str:
        """Record one call and return the configured response.

        Inputs: generated prompt.
        Outputs: configured response text.
        Assumptions: prompt content is tested separately.
        """

        self.calls += 1
        return self.response


def test_low_confidence_detection_identifies_table_and_column() -> None:
    """Verify unknown types and weak column mappings enter the review queue."""

    items = detect_low_confidence_items({"tables": [_uncertain_table()]})

    assert len(items) == 1
    assert items[0].table_id == "workbook__datos__table_01"
    assert "unknown_table_type" in items[0].reasons
    assert items[0].uncertain_columns == ("Ciclo AcadÃ©mico",)


def test_prompt_is_compact_and_contains_only_five_sample_rows() -> None:
    """Verify prompt evidence is bounded and contains the required schema context."""

    table = _uncertain_table()
    item = LowConfidenceItem(
        table_id="workbook__datos__table_01",
        reasons=("unknown_table_type",),
        uncertain_columns=("Ciclo AcadÃ©mico",),
    )

    prompt = build_structure_prompt(table, item)
    prompt_payload = json.loads(prompt.split("\nINPUT:\n", maxsplit=1)[1])

    assert prompt_payload["source_sheet_name"] == "Datos"
    assert len(prompt_payload["sample_rows"]) == 5
    assert "allowed_table_types" in prompt_payload
    assert "allowed_canonical_fields" in prompt_payload
    assert "source_workbook" not in prompt_payload


def test_valid_json_response_is_parsed() -> None:
    """Verify a contract-compliant response becomes a typed suggestion."""

    suggestion = parse_and_validate_structure_response(
        _valid_response(),
        original_columns=["Ciclo AcadÃ©mico", "Monto"],
    )

    assert suggestion.suggested_table_type == "Revenue"
    assert suggestion.table_confidence == 0.87
    assert suggestion.column_mappings["Ciclo AcadÃ©mico"] == "student_year"


def test_invalid_json_response_is_rejected() -> None:
    """Verify prose or malformed model output cannot affect deterministic data."""

    with pytest.raises(StructureResponseError, match="strict JSON"):
        parse_and_validate_structure_response(
            "```json\n{}\n```",
            original_columns=["Ciclo AcadÃ©mico", "Monto"],
        )


def test_invalid_canonical_field_is_rejected() -> None:
    """Verify Ollama cannot introduce canonical fields outside Python's allowlist."""

    payload = json.loads(_valid_response())
    payload["column_mappings"]["Monto"] = "made_up_finance_field"

    with pytest.raises(StructureResponseError, match="invalid canonical"):
        parse_and_validate_structure_response(
            json.dumps(payload),
            original_columns=["Ciclo AcadÃ©mico", "Monto"],
        )


def test_unavailable_ollama_preserves_deterministic_fallback() -> None:
    """Verify unavailable Ollama produces a reviewable enriched model without calls."""

    original = {"model_version": "2.0", "tables": [_uncertain_table()]}
    interpreter = FakeInterpreter(available=False)

    enriched, summary = enrich_intermediate_model(original, interpreter)
    table = enriched["tables"][0]

    assert interpreter.calls == 0
    assert summary.ollama_available is False
    assert summary.items_reviewed == 0
    assert summary.requiring_human_review == 1
    assert table["llm_reviewed"] is False
    assert table["final_table_type"] == "Unknown"
    assert table["final_column_mappings"]["Ciclo AcadÃ©mico"] == "ciclo_academico"
    assert table["requires_human_review"] is True


def test_enriched_model_accepts_only_validated_uncertain_mapping() -> None:
    """Verify accepted output enriches structure while locking strong Python mappings."""

    original = {"model_version": "2.0", "tables": [_uncertain_table()]}
    interpreter = FakeInterpreter(available=True, response=_valid_response())

    enriched, summary = enrich_intermediate_model(original, interpreter)
    table = enriched["tables"][0]

    assert interpreter.calls == 1
    assert summary.items_reviewed == 1
    assert summary.accepted == 1
    assert summary.rejected == 0
    assert table["llm_reviewed"] is True
    assert table["llm_suggested_type"] == "Revenue"
    assert table["final_table_type"] == "Revenue"
    assert table["final_column_mappings"]["Ciclo AcadÃ©mico"] == "student_year"
    # "Monto" was already a strong deterministic mapping and remains locked.
    assert table["final_column_mappings"]["Monto"] == "amount"
    assert table["requires_human_review"] is False
    assert enriched["enrichment"]["deterministic_fields_preserved"] is True


def test_invalid_response_is_rejected_and_keeps_original_mapping() -> None:
    """Verify invalid model output is counted and cannot overwrite Python results."""

    original = {"model_version": "2.0", "tables": [_uncertain_table()]}
    interpreter = FakeInterpreter(available=True, response="not json")

    enriched, summary = enrich_intermediate_model(original, interpreter)
    table = enriched["tables"][0]

    assert summary.rejected == 1
    assert table["final_table_type"] == "Unknown"
    assert table["final_column_mappings"]["Ciclo AcadÃ©mico"] == "ciclo_academico"
    assert table["requires_human_review"] is True


def test_unknown_suggestion_cannot_resolve_unknown_table() -> None:
    """Verify an unresolved type remains queued even with high LLM confidence."""

    payload = json.loads(_valid_response())
    payload["suggested_table_type"] = "Unknown"
    original = {"model_version": "2.0", "tables": [_uncertain_table()]}
    interpreter = FakeInterpreter(
        available=True,
        response=json.dumps(payload),
    )

    enriched, summary = enrich_intermediate_model(original, interpreter)

    assert summary.accepted == 0
    assert summary.rejected == 1
    assert enriched["tables"][0]["requires_human_review"] is True


def test_conflicting_type_cannot_override_strong_deterministic_type() -> None:
    """Verify disagreement with a confident Python type requires human review."""

    table = _uncertain_table()
    table["detected_type"] = "Department_Summary"
    table["confidence"] = 0.90
    original = {"model_version": "2.0", "tables": [table]}
    interpreter = FakeInterpreter(available=True, response=_valid_response())

    enriched, summary = enrich_intermediate_model(original, interpreter)
    enriched_table = enriched["tables"][0]

    assert summary.accepted == 0
    assert summary.rejected == 1
    assert enriched_table["final_table_type"] == "Department_Summary"
    assert enriched_table["requires_human_review"] is True
