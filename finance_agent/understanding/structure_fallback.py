"""Optional Ollama fallback for low-confidence intermediate-model structure."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from finance_agent.understanding.classification import CLASSIFICATION_RULES
from finance_agent.understanding.normalization import COLUMN_CONFIDENCE_THRESHOLD
from finance_agent.llm.ollama_client import OllamaError
from finance_agent.ingestion.schema import COLUMN_ALIASES


TABLE_CONFIDENCE_THRESHOLD = 0.75
FEATURE_CONFIDENCE_THRESHOLD = 0.50
LLM_ACCEPTANCE_THRESHOLD = 0.75

ALLOWED_TABLE_TYPES = frozenset(
    {"Unknown", *(rule.table_type for rule in CLASSIFICATION_RULES)}
)

# The LLM may select only fields already understood by the Python schema. Extra
# fields cover deterministic compound columns used by the current finance model.
ALLOWED_CANONICAL_FIELDS = frozenset(
    {
        "unknown",
        *COLUMN_ALIASES.values(),
        "anomaly_id",
        "anomaly_type",
        "capital_outflows",
        "description",
        "detected_period",
        "expense_variance",
        "expense_variance_pct",
        "high_value_flag",
        "net_contribution",
        "observed_value",
        "payment_id",
        "potential_duplicate",
        "severity",
        "source_sheet",
        "threshold",
        "utilization_pct",
    }
)

# Each tuple is an important semantic group; at least one member of every group
# should be present before a known table type is considered structurally clear.
IMPORTANT_FIELD_GROUPS: dict[str, tuple[frozenset[str], ...]] = {
    "Revenue": (frozenset({"actual_revenue", "revenue", "amount"}),),
    "Expenses": (frozenset({"actual_expense", "expenses", "amount"}),),
    "Budget_vs_Actual": (
        frozenset({"budget", "budget_amount", "budget_revenue", "budget_expense"}),
        frozenset({"actual", "actual_amount", "actual_revenue", "actual_expense"}),
    ),
    "Payroll": (
        frozenset({"total_payroll", "payroll", "salary", "base_salary", "amount"}),
    ),
    "Student_Payments": (
        frozenset({"student", "student_id"}),
        frozenset({"amount_paid", "payment", "amount"}),
    ),
    "Scholarships": (
        frozenset({"scholarship", "scholarship_type"}),
        frozenset({"allocated", "awarded", "amount"}),
    ),
    "Cash_Flow": (
        frozenset(
            {"net_cash_flow", "cash_inflows", "cash_outflows", "ending_cash"}
        ),
    ),
    "Vendor_Payments": (
        frozenset({"vendor"}),
        frozenset({"amount", "payment"}),
    ),
    "Department_Summary": (
        frozenset({"department"}),
        frozenset({"actual_revenue", "revenue"}),
        frozenset({"actual_expense", "expenses"}),
    ),
    "Executive_Summary": (
        frozenset({"metric"}),
        frozenset({"actual", "amount"}),
    ),
}


class StructureInterpreter(Protocol):
    """Protocol implemented by the Ollama client and test doubles."""

    def is_available(self) -> bool:
        """Return whether the interpreter service can be reached."""

    def generate(self, prompt: str) -> str:
        """Return model-authored structure JSON as text."""


@dataclass(frozen=True)
class LowConfidenceItem:
    """Evidence explaining why one table needs optional LLM interpretation."""

    table_id: str
    reasons: tuple[str, ...]
    uncertain_columns: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedStructureSuggestion:
    """Typed, allowlisted structure suggestion parsed from Ollama JSON."""

    suggested_table_type: str
    table_confidence: float
    column_mappings: dict[str, str]
    dimension_fields: list[str]
    metric_fields: list[str]
    reasoning_short: str
    requires_human_review: bool


@dataclass(frozen=True)
class FallbackSummary:
    """Execution counts for one enriched-model fallback run."""

    ollama_available: bool
    items_reviewed: int
    accepted: int
    rejected: int
    requiring_human_review: int
    uncertain_columns: int


class StructureResponseError(ValueError):
    """Raised when model-authored structure JSON violates the Python contract."""


def _mapping_confidence(mapping: dict[str, Any]) -> float:
    """Read a mapping confidence defensively.

    Inputs: serialized deterministic column mapping.
    Outputs: numeric confidence, defaulting to zero when malformed.
    Assumptions: malformed deterministic metadata should trigger review.
    """

    value = mapping.get("confidence", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def detect_low_confidence_items(
    model: dict[str, Any],
    *,
    table_threshold: float = TABLE_CONFIDENCE_THRESHOLD,
    column_threshold: float = COLUMN_CONFIDENCE_THRESHOLD,
) -> list[LowConfidenceItem]:
    """Find tables whose deterministic structure needs limited LLM assistance.

    Inputs: serialized intermediate model and confidence thresholds.
    Outputs: one review item per uncertain table, including uncertain columns.
    Assumptions: a table is the smallest prompt unit because column roles need context.
    """

    items: list[LowConfidenceItem] = []
    for table in model.get("tables", []):
        reasons: list[str] = []
        detected_type = table.get("detected_type", "Unknown")
        confidence = table.get("confidence", 0.0)
        if detected_type == "Unknown":
            reasons.append("unknown_table_type")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or confidence < table_threshold:
            reasons.append("low_table_confidence")

        uncertain_columns = tuple(
            str(mapping.get("original_name", ""))
            for mapping in table.get("column_mappings", [])
            if _mapping_confidence(mapping) < column_threshold
        )
        if uncertain_columns:
            reasons.append("low_column_confidence")

        normalized_fields = set(table.get("normalized_columns", []))
        missing_groups = [
            sorted(group)
            for group in IMPORTANT_FIELD_GROUPS.get(str(detected_type), ())
            if normalized_fields.isdisjoint(group)
        ]
        if missing_groups:
            reasons.append("important_columns_missing")

        features = [
            *table.get("extracted_dimensions", []),
            *table.get("extracted_metrics", []),
        ]
        unclear_roles = any(
            isinstance(feature.get("confidence"), (int, float))
            and not isinstance(feature.get("confidence"), bool)
            and feature["confidence"] < FEATURE_CONFIDENCE_THRESHOLD
            for feature in features
        )
        if (
            not table.get("extracted_dimensions")
            or not table.get("extracted_metrics")
            or unclear_roles
        ):
            reasons.append("dimensions_or_metrics_unclear")

        if table.get("requires_future_ollama_interpretation") and not reasons:
            reasons.append("deterministic_review_requested")
        if reasons:
            items.append(
                LowConfidenceItem(
                    table_id=str(table.get("table_id", "")),
                    reasons=tuple(dict.fromkeys(reasons)),
                    uncertain_columns=uncertain_columns,
                )
            )
    return items


def build_structure_prompt(
    table: dict[str, Any],
    item: LowConfidenceItem,
) -> str:
    """Build a compact structure-only prompt from bounded intermediate evidence.

    Inputs: one serialized table and its low-confidence assessment.
    Outputs: prompt containing metadata and at most five existing sample rows.
    Assumptions: no raw workbook, PDF, or full normalized CSV is included.
    """

    prompt_input = {
        "source_sheet_name": table.get("sheet"),
        "table_title_or_context": table.get("table_title"),
        "original_columns": table.get("original_columns", []),
        "normalized_candidate_columns": table.get("normalized_columns", []),
        "sample_rows": table.get("sample_rows", [])[:5],
        "detected_dimensions": [
            feature.get("semantic_name")
            for feature in table.get("extracted_dimensions", [])
        ],
        "detected_metrics": [
            feature.get("semantic_name")
            for feature in table.get("extracted_metrics", [])
        ],
        "current_confidence_scores": {
            "table": table.get("confidence"),
            "columns": {
                mapping.get("original_name"): mapping.get("confidence")
                for mapping in table.get("column_mappings", [])
            },
        },
        "review_reasons": list(item.reasons),
        "allowed_table_types": sorted(ALLOWED_TABLE_TYPES),
        "allowed_canonical_fields": sorted(ALLOWED_CANONICAL_FIELDS),
    }
    return (
        "You are a financial table structure interpreter. Do not calculate, "
        "summarize finances, or invent fields. Return exactly one JSON object "
        "with keys suggested_table_type, table_confidence, column_mappings, "
        "dimension_fields, metric_fields, reasoning_short, and "
        "requires_human_review. Map original column names only to an allowed "
        "canonical field or \"unknown\". Keep reasoning_short under 240 "
        "characters.\nINPUT:\n"
        + json.dumps(prompt_input, ensure_ascii=False, separators=(",", ":"))
    )


def _validate_field_list(value: Any, field_name: str) -> list[str]:
    """Validate one model-authored dimension or metric field list.

    Inputs: untrusted value and response field name.
    Outputs: validated canonical field names.
    Assumptions: unknown is permitted to preserve ambiguity explicitly.
    """

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise StructureResponseError(f"{field_name} must be a list of strings.")
    invalid = sorted(set(value) - ALLOWED_CANONICAL_FIELDS)
    if invalid:
        raise StructureResponseError(
            f"{field_name} contains invalid canonical fields: {invalid}."
        )
    return list(value)


def parse_and_validate_structure_response(
    response_text: str,
    *,
    original_columns: list[str],
) -> ValidatedStructureSuggestion:
    """Parse and fully validate untrusted Ollama structure JSON.

    Inputs: raw response text and the table's exact original columns.
    Outputs: typed allowlisted suggestion.
    Assumptions: markdown fences or surrounding prose are invalid by design.
    """

    try:
        payload = json.loads(response_text.strip())
    except (json.JSONDecodeError, AttributeError) as exc:
        raise StructureResponseError("Ollama response is not strict JSON.") from exc
    if not isinstance(payload, dict):
        raise StructureResponseError("Ollama response must be a JSON object.")

    required_keys = {
        "suggested_table_type",
        "table_confidence",
        "column_mappings",
        "dimension_fields",
        "metric_fields",
        "reasoning_short",
        "requires_human_review",
    }
    if set(payload) != required_keys:
        raise StructureResponseError("Ollama response has missing or extra keys.")

    suggested_type = payload["suggested_table_type"]
    if suggested_type not in ALLOWED_TABLE_TYPES:
        raise StructureResponseError("Suggested table type is not allowed.")
    confidence = payload["table_confidence"]
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        raise StructureResponseError("Table confidence must be numeric from 0 to 1.")

    column_mappings = payload["column_mappings"]
    if not isinstance(column_mappings, dict) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in column_mappings.items()
    ):
        raise StructureResponseError("Column mappings must be a string dictionary.")
    if not set(column_mappings).issubset(set(original_columns)):
        raise StructureResponseError("Column mappings contain unknown source columns.")
    invalid_targets = sorted(
        set(column_mappings.values()) - ALLOWED_CANONICAL_FIELDS
    )
    if invalid_targets:
        raise StructureResponseError(
            f"Column mappings contain invalid canonical fields: {invalid_targets}."
        )

    dimensions = _validate_field_list(payload["dimension_fields"], "dimension_fields")
    metrics = _validate_field_list(payload["metric_fields"], "metric_fields")
    reasoning = payload["reasoning_short"]
    if not isinstance(reasoning, str) or len(reasoning.strip()) > 240:
        raise StructureResponseError("reasoning_short must be a short string.")
    human_review = payload["requires_human_review"]
    if not isinstance(human_review, bool):
        raise StructureResponseError("requires_human_review must be boolean.")

    return ValidatedStructureSuggestion(
        suggested_table_type=suggested_type,
        table_confidence=float(confidence),
        column_mappings=dict(column_mappings),
        dimension_fields=dimensions,
        metric_fields=metrics,
        reasoning_short=reasoning.strip(),
        requires_human_review=human_review,
    )


def _deterministic_mapping(table: dict[str, Any]) -> dict[str, str]:
    """Convert serialized deterministic mappings to original-to-normalized form.

    Inputs: one intermediate table.
    Outputs: compact mapping dictionary.
    Assumptions: original names are stable identifiers within the detected table.
    """

    return {
        str(mapping.get("original_name", "")): str(mapping.get("normalized_name", ""))
        for mapping in table.get("column_mappings", [])
    }


def _base_enrichment(table: dict[str, Any]) -> dict[str, Any]:
    """Create fail-safe enrichment fields that preserve deterministic results.

    Inputs: one original table dictionary.
    Outputs: enrichment fields before any validated Ollama suggestion.
    Assumptions: deterministic mappings are always the fallback source of truth.
    """

    return {
        "llm_reviewed": False,
        "llm_suggested_type": None,
        "llm_column_mappings": {},
        "llm_confidence": None,
        "llm_reasoning_short": None,
        "final_table_type": table.get("detected_type", "Unknown"),
        "final_column_mappings": _deterministic_mapping(table),
        "final_confidence": table.get("confidence", 0.0),
        "requires_human_review": False,
    }


def preserve_deterministic_enrichment(model: dict[str, Any]) -> dict[str, Any]:
    """Add final deterministic structure fields without calling an LLM.

    Inputs: serialized intermediate model.
    Outputs: enriched copy with deterministic final mappings and no LLM review.
    Assumptions: caller already verified no low-confidence structure needs review.
    """

    enriched = copy.deepcopy(model)
    for table in enriched.get("tables", []):
        table.update(_base_enrichment(table))
    enriched["enrichment"] = {
        "stage": "ollama_structure_fallback",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_available": None,
        "items_detected": 0,
        "acceptance_threshold": LLM_ACCEPTANCE_THRESHOLD,
        "deterministic_fields_preserved": True,
        "skipped_reason": "deterministic_structure_high_confidence",
    }
    return enriched


def enrich_intermediate_model(
    model: dict[str, Any],
    interpreter: StructureInterpreter,
    *,
    table_threshold: float = TABLE_CONFIDENCE_THRESHOLD,
    column_threshold: float = COLUMN_CONFIDENCE_THRESHOLD,
) -> tuple[dict[str, Any], FallbackSummary]:
    """Apply validated Ollama suggestions only to low-confidence structure.

    Inputs: serialized model, interpreter/client, and confidence thresholds.
    Outputs: enriched deep copy and execution summary.
    Assumptions: the original model object and deterministic fields remain unchanged.
    """

    enriched = copy.deepcopy(model)
    items = detect_low_confidence_items(
        enriched,
        table_threshold=table_threshold,
        column_threshold=column_threshold,
    )
    items_by_id = {item.table_id: item for item in items}
    available = interpreter.is_available()
    accepted = 0
    rejected = 0

    for table in enriched.get("tables", []):
        table.update(_base_enrichment(table))
        item = items_by_id.get(str(table.get("table_id", "")))
        if item is None:
            continue

        # Unavailable Ollama leaves the deterministic result intact and makes the
        # unresolved item visible instead of silently pretending it was reviewed.
        table["requires_human_review"] = True
        if not available:
            continue

        table["llm_reviewed"] = True
        try:
            response_text = interpreter.generate(build_structure_prompt(table, item))
            suggestion = parse_and_validate_structure_response(
                response_text,
                original_columns=list(table.get("original_columns", [])),
            )
        except (OllamaError, StructureResponseError):
            rejected += 1
            continue

        table["llm_suggested_type"] = suggestion.suggested_table_type
        table["llm_column_mappings"] = suggestion.column_mappings
        table["llm_confidence"] = suggestion.table_confidence
        table["llm_reasoning_short"] = suggestion.reasoning_short

        deterministic_type = table.get("detected_type", "Unknown")
        deterministic_confidence = float(table.get("confidence", 0.0))
        conflicts_with_strong_type = (
            deterministic_type != "Unknown"
            and deterministic_confidence >= TABLE_CONFIDENCE_THRESHOLD
            and suggestion.suggested_table_type != deterministic_type
        )
        is_accepted = (
            suggestion.table_confidence >= LLM_ACCEPTANCE_THRESHOLD
            and not suggestion.requires_human_review
            and not conflicts_with_strong_type
            and not (
                deterministic_type == "Unknown"
                and suggestion.suggested_table_type == "Unknown"
            )
        )
        if not is_accepted:
            rejected += 1
            continue

        # A validated suggestion may resolve only uncertain deterministic fields.
        # High-confidence Python mappings remain locked even when the LLM differs.
        uncertain = set(item.uncertain_columns)
        final_mappings = dict(table["final_column_mappings"])
        for original_name, canonical_name in suggestion.column_mappings.items():
            if original_name in uncertain and canonical_name != "unknown":
                final_mappings[original_name] = canonical_name
        table["final_column_mappings"] = final_mappings

        if (
            deterministic_type == "Unknown"
            or deterministic_confidence < TABLE_CONFIDENCE_THRESHOLD
        ):
            table["final_table_type"] = suggestion.suggested_table_type
            table["final_confidence"] = suggestion.table_confidence
        table["requires_human_review"] = False
        accepted += 1

    enriched["enrichment"] = {
        "stage": "ollama_structure_fallback",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_available": available,
        "items_detected": len(items),
        "acceptance_threshold": LLM_ACCEPTANCE_THRESHOLD,
        "deterministic_fields_preserved": True,
    }
    human_review_count = sum(
        bool(table.get("requires_human_review"))
        for table in enriched.get("tables", [])
    )
    summary = FallbackSummary(
        ollama_available=available,
        items_reviewed=len(items) if available else 0,
        accepted=accepted,
        rejected=rejected,
        requiring_human_review=human_review_count,
        uncertain_columns=sum(len(item.uncertain_columns) for item in items),
    )
    return enriched, summary


def load_intermediate_model_json(model_path: str | Path) -> dict[str, Any]:
    """Load the existing serialized intermediate financial model.

    Inputs: path to financial_document_model.json.
    Outputs: decoded model dictionary.
    Assumptions: Step 2 produced a JSON object with a tables list.
    """

    path = Path(model_path)
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load intermediate model {path}: {exc}") from exc
    if not isinstance(model, dict) or not isinstance(model.get("tables"), list):
        raise ValueError("Intermediate model must be a JSON object with a tables list.")
    return model


def save_enriched_model(
    model: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save the enriched model without changing the original Step 2 artifact.

    Inputs: enriched model dictionary and destination path.
    Outputs: resolved path written as readable UTF-8 JSON.
    Assumptions: parent directories may be created safely.
    """

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
