"""Deterministic fact registry for placeholder-based reasoning."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from typing import Any


PLACEHOLDER_PATTERN = re.compile(r"\{\{(FACT_\d{3,})\}\}")
MALFORMED_PLACEHOLDER_PATTERN = re.compile(r"\{\{[^}]*\}\}|\{FACT_\d{3,}\}|\{\{FACT_[^}]+\}\}")


@dataclass(frozen=True)
class RegisteredFact:
    """One deterministic fact exposed to Ollama through a placeholder.

    Inputs: stable fact ID, evidence link, raw/display values, semantic metadata,
    and source metadata.
    Outputs: serializable fact entry for prompts, validation and substitution.
    Assumptions: values come only from deterministic processed artifacts.
    """

    fact_id: str
    placeholder: str
    evidence_ids: tuple[str, ...]
    raw_value: Any
    display_value: str
    value_type: str
    metric_name: str
    unit: str
    period: str
    entity: str
    source_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize a registered fact.

        Inputs: this fact.
        Outputs: JSON-compatible dictionary.
        Assumptions: source metadata is already JSON-compatible.
        """

        data = asdict(self)
        data["evidence_ids"] = list(self.evidence_ids)
        return data


@dataclass(frozen=True)
class PlaceholderValidationResult:
    """Result of placeholder validation.

    Inputs: validity flag, errors, placeholder usage count and audit details.
    Outputs: immutable validation object.
    Assumptions: callers fail closed when invalid.
    """

    is_valid: bool
    errors: tuple[str, ...]
    placeholder_count: int
    audit: tuple[dict[str, Any], ...]


class FactRegistry:
    """Registry mapping deterministic facts to stable placeholders."""

    def __init__(self, facts: tuple[RegisteredFact, ...]) -> None:
        """Create a fact registry.

        Inputs: registered facts.
        Outputs: initialized registry with placeholder/evidence indexes.
        Assumptions: fact IDs are unique within one run.
        """

        self.facts = facts
        self.by_fact_id = {fact.fact_id: fact for fact in facts}
        self.by_placeholder = {fact.placeholder: fact for fact in facts}

    @classmethod
    def from_evidence_ledger(cls, evidence_ledger: dict[str, Any]) -> "FactRegistry":
        """Create a registry from the validated evidence ledger.

        Inputs: evidence ledger facts.
        Outputs: FactRegistry with stable FACT_### IDs sorted by evidence ID.
        Assumptions: no new analytical facts are derived here.
        """

        registered: list[RegisteredFact] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        source_facts = [
            fact
            for fact in evidence_ledger.get("facts", [])
            if isinstance(fact, dict) and fact.get("evidence_id")
        ]
        source_facts.sort(key=lambda item: str(item.get("evidence_id")))
        supplemental: list[dict[str, Any]] = []
        for fact in source_facts:
            # Periods and entities are deterministic facts too.  Registering
            # them lets Ollama refer to them through placeholders instead of
            # spelling report-specific literals directly in prose.
            for supplemental_field, supplemental_type in (("period", "period"), ("entity", "entity")):
                supplemental_value = str(fact.get(supplemental_field) or "").strip()
                if supplemental_value:
                    supplemental.append(
                        {
                            "evidence_id": fact["evidence_id"],
                            "metric": supplemental_field,
                            "display_value": supplemental_value,
                            "raw_value": supplemental_value,
                            "unit": supplemental_type,
                            "period": fact.get("period"),
                            "entity": fact.get("entity"),
                            "source_reference": fact.get("source_reference"),
                            "category": fact.get("category"),
                            "claim": fact.get("claim"),
                        }
                    )
        source_facts = [*source_facts, *supplemental]
        source_facts.sort(
            key=lambda item: (
                str(item.get("evidence_id")),
                str(item.get("metric") or item.get("field") or ""),
                str(item.get("display_value") or item.get("raw_value") or ""),
            )
        )
        for fact in source_facts:
            display_value = str(fact.get("display_value") or "").strip()
            raw_value = fact.get("raw_value")
            if display_value in {"", "None"} and raw_value in {None, ""}:
                continue
            evidence_id = str(fact["evidence_id"])
            key = (
                evidence_id,
                str(fact.get("metric") or fact.get("field") or ""),
                display_value if display_value else str(raw_value),
                str(fact.get("period") or ""),
                str(fact.get("entity") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fact_id = f"FACT_{len(registered) + 1:03d}"
            registered.append(
                RegisteredFact(
                    fact_id=fact_id,
                    placeholder=f"{{{{{fact_id}}}}}",
                    evidence_ids=(evidence_id,),
                    raw_value=raw_value,
                    display_value=display_value if display_value else str(raw_value),
                    value_type=_value_type(fact),
                    metric_name=str(fact.get("metric") or fact.get("field") or ""),
                    unit=str(fact.get("unit") or ""),
                    period=str(fact.get("period") or ""),
                    entity=str(fact.get("entity") or ""),
                    source_metadata={
                        "source_reference": fact.get("source_reference"),
                        "category": fact.get("category"),
                        "claim": fact.get("claim"),
                    },
                )
            )
        return cls(tuple(registered))

    def prompt_facts(self) -> list[dict[str, Any]]:
        """Return placeholder facts for Ollama prompts.

        Inputs: registry.
        Outputs: compact fact descriptors with placeholders and numeric metadata.
        Assumptions: display values are intentionally omitted from model context.
        """

        values: list[dict[str, Any]] = []
        for fact in self.facts:
            values.append(
                {
                    "fact_id": fact.fact_id,
                    "placeholder": fact.placeholder,
                    "semantic_label": _semantic_label(fact),
                    "numeric_value": fact.raw_value if isinstance(fact.raw_value, (int, float)) else None,
                    "value_type": fact.value_type,
                    "metric_name": fact.metric_name,
                    "unit": fact.unit,
                    "period": fact.period,
                    "entity": fact.entity,
                    "evidence_ids": list(fact.evidence_ids),
                }
            )
        return values

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry.

        Inputs: registry.
        Outputs: JSON-compatible registry snapshot.
        Assumptions: used for audit/checkpoint artifacts.
        """

        return {
            "fact_count": len(self.facts),
            "facts": [fact.to_dict() for fact in self.facts],
        }

    def substitute(self, payload: Any) -> tuple[Any, list[dict[str, Any]]]:
        """Replace placeholders recursively with deterministic display values.

        Inputs: JSON-compatible payload.
        Outputs: substituted payload plus substitution audit.
        Assumptions: validation already confirmed every placeholder exists.
        """

        audit: list[dict[str, Any]] = []

        def replace(value: Any) -> Any:
            """Recursively replace placeholders in one value."""

            if isinstance(value, str):
                def repl(match: re.Match[str]) -> str:
                    fact = self.by_fact_id[match.group(1)]
                    audit.append(
                        {
                            "fact_id": fact.fact_id,
                            "placeholder": fact.placeholder,
                            "display_value": fact.display_value,
                        }
                    )
                    return fact.display_value

                return PLACEHOLDER_PATTERN.sub(repl, value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        substituted = replace(copy.deepcopy(payload))
        unresolved = find_placeholders(substituted)
        if unresolved:
            raise ValueError(f"Unresolved placeholders after substitution: {sorted(unresolved)}")
        return substituted, audit


def validate_placeholders_in_payload(
    payload: Any,
    registry: FactRegistry,
) -> PlaceholderValidationResult:
    """Validate placeholder usage before deterministic substitution.

    Inputs: model payload and fact registry.
    Outputs: placeholder validation result.
    Assumptions: numeric literals in narrative are rejected unless they are part
    of a placeholder token or an approved confidence value outside prose.
    """

    errors: list[str] = []
    audit: list[dict[str, Any]] = []

    def visit(value: Any, path: str, evidence_ids: tuple[str, ...] = ()) -> None:
        """Walk model payload and validate placeholder-bearing text."""

        if isinstance(value, dict):
            local_evidence = tuple(str(item) for item in value.get("evidence_ids", []) if isinstance(item, str))
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else key, local_evidence or evidence_ids)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", evidence_ids)
            return
        if not isinstance(value, str):
            return
        malformed = [
            match.group(0)
            for match in MALFORMED_PLACEHOLDER_PATTERN.finditer(value)
            if not PLACEHOLDER_PATTERN.fullmatch(match.group(0))
        ]
        for token in malformed:
            errors.append(f"{path} contains malformed placeholder: {token}")
        for fact_id in PLACEHOLDER_PATTERN.findall(value):
            fact = registry.by_fact_id.get(fact_id)
            if fact is None:
                errors.append(f"{path} contains unknown placeholder: {fact_id}")
                continue
            if evidence_ids and not set(fact.evidence_ids).issubset(set(evidence_ids)):
                errors.append(
                    f"{path} uses {fact.placeholder} without supporting evidence_ids {list(fact.evidence_ids)}"
                )
            audit.append(
                {
                    "path": path,
                    "fact_id": fact.fact_id,
                    "placeholder": fact.placeholder,
                    "evidence_ids": list(fact.evidence_ids),
                }
            )
        if _is_narrative_path(path):
            stripped = PLACEHOLDER_PATTERN.sub("", value)
            if re.search(r"-?\d+(?:[\.,]\d+)?%?", stripped):
                errors.append(f"{path} contains unsupported numeric literal")
            for fact in registry.facts:
                if fact.value_type not in {"entity", "period"}:
                    continue
                literal = str(fact.display_value or "").strip()
                # Long deterministic literals must be referenced by placeholder
                # while the LLM is reasoning.  Short words are skipped to avoid
                # false positives on ordinary Spanish prose.
                if len(literal) >= 3 and literal in stripped:
                    errors.append(
                        f"{path} contains deterministic literal for {fact.fact_id} instead of {fact.placeholder}"
                    )

    visit(payload, "")
    return PlaceholderValidationResult(
        is_valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        placeholder_count=len(audit),
        audit=tuple(audit),
    )


def find_placeholders(payload: Any) -> set[str]:
    """Find placeholders recursively in a payload.

    Inputs: JSON-compatible payload.
    Outputs: set of placeholder strings.
    Assumptions: useful for post-substitution fail-closed checks.
    """

    found: set[str] = set()
    if isinstance(payload, str):
        found.update(match.group(0) for match in PLACEHOLDER_PATTERN.finditer(payload))
    elif isinstance(payload, list):
        for item in payload:
            found.update(find_placeholders(item))
    elif isinstance(payload, dict):
        for item in payload.values():
            found.update(find_placeholders(item))
    return found


def _value_type(fact: dict[str, Any]) -> str:
    """Infer a display value type from a ledger fact.

    Inputs: ledger fact.
    Outputs: value type string.
    Assumptions: this is metadata for prompts, not new calculation logic.
    """

    unit = str(fact.get("unit") or "").casefold()
    raw = fact.get("raw_value")
    if unit in {"period", "date"}:
        return "period"
    if unit == "entity":
        return "entity"
    if unit in {"usd", "pen"}:
        return "currency"
    if unit == "ratio" or str(fact.get("display_value", "")).endswith("%"):
        return "percentage"
    if unit == "count":
        return "integer"
    if isinstance(raw, (int, float)):
        return "decimal"
    if fact.get("entity"):
        return "entity"
    if fact.get("period"):
        return "period"
    return "text"


def _is_narrative_path(path: str) -> bool:
    """Return whether a JSON path contains user-facing narrative prose.

    Inputs: recursive JSON path string.
    Outputs: True for prose fields that must use placeholders for literals.
    Assumptions: structural fields such as evidence IDs, source paths, and
    confidence values are validated elsewhere and should not be scanned as prose.
    """

    if "narrative_evidence" in path:
        return False
    last = path.split(".")[-1]
    if "[" in last:
        last = last.split("[")[0]
    narrative_names = {
        "text",
        "executive_summary",
        "financial_health_analysis",
        "kpi_analysis",
        "department_analysis",
        "anomaly_analysis",
        "recommendation_follow_up_analysis",
        "longitudinal_risk_analysis",
        "historical_summary",
        "historical_trend_analysis",
        "reasoning_summary",
        "rationale",
        "expected_impact",
        "action",
        "missing_information",
        "finding",
        "cause",
        "priority",
    }
    return last in narrative_names


def _semantic_label(fact: RegisteredFact) -> str:
    """Build a compact semantic label for a registered fact.

    Inputs: registered fact.
    Outputs: label string without deterministic display value.
    Assumptions: labels describe meaning only and are not analytical prose.
    """

    parts = [fact.period, fact.entity, fact.metric_name]
    return " ".join(part for part in parts if part).strip() or fact.fact_id
