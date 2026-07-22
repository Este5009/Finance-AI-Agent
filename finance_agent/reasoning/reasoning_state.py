"""Ephemeral shared working memory for one report reasoning run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from finance_agent.reasoning.reasoning_models import ReasoningStageResult


@dataclass
class ReasoningState:
    """Shared state passed across modular Ollama reasoning stages.

    Inputs: period slug and validated evidence ledger for one report execution.
    Outputs: mutable in-run state that can be serialized for audit/debugging.
    Assumptions: this state is not SQLite memory and is discarded after a run
    unless written as a debugging artifact.
    """

    period_slug: str
    evidence_ledger: dict[str, Any]
    fact_registry: dict[str, Any] | None = None
    validated_claims: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    unresolved_conflicts: list[dict[str, Any]] = field(default_factory=list)
    reasoning_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_placeholder_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    substituted_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    placeholder_validation_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    substitution_audit_log: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    normalization_log: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    evidence_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    cross_stage_dependencies: list[dict[str, Any]] = field(default_factory=list)
    stage_results: list[ReasoningStageResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Index ledger facts by evidence ID after initialization.

        Inputs: initialized dataclass fields.
        Outputs: populates evidence_references.
        Assumptions: ledger facts are compact JSON dictionaries.
        """

        for fact in self.evidence_ledger.get("facts", []):
            if isinstance(fact, dict) and fact.get("evidence_id"):
                self.evidence_references[str(fact["evidence_id"])] = fact

    def add_stage_result(self, result: ReasoningStageResult) -> None:
        """Append a validated stage result and propagate structured outputs.

        Inputs: one stage result.
        Outputs: updates stage history plus claims/risks/opportunities/questions.
        Assumptions: only accepted payloads should influence later prompts.
        """

        self.stage_results.append(result)
        raw_payload = result.payload.get("_raw_placeholder_payload", result.payload)
        substituted_payload = result.payload.get("_substituted_payload", result.payload)
        self.raw_placeholder_outputs[result.stage_id] = raw_payload
        self.substituted_outputs[result.stage_id] = substituted_payload
        self.placeholder_validation_results[result.stage_id] = result.payload.get("_placeholder_validation", {})
        self.substitution_audit_log[result.stage_id] = list(result.payload.get("_substitution_audit", []))
        self.normalization_log[result.stage_id] = list(result.payload.get("_schema_normalizations", []))
        # Earlier stages are passed forward with placeholders intact.  The final
        # synthesis is stored substituted so existing report generation receives
        # deterministic Spanish prose with Python-owned values.
        self.reasoning_outputs[result.stage_id] = (
            substituted_payload if result.stage_id == "strategic_synthesis" else raw_payload
        )
        if not result.accepted:
            self.unresolved_conflicts.append(
                {
                    "stage_id": result.stage_id,
                    "reason": "reasoning_stage_rejected",
                    "errors": list(result.validation_errors),
                }
            )
            return
        for key in ("claims", "validated_financial_claims", "validated_historical_claims"):
            self._extend_structured_items(self.validated_claims, raw_payload.get(key), result.stage_id)
        for key in ("risks", "identified_financial_risks", "persistent_risks"):
            self._extend_structured_items(self.risks, raw_payload.get(key), result.stage_id)
        self._extend_structured_items(
            self.opportunities,
            raw_payload.get("opportunities", raw_payload.get("financial_opportunities")),
            result.stage_id,
        )
        self._extend_structured_items(
            self.open_questions,
            raw_payload.get("open_questions"),
            result.stage_id,
        )

    def to_prompt_context(self) -> dict[str, Any]:
        """Return compact validated reasoning for downstream stages.

        Inputs: current state.
        Outputs: bounded dictionary with accepted claims/risks/questions and
        stage dependencies, not the full evidence ledger.
        Assumptions: strategic synthesis should reason from prior validated
        outputs rather than re-reading all raw facts.
        """

        return {
            "period_slug": self.period_slug,
            "validated_claims": self.validated_claims[:12],
            "risks": self.risks[:10],
            "opportunities": self.opportunities[:8],
            "open_questions": self.open_questions[:8],
            "unresolved_conflicts": self.unresolved_conflicts[:8],
            "accepted_stage_ids": [
                stage.stage_id for stage in self.stage_results if stage.accepted
            ],
            "cross_stage_dependencies": self.cross_stage_dependencies[:8],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reasoning state.

        Inputs: current state.
        Outputs: JSON-compatible audit snapshot.
        Assumptions: this dump is for debugging/reproducibility, not final prose.
        """

        return {
            "period_slug": self.period_slug,
            "evidence_ledger_summary": {
                "fact_count": len(self.evidence_ledger.get("facts", [])),
                "approved_number_count": len(self.evidence_ledger.get("approved_numbers", [])),
                "approved_period_count": len(self.evidence_ledger.get("approved_periods", [])),
                "approved_entity_count": len(self.evidence_ledger.get("approved_entities", [])),
            },
            "validated_claims": self.validated_claims,
            "risks": self.risks,
            "opportunities": self.opportunities,
            "open_questions": self.open_questions,
            "unresolved_conflicts": self.unresolved_conflicts,
            "reasoning_outputs": self.reasoning_outputs,
            "fact_registry": self.fact_registry or {},
            "raw_placeholder_outputs": self.raw_placeholder_outputs,
            "substituted_outputs": self.substituted_outputs,
            "placeholder_validation_results": self.placeholder_validation_results,
            "substitution_audit_log": self.substitution_audit_log,
            "normalization_log": self.normalization_log,
            "evidence_references": self.evidence_references,
            "cross_stage_dependencies": self.cross_stage_dependencies,
            "stage_results": [stage.to_dict() for stage in self.stage_results],
        }

    def _extend_structured_items(
        self,
        target: list[dict[str, Any]],
        items: Any,
        stage_id: str,
    ) -> None:
        """Copy structured model outputs into one state collection.

        Inputs: target collection, model output items, and producing stage ID.
        Outputs: target collection updated with stage provenance.
        Assumptions: non-dictionary items are ignored rather than failing state
        propagation after validation.
        """

        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied.setdefault("stage_id", stage_id)
            target.append(copied)
