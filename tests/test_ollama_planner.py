"""Tests for validated Ollama planning and deterministic fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass

from finance_agent.agent.ollama_planner import (
    build_ollama_planner_prompt,
    create_ollama_investigation_plan,
)
from finance_agent.agent.planner_models import (
    EvidenceRequest,
    InvestigationPlan,
    InvestigationTask,
    PriorityLevel,
)
from finance_agent.agent.planner_validation import (
    MAX_PLAN_STEPS,
    validate_ollama_plan_response,
)


@dataclass
class FakeOllamaClient:
    """Configurable test double for the reusable Ollama client."""

    available: bool
    response: str = ""
    generate_calls: int = 0
    last_prompt: str = ""

    def is_available(self) -> bool:
        """Return configured local-service availability.

        Inputs: fixture state.
        Outputs: configured boolean.
        Assumptions: no network request occurs.
        """

        return self.available

    def generate(self, prompt: str) -> str:
        """Record the prompt and return configured model output.

        Inputs: planner prompt.
        Outputs: configured response.
        Assumptions: tests control every response byte.
        """

        self.generate_calls += 1
        self.last_prompt = prompt
        return self.response


def _baseline_plan() -> InvestigationPlan:
    """Build a one-task deterministic fallback fixture.

    Inputs: none.
    Outputs: trusted InvestigationPlan.
    Assumptions: the first evidence request is the primary fallback call.
    """

    request = EvidenceRequest(
        request_id="INV-2026-001-E01",
        tool_name="get_payroll_history",
        parameters={"department": "all", "months": 6},
        purpose="Review payroll history.",
    )
    task = InvestigationTask(
        task_id="INV-2026-001",
        anomaly_id="ANOM-2026-001",
        priority=PriorityLevel.HIGH,
        priority_score=80,
        question_to_answer="What caused the payroll ratio increase?",
        reason="Payroll exceeded its threshold.",
        required_evidence=(request,),
        suggested_tool="get_payroll_history",
        expected_output="Payroll driver breakdown.",
    )
    return InvestigationPlan(
        plan_id="PLAN-2026",
        report_period="2026",
        period_slug="2026",
        source_files=("finance_summary_2026.json",),
        tasks=(task,),
    )


def _valid_response() -> str:
    """Return one strict, valid mocked Ollama plan.

    Inputs: none.
    Outputs: serialized plan response.
    Assumptions: anomaly ID exists in the test anomaly context.
    """

    return json.dumps(
        {
            "investigation_steps": [
                {
                    "step_id": "STEP-001",
                    "anomaly_id": "ANOM-2026-001",
                    "priority": "high",
                    "question": "Which payroll components drove the ratio increase?",
                    "tool_name": "get_payroll_history",
                    "arguments": {"department": "all", "months": 6},
                    "reasoning": "Payroll breached its limit and needs component history.",
                    "expected_output": "Six-month payroll component trend.",
                }
            ]
        }
    )


def _finance_document() -> dict[str, object]:
    """Build compact finance/KPI input with an intentionally unused detail.

    Inputs: none.
    Outputs: Step 3-like finance document.
    Assumptions: prompt compression should omit arbitrary full-report detail.
    """

    return {
        "report_period": "2026",
        "finance_summary": {
            "total_revenue": 2_000_000,
            "payroll_total": 1_000_000,
            "payroll_percentage_of_revenue": 0.50,
            "unused_full_report_detail": "MUST_NOT_APPEAR",
        },
        "kpi_summary": [
            {
                "metric": "payroll_percentage_of_revenue",
                "value": 0.50,
                "unit": "ratio",
                "availability": "available",
                "source": "full source detail omitted",
            }
        ],
        "calculation_warnings": [],
    }


def _anomaly_report() -> dict[str, object]:
    """Build one-anomaly Step 4 input fixture.

    Inputs: none.
    Outputs: compact anomaly report.
    Assumptions: anomaly ID is the allowed source ID for mocked plans.
    """

    return {
        "report_period": "2026",
        "total_anomalies": 1,
        "anomalies_by_severity": {"high": 1},
        "thresholds": {"payroll_percent_max": 42},
        "anomalies": [
            {
                "anomaly_id": "ANOM-2026-001",
                "title": "Payroll exceeds revenue threshold",
                "metric": "payroll_percentage_of_revenue",
                "observed_value": 50,
                "threshold_value": 42,
                "severity": "high",
                "period": "2026",
                "evidence": "Payroll ratio is 50%.",
                "rule_id": "PAYROLL_RATIO_MAX",
            }
        ],
    }


def _run(client: FakeOllamaClient):
    """Run the Step 7 planner with compact standard fixtures.

    Inputs: mocked Ollama client.
    Outputs: OllamaPlannerResult.
    Assumptions: enriched model has no unresolved tables.
    """

    return create_ollama_investigation_plan(
        client=client,
        finance_document=_finance_document(),
        anomaly_report=_anomaly_report(),
        risk_summary={"top_risks": []},
        enriched_model={"tables": []},
        baseline_plan=_baseline_plan(),
        period_slug="2026",
    )


def test_valid_mocked_ollama_plan_is_accepted_and_queued() -> None:
    """Verify a valid model response becomes the primary pending queue."""

    client = FakeOllamaClient(True, _valid_response())

    result = _run(client)

    assert result.ollama_plan_accepted is True
    assert result.fallback_used is False
    assert result.plan_document["planner_source"] == "ollama"
    assert result.plan_document["validation_status"] == "accepted"
    assert result.execution_queue["tools_executed"] is False
    assert result.execution_queue["items"][0]["status"] == "queued"
    assert client.generate_calls == 1


def test_unavailable_ollama_uses_deterministic_fallback() -> None:
    """Verify unavailable Ollama skips generation and preserves baseline planning."""

    client = FakeOllamaClient(False)

    result = _run(client)

    assert client.generate_calls == 0
    assert result.fallback_used is True
    assert result.plan_document["validation_status"] == "unavailable"
    assert result.plan_document["investigation_steps"][0]["step_id"].startswith(
        "FALLBACK-"
    )
    assert result.execution_queue["planner_source"] == "deterministic_fallback"


def test_invalid_tool_rejects_ollama_plan_and_uses_fallback() -> None:
    """Verify tools outside the interface allowlist trigger whole-plan fallback."""

    payload = json.loads(_valid_response())
    payload["investigation_steps"][0]["tool_name"] = "send_email"
    client = FakeOllamaClient(True, json.dumps(payload))

    result = _run(client)

    assert result.ollama_plan_accepted is False
    assert result.fallback_used is True
    assert result.plan_document["validation_status"] == "rejected"
    assert any("tool_name is not allowed" in error for error in result.validation_errors)


def test_invalid_argument_range_is_rejected() -> None:
    """Verify tool argument ranges are enforced before queue creation."""

    payload = json.loads(_valid_response())
    payload["investigation_steps"][0]["arguments"]["months"] = 60

    validation = validate_ollama_plan_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("months must be" in error for error in validation.errors)


def test_invalid_json_is_rejected() -> None:
    """Verify prose or malformed output cannot become an execution queue."""

    validation = validate_ollama_plan_response("Here is the plan: []")

    assert validation.is_valid is False
    assert validation.errors == ("response is not strict JSON",)


def test_long_reasoning_is_safely_repaired() -> None:
    """Verify oversized prose is capped without changing tool semantics."""

    payload = json.loads(_valid_response())
    payload["investigation_steps"][0]["reasoning"] = "R" * 900

    validation = validate_ollama_plan_response(json.dumps(payload))

    assert validation.is_valid is True
    assert validation.repaired_text_fields == 1
    assert len(validation.steps[0]["reasoning"]) == 500
    assert validation.steps[0]["reasoning"].endswith("â€¦")
    assert validation.steps[0]["arguments"] == {
        "department": "all",
        "months": 6,
    }


def test_equivalent_tool_calls_are_safely_deduplicated() -> None:
    """Verify equivalent calls merge while retaining urgency and intent."""

    payload = json.loads(_valid_response())
    duplicate = dict(payload["investigation_steps"][0])
    duplicate["step_id"] = "STEP-002"
    duplicate["anomaly_id"] = "ANOM-2026-002"
    duplicate["priority"] = "critical"
    duplicate["question"] = "A differently worded duplicate question?"
    duplicate["reasoning"] = "The same payroll history supports another anomaly."
    duplicate["expected_output"] = "Shared payroll evidence for both anomalies."
    payload["investigation_steps"].append(duplicate)

    validation = validate_ollama_plan_response(json.dumps(payload))

    assert validation.is_valid is True
    assert validation.deduplicated_steps == 1
    assert len(validation.steps) == 1
    assert validation.steps[0]["priority"] == "critical"
    assert validation.steps[0]["anomaly_id"] is None
    assert "differently worded" in validation.steps[0]["question"]


def test_long_merged_duplicate_text_is_repaired() -> None:
    """Verify safe duplicate intent is capped instead of forcing fallback."""

    payload = json.loads(_valid_response())
    payload["investigation_steps"][0]["question"] = "A" * 240
    duplicate = dict(payload["investigation_steps"][0])
    duplicate["step_id"] = "STEP-002"
    duplicate["question"] = "B" * 240
    payload["investigation_steps"].append(duplicate)

    validation = validate_ollama_plan_response(json.dumps(payload))

    assert validation.is_valid is True
    assert validation.deduplicated_steps == 1
    assert validation.repaired_text_fields >= 1
    assert len(validation.steps[0]["question"]) == 320


def test_mocked_ollama_duplicate_calls_are_cleaned_without_fallback() -> None:
    """Verify safe deduplication keeps the Ollama plan as the primary plan."""

    payload = json.loads(_valid_response())
    duplicate = dict(payload["investigation_steps"][0])
    duplicate["step_id"] = "STEP-002"
    duplicate["question"] = "What historical payroll trend confirms the issue?"
    payload["investigation_steps"].append(duplicate)
    client = FakeOllamaClient(True, json.dumps(payload))

    result = _run(client)

    assert result.ollama_plan_accepted is True
    assert result.fallback_used is False
    assert result.plan_document["deduplicated_tool_calls"] == 1
    assert result.execution_queue["total_items"] == 1


def test_conflicting_duplicate_step_ids_are_rejected() -> None:
    """Verify reused step IDs with different calls remain a hard conflict."""

    payload = json.loads(_valid_response())
    conflict = dict(payload["investigation_steps"][0])
    conflict["tool_name"] = "get_full_report"
    conflict["arguments"] = {"period": "2026"}
    payload["investigation_steps"].append(conflict)

    validation = validate_ollama_plan_response(json.dumps(payload))

    assert validation.is_valid is False
    assert any("conflicting duplicate step_id" in error for error in validation.errors)


def test_maximum_plan_size_is_enforced() -> None:
    """Verify oversized model plans are rejected before individual execution."""

    steps = [
        {
            **json.loads(_valid_response())["investigation_steps"][0],
            "step_id": f"STEP-{index:03d}",
            "tool_name": "get_full_report",
            "arguments": {"period": f"2026-{index:02d}"},
        }
        for index in range(1, MAX_PLAN_STEPS + 2)
    ]

    validation = validate_ollama_plan_response(
        json.dumps({"investigation_steps": steps})
    )

    assert validation.is_valid is False
    assert any("cleaned plan exceeds maximum size" in error for error in validation.errors)


def test_prompt_contains_only_compressed_summaries_and_interfaces() -> None:
    """Verify the model prompt omits full-report and table-row payloads."""

    prompt = build_ollama_planner_prompt(
        finance_document=_finance_document(),
        anomaly_report=_anomaly_report(),
        risk_summary={"top_risks": []},
        enriched_model={
            "tables": [
                {
                    "table_id": "annual__unknown__table_01",
                    "source_workbook": "annual_report.xlsx",
                    "sheet": "Unknown",
                    "final_table_type": "Unknown",
                    "final_confidence": 0.3,
                    "requires_human_review": True,
                    "sample_rows": [{"secret": "MUST_NOT_APPEAR"}],
                    "final_column_mappings": {"secret": "unknown"},
                }
            ]
        },
        baseline_plan=_baseline_plan(),
        period_slug="2026",
    )

    assert "available_tool_interfaces" in prompt
    assert "under 180 characters" in prompt
    assert MAX_PLAN_STEPS == 8
    assert "MUST_NOT_APPEAR" not in prompt
    assert "sample_rows" not in prompt
    assert "final_column_mappings" not in prompt
