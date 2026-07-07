"""Strict validation for untrusted Ollama investigation plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


MAX_PLAN_STEPS = 8
MAX_RESPONSE_CHARACTERS = 60_000
TRUNCATION_MARKER = "\u00e2\u20ac\u00a6"
ALLOWED_PRIORITIES = frozenset({"critical", "high", "medium", "low"})
TEXT_FIELD_LIMITS = {
    "question": 320,
    "reasoning": 500,
    "expected_output": 320,
}
PRIORITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

# These are interfaces only. Step 7 gives their signatures to Ollama but does
# not import, implement, or execute any retrieval function.
TOOL_INTERFACES: dict[str, dict[str, Any]] = {
    "get_previous_cycle_memory": {
        "arguments": {},
        "description": "Retrieve the compact analysis memory from the previous cycle.",
    },
    "get_department_history": {
        "arguments": {
            "department": "string",
            "months": "integer 1..24",
        },
        "description": "Retrieve department-level financial history.",
    },
    "get_vendor_history": {
        "arguments": {
            "vendor": "string",
            "months": "integer 1..24",
        },
        "description": "Retrieve payment history for a known vendor.",
    },
    "get_payroll_history": {
        "arguments": {
            "department": "string; use 'all' only for institution-wide analysis",
            "months": "integer 1..24",
        },
        "description": "Retrieve payroll, benefits, overtime, and headcount history.",
    },
    "get_transactions": {
        "arguments": {
            "filters": {
                "allowed_keys": [
                    "department",
                    "vendor",
                    "period",
                    "type",
                    "expense_category",
                    "minimum_amount",
                    "include_aging",
                    "table_id",
                    "limit",
                    "status",
                ]
            }
        },
        "description": "Retrieve processed transactions matching bounded filters.",
    },
    "get_full_report": {
        "arguments": {"period": "string"},
        "description": "Retrieve one processed report for a specific period.",
    },
}

STEP_FIELDS = frozenset(
    {
        "step_id",
        "anomaly_id",
        "priority",
        "question",
        "tool_name",
        "arguments",
        "reasoning",
        "expected_output",
    }
)


@dataclass(frozen=True)
class PlanValidationResult:
    """Result of parsing and validating one untrusted Ollama response."""

    is_valid: bool
    steps: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    deduplicated_steps: int = 0
    repaired_text_fields: int = 0


def _is_int(value: Any) -> bool:
    """Check for a real integer while excluding booleans.

    Inputs: untrusted scalar.
    Outputs: True only for integer values.
    Assumptions: JSON booleans must not pass numeric range validation.
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _valid_string(value: Any, *, maximum: int) -> bool:
    """Validate a bounded non-empty string.

    Inputs: untrusted value and maximum character count.
    Outputs: True when the value is a trimmed bounded string.
    Assumptions: surrounding whitespace carries no planner meaning.
    """

    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value.strip()) <= maximum
    )


def _cap_text(value: str, *, maximum: int) -> tuple[str, bool]:
    """Trim whitespace and cap descriptive text at a safe schema limit.

    Inputs: descriptive model text and maximum output length.
    Outputs: repaired text and whether repair changed the value.
    Assumptions: leading text contains the primary investigation intent.
    """

    stripped = value.strip()
    if len(stripped) <= maximum:
        return stripped, stripped != value
    return stripped[: maximum - len(TRUNCATION_MARKER)].rstrip() + TRUNCATION_MARKER, True


def _repair_descriptive_text(step: Any) -> tuple[Any, int]:
    """Repair bounded prose without modifying semantic planner fields.

    Inputs: untrusted investigation step.
    Outputs: shallow repaired step and changed-field count.
    Assumptions: IDs, priorities, tools, and arguments must never be rewritten.
    """

    if not isinstance(step, dict):
        return step, 0
    repaired = dict(step)
    repair_count = 0
    for field_name, maximum in TEXT_FIELD_LIMITS.items():
        value = repaired.get(field_name)
        if not isinstance(value, str):
            continue
        repaired_value, changed = _cap_text(value, maximum=maximum)
        repaired[field_name] = repaired_value
        repair_count += int(changed)
    return repaired, repair_count


def _validate_exact_arguments(
    arguments: Any,
    expected_keys: set[str],
) -> list[str]:
    """Validate a tool argument object with an exact key set.

    Inputs: untrusted arguments and required keys.
    Outputs: validation error messages.
    Assumptions: optional arguments are represented by separate validators.
    """

    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]
    if set(arguments) != expected_keys:
        return [
            "arguments must contain exactly "
            f"{sorted(expected_keys)}; received {sorted(arguments)}"
        ]
    return []


def _validate_history_arguments(arguments: Any, entity_key: str) -> list[str]:
    """Validate department/vendor/payroll history arguments.

    Inputs: untrusted arguments and required entity key.
    Outputs: validation error messages.
    Assumptions: history windows are limited to 24 months.
    """

    errors = _validate_exact_arguments(arguments, {entity_key, "months"})
    if errors:
        return errors
    if not _valid_string(arguments[entity_key], maximum=120):
        errors.append(f"{entity_key} must be a non-empty string up to 120 characters")
    if not _is_int(arguments["months"]) or not 1 <= arguments["months"] <= 24:
        errors.append("months must be an integer from 1 to 24")
    return errors


def _validate_transactions_arguments(arguments: Any) -> list[str]:
    """Validate bounded processed-transaction filter arguments.

    Inputs: untrusted tool arguments.
    Outputs: validation error messages.
    Assumptions: filters select processed records and never contain raw SQL/code.
    """

    errors = _validate_exact_arguments(arguments, {"filters"})
    if errors:
        return errors
    filters = arguments["filters"]
    if not isinstance(filters, dict) or not filters:
        return ["filters must be a non-empty JSON object"]
    allowed_keys = set(
        TOOL_INTERFACES["get_transactions"]["arguments"]["filters"]["allowed_keys"]
    )
    unknown_keys = set(filters) - allowed_keys
    if unknown_keys:
        errors.append(f"filters contain unsupported keys: {sorted(unknown_keys)}")
    if len(filters) > 8:
        errors.append("filters may contain at most 8 keys")

    string_keys = {
        "department",
        "vendor",
        "period",
        "type",
        "expense_category",
        "table_id",
        "status",
    }
    for key in string_keys & set(filters):
        if not _valid_string(filters[key], maximum=160):
            errors.append(f"filters.{key} must be a bounded non-empty string")
    if "minimum_amount" in filters:
        amount = filters["minimum_amount"]
        if (
            not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not 0 <= amount <= 1_000_000_000
        ):
            errors.append("filters.minimum_amount must be between 0 and 1,000,000,000")
    if "include_aging" in filters and not isinstance(
        filters["include_aging"],
        bool,
    ):
        errors.append("filters.include_aging must be boolean")
    if "limit" in filters and (
        not _is_int(filters["limit"]) or not 1 <= filters["limit"] <= 1000
    ):
        errors.append("filters.limit must be an integer from 1 to 1000")
    return errors


def _validate_full_report_arguments(arguments: Any) -> list[str]:
    """Validate a processed full-report request.

    Inputs: untrusted arguments.
    Outputs: validation error messages.
    Assumptions: period is a short report label, not a filesystem path.
    """

    errors = _validate_exact_arguments(arguments, {"period"})
    if errors:
        return errors
    if not _valid_string(arguments["period"], maximum=60):
        errors.append("period must be a non-empty string up to 60 characters")
    return errors


def _validate_no_arguments(arguments: Any) -> list[str]:
    """Validate a tool interface that accepts no arguments.

    Inputs: untrusted arguments.
    Outputs: validation error messages.
    Assumptions: callers must still provide an empty JSON object.
    """

    return _validate_exact_arguments(arguments, set())


def _validate_department_history_arguments(arguments: Any) -> list[str]:
    """Validate department-history arguments.

    Inputs: untrusted arguments.
    Outputs: validation error messages.
    Assumptions: department is explicit and the window is at most 24 months.
    """

    return _validate_history_arguments(arguments, "department")


def _validate_vendor_history_arguments(arguments: Any) -> list[str]:
    """Validate vendor-history arguments.

    Inputs: untrusted arguments.
    Outputs: validation error messages.
    Assumptions: vendor is explicit and the window is at most 24 months.
    """

    return _validate_history_arguments(arguments, "vendor")


def _validate_payroll_history_arguments(arguments: Any) -> list[str]:
    """Validate payroll-history arguments.

    Inputs: untrusted arguments.
    Outputs: validation error messages.
    Assumptions: institution-wide history uses department='all'.
    """

    return _validate_history_arguments(arguments, "department")


TOOL_ARGUMENT_VALIDATORS: dict[str, Callable[[Any], list[str]]] = {
    "get_previous_cycle_memory": _validate_no_arguments,
    "get_department_history": _validate_department_history_arguments,
    "get_vendor_history": _validate_vendor_history_arguments,
    "get_payroll_history": _validate_payroll_history_arguments,
    "get_transactions": _validate_transactions_arguments,
    "get_full_report": _validate_full_report_arguments,
}


def _validate_step(step: Any, index: int) -> list[str]:
    """Validate one proposed investigation step.

    Inputs: untrusted step and one-based list index.
    Outputs: path-prefixed validation errors.
    Assumptions: exact fields prevent silent schema drift.
    """

    prefix = f"investigation_steps[{index}]"
    if not isinstance(step, dict):
        return [f"{prefix} must be a JSON object"]
    errors: list[str] = []
    if set(step) != STEP_FIELDS:
        errors.append(
            f"{prefix} must contain exactly {sorted(STEP_FIELDS)}"
        )
        return errors

    if not _valid_string(step["step_id"], maximum=64):
        errors.append(f"{prefix}.step_id must be a bounded non-empty string")
    anomaly_id = step["anomaly_id"]
    if anomaly_id is not None and not _valid_string(anomaly_id, maximum=180):
        errors.append(f"{prefix}.anomaly_id must be null or a bounded string")
    if step["priority"] not in ALLOWED_PRIORITIES:
        errors.append(f"{prefix}.priority is not allowed")
    if not _valid_string(
        step["question"],
        maximum=TEXT_FIELD_LIMITS["question"],
    ):
        errors.append(f"{prefix}.question must be a bounded non-empty string")
    if not _valid_string(
        step["reasoning"],
        maximum=TEXT_FIELD_LIMITS["reasoning"],
    ):
        errors.append(f"{prefix}.reasoning must be a bounded non-empty string")
    if not _valid_string(
        step["expected_output"],
        maximum=TEXT_FIELD_LIMITS["expected_output"],
    ):
        errors.append(f"{prefix}.expected_output must be a bounded non-empty string")

    tool_name = step["tool_name"]
    if tool_name not in TOOL_INTERFACES:
        errors.append(f"{prefix}.tool_name is not allowed")
    else:
        tool_errors = TOOL_ARGUMENT_VALIDATORS[tool_name](step["arguments"])
        errors.extend(
            f"{prefix}.{error}"
            for error in tool_errors
        )
    return errors


def _tool_call_signature(step: dict[str, Any]) -> str:
    """Create a stable equivalence signature for one proposed tool call.

    Inputs: individually validated investigation step.
    Outputs: canonical JSON signature for tool name and arguments.
    Assumptions: object key order does not change tool-call meaning.
    """

    return json.dumps(
        {
            "tool_name": step["tool_name"],
            "arguments": step["arguments"],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _merge_text_values(
    steps: list[dict[str, Any]],
    field_name: str,
    *,
    maximum: int,
) -> tuple[str, bool]:
    """Combine unique duplicate-step text and safely cap the result.

    Inputs: equivalent steps, text field name, and maximum merged length.
    Outputs: bounded merged text and whether it required repair.
    Assumptions: highest-priority text appears first.
    """

    unique_values = list(
        dict.fromkeys(str(step[field_name]).strip() for step in steps)
    )
    merged = " | ".join(unique_values)
    return _cap_text(merged, maximum=maximum)


def _merge_equivalent_steps(
    equivalent_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Safely merge steps that request the same tool with the same arguments.

    Inputs: two or more individually valid equivalent tool-call steps.
    Outputs: merged step and number of repaired merged text fields.
    Assumptions: differing priorities are compatible; highest urgency is preserved.
    """

    # Stable max keeps the first model-proposed step when priorities tie.
    representative = max(
        equivalent_steps,
        key=lambda step: PRIORITY_RANK[step["priority"]],
    )
    merged = dict(representative)
    repair_count = 0
    ordered_steps = [
        representative,
        *[step for step in equivalent_steps if step is not representative],
    ]
    for field_name, maximum in TEXT_FIELD_LIMITS.items():
        merged_text, repaired = _merge_text_values(
            ordered_steps,
            field_name,
            maximum=maximum,
        )
        merged[field_name] = merged_text
        repair_count += int(repaired)

    anomaly_ids = {
        step["anomaly_id"]
        for step in equivalent_steps
        if step["anomaly_id"] is not None
    }
    # One retrieval may support several anomalies. Null accurately marks the
    # merged call as cross-cutting without inventing a compound identifier.
    merged["anomaly_id"] = next(iter(anomaly_ids)) if len(anomaly_ids) == 1 else None
    return merged, repair_count


def _deduplicate_tool_calls(
    steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Collapse safely equivalent tool calls and identify merge conflicts.

    Inputs: individually valid steps with unique step IDs.
    Outputs: cleaned steps, removed count, and merged-text repair count.
    Assumptions: first occurrence determines queue position.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    signature_order: list[str] = []
    for step in steps:
        signature = _tool_call_signature(step)
        if signature not in grouped:
            grouped[signature] = []
            signature_order.append(signature)
        grouped[signature].append(step)

    cleaned: list[dict[str, Any]] = []
    merged_repair_count = 0
    for signature in signature_order:
        equivalent_steps = grouped[signature]
        if len(equivalent_steps) == 1:
            cleaned.append(dict(equivalent_steps[0]))
            continue
        merged, repair_count = _merge_equivalent_steps(equivalent_steps)
        cleaned.append(merged)
        merged_repair_count += repair_count
    return cleaned, len(steps) - len(cleaned), merged_repair_count


def validate_ollama_plan_response(
    response_text: str,
    *,
    allowed_source_ids: set[str] | None = None,
) -> PlanValidationResult:
    """Parse and validate a strict Ollama investigation-plan response.

    Inputs: raw model response text and optional allowed anomaly/data source IDs.
    Outputs: validation result containing safe steps or rejection errors.
    Assumptions: surrounding prose/markdown is invalid; equivalent calls may merge.
    """

    if not isinstance(response_text, str):
        return PlanValidationResult(False, (), ("response must be text",))
    if len(response_text) > MAX_RESPONSE_CHARACTERS:
        return PlanValidationResult(
            False,
            (),
            ("response exceeds maximum character count",),
        )
    try:
        payload = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return PlanValidationResult(False, (), ("response is not strict JSON",))
    if not isinstance(payload, dict) or set(payload) != {"investigation_steps"}:
        received_keys = sorted(payload) if isinstance(payload, dict) else []
        return PlanValidationResult(
            False,
            (),
            (
                "response must contain only investigation_steps; "
                f"received keys {received_keys}",
            ),
        )

    raw_steps = payload["investigation_steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        return PlanValidationResult(
            False,
            (),
            ("investigation_steps must be a non-empty list",),
        )

    steps: list[Any] = []
    repaired_text_fields = 0
    for raw_step in raw_steps:
        repaired_step, repair_count = _repair_descriptive_text(raw_step)
        steps.append(repaired_step)
        repaired_text_fields += repair_count

    errors: list[str] = []
    for index, step in enumerate(steps):
        errors.extend(_validate_step(step, index))
        if (
            allowed_source_ids is not None
            and isinstance(step, dict)
            and step.get("anomaly_id") is not None
            and step.get("anomaly_id") not in allowed_source_ids
        ):
            errors.append(
                f"investigation_steps[{index}].anomaly_id is not in supplied context"
            )

    step_ids = [
        step.get("step_id")
        for step in steps
        if isinstance(step, dict)
    ]
    if len(step_ids) != len(set(step_ids)):
        errors.append("conflicting duplicate step_id values must be unique")

    if errors:
        return PlanValidationResult(
            False,
            (),
            tuple(errors),
            repaired_text_fields=repaired_text_fields,
        )

    cleaned_steps, deduplicated_steps, merged_repair_count = _deduplicate_tool_calls(
        [dict(step) for step in steps]
    )
    repaired_text_fields += merged_repair_count
    if len(cleaned_steps) > MAX_PLAN_STEPS:
        return PlanValidationResult(
            False,
            (),
            (f"cleaned plan exceeds maximum size of {MAX_PLAN_STEPS}",),
            deduplicated_steps,
            repaired_text_fields,
        )
    return PlanValidationResult(
        True,
        tuple(cleaned_steps),
        (),
        deduplicated_steps,
        repaired_text_fields,
    )
