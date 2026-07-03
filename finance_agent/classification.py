"""Confidence-based classification of normalized financial tables."""

from __future__ import annotations

from dataclasses import dataclass

from finance_agent.models import DetectedRawTable
from finance_agent.normalization import NormalizedTableData
from finance_agent.schema import clean_column_name


TABLE_CONFIDENCE_THRESHOLD = 0.65


@dataclass(frozen=True)
class ClassificationResult:
    """Detected financial table type and confidence evidence."""

    detected_type: str
    confidence: float
    requires_interpretation: bool
    matched_signals: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationRule:
    """Internal keyword rule for one supported table type."""

    table_type: str
    sheet_aliases: frozenset[str]
    column_signals: frozenset[str]


CLASSIFICATION_RULES = (
    ClassificationRule(
        "Revenue",
        frozenset({"revenue", "revenues", "ingresos", "income"}),
        frozenset({"revenue", "revenue_category", "budget_revenue", "actual_revenue"}),
    ),
    ClassificationRule(
        "Expenses",
        frozenset({"expenses", "expense", "gastos", "egresos"}),
        frozenset({"expenses", "expense_category", "budget_expense", "actual_expense"}),
    ),
    ClassificationRule(
        "Budget_vs_Actual",
        frozenset(
            {
                "budget_vs_actual",
                "budget_actual",
                "presupuesto_vs_ejecutado",
                "presupuesto_ejecutado",
            }
        ),
        frozenset({"budget", "actual", "variance", "variance_pct"}),
    ),
    ClassificationRule(
        "Payroll",
        frozenset({"payroll", "nomina", "planilla"}),
        frozenset({"payroll", "salary", "benefits", "overtime", "headcount"}),
    ),
    ClassificationRule(
        "Student_Payments",
        frozenset(
            {
                "student_payments",
                "student_payment",
                "pagos_estudiantes",
                "cuentas_estudiantes",
            }
        ),
        frozenset({"student", "invoice", "amount_due", "amount_paid", "outstanding"}),
    ),
    ClassificationRule(
        "Scholarships",
        frozenset({"scholarships", "scholarship", "becas", "beca"}),
        frozenset({"scholarship", "allocated", "awarded", "recipients", "remaining"}),
    ),
    ClassificationRule(
        "Cash_Flow",
        frozenset({"cash_flow", "cashflow", "flujo_caja", "flujo_efectivo"}),
        frozenset({"cash", "cash_flow", "beginning_cash", "ending_cash", "inflows", "outflows"}),
    ),
    ClassificationRule(
        "Vendor_Payments",
        frozenset(
            {
                "vendor_payments",
                "vendor_payment",
                "pagos_proveedores",
                "cuentas_proveedores",
            }
        ),
        frozenset({"vendor", "invoice", "amount"}),
    ),
    ClassificationRule(
        "Department_Summary",
        frozenset(
            {
                "department_summary",
                "resumen_departamento",
                "resumen_departamental",
            }
        ),
        frozenset({"department", "revenue", "expenses", "variance", "status"}),
    ),
    ClassificationRule(
        "Executive_Summary",
        frozenset({"executive_summary", "resumen_ejecutivo"}),
        frozenset({"metric", "actual", "goal", "budget", "status"}),
    ),
)


def _semantic_tokens(columns: list[str]) -> set[str]:
    """Expand normalized columns into full names and component tokens.

    Inputs: normalized column names.
    Outputs: searchable semantic signals.
    Assumptions: underscores separate meaningful schema tokens.
    """

    tokens = set(columns)
    for column in columns:
        tokens.update(part for part in column.split("_") if part)
    return tokens


def _sheet_matches(sheet_text: str, aliases: frozenset[str]) -> bool:
    """Check whether normalized sheet/title text contains a type alias.

    Inputs: normalized contextual text and known aliases.
    Outputs: True when an alias occurs as the full text or a contained phrase.
    Assumptions: sheet/title labels are stronger evidence than isolated tokens.
    """

    return any(
        sheet_text == alias
        or sheet_text.startswith(f"{alias}_")
        or sheet_text.endswith(f"_{alias}")
        or f"_{alias}_" in sheet_text
        for alias in aliases
    )


def classify_table(
    raw_table: DetectedRawTable,
    normalized_table: NormalizedTableData,
) -> ClassificationResult:
    """Classify one table using sheet, title, and normalized-column evidence.

    Inputs: detected raw table and normalized table data.
    Outputs: financial type, confidence, signals, and future-interpretation flag.
    Assumptions: ambiguous or weak deterministic evidence must remain Unknown.
    """

    context = clean_column_name(
        " ".join(
            value
            for value in [raw_table.sheet_name, raw_table.title or ""]
            if value
        )
    )
    semantic_tokens = _semantic_tokens(normalized_table.normalized_columns)
    scored_rules: list[tuple[float, ClassificationRule, list[str], bool]] = []

    for rule in CLASSIFICATION_RULES:
        sheet_match = _sheet_matches(context, rule.sheet_aliases)
        matched_columns = sorted(rule.column_signals & semantic_tokens)
        coverage = len(matched_columns) / max(1, len(rule.column_signals))
        # Column coverage supports generic sheet names; contextual labels then
        # strengthen classification without becoming a hardcoded structure.
        score = min(0.99, 0.17 + (0.65 * coverage) + (0.18 if sheet_match else 0.0))
        signals = [f"column:{signal}" for signal in matched_columns]
        if sheet_match:
            signals.append("sheet_or_title")
        scored_rules.append((score, rule, signals, sheet_match))

    scored_rules.sort(key=lambda item: item[0], reverse=True)
    best_score, best_rule, best_signals, best_sheet_match = scored_rules[0]
    second_score = scored_rules[1][0] if len(scored_rules) > 1 else 0.0
    ambiguous = (best_score - second_score) < 0.06 and not best_sheet_match

    if best_score < TABLE_CONFIDENCE_THRESHOLD or ambiguous:
        return ClassificationResult(
            detected_type="Unknown",
            confidence=round(best_score, 4),
            requires_interpretation=True,
            matched_signals=tuple(best_signals),
        )
    return ClassificationResult(
        detected_type=best_rule.table_type,
        confidence=round(best_score, 4),
        requires_interpretation=False,
        matched_signals=tuple(best_signals),
    )
