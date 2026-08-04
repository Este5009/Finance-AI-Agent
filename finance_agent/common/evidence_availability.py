"""Deterministic checks for whether claimed-missing evidence exists."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


DEPARTMENT_DISPLAY_ALIASES: dict[str, str] = {
    "Arts & Humanities": "Artes y Humanidades",
    "Health Sciences": "Ciencias de la Salud",
    "University": "Universidad",
    "Engineering": "Ingeniería",
    "Business": "Negocios",
    "Student Services": "Servicios Estudiantiles",
    "Administration": "Administración",
}

DEPARTMENT_FIELD_TERMS: dict[str, tuple[str, ...]] = {
    "budget_revenue": ("budget revenue", "presupuesto de ingresos", "ingresos presupuestados"),
    "actual_revenue": ("actual revenue", "ingresos reales", "ingresos del departamento", "ingresos"),
    "budget_expenses": ("budget expense", "budget expenses", "presupuesto de gastos", "gastos presupuestados"),
    "actual_expenses": ("actual expense", "actual expenses", "gastos reales", "gastos del departamento", "gastos"),
    "net_operating_result": ("net contribution", "resultado operativo", "contribución neta", "contribucion neta"),
    "expense_variance": ("expense variance", "variación de gastos", "variacion de gastos"),
    "expense_variance_pct": ("expense variance", "variación de gastos", "variacion de gastos"),
}

ABSENCE_TERMS: tuple[str, ...] = (
    "missing",
    "faltante",
    "falta",
    "no se proporciona",
    "no se proporcionan",
    "no disponible",
    "no están disponibles",
    "no estan disponibles",
    "para completar",
)


def normalize_text_key(value: Any) -> str:
    """Normalize a user/data text value for conservative matching.

    Inputs: any text-like value.
    Outputs: Unicode-normalized, whitespace-collapsed, casefolded string.
    Assumptions: this is for comparison only; source values are not mutated.
    """

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def department_aliases(value: Any) -> set[str]:
    """Return normalized aliases for a department display/canonical name.

    Inputs: department text from artifacts or prose.
    Outputs: normalized aliases including known English/Spanish display forms.
    Assumptions: aliases are presentation/data labels, not SQL identifiers.
    """

    text = str(value or "").strip()
    aliases = {normalize_text_key(text)}
    if text in DEPARTMENT_DISPLAY_ALIASES:
        aliases.add(normalize_text_key(DEPARTMENT_DISPLAY_ALIASES[text]))
    for canonical, display in DEPARTMENT_DISPLAY_ALIASES.items():
        if normalize_text_key(text) == normalize_text_key(display):
            aliases.add(normalize_text_key(canonical))
    return {alias for alias in aliases if alias}


def _value_is_present(value: Any) -> bool:
    """Return whether a processed scalar is materially present.

    Inputs: scalar value from processed finance artifacts.
    Outputs: True when not empty/null/NaN.
    Assumptions: zero is a valid financial value and must remain present.
    """

    if value is None or value == "":
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _department_rows(finance_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return processed department rows from a finance summary document.

    Inputs: Step 3 finance summary JSON.
    Outputs: list of department dictionaries.
    Assumptions: report generation passes processed outputs, never raw Excel.
    """

    rows = finance_summary.get("department_summary", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def find_department_row(finance_summary: dict[str, Any], department: Any) -> dict[str, Any] | None:
    """Find a department row using canonical or Spanish display aliases.

    Inputs: finance summary and department text.
    Outputs: matching department row or None.
    Assumptions: normalized aliases are only used for matching data values.
    """

    wanted = department_aliases(department)
    if not wanted:
        return None
    for row in _department_rows(finance_summary):
        if department_aliases(row.get("department")) & wanted:
            return row
    return None


def _period_slug(finance_summary: dict[str, Any]) -> str:
    """Return a filename-safe period slug for provenance hints.

    Inputs: processed finance summary document.
    Outputs: slug such as ``2026_11``.
    Assumptions: report_period is already deterministic pipeline metadata.
    """

    period = str(finance_summary.get("report_period") or "").strip()
    return period.replace("-", "_") if period else "unknown"


def department_field_provenance(
    finance_summary: dict[str, Any],
    department: Any,
    field: str,
) -> dict[str, Any] | None:
    """Return provenance when one department field exists.

    Inputs: finance summary, department name/alias, and canonical field.
    Outputs: provenance dictionary or None when absent.
    Assumptions: source values come from processed calculation outputs.
    """

    row = find_department_row(finance_summary, department)
    if not row or not _value_is_present(row.get(field)):
        return None
    return {
        "department": row.get("department"),
        "field": field,
        "value": row.get(field),
        "source_table": "department_summary",
        "source_artifact": f"outputs/calculations/department_summary_{_period_slug(finance_summary)}.csv",
    }


def contradicted_department_missing_claims(
    text: Any,
    finance_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find missing-data claims contradicted by processed department rows.

    Inputs: prose/missing-information text and finance summary document.
    Outputs: provenance records for each contradicted entity/field claim.
    Assumptions: only objective availability claims are checked; business
    interpretation remains untouched.
    """

    normalized = normalize_text_key(text)
    if not normalized or not any(term in normalized for term in ABSENCE_TERMS):
        return []
    contradictions: list[dict[str, Any]] = []
    for row in _department_rows(finance_summary):
        department = row.get("department")
        if not (department_aliases(department) & {alias for alias in department_aliases(department) if alias in normalized}):
            # Check each alias as a substring so Spanish display names in prose
            # match English canonical names in processed artifacts.
            if not any(alias and alias in normalized for alias in department_aliases(department)):
                continue
        for field, terms in DEPARTMENT_FIELD_TERMS.items():
            if any(term in normalized for term in terms):
                provenance = department_field_provenance(finance_summary, department, field)
                if provenance:
                    contradictions.append(provenance)
    return contradictions


def filter_contradicted_missing_information(
    items: list[Any],
    finance_summary: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Remove missing-information items contradicted by processed evidence.

    Inputs: missing-information items and finance summary.
    Outputs: kept items plus provenance for removed contradictions.
    Assumptions: removal is limited to objective missing-data claims whose
    referenced department/metric exists in deterministic outputs.
    """

    kept: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for item in items:
        contradictions = contradicted_department_missing_claims(item, finance_summary)
        if contradictions:
            provenance.append({"removed_item": str(item), "checked_sources": contradictions})
            continue
        kept.append(item)
    return kept, provenance


def remove_contradicted_department_absence_text(text: Any, finance_summary: dict[str, Any]) -> str:
    """Blank a department narrative when it contains contradicted absence claims.

    Inputs: narrative text and processed finance summary.
    Outputs: original text or an empty string when contradicted by evidence.
    Assumptions: callers can fall back to deterministic summaries when this
    returns an empty string; this never writes replacement analysis.
    """

    return "" if contradicted_department_missing_claims(text, finance_summary) else str(text or "")
