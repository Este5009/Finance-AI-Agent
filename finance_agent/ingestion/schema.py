"""Starter helpers for deterministic column-name normalization."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping


COLUMN_ALIASES: dict[str, str] = {
    "department": "department",
    "departamento": "department",
    "area": "department",
    "unidad": "department",
    "amount": "amount",
    "monto": "amount",
    "importe": "amount",
    "total": "amount",
    "valor": "amount",
    "date": "date",
    "fecha": "date",
    "revenue": "revenue",
    "ingresos": "revenue",
    "expenses": "expenses",
    "expense": "expenses",
    "gastos": "expenses",
    "egresos": "expenses",
    "budget": "budget",
    "presupuesto": "budget",
    "actual": "actual",
    "ejecutado": "actual",
    "vendor": "vendor",
    "proveedor": "vendor",
    "student": "student",
    "estudiante": "student",
    "student_id": "student_id",
    "id_estudiante": "student_id",
    "codigo_estudiante": "student_id",
    "invoice": "invoice",
    "invoice_id": "invoice_id",
    "factura": "invoice",
    "numero_factura": "invoice_number",
    "invoice_number": "invoice_number",
    "payment": "payment",
    "pago": "payment",
    "payment_date": "payment_date",
    "fecha_pago": "payment_date",
    "due_date": "due_date",
    "fecha_vencimiento": "due_date",
    "billing_period": "billing_period",
    "periodo_facturacion": "billing_period",
    "period": "period",
    "periodo": "period",
    "month": "month",
    "mes": "month",
    "year": "year",
    "ano": "year",
    "anio": "year",
    "student_year": "student_year",
    "ano_estudiante": "student_year",
    "program": "program",
    "programa": "program",
    "category": "category",
    "categoria": "category",
    "revenue_category": "revenue_category",
    "categoria_ingreso": "revenue_category",
    "tipo_ingreso": "revenue_category",
    "expense_category": "expense_category",
    "categoria_gasto": "expense_category",
    "tipo_gasto": "expense_category",
    "budget_revenue": "budget_revenue",
    "ingreso_presupuestado": "budget_revenue",
    "actual_revenue": "actual_revenue",
    "ingreso_ejecutado": "actual_revenue",
    "budget_expense": "budget_expense",
    "gasto_presupuestado": "budget_expense",
    "actual_expense": "actual_expense",
    "gasto_ejecutado": "actual_expense",
    "budget_amount": "budget_amount",
    "monto_presupuestado": "budget_amount",
    "actual_amount": "actual_amount",
    "monto_ejecutado": "actual_amount",
    "variance": "variance",
    "variacion": "variance",
    "desviacion": "variance",
    "variance_pct": "variance_pct",
    "variacion_pct": "variance_pct",
    "porcentaje_variacion": "variance_pct",
    "amount_due": "amount_due",
    "monto_adeudado": "amount_due",
    "amount_paid": "amount_paid",
    "monto_pagado": "amount_paid",
    "outstanding": "outstanding",
    "saldo_pendiente": "outstanding",
    "status": "status",
    "estado": "status",
    "days_overdue": "days_overdue",
    "dias_vencidos": "days_overdue",
    "payroll": "payroll",
    "nomina": "payroll",
    "planilla": "payroll",
    "salary": "salary",
    "salario": "salary",
    "base_salary": "base_salary",
    "salario_base": "base_salary",
    "benefits": "benefits",
    "beneficios": "benefits",
    "overtime": "overtime",
    "horas_extra": "overtime",
    "total_payroll": "total_payroll",
    "payroll_budget": "payroll_budget",
    "headcount": "headcount",
    "headcount_fte": "headcount_fte",
    "scholarship": "scholarship",
    "scholarships": "scholarship",
    "beca": "scholarship",
    "becas": "scholarship",
    "scholarship_type": "scholarship_type",
    "tipo_beca": "scholarship_type",
    "allocated": "allocated",
    "asignado": "allocated",
    "awarded": "awarded",
    "otorgado": "awarded",
    "remaining": "remaining",
    "remanente": "remaining",
    "recipients": "recipients",
    "beneficiarios": "recipients",
    "cash": "cash",
    "caja": "cash",
    "cash_flow": "cash_flow",
    "flujo_caja": "cash_flow",
    "flujo_de_caja": "cash_flow",
    "beginning_cash": "beginning_cash",
    "saldo_inicial": "beginning_cash",
    "ending_cash": "ending_cash",
    "saldo_final": "ending_cash",
    "cash_inflows": "cash_inflows",
    "entradas_efectivo": "cash_inflows",
    "cash_outflows": "cash_outflows",
    "salidas_efectivo": "cash_outflows",
    "net_cash_flow": "net_cash_flow",
    "flujo_neto": "net_cash_flow",
    "vendor_name": "vendor",
    "nombre_proveedor": "vendor",
    "payment_method": "payment_method",
    "metodo_pago": "payment_method",
    "approval_status": "approval_status",
    "estado_aprobacion": "approval_status",
    "metric": "metric",
    "metrica": "metric",
    "goal": "goal",
    "meta": "goal",
    "goal_or_budget": "goal_or_budget",
    "format": "format",
    "formato": "format",
    "count": "count",
    "cantidad": "count",
    "rate": "rate",
    "tasa": "rate",
}


def clean_column_name(column_name: object) -> str:
    """Convert a raw column label to a snake_case identifier.

    Inputs: any scalar column label.
    Outputs: lowercase ASCII-oriented text with normalized separators.
    Assumptions: removing accents is acceptable for internal identifiers.
    """

    raw_name = "" if column_name is None else str(column_name)
    # Some historical fixtures contain UTF-8 text decoded as Latin-1. Repairing
    # that narrow mojibake case keeps normalization stable after file moves.
    if "Ã" in raw_name:
        try:
            raw_name = raw_name.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    # NFKD separates accents from base characters so accents can be discarded.
    decomposed = unicodedata.normalize("NFKD", raw_name)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    lowered = without_accents.strip().lower()
    with_underscores = re.sub(r"\s+", "_", lowered)
    alphanumeric_only = re.sub(r"[^a-z0-9_]+", "_", with_underscores)
    return re.sub(r"_+", "_", alphanumeric_only).strip("_")


def map_column_alias(
    column_name: object,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Map a cleaned Spanish/English alias to a canonical name.

    Inputs: raw column label and optional alias mapping override.
    Outputs: a canonical alias, or the cleaned name when no alias exists.
    Assumptions: exact deterministic aliases are safest in this phase.
    """

    cleaned_name = clean_column_name(column_name)
    alias_map = COLUMN_ALIASES if aliases is None else aliases
    normalized_aliases = {
        clean_column_name(source): clean_column_name(target)
        for source, target in alias_map.items()
    }
    return normalized_aliases.get(cleaned_name, cleaned_name)


def normalize_column_names(
    column_names: Iterable[object],
    *,
    aliases: Mapping[str, str] | None = None,
) -> list[str]:
    """Clean, alias-map, and de-duplicate column labels.

    Inputs: column labels and an optional alias mapping.
    Outputs: unique canonical labels in original order.
    Assumptions: duplicate canonical names receive numeric suffixes.
    """

    normalized_names: list[str] = []
    occurrences: dict[str, int] = {}
    for position, column_name in enumerate(column_names, start=1):
        normalized = map_column_alias(column_name, aliases)
        if not normalized:
            normalized = f"unnamed_column_{position}"
        occurrences[normalized] = occurrences.get(normalized, 0) + 1
        occurrence = occurrences[normalized]
        normalized_names.append(normalized if occurrence == 1 else f"{normalized}_{occurrence}")
    return normalized_names
