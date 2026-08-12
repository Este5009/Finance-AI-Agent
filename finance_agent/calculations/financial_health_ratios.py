"""Deterministic financial-health ratio definitions and calculations.

This module is the single source of truth for internal management ratio
thresholds used by calculations, report models, Streamlit, HTML, and PDF. The
thresholds are analytical management references, not regulatory or
institutional policy limits unless a future source explicitly proves otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from finance_agent.calculations.calculation_loader import LoadedIntermediateTable
from finance_agent.ingestion.schema import clean_column_name, map_column_alias


@dataclass(frozen=True)
class RatioThresholdBand:
    """One display/classification band for a management ratio.

    Inputs: Spanish label, lower/upper numeric bounds, and inclusivity flags.
    Outputs: immutable threshold metadata consumed by the classifier.
    Assumptions: bounds are expressed in raw ratio units, not display percent.
    """

    label: str
    minimum: float | None = None
    maximum: float | None = None
    include_minimum: bool = True
    include_maximum: bool = True


@dataclass(frozen=True)
class FinancialHealthRatioDefinition:
    """Canonical definition for one deterministic financial-health ratio.

    Inputs: metric identity, Spanish labels, formula input keys, unit, and
        threshold bands.
    Outputs: shared ratio definition used across calculation/presentation.
    Assumptions: all formulas are simple numerator/denominator ratios.
    """

    metric_id: str
    label_es: str
    numerator_key: str
    denominator_key: str
    unit: str
    formula_es: str
    reference_range_es: str
    interpretation_es: str
    bands: tuple[RatioThresholdBand, ...]


FINANCIAL_HEALTH_RATIO_DEFINITIONS: tuple[FinancialHealthRatioDefinition, ...] = (
    FinancialHealthRatioDefinition(
        "current_ratio",
        "Razón corriente",
        "current_assets",
        "current_liabilities",
        "ratio_number",
        "Activo corriente / Pasivo corriente",
        "Crítico < 1.00; Aceptable 1.00–1.49; Bueno ≥ 1.50",
        "Mide la capacidad de cubrir obligaciones corrientes con activos corrientes.",
        (
            RatioThresholdBand("Crítico", maximum=1.0, include_maximum=False),
            RatioThresholdBand("Aceptable", minimum=1.0, maximum=1.49),
            RatioThresholdBand("Bueno", minimum=1.5),
        ),
    ),
    FinancialHealthRatioDefinition(
        "cash_ratio",
        "Razón de efectivo",
        "cash_and_equivalents",
        "current_liabilities",
        "ratio_number",
        "Efectivo y equivalentes / Pasivo corriente",
        "Crítico < 0.30; Aceptable 0.30–0.59; Bueno ≥ 0.60",
        "Mide la cobertura inmediata de obligaciones corrientes con efectivo disponible.",
        (
            RatioThresholdBand("Crítico", maximum=0.30, include_maximum=False),
            RatioThresholdBand("Aceptable", minimum=0.30, maximum=0.59),
            RatioThresholdBand("Bueno", minimum=0.60),
        ),
    ),
    FinancialHealthRatioDefinition(
        "total_debt_ratio",
        "Endeudamiento total",
        "total_liabilities",
        "total_assets",
        "ratio",
        "Pasivo total / Activo total",
        "Crítico > 60%; Aceptable 50–60%; Bueno < 50%",
        "Mide qué proporción de los activos está financiada con pasivos.",
        (
            RatioThresholdBand("Bueno", maximum=0.50, include_maximum=False),
            RatioThresholdBand("Aceptable", minimum=0.50, maximum=0.60),
            RatioThresholdBand("Crítico", minimum=0.60, include_minimum=False),
        ),
    ),
    FinancialHealthRatioDefinition(
        "ebitda_margin",
        "Margen EBITDA",
        "ebitda",
        "total_revenue",
        "ratio",
        "EBITDA / Ingresos",
        "Crítico < 20%; Aceptable 20–29.9%; Bueno ≥ 30%",
        "Mide la rentabilidad operativa antes de intereses, impuestos, depreciación y amortización.",
        (
            RatioThresholdBand("Crítico", maximum=0.20, include_maximum=False),
            RatioThresholdBand("Aceptable", minimum=0.20, maximum=0.299),
            RatioThresholdBand("Bueno", minimum=0.30),
        ),
    ),
    FinancialHealthRatioDefinition(
        "net_margin",
        "Margen neto",
        "net_income",
        "total_revenue",
        "ratio",
        "Utilidad neta / Ingresos",
        "Crítico < 10%; Aceptable 10–19.9%; Bueno ≥ 20%",
        "Mide la proporción de ingresos que queda como utilidad neta.",
        (
            RatioThresholdBand("Crítico", maximum=0.10, include_maximum=False),
            RatioThresholdBand("Aceptable", minimum=0.10, maximum=0.199),
            RatioThresholdBand("Bueno", minimum=0.20),
        ),
    ),
    FinancialHealthRatioDefinition(
        "return_on_assets",
        "ROA",
        "net_income",
        "total_assets",
        "ratio",
        "Utilidad neta / Activo total",
        "Crítico < 8%; Aceptable 8–14.9%; Bueno ≥ 15%",
        "Mide la utilidad neta generada por cada unidad de activo total.",
        (
            RatioThresholdBand("Crítico", maximum=0.08, include_maximum=False),
            RatioThresholdBand("Aceptable", minimum=0.08, maximum=0.149),
            RatioThresholdBand("Bueno", minimum=0.15),
        ),
    ),
)


RATIO_DEFINITIONS_BY_ID = {
    definition.metric_id: definition for definition in FINANCIAL_HEALTH_RATIO_DEFINITIONS
}


def number_value(value: Any) -> float | None:
    """Coerce a processed scalar to float.

    Inputs: arbitrary table or summary value.
    Outputs: float when valid, otherwise None.
    Assumptions: missing values should remain unavailable, not zero-filled.
    """

    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_financial_health_ratio(metric_id: str, value: float | None) -> str:
    """Classify one ratio using the canonical internal-management bands.

    Inputs: metric identifier and ratio value.
    Outputs: Spanish classification label.
    Assumptions: unavailable values produce ``No disponible``.
    """

    if value is None:
        return "No disponible"
    definition = RATIO_DEFINITIONS_BY_ID.get(metric_id)
    if definition is None:
        return "No disponible"
    for band in definition.bands:
        if band.minimum is not None:
            if value < band.minimum or (value == band.minimum and not band.include_minimum):
                continue
        if band.maximum is not None:
            if value > band.maximum or (value == band.maximum and not band.include_maximum):
                continue
        return band.label
    return "No disponible"


def _canonical_metric_name(value: Any) -> str:
    """Normalize a metric label from wide or metric/value source tables.

    Inputs: raw source label.
    Outputs: canonical metric key when an alias exists, otherwise cleaned key.
    Assumptions: matching is deterministic and exact after normalization.
    """

    text = str(value or "")
    mapped = map_column_alias(text)
    cleaned = clean_column_name(mapped)
    synonyms = {
        "activo_corriente": "current_assets",
        "activos_corrientes": "current_assets",
        "total_current_assets": "current_assets",
        "pasivo_corriente": "current_liabilities",
        "pasivos_corrientes": "current_liabilities",
        "total_current_liabilities": "current_liabilities",
        "efectivo": "cash_and_equivalents",
        "efectivo_y_equivalentes": "cash_and_equivalents",
        "efectivo_y_equivalentes_de_efectivo": "cash_and_equivalents",
        "equivalentes_de_efectivo": "cash_and_equivalents",
        "cash": "cash_and_equivalents",
        "cash_and_cash_equivalents": "cash_and_equivalents",
        "cash_equivalents": "cash_and_equivalents",
        "activo_total": "total_assets",
        "activos_totales": "total_assets",
        "assets_total": "total_assets",
        "pasivo_total": "total_liabilities",
        "pasivos_totales": "total_liabilities",
        "liabilities_total": "total_liabilities",
        "utilidad_neta": "net_income",
        "utilidad_del_ejercicio": "net_income",
        "resultado_neto": "net_income",
        "excedente_neto": "net_income",
        "net_earnings": "net_income",
        "ingresos": "total_revenue",
        "revenue": "total_revenue",
    }
    return synonyms.get(cleaned, cleaned)


def _collect_values_from_table(table: LoadedIntermediateTable) -> dict[str, dict[str, Any]]:
    """Extract canonical financial-position values from one normalized table.

    Inputs: loaded intermediate table.
    Outputs: mapping from canonical metric to value/provenance.
    Assumptions: supported tables may be wide or metric/value shaped.
    """

    frame = table.dataframe
    values: dict[str, dict[str, Any]] = {}
    if frame.empty:
        return values

    # Wide format: one or more rows with canonical columns such as
    # current_assets, total_liabilities, ebitda, and net_income.
    for column in frame.columns:
        canonical = _canonical_metric_name(column)
        if canonical not in _required_source_keys():
            continue
        first_value = next((number_value(value) for value in frame[column] if number_value(value) is not None), None)
        if first_value is not None:
            values[canonical] = {
                "value": first_value,
                "source_table": table.detected_type,
                "source_sheet": table.sheet,
                "source_column": column,
                "source_table_id": table.table_id,
            }

    # Metric/value format: e.g. Metric = "Activo corriente", Value = 4500000.
    metric_columns = [column for column in frame.columns if _canonical_metric_name(column) in {"metric", "account", "category", "concept"}]
    value_columns = [column for column in frame.columns if _canonical_metric_name(column) in {"value", "amount", "actual", "balance"}]
    if metric_columns and value_columns:
        metric_column = metric_columns[0]
        value_column = value_columns[0]
        for _, row in frame.iterrows():
            canonical = _canonical_metric_name(row.get(metric_column))
            numeric = number_value(row.get(value_column))
            if canonical in _required_source_keys() and numeric is not None:
                values[canonical] = {
                    "value": numeric,
                    "source_table": table.detected_type,
                    "source_sheet": table.sheet,
                    "source_column": value_column,
                    "source_metric_label": row.get(metric_column),
                    "source_table_id": table.table_id,
                }
    return values


def _required_source_keys() -> set[str]:
    """Return all source fields required by canonical ratio definitions.

    Inputs: none.
    Outputs: set of canonical source metric IDs.
    Assumptions: revenue may be supplied by the main finance summary.
    """

    keys: set[str] = set()
    for definition in FINANCIAL_HEALTH_RATIO_DEFINITIONS:
        keys.add(definition.numerator_key)
        keys.add(definition.denominator_key)
    keys.update({"value", "amount", "actual", "balance", "metric", "account", "concept"})
    return keys


def collect_financial_health_inputs(
    tables: Iterable[LoadedIntermediateTable],
    finance_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Collect financial-health inputs from processed tables and summaries.

    Inputs: selected normalized tables and current finance summary.
    Outputs: canonical value/provenance records keyed by source metric.
    Assumptions: deterministic processed summaries can supply total revenue.
    """

    collected: dict[str, dict[str, Any]] = {}
    revenue = number_value(finance_summary.get("total_revenue"))
    if revenue is not None:
        collected["total_revenue"] = {
            "value": revenue,
            "source_table": "finance_summary",
            "source_sheet": "Revenue",
            "source_column": "total_revenue",
            "source_table_id": "finance_summary",
        }
    for table in tables:
        for key, record in _collect_values_from_table(table).items():
            collected[key] = record
    return collected


def calculate_financial_health_ratios(
    tables: Iterable[LoadedIntermediateTable],
    finance_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calculate all configured financial-health ratios deterministically.

    Inputs: selected normalized tables and existing finance summary.
    Outputs: list of ratio result dictionaries with provenance and
        classification metadata.
    Assumptions: missing numerator/denominator values produce unavailable rows.
    """

    inputs = collect_financial_health_inputs(tables, finance_summary)
    rows: list[dict[str, Any]] = []
    for definition in FINANCIAL_HEALTH_RATIO_DEFINITIONS:
        numerator = number_value((inputs.get(definition.numerator_key) or {}).get("value"))
        denominator = number_value((inputs.get(definition.denominator_key) or {}).get("value"))
        value = numerator / denominator if numerator is not None and denominator not in {None, 0.0} else None
        classification = classify_financial_health_ratio(definition.metric_id, value)
        rows.append(
            {
                "metric": definition.metric_id,
                "display_label": definition.label_es,
                "value": value,
                "unit": definition.unit,
                "availability": "available" if value is not None else "unavailable",
                "classification": classification,
                "reference_range": definition.reference_range_es,
                "reference_origin": "configurable_internal_management_threshold",
                "reference_origin_es": "Referencia interna configurable de gestión",
                "is_regulatory_limit": False,
                "formula": definition.formula_es,
                "interpretation": definition.interpretation_es,
                "required_inputs": [definition.numerator_key, definition.denominator_key],
                "missing_inputs": [
                    key
                    for key, observed in (
                        (definition.numerator_key, numerator),
                        (definition.denominator_key, denominator),
                    )
                    if observed is None
                ],
                "source_provenance": {
                    "numerator": inputs.get(definition.numerator_key),
                    "denominator": inputs.get(definition.denominator_key),
                },
            }
        )
    return rows
