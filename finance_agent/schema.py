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
}


def clean_column_name(column_name: object) -> str:
    """Convert a raw column label to a snake_case identifier.

    Inputs: any scalar column label.
    Outputs: lowercase ASCII-oriented text with normalized separators.
    Assumptions: removing accents is acceptable for internal identifiers.
    """

    raw_name = "" if column_name is None else str(column_name)
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
