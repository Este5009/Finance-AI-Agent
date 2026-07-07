"""Tests for starter schema-normalization helpers."""

import pytest

from finance_agent.ingestion.schema import clean_column_name, map_column_alias, normalize_column_names


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("  Fecha de Pago  ", "fecha_de_pago"),
        ("ÃREA / UNIDAD", "area_unidad"),
        ("Total   Ejecutado", "total_ejecutado"),
        ("Proveedor-Nombre", "proveedor_nombre"),
    ],
)
def test_clean_column_name(raw_name: str, expected: str) -> None:
    """Verify names are trimmed, lowercased, de-accented, and underscored."""

    assert clean_column_name(raw_name) == expected


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("Departamento", "department"),
        ("Ã¡rea", "department"),
        ("importe", "amount"),
        ("Fecha", "date"),
        ("Ingresos", "revenue"),
        ("egresos", "expenses"),
        ("Presupuesto", "budget"),
        ("Ejecutado", "actual"),
        ("Proveedor", "vendor"),
        ("Estudiante", "student"),
    ],
)
def test_map_column_alias(alias: str, canonical: str) -> None:
    """Verify starter Spanish/English aliases map deterministically."""

    assert map_column_alias(alias) == canonical


def test_normalize_column_names_makes_alias_collisions_unique() -> None:
    """Verify duplicate aliases receive stable suffixes instead of being lost."""

    assert normalize_column_names(["Monto", "Importe", "Fecha", None]) == [
        "amount",
        "amount_2",
        "date",
        "unnamed_column_4",
    ]


def test_composite_financial_aliases_are_normalized() -> None:
    """Verify common compound financial headers map across languages."""

    assert normalize_column_names(
        [
            "Ingreso Presupuestado",
            "Gasto Ejecutado",
            "Monto Pagado",
            "Saldo Pendiente",
            "Flujo de Caja",
        ]
    ) == [
        "budget_revenue",
        "actual_expense",
        "amount_paid",
        "outstanding",
        "cash_flow",
    ]
