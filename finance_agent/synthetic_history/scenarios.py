"""Deterministic scenario definitions for synthetic financial histories."""

from __future__ import annotations

from finance_agent.synthetic_history.models import MonthlyScenarioPoint


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


MONTH_NAMES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def get_recovery_year_scenario() -> list[MonthlyScenarioPoint]:
    """Return the default 12-month recovery scenario.

    Inputs:
        None.
    Outputs:
        Ordered monthly scenario points for the recovery-year narrative.
    Assumptions:
        The scenario is coherent by construction: payroll pressure rises before
        stabilization, collections recover after the campaign, and cash flow
        improves late in the year.
    """

    return [
        MonthlyScenarioPoint(1, "Línea base saludable con márgenes estables.", 1.00, 0.38, 0.95, 150_000),
        MonthlyScenarioPoint(2, "Ligera caída de matrícula e ingresos.", 0.97, 0.39, 0.93, 80_000),
        MonthlyScenarioPoint(3, "La nómina empieza a crecer por cobertura operativa.", 0.96, 0.42, 0.91, 20_000),
        MonthlyScenarioPoint(4, "Aumenta el tiempo extra en Ciencias de la Salud.", 0.95, 0.45, 0.89, -120_000, 1.65),
        MonthlyScenarioPoint(
            5,
            "Se introduce meta de reducción de horas extra.",
            0.94,
            0.47,
            0.86,
            -220_000,
            1.85,
            recommendation_milestone=True,
            policy_action_es="Meta: reducir 18% el tiempo extra de Ciencias de la Salud antes de septiembre.",
        ),
        MonthlyScenarioPoint(6, "El problema persiste y el flujo de caja se debilita.", 0.93, 0.53, 0.84, -680_000, 2.05),
        MonthlyScenarioPoint(7, "Aparece anomalía recurrente de proveedor crítico.", 0.94, 0.52, 0.85, -350_000, 1.75, True),
        MonthlyScenarioPoint(
            8,
            "Campaña de cobranza mejora la recuperación estudiantil.",
            0.96,
            0.49,
            0.90,
            -120_000,
            1.35,
            True,
            policy_action_es="Campaña de cobranza: seguimiento semanal a saldos vencidos.",
        ),
        MonthlyScenarioPoint(
            9,
            "Congelamiento de contrataciones y control de costos.",
            0.98,
            0.46,
            0.92,
            50_000,
            1.12,
            True,
            policy_action_es="Congelamiento selectivo de contrataciones no críticas.",
        ),
        MonthlyScenarioPoint(10, "La nómina se estabiliza tras controles operativos.", 1.00, 0.43, 0.94, 180_000, 1.02),
        MonthlyScenarioPoint(11, "El flujo de caja mejora por cobranza y disciplina de gasto.", 1.02, 0.41, 0.95, 320_000),
        MonthlyScenarioPoint(12, "La mayoría de metas anuales mejora; quedan riesgos abiertos.", 1.04, 0.40, 0.96, 450_000),
    ]


def get_scenario_points(scenario: str) -> list[MonthlyScenarioPoint]:
    """Return scenario points for a supported scenario name.

    Inputs:
        scenario: Scenario name supplied in configuration.
    Outputs:
        Ordered monthly scenario points.
    Assumptions:
        Phase 12A intentionally supports only the recovery scenario.
    """

    if scenario != "recovery":
        raise ValueError(f"Unsupported synthetic history scenario: {scenario}")
    return get_recovery_year_scenario()
