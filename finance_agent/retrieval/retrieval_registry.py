"""Registry of deterministic retrieval interfaces for execution queues."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finance_agent.retrieval.retrieval_engine import RetrievalContext
    from finance_agent.retrieval.retrieval_models import RetrievalResult


RetrievalFunction = Callable[["RetrievalContext", dict[str, Any]], "RetrievalResult"]


@dataclass(frozen=True)
class RetrievalTool:
    """Metadata and callable implementation for one retrieval interface.

    Inputs: stable interface name, implementation, and a short description.
    Outputs: registry entry used by the queue executor.
    Assumptions: future SQL/API implementations can keep the same callable shape.
    """

    name: str
    function: RetrievalFunction
    description: str


class RetrievalRegistry:
    """Name-to-function registry for deterministic evidence retrieval.

    Inputs: registered retrieval tools.
    Outputs: lookup and execution helpers.
    Assumptions: interface names are stable even if implementations change later.
    """

    def __init__(self) -> None:
        """Create an empty retrieval registry.

        Inputs: none.
        Outputs: initialized registry.
        Assumptions: tools are registered explicitly by setup code.
        """

        self._tools: dict[str, RetrievalTool] = {}

    def register(
        self,
        name: str,
        function: RetrievalFunction,
        description: str,
    ) -> None:
        """Register one retrieval function by public interface name.

        Inputs: interface name, callable, and description.
        Outputs: mutates the registry.
        Assumptions: duplicate registration indicates a programming error.
        """

        if name in self._tools:
            raise ValueError(f"Retrieval tool already registered: {name}")
        self._tools[name] = RetrievalTool(name, function, description)

    def get(self, name: str) -> RetrievalTool:
        """Look up a registered retrieval tool.

        Inputs: public interface name.
        Outputs: retrieval tool metadata and callable.
        Assumptions: callers handle unknown names as retrieval failures.
        """

        if name not in self._tools:
            raise KeyError(f"Unknown retrieval tool: {name}")
        return self._tools[name]

    def names(self) -> tuple[str, ...]:
        """Return all registered interface names.

        Inputs: none.
        Outputs: sorted tuple of names.
        Assumptions: deterministic ordering helps tests and summaries.
        """

        return tuple(sorted(self._tools))


def create_default_registry() -> RetrievalRegistry:
    """Create the default local-output-backed retrieval registry.

    Inputs: none.
    Outputs: registry containing Step 8 retrieval interfaces and Step 7 aliases.
    Assumptions: functions read processed outputs only through RetrievalContext.
    """

    from finance_agent.retrieval.retrieval_engine import (
        retrieve_cashflow_history,
        retrieve_department_history,
        retrieve_financial_report,
        retrieve_payroll_history,
        retrieve_previous_cycle_memory,
        retrieve_student_payment_history,
        retrieve_transactions,
        retrieve_vendor_history,
    )

    registry = RetrievalRegistry()
    registry.register(
        "department_history",
        retrieve_department_history,
        "Retrieve department-level processed history.",
    )
    registry.register(
        "payroll_history",
        retrieve_payroll_history,
        "Retrieve processed payroll history.",
    )
    registry.register(
        "vendor_history",
        retrieve_vendor_history,
        "Retrieve processed vendor payment history.",
    )
    registry.register(
        "student_payment_history",
        retrieve_student_payment_history,
        "Retrieve processed student payment history.",
    )
    registry.register(
        "cashflow_history",
        retrieve_cashflow_history,
        "Retrieve processed cash-flow history.",
    )
    registry.register(
        "transactions",
        retrieve_transactions,
        "Retrieve bounded processed transaction rows.",
    )
    registry.register(
        "previous_cycle_memory",
        retrieve_previous_cycle_memory,
        "Retrieve compact previous-cycle memory placeholders and data-quality notes.",
    )
    registry.register(
        "financial_report",
        retrieve_financial_report,
        "Retrieve one processed financial report summary.",
    )

    # Step 7 queues use public get_* tool names. They are aliases to the same
    # generic interfaces so future SQL/API implementations can replace internals.
    registry.register("get_department_history", retrieve_department_history, "")
    registry.register("get_payroll_history", retrieve_payroll_history, "")
    registry.register("get_vendor_history", retrieve_vendor_history, "")
    registry.register("get_transactions", retrieve_transactions, "")
    registry.register("get_previous_cycle_memory", retrieve_previous_cycle_memory, "")
    registry.register("get_full_report", retrieve_financial_report, "")

    from finance_agent.memory.retrieval import (
        memory_result_to_retrieval_result,
        registry_adapter,
    )

    def memory_tool(name: str) -> RetrievalFunction:
        """Create a registry adapter for one historical memory tool.

        Inputs: memory tool name.
        Outputs: RetrievalFunction compatible with Step 8 registry.
        Assumptions: memory tools stay read-only and are not planner-integrated yet.
        """

        def run(context: "RetrievalContext", arguments: dict[str, Any]) -> "RetrievalResult":
            """Run one memory retrieval tool through the existing registry shape."""

            args = dict(arguments)
            args.setdefault(
                "database_path",
                str(context.project_root / "data" / "memory" / "finance_memory.db"),
            )
            return memory_result_to_retrieval_result(registry_adapter(name, args))

        return run

    registry.register("get_previous_period", memory_tool("get_previous_period"), "")
    registry.register("get_period_history", memory_tool("get_period_history"), "")
    registry.register("get_metric_history", memory_tool("get_metric_history"), "")
    registry.register(
        "get_memory_department_history",
        memory_tool("get_memory_department_history"),
        "Retrieve historical department memory without replacing current-period retrieval.",
    )
    registry.register(
        "get_historical_department_history",
        memory_tool("get_memory_department_history"),
        "Alias for historical department memory retrieval.",
    )
    registry.register("get_repeated_anomalies", memory_tool("get_repeated_anomalies"), "")
    registry.register("get_previous_recommendations", memory_tool("get_previous_recommendations"), "")
    registry.register("get_goal_progress", memory_tool("get_goal_progress"), "")
    registry.register("get_memory_facts", memory_tool("get_memory_facts"), "")
    registry.register("get_full_period_record", memory_tool("get_full_period_record"), "")
    registry.register("get_artifact_references", memory_tool("get_artifact_references"), "")
    return registry
