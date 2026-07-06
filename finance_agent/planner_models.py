"""Structured models for deterministic financial investigation plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class PriorityLevel(str, Enum):
    """Supported investigation priority levels in descending urgency."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


PRIORITY_ORDER = {
    PriorityLevel.CRITICAL: 4,
    PriorityLevel.HIGH: 3,
    PriorityLevel.MEDIUM: 2,
    PriorityLevel.LOW: 1,
}


@dataclass(frozen=True)
class EvidenceRequest:
    """Evidence a future retrieval layer should obtain for one investigation."""

    request_id: str
    tool_name: str
    parameters: dict[str, Any]
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize a future evidence request.

        Inputs: this request.
        Outputs: JSON-compatible request metadata.
        Assumptions: the planner describes calls but never executes them.
        """

        return asdict(self)


@dataclass(frozen=True)
class InvestigationTask:
    """One planned investigation tied to an anomaly or data-quality issue."""

    task_id: str
    anomaly_id: str
    priority: PriorityLevel
    priority_score: int
    question_to_answer: str
    reason: str
    required_evidence: tuple[EvidenceRequest, ...]
    suggested_tool: str
    expected_output: str
    status: str = "planned"
    prioritization_factors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize one investigation task and its evidence requests.

        Inputs: this task.
        Outputs: JSON-compatible task dictionary.
        Assumptions: status remains planned until a later orchestration step.
        """

        return {
            "task_id": self.task_id,
            "anomaly_id": self.anomaly_id,
            "priority": self.priority.value,
            "priority_score": self.priority_score,
            "question_to_answer": self.question_to_answer,
            "reason": self.reason,
            "required_evidence": [
                request.to_dict() for request in self.required_evidence
            ],
            "suggested_tool": self.suggested_tool,
            "expected_output": self.expected_output,
            "status": self.status,
            "prioritization_factors": list(self.prioritization_factors),
        }


@dataclass(frozen=True)
class InvestigationPlan:
    """Prioritized deterministic investigation plan for one reporting scope."""

    plan_id: str
    report_period: str
    period_slug: str
    source_files: tuple[str, ...]
    tasks: tuple[InvestigationTask, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize a complete plan with summary counts.

        Inputs: this plan.
        Outputs: JSON-compatible plan metadata, counts, and ordered tasks.
        Assumptions: tasks are already sorted by priority and score.
        """

        counts = Counter(task.priority.value for task in self.tasks)
        return {
            "plan_id": self.plan_id,
            "report_period": self.report_period,
            "period_slug": self.period_slug,
            "source_files": list(self.source_files),
            "total_tasks": len(self.tasks),
            "tasks_by_priority": {
                priority.value: counts.get(priority.value, 0)
                for priority in PriorityLevel
            },
            "tasks": [task.to_dict() for task in self.tasks],
        }
