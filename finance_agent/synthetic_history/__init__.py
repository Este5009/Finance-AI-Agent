"""Synthetic multi-period university finance history generation package."""

from finance_agent.synthetic_history.generator import generate_synthetic_history
from finance_agent.synthetic_history.models import SyntheticHistoryConfig
from finance_agent.synthetic_history.validation import validate_generated_history

__all__ = [
    "SyntheticHistoryConfig",
    "generate_synthetic_history",
    "validate_generated_history",
]
