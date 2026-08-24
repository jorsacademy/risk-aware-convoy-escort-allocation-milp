"""Risk-aware convoy escort allocation package."""

from .model import AllocationResult, Convoy, Escort, solve_allocation
from .uncertainty import (
    ScenarioOutcome,
    ThreatScenario,
    UncertainAllocationResult,
    solve_uncertain_allocation,
)

__all__ = [
    "AllocationResult",
    "Convoy",
    "Escort",
    "ScenarioOutcome",
    "ThreatScenario",
    "UncertainAllocationResult",
    "solve_allocation",
    "solve_uncertain_allocation",
]
