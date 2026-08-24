from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value

from .model import Convoy, Escort
from .uncertainty import (
    ScenarioOutcome,
    ThreatScenario,
    _EPS,
    _build_uncertain_core,
    _extract_outcomes,
    _finite,
    _validate_uncertain_inputs,
)


@dataclass(frozen=True)
class DROAllocationResult:
    """Result of a distributionally robust finite-scenario allocation."""

    status: str
    objective_value: float
    ambiguity_radius: float
    assigned_escorts: Mapping[str, tuple[str, ...]]
    scenarios: tuple[ScenarioOutcome, ...]
    worst_case_distribution: Mapping[str, float]

    @property
    def nominal_expected_survivors(self) -> float:
        return sum(item.probability * item.expected_survivors for item in self.scenarios)

    @property
    def worst_case_expected_survivors(self) -> float:
        return sum(
            self.worst_case_distribution[item.scenario_id] * item.expected_survivors
            for item in self.scenarios
        )


def _validate_ambiguity_radius(ambiguity_radius: float) -> None:
    if not _finite(ambiguity_radius) or not 0 <= ambiguity_radius <= 2:
        raise ValueError("ambiguity_radius must be in [0, 2].")


def _worst_case_distribution(
    outcomes: Sequence[ScenarioOutcome], ambiguity_radius: float
) -> dict[str, float]:
    """Compute the worst distribution in an L1 ball around nominal probabilities.

    For a fixed allocation, minimizing expected survivors over

        ||q - p||_1 <= ambiguity_radius, q >= 0, sum(q) = 1

    is a small linear program. The L1 radius ranges from zero (nominal
    probabilities only) to two (the largest possible distance between two
    probability distributions).
    """
    scenario_ids = tuple(item.scenario_id for item in outcomes)
    nominal = {item.scenario_id: item.probability for item in outcomes}
    survivors = {item.scenario_id: item.expected_survivors for item in outcomes}

    model = LpProblem("Worst_Case_Scenario_Distribution", sense=1)
    q = {s: LpVariable(f"q_{i}", lowBound=0, upBound=1) for i, s in enumerate(scenario_ids)}
    deviation = {
        s: LpVariable(f"deviation_{i}", lowBound=0)
        for i, s in enumerate(scenario_ids)
    }

    model += lpSum(survivors[s] * q[s] for s in scenario_ids)
    model += lpSum(q.values()) == 1
    model += lpSum(deviation.values()) <= ambiguity_radius
    for s in scenario_ids:
        model += deviation[s] >= q[s] - nominal[s]
        model += deviation[s] >= nominal[s] - q[s]

    model.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Worst-case distribution solver failed: {status}")
    return {s: float(q[s].value() or 0.0) for s in scenario_ids}


def solve_dro_allocation(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    scenarios: Mapping[str, ThreatScenario],
    max_available_escorts: int,
    *,
    ambiguity_radius: float = 0.20,
    diminishing_returns: Sequence[float] = (1.0, 0.72, 0.52, 0.38, 0.28, 0.20),
    effectiveness_scale: float = 0.9,
) -> DROAllocationResult:
    """Solve an L1 probability-ambiguity distributionally robust MILP.

    The ambiguity set contains every scenario probability vector q satisfying
    ||q - p||_1 <= ambiguity_radius, where p is the supplied nominal scenario
    distribution. The model maximizes expected survivors under the worst
    probability vector in that ambiguity set.

    The inner distribution minimization is dualized, leaving one MILP that can
    be solved by the same CBC backend as the other package models.
    """
    _validate_uncertain_inputs(
        convoys,
        escorts,
        scenarios,
        max_available_escorts,
        diminishing_returns,
        effectiveness_scale,
        risk_aversion=0.0,
    )
    _validate_ambiguity_radius(ambiguity_radius)

    model, x, scenario_totals, convoy_ids, escort_ids, scenario_ids = _build_uncertain_core(
        convoys,
        escorts,
        scenarios,
        max_available_escorts,
        diminishing_returns,
        effectiveness_scale,
        "Distributionally_Robust_Convoy_Escort_Allocation",
    )

    # Dual of min_q sum_s q_s Q_s subject to q in an L1 ball around p.
    # q = p + u - v, with u,v >= 0 and sum(u+v) <= radius. The compact
    # dual below uses a free normalization multiplier and nonnegative
    # multipliers for the absolute-deviation budget.
    theta = LpVariable("dro_theta", lowBound=None, upBound=None)
    gamma = LpVariable("dro_gamma", lowBound=0)
    a = {s: LpVariable(f"dro_a_{i}", lowBound=0) for i, s in enumerate(scenario_ids)}
    b = {s: LpVariable(f"dro_b_{i}", lowBound=0) for i, s in enumerate(scenario_ids)}

    for s in scenario_ids:
        model += theta - a[s] + b[s] <= scenario_totals[s]
        model += a[s] + b[s] <= gamma

    model += theta - ambiguity_radius * gamma + lpSum(
        scenarios[s].probability * (-a[s] + b[s]) for s in scenario_ids
    )

    model.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not return an optimal solution: {status}")

    assigned = {
        c: tuple(e for e in escort_ids if (x[c, e].value() or 0.0) > 0.5)
        for c in convoy_ids
    }
    outcomes = _extract_outcomes(convoys, scenarios, scenario_totals, scenario_ids)
    worst_distribution = _worst_case_distribution(outcomes, ambiguity_radius)
    reported_objective = sum(
        worst_distribution[item.scenario_id] * item.expected_survivors for item in outcomes
    )

    return DROAllocationResult(
        status=status,
        objective_value=reported_objective,
        ambiguity_radius=ambiguity_radius,
        assigned_escorts=assigned,
        scenarios=outcomes,
        worst_case_distribution=worst_distribution,
    )
