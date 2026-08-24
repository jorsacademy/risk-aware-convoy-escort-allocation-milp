from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from pulp import LpBinary, LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value


@dataclass(frozen=True)
class Convoy:
    ships: int
    threat: float
    baseline_survival: float
    min_escorts: int = 0
    max_escorts: int | None = None


@dataclass(frozen=True)
class Escort:
    protection: float


@dataclass(frozen=True)
class ConvoyResult:
    convoy_id: str
    ships: int
    assigned_escorts: tuple[str, ...]
    baseline_survival: float
    survival_probability: float
    expected_survivors: float
    expected_losses: float


@dataclass(frozen=True)
class AllocationResult:
    status: str
    objective_value: float
    convoys: tuple[ConvoyResult, ...]

    @property
    def total_ships(self) -> int:
        return sum(item.ships for item in self.convoys)

    @property
    def expected_survivors(self) -> float:
        return sum(item.expected_survivors for item in self.convoys)

    @property
    def expected_losses(self) -> float:
        return sum(item.expected_losses for item in self.convoys)

    @property
    def survival_rate(self) -> float:
        if self.total_ships == 0:
            return 0.0
        return self.expected_survivors / self.total_ships


def _validate_inputs(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    max_available_escorts: int,
    diminishing_returns: Sequence[float],
) -> None:
    if not convoys:
        raise ValueError("At least one convoy is required.")
    if not escorts:
        raise ValueError("At least one escort is required.")
    if max_available_escorts < 0:
        raise ValueError("max_available_escorts must be non-negative.")
    if max_available_escorts > len(escorts):
        raise ValueError("max_available_escorts cannot exceed the number of escorts.")
    if not diminishing_returns:
        raise ValueError("diminishing_returns must contain at least one value.")
    if any(value_ <= 0 for value_ in diminishing_returns):
        raise ValueError("diminishing_returns values must be positive.")
    if any(a < b for a, b in zip(diminishing_returns, diminishing_returns[1:])):
        raise ValueError("diminishing_returns must be non-increasing.")

    for convoy_id, convoy in convoys.items():
        if convoy.ships <= 0:
            raise ValueError(f"Convoy {convoy_id} must contain at least one ship.")
        if convoy.threat <= 0:
            raise ValueError(f"Convoy {convoy_id} threat must be positive.")
        if not 0 <= convoy.baseline_survival <= 1:
            raise ValueError(f"Convoy {convoy_id} baseline_survival must be in [0, 1].")
        if convoy.min_escorts < 0:
            raise ValueError(f"Convoy {convoy_id} min_escorts must be non-negative.")
        upper = len(escorts) if convoy.max_escorts is None else convoy.max_escorts
        if upper < 0:
            raise ValueError(f"Convoy {convoy_id} max_escorts must be non-negative.")
        if convoy.min_escorts > upper:
            raise ValueError(f"Convoy {convoy_id} min_escorts cannot exceed max_escorts.")

    for escort_id, escort in escorts.items():
        if escort.protection <= 0:
            raise ValueError(f"Escort {escort_id} protection must be positive.")


def _marginal_gain(
    convoy: Convoy,
    escort_protection: float,
    return_factor: float,
    effectiveness_scale: float,
) -> float:
    raw_gain = effectiveness_scale * escort_protection * return_factor / convoy.threat
    return max(0.0, raw_gain)


def solve_allocation(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    max_available_escorts: int,
    *,
    diminishing_returns: Sequence[float] = (1.0, 0.72, 0.52, 0.38, 0.28, 0.20),
    effectiveness_scale: float = 0.9,
) -> AllocationResult:
    """Solve the risk-aware escort allocation MILP.

    The optimization and the reported expected-survival metrics use the same
    mathematical formulation. Each additional escort occupies an ordered slot
    with a smaller marginal effectiveness factor.
    """
    _validate_inputs(convoys, escorts, max_available_escorts, diminishing_returns)
    if effectiveness_scale <= 0:
        raise ValueError("effectiveness_scale must be positive.")

    convoy_ids = tuple(convoys)
    escort_ids = tuple(escorts)
    max_slots = min(max_available_escorts, len(diminishing_returns), len(escort_ids))

    model = LpProblem("Risk_Aware_Convoy_Escort_Allocation", LpMaximize)

    x = {
        (convoy_id, escort_id): LpVariable(
            f"assign_{convoy_id}_{escort_id}", cat=LpBinary
        )
        for convoy_id in convoy_ids
        for escort_id in escort_ids
    }

    z = {
        (convoy_id, escort_id, slot): LpVariable(
            f"slot_{convoy_id}_{escort_id}_{slot}", cat=LpBinary
        )
        for convoy_id in convoy_ids
        for escort_id in escort_ids
        for slot in range(max_slots)
    }

    slot_used = {
        (convoy_id, slot): LpVariable(
            f"slot_used_{convoy_id}_{slot}", cat=LpBinary
        )
        for convoy_id in convoy_ids
        for slot in range(max_slots)
    }

    baseline_expected_survivors = sum(
        convoys[convoy_id].ships * convoys[convoy_id].baseline_survival
        for convoy_id in convoy_ids
    )

    marginal_objective_terms = []
    for convoy_id in convoy_ids:
        convoy = convoys[convoy_id]
        remaining_probability = 1.0 - convoy.baseline_survival
        for slot in range(max_slots):
            factor = diminishing_returns[slot]
            for escort_id in escort_ids:
                gain = _marginal_gain(
                    convoy,
                    escorts[escort_id].protection,
                    factor,
                    effectiveness_scale,
                )
                capped_gain = min(gain, remaining_probability)
                marginal_objective_terms.append(
                    convoy.ships * capped_gain * z[convoy_id, escort_id, slot]
                )

    model += baseline_expected_survivors + lpSum(marginal_objective_terms)

    for escort_id in escort_ids:
        model += lpSum(x[convoy_id, escort_id] for convoy_id in convoy_ids) <= 1

    model += lpSum(x.values()) <= max_available_escorts

    for convoy_id in convoy_ids:
        convoy = convoys[convoy_id]
        max_for_convoy = len(escort_ids) if convoy.max_escorts is None else convoy.max_escorts
        max_for_convoy = min(max_for_convoy, max_slots)

        assigned_count = lpSum(x[convoy_id, escort_id] for escort_id in escort_ids)
        model += assigned_count >= convoy.min_escorts
        model += assigned_count <= max_for_convoy

        model += lpSum(slot_used[convoy_id, slot] for slot in range(max_slots)) == assigned_count

        for slot in range(max_slots):
            model += (
                lpSum(z[convoy_id, escort_id, slot] for escort_id in escort_ids)
                == slot_used[convoy_id, slot]
            )

        for slot in range(1, max_slots):
            model += slot_used[convoy_id, slot] <= slot_used[convoy_id, slot - 1]

        for escort_id in escort_ids:
            model += (
                lpSum(z[convoy_id, escort_id, slot] for slot in range(max_slots))
                == x[convoy_id, escort_id]
            )

        survival_gain = lpSum(
            min(
                _marginal_gain(
                    convoy,
                    escorts[escort_id].protection,
                    diminishing_returns[slot],
                    effectiveness_scale,
                ),
                1.0 - convoy.baseline_survival,
            )
            * z[convoy_id, escort_id, slot]
            for escort_id in escort_ids
            for slot in range(max_slots)
        )
        model += survival_gain <= 1.0 - convoy.baseline_survival

    model.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not return an optimal solution: {status}")

    convoy_results = []
    for convoy_id in convoy_ids:
        convoy = convoys[convoy_id]
        assigned = tuple(
            escort_id
            for escort_id in escort_ids
            if (x[convoy_id, escort_id].value() or 0.0) > 0.5
        )

        survival_probability = convoy.baseline_survival
        for escort_id in escort_ids:
            for slot in range(max_slots):
                if (z[convoy_id, escort_id, slot].value() or 0.0) > 0.5:
                    gain = _marginal_gain(
                        convoy,
                        escorts[escort_id].protection,
                        diminishing_returns[slot],
                        effectiveness_scale,
                    )
                    survival_probability += gain

        survival_probability = min(1.0, max(0.0, survival_probability))
        expected_survivors = convoy.ships * survival_probability
        expected_losses = convoy.ships - expected_survivors

        convoy_results.append(
            ConvoyResult(
                convoy_id=convoy_id,
                ships=convoy.ships,
                assigned_escorts=assigned,
                baseline_survival=convoy.baseline_survival,
                survival_probability=survival_probability,
                expected_survivors=expected_survivors,
                expected_losses=expected_losses,
            )
        )

    objective_value = float(value(model.objective))
    return AllocationResult(
        status=status,
        objective_value=objective_value,
        convoys=tuple(convoy_results),
    )
