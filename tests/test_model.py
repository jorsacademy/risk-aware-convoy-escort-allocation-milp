import pytest

from convoy_allocation.model import Convoy, Escort, solve_allocation
from convoy_allocation.scenario import build_demo_scenario


def test_demo_scenario_solves_optimally() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    assert result.status == "Optimal"
    assert result.total_ships == sum(item.ships for item in convoys.values())
    assert 0.0 <= result.survival_rate <= 1.0
    assert result.expected_survivors == pytest.approx(result.objective_value)


def test_each_escort_is_used_at_most_once() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    assigned = [escort for convoy in result.convoys for escort in convoy.assigned_escorts]
    assert len(assigned) == len(set(assigned))
    assert len(assigned) <= max_available_escorts


def test_survival_probability_respects_bounds() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    for convoy in result.convoys:
        assert convoy.baseline_survival <= convoy.survival_probability <= 1.0
        assert convoy.expected_losses >= 0.0


def test_rejects_invalid_resource_limit() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}

    with pytest.raises(ValueError, match="cannot exceed"):
        solve_allocation(convoys, escorts, max_available_escorts=2)


def test_rejects_increasing_return_schedule() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}

    with pytest.raises(ValueError, match="non-increasing"):
        solve_allocation(
            convoys,
            escorts,
            max_available_escorts=1,
            diminishing_returns=(0.5, 0.8),
        )
