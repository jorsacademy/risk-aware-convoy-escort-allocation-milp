import pytest

from convoy_allocation.model import Convoy, Escort, solve_allocation
from convoy_allocation.scenario import (
    build_demo_scenario,
    build_uncertain_demo_scenarios,
)
from convoy_allocation.uncertainty import ThreatScenario, solve_uncertain_allocation


def test_risk_neutral_objective_matches_expected_survivors() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    result = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=0.0,
    )

    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(result.expected_survivors, abs=1e-8)


def test_full_risk_aversion_matches_worst_case_survivors() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    result = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=1.0,
    )

    assert result.objective_value == pytest.approx(result.worst_case_survivors, abs=1e-8)


def test_single_nominal_scenario_matches_deterministic_model() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = {
        "Nominal": ThreatScenario(
            probability=1.0,
            threat_multipliers={convoy_id: 1.0 for convoy_id in convoys},
        )
    }

    deterministic = solve_allocation(convoys, escorts, limit)
    uncertain = solve_uncertain_allocation(convoys, escorts, scenarios, limit)

    assert uncertain.expected_survivors == pytest.approx(
        deterministic.expected_survivors,
        abs=1e-8,
    )


def test_uncertain_assignment_respects_escort_exclusivity() -> None:
    convoys, escorts, limit = build_demo_scenario()
    result = solve_uncertain_allocation(
        convoys,
        escorts,
        build_uncertain_demo_scenarios(),
        limit,
    )

    assigned = [
        escort_id
        for escort_ids in result.assigned_escorts.values()
        for escort_id in escort_ids
    ]
    assert len(assigned) == len(set(assigned))
    assert len(assigned) <= limit


def test_worst_case_solution_cannot_have_lower_worst_case_than_risk_neutral_solution() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    neutral = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=0.0,
    )
    robust = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=1.0,
    )

    assert robust.worst_case_survivors + 1e-8 >= neutral.worst_case_survivors


def test_rejects_scenario_probabilities_that_do_not_sum_to_one() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {
        "S1": ThreatScenario(0.6, {"Convoy-A": 1.0}),
        "S2": ThreatScenario(0.3, {"Convoy-A": 1.2}),
    }

    with pytest.raises(ValueError, match="sum to 1.0"):
        solve_uncertain_allocation(convoys, escorts, scenarios, 1)


def test_rejects_incomplete_scenario_multiplier_map() -> None:
    convoys = {
        "Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5),
        "Convoy-B": Convoy(ships=10, threat=5.0, baseline_survival=0.5),
    }
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {"S1": ThreatScenario(1.0, {"Convoy-A": 1.0})}

    with pytest.raises(ValueError, match="every convoy"):
        solve_uncertain_allocation(convoys, escorts, scenarios, 1)


def test_rejects_risk_aversion_outside_unit_interval() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {"S1": ThreatScenario(1.0, {"Convoy-A": 1.0})}

    with pytest.raises(ValueError, match="risk_aversion"):
        solve_uncertain_allocation(
            convoys,
            escorts,
            scenarios,
            1,
            risk_aversion=1.1,
        )
