import pytest

from convoy_allocation import Convoy, Escort, ThreatScenario, solve_dro_allocation


def _data():
    convoys = {
        "C1": Convoy(ships=12, threat=1.0, baseline_survival=0.50),
        "C2": Convoy(ships=8, threat=1.2, baseline_survival=0.55),
    }
    escorts = {
        "E1": Escort(protection=0.30),
        "E2": Escort(protection=0.22),
    }
    scenarios = {
        "nominal": ThreatScenario(0.60, {"C1": 1.0, "C2": 1.0}),
        "c1_stress": ThreatScenario(0.25, {"C1": 1.8, "C2": 0.9}),
        "c2_stress": ThreatScenario(0.15, {"C1": 0.9, "C2": 2.0}),
    }
    return convoys, escorts, scenarios


def test_zero_radius_matches_nominal_expectation():
    convoys, escorts, scenarios = _data()
    result = solve_dro_allocation(convoys, escorts, scenarios, 2, ambiguity_radius=0.0)
    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(result.nominal_expected_survivors)
    assert result.worst_case_distribution == pytest.approx(
        {name: scenario.probability for name, scenario in scenarios.items()}
    )


def test_positive_radius_moves_probability_toward_bad_outcomes():
    convoys, escorts, scenarios = _data()
    nominal = solve_dro_allocation(convoys, escorts, scenarios, 2, ambiguity_radius=0.0)
    robust = solve_dro_allocation(convoys, escorts, scenarios, 2, ambiguity_radius=0.4)
    assert robust.objective_value <= nominal.nominal_expected_survivors + 1e-7
    assert sum(robust.worst_case_distribution.values()) == pytest.approx(1.0)
    l1 = sum(
        abs(robust.worst_case_distribution[name] - scenarios[name].probability)
        for name in scenarios
    )
    assert l1 <= 0.4 + 1e-7


def test_invalid_ambiguity_radius_is_rejected():
    convoys, escorts, scenarios = _data()
    with pytest.raises(ValueError, match="ambiguity_radius"):
        solve_dro_allocation(convoys, escorts, scenarios, 2, ambiguity_radius=-0.1)
    with pytest.raises(ValueError, match="ambiguity_radius"):
        solve_dro_allocation(convoys, escorts, scenarios, 2, ambiguity_radius=2.1)
