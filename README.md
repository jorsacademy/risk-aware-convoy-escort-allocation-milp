# Risk-Aware Convoy Escort Allocation MILP

A compact operations-research project for allocating a limited set of escort resources across multiple synthetic convoys under heterogeneous risk, diminishing marginal returns, and uncertain threat conditions.

The repository is intentionally synthetic. All convoy identifiers, escort identifiers, parameters, scenarios, and numerical values are fictional and are provided only to demonstrate optimization modeling techniques.

## Models included

The repository now contains three closely related formulations:

1. **Deterministic MILP** — assumes each convoy threat score is known.
2. **Risk-neutral stochastic MILP** — selects one allocation before a finite threat scenario is observed and maximizes probability-weighted expected survivors.
3. **Finite-scenario robust MILP** — selects one allocation that maximizes the worst scenario outcome among the supplied scenarios.

A continuous `risk_aversion` parameter also allows intermediate solutions between the stochastic expected-value objective and the robust maximin objective.

## Deterministic model

A planner assigns a limited number of escorts to convoys. Each convoy has:

- a ship count,
- a baseline survival probability,
- a threat score,
- optional minimum and maximum escort requirements.

Each escort has a protection score. Escort effectiveness is adjusted by convoy threat and by an ordered diminishing-return schedule. The model maximizes total expected surviving ships while respecting resource and assignment constraints.

The principal binary variables are:

- `x[c,e]`: escort `e` is assigned to convoy `c`,
- `z[c,e,k]`: escort `e` occupies diminishing-return slot `k` for convoy `c`,
- `s[c,k]`: slot `k` is active for convoy `c`.

For convoy `c`, escort `e`, and slot `k`, the synthetic survival increment is based on:

`effectiveness_scale * protection[e] * return_factor[k] / threat[c]`

The coefficient is capped by the convoy's remaining probability mass, and a convoy-level linear constraint ensures modeled survival probability cannot exceed one.

See [`docs/formulation.md`](docs/formulation.md) for the deterministic formulation.

## Uncertainty-aware model

The uncertainty extension introduces a finite scenario set. Each scenario has:

- a probability,
- one positive threat multiplier for every convoy.

Escort assignments are **here-and-now decisions**: the same assignment must be used in every scenario. Scenario-dependent threat multipliers change the survival contribution of each selected escort-slot pair.

If `Q[s]` denotes total expected survivors under scenario `s`, then the risk-neutral model maximizes:

`sum_s probability[s] * Q[s]`

For robust optimization, a variable `eta` is constrained by:

`eta <= Q[s]` for every scenario `s`

so that maximizing `eta` maximizes the worst supplied scenario outcome.

The implemented risk-adjusted objective is:

`(1 - alpha) * expected_survivors + alpha * worst_case_survivors`

where `alpha = risk_aversion`.

- `alpha = 0.0`: risk-neutral stochastic allocation,
- `0.0 < alpha < 1.0`: expected/worst-case compromise,
- `alpha = 1.0`: finite-scenario maximin robust allocation.

See [`docs/uncertainty.md`](docs/uncertainty.md) for the full uncertainty formulation and interpretation limits.

## Why this formulation is stronger than a naive allocation model

A naive formulation can accidentally optimize one expression and then report outcomes using a different expression. It can also value every additional escort identically, encourage unrealistic resource concentration, or ignore uncertainty entirely.

This project addresses those problems by:

- using the same survival-gain coefficients in the objective, feasibility constraints, and result reconstruction,
- representing heterogeneous escorts explicitly,
- introducing ordered marginal-effect slots for diminishing returns,
- preventing survival probability from exceeding `1.0`,
- validating resource limits and aggregate minimum requirements before solving,
- rejecting non-finite and invalid numeric inputs,
- checking solver status before results are returned,
- supporting finite threat scenarios with explicit probabilities,
- supporting both expected-value and worst-case optimization,
- covering deterministic and uncertainty-aware invariants with automated tests.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── docs/
│   ├── formulation.md
│   └── uncertainty.md
├── src/convoy_allocation/
│   ├── __init__.py
│   ├── cli.py
│   ├── model.py
│   ├── scenario.py
│   ├── uncertainty.py
│   └── uncertainty_cli.py
├── tests/
│   ├── test_model.py
│   └── test_uncertainty.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Run the deterministic demonstration

```bash
convoy-allocation
```

or:

```bash
python -m convoy_allocation.cli
```

## Run the uncertainty demonstration

```bash
convoy-allocation-uncertain
```

or:

```bash
python -m convoy_allocation.uncertainty_cli
```

The uncertainty demonstration solves both a risk-neutral stochastic model and a fully risk-averse maximin model using the same synthetic scenario set.

## Python API example

```python
from convoy_allocation.scenario import (
    build_demo_scenario,
    build_uncertain_demo_scenarios,
)
from convoy_allocation.uncertainty import solve_uncertain_allocation

convoys, escorts, limit = build_demo_scenario()
scenarios = build_uncertain_demo_scenarios()

result = solve_uncertain_allocation(
    convoys,
    escorts,
    scenarios,
    limit,
    risk_aversion=0.5,
)

print(result.expected_survivors)
print(result.worst_case_survivors)
```

## Run the test suite

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite verifies, among other properties:

- deterministic objective/reporting consistency,
- escort exclusivity,
- survival-probability bounds,
- convoy minimum and maximum assignment limits,
- preference for a stronger escort in a controlled single-slot case,
- diminishing marginal benefit across slots,
- rejection of infeasible aggregate minimum requirements,
- rejection of non-finite numeric inputs,
- correct baseline behavior when no escort resources are available,
- stochastic objective consistency with probability-weighted outcomes,
- robust objective consistency with the worst supplied scenario,
- equivalence between the deterministic model and a one-scenario nominal uncertainty model,
- uncertainty-model escort exclusivity,
- rejection of invalid scenario probability distributions and incomplete scenario definitions.

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12 for pushes and pull requests targeting `main`.

## Model scope and interpretation

This repository is an educational operations-research example. It is not a historical reconstruction, tactical tool, empirical survival model, forecast, or validated operational decision system.

The protection scores, threat scores, survival probabilities, scenario probabilities, threat multipliers, scale parameter, and diminishing-return factors are synthetic assumptions. The optimization is mathematically conditional on those assumptions, but the numerical outputs should not be interpreted as estimates for any real-world operation.

## Design limitations

The current models do not include:

- travel time or multi-period escort scheduling,
- escort endurance or refueling,
- route compatibility,
- correlated loss processes,
- convoy splitting or merging,
- dynamic reallocation after scenario observation,
- recourse decisions,
- distributionally robust ambiguity sets,
- CVaR or other tail-risk measures,
- empirically calibrated nonlinear survival functions.

These are natural extensions for more advanced operations-research work.

## License

This project is source-available for non-commercial research, education, and personal study only. Commercial use is prohibited. It is not distributed under an OSI-approved open-source license.

See [`LICENSE`](LICENSE) for the full terms.
