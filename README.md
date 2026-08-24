# Risk-Aware Convoy Escort Allocation MILP

A compact mixed-integer linear programming model for allocating a limited set of escort resources across multiple synthetic convoys under heterogeneous risk and diminishing marginal returns.

The repository is intentionally synthetic. All convoy identifiers, escort identifiers, parameters, and scenario values are fictional and are provided only to demonstrate optimization modeling techniques.

## What the model does

A planner must assign a limited number of escorts to convoys. Each convoy has:

- a ship count,
- a baseline survival probability,
- a threat score,
- optional minimum and maximum escort requirements.

Each escort has a protection score. Escort effectiveness is adjusted by convoy threat and by an ordered diminishing-return schedule. The model maximizes total expected surviving ships while respecting resource and assignment constraints.

## Why this version is stronger than a naive allocation model

A naive linear formulation can accidentally optimize one expression and then report outcomes using a different expression. It can also value the first and fifth escort identically, which encourages unrealistic resource concentration.

This implementation avoids those issues by:

- using one shared survival-gain coefficient table in the objective, feasibility constraints, and result reconstruction,
- representing heterogeneous escorts explicitly,
- introducing ordered escort-effect slots for diminishing marginal returns,
- preventing survival probability from exceeding `1.0`,
- validating resource limits and aggregate minimum requirements before solving,
- rejecting non-finite and invalid numeric inputs,
- checking solver status before results are returned,
- covering core invariants with automated tests.

## Optimization structure

The principal binary variables are:

- `x[c,e]`: escort `e` is assigned to convoy `c`,
- `z[c,e,k]`: escort `e` occupies diminishing-return slot `k` for convoy `c`,
- `s[c,k]`: slot `k` is active for convoy `c`.

For convoy `c`, escort `e`, and slot `k`, the synthetic survival increment is based on:

`effectiveness_scale * protection[e] * return_factor[k] / threat[c]`

The coefficient is capped by the convoy's remaining probability mass. A convoy-level linear constraint then ensures the sum of selected increments cannot push survival probability above one.

The objective is:

`maximize total expected surviving ships`

subject to:

- each escort is assigned to at most one convoy,
- total assignments do not exceed the available resource limit,
- convoy minimum and maximum escort requirements are respected,
- each assigned escort occupies exactly one marginal-effect slot,
- each active slot contains exactly one escort,
- later slots cannot activate before earlier slots,
- modeled survival probability remains in `[0, 1]`.

See [`docs/formulation.md`](docs/formulation.md) for the complete mathematical formulation.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── docs/formulation.md
├── src/convoy_allocation/
│   ├── __init__.py
│   ├── cli.py
│   ├── model.py
│   └── scenario.py
├── tests/test_model.py
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

## Run the demonstration

```bash
convoy-allocation
```

or:

```bash
python -m convoy_allocation.cli
```

## Run the test suite

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite verifies, among other properties:

- objective/reporting consistency,
- escort exclusivity,
- survival-probability bounds,
- convoy minimum and maximum assignment limits,
- preference for a stronger escort in a controlled single-slot case,
- diminishing marginal benefit across slots,
- rejection of infeasible aggregate minimum requirements,
- rejection of non-finite numeric inputs,
- correct baseline behavior when no escort resources are available.

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12 for pushes and pull requests targeting `main`.

## Model scope and interpretation

This repository is an educational operations-research example. It is not a historical reconstruction, tactical tool, empirical survival model, or validated operational decision system.

The protection scores, threat scores, survival probabilities, scale parameter, and diminishing-return factors are synthetic assumptions. The model is mathematically consistent with those assumptions, but the resulting probabilities should not be interpreted as estimates for any real-world operation.

## Design limitations

The model is intentionally compact. It does not currently model:

- travel time or multi-period escort scheduling,
- escort endurance or refueling,
- route compatibility,
- uncertain threat scenarios,
- correlated losses,
- convoy splitting or merging,
- dynamic reallocation,
- empirically calibrated nonlinear survival functions.

These are natural extensions for more advanced operations-research work.

## License

This project is source-available for non-commercial research, education, and personal study only. Commercial use is prohibited. It is not distributed under an OSI-approved open-source license.

See [`LICENSE`](LICENSE) for the full terms.
