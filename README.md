# Risk-Aware Convoy Escort Allocation MILP

A compact mixed-integer linear programming model for allocating a limited escort fleet across multiple convoys under heterogeneous risk.

The project is intentionally synthetic. All convoy identifiers, escort identifiers, parameters, and scenario values are fictional and are provided only to demonstrate optimization modeling techniques.

## Problem

A planner must assign a limited number of escort units to convoys. Each convoy has:

- a ship count,
- a baseline survival probability,
- a threat level,
- optional minimum and maximum escort counts.

Each escort has a protection score. Escort effectiveness is adjusted by convoy threat and subject to diminishing marginal returns. The model maximizes expected surviving ships.

## Mathematical structure

Let:

- `x[c,e] = 1` if escort `e` is assigned to convoy `c`, otherwise `0`.
- `y[c,k] = 1` if convoy `c` receives at least `k` escorts.

The model maximizes:

`sum_c ships[c] * (baseline_survival[c] + sum_k marginal_gain[c,k] * y[c,k])`

subject to:

- each escort can be assigned to at most one convoy,
- the number of assigned escorts does not exceed the available fleet limit,
- optional convoy-level minimum and maximum escort limits,
- consistency constraints linking `x` and `y`,
- ordered activation of marginal escort slots.

The marginal gains are generated from escort protection scores, convoy threat, and a diminishing-return schedule, then capped so survival probability never exceeds `1.0`.

## Why this formulation

A naive linear objective that values every additional escort identically can concentrate all resources on one convoy and can become inconsistent if post-processing uses a different loss formula. This implementation uses the same survival model for both optimization and reporting.

## Repository layout

- `src/convoy_allocation/model.py` — model construction and solution logic
- `src/convoy_allocation/scenario.py` — synthetic scenario data
- `src/convoy_allocation/cli.py` — command-line entry point
- `tests/` — unit tests
- `pyproject.toml` — package and tooling configuration
- `LICENSE` — non-commercial source-available license

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Usage

```bash
convoy-allocation
```

or:

```bash
python -m convoy_allocation.cli
```

## Testing

```bash
pip install -e .[dev]
pytest
```

## Model scope

This repository is an educational operations-research example, not a historical reconstruction, tactical tool, or validated operational model. The numerical parameters are synthetic and are not derived from archival or contemporary military data.

## License

This project is source-available for non-commercial research, education, and personal study only. Commercial use is prohibited. See `LICENSE` for the full terms.
