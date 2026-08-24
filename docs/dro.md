# Distributionally robust optimization

The DRO extension treats the supplied scenario probabilities as nominal estimates rather than exact probabilities.

For nominal probability vector `p`, the ambiguity set is

```text
P(rho) = { q : q >= 0, sum(q) = 1, ||q - p||_1 <= rho }
```

where `rho = ambiguity_radius` is between 0 and 2. The allocation solves

```text
maximize_x  min_{q in P(rho)} sum_s q[s] Q_s(x)
```

where `Q_s(x)` is the survivor total under scenario `s` for a common first-stage escort allocation `x`.

`rho = 0` reproduces the nominal probability model. Increasing `rho` permits an adversarial probability distribution to move more probability mass toward scenarios with poorer survivor outcomes. At the upper end of the range the ambiguity set can contain distributions far from the nominal estimate, producing a more conservative allocation.

The inner probability minimization is represented through linear-program duality, so the outer allocation remains a MILP and uses the same CBC solver backend as the deterministic, stochastic, maximin, and CVaR formulations.

## API

```python
from convoy_allocation import solve_dro_allocation

result = solve_dro_allocation(
    convoys,
    escorts,
    scenarios,
    max_available_escorts=4,
    ambiguity_radius=0.25,
)

print(result.objective_value)
print(result.nominal_expected_survivors)
print(result.worst_case_expected_survivors)
print(result.worst_case_distribution)
```

The returned worst-case distribution is recomputed from the solved allocation with a small explicit LP. This provides an auditable probability vector and keeps the reported objective aligned with the realized scenario outcomes.
