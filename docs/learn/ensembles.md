# Coupled ensembles

!!! info "Why this exists"
    A single ESN seed can fail on chaotic data. **Ensembles** average multiple
    independently initialized models. ResDAG's `CoupledEnsembleESNModel` goes further:
    at forecast time every member receives the **same aggregated feedback**, coupling
    trajectories while preserving weight diversity.

## Naive vs coupled averaging

| Strategy | Training | Forecast feedback |
|----------|----------|---------------------|
| Train $N$ models, average outputs post hoc | Independent | Each uses its own prediction |
| **Coupled ensemble** | Independent | All use mean/median/robust aggregate of members' outputs |

Coupling shares one autoregressive input drive, which often stabilizes long-horizon chaos
forecasts compared to uncoupled rollouts.

## How it works

1. Build $N$ `ESNModel` instances (different reservoir draws / topologies).
2. Wrap in `CoupledEnsembleESNModel(models, aggregator="mean")`.
3. `ensemble.fit(...)` trains each member with `ESNTrainer`.
4. `ensemble.forecast(warmup, horizon=H)` — each step:
   - each model predicts;
   - `aggregator` combines outputs → shared feedback for the next step.

```python
from resdag.models import coupled_ensemble_esn

ensemble = coupled_ensemble_esn(
    n_models=5,
    reservoir_size=300,
    feedback_size=3,
    output_size=3,
)
ensemble.fit((warmup,), (train,), {"output": targets})
ensemble.reset_reservoirs()
pred = ensemble.forecast(f_warmup, horizon=200)
```

## Aggregators

| `aggregator` | Behaviour |
|--------------|-----------|
| `"mean"` | Arithmetic mean across models |
| `"median"` | Per-feature median |
| `OutliersFilteredMean` | Drop outlier members by norm, then mean |

Custom modules accept stacked tensors `(N, batch, time, features)` and return
`(batch, time, features)` — see [Extend: custom aggregator](../extending/custom-aggregator.md).

## Where diversity comes from

Members differ because each `ESNModel` draws its own $W_{\mathrm{res}}$ and input weights
at initialization. You do **not** need different data splits — same `(warmup, train, target)`
for all.

## See also

- [`CoupledEnsembleESNModel`](../reference/ensemble.md)
- [Coupled ensemble guide](../guides/coupled-ensembles.md)
- [Chaos & losses](chaos-and-losses.md) — tuning members with HPO
