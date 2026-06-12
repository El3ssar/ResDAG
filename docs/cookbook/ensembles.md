---
description: Coupled ensemble forecasting — N independently-trained ESNs sharing one aggregated feedback signal, for variance reduction on chaotic systems.
---

<span class="rd-eyebrow">Cookbook</span>

# Coupled ensembles

A coupled ensemble trains N independent ESNs on the same data, then
forecasts with all of them at once: at every autoregressive step, each
model receives the *same* aggregated output (mean, by default) of all
models as its next feedback. Diversity comes from independent reservoir
initialization; the shared feedback damps the per-model drift that ruins
single-model chaotic forecasts.

## The whole thing

<div class="rd-window" data-title="coupled_ensemble.py" markdown>

```python
import resdag as rd

# data: (1, T, 3) — e.g. Lorenz-63
warmup, train, target, f_warmup, val = rd.utils.data.prepare_esn_data(
    data, warmup_steps=2_000, train_steps=15_000, val_steps=5_000,
    normalize=True,
)

ensemble = rd.coupled_ensemble_esn(
    n_models=5,            # sub-models built by model_factory (default: ott_esn)
    reservoir_size=600,    # forwarded to every factory call
    feedback_size=3,
    output_size=3,
    seed=0,                # sub-model i is built under seed + i
)

ensemble.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
    n_workers=2,           # thread-pool fit; PyTorch releases the GIL in BLAS
)

ensemble.reset_reservoirs()
pred, individuals = ensemble.forecast(
    f_warmup, horizon=val.shape[1], return_individuals=True
)
print(pred.shape)              # (1, 5000, 3) — aggregated forecast
print(individuals[0].shape)    # (1, 5000, 3) — one trajectory per sub-model
```

</div>

`fit` trains each sub-model independently with its own
[`ESNTrainer`](../reference/training.md) — coupling exists only at forecast
time. `forecast` runs two phases: every model is teacher-forced on the same
warmup window, then the autoregressive loop aggregates the N outputs at
each step and feeds that single tensor back to all of them.

Set `return_individuals=True` only when you want the per-model
trajectories (ensemble spread, divergence-time analysis): it allocates one
extra `(batch, horizon, output)` buffer per sub-model.

## Choosing an aggregator

The `aggregate=` argument takes a string or any `nn.Module` mapping a
stacked `(N, batch, time, features)` tensor to `(batch, time, features)`:

| Aggregator | When to use |
|---|---|
| `"mean"` (default) | Symmetric, fast, robust on bounded series. |
| `"median"` | Survives a single sub-model going haywire. |
| `OutliersFilteredMean` | Per-timestep outlier rejection by member norm — `method="z_score"` or `"iqr"`, with a `threshold`. Falls back to the plain mean if everything is flagged. |

```python
from resdag.ensemble.aggregators import OutliersFilteredMean

ensemble = rd.coupled_ensemble_esn(
    n_models=10, reservoir_size=400, feedback_size=3, output_size=3,
    aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
)
```

Any factory works as the sub-model architecture — pass
`model_factory=rd.classic_esn` (or your own `Callable(**kwargs) -> ESNModel`)
and its keyword arguments flow through unchanged.

!!! note "Driver alignment is the single-model convention"
    For input-driven ensembles, `forecast_inputs[i][:, t, :]` is the driver
    at the t-th step *after* the warmup window — pass driver series of
    length `horizon - 1` (or `horizon`; the last step is unused) continuing
    exactly where the warmup drivers ended. Same rule as
    `ESNModel.forecast`, applied to all N models at once.

## Related

- [Data preparation](data.md) — why `f_warmup` must be the tail of `train`.
- [Forecasting](../learn/forecasting.md) — the two-phase loop each sub-model runs.
- [Timing & alignment](../under-the-hood/timing-and-alignment.md) — the driver convention, step by step.
