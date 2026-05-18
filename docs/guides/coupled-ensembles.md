# Coupled ensemble forecasting

A coupled ensemble trains N independent ESN sub-models on the same data,
then forecasts autoregressively while **sharing a single aggregated
feedback** between them. Diversity comes from independent reservoir
initialisation; coupling damps the per-model drift that ruins single-model
chaotic forecasts.

```python
import torch
from resdag import coupled_ensemble_esn
from resdag.utils.data import prepare_esn_data

# data: (1, T, 3) — e.g. Lorenz-63
warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=2_000,
    train_steps=15_000,
    val_steps=5_000,
    discard_steps=2_000,
    normalize=True,
)

ensemble = coupled_ensemble_esn(
    n_models=5,
    reservoir_size=600,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    seed=0,                # deterministic per-sub-model init
)

ensemble.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
    n_workers=2,           # PyTorch releases the GIL → real CPU parallelism
)

ensemble.reset_reservoirs()
pred, individuals = ensemble.forecast(
    f_warmup, horizon=val.shape[1], return_individuals=True,
)
```

`pred` is the aggregated forecast (`mean` by default). `individuals` is a
list of per-sub-model trajectories, useful when you want to see the
ensemble spread.

## Aggregators

Pick the aggregation strategy via the `aggregate=` argument:

| Aggregator | When to use |
|---|---|
| `"mean"` (default) | Symmetric, fast, robust on bounded series. |
| `"median"` | Resists a single sub-model going haywire. |
| [`OutliersFilteredMean`](../reference/ensemble.md) | Z-score or IQR-based outlier rejection per timestep. |

```python
from resdag.ensemble.aggregators import OutliersFilteredMean

ensemble = coupled_ensemble_esn(
    n_models=10, reservoir_size=400, feedback_size=3, output_size=3,
    aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
)
```

## Reference

See [`CoupledEnsembleESNModel`](../reference/ensemble.md) for the full
class API (state management, save/load, per-sub-model forecasting).
