# Coupled ensemble

Train five `ott_esn` sub-models on the same Lorenz series, then forecast
autoregressively with **shared feedback** between them. The aggregated
output is more stable than any individual sub-model.

## Setup

```python
import torch
from resdag import coupled_ensemble_esn
from resdag.utils.data import prepare_esn_data
```

We reuse the `lorenz63` helper from the [Lorenz example](lorenz.md). The
data splits are identical.

```python
data = lorenz63()
warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=2_000,
    train_steps=15_000,
    val_steps=5_000,
    discard_steps=2_000,
    normalize=True,
)
```

## Build the ensemble

```python
ensemble = coupled_ensemble_esn(
    n_models=5,
    reservoir_size=600,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    topology="erdos_renyi",
    seed=0,                # reproducible — same seed → same sub-models
)
```

Each sub-model is a complete `ESNModel` (here `ott_esn` under the
hood — that's the default factory). Diversity comes from each
sub-model's independent random reservoir.

## Train and forecast

```python
ensemble.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
    n_workers=2,        # CPU-parallel ridge solves
)

ensemble.reset_reservoirs()
pred, individuals = ensemble.forecast(
    f_warmup,
    horizon=val.shape[1],
    return_individuals=True,
)
```

`pred` is the aggregated forecast — by default a simple mean across
sub-models, fed back into every sub-model as the next-step input.
`individuals` is the list of per-sub-model trajectories, useful for
plotting the spread or computing per-model error.

## Aggregator choice

The aggregation strategy is set when the ensemble is built. For chaotic
systems with occasional sub-model blow-ups, an outlier-robust aggregator
helps:

```python
from resdag.ensemble.aggregators import OutliersFilteredMean

ensemble = coupled_ensemble_esn(
    n_models=10, reservoir_size=400, feedback_size=3, output_size=3,
    aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
)
```

## Reference

- [`CoupledEnsembleESNModel`](../reference/ensemble.md) — class API.
- [`OutliersFilteredMean`](../reference/ensemble.md) — aggregator
  module.
- [Guide: coupled ensembles](../guides/coupled-ensembles.md) — task-level
  treatment.
