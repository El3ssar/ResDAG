# Coupled ensemble forecasting

Train $N$ independent `ott_esn` models; at forecast time they share one aggregated
feedback signal.

```python
import torch
from resdag import coupled_ensemble_esn
from resdag.ensemble.aggregators import OutliersFilteredMean

# ... build warmup, train, target, f_warmup, ground_truth tensors (B, T, 3) ...

ensemble = coupled_ensemble_esn(
    n_models=5,
    reservoir_size=300,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    # aggregate="mean"  # default
)

ensemble.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": train.clone()},
)

ensemble.reset_reservoirs()
pred = ensemble.forecast(f_warmup, horizon=ground_truth.shape[1])
```

## Robust aggregation

```python
ensemble = coupled_ensemble_esn(
    n_models=10,
    feedback_size=3,
    output_size=3,
    aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
    reservoir_size=300,
)
```

## Gotchas

- Same training data for all members — diversity is from random initialization only.
- `coupled_ensemble_esn` is a factory; use `CoupledEnsembleESNModel` directly if members
  need different topologies.
- Always `reset_reservoirs()` before forecast after training.

## See also

- [Ensembles (Learn)](../learn/ensembles.md)
- [Example 11](https://github.com/El3ssar/resdag/blob/main/examples/11_ensemble_forecasting.py)
