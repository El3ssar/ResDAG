# Coupled ensemble forecasting

```python
import torch
from resdag import coupled_ensemble_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

# data: (1, T, 3)
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
)

ensemble.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

ensemble.reset_reservoirs()
pred = ensemble.forecast(f_warmup, horizon=val.shape[1])
```

Members are trained on the same splits; diversity comes from independent reservoir
initialization. During forecast, sub-models share aggregated feedback (`mean`,
`median`, or a custom module).

See [`CoupledEnsembleESNModel`](../reference/ensemble.md).
