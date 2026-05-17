# Your first ESN

Sine wave, `prepare_esn_data`, `classic_esn`, train, forecast on the validation segment.

```python
import torch
from resdag import classic_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

torch.manual_seed(0)
t = torch.linspace(0, 40 * torch.pi, 4_000)
data = torch.sin(t).view(1, -1, 1)

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=400,
    train_steps=3_000,
    val_steps=600,
    normalize=True,
    norm_method="minmax",
)

model = classic_esn(
    reservoir_size=400,
    feedback_size=1,
    output_size=1,
    spectral_radius=0.9,
)

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

model.reset_reservoirs()
pred = model.forecast(f_warmup, horizon=val.shape[1])
mse = torch.mean((pred - val) ** 2).item()
print("val MSE:", mse)
```

`f_warmup` is the last `warmup_steps` rows of `train` (see
[`prepare_esn_data`](../reference/utils/data.md)). `val` is the unseen tail used
only for evaluation.

## Next

[Lorenz walkthrough](lorenz-walkthrough.md)
