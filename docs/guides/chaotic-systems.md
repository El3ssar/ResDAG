# Forecasting chaotic systems

Use a long series, `prepare_esn_data`, a premade model such as `ott_esn`, and score
on `val`.

```python
import torch
from resdag import ott_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data


def lorenz63(n_steps: int = 30_000, dt: float = 0.02, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    xyz = torch.zeros(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 0.0, 0.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        d = torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[t] = xyz[t - 1] + dt * d
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    return xyz.unsqueeze(0)


data = lorenz63()

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=3_000,
    train_steps=18_000,
    val_steps=6_000,
    discard_steps=3_000,
    normalize=True,
    norm_method="minmax",
)

model = ott_esn(
    reservoir_size=800,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    topology="erdos_renyi",
)

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

model.reset_reservoirs()
pred = model.forecast(f_warmup, horizon=val.shape[1])
print(torch.mean((pred - val) ** 2).item())
```

Tune `warmup_steps`, `train_steps`, reservoir size, and `readout_alpha`. For search
over hyperparameters use [HPO](hyperparameter-optimization.md) with horizon-oriented
losses (`efh`, `forecast_horizon`, etc.).
