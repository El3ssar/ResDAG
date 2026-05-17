# Forecasting chaotic systems

When to use this: Lorenz, Rössler, or any sensitive dependence on initial conditions.
Use `ott_esn` and enough warmup before trusting long horizons.

## Full script

```python
import torch
from resdag import ott_esn
from resdag.training import ESNTrainer


def lorenz63(n_steps=2500, dt=0.02, seed=42):
    torch.manual_seed(seed)
    xyz = torch.zeros(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 0.0, 0.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        d = torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[t] = xyz[t - 1] + dt * d
    return ((xyz - xyz.mean(0)) / xyz.std(0)).unsqueeze(0)


data = lorenz63()
warmup = data[:, :300, :]
train = data[:, 300:1300, :]
target = data[:, 301:1301, :]
f_warmup = data[:, 1300:1500, :]
val = data[:, 1500:1800, :]

model = ott_esn(
    reservoir_size=500,
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
print("MSE:", torch.mean((pred - val) ** 2).item())
```

## Gotchas

- **Warmup length** — too short and the reservoir never sits on the attractor.
- **Spectral radius** — try $0.7$–$1.0$; measure with [`esp_index`](../reference/utils/states.md).
- **Readout `alpha`** — scan `1e-8` … `1e-3` if validation error is flat.
- **Evaluation** — use horizon-based losses for HPO ([chaos losses](../learn/chaos-and-losses.md)), not raw MSE alone.

## See also

- [Lorenz walkthrough](../getting-started/lorenz-walkthrough.md)
- [Coupled ensembles](coupled-ensembles.md)
- [Hyperparameter optimization](hyperparameter-optimization.md)
