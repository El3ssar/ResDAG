# Lorenz walkthrough

Forecast the Lorenz-63 attractor with Ott's state-augmented ESN — no data files required.

## Generate data

Euler integration of Lorenz-63, normalized per channel:

```python
import torch
from resdag import ott_esn
from resdag.training import ESNTrainer


def lorenz63(n_steps: int = 2500, dt: float = 0.02, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    xyz = torch.zeros(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 0.0, 0.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        d = torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[t] = xyz[t - 1] + dt * d
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    return xyz.unsqueeze(0)  # (1, T, 3)


data = lorenz63()
warmup = data[:, :300, :]
train = data[:, 300:1300, :]
target = data[:, 301:1301, :]
f_warmup = data[:, 1300:1500, :]
ground_truth = data[:, 1500:1800, :]
```

## Model

`ott_esn` adds selective squaring on even reservoir units — strong default for chaos
([premade models](../reference/models.md)).

```python
model = ott_esn(
    reservoir_size=500,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    topology="erdos_renyi",
)
```

## Train and forecast

```python
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

pred = model.forecast(f_warmup, horizon=ground_truth.shape[1])
mse = torch.mean((pred - ground_truth) ** 2).item()
print(f"Validation MSE: {mse:.6f}")
```

## Tips

- Increase `reservoir_size` or warmup length if error is high.
- For production workflows use [`prepare_esn_data`](../reference/utils/data.md)
  and the [data preparation guide](../guides/data-preparation.md).
- Hyperparameter search: [HPO guide](../guides/hyperparameter-optimization.md).

## Next

[Learn](../learn/index.md) for theory, or [chaotic systems guide](../guides/chaotic-systems.md).
