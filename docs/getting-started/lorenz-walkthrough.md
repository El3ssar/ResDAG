# Lorenz walkthrough

Lorenz-63 integration, splits via `prepare_esn_data`, `ott_esn`, forecast on `val`.

```python
import torch
from resdag import ott_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data


def lorenz63(n_steps: int = 25_000, dt: float = 0.02, seed: int = 42) -> torch.Tensor:
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
    warmup_steps=2_000,
    train_steps=15_000,
    val_steps=5_000,
    discard_steps=2_000,
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
print("val MSE:", torch.mean((pred - val) ** 2).item())
```

## Notes

- Increase `train_steps`, `warmup_steps`, or `reservoir_size` if error remains high.
- Load external series with [`load_file`](../reference/utils/data.md) then pass the
  tensor to `prepare_esn_data`.
- HPO: [hyperparameter optimization guide](../guides/hyperparameter-optimization.md).

## Next

[Chaotic systems guide](../guides/chaotic-systems.md)
