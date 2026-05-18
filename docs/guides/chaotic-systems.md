# Forecasting chaotic systems

The recipe is short: a long series, `prepare_esn_data` for splits, a
premade model — `ott_esn` works well on most low-dimensional chaos — and
scoring on `val`.

<figure markdown>
  ![Lorenz timeseries](../assets/figures/signal_lorenz.png){ width="720" }
  <figcaption>Lorenz-63 — 3 000 normalised timesteps. The three
  coordinates show the characteristic non-periodic, bounded
  trajectory.</figcaption>
</figure>

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

## Forecast horizon

Chaotic systems eventually diverge from any predictor. The useful question
is *how long does the forecast stay close before it does?* Component-wise
overlay of an `ott_esn` autoregressive forecast on the held-out truth:

<figure markdown>
  ![Lorenz prediction overlay](../assets/figures/predict_lorenz.png){ width="720" }
  <figcaption>Grey: held-out truth. Amber: ESN forecast. The model tracks
  the attractor's dynamics for hundreds of steps before phase
  drift dominates.</figcaption>
</figure>

## Tuning notes

- **Spectral radius** controls memory and stability. Start at `0.9`; for
  systems with longer memory, push to `0.95–1.05` (just above 1 is
  sometimes useful for the "edge of chaos" regime).
- **Reservoir size** is the biggest lever on accuracy and the biggest
  driver of cost. 500–1000 is a sensible range for Lorenz.
- **Warmup length** must be long enough for the reservoir state to forget
  its zero-init. A few hundred to a few thousand steps depending on
  `leak_rate`.
- **`readout_alpha`** controls the readout's regularization. Smaller
  values fit harder; larger values stabilise long-horizon forecasts.
  Sweep on a log scale.

Search systematically with the
[HPO guide](hyperparameter-optimization.md) and a horizon-oriented loss
(`efh`, `forecast_horizon`).
