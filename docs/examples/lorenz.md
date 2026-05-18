# Lorenz-63

Forecast the canonical chaotic system with `ott_esn`. The accuracy
metric for chaotic series is *how many Lyapunov times the forecast
tracks before drifting* — not RMSE over the full horizon, which is
meaningless once the trajectories have diverged.

## The system

Lorenz-63 integrates three coupled ODEs:

$$
\dot{x} = \sigma(y - x), \quad
\dot{y} = x(\rho - z) - y, \quad
\dot{z} = xy - \beta z
$$

with the chaotic parameters $\sigma = 10$, $\rho = 28$, $\beta = 8/3$
and integrator step $\Delta t = 0.02$. The leading Lyapunov exponent is
$\lambda_1 \approx 0.9$, giving a Lyapunov time $T_\lambda \approx 1.1 /
\lambda_1 \approx 55$ integration steps.

<figure markdown>
  ![Lorenz phase portrait](../assets/figures/signal_lorenz_phase.png){ width="460" }
  <figcaption>Phase portrait — the "butterfly" attractor.</figcaption>
</figure>

<figure markdown>
  ![Lorenz timeseries](../assets/figures/signal_lorenz.png){ width="720" }
  <figcaption>Three normalised state coordinates over 3 000
  timesteps.</figcaption>
</figure>

## Generate

```python
import torch

def lorenz63(n_steps: int = 40_000, dt: float = 0.02, seed: int = 42):
    torch.manual_seed(seed)
    xyz = torch.zeros(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 1.0, 1.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        d = torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[t] = xyz[t - 1] + dt * d
    return ((xyz - xyz.mean(0)) / xyz.std(0)).unsqueeze(0)
```

## Architecture

`ott_esn` augments reservoir states by squaring their even-indexed units
and concatenates them with the input before the readout:

<figure markdown>
  ![ott_esn architecture](../assets/figures/arch_ott_esn.svg){ width="720" }
</figure>

## Train and forecast

```python
from resdag import ott_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

data = lorenz63()
torch.manual_seed(1)            # controls reservoir initialisation

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=2_000,
    train_steps=20_000,
    val_steps=1_500,
    discard_steps=3_000,
    normalize=True,
    norm_method="minmax",
)

model = ott_esn(
    reservoir_size=1_500,
    feedback_size=3,
    output_size=3,
    spectral_radius=1.0,
    leak_rate=1.0,
    readout_alpha=1e-7,
)
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
model.reset_reservoirs()
pred = model.forecast(f_warmup, horizon=val.shape[1])
```

## Forecast quality

The held-out validation segment is purely autoregressive — no teacher
forcing once `f_warmup` ends. The model tracks the truth for ~292 steps
(≈ 5.3 Lyapunov times) before phase drift becomes visible:

<figure markdown>
  ![Lorenz forecast](../assets/figures/predict_lorenz.png){ width="720" }
  <figcaption>Timeseries overlay. The dashed indigo line marks the
  292-step tracking horizon (running RMSE first crosses 0.5).</figcaption>
</figure>

In phase space, the prediction traces the **correct attractor** — both
lobes, correct geometry, correct amplitude — for the entire accurate
window:

<figure markdown>
  ![Phase portrait comparison](../assets/figures/predict_lorenz_phase.png){ width="720" }
  <figcaption>Forecast restricted to its accurate window plus a small
  margin. The ESN has learned the geometry of the attractor, not just
  short-term pointwise dynamics.</figcaption>
</figure>

## Tuning

- **Reservoir size** is the biggest lever. 1 500 is a good docs/CI
  trade-off; Pathak 2018 use 5 000 for state-of-the-art results.
- **Spectral radius** = 1.0 (edge of chaos) is the sweet spot for this
  setup. Sweep `[0.85, 1.1]` to see the trade-off between memory and
  stability.
- **Topology**: dense uniform random (the default) is robust. Sparse
  graph topologies sometimes win — see [Graph topologies](../guides/topologies.md)
  for a head-to-head comparison.
- **HPO** on these knobs typically adds another 50–100 tracked steps.
  See the [HPO guide](../guides/hyperparameter-optimization.md).
