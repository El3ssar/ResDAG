---
description: The projects ResDAG builds on and pairs with — pytorch_symbolic and TSDynamics.
---

<span class="nb-kicker">Project</span>

# Ecosystem

## pytorch_symbolic

ResDAG's composition layer is built on
[pytorch_symbolic](https://github.com/gahaalt/pytorch-symbolic) by Szymon
Mikler. It provides the symbolic tracing that turns layer calls on
placeholder tensors into executable model graphs — the mechanism behind
`Input`, the functional API, and the model class. Without it, the central
premise of this library — reservoir models composed as arbitrary DAGs —
would not exist in its current form. Credit where it is due.

## TSDynamics

[TSDynamics](https://github.com/El3ssar/TSDynamics) is a companion library
of dynamical systems by the same author: a catalog of chaotic ODEs, maps,
and delay systems with a uniform integration interface, plus Lyapunov
analysis utilities.

TSDynamics generates trajectories and ResDAG learns to forecast them. Install
`pip install "resdag[dynamics]"` for the tested optional integration. The example
below uses TSDynamics 6's `systems` and `analysis` namespaces. No TSDynamics
import is required for the rest of ResDAG.

## Generate, train and forecast

```python
import numpy as np
import torch
import tsdynamics as ts
import resdag as rd
from resdag.metrics import largest_lyapunov, valid_prediction_time
from resdag.utils import prepare_esn_data

system = ts.systems.Lorenz()
trajectory = system.run(final_time=140.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(20.0)
# Keep the recorded time axis: dt is the output sample spacing.
spacing = np.diff(trajectory.t)
assert np.allclose(spacing, spacing[0])
dt = float(spacing[0])
series = torch.tensor(trajectory.y, dtype=torch.float64).unsqueeze(0)

warmup_steps, train_steps, val_steps = 500, 4000, 1000
history = series[:, :warmup_steps + train_steps + 1]
mean = history.mean(dim=(0, 1), keepdim=True)
scale = history.std(dim=(0, 1), keepdim=True, correction=0).clamp_min(1e-8)
normalized = (series - mean) / scale  # training statistics apply to future data too
warmup, train, target, forecast_warmup, truth = prepare_esn_data(
    normalized, warmup_steps=warmup_steps, train_steps=train_steps,
    val_steps=val_steps,
)
model = rd.classic_esn(
    reservoir_size=200, feedback_size=3, output_size=3,
    seed=7, dtype=torch.float64,
)
rd.ESNTrainer(model).fit((warmup,), (train,), {"output": target})
prediction = model.forecast(forecast_warmup, horizon=truth.shape[1])
# First forecast sample and first truth sample describe the same next timestep.
score = valid_prediction_time(
    prediction, truth, units="lyapunov", dt=dt,
    lambda_max=largest_lyapunov("lorenz"),
)
physical_prediction = prediction * scale + mean
print(f"Valid prediction time: {score:.2f} Lyapunov times")
```

The canonical Lorenz exponent above assumes the default Lorenz parameters.
For a customized continuous system, compute its exponent explicitly:

```python
lambda_max = largest_lyapunov(system, dt=dt, final_time=500.0)
# Internally this calls ts.analysis.lyapunov_spectrum(system, ...).
```

Increase integration/averaging time and check convergence when interpreting
that estimate. Data-based exponents are also available through
`largest_lyapunov(series, dt=dt, from_="data")`; they are noisy diagnostics.

ResDAG currently consumes the trajectory's numeric samples through this explicit
conversion. It does not automatically carry the time axis or system metadata
through training. Preserve `trajectory.t`, its parameters and preprocessing
statistics with experiment results. Resample irregularly spaced observations
before using a fixed-step reservoir; the `dt` used for Lyapunov conversion must
match the observations, including any stride or downsampling.

## See also

- [Citation](citation.md) — how to cite ResDAG and the methods it implements
- [Contributing](contributing.md) — development setup
