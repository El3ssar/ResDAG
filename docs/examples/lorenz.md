# Lorenz Attractor

The Lorenz system is the canonical benchmark for reservoir computing — a 3D chaotic attractor with a positive Lyapunov exponent of ≈0.906 (in units of the integration time step).

This example walks through a complete forecast pipeline, from data generation through evaluation.

---

## The Lorenz System

\[
\dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y, \quad \dot{z} = xy - \beta z
\]

Standard parameters: \(\sigma = 10\), \(\rho = 28\), \(\beta = 8/3\). The Lyapunov time \(\Lambda = 1/\lambda_1 \approx 1.1\) time units (where \(dt = 0.01\)).

---

## Generate Data

```python
import numpy as np
import torch


def lorenz(
    n_steps: int = 20000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
    seed: int = 0,
) -> torch.Tensor:
    """Integrate Lorenz with RK4."""
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_steps, 3))
    xyz[0] = rng.standard_normal(3) * 0.1

    def deriv(state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    for i in range(n_steps - 1):
        k1 = deriv(xyz[i])
        k2 = deriv(xyz[i] + dt / 2 * k1)
        k3 = deriv(xyz[i] + dt / 2 * k2)
        k4 = deriv(xyz[i] + dt * k3)
        xyz[i + 1] = xyz[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)  # (1, T, 3)


torch.manual_seed(42)
data = lorenz(n_steps=15000)

# Split phases
warmup   = data[:, :1000,    :]   # (1, 1000,  3) — state synchronization
train    = data[:, 1000:7000, :]  # (1, 6000,  3) — readout training
target   = data[:, 1001:7001, :]  # (1, 6000,  3) — one-step-ahead targets
f_warmup = data[:, 7000:8000, :]  # (1, 1000,  3) — forecast warmup
val      = data[:, 8000:9000, :]  # (1, 1000,  3) — validation ground truth

print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
```

---

## Build and Train

```python
from resdag.models import ott_esn
from resdag.training import ESNTrainer

model = ott_esn(
    reservoir_size=500,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    topology="erdos_renyi",
    readout_alpha=1e-6,
)

trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

print("Training complete!")
```

---

## Evaluate

```python
# Forecast 1000 steps (10 Lyapunov times)
predictions = model.forecast(f_warmup, horizon=1000)

# RMSE
rmse = torch.sqrt(torch.mean((predictions - val) ** 2)).item()
print(f"RMSE: {rmse:.4f}")

# Normalized RMSE (relative to data std)
data_std = data.std().item()
nrmse = rmse / data_std
print(f"Normalized RMSE: {nrmse:.4f}")

# Valid forecast horizon (error < 0.4 * std)
errors = torch.sqrt(torch.mean((predictions - val) ** 2, dim=-1))  # (1, 1000)
threshold = 0.4 * data_std
valid_mask = (errors[0] < threshold)
if valid_mask.any():
    horizon_length = valid_mask.float().cumsum(0).argmax().item() + 1
else:
    horizon_length = 0

lyapunov_time = 0.906 * 0.01  # λ₁ * dt
horizon_lt = horizon_length * 0.01 / (1 / 0.906)
print(f"Valid horizon: {horizon_length} steps ({horizon_lt:.1f} Lyapunov times)")
```

---

## Visualize

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

pred = predictions[0].numpy()   # (1000, 3)
true = val[0].numpy()

fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# --- Time series ---
labels = ["x", "y", "z"]
colors = ["#3949AB", "#00BCD4", "#00897B"]

for i, (lab, col) in enumerate(zip(labels, colors)):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(true[:, i], label="True", color=col, alpha=0.8, linewidth=1.2)
    ax.plot(pred[:, i], label="Predicted", color=col, linestyle="--",
            alpha=0.9, linewidth=1.2)
    ax.axvline(horizon_length, color="red", linestyle=":", alpha=0.5,
               label="Valid horizon" if i == 0 else "")
    ax.set_ylabel(lab, fontsize=12)
    ax.set_xlim(0, 1000)
    if i == 0:
        ax.legend(loc="upper right", fontsize=9)
    if i == 2:
        ax.set_xlabel("Timestep", fontsize=11)

# --- Phase portrait ---
ax3d = fig.add_subplot(gs[:, 1], projection="3d")
ax3d.plot(*true.T, alpha=0.4, linewidth=0.7, color="#3949AB", label="True")
ax3d.plot(*pred.T, alpha=0.5, linewidth=0.7, color="#00BCD4", linestyle="--", label="Predicted")
ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
ax3d.set_title("Phase Portrait", fontsize=12)
ax3d.legend(fontsize=9)

fig.suptitle(
    f"Lorenz Attractor — ott_esn (N=500)\n"
    f"RMSE={rmse:.4f}  |  Horizon={horizon_length} steps ({horizon_lt:.1f} LT)",
    fontsize=13,
)
plt.savefig("lorenz_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Expected Results

With the default configuration (`reservoir_size=500`, `spectral_radius=0.9`, `alpha=1e-6`):

| Metric | Expected Value |
|---|---|
| Valid Horizon | 500–1200 steps |
| Lyapunov Times | 4.5–11 |
| RMSE | 0.5–2.0 |
| Training Time | < 2 seconds (CPU) |

!!! tip "Improve results with HPO"
    Use `run_hpo()` to automatically find the best `reservoir_size`, `spectral_radius`,
    `alpha`, and `topology`. See the [HPO example](hpo-example.md).

---

## Variants

### Custom Topology

```python
model = ott_esn(
    reservoir_size=500,
    feedback_size=3,
    output_size=3,
    topology=("watts_strogatz", {"k": 6, "p": 0.1}),
    spectral_radius=0.95,
)
```

### Larger Reservoir

```python
model = ott_esn(
    reservoir_size=2000,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.9,
    readout_alpha=1e-8,
)
```

### With Leaky Integration

```python
model = ott_esn(
    reservoir_size=500,
    feedback_size=3,
    output_size=3,
    spectral_radius=1.1,  # higher SR compensated by leaking
    leak_rate=0.5,
)
```
