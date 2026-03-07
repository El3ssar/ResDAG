# Quickstart

This guide walks you through a complete ESN workflow — from raw time-series data to trained model and forecasts — in about 5 minutes.

---

## 1. Generate Data

We'll use the Lorenz attractor, a classic chaotic system often used to benchmark reservoir computers:

```python
import torch
import numpy as np

def lorenz(n_steps=10000, dt=0.01, sigma=10, rho=28, beta=8/3, seed=42):
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_steps, 3))
    xyz[0] = rng.standard_normal(3)
    for i in range(n_steps - 1):
        x, y, z = xyz[i]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xyz[i+1] = xyz[i] + dt * np.array([dx, dy, dz])
    return torch.tensor(xyz, dtype=torch.float32)

data = lorenz(n_steps=12000)  # (12000, 3)
data = data.unsqueeze(0)      # (1, 12000, 3) — add batch dim
```

Split into phases:

```python
warmup  = data[:, :1000,  :]   # reservoir synchronization  (1, 1000,  3)
train   = data[:, 1000:6000, :] # readout training          (1, 5000,  3)
target  = data[:, 1001:6001, :] # one-step-ahead targets    (1, 5000,  3)
f_warm  = data[:, 6000:7000, :] # forecast warmup           (1, 1000,  3)
val     = data[:, 7000:8000, :] # validation                (1, 1000,  3)
```

---

## 2. Choose a Model

=== "Premade (recommended)"
    ```python
    from resdag.models import ott_esn

    # Ott's architecture: best for chaotic systems
    model = ott_esn(
        reservoir_size=500,
        feedback_size=3,
        output_size=3,
        spectral_radius=0.9,
        topology="erdos_renyi",
    )
    ```

=== "Custom architecture"
    ```python
    import pytorch_symbolic as ps
    from resdag import ESNModel, ESNLayer, CGReadoutLayer

    inp = ps.Input((100, 3))           # (seq_len, features)
    reservoir = ESNLayer(
        reservoir_size=500,
        feedback_size=3,
        spectral_radius=0.9,
        topology="erdos_renyi",
    )(inp)
    readout = CGReadoutLayer(
        in_features=500,
        out_features=3,
        alpha=1e-6,
        name="output",
    )(reservoir)
    model = ESNModel(inp, readout)
    ```

---

## 3. Train

No gradient descent — `ESNTrainer` fits the readout algebraically:

```python
from resdag.training import ESNTrainer

trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),       # synchronize reservoir states
    train_inputs=(train,),         # fit readout on these states
    targets={"output": target},    # ground-truth targets
)

print("Training complete!")
```

The trainer:

1. Resets reservoir state
2. Runs teacher-forced warmup to synchronize states
3. Runs a single forward pass with pre-hooks that fit each readout via CG ridge regression

---

## 4. Forecast

```python
predictions = model.forecast(f_warm, horizon=1000)
print(predictions.shape)  # torch.Size([1, 1000, 3])
```

Measure forecast quality:

```python
mse = torch.mean((predictions - val) ** 2)
print(f"Validation MSE: {mse.item():.6f}")
```

---

## 5. Visualize (optional)

```python
import matplotlib.pyplot as plt

pred = predictions[0].numpy()  # (1000, 3)
true = val[0].numpy()

fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
labels = ["x", "y", "z"]
for i, (ax, lab) in enumerate(zip(axes, labels)):
    ax.plot(true[:, i], label="True", alpha=0.7)
    ax.plot(pred[:, i], label="Predicted", linestyle="--")
    ax.set_ylabel(lab)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Timestep")
fig.suptitle("Lorenz Attractor Forecast")
plt.tight_layout()
plt.savefig("lorenz_forecast.png", dpi=150)
```

---

## 6. Model Architecture

Print a summary or visualize:

```python
model.summary()       # text summary
model.plot_model()    # graphviz visualization (requires graphviz)
```

---

## Complete Script

```python
import torch
import numpy as np
from resdag.models import ott_esn
from resdag.training import ESNTrainer


def lorenz(n_steps=12000, dt=0.01, seed=42):
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_steps, 3))
    xyz[0] = rng.standard_normal(3)
    for i in range(n_steps - 1):
        x, y, z = xyz[i]
        dxyz = np.array([10*(y-x), x*(28-z)-y, x*y - (8/3)*z])
        xyz[i+1] = xyz[i] + dt * dxyz
    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)


torch.manual_seed(0)
data = lorenz()

warmup = data[:, :1000,  :]
train  = data[:, 1000:6000, :]
target = data[:, 1001:6001, :]
f_warm = data[:, 6000:7000, :]
val    = data[:, 7000:8000, :]

model   = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
trainer = ESNTrainer(model)
trainer.fit(warmup_inputs=(warmup,), train_inputs=(train,), targets={"output": target})

preds = model.forecast(f_warm, horizon=1000)
mse   = torch.mean((preds - val) ** 2).item()
print(f"Validation MSE: {mse:.6f}")
```

---

## Next Steps

- Read [Core Concepts](concepts.md) to understand what's happening inside the reservoir
- Explore [ESN Layer](../guide/esn-layer.md) for all configuration options
- Try [NG-RC](../guide/ngrc.md) for a weight-free alternative
- Set up [HPO](../hpo/overview.md) to tune hyperparameters automatically
