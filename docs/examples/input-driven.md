# Input-Driven Systems

This example shows how to build, train, and forecast with ESN models that receive an **exogenous driving signal** in addition to their own feedback.

Input-driven models are useful for:
- Systems with known future inputs (weather, control signals, schedules)
- Conditional forecasting
- Systems with multiple interacting variables

---

## What Changes vs Feedback-Only

For a feedback-only model:
- Single input: the model's own output is fed back
- Training: `warmup_inputs=(feedback,)`, `train_inputs=(feedback,)`
- Forecasting: `model.forecast(warmup_feedback, horizon=N)`

For an input-driven model:
- Two inputs: feedback + external driver
- Training: `warmup_inputs=(feedback, driver)`, `train_inputs=(feedback, driver)`
- Forecasting: `model.forecast(w_fb, w_driver, horizon=N, forecast_drivers=(future_driver,))`

---

## Example: Forced Van der Pol Oscillator

A Van der Pol oscillator driven by a sinusoidal forcing signal:

```python
import numpy as np
import torch

def vanderpol_forced(
    n_steps=5000,
    dt=0.05,
    mu=1.0,
    omega=0.8,
    A=1.2,
    seed=0,
):
    """Forced Van der Pol oscillator."""
    rng = np.random.default_rng(seed)
    state = np.zeros((n_steps, 2))  # [x, y] — oscillator state
    force = np.zeros((n_steps, 1))  # forcing signal

    state[0] = rng.standard_normal(2) * 0.5

    for i in range(n_steps - 1):
        x, y = state[i]
        t = i * dt
        f = A * np.cos(omega * t)
        force[i] = f

        dx = y
        dy = mu * (1 - x**2) * y - x + f
        state[i+1] = state[i] + dt * np.array([dx, dy])

    force[-1] = A * np.cos(omega * (n_steps - 1) * dt)

    return (
        torch.tensor(state, dtype=torch.float32).unsqueeze(0),  # (1, T, 2)
        torch.tensor(force, dtype=torch.float32).unsqueeze(0),  # (1, T, 1)
    )


state, force = vanderpol_forced(n_steps=6000)

# One-step-ahead prediction
warmup_len = 200
train_len  = 2000
f_warm_len = 300
val_len    = 1000

warmup_s = state[:, :warmup_len, :]
warmup_f = force[:, :warmup_len, :]

train_s  = state[:, warmup_len:warmup_len+train_len, :]
train_f  = force[:, warmup_len:warmup_len+train_len, :]

target_s = state[:, warmup_len+1:warmup_len+train_len+1, :]

fw_start = warmup_len + train_len
fw_end   = fw_start + f_warm_len
fwarm_s  = state[:, fw_start:fw_end, :]
fwarm_f  = force[:, fw_start:fw_end, :]

val_start = fw_end
val_end   = val_start + val_len
val_s     = state[:, val_start:val_end, :]
future_f  = force[:, val_start:val_end, :]  # known future forcing
```

---

## Build the Input-Driven Model

```python
import pytorch_symbolic as ps
from resdag import ESNModel, ESNLayer, CGReadoutLayer

feedback_inp = ps.Input((100, 2))  # oscillator state (dim=2)
driver_inp   = ps.Input((100, 1))  # forcing signal (dim=1)

reservoir = ESNLayer(
    reservoir_size=300,
    feedback_size=2,       # feedback = oscillator state
    input_size=1,          # driving input = forcing signal
    spectral_radius=0.9,
)(feedback_inp, driver_inp)

readout = CGReadoutLayer(300, 2, alpha=1e-6, name="output")(reservoir)
model   = ESNModel([feedback_inp, driver_inp], readout)

model.summary()
```

---

## Train

```python
from resdag.training import ESNTrainer

trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup_s, warmup_f),   # (feedback, driver) — both needed
    train_inputs=(train_s, train_f),
    targets={"output": target_s},
)
print("Training complete!")
```

---

## Forecast with Known Future Drivers

When the future forcing is known (e.g., a scheduled signal or weather forecast):

```python
# Forecast 1000 steps with known future forcing
predictions = model.forecast(
    fwarm_s,                              # warmup feedback
    fwarm_f,                              # warmup driver
    horizon=val_len,
    forecast_drivers=(future_f,),         # (batch, horizon, 1) — known future
)

rmse = torch.sqrt(torch.mean((predictions - val_s) ** 2)).item()
print(f"Forecast RMSE: {rmse:.4f}")
```

!!! important "`forecast_drivers` shape"
    Each tensor in `forecast_drivers` must have shape `(batch, horizon, feat)`.
    The order must match the order of driving inputs in the model (after the feedback input).

---

## Visualize

```python
import matplotlib.pyplot as plt

pred = predictions[0].numpy()
true = val_s[0].numpy()
drv  = future_f[0].numpy()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(true[:, 0], label="True x", alpha=0.7, color="#3949AB")
axes[0].plot(pred[:, 0], label="Pred x", linestyle="--", alpha=0.9, color="#3949AB")
axes[0].set_ylabel("x (state)")
axes[0].legend()

axes[1].plot(true[:, 1], label="True y", alpha=0.7, color="#00BCD4")
axes[1].plot(pred[:, 1], label="Pred y", linestyle="--", alpha=0.9, color="#00BCD4")
axes[1].set_ylabel("y (state)")
axes[1].legend()

axes[2].plot(drv[:, 0], label="Forcing f(t)", color="#E91E63", alpha=0.8)
axes[2].set_ylabel("f(t)")
axes[2].set_xlabel("Timestep")
axes[2].legend()

plt.suptitle(f"Forced Van der Pol — RMSE={rmse:.4f}")
plt.tight_layout()
plt.savefig("vanderpol_forecast.png", dpi=150)
```

---

## Multi-Driver Example

For systems with multiple external signals:

```python
fb_inp  = ps.Input((100, 3))
drv1    = ps.Input((100, 2))
drv2    = ps.Input((100, 1))

res = ESNLayer(
    reservoir_size=400,
    feedback_size=3,
    input_size=3,          # total driver dim = 2 + 1 = 3
)(fb_inp, Concatenate()(drv1, drv2))

out   = CGReadoutLayer(400, 3, name="output")(res)
model = ESNModel([fb_inp, drv1, drv2], out)

# Training
trainer.fit(
    warmup_inputs=(w_fb, w_d1, w_d2),
    train_inputs=(t_fb, t_d1, t_d2),
    targets={"output": t_targets},
)

# Forecasting
preds = model.forecast(
    w_fb, w_d1, w_d2,
    horizon=500,
    forecast_drivers=(future_d1, future_d2),
)
```
