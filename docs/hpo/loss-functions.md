# HPO Loss Functions

resdag provides five specialist loss functions for evaluating ESN forecasting performance. These are used as the optimization objective in `run_hpo()`.

!!! note "Always available"
    These loss functions can be used directly — no Optuna required.
    ```python
    from resdag.hpo.losses import efh, horizon, lyap, standard, discounted
    ```

---

## Choosing a Loss

| Loss | Best For | Notes |
|---|---|---|
| `"efh"` | Chaotic systems (default) | Robust to random seeding |
| `"horizon"` | Deterministic systems | Sensitive to threshold |
| `"lyap"` | Comparing with Lyapunov exponent | Requires known Lyapunov time |
| `"standard"` | Smooth signal tasks | Classical MSE-based |
| `"discounted"` | Near-term accuracy matters | Exponential time weighting |

---

## `efh` — Expected Forecast Horizon

**Expected Forecast Horizon** is the recommended loss for chaotic systems. It averages the valid forecast horizon over multiple random initial reservoir states, making it robust to the sensitivity of chaotic systems to initial conditions.

```python
study = run_hpo(..., loss="efh", loss_params={"threshold": 0.4, "n_samples": 5})
```

| Param | Default | Description |
|---|---|---|
| `threshold` | `0.4` | Normalized error threshold below which forecast is "valid" |
| `n_samples` | `5` | Number of random initial states to average over |

**How it works:**

1. For each of `n_samples` random reservoir initializations, compute the forecast
2. Find the first timestep where normalized error exceeds `threshold`
3. Return the **negative mean** of these horizon lengths (negative because Optuna minimizes)

Higher EFH → longer valid predictions → better model.

---

## `horizon` — Forecast Horizon

Computes the length of the longest **contiguous** window where the normalized error stays below a threshold.

```python
study = run_hpo(..., loss="horizon", loss_params={"threshold": 0.4})
```

| Param | Default | Description |
|---|---|---|
| `threshold` | `0.4` | Error threshold |

**Difference from EFH**: Does not average over multiple random states. Faster but more sensitive to lucky/unlucky initialization.

---

## `lyap` — Lyapunov-Weighted Loss

Weights the forecast error exponentially according to the system's Lyapunov exponent, measuring accuracy in Lyapunov time units.

```python
study = run_hpo(
    ...,
    loss="lyap",
    loss_params={"lyapunov_time": 0.906}  # Lorenz attractor (1/λ₁)
)
```

| Param | Default | Description |
|---|---|---|
| `lyapunov_time` | `1.0` | Lyapunov time (1/largest Lyapunov exponent) |
| `threshold` | `0.4` | Error threshold for counting valid steps |

**Best for**: Research contexts where you want to report results in Lyapunov time units (the standard in chaos prediction literature).

---

## `standard` — Standard Loss

Geometric mean of the per-timestep RMSE across the forecast horizon. A classical metric.

```python
study = run_hpo(..., loss="standard")
```

No parameters. Returns the mean normalized RMSE over the validation horizon.

**Best for**: Smooth, non-chaotic time series where you care about average prediction accuracy rather than valid horizon length.

---

## `discounted` — Discounted RMSE

Time-discounted RMSE: earlier timesteps are weighted more heavily using exponential decay.

```python
study = run_hpo(
    ...,
    loss="discounted",
    loss_params={"half_life": 50}  # error weight halves every 50 steps
)
```

| Param | Default | Description |
|---|---|---|
| `half_life` | `50` | Steps after which error weight halves |

**Best for**: Applications where short-term accuracy is more important than long-term.

---

## Monitor Losses

You can compute additional losses (not optimized, just logged) for each trial:

```python
study = run_hpo(
    ...,
    loss="efh",
    monitor_losses=["horizon", "standard"],
    monitor_params={
        "horizon": {"threshold": 0.4},
        "standard": {},
    },
)

# Access monitored values in trial user_attrs
for trial in study.trials:
    print(f"Trial {trial.number}: EFH={trial.value:.2f}, "
          f"Horizon={trial.user_attrs['horizon']:.2f}, "
          f"Standard={trial.user_attrs['standard']:.4f}")
```

---

## Using Loss Functions Directly

```python
import torch
from resdag.hpo.losses import efh, horizon, standard

predictions = torch.randn(1, 1000, 3)   # (batch, horizon, feat)
ground_truth = torch.randn(1, 1000, 3)

# Forecast horizon
h = horizon(predictions, ground_truth, threshold=0.4)
print(f"Valid horizon: {h} steps")

# Standard loss
s = standard(predictions, ground_truth)
print(f"Standard loss: {s:.4f}")

# EFH (requires a model for multiple random initializations)
# Use via run_hpo for this one
```
