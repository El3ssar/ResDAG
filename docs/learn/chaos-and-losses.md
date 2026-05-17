# Chaos & HPO loss functions

!!! info "Why this exists"
    Chaotic attractors amplify small errors exponentially. **Plain MSE** over long
    horizons punishes models uniformly and correlates poorly with *how long* a forecast
    stays useful. ResDAG ships five Optuna-ready losses that target **valid forecast
    length** or **Lyapunov-aware** weighting.

## The forecast-horizon problem

Define per-step error $e_t$ (e.g. RMSE across batch and dimensions). For chaos:

- $e_t$ grows roughly as $e_t \sim e_0 \cdot e^{t/\tau}$ (Lyapunov time $\tau$).
- A model that is good for 20 steps and bad for 200 should beat one that is mediocre
  everywhere — but MSE averages both regimes equally.

HPO losses therefore emphasize **early accurate steps** or **expected horizon** under a
threshold.

## Registered losses (`LOSSES`)

| Key | Name | Idea | Returns |
|-----|------|------|---------|
| `efh` | Expected Forecast Horizon | Soft threshold + cumulative survival $\sum_t \prod_{i\le t} g_i$ | $-\mathbb{E}[\text{horizon}]$ (minimize) |
| `forecast_horizon` | Hard contiguous horizon | Count steps while $e_t < \theta$ from $t=0$ | $-\log(\text{length})$ |
| `soft_horizon` | Hill-gated survival | $g_t = 1/(1+(e_t/\theta)^n)$, cumprod | $-\sum_t \prod g$ |
| `lyapunov` | Lyapunov-weighted error | Weights $\propto e^{-t/\tau}$ on $e_t$ | weighted mean error |
| `standard` | Baseline | Mean of geometric mean errors over time | scalar error |

Use in `run_hpo`:

```python
from resdag.hpo import run_hpo

study = run_hpo(..., loss="efh", loss_params={"threshold": 0.2})
```

### `efh` (recommended for chaos)

Soft indicator $\sigma((\theta - e_t)/\text{softness})$ at each step, cumulative product
for survival, sum for expected horizon. Differentiable proxy — good for TPE search.

### `forecast_horizon`

Strict: first time $e_t \ge \theta$ ends the valid run. Non-differentiable but interpretable.

### `soft_horizon`

Hill function sharpness `n` controls gate; harsher than sigmoid `efh`, still smoother than hard horizon.

### `lyapunov`

Does not maximize horizon directly — down-weights late timesteps where error growth is
expected: $w_t = e^{-t/\tau}$.

### `standard`

Use when dynamics are stable or you want a simple geometric-mean error baseline.

## Custom losses

Any callable `loss(y_true, y_pred, /, **kwargs) -> float` matching `LossProtocol` can be
passed as `loss=my_fn` or monitored via `monitor_losses`.

## See also

- [HPO losses reference](../reference/hpo/losses.md)
- [Hyperparameter optimization guide](../guides/hyperparameter-optimization.md)
- [Chaotic systems guide](../guides/chaotic-systems.md)
