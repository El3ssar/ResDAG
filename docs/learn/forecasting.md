# Autoregressive forecasting

!!! info "Why this exists"
    Many RC tasks predict the future of a dynamical system from its past. After training,
    the model must run **closed-loop**: outputs become the next feedback input. ResDAG
    packages this as a two-phase `ESNModel.forecast()` — warmup, then autoregression.

## Two phases

```mermaid
sequenceDiagram
    participant User
    participant Model
  User->>Model: warmup_inputs (true trajectory)
  Note over Model: Teacher-forced; states sync to attractor
  loop horizon steps
    Model->>Model: predict one step
    Model->>Model: feed prediction back as feedback
  end
  Model->>User: predictions (batch, horizon, out_dim)
```

### Phase 1 — Warmup

Same idea as training warmup: `model.forecast(..., reset=True)` calls `warmup()` internally
unless you continue from an existing state (`reset=False`).

### Phase 2 — Autoregressive loop

For each step $t = 1 \ldots H$:

1. Forward one timestep with current feedback (and optional drivers from `forecast_inputs`).
2. Take model output as next feedback (first output if multi-head).
3. Append to prediction buffer.

## Basic usage

```python
pred = model.forecast(warmup_feedback, horizon=500)
# shape: (batch, 500, output_dim)
```

With exogenous drivers during the forecast window:

```python
pred = model.forecast(
    warmup_feedback,
    warmup_driver,
    forecast_inputs=(future_driver,),  # (batch, horizon, driver_dim)
    horizon=500,
)
```

Include warmup outputs in the returned tensor:

```python
full = model.forecast(warmup, horizon=500, return_warmup=True)
```

## The feedback dimension constraint

Autoregression requires the **first** model output to match the **feedback input** size.
If you forecast with feedback of shape `(..., 3)` but the first readout emits 10 features,
`forecast()` raises:

```text
ValueError: forecast(): feedback dimension mismatch.
Feedback input has 3 features but the model output used as feedback has 10 features.
```

**Fixes:**

- Set the primary readout `out_features` equal to `feedback_size`.
- For multi-readout models, order outputs so the feedback-sized head is first, or add a
  dedicated low-dimensional head for autoregression.

Multi-output models always use **output index 0** as feedback.

## Input convention

| Argument | Role |
|----------|------|
| `warmup_inputs` | Tuple `(feedback, *drivers)` or feedback tensor alone |
| `forecast_inputs` | Drivers only for the $H$ forecast steps (no feedback) |
| `initial_feedback` | Optional override for step 0 feedback `(batch, 1, fb_dim)` |

## See also

- [Two-phase training](two-phase-training.md)
- [`ESNModel.forecast`](../reference/core.md)
- [Chaotic systems guide](../guides/chaotic-systems.md)
- [Lorenz walkthrough](../getting-started/lorenz-walkthrough.md)
