---
description: Next-sample training, forecast indices, driver alignment, and recurrent state across calls.
---

<span class="nb-kicker">Theory · Timing</span>

# Timing conventions

A next-step readout consumes feedback $x_t$ and an optional driver $d_t$ and
predicts $x_{t+1}$. This same rule applies to training, teacher-forced evaluation,
and autonomous forecasting.

The examples below describe default next-sample timing and the explicit
low-level `ESNModel`/ensemble APIs. The common `rd.Model` and `rd.ESN` share
that default; their optional prediction leads and named delayed connections
are explained in [feedback and timing](../start/feedback.md). Their future
drivers use `future_inputs`, and each forecast reconstructs its rollout from
history; the raw continuation arguments below do not apply to that interface.

## Training and validation

After discarding any initial transient, let `W = warmup_steps` and
`L = train_steps`. Both counts must be positive integers. Preparation builds:

```python
warmup = data[:, :W]
train = data[:, W:W+L]
target = data[:, W+1:W+L+1]
f_warmup = data[:, L+1:W+L+1]
val = data[:, W+L+1:W+L+1+H]
```

The trainer fits each input row to the matching target row without another
shift. The final training target, `data[:, W+L]`, has already been observed
during fitting. Forecast warmup includes it; validation starts after it.
`f_warmup` always contains `W` observations, even when `L < W`.

Normalization statistics are fitted on training inputs alone and reused across
all windows. Validation therefore contributes neither targets to fitting nor
statistics to preprocessing.

## Forecast the immediate unseen sample

If `observed` contains `x[:, :N]`, then:

```python
predictions = model.forecast(observed, horizon=H)
# predictions[:, i] estimates x[:, N+i]
```

The last warmup readout already predicts the next sample. Forecast slot zero
uses that prediction. Only the remaining `H-1` predictions need further model
updates. `horizon` must be a positive integer; zero is rejected. A one-step
forecast is useful and does not require any future drivers.

| Output | Feedback consumed | Driver consumed | Sample estimated |
| --- | --- | --- | --- |
| last warmup output / forecast slot 0 | $x_{N-1}$ | $d_{N-1}$ | $x_N$ |
| forecast slot 1 | $\hat x_N$ | $d_N$ | $x_{N+1}$ |
| forecast slot 2 | $\hat x_{N+1}$ | $d_{N+1}$ | $x_{N+2}$ |

For a driven model, future drivers start immediately after the observed driver
history. They accompany the *input being consumed*, not its next-sample target:

```python
predictions = model.forecast(
    (x[:, :N], d[:, :N]),
    forecast_inputs=d[:, N:N+H-1],
    horizon=H,
)
truth = x[:, N:N+H]
```

Longer driver windows are accepted; only the required prefix is consumed.
With `return_warmup=True`, historical predictions are prepended. The final
warmup prediction is already slot zero, so it appears once: total length is
`W-1+H`. Core, coupled ensemble, and parallel models share these semantics.
With the default lead/delay of one, the `ESN` preset retains `washout+1`
observations and therefore returns `washout+H` values when including historical
predictions. Larger leads/delays can require a longer cached history. The common
workflow's general length is `W - max(model.prediction_leads.values()) + H`,
where `W` is the history actually supplied or cached.

## Explicit seeds and continuation

This section uses the raw graph APIs, not `rd.Model` or the `rd.ESN` preset.

`initial_feedback` is a distinct extra input consumed *after* the complete
warmup. Slot zero predicts the sample after that seed. Do not repeat the last
warmup observation as the seed. Explicit seeds need `H` future driver samples,
starting at the same time as the seed, and `return_warmup=True` has length `W+H`.

A forecast leaves its final returned prediction unconsumed. Continue the same
trajectory without replaying history by consuming it as the next explicit seed:

```python
first = model.forecast(observed, horizon=100)
next_part = model.forecast(
    observed[:, :0],
    initial_feedback=first[:, -1:],
    horizon=100,
    reset=False,
)
```

For driven models, pass empty history for every stream and future drivers
beginning at the seed's time. Empty warmup without an explicit seed raises.
`reset=False` uses current states; `reset=True` starts from zero states.

`windowed_forecast` fills gaps at their own timeline indices. It consumes the
final predicted gap input before the next observed window so recurrent state
remains continuous. Score only gap positions using its returned mask.

## Reproducing earlier low-level forecasts

`alignment="legacy"` preserves the old core and ensemble convention: it
consumes the last warmup prediction before returning slot zero, skipping the
immediate unseen sample. It requires `H` drivers and prepends `W` predictions
when `return_warmup=True`. Pair it with
`prepare_esn_data(..., alignment="legacy")`, whose forecast history ends one
sample earlier. Training and validation splits stay unchanged.

Earlier ParallelESN forecasts already started at the immediate unseen sample;
keep the default to preserve their future timing. Their old `return_warmup`
prefix contained raw observations; the prefix now contains predictions.
See [Migration](../migrating.md) for the compatibility recipe.

## State across calls

Ordinary `model(x)` calls carry reservoir state. `warmup` and `forecast` reset
by default. Use `reset_reservoirs()` between independent trajectories or use
`model.as_module()` for independent PyTorch minibatches. The full model's
`forward_stateless(..., states=...)` returns predictions and explicit state
without changing stored reservoir buffers; it supports deliberate chunking and
backpropagation across chunks.

Stored states move with `.to(device)` and dtype conversion. Inputs must match
the model's device and dtype. By default, stored state detaches at ordinary
call boundaries. Set `detach_state_between_calls=False` only when managing
backpropagation across those calls. Explicit stateless states retain gradients
until you detach them yourself.
