---
description: The one-step target shift, default and explicitly seeded forecast indices, driver alignment, and state across calls.
---

<span class="nb-kicker">Theory · Timing</span>

# Timing conventions

A next-step readout maps feedback $f_t$ and an optional driver $d_t$ to
$\hat f_{t+1}$. The reservoir state after consuming that pair also belongs to
time $t$. Keeping those two indices distinct prevents a plausible forecast
from being scored one sample early.

## Training pairs

After discarding any initial transient, let `warmup_steps = W` and
`train_steps = L`. `prepare_esn_data` builds:

```python
warmup = data[:, :W]
train = data[:, W : W + L]
target = data[:, W + 1 : W + L + 1]
```

State row `t` is fitted to target row `t`. The trainer performs no additional
shift. The training target includes `data[:, W + L]`, one sample beyond the
last training input, so preparing a split requires that extra sample.

The forecast warmup is the final `W` samples of the entire observed input
prefix `data[:, :W + L]`. This remains exactly `W` samples even when `L < W`.
With `W = 0` it is empty. The validation split starts at
`data[:, W + L + 1]`, as required by the default forecast convention below.

## Default forecast: seed from the model's prediction

`forecast(warmup_inputs, horizon=H)` first processes the warmup. Let $T$
index its last input pair $(f_T, d_T)$. The last warmup output estimates
$f_{T+1}$; it seeds the autoregressive loop and is **not** included in the
returned forecast unless `return_warmup=True`.

Every returned slot performs a new model update:

| Phase | Feedback consumed | Driver consumed | Output estimates |
| --- | --- | --- | --- |
| last warmup step | $f_T$ | $d_T$ | $f_{T+1}$, used as the seed |
| forecast slot 0 | $\hat f_{T+1}$ | $d_{T+1}$ | $f_{T+2}$ |
| forecast slot 1 | $\hat f_{T+2}$ | $d_{T+2}$ | $f_{T+3}$ |
| forecast slot $i$ | $\hat f_{T+1+i}$ | $d_{T+1+i}$ | $f_{T+2+i}$ |

Thus `predictions[:, i]` estimates $f_{T+2+i}$. With prepared data,
$T = W + L - 1$, so prediction 0 aligns with `data[:, W + L + 1]`,
the first validation sample. The preceding seam sample `data[:, W + L]`
is the final training target and the model's predicted seed.

`forecast_inputs` holds **one driver value per returned step**, starting
immediately after the warmup driver window. It must contain at least `H`
steps; only its first `H` values are consumed. A single driver tensor can be
passed directly; multiple drivers use a tuple.

`return_warmup=True` concatenates all warmup outputs with these `H` new
outputs. It adds no repeated frame at the boundary.

## Predict immediately after the last observation

For a continuous observed prefix of length `N`, the next unobserved sample
has index `N`. To predict it in slot 0, keep the final observation as explicit
feedback and warm up through the preceding sample:

```python
predictions = model.forecast(
    observed[:, :-1],
    initial_feedback=observed[:, -1:],
    horizon=H,
)
# predictions[:, i] estimates the sample at index N + i
```

The first update consumes the actual observation at index `N - 1`, producing
the prediction for index `N`. For a driven model, warm up on the corresponding
prefix of the driver and start `forecast_inputs` at driver index `N - 1`:
the driver must have the same time index as the feedback it accompanies.

An empty warmup is supported with an explicit `initial_feedback`, which is
useful when starting from zero or deliberately restored states. Without a
seed, an empty warmup raises because there is no prediction to feed back.
The high-level `ESN.forecast` facade handles this observed-prefix alignment
for you.

`windowed_forecast` also aligns gap predictions to the original timeline.
It starts each gap with the final teacher-forced prediction and follows it
with the required autoregressive predictions. Score only its gap positions
using the returned observation mask.

## State across calls

Reservoir layers carry mutable state between ordinary `forward` calls.

- `warmup` and `forecast` reset by default (`reset=True`). A bare `model(x)`
  or `layer(x)` continues the current state; use `reset_reservoirs` or
  `reset_state` between independent trajectories.
- `.to(device)` and dtype conversions move the stored state along with the
  parameters, preserving its values. Move the inputs to match the module.
  Naturally accumulated states reinitialize to zeros when batch size changes.
  A state deliberately restored with `set_state` instead raises on a batch
  mismatch until it is consumed or explicitly reset.
- By default, stored states detach at call boundaries while returned outputs
  retain gradients within the call. Set `detach_state_between_calls=False`
  only when intentionally backpropagating across calls and managing those
  retained graphs.
- `ReservoirFeatureExtractor(stateful=False)` starts independent minibatches
  from zeros without changing stored states. Its `forward_stateless` method
  accepts and returns explicit states, keeping gradients across chunks until
  you detach them yourself.

## Next

[**Design of the library**](design.md) explains the composition and execution
choices behind these interfaces.
