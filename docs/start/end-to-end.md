---
description: A practical path from raw time series to a saved reservoir model inside a larger PyTorch pipeline.
---

<span class="nb-kicker">Start · End to end</span>

# Train it, forecast with it, reuse it

A reservoir turns recent observations into a useful memory of the signal.
The **readout** learns how to turn that memory into a prediction. Usually you
fit the readout with one algebraic solve; you do not need an epoch loop.

## 1. Start with your series

Install with `pip install resdag`. The shortest path accepts either NumPy or
PyTorch data, shaped `(time, features)` or `(batch, time, features)`:

```python
import torch
from resdag import ESN

t = torch.arange(1400, dtype=torch.float64) * 0.05
series = torch.stack((t.sin(), t.cos()), dim=-1)

esn = ESN(reservoir_size=100, washout=100, seed=0, dtype=torch.float64)
esn.fit(series[:1200])
prediction = esn.forecast(horizon=200)
truth = series[1200:1400]
rmse = (prediction - truth).square().mean().sqrt()
```

The first prediction is the sample **immediately after the last observation**.
A horizon of 200 returns exactly 200 future samples. Keep your test period
separate from training; for real data, fit scaling statistics on training data
only. The facade does not normalize automatically.

## 2. Use explicit splits when you want more control

The low-level API uses `(batch, time, features)` everywhere. Warmup means
“let the reservoir settle before fitting.” Training pairs each observation
with the next sample as its target:

```python
import resdag as rd

data = series.unsqueeze(0)
warmup, train, target, history, truth, stats = rd.data.prepare_esn_data(
    data, warmup_steps=100, train_steps=1000, val_steps=200,
    normalize=True, norm_method="standard", return_stats=True,
)
model = rd.classic_esn(
    reservoir_size=100, feedback_size=2, output_size=2, readout="svd",
).double()
rd.ESNTrainer(model).fit(warmup, train, target)
model.eval()
prediction = model.forecast(history, horizon=truth.shape[1])

# Return predictions to the original physical units.
prediction = rd.data.denormalize_data(prediction, "standard", stats)
truth = rd.data.denormalize_data(truth, "standard", stats)
```

`prepare_esn_data` fits scaling on training inputs only and aligns `history`
with `truth`. You can compare them directly. The last training target is
already observed, so history includes it before predicting held-out data.

For several input streams, use `(feedback, driver, ...)`. For several readouts,
pass a dictionary such as `{"position": position_target, "speed": speed_target}`.
For very long ordered sequences, use `readout="incremental"` and
`ESNTrainer(model).fit_stream(warmup, chunks)`, where each chunk is
`(inputs, targets)`. Keep chunks contiguous and keep the same trajectory in
each batch slot.

## 3. Save the model and the preprocessing

```python
model.save_full("trained.pt", normalization=stats, norm_method="standard")
restored, metadata = rd.ESNModel.load_full(
    "trained.pt", return_metadata=True, map_location="cpu",
)
```

This stores the architecture, fitted weights and current recurrent state.
Only load full model files you trust. For weights alone, use `model.save(...)`,
rebuild the same architecture, and call `model.load(..., map_location="cpu")`.
Add `include_states=True` when saving and `load_states=True` when loading if
you also want to resume the stored recurrent state. The same methods work for
parallel and coupled ensemble models. Save the scaling statistics too: a model
trained on scaled data expects that scaling at inference.

## 4. Put the trained model inside a bigger model

```python
from torch import nn

pipeline = nn.Sequential(
    restored.as_module(),       # complete reservoir model, including its readout
    nn.Linear(2, 1).double(),    # your task-specific head
)
output = pipeline(train)
```

`as_module()` starts each call from the configured reset state (zero by default),
making independent windows safe to batch or shuffle. It shares weights with the original model.
Use ordinary PyTorch loss functions and optimizers to train your added head.
Gradients can also flow to an encoder placed before the reservoir, while its
frozen weights stay fixed. Keep preprocessing consistent with the data used to
fit its readout; changing an encoder changes what the reservoir sees.

For an ordered stream, carry state explicitly instead:

```python
first, state = restored.forward_stateless(train[:, :400])
second, state = restored.forward_stateless(train[:, 400:], states=state)
```

These calls leave stored reservoir states unchanged. Other layers keep their
usual PyTorch `train()`/`eval()` behavior. Detach the returned state between
chunks if you want to stop gradients at a chunk boundary. Use
`as_module(stateful=True)` only when you deliberately want calls to share the
model's stored state; reset it between independent sequences.

Use a distinct reservoir layer instance at each position in a graph. Reusing
one instance mixes its recurrent state across positions and makes results
depend on chunk boundaries, so the explicit-state, pipeline and forecast APIs
reject it. Separate layer instances can still share the same weight tensors.

## 5. Match the device and precision

Move both the model and its data with `.to(device=..., dtype=...)`. Keep them
on that device during fitting and forecasting. Start with ordinary execution;
use `forecast(..., compile=True)` for repeated long rollouts and measure the
first compilation separately from steady-state runtime. Compilation support
and benefit depend on the model and hardware.

Run the complete small example, including saving, neural-head training and
chunk continuity:

```bash
uv run python examples/15_end_to_end.py
```

For the forecast timing change and old-result reproduction, see
[Migration](../migrating.md) and [Timing](../theory/timing.md).

## 6. Customize the reservoir

Use `recurrent_initializer=` to choose or register the recurrent weights;
`topology=` still works. For finer control, set input, feedback, bias and reset
state initializers separately. [Custom registration](../build/initialization/custom.md)
walks through defining your own function and selecting it by name.

You can also choose [oscillator dynamics](../build/layers/oscillator.md),
[Legendre memory](../build/layers/legendre-memory.md), or wrap a
[PyTorch GRU/LSTM](../build/layers/pytorch-recurrent.md). The same trainer,
forecasting and `as_module()` integration apply to all of them. Optional
[calibration](../workflows/calibration.md) adapts a standard tanh ESN on training
inputs before its readout is fitted.

The friendly `ESN.forecast()` temporarily enters evaluation mode and restores
the previous module modes afterward. When using the lower-level graph model
directly, call `model.eval()` yourself before inference, as in the examples
above. This disables training-only reservoir noise and ordinary PyTorch dropout.

Run `uv run python examples/16_custom_reservoirs.py` for a small comparison that
registers a custom initializer, calibrates, trains, forecasts, saves/restores,
and verifies gradients through each fitted model.
