---
description: Train a reservoir, forecast in original units, save its state and reuse it inside PyTorch.
---

<span class="nb-kicker">Start · End to end</span>

# Train it, forecast with it, reuse it

A reservoir turns recent observations into a useful memory. The readout learns
how to turn that memory into a prediction. Usually training needs one algebraic
solve, rather than an epoch loop.

## 1. Keep training and test data separate

Install with `pip install resdag`. Start with a floating Torch tensor shaped
`(time, signals)` or `(batch, time, signals)`. Convert NumPy data explicitly
with `torch.as_tensor(array)`.

```python
import torch
import resdag as rd

t = torch.arange(1400, dtype=torch.float64) * 0.05
series = torch.stack((t.sin(), t.cos()), dim=-1)
training, truth = series[:1200], series[1200:]
options = dict(reservoir_size=100, washout=100, seed=0, readout="svd", dtype=torch.float64)
model = rd.ESN(**options).fit(training)
```

`fit` infers the signal count, learns scaling from the supervised training
inputs only, and fits the next-sample readout. Washout is the initial period
used to synchronize the reservoir before solving. Test samples never enter
that solve or the scaling statistics.

For a TSDynamics simulation, install `resdag[dynamics]` and pass its trajectory
directly. The [trajectory guide](trajectory-workflow.md) explains timestamps,
signal names, resampling and trajectory forecasts.

## 2. Forecast and measure error

```python
prediction = model.forecast(horizon=len(truth))
assert prediction.shape == truth.shape
rmse = (prediction - truth).square().mean().sqrt()
assert torch.isfinite(rmse)
```

The first prediction estimates the sample immediately after the observations.
A horizon of 200 returns 200 future samples, in the original units. Supply
`history=new_observations` to synchronize on a different observed sequence.
Otherwise the model uses its saved training history.

Chaotic forecasts eventually diverge. Compare several starting points and
seeds on validation data, then evaluate the final choice on an untouched test
period. [Automatic tuning](../workflows/tune.md) handles the validation split,
candidate comparison and final refit. Matched speed–accuracy comparisons remain
in [roadmap phase five](../project/roadmap.md).

## 3. Save and restore the complete model

```python
model.save("trained.pt")
restored = rd.load("trained.pt", map_location="cpu")
restored_prediction = restored.forecast(horizon=len(truth))
torch.testing.assert_close(restored_prediction, prediction, rtol=0, atol=0)
model.save("model.svg")  # diagram; requires the optional visualization tools
```

The checkpoint includes the architecture, fitted weights, preprocessing,
observed history and ordinary-forward reservoir memory. You do not reconstruct
the DAG. Forecasts replay their supplied or saved history; a checkpoint does
not resume an in-progress delayed rollout. See [save and load](../workflows/save-load.md)
for custom components, notebook bundles and diagram formats. Standard
`state_dict()` remains available for larger PyTorch pipelines that already own
the architecture.

## 4. Put it inside a larger PyTorch model

```python
from torch import nn

pipeline = nn.Sequential(
    nn.Linear(2, 2, dtype=torch.float64),
    restored,
    nn.Linear(2, 1, dtype=torch.float64),
)
restored.reset_state()
batch = training[-20:].unsqueeze(0).detach().requires_grad_()
loss = pipeline(batch).square().mean()
loss.backward()
assert batch.grad is not None and torch.isfinite(batch.grad).all()
```

Frozen reservoir weights still pass gradients to the preceding encoder. Use
ordinary Torch losses and optimizers. Fit or `build(input_size=...)` before
attaching an optimizer, so it sees every parameter.

Calls carry reservoir state. Reset before unrelated windows; `train()` and
`eval()` do not reset it. For explicit state across chunks:

```python
first, state = restored.forward_stateless(training[:20])
second, state = restored.forward_stateless(training[20:40], states=state)
assert first.shape == second.shape == (20, 2)
```

These calls leave stored reservoir state unchanged. The returned states can
retain gradients across chunks. Detach them yourself when intentionally
truncating backpropagation.

## 5. Customize only what you need

Use `readout` and `readout_kwargs` to select or configure a solver. Supply
custom initializers directly or register them by name. Use `model.graph` for
raw, already-preprocessed graph execution.

For branches, drivers, multiple outputs or a different recurrent cell, build
`rd.Model(inputs={...}, outputs={...})`. It uses the same `fit`, `forecast`,
forward, state and checkpoint commands. Start with the
[common model guide](model-workflow.md), then the
[public component recipes](../advanced/custom-components.md).
Use [custom preprocessing](../advanced/custom-preprocessing.md) to change the
units of individual inputs or outputs while retaining automatic inverse
conversion. [Named trajectories](named-trajectories.md) align several
timestamped inputs and targets for a custom graph.

## 6. Choose device and precision explicitly

Move both the model and tensor data with `.to(device=..., dtype=...)`. Fitted
statistics move with the model. An untouched lazy ESN adopts its first training
tensor's dtype/device; explicit constructor placement or `.to(...)` fixes them.

On MPS, use float32. The default automatic readout uses supported native solves
when its numerical checks pass, and reports/warns about any CPU float64
fallback. Set `readout_kwargs={"cpu_fallback": False}` to require native fitting,
or explicitly choose a solver. See [readout solvers](../guides/readout-solvers.md).

Use `compile=True` only for supported fast execution, and measure first-time
compilation separately. For custom operations that need the whole sequence,
choose [history execution](../advanced/forecast-capabilities.md). It trades
extra computation and memory for complete-prefix semantics.
