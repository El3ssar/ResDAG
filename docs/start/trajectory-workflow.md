---
description: Fit TSDynamics trajectories and forecast named signals on their physical time axis.
---

# Forecast a TSDynamics trajectory

`rd.Model.fit(trajectory)` reads a TSDynamics `Trajectory` directly. The model
learns next-sample targets and training-only scaling, and `forecast()` returns
a new trajectory in the original signal units, with component names and future
timestamps. Ordinary `model(tensor)` calls still return differentiable tensors.

Install the optional dynamics extra on Python 3.12 or newer:

```sh
python -m pip install "resdag[dynamics]"
```

## Simulate, fit and forecast

Use TSDynamics' public `run` method to generate a trajectory. Split training
and evaluation samples before fitting or resampling.

```python
import numpy as np
import torch
import tsdynamics as ts
import resdag as rd

trajectory = ts.systems.Lorenz().run(final_time=2.0, dt=0.02)
training, truth = trajectory[:81], trajectory[81:91]
model = rd.Model(
    graph=rd.classic_esn(16, 3, 3, seed=7, dtype=torch.float64, readout="svd"),
    washout=10,
)
model.fit(training)
prediction = model.forecast(history=training[-20:], horizon=10)

assert isinstance(prediction, ts.Trajectory)
assert prediction.y.shape == truth.y.shape == (10, 3)
assert prediction.variables == ("x", "y", "z")
assert np.isfinite(prediction.y).all()
np.testing.assert_allclose(prediction.t, truth.t, rtol=0, atol=1e-12)
```

The trajectory stores times in `.t`, values in `.y` and component labels in
`.variables`. The forecast starts one sample after the supplied history.
Omitting `history` uses the training tail retained by `fit`.

The example checks the interface and sample alignment; its small reservoir
does not establish a forecast-accuracy benchmark for Lorenz.

## Check the physical-time boundary

A measured signal needs no producing system. Here the observed position rises
by one every 0.25 time units. A terminal linear readout learns that increment
exactly, making the first unseen value and timestamp explicit.

```python
def ramp_model(**options):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
    return rd.Model(inputs={"state": x}, outputs={"next": y}, washout=0, **options)

observed = ts.Trajectory(
    10.0 + 0.25 * np.arange(8),
    np.arange(8.0)[:, None],
    variables=("position",),
)
ramp = ramp_model().fit(observed)
future = ramp.forecast(horizon=4)
np.testing.assert_allclose(future.y, [[8.0], [9.0], [10.0], [11.0]], atol=1e-11)
np.testing.assert_allclose(future.t, [12.0, 12.25, 12.5, 12.75], rtol=0, atol=1e-12)

with_history = ramp.forecast(history=observed, horizon=4, return_warmup=True)
np.testing.assert_allclose(with_history.y, np.arange(1.0, 12.0)[:, None], atol=1e-11)
np.testing.assert_allclose(
    with_history.t, 10.0 + 0.25 * np.arange(1, 12), rtol=0, atol=1e-12
)
```

With `return_warmup=True`, each historical input contributes its next-sample
prediction. Those dates are the history dates after the first observation,
followed by the forecast dates. The boundary prediction appears once.

Scaling statistics use supervised training inputs after washout, excluding
the final observation that supplies the shifted target. Validation and test
samples never belong in that fit call. Forecasting leaves those statistics
and the cached training tail unchanged.

## Choose a sampling grid

Valid irregular trajectories are resampled automatically. By default, the new
grid preserves the first time, last time and sample count. Its interval is
`(last_time - first_time) / (n_samples - 1)`. Existing regular observations pass
through. Sampling comes from the actual time axis, even if source metadata
contains an older `dt`.

Set `resample_dt` to choose an interval. That grid ends at the final
observation and stays inside the observed time range, so it can drop a
fractional interval at the beginning. Interpolation never extrapolates.
`resample_method` accepts `"linear"` (default), `"pchip"` or `"cubic"`.
Interpolation does not apply an anti-alias filter when choosing a larger
interval; apply any required filtering before supplying the training data.

```python
irregular_times = np.array([2.0, 2.2, 2.9, 4.0])
irregular = ts.Trajectory(
    irregular_times,
    (3 * irregular_times + 10)[:, None],
    variables=("position",),
)
automatic = ramp_model().fit(irregular)
automatic_future = automatic.forecast(horizon=2)
np.testing.assert_allclose(automatic_future.t, [4 + 2 / 3, 4 + 4 / 3], atol=1e-12)
np.testing.assert_allclose(automatic_future.y, [[24.0], [26.0]], atol=1e-10)

fixed = ramp_model(resample_dt=0.75, resample_method="pchip").fit(irregular)
fixed_future = fixed.forecast(horizon=2)
np.testing.assert_allclose(fixed_future.t, [4.75, 5.5], rtol=0, atol=1e-12)
np.testing.assert_allclose(fixed_future.y, [[24.25], [26.5]], atol=1e-10)
```

The fixed grid above is `[2.5, 3.25, 4.0]`. A later trajectory supplied as
forecast history is resampled to the fitted interval, keeping each recurrent
step at the duration used for training. Its component names and order must
match the training trajectory. Duplicate, decreasing or nonfinite times are
rejected before execution.
The time origin and interval must also remain distinguishable at float64
precision. If a large absolute time prevents a uniform grid, subtract a time
origin or choose a coarser interval.

## Keep tensors on the model device

Choose `return_type="tensor"` to receive the usual Torch output on the model
device. Trajectory output makes a detached CPU/NumPy copy after forecasting;
the tensor path makes no such copy. Since NumPy has no bfloat16 dtype,
bfloat16 trajectory values are converted to float32 at that output boundary.

```python
tensor_future = ramp.forecast(horizon=4, return_type="tensor")
assert isinstance(tensor_future, torch.Tensor)
torch.testing.assert_close(tensor_future, torch.from_numpy(future.y))

tensor_history = torch.from_numpy(observed.y.copy())
tensor_continuation = ramp.forecast(
    history=tensor_history, horizon=4, return_type="tensor"
)
torch.testing.assert_close(tensor_continuation, tensor_future)
```

`return_type="auto"` is the default: trajectory fitting or explicit trajectory
history selects trajectory output; a tensor-only workflow selects tensors.
Tensor history supplies no timestamps, so requesting trajectory output from
it raises a clear error. Supply a trajectory history or request tensor output.
A successful tensor refit returns the default to tensors.

## Restore a checkpoint

The ordinary state dictionary includes scaling, the observed tail, sampling
options, component names, clock information and portable source metadata.
Rebuild the same named graph before loading.

```python
import io

checkpoint = io.BytesIO()
torch.save(fixed.state_dict(), checkpoint)
checkpoint.seek(0)
restored = ramp_model()
restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
restored_future = restored.forecast(horizon=2)
np.testing.assert_array_equal(restored_future.t, fixed_future.t)
np.testing.assert_allclose(restored_future.y, fixed_future.y, rtol=0, atol=0)
assert restored_future.variables == ("position",)
```

The producing system is a transient reference and is not serialized. A
restored forecast retains its source identity in `.meta["source"]`, while
`.system` is `None`. Model device/dtype conversions move tensors and fitted
scaling; physical timestamps retain float64 precision.
Source arrays become plain lists in checkpoint metadata. Opaque Python
metadata values are represented by type markers instead of serialized objects.

Bare trajectory fitting infers shifted targets for one input/output of equal
width. `prediction_lead` can select a larger distance. Warmup outputs then
start at that lead's historical timestamp, while future predictions still
begin at the first unseen sample. Named inputs and explicit trajectory
targets are supported by the [named trajectory workflow](named-trajectories.md),
which aligns all ports on a shared observed range and applies per-head leads
to physical target timestamps. Tensor targets remain caller-aligned.

Both bare and named trajectory checkpoints use the current version 2 clock
envelope. Older common-workflow trajectory metadata is rejected; reconstruct
and refit those models to save a current checkpoint. There is no automatic
converter. Ordinary tensor forward and explicit-state calls retain the
PyTorch behavior described in the [common model guide](model-workflow.md).
