---
description: Train custom models from named observed signals and predict on a shared physical clock.
---

# Named trajectories

Use named TSDynamics trajectories when a model has several observed inputs or
prediction heads. ResDAG aligns the observed inputs on one clock and uses each
head's `prediction_lead` to find its target labels. You still call `fit` and
`forecast`; you do not need to pack signals into one array.

Install `resdag[dynamics]` on Python 3.12 or newer. Every named input in one
call must be a `Trajectory`; every explicit target must also be a trajectory.
Mixed tensor and trajectory ports are rejected because tensors have no
physical timestamps. The ordinary named tensor workflow remains available
when you already prepared aligned arrays.

## Build a model with two prediction heads

Here position follows `x[t+1] = x[t] + force[t]`. One head predicts one sample
ahead; another predicts two. The two-step head feeds position back after two
graph steps. Their returned forecasts refer to the same physical dates.

```python
import numpy as np
import torch
import tsdynamics as ts
import resdag as rd

def build_model():
    state = rd.reservoir_input(1, dtype=torch.float64)
    force = rd.reservoir_input(1, dtype=torch.float64)
    features = rd.Concatenate()(state, force)
    near = rd.RidgeReadoutLayer(2, 1, alpha=0.0).double()(features)
    far = rd.RidgeReadoutLayer(2, 1, alpha=0.0).double()(features)
    return rd.Model(
        inputs={"state": state, "force": force},
        outputs={"near": near, "far": far},
        prediction_lead={"near": 1, "far": 2},
        feedback=[rd.Feedback("far", "state", delay=2)],
        washout=2,
    )

def position(time):
    return time * (time + 1) / 2

state_times = np.arange(13, dtype=np.float64)
force_times = np.arange(0.25, 12.26, 0.5)
observations = {
    "force": ts.Trajectory(
        force_times, (force_times + 1)[:, None], variables=("forcing",)
    ),
    "state": ts.Trajectory(
        state_times, position(state_times)[:, None], variables=("position",)
    ),
}
# Labels may begin later than observations. Each target owns its component name.
near_times = np.arange(6, 17, dtype=np.float64)
far_times = np.arange(5, 17, dtype=np.float64)
targets = {
    "near": ts.Trajectory(
        near_times, position(near_times)[:, None], variables=("near_position",)
    ),
    "far": ts.Trajectory(
        far_times, position(far_times)[:, None], variables=("far_position",)
    ),
}
model = build_model().fit(observations, targets)
```

The default interval comes from the **first declared model input**, `state`,
so it is one time unit here. Caller dictionary order does not change it. The
shared grid stays inside both observed ranges and ends at their latest common
endpoint: `[1, 2, ..., 12]`. Set `resample_dt` to choose another interval;
`resample_method` accepts `"linear"`, `"pchip"` or `"cubic"`.

For each origin `t`, the near head uses the target at `t + dt`, and the far
head uses the target at `t + 2*dt`. Both labels exist for origins `5..10`.
Only those origins and their actual labels fit the scaling and readouts.
`washout=2` excludes the first two shared input samples; it does not discard
another two samples after the late labels begin. Earlier observations still
warm the recurrent layers. The cached tail includes observations through time
12, even after the last supervised origin. Fast framewise models retain
`washout + required_history` input samples; complete-history models retain the
full aligned observations. Pass explicit history when you need a longer
warmup window.

Target observations after the latest common input endpoint are held out.
They do not enter interpolation coefficients, fitted statistics or weights.
Keep validation/test data outside the fit call as usual. Input interpolation
may use available source observations surrounding a shared-grid point within
that input's observed range. Resampling never extrapolates and does not apply
an anti-alias filter when increasing the interval.

## Forecast with future drivers

Position is fed back by the model, while force needs external future values.
Drivers are sampled at the required fitted ticks after the history endpoint;
their own endpoints do not move the model's clock. With this one-step/two-step
pair, a horizon of four needs three future driver samples.

```python
driver_times = np.array([13.0, 13.25, 14.5, 15.0])
future_force = ts.Trajectory(
    driver_times, (driver_times + 1)[:, None], variables=("forcing",)
)
prediction = model.forecast(horizon=4, future_inputs={"force": future_force})
assert list(prediction) == ["near", "far"]
for name, result in prediction.items():
    assert isinstance(result, ts.Trajectory)
    assert result.variables == (f"{name}_position",)
    np.testing.assert_array_equal(result.t, [13, 14, 15, 16])
    np.testing.assert_allclose(result.y[:, 0], position(result.t), atol=1e-10)

tensors = model.forecast(
    horizon=4, future_inputs={"force": future_force}, return_type="tensor"
)
for name in model.output_names:
    torch.testing.assert_close(tensors[name], torch.from_numpy(prediction[name].y))

with_history = model.forecast(
    observations, horizon=4, future_inputs={"force": future_force}, return_warmup=True
)
np.testing.assert_array_equal(with_history["near"].t, np.arange(3, 17))
np.testing.assert_array_equal(with_history["near"].t, with_history["far"].t)
```

The warmup portion starts at history index `max(prediction_leads)`, so every
head shares the same timeline. A horizon of one needs no future driver here.
If exactly one driver sample is needed, a one-row trajectory is supported.
Tensor drivers are also accepted and are treated as caller-aligned samples.

Pass `history={input_name: trajectory, ...}` to forecast from another observed
window. History is resampled to the fitted interval and must keep each input's
component names and order. A new tensor history has no absolute clock: request
tensor results and use tensor drivers, or provide timestamped history.

Ordinary `model(named_tensors)` preserves Torch gradients. Tensor forecasts
stay on the model's device and dtype. Only a trajectory result converts its
predictions to NumPy on CPU; bfloat16 values become float32 at that boundary.
One output returns a single trajectory. Several outputs return a dictionary
in declared model order, each with its target component names and source metadata.

## Restore a matching graph

The state dictionary records the shared clock, ordered port metadata, target
component names, scaling and the cached observed tail. Reconstruct the
same graph and custom preprocessing before loading its state.

```python
torch.save(model.state_dict(), "named-model.pt")
restored = build_model()
restored.load_state_dict(torch.load("named-model.pt", weights_only=True))
replayed = restored.forecast(horizon=4, future_inputs={"force": future_force})
for name in model.output_names:
    np.testing.assert_array_equal(replayed[name].t, prediction[name].t)
    np.testing.assert_array_equal(replayed[name].y, prediction[name].y)
```

The current trajectory checkpoint envelope is version 2 for both bare and
named inputs. Older common-workflow trajectory checkpoints are rejected;
reconstruct and refit those models before saving a current checkpoint.
There is no automatic checkpoint converter. Live producing-system objects
are kept only in memory; restored forecasts retain plain source metadata
and have `.system is None`. Generic artifact loading without reconstructing
the graph remains a separate roadmap phase.

For a single observed signal with implicit next-sample targets, start with
the [basic trajectory workflow](trajectory-workflow.md). For custom reversible
scaling, see [custom preprocessing](../advanced/custom-preprocessing.md).
