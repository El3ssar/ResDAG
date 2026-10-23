---
description: Build and draw a custom reservoir DAG, compare it with vanilla and OTT models, and measure an autonomous forecast on TSDynamics data.
---

# Run a forecast experiment

This example builds a network, draws its DAG, trains it and measures how long
its autonomous prediction follows a Lorenz trajectory. The same commands work
for a vanilla ESN, an OTT model and your own DAG.

Install on Python 3.12 or newer:

```sh
python -m pip install "resdag[hpo,dynamics,viz]"
```

SVG diagrams also need the Graphviz executable: `brew install graphviz` on
macOS, or your system's Graphviz package. Saving `.dot` works without it.

## Generate data and keep the future separate

```python
from time import perf_counter

import numpy as np
import torch
import tsdynamics as ts
import resdag as rd

trajectory = ts.systems.Lorenz().run(
    final_time=100, dt=0.02, transient=20,
    solver="dop853", rtol=1e-10, atol=1e-12,
)
training, truth = trajectory[:4000], trajectory[4000:4500]
washout = 200
```

`training` contains observed samples. `truth` is the future we will check
after training. Neither the readout nor preprocessing sees `truth` during fit.
The model learns scaling from training data and returns predictions in the
original units.

## Choose a ready-made model

```python
models = {
    "vanilla": rd.ESN(
        reservoir_size=192, input_size=3, washout=washout,
        readout="svd", seed=7, dtype=torch.float64,
    ),
    "ott": rd.Model(
        graph=rd.ott_esn(192, 3, 3, readout="svd", seed=7, dtype=torch.float64),
        washout=washout,
    ),
}
```

Both have 192 reservoir units. The OTT model adds squared state features;
the vanilla model uses ordinary reservoir states. These are starting settings,
not tuned accuracy recipes.

## Or connect your own DAG

Two branches can remember the same signal at different rates. Concatenating
their states with the original input gives the readout all three sources.

```python
x = rd.reservoir_input(3, dtype=torch.float64)
fast = rd.ESNLayer(
    96, feedback_size=3, spectral_radius=0.9, leak_rate=1.0,
    seed=7, dtype=torch.float64,
)(x)
slow = rd.ESNLayer(
    96, feedback_size=3, spectral_radius=0.9, leak_rate=0.3,
    seed=8, dtype=torch.float64,
)(x)
features = rd.Concatenate()(x, fast, slow)  # 3 + 96 + 96 signals
y = rd.SVDReadoutLayer(195, 3, alpha=1e-6).double()(features)
models["custom"] = rd.Model(
    inputs={"state": x}, outputs={"prediction": y}, washout=washout,
)
models["custom"].save("custom.svg")
```

Open `custom.svg` to inspect the connections. In a notebook,
`models["custom"].plot_model()` returns a displayable Graphviz diagram.
Changing a layer, its initializer or solver changes the model you train;
the surrounding fit/forecast workflow stays the same.

## Train, forecast and measure

For each future sample, compare the prediction with the true signals. Normalize
each signal's error with the standard deviation of its **training targets**
after washout. For next-sample training, those are `training.y[washout + 1:]`:
the final observed sample is a known training label. This keeps the score's
scale fixed before seeing the future.

The valid prediction horizon ends at the **first** error of 0.3 or higher.
Converting it to Lyapunov times uses the actual sample interval and the
canonical Lorenz exponent, approximately 0.9056 per unit of physical time.

```python
scale = training.y[washout + 1:].std(axis=0)
scale = np.where(scale == 0, 1.0, scale)
dt = float(training.t[1] - training.t[0])
lambda_max = 0.9056
results = {}

for name, model in models.items():
    start = perf_counter()
    model.fit(training)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    prediction = model.forecast(history=training[-500:], horizon=len(truth))
    forecast_seconds = perf_counter() - start
    np.testing.assert_allclose(prediction.t, truth.t, rtol=0, atol=1e-10)

    error = np.sqrt(np.mean(((prediction.y - truth.y) / scale) ** 2, axis=-1))
    valid = np.isfinite(error) & (error < 0.3)
    steps = len(valid) if valid.all() else int(np.argmax(~valid))
    secondary = np.isfinite(error) & (error < 0.5)
    secondary_steps = len(error) if secondary.all() else int(np.argmax(~secondary))
    results[name] = {
        "fit_seconds": fit_seconds,
        "forecast_seconds": forecast_seconds,
        "valid_steps": steps,
        "valid_lyapunov_times": steps * dt * lambda_max,
        "reached_horizon": bool(valid.all()),
        "secondary_valid_lyapunov_times_at_0_5": secondary_steps * dt * lambda_max,
        "secondary_reached_horizon": bool(secondary.all()),
    }
    print(name, results[name])
```

The forecast receives observed history to synchronize the reservoir. Its first
prediction is the sample immediately after that history. After that, its own
predictions feed the reservoir: it does not receive future truth.

If `reached_horizon` is true, the result is a **lower bound**: extend the test
window before claiming that you measured the model's full limit. A single
origin and seed do not establish general accuracy. These illustrative settings
do not promise 7–8 Lyapunov times. Choose settings on separate validation data,
then compare several seeds and starting points on an untouched test period.
See [automatic tuning](../workflows/tune.md).

The stricter 0.3 threshold is the main score; 0.5 is a secondary, more permissive
score. The phase-five 7–8 Lyapunov-time target applies to the **median** across
seeds and starting points at 0.3, with the full spread reported. A lucky single
forecast is not enough.

These training-normalized scores have a different scale from
`resdag.metrics.valid_prediction_time`, which uses the supplied truth window's
standard deviation. State the normalization and threshold when comparing
reported numbers. Do not compare horizons measured with different definitions.

## Tune without using the final test

Pass only `training` to tuning. It holds out the final 20% of that data for
candidate selection, then refits the winner on all of `training`. The separate
`truth` segment stays outside the call.

```python
tuned = models["vanilla"]
tuned.tune(
    training, preset="balanced", threshold=0.3, horizon=200, seed=7,
    n_trials=2, seeds=[7], n_origins=1,  # small tutorial budget
)
tuned_prediction = tuned.forecast(horizon=len(truth))
np.testing.assert_allclose(tuned_prediction.t, truth.t, rtol=0, atol=1e-10)
tuned_error = np.sqrt(np.mean(((tuned_prediction.y - truth.y) / scale) ** 2, axis=-1))
tuned_valid = np.isfinite(tuned_error) & (tuned_error < 0.3)
tuned_steps = len(truth) if tuned_valid.all() else int(np.argmax(~tuned_valid))
print("Winning settings:", tuned.tuning_report.best_params)
print("Final-test Lyapunov times:", tuned_steps * dt * lambda_max)
```

The explicit overrides keep this tutorial short. Remove `n_trials`, `seeds` and
`n_origins` for the full balanced budget: **24 configurations × 3 seeds × 3
forecast starting points**. Increase the validation horizon when measuring
longer predictions; it must fit completely within the held-out validation span.
Two candidates over a short validation horizon are a workflow demonstration,
not evidence of the 7–8 target. A final test that stays valid throughout remains
a lower bound.

For a separate validation trajectory, supply
`validation=(validation_trajectory, None)` while keeping the final test out of
the call. For custom DAGs, [provide a model-building recipe](../workflows/tune.md#change-sizes-layers-or-initializers)
to search over layer widths, wiring and initializers. The winner stays in the
same model object and is ready to forecast.

## Save the trained DAG and reload it

```python
custom = models["custom"]
custom.save("custom.pt")
restored = rd.load("custom.pt")
expected = custom.forecast(history=training[-500:], horizon=len(truth))
actual = restored.forecast(history=training[-500:], horizon=len(truth))
np.testing.assert_allclose(actual.y, expected.y, rtol=0, atol=0)
```

The file contains the graph, fitted readout, preprocessing and current memory.
You do not need to know or rebuild its DAG. See [save and load](../workflows/save-load.md)
for registered custom components, trusted notebook bundles and device mapping.

## Try Rössler or a larger reservoir

The runnable script generates data, evaluates all three models, saves their
diagrams and portable checkpoints, and checks reload parity:

```sh
python examples/17_forecast_experiment.py --system lorenz --output lorenz-run
python examples/17_forecast_experiment.py --system rossler --reservoir-size 400 --output rossler-run
```

The script uses canonical Rössler parameters `(a, b, c) = (0.2, 0.2, 5.7)` and
an approximate largest exponent of 0.0714. Its default evaluation window covers
at least ten Lyapunov times for either system; that is the available scoring
window, not a prediction-quality claim. Override `--dt`, `--horizon`, `--seed`,
`--washout` or `--history` to make the experiment yours. Changing system
parameters also requires a corresponding exponent; never reuse a canonical
exponent for a different dynamical regime.

`--diagram-format dot` avoids visualization dependencies. The reported CPU
wall times include the convenient workflow and are exploratory measurements,
not a controlled speed comparison. Phase-five
[benchmarks](../workflows/benchmarks.md) define the reproducible comparison.
