---
description: Fit a reservoir on training observations, forecast unseen Lorenz samples and measure the error in original units.
---

<span class="nb-kicker">Start · First forecast</span>

# First forecast

An ESN learns a map from the signal now to the next sample. Its reservoir
weights stay fixed; fitting solves the linear prediction head. During an
autonomous forecast, each prediction becomes the next input.

This example needs only the core installation. It separates training and
unseen observations before fitting. Scaling is learned from training inputs
and predictions are returned in the original units.

```python
import torch
import resdag as rd

series = rd.datasets.lorenz(6001, normalize=None, seed=7)[0]
training, truth = series[:5001], series[5001:]

model = rd.ESN(reservoir_size=300, washout=200, seed=7)
model.fit(training)
prediction = model.forecast(horizon=len(truth))

assert prediction.shape == truth.shape == (1000, 3)
rmse = (prediction - truth).square().mean().sqrt()
print(f"Forecast RMSE: {rmse.item():.4g}")
```

`washout=200` lets the reservoir synchronize before collecting supervised
training rows. The last observed sample supplies the last shifted target.
`forecast()` uses the saved training history and begins at the first unseen
sample, `truth[0]`. It never needs `truth` to generate those predictions.

The default readout checks numerical safety while preserving the requested
ridge penalty. Use `readout_kwargs={"solver": "svd"}` to force SVD, or choose
another [readout solver](../guides/readout-solvers.md). CUDA placement uses
ordinary `.to("cuda")` on both model and tensors; measure your workload using
[the deployment guide](../workflows/deploy.md).

## Save and reuse the model

```python
model.save("lorenz.pt")
restored = rd.load("lorenz.pt")
replayed = restored.forecast(horizon=len(truth))
torch.testing.assert_close(replayed, prediction, rtol=0, atol=0)
```

The artifact includes the graph, fitted scaling, readout weights, observed
history and current recurrent memory. Loading does not require the original
DAG-building code for built-in components. See [save, load and diagrams](../workflows/save-load.md)
for custom layers and image exports.

## Improve forecast accuracy

The small example above demonstrates the workflow; its parameters are not a
claim of long-range chaotic prediction. Tune on observations reserved for
validation and keep final test samples outside the tuning call:

```python
# Optional: pip install "resdag[hpo]"
# model.tune(training, preset="quick")
# prediction = model.forecast(horizon=len(truth))
```

Tuning refits the winning configuration on the supplied training observations.
Its built-in validation split never uses `truth` above. The balanced and
thorough presets check more configurations, random initializations and
forecast starting points. See [tuning](../workflows/tune.md) for custom search
ranges and objectives.

<figure markdown>
![All three Lorenz components over the complete recorded test window, comparing the observed continuation and autonomous forecast](../assets/figures/first_forecast.png)
<figcaption>A separate recorded Lorenz experiment, with its trajectory, reservoir
seed and forecast start fixed before final evaluation. The full supplied window
is shown in original units. The valid prefix stops at the first
component-normalized RMS error of 0.3; training observations supply the scales.
This image is not a benchmark of the small example above. The
[study protocol and complete results](../project/phase-five-robustness.md)
report all independent trajectories and both 0.3 and 0.5 thresholds.</figcaption>
</figure>

For your own data, use the same `fit`/`forecast` calls with a `(time, signals)`
or `(batch, time, signals)` tensor. [TSDynamics trajectories](trajectory-workflow.md)
carry timestamps and signal names automatically. [Plot and animate a forecast](../workflows/visualize.md)
adds a custom model diagram, component plots and an attractor movie.
