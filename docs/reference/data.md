---
description: Time-series file I/O, preparation, streaming windows and built-in benchmark generators.
---

<span class="nb-kicker">Reference</span>

# Data and datasets

`resdag.data` is the canonical home for time-series preparation, file I/O and
windowed datasets. `resdag.datasets` contains benchmark generators. Existing
imports through `resdag.utils.data` use the same functions.

## Tensor preparation

`prepare_esn_data` accepts `(batch, time, features)` and returns warmup, train,
target, forecast-warmup and validation tensors. Its default `alignment="next"`
includes the final observed training target in forecast warmup, so forecasts
align with the held-out validation tensor. Inspect the alignment option when
working with older experiments.

`normalize_data` fits statistics on exactly the tensor passed to it. To apply
one transformation to later observations, retain those statistics and pass
them back through `stats=`; `denormalize_data` restores the original units.
Keep final test observations out of any statistics fitted for model selection.

::: resdag.data.prepare_esn_data

::: resdag.data.normalize_data

::: resdag.data.denormalize_data

::: resdag.data.load_and_prepare

## Streaming windows

`TimeSeriesWindowDataset` accepts one trajectory, a batched tensor, or a list
of trajectories. Windows never cross trajectory boundaries. With external
targets, input and target windows share their origin; otherwise `horizon`
sets the target offset. `make_dataloader` batches `(inputs, targets, washout)`.
See [streaming and DataLoaders](../workflows/streaming.md) for training loops.

::: resdag.data.TimeSeriesWindowDataset

::: resdag.data.make_dataloader

## Files

Loaders return `(batch, time, features)`. CSV rows represent timesteps and
columns represent features, including single-row and single-column files.
An NPY/NPZ/NetCDF vector is interpreted as a univariate series and emits an
ambiguity warning; store a `(time, features)` array to make the axes explicit.
NPY and NPZ loading use NumPy's default restricted object-array policy.
NetCDF requires the optional `xarray` dependency.

::: resdag.data.load_file

::: resdag.data.load_csv

::: resdag.data.load_npy

::: resdag.data.load_npz

::: resdag.data.load_nc

::: resdag.data.save_csv

::: resdag.data.save_npy

::: resdag.data.save_npz

::: resdag.data.save_nc

::: resdag.data.list_files

## Dataset generators

Each generator takes `n_timesteps` first and returns a `(1, time, features)`
tensor. Continuous Lorenz and Rössler generators use RK4; these convenience
generators are separate from optional TSDynamics trajectory integration.
All generators return raw values by default (`normalize=None`). The common
`rd.ESN.fit` / `rd.Model.fit` workflow learns scaling from training observations.
For raw graph workflows, split first and fit normalization on the training
prefix with `prepare_esn_data(..., normalize=True)` or `normalize_data`.
Explicit `normalize="standard"` or `normalize="minmax"` scales the entire returned
window; use that option only when the complete window belongs to one training
split, or when global normalization is intentionally part of your experiment. Record the
actual `dt`, system parameters, initial-condition/seed settings and discarded
transient alongside any forecast result.

::: resdag.datasets.lorenz

::: resdag.datasets.rossler

::: resdag.datasets.henon

::: resdag.datasets.mackey_glass

::: resdag.datasets.narma

::: resdag.datasets.sine

::: resdag.datasets.kuramoto_sivashinsky

Default sampling intervals and reference exponent lookup remain available
from the existing data utility namespace; explicitly supplied system or
sampling parameters take precedence in an experiment.

::: resdag.utils.data.DATASET_DT

::: resdag.utils.data.dataset_lambda_max