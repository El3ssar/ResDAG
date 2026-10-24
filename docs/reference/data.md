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

::: resdag.data
    options:
      members:
        - prepare_esn_data
        - normalize_data
        - denormalize_data
        - load_and_prepare

## Streaming windows

`TimeSeriesWindowDataset` accepts one trajectory, a batched tensor, or a list
of trajectories. Windows never cross trajectory boundaries. With external
targets, input and target windows share their origin; otherwise `horizon`
sets the target offset. `make_dataloader` batches `(inputs, targets, washout)`.
See [streaming and DataLoaders](../workflows/streaming.md) for training loops.

::: resdag.data
    options:
      members:
        - TimeSeriesWindowDataset
        - make_dataloader

## Files

Loaders return `(batch, time, features)`. CSV rows represent timesteps and
columns represent features, including single-row and single-column files.
An NPY/NPZ/NetCDF vector is interpreted as a univariate series and emits an
ambiguity warning; store a `(time, features)` array to make the axes explicit.
NPY and NPZ loading use NumPy's default restricted object-array policy.
NetCDF requires the optional `xarray` dependency.

::: resdag.data
    options:
      members:
        - load_file
        - load_csv
        - load_npy
        - load_npz
        - load_nc
        - save_csv
        - save_npy
        - save_npz
        - save_nc
        - list_files

## Dataset generators

Each generator takes `n_timesteps` first and returns a `(1, time, features)`
tensor. Continuous Lorenz and Rössler generators use RK4; these convenience
generators are separate from optional TSDynamics trajectory integration.
Generator normalization acts on the returned window. Use `normalize=None`
when fitting normalization separately on a training prefix. Record the
actual `dt`, system parameters, initial-condition/seed settings and discarded
transient alongside any forecast result.

::: resdag.datasets
    options:
      members:
        - lorenz
        - rossler
        - henon
        - mackey_glass
        - narma
        - sine
        - kuramoto_sivashinsky

Default sampling intervals and reference exponent lookup remain available
from the existing data utility namespace; explicitly supplied system or
sampling parameters take precedence in an experiment.

::: resdag.utils.data
    options:
      members:
        - DATASET_DT
        - dataset_lambda_max
