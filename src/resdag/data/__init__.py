"""
Data Pipeline: Preparation, I/O, and Streaming Windows
======================================================

Canonical public home for resdag's data plumbing. Three layers live here:

**Whole-tensor preparation** — split a long trajectory into the fixed
warmup/train/target/f_warmup/val tensors the algebraic (`resdag.training.ESNTrainer`)
path consumes, with optional normalization.

**File I/O** — load and save time series from CSV / NPY / NPZ / NetCDF, always
returning the `(B, T, D)` convention.

**Streaming / SGD windows** — slice a long trajectory (or many) into
fixed-length, batchable windows behind a standard
`torch.utils.data.DataLoader`, for the SGD and streaming-algebraic paths.

Classes
-------
TimeSeriesWindowDataset
    Sliding-window `Dataset` over a `(T, D)` tensor, a `(B, T, D)` tensor,
    or a list of variable-length trajectories. Yields
    `(input_window, target_window, washout)` triples that never straddle a
    trajectory boundary.

Functions
---------
make_dataloader
    Wrap `TimeSeriesWindowDataset` in a `DataLoader` whose batches are
    `(B, window_len, D)` and feed straight into `ESNLayer` / `ESNModel`.
prepare_esn_data
    Split a series into warmup, train, target, f_warmup, val.
normalize_data, denormalize_data
    Normalize a series (and invert it) using various methods.
load_and_prepare
    Load and prepare data from file(s) in one step.
load_file, load_csv, load_npy, load_npz, load_nc
    Auto-detecting and format-specific loaders.
save_csv, save_npy, save_npz, save_nc, list_files
    Savers and a directory listing helper.

Examples
--------
Whole-tensor preparation for the algebraic path:

>>> from resdag.data import load_file, prepare_esn_data
>>> data = load_file("lorenz.csv")  # doctest: +SKIP
>>> warmup, train, target, f_warmup, val = prepare_esn_data(  # doctest: +SKIP
...     data, warmup_steps=100, train_steps=500, val_steps=200, normalize=True, norm_method="minmax"
... )

Streaming windows for the SGD path:

>>> import torch
>>> from resdag.data import make_dataloader
>>> series = torch.randn(2000, 3)
>>> loader = make_dataloader(series, batch_size=16, window_len=200, washout=50)
>>> for x, y, washout in loader:  # doctest: +SKIP
...     model.reset_reservoirs()
...     pred = model(x)
...     loss = mse(pred[:, washout:], y[:, washout:])

See Also
--------
resdag.datasets : Canonical benchmark time-series generators.
resdag.layers.IncrementalRidgeReadoutLayer : Streaming ridge readout for the loader.
"""

from .dataset import TimeSeriesWindowDataset, make_dataloader
from .io import (
    list_files,
    load_csv,
    load_file,
    load_nc,
    load_npy,
    load_npz,
    save_csv,
    save_nc,
    save_npy,
    save_npz,
)
from .prepare import denormalize_data, load_and_prepare, normalize_data, prepare_esn_data

__all__ = [
    # Streaming / SGD data pipeline
    "TimeSeriesWindowDataset",
    "make_dataloader",
    # Data preparation
    "prepare_esn_data",
    "normalize_data",
    "denormalize_data",
    "load_and_prepare",
    # File I/O
    "load_file",
    "load_csv",
    "load_npy",
    "load_npz",
    "load_nc",
    "save_csv",
    "save_npy",
    "save_npz",
    "save_nc",
    "list_files",
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
