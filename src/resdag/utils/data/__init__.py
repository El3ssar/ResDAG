"""
Data Loading and Preparation
============================

This module provides utilities for loading time series data and preparing
it for ESN training and forecasting.

File I/O Functions
------------------
load_file
    Auto-detect format and load time series data.
load_csv
    Load data from CSV file.
load_npy
    Load data from NumPy .npy file.
load_npz
    Load data from NumPy .npz archive.
load_nc
    Load data from NetCDF file.
save_csv, save_npy, save_npz, save_nc
    Save data in respective formats.
list_files
    List files matching a pattern.

Data Preparation Functions
--------------------------
prepare_esn_data
    Split time series into warmup, train, target, f_warmup, val.
normalize_data
    Normalize data using various methods.
denormalize_data
    Invert a normalization, mapping data back to its original scale.
load_and_prepare
    Load and prepare data in one step.

Canonical Dataset Generators
----------------------------
lorenz, rossler
    Chaotic continuous-time attractors (RK4 integration).
henon
    Chaotic discrete-time map.
mackey_glass
    Chaotic delay-differential-equation series.
narma
    Nonlinear system-identification benchmark (input/output series).
sine
    Simple periodic signal for sanity checks.

Each generator returns a ``(1, T, D)`` tensor, is seedable via ``create_rng``,
supports a transient/discard period, and supports optional normalization.

Examples
--------
Loading and preparing data:

>>> from resdag.utils.data import load_file, prepare_esn_data
>>> data = load_file("lorenz.csv")
>>> warmup, train, target, f_warmup, val = prepare_esn_data(
...     data,
...     warmup_steps=100,
...     train_steps=500,
...     val_steps=200,
...     normalize="minmax",
... )

Using load_and_prepare for convenience:

>>> from resdag.utils.data import load_and_prepare
>>> splits = load_and_prepare(
...     "lorenz.csv",
...     warmup_steps=100,
...     train_steps=500,
...     val_steps=200,
... )
>>> warmup, train, target, f_warmup, val = splits

Generating a canonical benchmark series:

>>> from resdag.utils.data import lorenz
>>> series = lorenz(2000)  # (1, 2000, 3) — chaotic Lorenz-63 trajectory

See Also
--------
resdag.training.ESNTrainer : Uses prepared data for training.
resdag.core.ESNModel.forecast : Uses prepared data for forecasting.
"""

from .datasets import henon, lorenz, mackey_glass, narma, rossler, sine
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
    # Data preparation
    "prepare_esn_data",
    "normalize_data",
    "denormalize_data",
    "load_and_prepare",
    # Canonical dataset generators
    "lorenz",
    "rossler",
    "henon",
    "mackey_glass",
    "narma",
    "sine",
]
