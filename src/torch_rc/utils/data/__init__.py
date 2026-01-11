"""Data loading and preparation utilities for torch_rc.

This module provides utilities for:
- Loading time series data from various file formats
- Preparing data for ESN training and forecasting
- Normalization and data splitting

Example:
    >>> from torch_rc.utils.data import load_file, prepare_esn_data
    >>> data = load_file("timeseries.csv")
    >>> warmup, train, target, forecast_warmup, val = prepare_esn_data(
    ...     data, warmup_steps=100, train_steps=500, val_steps=200
    ... )
"""

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
from .prepare import load_and_prepare, normalize_data, prepare_esn_data

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
    "load_and_prepare",
]
