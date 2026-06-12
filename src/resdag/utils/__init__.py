"""
Utility Functions
=================

This module provides utility functions for data loading, preparation,
and general operations in resdag.

Submodules
----------
data
    Data loading and preparation utilities for ESN training.

Functions
---------
create_rng
    Create a random number generator with optional seed.
load_file
    Load a time series from disk (re-exported from ``resdag.utils.data``).
prepare_esn_data
    Split a series into ESN train/forecast segments (re-exported from
    ``resdag.utils.data``).

Examples
--------
>>> import resdag as rd
>>> data = rd.utils.load_file("timeseries.csv")
>>> warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
...     data, warmup_steps=100, train_steps=500, val_steps=200
... )

See Also
--------
resdag.utils.data : Data loading and preparation.
"""

from . import data
from .data import load_file, prepare_esn_data
from .general import create_rng

__all__ = ["create_rng", "data", "load_file", "prepare_esn_data"]
