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

Examples
--------
>>> from resdag.utils.data import load_file, prepare_esn_data
>>> data = load_file("timeseries.csv")
>>> warmup, train, target, f_warmup, val = prepare_esn_data(
...     data, warmup_steps=100, train_steps=500, val_steps=200
... )

See Also
--------
resdag.utils.data : Data loading and preparation.
"""

from . import data
from .general import create_rng

__all__ = ["create_rng", "data"]
