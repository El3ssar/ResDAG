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
    Create a NumPy random number generator with optional seed.
create_torch_generator
    Create a torch ``Generator`` for reproducible weight/bias draws.
coerce_seed_to_int
    Reduce an int/``torch.Generator``/None seed to a plain int (or None).
SeedLike
    Type alias for accepted seed inputs (``int | torch.Generator | None``).
load_file
    Load a time series from disk (re-exported from ``resdag.utils.data``).
prepare_esn_data
    Split a series into ESN train/forecast segments (re-exported from
    ``resdag.utils.data``).
lorenz, rossler, henon, mackey_glass, narma, sine
    Canonical reservoir-computing dataset generators (re-exported from
    ``resdag.utils.data``).

Examples
--------
>>> import resdag as rd
>>> data = rd.utils.load_file("timeseries.csv")
>>> warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
...     data, warmup_steps=100, train_steps=500, val_steps=200
... )

>>> series = rd.utils.lorenz(2000)  # (1, 2000, 3) chaotic benchmark series

See Also
--------
resdag.utils.data : Data loading and preparation.
"""

from . import data
from .data import (
    henon,
    load_file,
    lorenz,
    mackey_glass,
    narma,
    prepare_esn_data,
    rossler,
    sine,
)
from .general import SeedLike, coerce_seed_to_int, create_rng, create_torch_generator

__all__ = [
    "SeedLike",
    "coerce_seed_to_int",
    "create_rng",
    "create_torch_generator",
    "data",
    "load_file",
    "prepare_esn_data",
    # Canonical dataset generators
    "lorenz",
    "rossler",
    "henon",
    "mackey_glass",
    "narma",
    "sine",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
