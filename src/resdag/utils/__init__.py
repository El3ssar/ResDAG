"""
Utility Functions
=================

This module provides utility functions for data loading, preparation,
and general operations in resdag.

Submodules
----------
data
    Data loading and preparation utilities for ESN training.
states
    Reservoir-state diagnostics (the Echo State Property index).

Functions
---------
seed_everything
    Seed Python, NumPy, and torch (CPU + CUDA) for reproducible runs.
resolve_device
    Resolve a device spec (``'auto'``/str/``torch.device``) to a ``torch.device``.
create_rng
    Create a NumPy random number generator with optional seed.
create_torch_generator
    Create a torch ``Generator`` for reproducible weight/bias draws.
coerce_seed_to_int
    Reduce an int/``torch.Generator``/None seed to a plain int (or None).
SeedLike
    Type alias for accepted seed inputs (``int | torch.Generator | None``).
DeviceLike
    Type alias for accepted device specs (``str | torch.device | None``).
load_file
    Load a time series from disk (re-exported from ``resdag.utils.data``).
prepare_esn_data
    Split a series into ESN train/forecast segments (re-exported from
    ``resdag.utils.data``).
normalize_data
    Normalize a series and return the fitted statistics (re-exported from
    ``resdag.utils.data``).
denormalize_data
    Invert a normalization back to the original scale (re-exported from
    ``resdag.utils.data``).
lorenz, rossler, henon, mackey_glass, narma, sine
    Canonical reservoir-computing dataset generators (re-exported from
    ``resdag.utils.data``).
esp_index
    Echo State Property index — the library's signature stability diagnostic
    (re-exported from ``resdag.utils.states``).

Examples
--------
>>> import resdag as rd
>>> data = rd.utils.load_file("timeseries.csv")
>>> warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
...     data, warmup_steps=100, train_steps=500, val_steps=200
... )

>>> series = rd.utils.lorenz(2000)  # (1, 2000, 3) chaotic benchmark series

>>> from resdag.utils import esp_index  # signature stability diagnostic

See Also
--------
resdag.utils.data : Data loading and preparation.
resdag.utils.states : Echo State Property diagnostics.
"""

from . import data, states
from .data import (
    denormalize_data,
    henon,
    load_file,
    lorenz,
    mackey_glass,
    narma,
    normalize_data,
    prepare_esn_data,
    rossler,
    sine,
)
from .general import (
    DeviceLike,
    SeedLike,
    coerce_seed_to_int,
    create_rng,
    create_torch_generator,
    resolve_device,
    seed_everything,
)
from .states import esp_index

__all__ = [
    "DeviceLike",
    "SeedLike",
    "coerce_seed_to_int",
    "create_rng",
    "create_torch_generator",
    "resolve_device",
    "seed_everything",
    "data",
    "states",
    "esp_index",
    "load_file",
    "prepare_esn_data",
    "normalize_data",
    "denormalize_data",
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
