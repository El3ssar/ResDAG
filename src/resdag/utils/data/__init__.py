"""
Data Loading and Preparation (compatibility shim)
=================================================

.. deprecated:: 0.7.0
    The canonical home for data preparation and file I/O is now
    :mod:`resdag.data`. This module re-exports those symbols unchanged for
    back-compat, so ``from resdag.utils.data import prepare_esn_data`` (etc.)
    keeps working. New code should import from :mod:`resdag.data`.

The re-exported callables are the *same* function objects as their
:mod:`resdag.data` counterparts, so identity checks hold::

    resdag.utils.data.prepare_esn_data is resdag.data.prepare_esn_data  # True

The canonical dataset generators (:func:`lorenz`, :func:`rossler`, …) still
live here and are re-exported by :mod:`resdag.datasets`.

File I/O Functions
------------------
load_file
    Auto-detect format and load time series data.
load_csv, load_npy, load_npz, load_nc
    Format-specific loaders.
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
Loading and preparing data (canonical home is :mod:`resdag.data`):

>>> from resdag.data import load_file, prepare_esn_data
>>> data = load_file("lorenz.csv")  # doctest: +SKIP
>>> warmup, train, target, f_warmup, val = prepare_esn_data(  # doctest: +SKIP
...     data, warmup_steps=100, train_steps=500, val_steps=200, normalize="minmax"
... )

Generating a canonical benchmark series:

>>> from resdag.datasets import lorenz
>>> series = lorenz(2000)  # (1, 2000, 3) — chaotic Lorenz-63 trajectory

See Also
--------
resdag.data : Canonical home for data preparation, file I/O, and streaming windows.
resdag.datasets : Canonical benchmark time-series generators.
"""

# Data prep / file I/O now live in ``resdag.data``; re-exported here unchanged.
from ...data.io import (
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
from ...data.prepare import (
    denormalize_data,
    load_and_prepare,
    normalize_data,
    prepare_esn_data,
)

# Canonical dataset generators still live in this package.
from .datasets import (
    DATASET_DT,
    dataset_lambda_max,
    henon,
    lorenz,
    mackey_glass,
    narma,
    rossler,
    sine,
)

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
    # Dataset metadata
    "DATASET_DT",
    "dataset_lambda_max",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
