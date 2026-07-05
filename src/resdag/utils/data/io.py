"""File I/O utilities (compatibility shim).

.. deprecated:: 0.7.0
    This module moved to :mod:`resdag.data.io`. It is re-exported here unchanged
    so ``from resdag.utils.data.io import load_file`` (etc.) keeps working. New
    code should import from :mod:`resdag.data` (or :mod:`resdag.data.io`).

Every name below is the *same* object as its :mod:`resdag.data.io` counterpart.
"""

from ...data.io import (
    PathLike,
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

# Private helpers, re-exported for tests that exercised them at the old path.
# The redundant ``as`` aliases mark these as intentional re-exports for ruff.
from ...data.io import _ensure_3d as _ensure_3d
from ...data.io import _torch_to_numpy_dtype as _torch_to_numpy_dtype
from ...data.io import _warn_if_1d as _warn_if_1d

__all__ = [
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
    "PathLike",
]
