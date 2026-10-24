"""Data preparation utilities (compatibility shim).

**Deprecated since 0.7.0.**

This module moved to `resdag.data.prepare`. It is re-exported here
unchanged so `from resdag.utils.data.prepare import prepare_esn_data` (etc.)
keeps working. New code should import from `resdag.data` (or
`resdag.data.prepare`).

Every name below is the *same* object as its `resdag.data.prepare`
counterpart.
"""

from ...data.prepare import (
    ESNDataSplits,
    ESNDataSplitsWithStats,
    NormMethod,
    denormalize_data,
    load_and_prepare,
    normalize_data,
    prepare_esn_data,
)

__all__ = [
    "prepare_esn_data",
    "normalize_data",
    "denormalize_data",
    "load_and_prepare",
    "NormMethod",
    "ESNDataSplits",
    "ESNDataSplitsWithStats",
]
