"""
Backward compatibility shim — ``resdag.layers.custom`` moved to ``resdag.layers.transforms``.

All names are re-exported from :mod:`resdag.layers.transforms` so existing
code continues to work without modification.  New code should import from
``resdag.layers.transforms`` directly.
"""

import warnings as _warnings

_warnings.warn(
    "resdag.layers.custom is deprecated; import from resdag.layers.transforms instead. "
    "resdag.layers.custom will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from resdag.layers.transforms import (  # noqa: F401, E402
    Concatenate,
    FeaturePartitioner,
    Power,
    SelectiveDropout,
    SelectiveExponentiation,
)

__all__ = [
    "Concatenate",
    "FeaturePartitioner",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
]
