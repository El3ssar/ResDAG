"""
Backward compatibility shim — ``resdag.composition`` moved to ``resdag.core``.

All names are re-exported from :mod:`resdag.core` so existing code continues
to work without modification.  New code should import from ``resdag.core``
directly.
"""

import warnings as _warnings

_warnings.warn(
    "resdag.composition is deprecated; import from resdag.core instead. "
    "resdag.composition will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from resdag.core import ESNModel, Input, reservoir_input  # noqa: F401, E402

__all__ = ["ESNModel", "Input", "reservoir_input"]
