"""
Matrix-Builder Topologies
=========================

Direct matrix-construction functions registered as topologies — the
non-graph counterpart of :mod:`resdag.init.graphs`. Each function takes the
matrix size ``n`` first, returns an ``(n, n)`` matrix, and is registered via
:func:`~resdag.init.topology.register_matrix_topology`, making it available
by name in ``ESNLayer(topology="...")``.

Functions
---------
orthogonal_matrix
    Haar-random orthogonal matrix via QR decomposition (``"orthogonal"``).
fast_spectral_initialization
    Recurrent matrix built at a target spectral radius analytically, with no
    eigendecomposition (``"fast_spectral_initialization"``).

See Also
--------
resdag.init.graphs : Graph-based topology generators.
resdag.init.topology : Registry and initializer classes.
"""

from .fsi import fast_spectral_initialization
from .orthogonal import orthogonal_matrix

__all__ = [
    "fast_spectral_initialization",
    "orthogonal_matrix",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
