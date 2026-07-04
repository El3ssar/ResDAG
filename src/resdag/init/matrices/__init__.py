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
antisymmetric_matrix
    Random antisymmetric (skew-symmetric) matrix ``A - A^T`` with purely
    imaginary eigenvalues (``"antisymmetric"``).
fast_spectral_initialization
    Recurrent matrix built at a target spectral radius analytically, with no
    eigendecomposition (``"fast_spectral_initialization"``).
delay_line, delay_line_backward, simple_cycle, cycle_jumps, self_loop_cycle
    Deterministic-minimal (Rodan & Tiňo) reservoirs — a pure delay line, a
    delay line with a backward connection, a unidirectional cycle, a cycle with
    regular jumps, and a self-loop cycle (``"delay_line"``,
    ``"delay_line_backward"``, ``"simple_cycle"``, ``"cycle_jumps"``,
    ``"self_loop_cycle"``).
toeplitz, band, block_diagonal
    Structured reservoirs — a Toeplitz (constant-per-diagonal) matrix, a banded
    matrix, and a block-diagonal matrix (``"toeplitz"``, ``"band"``,
    ``"block_diagonal"``).

See Also
--------
resdag.init.graphs : Graph-based topology generators.
resdag.init.topology : Registry and initializer classes.
"""

from .antisymmetric import antisymmetric_matrix
from .fsi import fast_spectral_initialization
from .minimal import (
    cycle_jumps,
    delay_line,
    delay_line_backward,
    self_loop_cycle,
    simple_cycle,
)
from .orthogonal import orthogonal_matrix
from .structured import band, block_diagonal, toeplitz

__all__ = [
    "antisymmetric_matrix",
    "band",
    "block_diagonal",
    "cycle_jumps",
    "delay_line",
    "delay_line_backward",
    "fast_spectral_initialization",
    "orthogonal_matrix",
    "self_loop_cycle",
    "simple_cycle",
    "toeplitz",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
