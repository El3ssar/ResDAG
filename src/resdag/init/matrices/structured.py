r"""Structured reservoir matrices (Toeplitz, banded, block-diagonal).

Structured recurrent matrices trade the isotropy of a random reservoir for a
fixed, interpretable connectivity skeleton — a constant along each diagonal
(Toeplitz), a fixed-bandwidth neighbourhood (banded), or independent blocks
(block-diagonal). They complement the deterministic-minimal family
(:mod:`resdag.init.matrices.minimal`) and mirror the structured initializers
shipped by ``ReservoirComputing.jl`` (``toepliz_init``, ``band_init``,
``block_diagonal`` in ``src/inits/inits_reservoir.jl``).

All three builders here are **deterministic and fully populated** within their
structure, which makes their pattern exact and reproducible with no seed. This
is a deliberate simplification of RC.jl's ``toepliz_init`` / ``band_init``,
which populate their diagonals at *random* positions to hit a sparsity target;
the fully-populated deterministic form is the cleaner, unambiguous primitive and
still scales to any target spectral radius via the layer-level rescale.
``block_diagonal`` matches RC.jl entry-for-entry: each block is a solid block of
a single constant weight.

See Also
--------
resdag.init.matrices.minimal : Deterministic-minimal (Rodan & Tiňo) matrices.
resdag.init.matrices.orthogonal : Haar-random orthogonal recurrent matrix.
resdag.init.matrices.fsi : Analytically spectral-radius-targeted random matrix.
"""

import torch

from resdag.init.topology import register_matrix_topology


def _validate_size(n: int) -> None:
    """Raise if ``n`` is not a usable matrix size."""
    if n <= 0:
        raise ValueError(f"Matrix size n must be positive, got {n}")


@register_matrix_topology("toeplitz", weight=0.1, decay=1.0, bandwidth=None)
def toeplitz(
    n: int,
    weight: float = 0.1,
    decay: float = 1.0,
    bandwidth: int | None = None,
) -> torch.Tensor:
    r"""Build a deterministic Toeplitz recurrent matrix (constant per diagonal).

    A Toeplitz matrix is constant along each diagonal: ``M[i, j]`` depends only
    on the offset ``k = j - i``. Here diagonal ``k`` takes the value
    ``weight * decay ** |k|``, so the main diagonal is ``weight`` and off-
    diagonals fall off geometrically with ``decay`` (``decay=1.0`` gives a flat
    Toeplitz with the same constant on every diagonal). Optionally restricted to
    a band of half-width ``bandwidth`` (diagonals with ``|k| > bandwidth`` are
    zero); ``bandwidth=None`` fills every diagonal (a dense Toeplitz).

    This is a deterministic, fully-populated Toeplitz — the clean structural
    primitive behind RC.jl's ``toepliz_init`` (which instead scatters one random
    value per diagonal to hit a sparsity target). Because the entry depends only
    on ``|k|`` (both the sub- and super-diagonal ``±k`` get the same value), the
    matrix is **symmetric**; its spectral radius is nonzero, so a layer-level
    ``spectral_radius`` rescales it cleanly.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    weight : float, default=0.1
        Value of the main diagonal, and the base value scaled by ``decay`` for
        off-diagonals.
    decay : float, default=1.0
        Geometric fall-off applied to diagonal ``k`` as ``decay ** |k|``. With
        ``1.0`` every diagonal shares the constant ``weight``; ``0 < decay < 1``
        concentrates weight near the main diagonal.
    bandwidth : int or None, default=None
        If given, only diagonals with ``|k| <= bandwidth`` are populated (a
        banded Toeplitz). ``None`` fills all diagonals. Must be ``>= 0``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` Toeplitz matrix.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="toeplitz", spectral_radius=0.9
    ... )
    >>> reservoir = ESNLayer(
    ...     500,
    ...     feedback_size=3,
    ...     topology=("toeplitz", {"decay": 0.5, "bandwidth": 5}),
    ... )

    References
    ----------
    RC.jl ``toepliz_init`` (Toeplitz-structured minimal-complexity ESN reservoir,
    Cossu et al. 2025); this builder is the deterministic fully-populated form.
    """
    _validate_size(n)
    if bandwidth is not None and bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")

    rows = torch.arange(n).view(n, 1)
    cols = torch.arange(n).view(1, n)
    offset = (cols - rows).abs()  # |k| for each entry

    matrix = float(weight) * (float(decay) ** offset.to(torch.float32))
    if bandwidth is not None:
        matrix = torch.where(offset <= bandwidth, matrix, torch.zeros_like(matrix))
    return matrix.to(torch.float32)


@register_matrix_topology("band", weight=0.1, bandwidth=1)
def band(n: int, weight: float = 0.1, bandwidth: int = 1) -> torch.Tensor:
    r"""Build a deterministic banded recurrent matrix of constant weight.

    Every entry within ``bandwidth`` of the main diagonal (``|j - i| <=
    bandwidth``) is set to the constant ``weight``; everything outside the band
    is zero. ``bandwidth=1`` gives a tridiagonal matrix (main diagonal plus the
    first sub- and super-diagonals); ``bandwidth=0`` degenerates to a scaled
    identity. This is a local-connectivity reservoir where each unit couples only
    to its ``bandwidth`` nearest neighbours on either side.

    This is the deterministic fully-populated form of RC.jl's ``band_init``
    (which scatters random values across the band to a sparsity target). The
    matrix is symmetric with a nonzero spectral radius, so a layer-level
    ``spectral_radius`` rescales it cleanly.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    weight : float, default=0.1
        Constant value assigned to every in-band entry.
    bandwidth : int, default=1
        Half-bandwidth: entries with ``|j - i| <= bandwidth`` are populated.
        Must satisfy ``0 <= bandwidth < n``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` banded matrix.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="band", spectral_radius=0.9
    ... )
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology=("band", {"bandwidth": 3})
    ... )

    References
    ----------
    RC.jl ``band_init`` (banded minimal-complexity ESN reservoir); this builder
    is the deterministic fully-populated form.
    """
    _validate_size(n)
    if not 0 <= bandwidth < n:
        raise ValueError(f"bandwidth must satisfy 0 <= bandwidth < n={n}, got {bandwidth}")

    rows = torch.arange(n).view(n, 1)
    cols = torch.arange(n).view(1, n)
    in_band = (cols - rows).abs() <= bandwidth
    return torch.where(
        in_band,
        torch.full((n, n), float(weight), dtype=torch.float32),
        torch.zeros((n, n), dtype=torch.float32),
    )


@register_matrix_topology("block_diagonal", weight=1.0, block_size=1)
def block_diagonal(n: int, weight: float = 1.0, block_size: int = 1) -> torch.Tensor:
    r"""Build a block-diagonal recurrent matrix of constant-valued blocks.

    Tiles the main diagonal with ``floor(n / block_size)`` square blocks of size
    ``block_size``, each block a solid block of the constant ``weight``; the rest
    of the matrix (and any leftover rows/columns when ``block_size`` does not
    divide ``n``) is zero. The reservoir then splits into independent
    ``block_size``-dimensional sub-reservoirs. Ports RC.jl's ``block_diagonal``
    entry-for-entry: each ``block_size × block_size`` diagonal block is filled
    with a single constant (RC.jl ``block_weight``, default ``1``; ``block_size``
    default ``1``).

    A single block of constant ``weight`` has rank 1 with eigenvalue
    ``weight * block_size`` (and zeros), so the whole matrix has spectral radius
    ``|weight| * block_size`` and a layer-level ``spectral_radius`` rescales it
    cleanly. When ``block_size`` does not divide ``n`` the leftover trailing
    rows/columns are left zero (matching RC.jl, which warns and drops them).

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    weight : float, default=1.0
        Constant value filling every block. RC.jl ``block_weight`` default ``1``.
    block_size : int, default=1
        Side length of each diagonal block. RC.jl ``block_size`` default ``1``
        (which yields a diagonal matrix). Must satisfy ``1 <= block_size <= n``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` block-diagonal matrix.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="block_diagonal", spectral_radius=0.9
    ... )
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology=("block_diagonal", {"block_size": 5})
    ... )

    References
    ----------
    RC.jl ``block_diagonal`` (block-diagonal reservoir).
    """
    _validate_size(n)
    if not 1 <= block_size <= n:
        raise ValueError(f"block_size must satisfy 1 <= block_size <= n={n}, got {block_size}")

    matrix = torch.zeros(n, n, dtype=torch.float32)
    num_blocks = n // block_size
    value = float(weight)
    for block in range(num_blocks):
        start = block * block_size
        end = start + block_size
        matrix[start:end, start:end] = value
    return matrix
