r"""Deterministic Toeplitz, banded, and block-diagonal reservoir matrices.

These builders expose connectivity patterns directly. All return dense tensors
on the requested device, and an outer spectral-radius rescale preserves their
patterns. Dense storage and multiplication remain O(n²).
"""

import torch

from resdag.init.topology import register_matrix_topology

from ._common import matrix_device


@register_matrix_topology("toeplitz", weight=0.1, decay=1.0, bandwidth=None)
def toeplitz(
    n: int,
    weight: float = 0.1,
    decay: float = 1.0,
    bandwidth: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Set ``M[i,j] = weight * decay**abs(i-j)`` inside an optional band.

    Every diagonal is constant, and the matrix is symmetric. ``bandwidth=None``
    fills all diagonals; otherwise only offsets up to a nonnegative bandwidth
    are populated. This is a deterministic geometric Toeplitz construction.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer.
    weight : float, default=0.1
        Main-diagonal weight and the scale of every other diagonal.
    decay : float, default=1.0
        Geometric factor per diagonal offset. ``decay=1`` fills each diagonal
        with the same weight; ``0 < decay < 1`` weakens distant connections.
        The sign alternates across offsets when decay is negative.
    bandwidth : int or None, default=None
        Maximum populated absolute offset. Must be nonnegative when supplied;
        None keeps every diagonal. Zero gives a scaled identity.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output and power-computation dtype.
    device : torch.device or str, optional
        Allocation and computation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` symmetric Toeplitz matrix with optional band masking.

    Raises
    ------
    ValueError
        If size, bandwidth, or dtype is invalid.

    Examples
    --------
    >>> matrix = toeplitz(6, decay=0.5, bandwidth=2, dtype=torch.float64)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3,
    ...     recurrent_initializer=("toeplitz", {"decay": 0.5, "bandwidth": 5}),
    ... )
    """
    target = matrix_device(n, dtype, device)
    if bandwidth is not None and bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")
    rows = torch.arange(n, device=target).view(n, 1)
    cols = torch.arange(n, device=target).view(1, n)
    offset = (cols - rows).abs()
    matrix = float(weight) * (float(decay) ** offset.to(dtype))
    if bandwidth is not None:
        matrix = torch.where(offset <= bandwidth, matrix, torch.zeros_like(matrix))
    return matrix


@register_matrix_topology("band", weight=0.1, bandwidth=1)
def band(
    n: int,
    weight: float = 0.1,
    bandwidth: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fill ``abs(i-j) <= bandwidth`` with ``weight`` and the rest with zero.

    Requires ``0 <= bandwidth < n``. Zero gives a scaled identity; one gives
    a tridiagonal matrix. The in-band weights are deterministic and constant.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer.
    weight : float, default=0.1
        Constant assigned to every in-band entry, including the main diagonal.
    bandwidth : int, default=1
        Half-bandwidth, satisfying ``0 <= bandwidth < n``. Two populates the
        main diagonal and two diagonals on either side.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense symmetric ``(n, n)`` matrix with constant in-band weights.

    Raises
    ------
    ValueError
        If size, bandwidth, or dtype is invalid.

    Examples
    --------
    >>> matrix = band(6, weight=0.2, bandwidth=2)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3,
    ...     recurrent_initializer=("band", {"bandwidth": 3}),
    ... )
    """
    target = matrix_device(n, dtype, device)
    if not 0 <= bandwidth < n:
        raise ValueError(f"bandwidth must satisfy 0 <= bandwidth < n={n}, got {bandwidth}")
    rows = torch.arange(n, device=target).view(n, 1)
    cols = torch.arange(n, device=target).view(1, n)
    in_band = (cols - rows).abs() <= bandwidth
    return torch.where(
        in_band,
        torch.full((n, n), float(weight), dtype=dtype, device=target),
        torch.zeros((n, n), dtype=dtype, device=target),
    )


@register_matrix_topology("block_diagonal", weight=1.0, block_size=1)
def block_diagonal(
    n: int,
    weight: float = 1.0,
    block_size: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fill complete diagonal blocks of size ``block_size`` with ``weight``.

    Requires ``1 <= block_size <= n``. Incomplete trailing rows/columns are
    left zero. Each full block has rank at most one and eigenvalues
    ``weight * block_size`` plus zeros. This isolates independent groups of
    reservoir units while retaining the original constant-block behavior.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer.
    weight : float, default=1.0
        Constant filling every element of each complete block.
    block_size : int, default=1
        Square block side length, satisfying ``1 <= block_size <= n``. One
        gives a scaled identity. Any trailing incomplete block is left zero.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` tensor with ``floor(n / block_size)`` constant blocks.
        When weight is nonzero its rank is the number of complete blocks.

    Raises
    ------
    ValueError
        If size, block_size, or dtype is invalid.

    Examples
    --------
    >>> matrix = block_diagonal(7, weight=0.5, block_size=3)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3,
    ...     recurrent_initializer=("block_diagonal", {"block_size": 5}),
    ... )
    """
    target = matrix_device(n, dtype, device)
    if not 1 <= block_size <= n:
        raise ValueError(f"block_size must satisfy 1 <= block_size <= n={n}, got {block_size}")
    matrix = torch.zeros(n, n, dtype=dtype, device=target)
    for start in range(0, n - block_size + 1, block_size):
        matrix[start : start + block_size, start : start + block_size] = float(weight)
    return matrix
