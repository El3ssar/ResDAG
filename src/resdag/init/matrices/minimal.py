r"""Deterministic delay-line and cycle reservoirs.

The matrices use fixed connections and require no random seed. A cycle has
spectral radius equal to the magnitude of its edge weight; a delay line is
nilpotent, so scaling cannot give it a nonzero spectral radius. All builders
return dense tensors on the requested device and retain the requested dtype.

References
----------
A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE
Transactions on Neural Networks*, 22(1):131–144, 2011.

A. Rodan and P. Tiňo, "Simple Deterministically Constructed Cycle Reservoirs
with Regular Jumps", *Neural Computation*, 24(7):1822–1852, 2012.
"""

import torch

from resdag.init.topology import register_matrix_topology

from ._common import matrix_device


def _build_simple_cycle(
    n: int, weight: float, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Place a subdiagonal and wrap-around corner of constant weight."""
    matrix = torch.zeros(n, n, dtype=dtype, device=device)
    idx = torch.arange(n - 1, device=device)
    matrix[idx + 1, idx] = weight
    matrix[0, n - 1] = weight
    return matrix


@register_matrix_topology("delay_line", weight=0.1, shift=1)
def delay_line(
    n: int,
    weight: float = 0.1,
    shift: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Place `weight` on a subdiagonal `shift` places below the diagonal.

    Requires `1 <= shift < n`. The matrix is nilpotent: its eigenvalues are
    all zero. Control its scale through `weight`; an outer spectral-radius
    rescale cannot assign it a nonzero radius.

    Parameters
    ----------
    n : int
        Number of reservoir units; a positive integer greater than `shift`.
    weight : float, default=0.1
        Constant weight of every forward connection. It controls how much
        past activity remains after each shift along the delay line.
    shift : int, default=1
        Subdiagonal offset. One connects each unit to its next neighbour;
        larger offsets skip units. Must satisfy `1 <= shift < n`.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype; float64 avoids a float32 conversion.
    device : torch.device or str, optional
        Allocation device, CPU by default. CUDA/MPS allocation is direct.

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` tensor with the chosen subdiagonal and zeros elsewhere.

    Raises
    ------
    ValueError
        If `n` is invalid, `shift` is outside its range, or dtype is not
        real floating-point.

    Examples
    --------
    >>> matrix = delay_line(6, weight=0.5, shift=2, dtype=torch.float64)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3, spectral_radius=None,
    ...     recurrent_initializer=("delay_line", {"weight": 0.5}),
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE
    Transactions on Neural Networks*, 22(1):131–144, 2011.
    """
    target = matrix_device(n, dtype, device)
    if not 1 <= shift < n:
        raise ValueError(f"shift must satisfy 1 <= shift < n={n}, got {shift}")
    matrix = torch.zeros(n, n, dtype=dtype, device=target)
    idx = torch.arange(n - shift, device=target)
    matrix[idx + shift, idx] = float(weight)
    return matrix


@register_matrix_topology(
    "delay_line_backward", weight=0.1, backward_weight=0.1, shift=1, backward_shift=1
)
def delay_line_backward(
    n: int,
    weight: float = 0.1,
    backward_weight: float = 0.1,
    shift: int = 1,
    backward_shift: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a forward delay line plus a backward superdiagonal connection.

    `shift` and `backward_shift` each satisfy `1 <= offset < n`. The
    weights and offsets of the two directions are independent.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer.
    weight : float, default=0.1
        Weight on the forward subdiagonal.
    backward_weight : float, default=0.1
        Weight on the reverse superdiagonal. A negative weight changes the
        sign of backward feedback independently of the forward path.
    shift : int, default=1
        Forward subdiagonal offset, satisfying `1 <= shift < n`.
    backward_shift : int, default=1
        Backward superdiagonal offset, satisfying `1 <= backward_shift < n`.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` tensor with one subdiagonal and one superdiagonal.

    Raises
    ------
    ValueError
        If size, offsets, or dtype are invalid.

    Examples
    --------
    >>> matrix = delay_line_backward(6, weight=0.1, backward_weight=-0.3)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3,
    ...     recurrent_initializer=("delay_line_backward", {"backward_weight": -1.0}),
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE
    Transactions on Neural Networks*, 22(1):131–144, 2011.
    """
    target = matrix_device(n, dtype, device)
    if not 1 <= shift < n:
        raise ValueError(f"shift must satisfy 1 <= shift < n={n}, got {shift}")
    if not 1 <= backward_shift < n:
        raise ValueError(
            f"backward_shift must satisfy 1 <= backward_shift < n={n}, got {backward_shift}"
        )
    matrix = torch.zeros(n, n, dtype=dtype, device=target)
    fwd = torch.arange(n - shift, device=target)
    matrix[fwd + shift, fwd] = float(weight)
    bwd = torch.arange(n - backward_shift, device=target)
    matrix[bwd, bwd + backward_shift] = float(backward_weight)
    return matrix


@register_matrix_topology("simple_cycle", weight=0.1)
def simple_cycle(
    n: int,
    weight: float = 0.1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build one directed ring with constant `weight` on every edge.

    The subdiagonal and top-right corner form a cyclic permutation scaled by
    `weight`. Every eigenvalue has magnitude `abs(weight)`; a requested
    outer radius rescales the cycle without changing its connections.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer. One gives a self-loop.
    weight : float, default=0.1
        Constant weight on every cycle edge. The raw spectral radius is
        `abs(weight)`. A layer's requested radius changes this overall scale.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` cyclic matrix; a subdiagonal plus wrap-around corner.

    Raises
    ------
    ValueError
        If size or dtype is invalid.

    Examples
    --------
    >>> matrix = simple_cycle(6, weight=0.9)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3, recurrent_initializer="simple_cycle",
    ...     spectral_radius=0.9,
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE
    Transactions on Neural Networks*, 22(1):131–144, 2011.
    """
    target = matrix_device(n, dtype, device)
    return _build_simple_cycle(n, float(weight), dtype=dtype, device=target)


@register_matrix_topology("cycle_jumps", cycle_weight=0.1, jump_weight=0.1, jump_size=3)
def cycle_jumps(
    n: int,
    cycle_weight: float = 0.1,
    jump_weight: float = 0.1,
    jump_size: int = 3,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Add bidirectional jumps between regularly spaced nodes of a cycle.

    The jump chain visits `0, jump_size, 2*jump_size, ...`. It closes back
    to zero only if `jump_size` divides `n`; otherwise the trailing partial
    jump is omitted. Jump edges overwrite any coincident cycle edges.
    Requires `1 <= jump_size < n`.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer greater than jump_size.
    cycle_weight : float, default=0.1
        Constant weight of the unidirectional cycle edges.
    jump_weight : float, default=0.1
        Constant weight of both directions of each jump edge.
    jump_size : int, default=3
        Distance between successive jump-chain nodes. Must satisfy
        `1 <= jump_size < n`; divisibility determines the closing jump.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` cycle with symmetric jump connections. For example,
        `n=7, jump_size=3` adds `0 <-> 3` and `3 <-> 6` only.

    Raises
    ------
    ValueError
        If size, jump offset, or dtype is invalid.

    Examples
    --------
    >>> matrix = cycle_jumps(7, cycle_weight=0.1, jump_weight=0.5, jump_size=3)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3,
    ...     recurrent_initializer=("cycle_jumps", {"jump_size": 5}),
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Simple Deterministically Constructed Cycle Reservoirs
    with Regular Jumps", *Neural Computation*, 24(7):1822–1852, 2012.
    """
    target = matrix_device(n, dtype, device)
    if not 1 <= jump_size < n:
        raise ValueError(f"jump_size must satisfy 1 <= jump_size < n={n}, got {jump_size}")
    matrix = _build_simple_cycle(n, float(cycle_weight), dtype=dtype, device=target)
    nodes = list(range(0, n, jump_size))
    for left, right in zip(nodes, nodes[1:]):
        matrix[left, right] = float(jump_weight)
        matrix[right, left] = float(jump_weight)
    if n % jump_size == 0:
        matrix[nodes[-1], 0] = float(jump_weight)
        matrix[0, nodes[-1]] = float(jump_weight)
    return matrix


@register_matrix_topology("self_loop_cycle", cycle_weight=0.1, self_loop_weight=0.1)
def self_loop_cycle(
    n: int,
    cycle_weight: float = 0.1,
    self_loop_weight: float = 0.1,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a directed cycle with a constant self-connection on every unit.

    The diagonal is assigned `self_loop_weight`. For `n=1`, this overwrites
    the cycle's single self-connection.

    Parameters
    ----------
    n : int
        Number of reservoir units, a positive integer.
    cycle_weight : float, default=0.1
        Constant weight of the cycle edges.
    self_loop_weight : float, default=0.1
        Weight on every diagonal self-connection. For `n > 1`, eigenvalues
        are those of the cycle shifted by this value.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` cycle with diagonal self-loops.

    Raises
    ------
    ValueError
        If size or dtype is invalid.

    Examples
    --------
    >>> matrix = self_loop_cycle(6, cycle_weight=0.9, self_loop_weight=0.1)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3,
    ...     recurrent_initializer=("self_loop_cycle", {"self_loop_weight": 0.2}),
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE
    Transactions on Neural Networks*, 22(1):131–144, 2011.
    """
    target = matrix_device(n, dtype, device)
    matrix = _build_simple_cycle(n, float(cycle_weight), dtype=dtype, device=target)
    matrix.fill_diagonal_(float(self_loop_weight))
    return matrix
