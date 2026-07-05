r"""Deterministic-minimal reservoir matrices (Rodan & Tiňo family).

Rodan & Tiňo showed that a fully *deterministic*, sparsely wired reservoir can
match the performance of a random Echo State Network on a range of benchmarks —
the "Minimum Complexity Echo State Network" programme. The recurrent matrix is
built from a handful of fixed structural primitives (a single delay line, a
unidirectional cycle, regularly spaced "jumps", optional self-loops) rather than
from random draws, so the reservoir is reproducible bit-for-bit with **no seed**.

This module ports the deterministic-minimal builders from
`ReservoirComputing.jl` (SciML) — ``delay_line``, ``delayline_backward``,
``simple_cycle``, ``cycle_jumps`` and ``selfloop_cycle`` — as resdag matrix
topologies. The nonzero *placements* are matched entry-for-entry to the RC.jl
component helpers in ``src/inits/inits_components.jl`` (``delay_line!``,
``backward_connection!``, ``simple_cycle!``, ``add_jumps!``, ``self_loop!``),
translated from 1-indexed Julia to 0-indexed NumPy/torch.

Unlike the pre-scaled ``orthogonal`` / ``fast_spectral_initialization``
topologies, these builders emit the raw structural weights and remain
**scalable**: a linear cycle of weight :math:`w` has spectral radius
:math:`|w|`, so the layer-level ``spectral_radius`` rescale (which multiplies the
whole matrix by a single factor) preserves the structure and simply retargets
the radius. The one exception is the pure ``delay_line``: a strictly
sub-diagonal matrix is *nilpotent* (all eigenvalues zero), so its spectral
radius cannot be rescaled — a layer ``spectral_radius`` is a no-op there (see
that builder's notes).

References
----------
A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE
Transactions on Neural Networks*, 22(1):131-144, 2011.

A. Rodan and P. Tiňo, "Simple Deterministically Constructed Cycle Reservoirs
with Regular Jumps", *Neural Computation*, 24(7):1822-1852, 2012.

C. Martinuzzi et al., *ReservoirComputing.jl*,
https://github.com/SciML/ReservoirComputing.jl (``src/inits/inits_reservoir.jl``
and ``src/inits/inits_components.jl``).

See Also
--------
resdag.init.matrices.structured : Toeplitz, banded, and block-diagonal matrices.
resdag.init.matrices.orthogonal : Haar-random orthogonal recurrent matrix.
resdag.init.matrices.fsi : Analytically spectral-radius-targeted random matrix.
"""

import torch

from resdag.init.topology import register_matrix_topology


def _validate_size(n: int) -> None:
    """Raise if ``n`` is not a usable matrix size."""
    if n <= 0:
        raise ValueError(f"Matrix size n must be positive, got {n}")


def _build_simple_cycle(n: int, weight: float) -> torch.Tensor:
    """Return the RC.jl ``simple_cycle!`` matrix (sub-diagonal + corner).

    Shared, precisely-typed core used by :func:`simple_cycle`,
    :func:`cycle_jumps`, and :func:`self_loop_cycle`. Kept separate from the
    registered builder so callers see a ``torch.Tensor`` return type (the
    ``@register_matrix_topology`` decorator erases it to a bare ``Callable``).
    """
    matrix = torch.zeros(n, n, dtype=torch.float32)
    idx = torch.arange(n - 1)
    matrix[idx + 1, idx] = weight
    matrix[0, n - 1] = weight
    return matrix


@register_matrix_topology("delay_line", weight=0.1, shift=1)
def delay_line(n: int, weight: float = 0.1, shift: int = 1) -> torch.Tensor:
    r"""Build a pure delay-line recurrent matrix (a single sub-diagonal).

    Places the constant ``weight`` on the ``shift``-th sub-diagonal and leaves
    every other entry zero, so the reservoir is a chain of neurons that copies
    (scaled) activity forward by ``shift`` steps each tick — the minimal linear
    memory of Rodan & Tiňo's deterministic ESN. Ports RC.jl's ``delay_line!``
    component (``inits_components.jl``): ``M[i + shift, i] = weight`` for
    ``i`` in ``0 .. n-1-shift`` (the Julia loop
    ``reservoir_matrix[idx + shift, idx] = weight`` over ``idx in 1:(N-shift)``).

    .. note::
        A strictly sub-diagonal matrix is **nilpotent**: all its eigenvalues are
        zero, so its spectral radius is ``0`` and cannot be rescaled. Passing a
        layer-level ``spectral_radius`` alongside this topology has no effect
        (the shared rescale is a no-op when the current radius is numerically
        zero); control the scale through ``weight`` instead.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    weight : float, default=0.1
        Constant value placed on every element of the sub-diagonal. Matches the
        RC.jl ``delay_weight`` default of ``0.1``.
    shift : int, default=1
        Distance of the delay line below the main diagonal. ``shift=1`` is the
        first sub-diagonal. Must satisfy ``1 <= shift < n``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` matrix with a single sub-diagonal.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(500, feedback_size=3, topology="delay_line")
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology=("delay_line", {"weight": 0.5})
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", *IEEE Trans.
    Neural Networks*, 22(1):131-144, 2011. RC.jl ``delay_line!``.
    """
    _validate_size(n)
    if not 1 <= shift < n:
        raise ValueError(f"shift must satisfy 1 <= shift < n={n}, got {shift}")

    matrix = torch.zeros(n, n, dtype=torch.float32)
    idx = torch.arange(n - shift)
    matrix[idx + shift, idx] = float(weight)
    return matrix


@register_matrix_topology(
    "delay_line_backward",
    weight=0.1,
    backward_weight=0.1,
    shift=1,
    backward_shift=1,
)
def delay_line_backward(
    n: int,
    weight: float = 0.1,
    backward_weight: float = 0.1,
    shift: int = 1,
    backward_shift: int = 1,
) -> torch.Tensor:
    r"""Build a delay line with an added backward (super-diagonal) connection.

    A ``delay_line`` (constant ``weight`` on the ``shift``-th sub-diagonal) with
    a symmetric-in-position back-connection: the constant ``backward_weight`` on
    the ``backward_shift``-th *super*-diagonal, giving the chain a short reverse
    path. Ports RC.jl's ``delayline_backward`` (``delay_line!`` +
    ``backward_connection!``): sub-diagonal ``M[i + shift, i] = weight`` and
    super-diagonal ``M[i, i + backward_shift] = backward_weight`` (the Julia
    ``backward_connection!`` loop ``reservoir_matrix[idx, idx + shift]``).

    Unlike the pure ``delay_line``, this matrix is **not** nilpotent once the
    back-connection is present (it carries cycles), so its spectral radius is
    generally nonzero and a layer-level ``spectral_radius`` rescales it as usual.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    weight : float, default=0.1
        Constant on the sub-diagonal (the forward delay line). RC.jl
        ``delay_weight`` default ``0.1``.
    backward_weight : float, default=0.1
        Constant on the super-diagonal (the backward connection). RC.jl
        ``fb_weight`` default ``0.1``.
    shift : int, default=1
        Sub-diagonal offset of the forward delay line. ``1 <= shift < n``.
    backward_shift : int, default=1
        Super-diagonal offset of the backward connection.
        ``1 <= backward_shift < n``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` matrix with one sub- and one super-diagonal.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="delay_line_backward"
    ... )
    >>> reservoir = ESNLayer(
    ...     500,
    ...     feedback_size=3,
    ...     topology=("delay_line_backward", {"backward_weight": -1.0}),
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", 2011. RC.jl
    ``delayline_backward``.
    """
    _validate_size(n)
    if not 1 <= shift < n:
        raise ValueError(f"shift must satisfy 1 <= shift < n={n}, got {shift}")
    if not 1 <= backward_shift < n:
        raise ValueError(
            f"backward_shift must satisfy 1 <= backward_shift < n={n}, got {backward_shift}"
        )

    matrix = torch.zeros(n, n, dtype=torch.float32)
    fwd = torch.arange(n - shift)
    matrix[fwd + shift, fwd] = float(weight)
    bwd = torch.arange(n - backward_shift)
    matrix[bwd, bwd + backward_shift] = float(backward_weight)
    return matrix


@register_matrix_topology("simple_cycle", weight=0.1)
def simple_cycle(n: int, weight: float = 0.1) -> torch.Tensor:
    r"""Build a unidirectional cycle (ring) recurrent matrix.

    A single constant ``weight`` on the sub-diagonal *plus* the wrap-around
    corner, wiring the units into one directed ring
    ``0 <- n-1 <- ... <- 1 <- 0``. This is Rodan & Tiňo's Simple Cycle Reservoir
    (SCR), the canonical deterministic-minimal topology. Ports RC.jl's
    ``simple_cycle!``: ``M[i + 1, i] = weight`` for ``i`` in ``0 .. n-2`` and the
    corner ``M[0, n - 1] = weight`` (the Julia ``reservoir_matrix[1, end]``
    wrap-around).

    A cyclic-permutation matrix scaled by ``weight`` has all eigenvalues on the
    circle of radius :math:`|weight|`, so its spectral radius is exactly
    :math:`|weight|` and a layer-level ``spectral_radius`` rescales it cleanly.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    weight : float, default=0.1
        Constant weight on every cycle edge. RC.jl ``cycle_weight`` default
        ``0.1``. The spectral radius of the unscaled matrix equals ``|weight|``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` cyclic matrix.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="simple_cycle", spectral_radius=0.9
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network", 2011 (Simple
    Cycle Reservoir). RC.jl ``simple_cycle!``.
    """
    _validate_size(n)
    return _build_simple_cycle(n, float(weight))


@register_matrix_topology("cycle_jumps", cycle_weight=0.1, jump_weight=0.1, jump_size=3)
def cycle_jumps(
    n: int,
    cycle_weight: float = 0.1,
    jump_weight: float = 0.1,
    jump_size: int = 3,
) -> torch.Tensor:
    r"""Build a simple cycle augmented with regularly spaced bidirectional jumps.

    Rodan & Tiňo's Cycle Reservoir with Jumps (CRJ, 2012): the unidirectional
    ``simple_cycle`` (constant ``cycle_weight``) plus symmetric "jump" edges of
    constant ``jump_weight`` connecting nodes ``jump_size`` apart around a ring,
    which shorten path lengths and boost memory capacity while staying fully
    deterministic. Ports RC.jl's ``cycle_jumps`` (``simple_cycle!`` +
    ``add_jumps!``).

    The jump wiring matches RC.jl's ``add_jumps!`` walk exactly: starting from
    node ``0`` it steps by ``jump_size`` around the ring, adding a *symmetric*
    edge ``M[i, j] = M[j, i] = jump_weight`` at each step. When ``n`` is a
    multiple of ``jump_size`` the walk closes into a full sub-ring; otherwise it
    stops at the last node reachable without wrapping past ``n`` (so the final
    partial jump is dropped, as in the RC.jl source).

    .. note::
        The nonzero jump *pattern* is matched to the current RC.jl **source**
        (the ``add_jumps!`` walk in ``inits_components.jl``). The illustrative
        ``add_jumps!`` docstring example in RC.jl is stale relative to that walk
        for some sizes; the source loop is authoritative and is what is ported.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    cycle_weight : float, default=0.1
        Constant weight of the unidirectional cycle. RC.jl ``cycle_weight``
        default ``0.1``.
    jump_weight : float, default=0.1
        Constant weight of the (symmetric) jump edges. RC.jl ``jump_weight``
        default ``0.1``.
    jump_size : int, default=3
        Number of nodes spanned by each jump. RC.jl ``jump_size`` default ``3``.
        Must satisfy ``1 <= jump_size < n``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` cycle-with-jumps matrix.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="cycle_jumps", spectral_radius=0.9
    ... )
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology=("cycle_jumps", {"jump_size": 5})
    ... )

    References
    ----------
    A. Rodan and P. Tiňo, "Simple Deterministically Constructed Cycle Reservoirs
    with Regular Jumps", *Neural Computation*, 24(7):1822-1852, 2012. RC.jl
    ``cycle_jumps`` / ``add_jumps!``.
    """
    _validate_size(n)
    if not 1 <= jump_size < n:
        raise ValueError(f"jump_size must satisfy 1 <= jump_size < n={n}, got {jump_size}")

    matrix = _build_simple_cycle(n, float(cycle_weight))

    # Port of RC.jl ``add_jumps!`` (inits_components.jl): a deterministic ring
    # walk from node 0 stepping by jump_size, adding a symmetric edge per step.
    # 1-indexed Julia ``start=1`` corresponds to 0-indexed ``start=0`` here.
    w = float(jump_weight)
    divisible = n % jump_size == 0
    seq: list[int] = [0]
    cur = 0
    while True:
        nxt = cur + jump_size
        if nxt > n - 1:
            if divisible:
                nxt = (cur + jump_size) % n
                if nxt == 0:
                    break
                seq.append(nxt)
                cur = nxt
                continue
            break
        seq.append(nxt)
        cur = nxt

    num_edges = len(seq) if divisible else len(seq) - 1
    for k in range(num_edges):
        i = seq[k]
        if divisible:
            j = 0 if k == num_edges - 1 else seq[k + 1]
        else:
            j = seq[k + 1]
        matrix[i, j] = w
        matrix[j, i] = w

    return matrix


@register_matrix_topology("self_loop_cycle", cycle_weight=0.1, self_loop_weight=0.1)
def self_loop_cycle(
    n: int,
    cycle_weight: float = 0.1,
    self_loop_weight: float = 0.1,
) -> torch.Tensor:
    r"""Build a unidirectional cycle with a self-loop on every node.

    The ``simple_cycle`` (constant ``cycle_weight`` on the sub-diagonal and the
    wrap-around corner) with the constant ``self_loop_weight`` added on the main
    diagonal, so each unit both feeds the next around the ring and retains a
    fraction of its own state — a leaky deterministic ring. Ports RC.jl's
    ``selfloop_cycle`` (``self_loop!`` + ``simple_cycle!``): diagonal
    ``M[i, i] = self_loop_weight`` plus the simple-cycle pattern.

    With equal weights ``w`` the matrix is ``w * (I + C)`` for a cyclic
    permutation ``C``, whose eigenvalues are ``w * (1 + e^{2πik/n})`` — the
    spectral radius is ``2|w|`` (attained at ``k=0``), and a layer-level
    ``spectral_radius`` rescales the whole matrix accordingly.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    cycle_weight : float, default=0.1
        Constant weight of the cycle edges. RC.jl ``cycle_weight`` default
        ``0.1``.
    self_loop_weight : float, default=0.1
        Constant weight of the diagonal self-loops. RC.jl ``selfloop_weight``
        default ``0.1``.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` cycle-with-self-loops matrix.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology="self_loop_cycle", spectral_radius=0.9
    ... )

    References
    ----------
    RC.jl ``selfloop_cycle`` (deterministic self-loop cycle reservoir), building
    on the Rodan & Tiňo minimal-complexity ESN family.
    """
    _validate_size(n)
    matrix = _build_simple_cycle(n, float(cycle_weight))
    diag = torch.arange(n)
    matrix[diag, diag] = float(self_loop_weight)
    return matrix
