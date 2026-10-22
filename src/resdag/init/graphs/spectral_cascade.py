from math import isqrt

import numpy as np
from networkx import Graph

from resdag.init.topology.registry import register_graph_topology

from ._dense import adjacency_to_graph, register_dense_adjacency


def spectral_cascade_adjacency(
    n: int,
    spectral_radius: float,
    self_loops: bool = False,
) -> np.ndarray:
    """Build the dense weighted adjacency of a spectral-cascade graph.

    Constructs the block-diagonal clique adjacency directly in NumPy — the same
    matrix :func:`spectral_cascade_graph` describes — without materialising a
    NetworkX object or the ``O(n^2)`` Python double loop that previously
    converted the dense matrix into edges only for
    :class:`~resdag.init.topology.GraphTopology` to collapse it straight back.

    Each clique ``k = 1..N`` (sizes ``1..N`` with ``n = N(N+1)/2``) occupies a
    contiguous diagonal block filled with deterministic ``{+1, -1}`` weights and
    spectrally scaled to its graded target radius (see
    :func:`spectral_cascade_graph` for the cascade definition).

    Parameters
    ----------
    n : int
        Total number of nodes (must be triangular, i.e. ``n = N(N+1)/2``).
    spectral_radius : float
        The "starting" radius for the 2-node clique, graded down across larger
        cliques.
    self_loops : bool, optional
        If ``True``, each node carries a self-loop of weight ``(-1)^i``.
        Default: ``False``.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` symmetric block-diagonal weighted adjacency.

    Raises
    ------
    ValueError
        If ``n`` is not triangular.

    See Also
    --------
    spectral_cascade_graph : NetworkX-returning wrapper around this builder.
    """
    # 1) Check that n is triangular:  n = N(N+1)/2.
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    if not np.isfinite(spectral_radius) or spectral_radius < 0:
        raise ValueError("spectral_radius must be finite and non-negative.")
    D = 1 + 8 * n
    sqrt_D = isqrt(D)
    if sqrt_D * sqrt_D != D or (sqrt_D - 1) % 2 != 0:
        raise ValueError(f"{n} is not a triangular number (N(N+1)/2 for some integer N).")

    N = (sqrt_D - 1) // 2  # number of cliques

    # 2) Prepare an n x n adjacency matrix
    A = np.zeros((n, n), dtype=np.float64)

    offset = 0
    for k in range(1, N + 1):
        size_k = k
        # If k=1 => trivial, spectral radius = 0 => skip edges
        if k == 1:
            offset += 1
            continue

        # Build the local adjacency block. With global indices
        # ``g = offset + i``, the off-diagonal weight ``(-1)^(g_i + g_j)`` is the
        # outer product of the per-node sign ``s_i = (-1)^(offset + i)``, and the
        # optional diagonal self-loop is ``(-1)^(offset + i) == s_i`` too.
        local = np.arange(size_k)
        signs = np.power(-1.0, offset + local)  # (-1)^(offset + i)
        A_sub = np.outer(signs, signs)  # (-1)^(g_i + g_j) elementwise
        if self_loops:
            np.fill_diagonal(A_sub, signs)  # diagonal = (-1)^g_i
        else:
            np.fill_diagonal(A_sub, 0.0)

        # 3) Desired spectral radius for this clique
        if k == 2:
            desired_r = spectral_radius
        else:
            desired_r = (N - k + 1) * spectral_radius / N

        # 4) Scale block by ratio (desired_r / actual_r)
        vals = np.linalg.eigvals(A_sub)
        current_r = max(abs(vals)) if len(vals) > 0 else 0.0

        if current_r > 1e-14:
            scale_factor = desired_r / current_r
            A_sub *= scale_factor

        # 5) Insert A_sub back into the global adjacency matrix A
        A[offset : offset + k, offset : offset + k] = A_sub

        offset += k

    # Symmetrize exactly as the original did (mirror upper triangle). The block
    # construction is already symmetric, so this is a no-op numerically but kept
    # for parity and to zero any floating asymmetry.
    A = np.triu(A) + np.triu(A, 1).T

    # Drop floating garbage below the original edge threshold so the adjacency
    # has the same sparsity pattern as the edge-by-edge NetworkX build.
    A[np.abs(A) <= 1e-14] = 0.0

    return A.astype(np.float32)


@register_graph_topology(
    "spectral_cascade",
    prescaled=True,
    spectral_radius=1.0,
    self_loops=False,
)
@register_dense_adjacency(spectral_cascade_adjacency)
def spectral_cascade_graph(
    n: int,
    spectral_radius: float,
    self_loops: bool = False,
) -> Graph:
    """
    Generates a graph of N disconnected components (cliques) of sizes 1..N, where n = N(N+1)/2.
    Each clique is fully connected (or complete in the directed sense),
    with deterministic weights in {+1, -1}:
      - off-diagonal: (-1)^(i + j)
      - diagonal (if self_loops=True): (-1)^i

    Then each clique's adjacency block is spectrally scaled so that:
      - The 1-node clique has spectral radius 0 (trivial).
      - The 2-node clique has spectral radius = `spectral_radius`.
      - Each larger clique k=3..N has spectral radius (N - k + 1)*spectral_radius / N.

    This topology is **pre-scaled**: the graded per-clique radii above *are* its
    spectral structure, and they are encoded in the edge weights by this builder
    itself. The layer-level spectral-radius rescale would flatten that cascade
    into a single global radius, so it is suppressed. The graded-radii property
    therefore holds only when the rescale is skipped — i.e. when the layer
    ``spectral_radius`` is ``None`` (or, equivalently, when this topology is used
    by name, in which case a layer ``spectral_radius`` is ignored with a
    warning). Set the cascade's top radius through this builder's own
    ``spectral_radius`` parameter instead.

    The adjacency is built with vectorized NumPy (see
    :func:`spectral_cascade_adjacency`); the NetworkX object is materialised only
    for direct callers. When this generator is used as a topology, the dense fast
    path skips NetworkX entirely.

    Parameters
    ----------
    n : int
        Total number of nodes (must be triangular, i.e. n = N(N+1)/2).
    spectral_radius : float
        The "starting" radius for the 2-node clique, which is then decreased
        across bigger cliques down to sr/N for the largest clique. This is the
        builder's own parameter and is independent of any layer-level
        ``spectral_radius`` (which is ignored for this pre-scaled topology).
    self_loops : bool, optional
        If True, each node has a self-loop of weight = (-1)^i. Default: False.

    Returns
    -------
    networkx.Graph
        A NetworkX graph with block-diagonal clique structure,
        spectrally scaled as specified.

    Raises
    ------
    ValueError
        If `n` is not triangular.
    """
    adjacency = spectral_cascade_adjacency(n, spectral_radius, self_loops)
    return adjacency_to_graph(adjacency, directed=False)
