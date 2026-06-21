import math

import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def _toroidal_distance_matrix(side: int) -> np.ndarray:
    """Pairwise toroidal Manhattan distances on a ``side x side`` grid.

    Returns an ``(N, N)`` matrix (``N = side * side``) where entry ``(a, b)`` is
    the wrapped Manhattan distance between the row-major grid nodes ``a`` and
    ``b``. Computed once and reused for every node's long-range draw, replacing
    the per-node ``O(N)`` Python candidate rebuild.
    """
    coords = np.arange(side)
    # 1-D wrapped distance along one axis, broadcast to the grid.
    axis = np.abs(coords[:, None] - coords[None, :])
    axis = np.minimum(axis, side - axis)  # (side, side)

    rows = np.repeat(np.arange(side), side)
    cols = np.tile(np.arange(side), side)
    di = axis[rows[:, None], rows[None, :]]
    dj = axis[cols[:, None], cols[None, :]]
    return np.asarray(di + dj, dtype=np.float64)


def kleinberg_small_world_adjacency(
    n: int,
    q: float = 2,
    k: int = 1,
    directed: bool = False,
    weighted: bool = False,
    beta: float = 2,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense weighted adjacency of a 2D Kleinberg small-world graph.

    NumPy construction of the same adjacency that
    :func:`kleinberg_small_world_graph` describes, on a ``sqrt(n) x sqrt(n)``
    torus with row-major node indexing (node ``(i, j)`` -> ``i * side + j``,
    matching ``sorted`` over tuple labels). The toroidal distance matrix is
    computed once and shared across every node's long-range draw, replacing the
    per-node ``O(n)`` Python candidate rebuild.

    Parameters
    ----------
    n : int
        Total number of nodes; must be a perfect square.
    q : float, optional
        Exponent controlling long-range connection probability
        (``probability ~ distance ** -q``).
    k : int, optional
        Number of long-range connections per node.
    directed : bool, optional
        If ``True``, the adjacency is directed; otherwise edges are symmetric.
    weighted : bool, optional
        If ``True``, long-range weights are ``distance ** beta``; otherwise they
        are random ``{-1, +1}``. Local edges are always random ``{-1, +1}``.
    beta : float, optional
        Exponent for long-range weights when ``weighted=True``.
    seed : int or numpy.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` weighted adjacency matrix.

    Raises
    ------
    ValueError
        If ``n`` is not a perfect square.

    See Also
    --------
    kleinberg_small_world_graph : NetworkX-returning wrapper around this builder.
    """
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(
            f"kleinberg_small_world requires a perfect-square number of nodes "
            f"(got n={n}); the graph lives on a sqrt(n) x sqrt(n) toroidal grid."
        )

    rng = create_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.float32)

    def _set_edge(u: int, v: int, weight: float) -> None:
        adjacency[u, v] = weight
        if not directed:
            adjacency[v, u] = weight

    # Per-node random "node weight" is drawn in the original (unused in the
    # adjacency); consume the same draws so the RNG stream stays aligned with the
    # NetworkX implementation, keeping the two statistically equivalent.
    rng.random(n)

    # Local edges: each node connects to its 4 toroidal neighbours.
    for i in range(side):
        for j in range(side):
            u = i * side + j
            neighbors = [
                ((i - 1) % side) * side + j,
                ((i + 1) % side) * side + j,
                i * side + (j - 1) % side,
                i * side + (j + 1) % side,
            ]
            for v in neighbors:
                _set_edge(u, v, _rand_sign(rng))

    # Long-range edges: probability ~ distance ** -q over all other nodes.
    distances = _toroidal_distance_matrix(side)
    with np.errstate(divide="ignore"):
        probs = distances**-q
    np.fill_diagonal(probs, 0.0)  # exclude self (distance 0 -> inf)

    for u in range(n):
        row = probs[u]
        total = row.sum()
        if total <= 0:
            continue
        weights = row / total
        candidates = np.flatnonzero(weights > 0)
        k_eff = min(k, candidates.size)
        if k_eff == 0:
            continue
        chosen = rng.choice(n, size=k_eff, replace=False, p=weights)
        for v in chosen:
            v = int(v)
            dist = float(distances[u, v])
            weight = (dist**beta) if weighted else _rand_sign(rng)
            _set_edge(u, v, weight)

    return adjacency


def _rand_sign(rng: np.random.Generator) -> float:
    """Draw a random weight uniformly from ``{-1.0, +1.0}``."""
    return -1.0 if rng.random() < 0.5 else 1.0


@register_graph_topology(
    "kleinberg_small_world",
    q=2,
    k=1,
    directed=False,
    weighted=False,
    beta=2,
    seed=None,
)
@register_dense_adjacency(kleinberg_small_world_adjacency)
def kleinberg_small_world_graph(
    n: int,
    q: float = 2,
    k: int = 1,
    directed: bool = False,
    weighted: bool = False,
    beta: float = 2,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a 2D Kleinberg small-world graph on an ``n x n`` toroidal grid.

    Each node corresponds to a position on the 2D torus (i, j). Local edges connect each
    node to its 4 immediate neighbors (up, down, left, right) with wrapping. Additionally,
    each node gains ``k`` long-range edges, where the probability of connecting to a
    particular node depends on the toroidal Manhattan distance raised to the power ``-q``.

    When ``weighted=True``, weights are assigned as ``distance^beta`` for long-range links.

    The adjacency is built with NumPy (see
    :func:`kleinberg_small_world_adjacency`); the NetworkX object is materialised
    only for direct callers, and the dense topology fast path skips it entirely.
    Nodes are indexed row-major (node ``(i, j)`` maps to index ``i * sqrt(n) + j``).

    Parameters
    ----------
    n : int
        Total number of nodes. Must be a perfect square — the graph lives
        on a sqrt(n) x sqrt(n) toroidal grid.
    q : float, optional
        Exponent controlling the probability of long-range connections. Default: 2.
    k : int, optional
        Number of long-range connections per node. Default: 1.
    directed : bool, optional
        If True, graph is directed; otherwise, undirected. Default: False.
    weighted : bool, optional
        If True, weight of each long-range link is ``distance^beta``; otherwise, it is
        randomly chosen from {-1, 1}. Default: False.
    beta : float, optional
        Exponent used when computing long-range weights if ``weighted=True``. Default: 2.
    seed : int or np.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A Kleinberg small-world graph on an ``n x n`` toroidal grid.
    """
    adjacency = kleinberg_small_world_adjacency(n, q, k, directed, weighted, beta, seed)
    return adjacency_to_graph(adjacency, directed=directed)
