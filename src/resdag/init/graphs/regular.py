import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def regular_adjacency(
    n: int,
    k: int,
    directed: bool = False,
    self_loops: bool = False,
    random_weights: bool = True,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense weighted adjacency of a regular ring-lattice graph.

    Vectorized NumPy construction of the same banded adjacency that
    :func:`regular_graph` describes, without the ``O(n*k)`` Python loop. Each
    node connects to its ``k // 2`` forward neighbours (wrapping on the ring);
    undirected graphs mirror those edges, directed graphs add the reverse edges
    as independent draws.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Number of neighbours per node (``k // 2`` on each side).
    directed : bool, optional
        If ``True``, the forward and reverse edges are drawn independently;
        otherwise the forward edges are mirrored to keep the matrix symmetric.
    self_loops : bool, optional
        If ``True``, the diagonal carries a self-loop on every node.
    random_weights : bool, optional
        If ``True``, weights are random ``{-1, +1}``; otherwise they follow
        ``(-1) ** (i + neighbour)``.
    seed : int or numpy.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` weighted adjacency matrix.

    Raises
    ------
    ValueError
        If ``k > n - (n % 2)`` (an invalid regular ring-lattice configuration).

    See Also
    --------
    regular_graph : NetworkX-returning wrapper around this builder.
    """
    if k > n - (n % 2):
        raise ValueError(f"k must be <= n - (n % 2). Got k={k}, n={n}.")

    rng = create_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.float32)
    rows = np.arange(n)

    def _weights(targets: np.ndarray) -> np.ndarray:
        if random_weights:
            return np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        signs = np.power(-1.0, rows + targets)
        return np.asarray(signs, dtype=np.float32)

    for j in range(1, (k // 2) + 1):
        neighbors = (rows + j) % n
        forward = _weights(neighbors)
        adjacency[rows, neighbors] = forward
        if directed:
            backward = _weights(neighbors)
            adjacency[neighbors, rows] = backward
        else:
            # Undirected: mirror the forward edge weight.
            adjacency[neighbors, rows] = forward

    if self_loops:
        if random_weights:
            diag = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        else:
            diag = ((-1.0) ** rows).astype(np.float32)
        np.fill_diagonal(adjacency, diag)

    return adjacency


@register_graph_topology(
    "regular",
    k=3,
    directed=False,
    self_loops=False,
    random_weights=True,
    seed=None,
)
@register_dense_adjacency(regular_adjacency)
def regular_graph(
    n: int,
    k: int,
    directed: bool = False,
    self_loops: bool = False,
    random_weights: bool = True,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a regular ring-lattice graph (each node has k neighbors).

    .. note::
        - If ``directed=True``, each undirected edge is replaced with two directed edges.
        - If ``self_loops=True``, each node also has a self-loop.
        - Weights can either be random in {-1, 1} or deterministically alternating.

    The adjacency is built with vectorized NumPy (see
    :func:`regular_adjacency`); the NetworkX object is materialised only for
    direct callers, and the dense topology fast path skips it entirely.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Number of neighbors each node is connected to. Must be <= n - (n % 2) in undirected mode.
    directed : bool, optional
        If True, the graph is directed; else undirected. Default: False.
    self_loops : bool, optional
        If True, adds a self-loop to each node. Default: False.
    random_weights : bool, optional
        If True, weights are drawn from {-1, 1} randomly; otherwise, they alternate according
        to (-1)^(i + j). Default: True.
    seed : int or np.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A regular ring-lattice graph.

    Raises
    ------
    ValueError
        If ``k > n - (n % 2)`` (an invalid regular ring-lattice configuration).
    """
    # ``regular_adjacency`` performs the ``k`` validation (raising the same
    # ValueError) so the fast path and the NetworkX path agree.
    adjacency = regular_adjacency(n, k, directed, self_loops, random_weights, seed)
    return adjacency_to_graph(adjacency, directed=directed)
