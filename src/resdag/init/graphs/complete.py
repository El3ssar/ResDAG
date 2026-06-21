import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def complete_adjacency(
    n: int,
    self_loops: bool = False,
    random_weights: bool = True,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense weighted adjacency of a complete (undirected) graph.

    Vectorized NumPy construction of the same symmetric adjacency that
    :func:`complete_graph` describes, without the ``O(n^2)`` Python double loop.
    Every off-diagonal pair is connected; weights are either random ``{-1, +1}``
    or the deterministic alternating pattern ``(-1) ** (i + j)``.

    Parameters
    ----------
    n : int
        Number of nodes.
    self_loops : bool, optional
        If ``True``, the diagonal carries a self-loop on every node.
    random_weights : bool, optional
        If ``True``, weights are random ``{-1, +1}``; otherwise they follow
        ``(-1) ** (i + j)`` off-diagonal and ``(-1) ** i`` on the diagonal.
    seed : int or numpy.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` symmetric weighted adjacency matrix.

    See Also
    --------
    complete_graph : NetworkX-returning wrapper around this builder.
    """
    rng = create_rng(seed)
    idx = np.arange(n)

    if random_weights:
        upper = np.where(rng.random((n, n)) < 0.5, -1.0, 1.0).astype(np.float32)
        # Symmetrize: draw the strict upper triangle and mirror it.
        adjacency = np.triu(upper, k=1)
        adjacency = adjacency + adjacency.T
    else:
        signs = ((-1.0) ** (idx[:, None] + idx[None, :])).astype(np.float32)
        adjacency = signs.copy()
        np.fill_diagonal(adjacency, 0.0)

    if self_loops:
        if random_weights:
            diag = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        else:
            diag = ((-1.0) ** idx).astype(np.float32)
        np.fill_diagonal(adjacency, diag)

    return adjacency


@register_graph_topology(
    "complete",
    self_loops=False,
    random_weights=True,
    seed=None,
)
@register_dense_adjacency(complete_adjacency)
def complete_graph(
    n: int,
    self_loops: bool = False,
    random_weights: bool = True,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a complete (undirected) graph of n nodes.

    Each pair of distinct nodes is connected by an edge. Optionally, self-loops can be included.
    Weights on edges can be random in {-1, 1} or follow a deterministic alternating pattern.

    The adjacency is built with vectorized NumPy (see
    :func:`complete_adjacency`); the NetworkX object is materialised only for
    direct callers, and the dense topology fast path skips it entirely.

    Parameters
    ----------
    n : int
        Number of nodes.
    self_loops : bool, optional
        If True, adds a self-loop to each node. Default: False.
    random_weights : bool, optional
        If True, weights are chosen randomly from {-1, 1}; otherwise, they alternate
        according to (-1)^(i + j). Default: True.
    seed : int or np.random.Generator or None, optional
        Seed for the RNG.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A complete (undirected) graph.
    """
    adjacency = complete_adjacency(n, self_loops, random_weights, seed)
    return adjacency_to_graph(adjacency, directed=False)
