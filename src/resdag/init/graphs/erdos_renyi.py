import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def erdos_renyi_adjacency(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense weighted adjacency of an Erdős–Rényi ``G(n, p)`` graph.

    Vectorized NumPy construction of the same adjacency that
    :func:`erdos_renyi_graph` describes, without materialising a NetworkX
    object or the ``O(n^2)`` Python edge comprehension. Every candidate edge is
    kept with probability ``p`` (a single ``rng.random((n, n)) < p`` draw) and
    assigned a weight drawn uniformly from ``{-1, +1}``.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of including each edge (in ``[0, 1]``).
    directed : bool, optional
        If ``True``, every ordered pair is an independent candidate edge. If
        ``False``, the upper triangle is drawn once and mirrored, yielding a
        symmetric adjacency.
    self_loops : bool, optional
        If ``True``, the diagonal is a candidate too; otherwise it is forced to
        zero.
    seed : int or numpy.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` weighted adjacency matrix with entries in
        ``{-1, 0, +1}``.

    See Also
    --------
    erdos_renyi_graph : NetworkX-returning wrapper around this builder.
    """
    rng = create_rng(seed)

    mask = rng.random((n, n)) < p
    signs = np.where(rng.random((n, n)) < 0.5, -1.0, 1.0).astype(np.float32)

    if not directed:
        # Draw the upper triangle once and mirror it so the adjacency is
        # symmetric, matching the undirected ``u <= v`` enumeration.
        upper = np.triu(np.ones((n, n), dtype=bool))
        mask = mask & upper
        mask = mask | mask.T
        signs = np.triu(signs)
        signs = signs + np.triu(signs, k=1).T

    if not self_loops:
        np.fill_diagonal(mask, False)

    return (mask.astype(np.float32)) * signs


@register_graph_topology(
    "erdos_renyi",
    p=0.1,
    directed=True,
    self_loops=True,
    seed=None,
)
@register_dense_adjacency(erdos_renyi_adjacency)
def erdos_renyi_graph(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates an Erdos-Renyi (G(n, p)) graph.

    Every possible edge is included with probability ``p``, independently of every other edge.
    Weights on edges are chosen randomly from the set ``{-1, 1}``.

    The adjacency is built with vectorized NumPy (see
    :func:`erdos_renyi_adjacency`); the NetworkX object is materialised only for
    direct callers. When this generator is used as a topology, the dense fast
    path skips NetworkX entirely.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Probability of including each edge (in [0, 1]).
    directed : bool
        If True, generates a directed graph; otherwise, an undirected graph.
    self_loops : bool
        If True, allows self-loops in the graph.
    seed : int or np.random.Generator or None
        Seed for the random number generator.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A Erdos-Renyi graph.
    """
    adjacency = erdos_renyi_adjacency(n, p, directed, self_loops, seed)
    return adjacency_to_graph(adjacency, directed=directed)
