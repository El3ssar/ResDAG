import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.init.utils.graph_tools import connected_adjacency, connected_graph

from ._dense import register_dense_adjacency
from .watts_strogatz import watts_strogatz_adjacency

# Adjacency-level fast path for the topology hot path: retry the vectorized dense
# builder and check weak connectivity via scipy, skipping the NetworkX
# round-trip. ``connected_watts_strogatz_graph`` defaults to ``directed=True``.
_connected_watts_strogatz_adjacency = connected_adjacency(
    watts_strogatz_adjacency,
    directed=True,
)


@register_graph_topology(
    "connected_watts_strogatz",
    k=6,
    p=0.1,
    directed=True,
    self_loops=True,
    seed=None,
)
@register_dense_adjacency(_connected_watts_strogatz_adjacency)
@connected_graph
def connected_watts_strogatz_graph(
    n: int,
    k: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a **connected** Watts-Strogatz graph.

    This function wraps :func:`watts_strogatz` with a decorator that attempts multiple
    generations until a connected graph is obtained (up to a certain number of tries).

    The direct-call path returns a NetworkX graph (via the ``@connected_graph``
    retry wrapper). When used as a topology, the adjacency-level fast path
    (:func:`~resdag.init.utils.graph_tools.connected_adjacency`) retries the
    vectorized dense builder and checks weak connectivity with scipy, skipping
    the NetworkX round-trip entirely.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to ``k/2`` predecessors and ``k/2`` successors.
    p : float
        Rewiring probability in the interval [0, 1].
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected graph.
        Default is True.
    self_loops : bool, optional
        If True, allows self-loops during the rewiring step. Default is True.
    seed : int or np.random.Generator or None, optional
        Seed for random number generator (RNG). If None, a random seed is used.

    Notes
    -----
    The ``tries`` argument is accepted by the ``@connected_graph`` wrapper, not
    this function's signature.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A connected Watts-Strogatz graph.
    """
    from .watts_strogatz import watts_strogatz_graph

    return watts_strogatz_graph(n, k, p, directed, self_loops, seed)
