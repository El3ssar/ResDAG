import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.init.utils.graph_tools import connected_adjacency, connected_graph

from ._dense import register_dense_adjacency
from .erdos_renyi import erdos_renyi_adjacency


def _connected_erdos_renyi_precondition(n: int, **kwargs: object) -> None:
    """Guard `p > ln(n)/n` before attempting a connected Erdős–Rényi build."""
    p = kwargs.get("p", 0.1)
    assert isinstance(p, (int, float))
    if p < np.log(n) / n:
        raise ValueError(
            f"Edge probability p must be > ln(n) / n to have a good chance of connectivity. "
            f"(Got p={p}, n={n})."
        )


# Adjacency-level fast path for the topology hot path: retry the vectorized dense
# builder and check weak connectivity via scipy, skipping the NetworkX
# round-trip. ``connected_erdos_renyi_graph`` defaults to ``directed=True``.
_connected_erdos_renyi_adjacency = connected_adjacency(
    erdos_renyi_adjacency,
    directed=True,
    precondition=_connected_erdos_renyi_precondition,
)


@register_graph_topology(
    "connected_erdos_renyi",
    p=0.1,
    directed=True,
    self_loops=True,
    seed=None,
)
@register_dense_adjacency(_connected_erdos_renyi_adjacency)
@connected_graph
def connected_erdos_renyi_graph(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a **connected** Erdos-Renyi graph.

    This function wraps `erdos_renyi` with a decorator that attempts multiple
    generations until a connected graph is obtained (up to a certain number of tries).

    The direct-call path returns a NetworkX graph (via the `@connected_graph`
    retry wrapper). When used as a topology, the adjacency-level fast path
    (`resdag.init.utils.graph_tools.connected_adjacency`) retries the
    vectorized dense builder and checks weak connectivity with scipy, skipping
    the NetworkX round-trip entirely.

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

    Notes
    -----
    The `tries` argument is accepted by the `@connected_graph` wrapper, not
    this function's signature.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A connected Erdos-Renyi graph.

    Raises
    ------
    ValueError
        If the probability `p` is too small to expect a connected graph. As a rough guideline,
        `p` should be greater than `ln(n)/n` for a good chance of connectivity.
    """
    _connected_erdos_renyi_precondition(n, p=p)
    from .erdos_renyi import erdos_renyi_graph

    return erdos_renyi_graph(n, p, directed, self_loops, seed)
