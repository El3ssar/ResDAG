import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.init.utils.graph_tools import connected_graph

from .erdos_renyi import erdos_renyi_graph


@register_graph_topology(
    "connected_erdos_renyi",
    p=0.1,
    directed=True,
    self_loops=True,
    seed=None,
)
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

    This function wraps :func:`erdos_renyi` with a decorator that attempts multiple
    generations until a connected graph is obtained (up to a certain number of tries).

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
    tries : int, optional
        Number of attempts to generate a connected graph. This parameter is handled
        by the ``@connected_graph`` decorator.

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
    if p < np.log(n) / n:
        raise ValueError(
            f"Edge probability p must be > ln(n) / n to have a good chance of connectivity. "
            f"(Got p={p}, n={n})."
        )
    return erdos_renyi_graph(n, p, directed, self_loops, seed)
