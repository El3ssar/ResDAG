import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.init.utils.graph_tools import connected_graph

from .watts_strogatz import watts_strogatz_graph


@register_graph_topology(
    "connected_watts_strogatz",
    k=6,
    p=0.1,
    directed=True,
    self_loops=True,
    seed=None,
)
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
    tries : int, optional
        Number of attempts to generate a connected graph. This parameter is handled
        by the ``@connected_graph`` decorator, not passed directly here.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A connected Watts-Strogatz graph.
    """
    return watts_strogatz_graph(n, k, p, directed, self_loops, seed)
