import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "erdos_renyi",
    p=0.1,
    directed=True,
    self_loops=True,
    seed=None,
)
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
    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    nodes = range(n)
    G.add_nodes_from(nodes)

    if directed:
        edges = [(u, v) for u in nodes for v in nodes if self_loops or u != v]
    else:
        # For undirected, only consider edges (u, v) with u <= v to avoid duplicates
        edges = [(u, v) for u in nodes for v in range(u, n) if self_loops or u != v]

    selected_edges = [edge for edge in edges if rng.random() < p]

    G.add_edges_from((u, v, {"weight": rng.choice([-1, 1])}) for u, v in selected_edges)

    return G
