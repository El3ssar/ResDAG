import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "regular",
    k=3,
    directed=False,
    self_loops=False,
    random_weights=True,
    seed=None,
)
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
    if k > n - (n % 2):
        raise ValueError(f"k must be <= n - (n % 2). Got k={k}, n={n}.")

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Connect each node to k/2 neighbors on each side
    for i in range(n):
        for j in range(1, (k // 2) + 1):
            neighbor = (i + j) % n
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** (i + neighbor)
            G.add_edge(i, neighbor, weight=weight)
            if directed:
                weight = rng.choice([-1, 1]) if random_weights else (-1) ** (i + neighbor)
                G.add_edge(neighbor, i, weight=weight)

    if self_loops:
        for i in range(n):
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** i
            G.add_edge(i, i, weight=weight)

    return G
