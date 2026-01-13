import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "complete",
    self_loops=False,
    random_weights=True,
    seed=None,
)
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
    rng = create_rng(seed)
    G = Graph()  # Always undirected in this function

    for i in range(n):
        for j in range(i + 1, n):
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** (i + j)
            G.add_edge(i, j, weight=weight)

    if self_loops:
        for i in range(n):
            weight = rng.choice([-1, 1]) if random_weights else (-1) ** i
            G.add_edge(i, i, weight=weight)

    return G
