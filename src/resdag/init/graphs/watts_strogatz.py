import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "watts_strogatz",
    k=6,
    p=0.1,
    directed=False,
    self_loops=False,
    seed=None,
)
def watts_strogatz_graph(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a Watts-Strogatz small-world graph.

    The function starts by creating a ring lattice where each node is connected to ``k/2`` neighbors
    on each side. Then, with probability ``p``, each edge is rewired to a new node (allowing
    for possible self-loops if specified). Weights on edges are chosen randomly from the set ``{-1, 1}``.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is initially connected to ``k/2`` predecessors and ``k/2`` successors.
        If ``k`` is odd, it will be incremented by 1 internally.
        Must be smaller than ``n``.
    p : float
        Rewiring probability in the interval [0, 1].
    directed : bool, optional
        If True, generates a directed graph; otherwise, generates an undirected graph.
    self_loops : bool, optional
        If True, allows self-loops during the rewiring step.
    seed : int or np.random.Generator or None, optional
        Seed for random number generator (RNG). If None, a random seed is used.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A Watts-Strogatz small-world graph.

    Raises
    ------
    ValueError
        If ``k >= n`` (not a valid ring lattice).
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n}).")

    # Ensure k is even
    if k % 2 != 0:
        k += 1

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initial ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n, weight=rng.choice([-1, 1]))  # forward edge
            if directed:
                G.add_edge(i, (i - j) % n, weight=rng.choice([-1, 1]))  # backward edge

    # Rewire edges with probability p
    edges = list(G.edges())
    for u, v in edges:
        if rng.random() < p:
            # Remove the original edge
            G.remove_edge(u, v)

            # Find a new candidate node for rewiring
            candidates = rng.permutation(n)
            for new_v in candidates:
                if (new_v != u or self_loops) and not G.has_edge(u, new_v):
                    G.add_edge(u, new_v, weight=rng.choice([-1, 1]))
                    break

    return G
