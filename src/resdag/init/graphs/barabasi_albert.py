import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "barabasi_albert",
    m=1,
    directed=False,
    seed=None,
)
def barabasi_albert_graph(
    n: int,
    m: int,
    directed: bool = False,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a Barabási-Albert scale-free network.

    The Barabási-Albert model grows a graph one node at a time, linking the new node to
    ``m`` existing nodes with probability proportional to their degrees.

    Parameters
    ----------
    n : int
        Total number of nodes in the final graph.
    m : int
        Number of edges each new node creates with already existing nodes.
        Must be >= 1 and < n.
    directed : bool, optional
        If True, creates a directed scale-free network; otherwise, undirected.
    seed : int or np.random.Generator or None, optional
        Seed for the RNG.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A Barabási-Albert scale-free network.

    Raises
    ------
    ValueError
        If ``m < 1 or m >= n``.
    """
    if m < 1 or m >= n:
        raise ValueError(f"m must be >= 1 and < n, got m={m}, n={n}.")

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initialize a complete graph with m nodes
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(j, i, weight=rng.choice([-1, 1]))

    # Keep track of node 'targets' with frequency proportional to node degree
    targets = list(G.nodes) * m

    # Add remaining nodes
    for i in range(m, n):
        G.add_node(i)
        new_edges = rng.choice(targets, size=m, replace=False)
        for t in new_edges:
            G.add_edge(i, t, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(t, i, weight=rng.choice([-1, 1]))

        # Update 'targets' to reflect new degrees
        targets.extend([i] * m)
        targets.extend(new_edges)

    return G
