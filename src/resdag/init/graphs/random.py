import networkx as nx
import numpy as np

from resdag.init.topology.registry import register_graph_topology


@register_graph_topology(
    "random",
    density=0.5,
    seed=None,
)
def random_graph(n: int, density: float, seed: int | None = None) -> nx.DiGraph:
    """
    Generate a random directed graph with a given density.

    The adjacency matrix A satisfies:
    - A_ij ~ Uniform(-1, 1) if edge exists
    - A_ij = 0 otherwise
    - Expected density of non-zero entries is `density`

    Parameters
    ----------
    n : int
        Number of nodes.
    density : float
        Proportion of non-zero entries in the adjacency matrix.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        Directed graph with weighted edges.
    """
    rng = np.random.default_rng(seed)

    # Random values
    values = rng.uniform(-1.0, 1.0, size=(n, n))

    # Sparsity mask
    num_nonzeros = int(np.round(density * n * n))
    indices = rng.choice(n * n, size=num_nonzeros, replace=False)
    mask = np.zeros(n * n, dtype=bool)
    mask[indices] = True
    mask = mask.reshape((n, n))

    A = values * mask

    # Build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    rows, cols = np.nonzero(A)
    for i, j in zip(rows, cols):
        G.add_edge(i, j, weight=A[i, j])

    return G
