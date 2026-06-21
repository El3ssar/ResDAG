import networkx as nx
import numpy as np

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def random_adjacency(n: int, density: float, seed: int | None = None) -> np.ndarray:
    """Build the dense weighted adjacency of a random directed graph.

    Vectorized NumPy construction of the same adjacency that
    :func:`random_graph` describes: ``Uniform(-1, 1)`` weights on a random subset
    of exactly ``round(density * n * n)`` entries.

    Parameters
    ----------
    n : int
        Number of nodes.
    density : float
        Proportion of non-zero entries in the adjacency matrix.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` weighted adjacency matrix.

    See Also
    --------
    random_graph : NetworkX-returning wrapper around this builder.
    """
    rng = create_rng(seed)

    values = rng.uniform(-1.0, 1.0, size=(n, n))

    num_nonzeros = int(np.round(density * n * n))
    indices = rng.choice(n * n, size=num_nonzeros, replace=False)
    mask = np.zeros(n * n, dtype=bool)
    mask[indices] = True
    mask = mask.reshape((n, n))

    return (values * mask).astype(np.float32)


@register_graph_topology(
    "random",
    density=0.5,
    seed=None,
)
@register_dense_adjacency(random_adjacency)
def random_graph(n: int, density: float, seed: int | None = None) -> nx.DiGraph:
    """
    Generate a random directed graph with a given density.

    The adjacency matrix A satisfies:
    - A_ij ~ Uniform(-1, 1) if edge exists
    - A_ij = 0 otherwise
    - Expected density of non-zero entries is `density`

    The adjacency is built with vectorized NumPy (see
    :func:`random_adjacency`); the NetworkX object is materialised only for
    direct callers, and the dense topology fast path skips it entirely.

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
    adjacency = random_adjacency(n, density, seed)
    graph = adjacency_to_graph(adjacency, directed=True)
    assert isinstance(graph, nx.DiGraph)
    return graph
