from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology


@register_graph_topology(
    "multi_cycle",
    k=3,
    weight=1.0,
    start_node=0,
)
def multi_cycle_graph(
    n: int,
    k: int,
    weight: float = 1.0,
    start_node: int = 0,
) -> DiGraph:
    """
    Build a Simple Multi-Cycle Reservoir (SMCR) topology as a NetworkX DiGraph.

    The graph is the disjoint union of `k` identical directed simple cycles (full cycles),
    each of length `n_per_cycle`. Every edge in every cycle carries the same `weight`.
    Node labels are contiguous integers starting at `start_node`.

    Parameters
    ----------
    n : int
        Total number of nodes in the graph.
    k : int
        Number of disjoint cycles (must be >= 1).
    weight : float, default=1.0
        Edge weight assigned to all recurrent connections (typically λ with 0 < λ < 1).
    Returns
    -------
    DiGraph
        Directed graph with the multi-cycle topology and `weight` on every edge.

    Notes
    -----
    - The associated recurrent matrix `W` is block-diagonal with `k` identical
      full-cycle permutation blocks scaled by `weight`.
    - To obtain `W` as a dense matrix, use `smcr_matrix_from_graph(G)` below.
    - This function is fully deterministic.

    Examples
    --------
    >>> G = multi_cycle(n=15, k=3, weight=0.9)
    (15, 15)
    >>> # Convert to dense recurrent matrix (node order is sorted by label):
    >>> W = smcr_matrix_from_graph(G)
    """
    if n % k != 0:
        raise ValueError("N must be divisible by k.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1.")

    n_per_cycle = n // k
    first = start_node
    last = start_node + n - 1

    G = DiGraph()
    G.add_nodes_from(range(first, last + 1))

    for b in range(k):
        base = start_node + b * n_per_cycle
        # nodes in this block: base .. base + n_per_cycle - 1
        for i in range(n_per_cycle):
            u = base + i
            v = base + ((i + 1) % n_per_cycle)
            G.add_edge(u, v, weight=weight)

    return G
