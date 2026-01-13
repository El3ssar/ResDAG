from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology


@register_graph_topology(
    "simple_cycle_jumps",
    jump_length=1,
    r_c=1.0,
    r_l=0.5,
)
def simple_cycle_jumps_graph(
    n: int,
    jump_length: int,
    r_c: float = 1.0,
    r_l: float = 0.5,
    seed: int | None = None,
) -> DiGraph:
    """
    Generate a directed cycle with bidirectional jumps.

    Parameters
    ----------
    n : int
        Total number of nodes.
    jump_length : int
        Jump step size.
    r_c : float
        Weight for cycle edges.
    r_l : float
        Weight for jump edges.

    Returns
    -------
    G : nx.DiGraph
        Directed graph with:
        - A directed cycle of n nodes (weight = r_c)
        - Bidirectional jump edges every jump_length nodes (weight = r_l)
          until n - (n % jump_length).
    """
    G = DiGraph()
    G.add_nodes_from(range(n))

    # Directed cycle
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=r_c)

    # Bidirectional jumps
    limit = n - (n % jump_length)
    for i in range(0, limit, jump_length):
        j = (i + jump_length) % n
        G.add_edge(i, j, weight=r_l)
        G.add_edge(j, i, weight=r_l)

    return G
