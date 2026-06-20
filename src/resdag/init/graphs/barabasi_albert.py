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

    The Barabási-Albert model grows a graph one node at a time, linking each new node to
    ``m`` *distinct* existing nodes chosen with probability proportional to their current
    degree (preferential attachment). This yields a degree distribution that is
    approximately power-law, ``P(k) ~ k^-3``, for large ``n``.

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
        A Barabási-Albert scale-free network. An undirected graph has exactly
        ``m * (m - 1) // 2 + m * (n - m)`` edges.

    Raises
    ------
    ValueError
        If ``m < 1 or m >= n``.

    Notes
    -----
    The ``m`` targets for each new node are drawn *without replacement at the node-id
    level* (not merely at the multiset-position level), so every new node receives
    exactly ``m`` distinct neighbours. This is the correct Barabási-Albert behaviour;
    sampling distinct positions of a degree-frequency multiset can return the same node
    id twice and leave nodes with fewer than ``m`` neighbours.

    Two deliberate departures from the textbook model:

    - When ``directed=True`` each attachment adds *both* ``i -> t`` and ``t -> i``,
      which doubles the edge count relative to the undirected graph and therefore
      changes the degree statistics (in-degree and out-degree are coupled).
    - The initial ``m``-node seed is a complete graph wired *uniformly* rather than by
      preferential attachment; the scale-free regime emerges only as ``n`` grows past
      this seed clique.
    """
    if m < 1 or m >= n:
        raise ValueError(f"m must be >= 1 and < n, got m={m}, n={n}.")

    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    # Initialize a complete graph with m nodes (uniform wiring, not preferential).
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(j, i, weight=rng.choice([-1, 1]))

    # Track node 'targets' with frequency proportional to node degree. Drawing a
    # uniform position from this multiset is equivalent to drawing a node with
    # probability proportional to its degree.
    targets = list(G.nodes) * m

    # Add remaining nodes.
    for i in range(m, n):
        G.add_node(i)

        # Sample ``m`` *distinct* existing node ids, each draw weighted by the
        # degree-frequency multiset. ``replace=False`` on the multiset only
        # guarantees distinct positions (which may repeat node ids), so collect
        # distinct ids explicitly until we have ``m`` of them.
        new_edges: list[int] = []
        seen: set[int] = set()
        while len(new_edges) < m:
            t = int(rng.choice(targets))
            if t not in seen:
                seen.add(t)
                new_edges.append(t)

        for t in new_edges:
            G.add_edge(i, t, weight=rng.choice([-1, 1]))
            if directed:
                G.add_edge(t, i, weight=rng.choice([-1, 1]))

        # Update 'targets' to reflect new degrees.
        targets.extend([i] * m)
        targets.extend(new_edges)

    return G
