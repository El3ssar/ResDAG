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
    `m` *distinct* existing nodes chosen with probability proportional to their current
    degree (preferential attachment). This yields a degree distribution that is
    approximately power-law, `P(k) ~ k^-3`, for large `n`.

    **Note.**

    The degree-frequency multiset is now held in a preallocated integer array
    and sampled with `rng.integers` instead of `rng.choice` over a growing
    Python list (an `O(len)` list-to-array conversion per draw). The result
    is still deterministic under a seed and statistically identical
    (preferential attachment with `m` distinct targets per node), but the
    exact edge set for a given `seed` differs from earlier versions.

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
        `m * (m - 1) // 2 + m * (n - m)` edges.

    Raises
    ------
    ValueError
        If `m < 1 or m >= n`.

    Notes
    -----
    The `m` targets for each new node are drawn *without replacement at the node-id
    level* (not merely at the multiset-position level), so every new node receives
    exactly `m` distinct neighbours. This is the correct Barabási-Albert behaviour;
    sampling distinct positions of a degree-frequency multiset can return the same node
    id twice and leave nodes with fewer than `m` neighbours.

    Two deliberate departures from the textbook model:

    - When `directed=True` each attachment adds *both* `i -> t` and `t -> i`,
      which doubles the edge count relative to the undirected graph and therefore
      changes the degree statistics (in-degree and out-degree are coupled).
    - The initial `m`-node seed is a complete graph wired *uniformly* rather than by
      preferential attachment; the scale-free regime emerges only as `n` grows past
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
            G.add_edge(i, j, weight=_rand_sign(rng))
            if directed:
                G.add_edge(j, i, weight=_rand_sign(rng))

    # Degree-frequency multiset used for preferential attachment: drawing a
    # uniform position from it picks a node with probability proportional to how
    # often it appears. The seed clique is entered as ``list(G.nodes) * m`` (each
    # of the m seed nodes m times), matching the pre-vectorization builder, and
    # each grown node then extends it by ``2*m`` positions. Held in a
    # preallocated int array (the final length is known) and filled with a write
    # cursor, avoiding the per-draw list-to-array conversion of
    # ``rng.choice(list)``. The ``m*m`` seed positions guarantee a nonempty
    # multiset even for ``m=1`` (where the seed node has degree 0).
    total_positions = m * m + 2 * m * (n - m)
    targets = np.empty(total_positions, dtype=np.int64)
    # Seed clique: each of the m nodes appears m times (as in ``list(nodes) * m``).
    cursor = 0
    for node in range(m):
        targets[cursor : cursor + m] = node
        cursor += m

    # Add remaining nodes.
    for i in range(m, n):
        G.add_node(i)

        # Sample ``m`` *distinct* existing node ids, each draw weighted by the
        # degree-frequency multiset. Positions may repeat node ids, so collect
        # distinct ids explicitly until we have ``m`` of them. ``rng.integers``
        # draws a uniform position index directly into the array — no per-draw
        # array conversion.
        new_edges: list[int] = []
        seen: set[int] = set()
        while len(new_edges) < m:
            pos = int(rng.integers(cursor))
            t = int(targets[pos])
            if t not in seen:
                seen.add(t)
                new_edges.append(t)

        for t in new_edges:
            G.add_edge(i, t, weight=_rand_sign(rng))
            if directed:
                G.add_edge(t, i, weight=_rand_sign(rng))

        # Update the multiset to reflect the new degrees: node ``i`` gains ``m``
        # incident edges (appears m times) and each chosen target gains one.
        targets[cursor : cursor + m] = i
        cursor += m
        targets[cursor : cursor + m] = new_edges
        cursor += m

    return G


def _rand_sign(rng: np.random.Generator) -> int:
    """Draw a random edge weight uniformly from `{-1, +1}`."""
    return -1 if rng.random() < 0.5 else 1
