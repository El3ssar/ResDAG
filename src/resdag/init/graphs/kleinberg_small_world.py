import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "kleinberg_small_world",
    q=2,
    k=1,
    directed=False,
    weighted=False,
    beta=2,
    seed=None,
)
def kleinberg_small_world_graph(
    n: int,
    q: float = 2,
    k: int = 1,
    directed: bool = False,
    weighted: bool = False,
    beta: float = 2,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a 2D Kleinberg small-world graph on an ``n x n`` toroidal grid.

    Each node corresponds to a position on the 2D torus (i, j). Local edges connect each
    node to its 4 immediate neighbors (up, down, left, right) with wrapping. Additionally,
    each node gains ``k`` long-range edges, where the probability of connecting to a
    particular node depends on the toroidal Manhattan distance raised to the power ``-q``.

    When ``weighted=True``, weights are assigned as ``distance^beta`` for long-range links.

    Parameters
    ----------
    n : int
        Dimension of the grid; total nodes = n^2.
    q : float, optional
        Exponent controlling the probability of long-range connections. Default: 2.
    k : int, optional
        Number of long-range connections per node. Default: 1.
    directed : bool, optional
        If True, graph is directed; otherwise, undirected. Default: False.
    weighted : bool, optional
        If True, weight of each long-range link is ``distance^beta``; otherwise, it is
        randomly chosen from {-1, 1}. Default: False.
    beta : float, optional
        Exponent used when computing long-range weights if ``weighted=True``. Default: 2.
    seed : int or np.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A Kleinberg small-world graph on an ``n x n`` toroidal grid.
    """
    rng = create_rng(seed)
    G = DiGraph() if directed else Graph()

    def toroidal_manhattan(i1, j1, i2, j2):
        # Wrap distances on a torus
        di = min(abs(i1 - i2), n - abs(i1 - i2))
        dj = min(abs(j1 - j2), n - abs(j1 - j2))
        return di + dj

    # Create nodes
    for i in range(n):
        for j in range(n):
            # Assign a random weight to the node (optionally used or not)
            G.add_node((i, j), weight=rng.choice([-1, 1]))

    # Local edges to 4 neighbors (toroidal wrap)
    for i in range(n):
        for j in range(n):
            neighbors = [
                ((i - 1) % n, j),  # up
                ((i + 1) % n, j),  # down
                (i, (j - 1) % n),  # left
                (i, (j + 1) % n),  # right
            ]
            for neighbor in neighbors:
                weight = rng.choice([-1, 1])
                G.add_edge((i, j), neighbor, weight=weight)
                if not directed:
                    G.add_edge(neighbor, (i, j), weight=weight)

    # Add k long-range connections per node
    for i in range(n):
        for j in range(n):
            candidates = [(x, y) for x in range(n) for y in range(n) if (x, y) != (i, j)]
            distances = np.array(
                [toroidal_manhattan(i, j, x, y) for (x, y) in candidates], dtype=float
            )

            # Probability ~ distance^-q
            probs = distances**-q
            probs /= probs.sum()

            k_eff = min(k, len(candidates))
            chosen = rng.choice(len(candidates), size=k_eff, replace=False, p=probs)
            for idx in chosen:
                target = candidates[idx]
                dist = toroidal_manhattan(i, j, *target)
                weight = (dist**beta) if weighted else rng.choice([-1, 1])
                G.add_edge((i, j), target, weight=weight)
                if not directed:
                    G.add_edge(target, (i, j), weight=weight)

    return G
