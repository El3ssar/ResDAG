import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def watts_strogatz_adjacency(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense weighted adjacency of a Watts-Strogatz small-world graph.

    NumPy construction of the same adjacency that :func:`watts_strogatz_graph`
    describes, without a NetworkX object: the ring lattice is filled directly
    into a dense matrix, and the rewiring step operates on that matrix using
    adjacency look-ups in place of ``has_edge``. The rewiring loop remains
    edge-sequential (it is intrinsic to the model — each rewire depends on the
    current connectivity), but it iterates over the ``O(n*k)`` ring edges rather
    than building and mutating a graph object.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to ``k // 2`` neighbours on each side.
        Incremented by one internally if odd.
    p : float
        Rewiring probability in ``[0, 1]``.
    directed : bool, optional
        If ``True``, the ring lattice adds both forward and backward edges.
    self_loops : bool, optional
        If ``True``, rewiring may create self-loops.
    seed : int or numpy.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        An ``(n, n)`` ``float32`` weighted adjacency matrix.

    Raises
    ------
    ValueError
        If ``k >= n`` (not a valid ring lattice).

    See Also
    --------
    watts_strogatz_graph : NetworkX-returning wrapper around this builder.
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n}).")

    if k % 2 != 0:
        k += 1

    rng = create_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.float32)

    def _connected(u: int, v: int) -> bool:
        """Adjacency look-up replacing ``G.has_edge`` (symmetric if undirected)."""
        if adjacency[u, v] != 0.0:
            return True
        return not directed and adjacency[v, u] != 0.0

    def _set_edge(u: int, v: int, weight: float) -> None:
        adjacency[u, v] = weight
        if not directed:
            adjacency[v, u] = weight

    def _clear_edge(u: int, v: int) -> None:
        adjacency[u, v] = 0.0
        if not directed:
            adjacency[v, u] = 0.0

    # Enumerate ring-lattice edges in the same insertion order as the NetworkX
    # build so the per-edge rewiring draws stay equivalent. Undirected edges are
    # written symmetrically (matching ``nx.Graph``); directed adds both the
    # forward and backward orientations as the original did.
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            forward = (i + j) % n
            _set_edge(i, forward, _rand_sign(rng))
            edges.append((i, forward))
            if directed:
                backward = (i - j) % n
                _set_edge(i, backward, _rand_sign(rng))
                edges.append((i, backward))

    # Rewire edges with probability p. Each rewire removes the original edge and
    # reconnects ``u`` to the first valid candidate in a fresh permutation,
    # mirroring the NetworkX implementation but checking the dense matrix.
    for u, v in edges:
        if rng.random() < p:
            _clear_edge(u, v)
            for new_v in rng.permutation(n):
                if (new_v != u or self_loops) and not _connected(u, int(new_v)):
                    _set_edge(u, int(new_v), _rand_sign(rng))
                    break

    return adjacency


def _rand_sign(rng: np.random.Generator) -> float:
    """Draw a random weight uniformly from ``{-1.0, +1.0}``."""
    return -1.0 if rng.random() < 0.5 else 1.0


@register_graph_topology(
    "watts_strogatz",
    k=6,
    p=0.1,
    directed=False,
    self_loops=False,
    seed=None,
)
@register_dense_adjacency(watts_strogatz_adjacency)
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

    The adjacency is built with NumPy (see :func:`watts_strogatz_adjacency`);
    the NetworkX object is materialised only for direct callers, and the dense
    topology fast path skips it entirely.

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
    adjacency = watts_strogatz_adjacency(n, k, p, directed, self_loops, seed)
    return adjacency_to_graph(adjacency, directed=directed)
