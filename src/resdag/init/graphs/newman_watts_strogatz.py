import numpy as np
from networkx import DiGraph, Graph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def newman_watts_strogatz_adjacency(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense weighted adjacency of a Newman-Watts-Strogatz graph.

    NumPy construction of the same adjacency `newman_watts_strogatz_graph`
    describes, without a NetworkX object. The ring lattice is written directly
    into a dense matrix and the shortcut-addition step operates on that matrix
    using adjacency look-ups in place of `has_edge`. Edges are enumerated in
    the same node-major insertion order as the pre-vectorization build, and the
    adjacency this builder returns equals the graph the wrapper returns by
    construction (the wrapper is a thin `adjacency_to_graph` wrap around it).

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to `k // 2` neighbours on each side.
        Incremented by one internally if odd. Must be smaller than `n`.
    p : float
        Probability of adding a long-range (shortcut) edge per existing edge.
    directed : bool, optional
        If `True`, the ring lattice adds both forward and backward edges.
    self_loops : bool, optional
        If `True`, shortcut addition may create self-loops.
    seed : int or numpy.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        An `(n, n)` `float32` weighted adjacency matrix.

    Raises
    ------
    ValueError
        If `k >= n` (not a valid ring lattice).

    See Also
    --------
    newman_watts_strogatz_graph : NetworkX-returning wrapper around this builder.
    """
    if k >= n:
        raise ValueError(f"k must be smaller than n (got k={k}, n={n}).")

    if k % 2 != 0:
        k += 1

    rng = create_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.float32)

    def _has_edge(u: int, v: int) -> bool:
        """Adjacency look-up replacing `G.has_edge` (symmetric if undirected)."""
        if adjacency[u, v] != 0.0:
            return True
        return not directed and adjacency[v, u] != 0.0

    def _set_edge(u: int, v: int, weight: float) -> None:
        adjacency[u, v] = weight
        if not directed:
            adjacency[v, u] = weight

    # Ring lattice. Enumerate edges in the same insertion order as the NetworkX
    # build (node-major, then j = 1..k//2, forward before backward) so the
    # per-edge sign draws stay in the original sequence.
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

    # Shortcut addition (no edge removal). For each existing edge, with
    # probability p connect ``u`` to the first valid node in a fresh permutation.
    # This mirrors the NetworkX loop over ``list(G.edges())`` exactly: an
    # undirected graph yields one ``(u, v)`` per edge with ``u < v`` — the order
    # the ring insertion above produced — so the draw sequence is preserved.
    for u, v in edges:
        if rng.random() < p:
            for new_v in rng.permutation(n):
                new_v = int(new_v)
                if (new_v != u or self_loops) and not _has_edge(u, new_v):
                    _set_edge(u, new_v, _rand_sign(rng))
                    break

    return adjacency


def _rand_sign(rng: np.random.Generator) -> float:
    """Draw a random weight uniformly from `{-1.0, +1.0}`."""
    return -1.0 if rng.random() < 0.5 else 1.0


@register_graph_topology(
    "newman_watts_strogatz",
    k=6,
    p=0.1,
    directed=False,
    self_loops=False,
    seed=None,
)
@register_dense_adjacency(newman_watts_strogatz_adjacency)
def newman_watts_strogatz_graph(
    n: int,
    k: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: int | np.random.Generator | None = None,
) -> DiGraph | Graph:
    """
    Generates a Newman-Watts-Strogatz small-world graph.

    Similar to Watts-Strogatz, except existing edges are **not removed** during rewiring.
    Instead, new edges are added with probability `p`.

    The adjacency is built with NumPy (see
    `newman_watts_strogatz_adjacency`); the NetworkX object is materialised
    only for direct callers, and the dense topology fast path skips it entirely.

    **Note.**

    The ring lattice is filled directly into a dense adjacency matrix instead
    of via per-edge `G.add_edge`/`rng.choice` calls, and the `{-1, +1}`
    edge-sign draw now uses `rng.random() < 0.5`. The result is still
    deterministic under a seed and statistically identical (a ring lattice
    plus Bernoulli(`p`) shortcuts with signed unit weights), but the exact
    edge set for a given `seed` differs from earlier versions.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node initially connects to `k/2` predecessors and `k/2` successors.
        If `k` is odd, it will be incremented by 1 internally.
        Must be smaller than `n`.
    p : float
        Probability of adding a long-range (random) edge.
    directed : bool, optional
        If True, generates a directed graph; otherwise, an undirected graph.
    self_loops : bool, optional
        If True, allows self-loops during the additional edge creation.
    seed : int or np.random.Generator or None, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A Newman-Watts-Strogatz small-world graph.

    Raises
    ------
    ValueError
        If `k >= n`.
    """
    adjacency = newman_watts_strogatz_adjacency(n, k, p, directed, self_loops, seed)
    return adjacency_to_graph(adjacency, directed=directed)
