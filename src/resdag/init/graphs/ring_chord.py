from typing import Iterable

from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology


@register_graph_topology(
    "ring_chord",
    L=1,
    w=0.5,
    alpha=1.0,
)
def ring_chord_graph(
    n: int,
    L: int | Iterable[int] = 1,
    w: float = 0.5,
    alpha: float = 1.0,
) -> DiGraph:
    """
    Generates a small-world digraph with a ring and backward chords.
      - ring edges i -> (i+1) mod n with weight 1.0
      - backward chords i -> (i - L_k) mod n with weight w * alpha**k

    Parameters
    ----------
    n : int
        Number of nodes (>= 3).
    L : int | Iterable[int]
        One delay or a small list of delays (each in [1, n//2]).
    w : float
        Base chord weight (ratio vs. ring=1). Must be >= 0.
    alpha : float
        Geometric decay across multiple chords (0<alpha<=1). Ignored if L is int.

    Returns
    -------
    networkx.DiGraph
        A small-world digraph with a ring and backward chords.
    """
    if n < 3:
        raise ValueError("n >= 3 required.")
    if isinstance(L, int):
        Ls = [L]
    else:
        Ls = list(L)
        if not Ls:
            raise ValueError("At least one delay L_k is required.")
    if any(not (1 <= Lk <= n // 2) for Lk in Ls):
        raise ValueError("Each L_k must satisfy 1 <= L_k <= n//2.")
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1]. Given alpha = {alpha}")

    G = DiGraph()
    G.add_nodes_from(range(n))

    # Ring (forward) with unit weight
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=1.0)

    # Backward chords with geometric weighting
    for k, Lk in enumerate(Ls):
        wk = w * (alpha**k)
        if wk == 0.0:
            continue
        for i in range(n):
            G.add_edge(i, (i - Lk) % n, weight=float(wk))

    return G
