from typing import Iterable

import numpy as np
from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "dendrocycle",
    c=0.5,
    d=0.2,
    core_weight=1.0,
    dendritic_weight=1.0,
    quiescent_weight=1.0,
    L=None,
    w=0.5,
    alpha=1.0,
    seed=None,
)
def dendrocycle_graph(
    n: int,
    c: float,
    d: float,
    core_weight: float = 1.0,
    dendritic_weight: float = 1.0,
    quiescent_weight: float = 1.0,
    L: int | Iterable[int] | None = None,
    w: float = 0.5,
    alpha: float = 1.0,
    seed: int | None = None,
) -> DiGraph:
    """
    Construct a directed dendrocycle graph with optional small-world chords
    on the core ring.

    The graph is composed of three structured parts:

    1. Core cycle
       A directed ring of C nodes (indices 0 … C-1), forming the only
       guaranteed directed cycle in the graph.

    2. Dendritic chains
       Directed acyclic chains that emanate outward from evenly spaced
       anchor nodes on the core ring. These chains introduce feed-forward
       memory without feedback.

    3. Quiescent DAG
       A directed acyclic subgraph on the remaining nodes, constructed by
       random forward connections under a shuffled topological ordering.

    Optionally, additional backward "chord" edges can be added exclusively
    on the core ring to introduce longer feedback paths. These chords do not
    affect dendritic or quiescent nodes.

    Parameters
    ----------
    n : int
        Total number of nodes in the graph.

    c : float
        Fraction of nodes assigned to the core cycle. The effective number
        of core nodes is ``C = max(2, round(c * n))``.

    d : float
        Fraction of nodes assigned to dendritic chains. Must satisfy
        ``0 <= d`` and ``c + d <= 1``.

    core_weight : float, default=1.0
        Weight assigned to edges in the core cycle.

    dendritic_weight : float, default=1.0
        Weight assigned to edges in dendritic chains.

    quiescent_weight : float, default=1.0
        Weight assigned to edges in the quiescent DAG.

    L : int or iterable of int, optional
        Backward chord lengths on the core ring. For each ``L_k`` in ``L``,
        edges of the form ``i -> (i - L_k) mod C`` are added for all core
        nodes ``i``. If ``None``, no chords are added.

    w : float, default=0.5
        Base weight assigned to chord edges.

    alpha : float, default=1.0
        Geometric decay factor for chord weights. The weight assigned to
        chords of index ``k`` in ``L`` is ``w * alpha**k``. Decay is applied
        according to the order of ``L``, not the magnitude of ``L_k``.

    seed : int, optional
        Seed for the random number generator used in constructing the
        quiescent DAG.

    Returns
    -------
    G : networkx.DiGraph
        A directed graph with a single core cycle, outward dendritic chains,
        an acyclic quiescent subgraph, and optional core-only chord feedback.

    Notes
    -----
    - Apart from the core cycle (and optional chords on it), the graph is
      acyclic.
    - No edges ever point from dendritic or quiescent nodes back into the
      core.
    - Chords only affect the internal feedback structure of the core and do
      not create cycles outside it.
    """
    if not (0 <= c <= 1):
        raise ValueError("c must be in (0, 1]")
    if not (0 <= d <= 1) or c + d > 1:
        raise ValueError("d must satisfy: 0 <= d and c + d <= 1")

    # ----- RNG
    rng = create_rng(seed)
    G = DiGraph()

    # ----- Node counts
    C = max(2, int(round(c * n)))
    D = max(0, int(round(d * n)))

    # Ensure C + D doesn't exceed n due to rounding
    if C + D > n:
        # Prioritize core size, adjust dendritic
        D = max(0, n - C)

    A = max(0, n - C - D)

    # ----- Node indexing: core first, dendritic next, quiescent last
    core_nodes = list(range(C))
    dend_nodes = list(range(C, C + D))
    q_nodes = list(range(C + D, n))

    G.add_nodes_from(core_nodes, role="core")
    G.add_nodes_from(dend_nodes, role="dendritic")
    G.add_nodes_from(q_nodes, role="quiescent")

    # ======================================================
    # 1. Core ring
    # ======================================================
    for i in range(C):
        G.add_edge(core_nodes[i], core_nodes[(i + 1) % C], weight=core_weight)

    # ======================================================
    # 2. Dendrites: deterministic spacing around the ring
    # ======================================================
    if D > 0:
        k = min(C, max(1, int(np.sqrt(D))))  # number of dendrites
        base_len = D // k
        remainder = D % k
        lengths = [base_len + (1 if i < remainder else 0) for i in range(k)]

        anchor_indices = np.linspace(0, C, num=k, endpoint=False, dtype=int)

        start_idx = 0
        for anchor_idx, Lchain in zip(anchor_indices, lengths):
            anchor_node = core_nodes[anchor_idx]
            prev = anchor_node
            for j in range(Lchain):
                node = dend_nodes[start_idx + j]
                G.add_edge(prev, node, weight=dendritic_weight)
                prev = node
            start_idx += Lchain

    # ======================================================
    # 3. Quiescent DAG
    # ======================================================
    if A > 0:
        topo = q_nodes.copy()
        rng.shuffle(topo)
        for i in range(len(topo)):
            for j in range(i + 1, len(topo)):
                if rng.random() < 0.1:
                    G.add_edge(topo[i], topo[j], weight=quiescent_weight)

    # ======================================================
    # 4. Optional small-world chords ON THE CORE RING
    # ======================================================
    if L is not None:
        if isinstance(L, int):
            Ls = [L]
        else:
            Ls = list(L)
            if not Ls:
                raise ValueError("L cannot be an empty iterable.")

        # validate L
        for Lk in Ls:
            if not (1 <= Lk <= C // 2):
                raise ValueError(f"Chord L_k must be in [1, {C // 2}]. Got {Lk}")

        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0,1]. Given {alpha}")

        # add chords i -> (i - Lk) % C on core only
        for k_idx, Lk in enumerate(Ls):
            wk = w * (alpha**k_idx)
            if wk == 0.0:
                continue
            for i in range(C):
                G.add_edge(i, (i - Lk) % C, weight=float(wk))

    return G
