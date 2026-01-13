from typing import Iterable

import numpy as np
from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng


@register_graph_topology(
    "chord_dendrocycle",
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
def dendrocycle_with_chords_graph(
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
    Dendrocycle + optional small-world chords on the core ring.

    Structure:
      - Core cycle (first C=round(c*n) nodes)                      | weight = core_weight
      - Dendritic chains from core (D=round(d*n) nodes after core) | weight = dendritic_weight
      - Optional quiescent DAG remainder (last A=n-C-D nodes)      | weight = quiescent_weight
      - Optional backward chords ONLY on core ring:
            i -> (i - L_k) mod C with weights w * alpha**k (where L_k is the k-th chord length)

    Parameters
    ----------
    n : int
    c : float
        Fraction for core cycle. (0 < c <= 1)
    d : float
        Fraction for dendrites. 0 <= d and c + d <= 1
    core_weight, dendritic_weight, quiescent_weight : float
    L : int or iterable of ints, optional
        Delays for backward chords. If None, no chords are added.
    w : float
        Base chord weight.
    alpha : float
        Geometric decay for multiple chord lengths.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.DiGraph
        A Dendrocycle + optional small-world chords on the core ring.
    """
    if not (0 < c <= 1):
        raise ValueError("c must be in (0, 1]")
    if not (0 <= d <= 1) or c + d > 1:
        raise ValueError("d must satisfy: 0 <= d and c + d <= 1")

    # ----- RNG
    rng = create_rng(seed)
    G = DiGraph()

    # ----- Node counts
    C = max(2, int(round(c * n)))
    D = max(0, int(round(d * n)))
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
