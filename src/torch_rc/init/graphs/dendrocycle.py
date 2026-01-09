from typing import Optional

import numpy as np
from networkx import DiGraph

from torch_rc.init.topology.registry import register_graph_topology
from torch_rc.utils.general import create_rng


@register_graph_topology(
    "dendrocycle",
    c=0.5,
    d=0.2,
    core_weight=1.0,
    dendritic_weight=1.0,
    quiescent_weight=1.0,
    seed=None,
)
def dendrocycle_graph(
    n: int,
    c: float,
    d: float,
    core_weight: float = 1.0,
    dendritic_weight: float = 1.0,
    quiescent_weight: float = 1.0,
    seed: Optional[int] = None,
) -> DiGraph:
    """
    Generate a directed "dendro-cycle" graph.

    Parameters
    ----------
    n : int
        Total number of nodes.
    c : float
        Fraction of nodes in the core cycle (0 < c < 1).
    d : float
        Fraction of nodes in dendrites (0 <= d < 1, c + d <= 1).
    core_weight: float
        Single weight for all core edges.
    dendritic_weight: float
        Single weight for all dendritic edges.
    quiescent_weight: float
        Single weight for all quiescent edges.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.DiGraph
        A directed graph with:
          - one directed cycle (core)
          - outward dendritic chains evenly distributed around it
          - optional disconnected quiescent DAG nodes
    """
    if not (0 < c <= 1):
        raise ValueError("c must be in (0, 1)")
    if not (0 <= d <= 1) or c + d > 1:
        raise ValueError("d must satisfy 0 <= d and c + d <= 1")

    rng = create_rng(seed)
    G = DiGraph()

    # --- 1. Compute node counts
    C = max(2, int(round(c * n)))  # cycle
    D = max(0, int(round(d * n)))  # dendritic
    A = max(0, n - C - D)  # quiescent

    # --- 2. Core cycle
    core_nodes = list(range(C))
    G.add_nodes_from(core_nodes, role="core")

    for i in range(C):
        G.add_edge(core_nodes[i], core_nodes[(i + 1) % C], weight=core_weight)

    # --- 3. Dendrites (uniformly distributed around the ring)
    dend_nodes = list(range(C, C + D))
    G.add_nodes_from(dend_nodes, role="dendritic")

    if D > 0:
        k = min(C, max(1, int(np.sqrt(D))))  # number of dendrites
        base_len = D // k
        remainder = D % k
        lengths = [base_len + (1 if i < remainder else 0) for i in range(k)]

        # Evenly spaced anchors around the cycle
        anchor_indices = np.linspace(0, C, num=k, endpoint=False, dtype=int)

        start_idx = 0
        for anchor_idx, L in zip(anchor_indices, lengths):
            anchor_node = core_nodes[anchor_idx]
            prev = anchor_node
            for j in range(L):
                node = dend_nodes[start_idx + j]
                G.add_edge(prev, node, weight=dendritic_weight)
                prev = node
            start_idx += L

    # --- 4. Quiescent DAG nodes
    if A > 0:
        q_nodes = list(range(C + D, n))
        G.add_nodes_from(q_nodes, role="quiescent")

        topo = q_nodes.copy()
        rng.shuffle(topo)
        for i in range(len(topo)):
            for j in range(i + 1, len(topo)):
                if rng.random() < 0.1:
                    G.add_edge(topo[i], topo[j], weight=quiescent_weight)
    return G
