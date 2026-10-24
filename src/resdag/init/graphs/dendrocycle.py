from typing import Iterable

import numpy as np
from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology
from resdag.utils.general import create_rng

from ._dense import adjacency_to_graph, register_dense_adjacency


def _dendrocycle_adjacency(
    n: int,
    c: float,
    d: float,
    core_weight: float,
    dendritic_weight: float,
    quiescent_weight: float,
    quiescent_prob: float,
    L: int | Iterable[int] | None,
    w: float,
    alpha: float,
    seed: int | np.random.Generator | None,
    *,
    require_positive_c: bool,
) -> np.ndarray:
    """Build the dense directed adjacency shared by both dendrocycle builders.

    The core ring, dendritic chains, and optional core chords are `O(n)`
    structural loops. The quiescent DAG — previously an `O(A^2)` Python pair
    loop drawing one scalar `rng.random()` per pair — is drawn as a single
    vectorized strict-upper-triangular `rng.random((A, A)) < quiescent_prob`
    mask under the shuffled topological ordering.

    `require_positive_c` selects the `0 <= c` (dendrocycle) vs. `0 < c`
    (chord_dendrocycle) validation to match each public wrapper's contract.

    Returns
    -------
    numpy.ndarray
        An `(n, n)` `float32` directed weighted adjacency.
    """
    if require_positive_c:
        if not (0 < c <= 1):
            raise ValueError("c must be in (0, 1]")
    else:
        if not (0 <= c <= 1):
            raise ValueError("c must be in (0, 1]")
    if not (0 <= d <= 1) or c + d > 1:
        raise ValueError("d must satisfy: 0 <= d and c + d <= 1")

    rng = create_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.float32)

    # ----- Node counts
    C = max(2, int(round(c * n)))
    D = max(0, int(round(d * n)))

    # Ensure C + D doesn't exceed n due to rounding (prioritize core size).
    if C + D > n:
        D = max(0, n - C)

    A = max(0, n - C - D)

    # Node indexing: core first, dendritic next, quiescent last.
    core_nodes = np.arange(C)
    dend_nodes = np.arange(C, C + D)
    q_nodes = np.arange(C + D, n)

    # ======================================================
    # 1. Core ring
    # ======================================================
    if C > 0:
        nxt = (core_nodes + 1) % C
        adjacency[core_nodes, nxt] = core_weight

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
            prev = int(core_nodes[anchor_idx])
            for j in range(Lchain):
                node = int(dend_nodes[start_idx + j])
                adjacency[prev, node] = dendritic_weight
                prev = node
            start_idx += Lchain

    # ======================================================
    # 3. Quiescent DAG (vectorized strict-upper-triangular draw)
    # ======================================================
    if A > 0:
        topo = q_nodes.copy()
        rng.shuffle(topo)
        # Draw the whole (A, A) block at once, then keep only the strict upper
        # triangle so edges run forward in the shuffled order (a DAG).
        mask = rng.random((A, A)) < quiescent_prob
        mask = np.triu(mask, k=1)
        src_local, dst_local = np.nonzero(mask)
        adjacency[topo[src_local], topo[dst_local]] = quiescent_weight

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

        for Lk in Ls:
            if not (1 <= Lk <= C // 2):
                raise ValueError(f"Chord L_k must be in [1, {C // 2}]. Got {Lk}")

        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0,1]. Given {alpha}")

        for k_idx, Lk in enumerate(Ls):
            wk = w * (alpha**k_idx)
            if wk == 0.0:
                continue
            src = core_nodes
            dst = (core_nodes - Lk) % C
            adjacency[src, dst] = float(wk)

    return adjacency


def dendrocycle_adjacency(
    n: int,
    c: float,
    d: float,
    core_weight: float = 1.0,
    dendritic_weight: float = 1.0,
    quiescent_weight: float = 1.0,
    L: int | Iterable[int] | None = None,
    w: float = 0.5,
    alpha: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Build the dense directed adjacency of a dendrocycle graph in NumPy.

    Vectorized construction of the same adjacency `dendrocycle_graph`
    describes. The quiescent DAG uses a strict-upper-triangular
    `rng.random((A, A)) < 0.5` draw rather than the original `O(A^2)` Python
    pair loop, so the draw order — and therefore the exact set of quiescent
    edges under a given seed — differs from the pre-vectorization builder while
    remaining deterministic and statistically equivalent (Bernoulli(0.5) forward
    edges on the shuffled ordering). See `dendrocycle_graph`.

    See Also
    --------
    dendrocycle_graph : NetworkX-returning wrapper around this builder.
    """
    return _dendrocycle_adjacency(
        n,
        c,
        d,
        core_weight,
        dendritic_weight,
        quiescent_weight,
        0.5,
        L,
        w,
        alpha,
        seed,
        require_positive_c=False,
    )


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
@register_dense_adjacency(dendrocycle_adjacency)
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

    The adjacency is built with vectorized NumPy (see
    `dendrocycle_adjacency`); the NetworkX object is materialised only for
    direct callers, and the dense topology fast path skips it entirely.

    **Note.**

    The quiescent DAG block is now drawn with a single vectorized
    `rng.random((A, A)) < 0.5` mask instead of a per-pair scalar loop. The
    result is still deterministic under a seed and statistically identical
    (Bernoulli(0.5) forward edges on the shuffled ordering), but the exact
    edge set for a given `seed` differs from earlier versions.

    Parameters
    ----------
    n : int
        Total number of nodes in the graph.

    c : float
        Fraction of nodes assigned to the core cycle. The effective number
        of core nodes is `C = max(2, round(c * n))`.

    d : float
        Fraction of nodes assigned to dendritic chains. Must satisfy
        `0 <= d` and `c + d <= 1`.

    core_weight : float, default=1.0
        Weight assigned to edges in the core cycle.

    dendritic_weight : float, default=1.0
        Weight assigned to edges in dendritic chains.

    quiescent_weight : float, default=1.0
        Weight assigned to edges in the quiescent DAG.

    L : int or iterable of int, optional
        Backward chord lengths on the core ring. For each `L_k` in `L`,
        edges of the form `i -> (i - L_k) mod C` are added for all core
        nodes `i`. If `None`, no chords are added.

    w : float, default=0.5
        Base weight assigned to chord edges.

    alpha : float, default=1.0
        Geometric decay factor for chord weights. The weight assigned to
        chords of index `k` in `L` is `w * alpha**k`. Decay is applied
        according to the order of `L`, not the magnitude of `L_k`.

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
    adjacency = dendrocycle_adjacency(
        n,
        c,
        d,
        core_weight,
        dendritic_weight,
        quiescent_weight,
        L,
        w,
        alpha,
        seed,
    )
    return adjacency_to_graph(adjacency, directed=True)
