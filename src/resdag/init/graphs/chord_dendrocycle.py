from typing import Iterable

import numpy as np
from networkx import DiGraph

from resdag.init.topology.registry import register_graph_topology

from ._dense import adjacency_to_graph, register_dense_adjacency
from .dendrocycle import _dendrocycle_adjacency


def dendrocycle_with_chords_adjacency(
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
    """Build the dense directed adjacency of a chord-dendrocycle graph in NumPy.

    Vectorized construction of the same adjacency
    :func:`dendrocycle_with_chords_graph` describes. Like
    :func:`~resdag.init.graphs.dendrocycle.dendrocycle_adjacency` but with a
    sparser quiescent DAG (edge probability ``0.1`` rather than ``0.5``). The
    quiescent block is a single strict-upper-triangular
    ``rng.random((A, A)) < 0.1`` draw, so the draw order — and the exact
    quiescent edge set under a given seed — differs from the pre-vectorization
    pair loop while staying deterministic and statistically equivalent.

    See Also
    --------
    dendrocycle_with_chords_graph : NetworkX-returning wrapper around this builder.
    """
    return _dendrocycle_adjacency(
        n,
        c,
        d,
        core_weight,
        dendritic_weight,
        quiescent_weight,
        0.1,
        L,
        w,
        alpha,
        seed,
        require_positive_c=True,
    )


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
@register_dense_adjacency(dendrocycle_with_chords_adjacency)
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

    The adjacency is built with vectorized NumPy (see
    :func:`dendrocycle_with_chords_adjacency`); the NetworkX object is
    materialised only for direct callers, and the dense topology fast path skips
    it entirely.

    .. note::
       The quiescent DAG block is now drawn with a single vectorized
       ``rng.random((A, A)) < 0.1`` mask instead of a per-pair scalar loop. The
       result is still deterministic under a seed and statistically identical
       (Bernoulli(0.1) forward edges on the shuffled ordering), but the exact
       edge set for a given ``seed`` differs from earlier versions.

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
    adjacency = dendrocycle_with_chords_adjacency(
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
