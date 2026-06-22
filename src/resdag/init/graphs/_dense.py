"""Vectorized dense-adjacency fast path for graph topologies.

Most reservoir topologies are *dense generators*: a graph builder constructs an
``nx.Graph`` only for :class:`~resdag.init.topology.GraphTopology` to immediately
collapse it back to a dense ``numpy`` adjacency. For those generators the
intermediate NetworkX object is pure overhead — building it with Python-level
``add_edge`` loops dominates init time at large ``n`` (verified: ``erdos_renyi``
at ``n=2000`` spent ~6.5s in the NetworkX build, larger than the dense
``eigvals``).

This module provides the two halves of the fast path:

- :func:`register_dense_adjacency` decorates a graph function with a companion
  ``adjacency_builder`` callable that returns the dense weighted adjacency
  directly in NumPy. The graph function keeps returning an ``nx`` graph for
  direct callers (and for the ``@connected_graph`` retry wrapper, which needs
  the real graph object), but advertises the vectorized builder for the
  topology hot path.
- :func:`adjacency_to_graph` converts a dense weighted adjacency back into the
  ``nx.Graph``/``nx.DiGraph`` a builder returns, so a single vectorized
  construction feeds both the NetworkX wrapper and the fast path — guaranteeing
  the two are equivalent by construction.

:class:`~resdag.init.topology.GraphTopology` reads the ``adjacency_builder``
attribute (see :func:`dense_adjacency_builder`) and, when present, skips
NetworkX entirely.
"""

from typing import Any, Callable

import networkx as nx
import numpy as np

#: Attribute name under which a graph function advertises its vectorized dense
#: adjacency builder. Read by :class:`~resdag.init.topology.GraphTopology`.
_ADJACENCY_BUILDER_ATTR = "adjacency_builder"


def register_dense_adjacency(
    adjacency_builder: Callable[..., np.ndarray],
) -> Callable[[Callable], Callable]:
    """Attach a vectorized dense-adjacency builder to a graph function.

    The decorated graph function keeps its NetworkX-returning behaviour, but
    gains an ``adjacency_builder`` attribute that
    :class:`~resdag.init.topology.GraphTopology` uses to build the dense
    weighted adjacency directly — skipping the NetworkX round-trip on the
    topology hot path.

    Parameters
    ----------
    adjacency_builder : callable
        A function ``(n, **kwargs) -> numpy.ndarray`` returning the ``(n, n)``
        dense weighted adjacency. It must accept the same keyword arguments as
        the graph function it is attached to.

    Returns
    -------
    callable
        A decorator that sets the ``adjacency_builder`` attribute and returns
        the graph function unchanged.

    Examples
    --------
    >>> @register_dense_adjacency(my_adjacency)
    ... def my_graph(n, p, seed=None):
    ...     return adjacency_to_graph(my_adjacency(n, p, seed=seed), directed=True)
    """

    def decorator(graph_func: Callable) -> Callable:
        setattr(graph_func, _ADJACENCY_BUILDER_ATTR, adjacency_builder)
        return graph_func

    return decorator


def dense_adjacency_builder(graph_func: Callable) -> Callable[..., np.ndarray] | None:
    """Return the vectorized adjacency builder for a graph function, if any.

    Parameters
    ----------
    graph_func : callable
        A graph function, possibly decorated with
        :func:`register_dense_adjacency`.

    Returns
    -------
    callable or None
        The attached ``(n, **kwargs) -> numpy.ndarray`` adjacency builder, or
        ``None`` when the function has no dense fast path.
    """
    builder = getattr(graph_func, _ADJACENCY_BUILDER_ATTR, None)
    return builder if callable(builder) else None


def adjacency_to_graph(
    adjacency: np.ndarray,
    *,
    directed: bool,
) -> nx.Graph | nx.DiGraph:
    """Convert a dense weighted adjacency into a NetworkX graph.

    Builds the ``nx`` object from a vectorized adjacency so that the graph a
    dense generator returns is, by construction, identical to the matrix its
    fast path produces. All ``n`` nodes are added (isolated nodes included) and
    every nonzero entry becomes a weighted edge.

    Parameters
    ----------
    adjacency : numpy.ndarray
        An ``(n, n)`` weighted adjacency matrix. Nonzero ``adjacency[i, j]`` is
        an edge ``i -> j`` with that weight.
    directed : bool
        If ``True``, return an ``nx.DiGraph`` (every nonzero entry is a directed
        edge). If ``False``, return an ``nx.Graph`` (the matrix is assumed
        symmetric; each undirected edge is added once).

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        The graph described by ``adjacency``, with integer node labels
        ``0..n-1`` and ``weight`` edge attributes.
    """
    n = adjacency.shape[0]
    graph: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(n))

    if directed:
        rows, cols = np.nonzero(adjacency)
    else:
        # Only enumerate the upper triangle (incl. diagonal) so each undirected
        # edge is added once; nx mirrors it internally.
        rows, cols = np.nonzero(np.triu(adjacency))

    weights = adjacency[rows, cols]
    edges: list[tuple[int, int, dict[str, Any]]] = [
        (int(i), int(j), {"weight": float(w)})
        for i, j, w in zip(rows.tolist(), cols.tolist(), weights.tolist())
    ]
    graph.add_edges_from(edges)
    return graph
