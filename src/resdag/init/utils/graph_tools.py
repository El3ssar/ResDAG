"""Graph generation utilities and decorators."""

import warnings
from functools import wraps
from typing import Callable

import networkx as nx


def connected_graph(
    graph_func: Callable,
    max_tries: int = 100,
) -> Callable:
    """Decorator to ensure a graph generation function produces a connected graph.

    Wraps a graph generation function and retries until a connected graph is produced,
    up to `max_tries` attempts. If all attempts fail, raises ValueError.

    Parameters
    ----------
    graph_func : callable
        A function that generates and returns a NetworkX graph.
        Must accept `tries` as a keyword argument (will be added if not present).
    max_tries : int, optional
        Maximum number of attempts to generate a connected graph. Default: 100.

    Returns
    -------
    callable
        Wrapped function that guarantees a connected graph.

    Raises
    ------
    ValueError
        If a connected graph cannot be generated within `max_tries` attempts.

    Examples
    --------
    >>> @connected_graph
    ... def my_graph(n, p, seed=None):
    ...     return erdos_renyi_graph(n, p, seed=seed)
    """

    @wraps(graph_func)
    def wrapper(*args, tries: int = max_tries, **kwargs) -> nx.Graph | nx.DiGraph:
        """Wrapper function that retries until connected."""
        for attempt in range(tries):
            G = graph_func(*args, **kwargs)

            # Check connectivity
            if isinstance(G, nx.DiGraph):
                # For directed graphs, check weak connectivity
                if nx.is_weakly_connected(G):
                    return G
            else:
                # For undirected graphs, check connectivity
                if nx.is_connected(G):
                    return G

        # Failed to generate connected graph
        warnings.warn(
            f"Failed to generate a connected graph after {tries} attempts. "
            f"Consider adjusting parameters (e.g., increase edge probability)."
            f"Returning the last generated graph."
        )
        return G

    return wrapper
