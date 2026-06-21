"""Graph generation utilities and decorators."""

import inspect
from functools import wraps
from typing import Any, Callable

import networkx as nx

from resdag.utils.general import create_rng


def connected_graph(
    graph_func: Callable,
    max_tries: int = 100,
) -> Callable:
    """Decorator to ensure a graph generation function produces a connected graph.

    Wraps a graph generation function and retries until a connected graph is
    produced, up to ``max_tries`` attempts. Each retry advances a single shared
    random generator, so successive attempts draw fresh randomness and differ
    even under a fixed integer ``seed`` (a fresh ``Generator`` rebuilt per call
    would otherwise make every attempt byte-identical). If no attempt yields a
    connected graph, a :class:`ValueError` is raised rather than silently
    returning a disconnected graph.

    Parameters
    ----------
    graph_func : callable
        A function that generates and returns a NetworkX graph. It must accept a
        ``seed`` parameter compatible with :func:`resdag.utils.create_rng`
        (an int, a :class:`numpy.random.Generator`, or ``None``).
    max_tries : int, optional
        Maximum number of attempts to generate a connected graph. Default: 100.

    Returns
    -------
    callable
        Wrapped function that guarantees a connected graph.

    Raises
    ------
    ValueError
        If ``tries`` is less than 1, or if a connected graph cannot be generated
        within ``tries`` attempts.

    Examples
    --------
    >>> @connected_graph
    ... def my_graph(n, p, seed=None):
    ...     return erdos_renyi_graph(n, p, seed=seed)
    """

    @wraps(graph_func)
    def wrapper(*args: Any, tries: int = max_tries, **kwargs: Any) -> nx.Graph | nx.DiGraph:
        """Wrapper function that retries until connected."""
        if tries < 1:
            raise ValueError(f"tries must be a positive integer, got {tries}.")

        # Resolve the requested seed (positional or keyword) into a single shared
        # numpy Generator. Reusing one Generator across attempts means each retry
        # consumes new randomness, so a fixed int seed no longer produces 100
        # byte-identical graphs. ``create_rng`` returns a Generator as-is and
        # derives one from torch's global RNG when seed is None.
        seed = _extract_seed(graph_func, args, kwargs)
        rng = create_rng(seed)
        # Hand the shared Generator to the wrapped function via the seed kwarg,
        # removing any positionally-bound seed so we do not pass it twice.
        args = _strip_positional_seed(graph_func, args)
        kwargs = {**kwargs, "seed": rng}

        for _ in range(tries):
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

        # Failed to generate a connected graph. Honour the documented contract
        # and raise rather than warn-and-return a disconnected graph.
        raise ValueError(
            f"Failed to generate a connected graph after {tries} attempts. "
            f"Consider adjusting parameters (e.g., increase edge probability)."
        )

    return wrapper


def _seed_parameter_index(graph_func: Callable) -> int | None:
    """Return the positional index of ``graph_func``'s ``seed`` parameter.

    Returns ``None`` when ``seed`` is not a positional-capable parameter (e.g.
    the function has no ``seed`` parameter or it is keyword-only).
    """
    try:
        params = list(inspect.signature(graph_func).parameters.values())
    except (TypeError, ValueError):
        return None
    for index, param in enumerate(params):
        if param.name == "seed" and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return index
    return None


def _extract_seed(graph_func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Resolve the seed value passed to ``graph_func`` (keyword or positional)."""
    if "seed" in kwargs:
        return kwargs["seed"]
    index = _seed_parameter_index(graph_func)
    if index is not None and index < len(args):
        return args[index]
    return None


def _strip_positional_seed(graph_func: Callable, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Drop a positionally-passed ``seed`` so it can be re-supplied as a kwarg."""
    index = _seed_parameter_index(graph_func)
    if index is not None and index < len(args):
        return args[:index] + args[index + 1 :]
    return args
