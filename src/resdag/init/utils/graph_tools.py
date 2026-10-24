"""Graph generation utilities and decorators."""

import inspect
from functools import wraps
from typing import Any, Callable

import networkx as nx
import numpy as np

from resdag.utils.general import create_rng


def connected_graph(
    graph_func: Callable,
    max_tries: int = 100,
) -> Callable:
    """Decorator to ensure a graph generation function produces a connected graph.

    Wraps a graph generation function and retries until a connected graph is
    produced, up to `max_tries` attempts. Each retry advances a single shared
    random generator, so successive attempts draw fresh randomness and differ
    even under a fixed integer `seed` (a fresh `Generator` rebuilt per call
    would otherwise make every attempt byte-identical). If no attempt yields a
    connected graph, a `ValueError` is raised rather than silently
    returning a disconnected graph.

    Parameters
    ----------
    graph_func : callable
        A function that generates and returns a NetworkX graph. It must accept a
        `seed` parameter compatible with `resdag.utils.create_rng`
        (an int, a `numpy.random.Generator`, or `None`).
    max_tries : int, optional
        Maximum number of attempts to generate a connected graph. Default: 100.

    Returns
    -------
    callable
        Wrapped function that guarantees a connected graph.

    Raises
    ------
    ValueError
        If `tries` is less than 1, or if a connected graph cannot be generated
        within `tries` attempts.

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
    """Return the positional index of `graph_func`'s `seed` parameter.

    Returns `None` when `seed` is not a positional-capable parameter (e.g.
    the function has no `seed` parameter or it is keyword-only).
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
    """Resolve the seed value passed to `graph_func` (keyword or positional)."""
    if "seed" in kwargs:
        return kwargs["seed"]
    index = _seed_parameter_index(graph_func)
    if index is not None and index < len(args):
        return args[index]
    return None


def _strip_positional_seed(graph_func: Callable, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Drop a positionally-passed `seed` so it can be re-supplied as a kwarg."""
    index = _seed_parameter_index(graph_func)
    if index is not None and index < len(args):
        return args[:index] + args[index + 1 :]
    return args


def _is_weakly_connected_adjacency(adjacency: np.ndarray, *, directed: bool) -> bool:
    """Return whether a dense adjacency is (weakly) connected.

    Uses `scipy.sparse.csgraph.connected_components` on a CSR view of the
    adjacency's nonzero pattern — no NetworkX object is built. For a directed
    graph this checks *weak* connectivity (`connection='weak'`), matching the
    `nx.is_weakly_connected` check the NetworkX `@connected_graph` wrapper
    performs; for an undirected graph weak and strong connectivity coincide.

    Parameters
    ----------
    adjacency : numpy.ndarray
        The `(n, n)` weighted adjacency to test.
    directed : bool
        Whether the graph is directed (weak connectivity is checked either way).

    Returns
    -------
    bool
        `True` when the graph has a single (weakly) connected component.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = adjacency.shape[0]
    if n <= 1:
        return True
    # Only the sparsity pattern matters for connectivity; a boolean CSR keeps the
    # component count independent of edge weights (including negative ones).
    sparse = csr_matrix(adjacency != 0.0)
    n_components, _ = connected_components(sparse, directed=directed, connection="weak")
    return bool(n_components == 1)


def connected_adjacency(
    adjacency_builder: Callable[..., np.ndarray],
    *,
    directed: bool,
    max_tries: int = 100,
    precondition: Callable[..., None] | None = None,
) -> Callable[..., np.ndarray]:
    """Wrap a dense-adjacency builder to guarantee a (weakly) connected result.

    The adjacency-level analogue of `connected_graph`: instead of building
    a NetworkX object per attempt and calling `nx.is_(weakly_)connected`, it
    calls the vectorized `adjacency_builder` directly and checks connectivity
    on a CSR view via `scipy.sparse.csgraph.connected_components`. This is
    the fast path used by `resdag.init.topology.GraphTopology` for the
    `connected_*` topologies, which otherwise fell back to the NetworkX
    round-trip (build a graph edge-by-edge, check, then densify back) — pure
    overhead when the inner builder already produces the dense matrix.

    Determinism matches `connected_graph`: a single shared
    `numpy.random.Generator` (resolved from `seed`) is advanced across
    attempts, so successive retries draw fresh randomness and a fixed integer
    `seed` does not produce identical attempts. Exhausting `max_tries`
    raises `ValueError` rather than returning a disconnected matrix,
    honouring the same contract.

    Parameters
    ----------
    adjacency_builder : callable
        A `(n, **kwargs) -> numpy.ndarray` dense-adjacency builder. It must
        accept a `seed` keyword compatible with
        `resdag.utils.create_rng`.
    directed : bool
        Whether the produced adjacency is directed (weak connectivity is checked
        either way, matching the NetworkX wrapper).
    max_tries : int, optional
        Maximum number of attempts. Default: 100.
    precondition : callable, optional
        An optional `(n, **kwargs) -> None` check run once before any attempt
        (e.g. the `p > ln(n)/n` guard for connected Erdős–Rényi). It should
        raise `ValueError` when a connected graph cannot be expected.

    Returns
    -------
    callable
        A `(n, **kwargs) -> numpy.ndarray` builder returning a connected
        adjacency, suitable for `register_dense_adjacency`.

    Raises
    ------
    ValueError
        If `tries` is less than 1, or if a connected adjacency cannot be built
        within `tries` attempts.
    """

    def builder(n: int, *, tries: int = max_tries, **kwargs: Any) -> np.ndarray:
        if tries < 1:
            raise ValueError(f"tries must be a positive integer, got {tries}.")

        if precondition is not None:
            precondition(n, **kwargs)

        # Resolve the requested seed into a single shared Generator and advance
        # it across attempts (mirroring ``connected_graph``): a fixed int seed
        # then yields *different* attempts, not identical ones.
        seed = kwargs.pop("seed", None)
        rng = create_rng(seed)

        for _ in range(tries):
            adjacency = adjacency_builder(n, seed=rng, **kwargs)
            if _is_weakly_connected_adjacency(adjacency, directed=directed):
                return adjacency

        raise ValueError(
            f"Failed to generate a connected graph after {tries} attempts. "
            f"Consider adjusting parameters (e.g., increase edge probability)."
        )

    return builder
