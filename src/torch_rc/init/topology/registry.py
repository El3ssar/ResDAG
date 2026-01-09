"""Registry for common graph topologies.

This module provides a convenient registry of pre-configured graph topologies
that can be referenced by name in ReservoirLayer.

The registry supports two families of initializers:
1. Graph topologies (square matrices) - for recurrent weights
2. Input/feedback initializers (rectangular matrices) - for input/feedback weights
"""

from typing import Any, Callable, Dict

from .base import GraphTopology

# Registry of topology names to (graph_func, default_kwargs)
_TOPOLOGY_REGISTRY: Dict[str, tuple[Callable, Dict[str, Any]]] = {}


def register_graph_topology(
    name: str,
    **default_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """Decorator to register a graph function as a topology.

    This decorator registers a graph generation function in the topology registry
    at definition time, making it available for use with ReservoirLayer.

    Parameters
    ----------
    name : str
        Name for the topology (must be unique)
    **default_kwargs
        Default keyword arguments for the graph function

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_graph_topology("my_graph", p=0.1, directed=True)
    ... def my_graph(n, p=0.1, directed=False, seed=None):
    ...     G = nx.DiGraph() if directed else nx.Graph()
    ...     # ... graph generation logic
    ...     return G

    Notes
    -----
    - Graph functions must accept `n` (number of nodes) as first parameter
    - Graph functions must return nx.Graph or nx.DiGraph with weighted edges
    - Registered topologies can be accessed via get_topology(name)
    """

    def decorator(graph_func: Callable) -> Callable:
        if name in _TOPOLOGY_REGISTRY:
            raise ValueError(f"Topology '{name}' is already registered")
        _TOPOLOGY_REGISTRY[name] = (graph_func, default_kwargs)
        return graph_func

    return decorator


def get_topology(
    name: str,
    **override_kwargs: Any,
) -> GraphTopology:
    """Get a pre-configured topology initializer by name.

    Parameters
    ----------
    name : str
        Name of the topology (e.g., "erdos_renyi", "watts_strogatz")
    **override_kwargs
        Keyword arguments to override default graph parameters

    Returns
    -------
    GraphTopology
        Topology initializer

    Raises
    ------
    ValueError
        If topology name is not registered

    Examples
    --------
    >>> topology = get_topology("erdos_renyi", p=0.15, seed=42)
    >>> weight = torch.empty(100, 100)
    >>> topology.initialize(weight, spectral_radius=0.9)
    """
    if name not in _TOPOLOGY_REGISTRY:
        available = ", ".join(_TOPOLOGY_REGISTRY.keys())
        raise ValueError(f"Unknown topology '{name}'. Available topologies: {available}")

    graph_func, default_kwargs = _TOPOLOGY_REGISTRY[name]

    # Merge default kwargs with overrides
    kwargs = {**default_kwargs, **override_kwargs}

    return GraphTopology(graph_func, kwargs)


def list_topologies() -> list[str]:
    """List all registered topology names.

    Returns
    -------
    list of str
        Names of all registered topologies
    """
    return sorted(_TOPOLOGY_REGISTRY.keys())
