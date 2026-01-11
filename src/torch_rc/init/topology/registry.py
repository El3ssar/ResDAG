"""Registry for common graph topologies.

This module provides a convenient registry of pre-configured graph topologies
that can be referenced by name in ReservoirLayer.

The registry supports two families of initializers:
1. Graph topologies (square matrices) - for recurrent weights
2. Input/feedback initializers (rectangular matrices) - for input/feedback weights
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

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


def show_topologies(name: Optional[str] = None) -> Union[List[str], Dict[str, Any]]:
    """Show available topologies or details for a specific topology.

    Parameters
    ----------
    name : str, optional
        Name of topology to inspect. If None, returns list of all topologies.

    Returns
    -------
    list of str or dict
        If name is None: sorted list of registered topology names.
        If name is provided: dict with 'name', 'defaults', and 'parameters' keys.

    Raises
    ------
    ValueError
        If the specified topology name is not registered.

    Examples
    --------
    >>> show_topologies()
    ['barabasi_albert', 'chain_of_neurons', 'dendrocycle', ...]

    >>> show_topologies("erdos_renyi")
    {
        'name': 'erdos_renyi',
        'defaults': {'p': 0.1, 'directed': False, 'seed': None},
        'parameters': {
            'n': {'type': 'int', 'default': <required>},
            'p': {'type': 'float', 'default': 0.1},
            ...
        }
    }
    """
    if name is None:
        return sorted(_TOPOLOGY_REGISTRY.keys())

    if name not in _TOPOLOGY_REGISTRY:
        available = ", ".join(sorted(_TOPOLOGY_REGISTRY.keys()))
        raise ValueError(f"Unknown topology '{name}'. Available: {available}")

    graph_func, default_kwargs = _TOPOLOGY_REGISTRY[name]

    # Extract function signature
    sig = inspect.signature(graph_func)
    parameters = {}
    for param_name, param in sig.parameters.items():
        param_info: Dict[str, Any] = {}

        # Get type annotation if available
        if param.annotation != inspect.Parameter.empty:
            param_info["type"] = getattr(param.annotation, "__name__", str(param.annotation))
        else:
            param_info["type"] = "Any"

        # Get default value
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default
        elif param_name in default_kwargs:
            param_info["default"] = default_kwargs[param_name]
        else:
            param_info["default"] = "<required>"

        parameters[param_name] = param_info

    return {
        "name": name,
        "defaults": default_kwargs,
        "parameters": parameters,
    }
