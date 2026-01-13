"""
Topology Registry
=================

This module provides a registry of pre-configured graph topologies that can
be referenced by name when creating :class:`~resdag.layers.ReservoirLayer`.

The registry allows convenient access to common topologies without needing
to import graph functions directly.

Functions
---------
register_graph_topology
    Decorator to register new topologies.
get_topology
    Get a topology initializer by name.
show_topologies
    List available topologies or get details.

Examples
--------
>>> from resdag.init.topology import get_topology, show_topologies
>>>
>>> # List all available topologies
>>> show_topologies()
>>>
>>> # Get details for a specific topology
>>> show_topologies("erdos_renyi")
>>>
>>> # Create a topology initializer
>>> topology = get_topology("erdos_renyi", p=0.15)
"""

import inspect
from typing import Any, Callable, get_args, get_origin

from .base import GraphTopology

# Registry of topology names to (graph_func, default_kwargs)
_TOPOLOGY_REGISTRY: dict[str, tuple[Callable, dict[str, Any]]] = {}


def register_graph_topology(
    name: str,
    **default_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """
    Decorator to register a graph function as a topology.

    Registers a graph generation function in the topology registry at
    definition time, making it available for use with
    :class:`~resdag.layers.ReservoirLayer`.

    Parameters
    ----------
    name : str
        Unique name for the topology.
    **default_kwargs
        Default keyword arguments for the graph function.

    Returns
    -------
    callable
        Decorator function that registers and returns the graph function.

    Raises
    ------
    ValueError
        If a topology with the same name is already registered.

    Examples
    --------
    >>> @register_graph_topology("my_graph", p=0.1, directed=True)
    ... def my_graph(n, p=0.1, directed=False, seed=None):
    ...     G = nx.DiGraph() if directed else nx.Graph()
    ...     # ... graph generation logic
    ...     return G

    Notes
    -----
    - Graph functions must accept ``n`` (number of nodes) as first parameter.
    - Graph functions must return ``nx.Graph`` or ``nx.DiGraph`` with weighted edges.
    - Registered topologies can be accessed via :func:`get_topology`.
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
    """
    Get a pre-configured topology initializer by name.

    Parameters
    ----------
    name : str
        Name of the topology (e.g., ``"erdos_renyi"``, ``"watts_strogatz"``).
    **override_kwargs
        Keyword arguments to override default graph parameters.

    Returns
    -------
    GraphTopology
        Configured topology initializer.

    Raises
    ------
    ValueError
        If topology name is not registered.

    Examples
    --------
    Basic usage:

    >>> topology = get_topology("erdos_renyi", p=0.15, seed=42)
    >>> weight = torch.empty(100, 100)
    >>> topology.initialize(weight, spectral_radius=0.9)

    With ReservoirLayer:

    >>> from resdag.layers import ReservoirLayer
    >>> reservoir = ReservoirLayer(
    ...     reservoir_size=500,
    ...     feedback_size=10,
    ...     topology=get_topology("watts_strogatz", k=4, p=0.3),
    ...     spectral_radius=0.95,
    ... )

    See Also
    --------
    show_topologies : List available topologies.
    register_graph_topology : Register new topologies.
    """
    if name not in _TOPOLOGY_REGISTRY:
        available = ", ".join(_TOPOLOGY_REGISTRY.keys())
        raise ValueError(f"Unknown topology '{name}'. Available topologies: {available}")

    graph_func, default_kwargs = _TOPOLOGY_REGISTRY[name]

    # Merge default kwargs with overrides
    kwargs = {**default_kwargs, **override_kwargs}

    return GraphTopology(graph_func, kwargs)


def show_topologies(name: str | None = None) -> list[str] | dict[str, Any] | None:
    """
    Show available topologies or details for a specific topology.

    Parameters
    ----------
    name : str, optional
        Name of topology to inspect. If None, returns list of all
        registered topology names.

    Returns
    -------
    list of str or dict or None
        If ``name`` is None: sorted list of registered topology names.
        If ``name`` is provided: prints details and returns None.

    Raises
    ------
    ValueError
        If the specified topology name is not registered.

    Examples
    --------
    List all topologies:

    >>> show_topologies()
    ['barabasi_albert', 'chain_of_neurons', 'dendrocycle', ...]

    Get details for a specific topology:

    >>> show_topologies("erdos_renyi")
    Topology: erdos_renyi

    Parameters:
      - directed: type=bool, default=True
      - p: type=float, default=0.1
      - seed: type=int | NoneType, default=None
    """
    if name is None:
        return sorted(_TOPOLOGY_REGISTRY.keys())

    if name not in _TOPOLOGY_REGISTRY:
        available = ", ".join(sorted(_TOPOLOGY_REGISTRY.keys()))
        raise ValueError(f"Unknown topology '{name}'. Available: {available}")

    graph_func, default_kwargs = _TOPOLOGY_REGISTRY[name]

    # Extract function signature
    sig = inspect.signature(graph_func)
    types = {}
    for param_name, param in sig.parameters.items():
        if param_name == "n":
            continue

        # Get type annotation if available
        if param.annotation is not inspect.Parameter.empty:
            origin = get_origin(param.annotation)
            if origin is None:
                types[param_name] = param.annotation.__name__
            else:
                args = get_args(param.annotation)
                types[param_name] = " | ".join(a.__name__ for a in args)
        else:
            types[param_name] = "Any"

    info = {
        "name": name,
        "parameters": {
            k: {
                "type": types.get(k, "Any"),
                "default": default_kwargs.get(k, "<required>"),
            }
            for k in sorted(set(types) | set(default_kwargs))
        },
    }
    return _format_topology(info)


def _format_topology(info: dict) -> None:
    """Print formatted topology information."""
    lines = [f"\nTopology: {info['name']}", "", "Parameters:"]
    for name, meta in info["parameters"].items():
        lines.append(f"  - {name}: type={meta['type']}, default={meta['default']}")
    print("\n".join(lines))
