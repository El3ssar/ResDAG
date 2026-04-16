"""
Topology Registry
=================

This module provides a registry of pre-configured graph topologies that can
be referenced by name when creating :class:`~resdag.layers.ESNLayer`.

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
    :class:`~resdag.layers.ESNLayer`.

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

    With ESNLayer:

    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
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


def show_topologies(name: str | None = None) -> None:
    """
    Show available topologies or details for a specific topology.

    Parameters
    ----------
    name : str, optional
        Name of topology to inspect. If None, prints all
        registered topology names.

    Returns
    -------
    None
        Prints formatted information to stdout.

    Raises
    ------
    ValueError
        If the specified topology name is not registered.
    """
    if name is None:
        print("\nAvailable topologies:\n")
        for n in sorted(_TOPOLOGY_REGISTRY):
            print(f"  - {n}")
        print(f"\nTotal: {len(_TOPOLOGY_REGISTRY)}\n")
        return

    if name not in _TOPOLOGY_REGISTRY:
        available = "\n".join(sorted(_TOPOLOGY_REGISTRY.keys()))
        raise ValueError(f"Unknown topology '{name}'.\nAvailable:\n{available}")

    graph_func, default_kwargs = _TOPOLOGY_REGISTRY[name]

    sig = inspect.signature(graph_func)

    print(f"\nTopology: {name}\n")
    print("Parameters:\n")

    for param_name, param in sig.parameters.items():
        if param_name == "n":
            continue

        # Type extraction
        if param.annotation is not inspect.Parameter.empty:
            origin = get_origin(param.annotation)
            if origin is None:
                type_str = param.annotation.__name__
            else:
                args = get_args(param.annotation)
                type_str = " | ".join(a.__name__ for a in args)
        else:
            type_str = "Any"

        # Default resolution
        if param.default is not inspect.Parameter.empty:
            default = param.default
        else:
            default = default_kwargs.get(param_name, "<required>")

        print(f"  - {param_name}")
        print(f"      type:    {type_str}")
        print(f"      default: {default}\n")
