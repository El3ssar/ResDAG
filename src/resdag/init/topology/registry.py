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

from .base import GraphTopology, MatrixTopology, TopologyInitializer

# Registry of topology names to (builder_func, default_kwargs, wrapper_class,
# prescaled).  wrapper_class is GraphTopology for NetworkX-based builders and
# MatrixTopology for direct matrix builders.  ``prescaled`` flags builders that
# bake their own spectral structure into the matrix, so the wrapper skips the
# outer spectral-radius rescale (see :class:`TopologyInitializer.prescaled`).
# The wrapper type is the concrete union (not the abstract base) so the
# ``wrapper_class(builder_func, kwargs, prescaled=...)`` construction in
# :func:`get_topology` type-checks against the real constructors.
_TOPOLOGY_REGISTRY: dict[
    str, tuple[Callable, dict[str, Any], type[GraphTopology] | type[MatrixTopology], bool]
] = {}


def register_graph_topology(
    name: str,
    prescaled: bool = False,
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
    prescaled : bool, default=False
        Mark the topology as already carrying its own spectral structure (e.g.
        ``spectral_cascade``'s graded per-clique radii). When ``True`` the
        wrapper :class:`~resdag.init.topology.GraphTopology` skips the outer
        :func:`~resdag.init.topology.scale_to_spectral_radius` rescale and warns
        if a layer-level ``spectral_radius`` is also requested.
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
        _TOPOLOGY_REGISTRY[name] = (graph_func, default_kwargs, GraphTopology, prescaled)
        return graph_func

    return decorator


def register_matrix_topology(
    name: str,
    prescaled: bool = False,
    **default_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """
    Decorator to register a matrix-building function as a topology.

    The complement of :func:`register_graph_topology` for builders that
    construct the recurrent weight matrix directly — no graph involved.
    Any logic that produces a square matrix qualifies.

    Parameters
    ----------
    name : str
        Unique name for the topology.
    prescaled : bool, default=False
        Mark the topology as already fixing its own spectral structure (e.g.
        ``orthogonal``'s unit singular values, or
        ``fast_spectral_initialization``'s analytically targeted radius). When
        ``True`` the wrapper :class:`~resdag.init.topology.MatrixTopology` skips
        the outer :func:`~resdag.init.topology.scale_to_spectral_radius` rescale
        and warns if a layer-level ``spectral_radius`` is also requested.
    **default_kwargs
        Default keyword arguments for the matrix function.

    Returns
    -------
    callable
        Decorator function that registers and returns the matrix function.

    Raises
    ------
    ValueError
        If a topology with the same name is already registered.

    Examples
    --------
    >>> @register_matrix_topology("block_diagonal", blocks=4)
    ... def block_diagonal(n, blocks=4, seed=None):
    ...     w = torch.zeros(n, n)
    ...     size = n // blocks
    ...     for b in range(blocks):
    ...         s = b * size
    ...         w[s : s + size, s : s + size] = torch.randn(size, size)
    ...     return w

    >>> reservoir = ESNLayer(500, feedback_size=3, topology="block_diagonal")

    Notes
    -----
    - Matrix functions must accept ``n`` (matrix size) as first parameter and
      return an ``(n, n)`` ``torch.Tensor`` or ``numpy.ndarray``.
    - Registered topologies are accessed via :func:`get_topology` or by name
      in ``ESNLayer(topology="...")``, identically to graph topologies.
    """

    def decorator(matrix_func: Callable) -> Callable:
        if name in _TOPOLOGY_REGISTRY:
            raise ValueError(f"Topology '{name}' is already registered")
        _TOPOLOGY_REGISTRY[name] = (matrix_func, default_kwargs, MatrixTopology, prescaled)
        return matrix_func

    return decorator


def get_topology(
    name: str,
    **override_kwargs: Any,
) -> TopologyInitializer:
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
    TopologyInitializer
        Configured topology initializer (:class:`GraphTopology` or
        :class:`MatrixTopology`, depending on how the name was registered).

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

    builder_func, default_kwargs, wrapper_class, prescaled = _TOPOLOGY_REGISTRY[name]

    # Merge default kwargs with overrides
    kwargs = {**default_kwargs, **override_kwargs}

    return wrapper_class(builder_func, kwargs, prescaled=prescaled)


def show_topologies(name: str | None = None) -> list[str] | None:
    """
    Show available topologies or details for a specific topology.

    Parameters
    ----------
    name : str, optional
        Name of topology to inspect. If None, prints all
        registered topology names *and* returns them as a list.

    Returns
    -------
    list of str or None
        When ``name is None``, returns the sorted list of registered
        topology names (in addition to printing them).  When ``name`` is
        provided, returns ``None`` after printing the parameter table.

    Raises
    ------
    ValueError
        If the specified topology name is not registered.
    """
    if name is None:
        names = sorted(_TOPOLOGY_REGISTRY)
        print("\nAvailable topologies:\n")
        for n in names:
            print(f"  - {n}")
        print(f"\nTotal: {len(names)}\n")
        return names

    if name not in _TOPOLOGY_REGISTRY:
        available = "\n".join(sorted(_TOPOLOGY_REGISTRY.keys()))
        raise ValueError(f"Unknown topology '{name}'.\nAvailable:\n{available}")

    builder_func, default_kwargs, wrapper_class, prescaled = _TOPOLOGY_REGISTRY[name]

    sig = inspect.signature(builder_func)

    kind = "graph" if wrapper_class is GraphTopology else "matrix"
    prescaled_note = ", pre-scaled" if prescaled else ""
    print(f"\nTopology: {name} ({kind}{prescaled_note})\n")
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

    return None
