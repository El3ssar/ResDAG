"""
Topology Initialization System
==============================

This module provides the interface between graph implementations and
PyTorch tensor initialization for reservoir recurrent weights.

Classes
-------
TopologyInitializer
    Abstract base class for topology initializers.
GraphTopology
    Concrete implementation using NetworkX graphs.

Functions
---------
get_topology
    Get a pre-configured topology by name.
show_topologies
    List available topologies or get details.
register_graph_topology
    Decorator to register new topologies.

Examples
--------
Using pre-registered topologies:

>>> from resdag.init.topology import get_topology
>>> topology = get_topology("erdos_renyi", p=0.1, seed=42)
>>> weight = torch.empty(100, 100)
>>> topology.initialize(weight, spectral_radius=0.9)

Creating custom topologies:

>>> from resdag.init.topology import GraphTopology
>>> from resdag.init.graphs import watts_strogatz_graph
>>> topology = GraphTopology(
...     watts_strogatz_graph,
...     {"k": 6, "p": 0.2, "directed": True}
... )
>>> topology.initialize(weight, spectral_radius=0.95)

Registering custom topologies:

>>> from resdag.init.topology import register_graph_topology
>>> @register_graph_topology("custom", param=1.0)
... def my_custom_graph(n, param=1.0, seed=None):
...     G = nx.DiGraph()
...     # ... graph generation logic
...     return G

See Also
--------
resdag.init.graphs : Graph generation functions.
resdag.layers.ReservoirLayer : Uses topologies for weight initialization.
"""

from .base import GraphTopology, TopologyInitializer
from .registry import (
    get_topology,
    register_graph_topology,
    show_topologies,
)

__all__ = [
    "GraphTopology",
    "TopologyInitializer",
    "get_topology",
    "register_graph_topology",
    "show_topologies",
]
