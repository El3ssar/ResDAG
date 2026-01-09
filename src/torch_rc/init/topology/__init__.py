"""Topology system for weight initialization.

This module provides the interface between graph implementations and
PyTorch tensor initialization. It wraps graph functions to create
weight initializers compatible with ReservoirLayer.

Basic Usage
-----------
Using pre-registered topologies:

>>> from torch_rc.init.topology import get_topology
>>> topology = get_topology("erdos_renyi", p=0.1, seed=42)
>>> weight = torch.empty(100, 100)
>>> topology.initialize(weight, spectral_radius=0.9)

Creating custom topologies:

>>> from torch_rc.init.topology import GraphTopology
>>> from torch_rc.init.graphs import watts_strogatz_graph
>>> topology = GraphTopology(
...     watts_strogatz_graph,
...     {"k": 6, "p": 0.2, "directed": True}
... )
>>> topology.initialize(weight, spectral_radius=0.95)

Registering custom topologies with decorator:

>>> from torch_rc.init.topology import register_graph_topology
>>> @register_graph_topology("custom", param=1.0)
... def my_custom_graph(n, param=1.0, seed=None):
...     G = nx.DiGraph()
...     # ... graph generation logic
...     return G
>>> topology = get_topology("custom")

Or programmatically:

>>> from torch_rc.init.topology import register_topology
>>> from my_graphs import my_custom_graph
>>> register_topology("custom", my_custom_graph, {"param": 1.0})
>>> topology = get_topology("custom")
"""

# Register all built-in graph topologies (import after registry to use decorator)
from .base import GraphTopology, TopologyInitializer
from .registry import (
    get_topology,
    list_topologies,
    register_graph_topology,
)

__all__ = [
    "GraphTopology",
    "TopologyInitializer",
    "get_topology",
    "list_topologies",
    "register_graph_topology",
]
