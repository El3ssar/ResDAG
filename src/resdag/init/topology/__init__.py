"""
Topology Initialization System
==============================

This module provides the interface between structure generators — NetworkX
graphs or direct matrix builders — and PyTorch tensor initialization for
reservoir recurrent weights.

Classes
-------
TopologyInitializer
    Abstract base class for topology initializers.
GraphTopology
    Concrete implementation using NetworkX graphs.
MatrixTopology
    Concrete implementation wrapping any matrix-building callable.

Functions
---------
get_topology
    Get a pre-configured topology by name.
show_topologies
    List available topologies or get details.
register_graph_topology
    Decorator to register graph-based topologies.
register_matrix_topology
    Decorator to register matrix-builder topologies.
scale_to_spectral_radius
    Rescale a square matrix to a target spectral radius.
estimate_spectral_radius
    Estimate a matrix's largest absolute eigenvalue (power iteration / sparse
    ``eigs`` / tiny-N dense fallback).

Examples
--------
Using pre-registered topologies:

>>> from resdag.init.topology import get_topology
>>> topology = get_topology("erdos_renyi", p=0.1, seed=42)
>>> weight = torch.empty(100, 100)
>>> topology.initialize(weight, spectral_radius=0.9)

Any function that builds a matrix is a topology:

>>> def block_diagonal(n, blocks=4):
...     ...  # return an (n, n) tensor
>>> reservoir = ESNLayer(500, feedback_size=3, topology=block_diagonal)

Registering custom topologies:

>>> from resdag.init.topology import register_graph_topology, register_matrix_topology
>>> @register_graph_topology("custom", param=1.0)
... def my_custom_graph(n, param=1.0, seed=None):
...     G = nx.DiGraph()
...     # ... graph generation logic
...     return G

>>> @register_matrix_topology("block_diagonal", blocks=4)
... def block_diagonal(n, blocks=4, seed=None):
...     # ... matrix construction logic
...     return w

See Also
--------
resdag.init.graphs : Graph generation functions.
resdag.init.matrices : Direct matrix-construction functions.
resdag.layers.ESNLayer : Uses topologies for weight initialization.
"""

from .base import (
    GraphTopology,
    MatrixTopology,
    TopologyInitializer,
    estimate_spectral_radius,
    scale_to_spectral_radius,
)
from .registry import (
    get_topology,
    register_graph_topology,
    register_matrix_topology,
    show_topologies,
)

__all__ = [
    "GraphTopology",
    "MatrixTopology",
    "TopologyInitializer",
    "estimate_spectral_radius",
    "get_topology",
    "register_graph_topology",
    "register_matrix_topology",
    "scale_to_spectral_radius",
    "show_topologies",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
