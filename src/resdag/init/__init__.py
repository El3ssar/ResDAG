"""
Weight Initialization
=====================

This module contains weight initialization strategies and graph topologies
for reservoir computing architectures.

Submodules
----------
graphs
    NetworkX-based graph generation functions.
input_feedback
    Initializers for input and feedback weight matrices.
topology
    Graph topology initializers for recurrent weights.
utils
    Utility functions for initialization.

Examples
--------
Using pre-registered topologies:

>>> from resdag.init.topology import get_topology
>>> topology = get_topology("erdos_renyi", p=0.1)

Using input/feedback initializers:

>>> from resdag.init.input_feedback import get_input_feedback
>>> initializer = get_input_feedback("random", input_scaling=0.5)

See Also
--------
resdag.layers.ReservoirLayer : Uses these initializers for weight matrices.
"""

from . import graphs, input_feedback, topology, utils

__all__ = [
    "graphs",
    "input_feedback",
    "topology",
    "utils",
]
