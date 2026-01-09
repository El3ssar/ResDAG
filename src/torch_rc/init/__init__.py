"""Initialization utilities for torch_rc.

This module contains weight initialization strategies and graph topologies
for reservoir computing architectures.

Submodules:
- graphs: NetworkX-based graph generation functions
- input_feedback: Initializers for input and feedback weights
- topology: Graph topology initializers for recurrent weights
- utils: Utility functions for initialization
"""

from . import graphs, input_feedback, topology, utils

__all__ = [
    "graphs",
    "input_feedback",
    "topology",
    "utils",
]
