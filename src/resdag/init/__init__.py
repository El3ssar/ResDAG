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
matrices
    Direct matrix-builder topologies (non-graph).
topology
    Topology initializers for recurrent weights (graph- or matrix-based).
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
resdag.layers.ESNLayer : Uses these initializers for weight matrices.
"""

from . import graphs, input_feedback, matrices, recurrent, topology, utils
from .recurrent import (
    get_recurrent_initializer,
    register_recurrent_initializer,
    show_recurrent_initializers,
)
from .utils import RecurrentInitializerSpec, resolve_recurrent_initializer, select_recurrent_spec

__all__ = [
    "graphs",
    "input_feedback",
    "matrices",
    "recurrent",
    "topology",
    "utils",
    "get_recurrent_initializer",
    "register_recurrent_initializer",
    "show_recurrent_initializers",
    "RecurrentInitializerSpec",
    "resolve_recurrent_initializer",
    "select_recurrent_spec",
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
