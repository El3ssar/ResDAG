"""
Input/Feedback Weight Initialization
====================================

This module contains initializers for rectangular weight matrices used in
reservoir input and feedback connections.

For reservoirs:
- Feedback weights: ``(reservoir_size, feedback_size)``
- Input weights: ``(reservoir_size, input_size)``

Classes
-------
InputFeedbackInitializer
    Abstract base class for all initializers.
RandomInputInitializer
    Uniform random in [-1, 1] (baseline).
RandomBinaryInitializer
    Binary {-1, +1} values.
PseudoDiagonalInitializer
    Structured block-diagonal pattern.
ChebyshevInitializer
    Deterministic chaotic initialization.
ChessboardInitializer
    Alternating {-1, +1} pattern.
BinaryBalancedInitializer
    Hadamard-based balanced initialization.
OppositeAnchorsInitializer
    Opposite anchor points on ring.
DendrocycleInputInitializer
    Specific to dendrocycle topology.
ChainOfNeuronsInputInitializer
    Specific to chain-of-neurons topology.
RingWindowInputInitializer
    Windowed inputs on ring topology.

Functions
---------
get_input_feedback
    Get an initializer by name.
show_input_initializers
    List available initializers or get details.
register_input_feedback
    Decorator to register new initializers.

Examples
--------
Using pre-registered initializers:

>>> from resdag.init.input_feedback import get_input_feedback
>>> initializer = get_input_feedback("random", input_scaling=0.5)
>>> weight = torch.empty(100, 10)
>>> initializer.initialize(weight)

Registering custom initializers:

>>> from resdag.init.input_feedback import register_input_feedback
>>> @register_input_feedback("my_init", scaling=0.5)
... class MyInitializer(InputFeedbackInitializer):
...     def __init__(self, scaling=0.5):
...         self.scaling = scaling
...     def initialize(self, weight, **kwargs):
...         torch.nn.init.uniform_(weight, -self.scaling, self.scaling)
...         return weight

See Also
--------
resdag.layers.ReservoirLayer : Uses these initializers for weight matrices.
resdag.init.topology : Topology initializers for recurrent weights.
"""

from .base import InputFeedbackInitializer
from .binary_balanced import BinaryBalancedInitializer
from .chain_of_neurons_input import ChainOfNeuronsInputInitializer
from .chebyshev import ChebyshevInitializer
from .chessboard import ChessboardInitializer
from .dendrocycle_input import DendrocycleInputInitializer
from .opposite_anchors import OppositeAnchorsInitializer
from .pseudo_diagonal import PseudoDiagonalInitializer
from .random import RandomInputInitializer
from .random_binary import RandomBinaryInitializer
from .registry import (
    get_input_feedback,
    register_input_feedback,
    show_input_initializers,
)
from .ring_window import RingWindowInputInitializer

__all__ = [
    "BinaryBalancedInitializer",
    "ChainOfNeuronsInputInitializer",
    "ChebyshevInitializer",
    "ChessboardInitializer",
    "DendrocycleInputInitializer",
    "InputFeedbackInitializer",
    "OppositeAnchorsInitializer",
    "PseudoDiagonalInitializer",
    "RandomBinaryInitializer",
    "RandomInputInitializer",
    "RingWindowInputInitializer",
    # Registry functions
    "register_input_feedback",
    "get_input_feedback",
    "show_input_initializers",
]
