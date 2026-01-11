"""Input and feedback weight initialization functions.

This module contains initializers for rectangular weight matrices used in
reservoir input and feedback connections. All initializers work with PyTorch's
F.linear convention where weights are (out_features, in_features).

For reservoirs:
- Feedback weights: (reservoir_size, feedback_size)
- Input weights: (reservoir_size, input_size)

Available Initializers
---------------------
- RandomInputInitializer: Uniform random in [-1, 1] (baseline)
- RandomBinaryInitializer: Binary {-1, +1} values
- PseudoDiagonalInitializer: Structured block-diagonal pattern
- ChebyshevInitializer: Deterministic chaotic initialization
- ChessboardInitializer: Alternating {-1, +1} pattern
- BinaryBalancedInitializer: Hadamard-based balanced initialization
- OppositeAnchorsInitializer: Opposite anchor points on ring
- DendrocycleInputInitializer: Specific to dendrocycle topology
- ChainOfNeuronsInputInitializer: Specific to chain-of-neurons topology
- RingWindowInputInitializer: Windowed inputs on ring topology

Registry System
---------------
Use the decorator @register_input_feedback to register new initializers:

    >>> from torch_rc.init.input_feedback import register_input_feedback
    >>> @register_input_feedback("my_init", scaling=0.5)
    ... class MyInitializer(InputFeedbackInitializer):
    ...     def __init__(self, scaling=0.5):
    ...         self.scaling = scaling
    ...     def initialize(self, weight, **kwargs):
    ...         return weight * self.scaling

Then access via:

    >>> from torch_rc.init.input_feedback import get_input_feedback
    >>> init = get_input_feedback("my_init")
"""

# Register all built-in initializers (import after registry to use decorator)
from .base import InputFeedbackInitializer
from .binary_balanced import BinaryBalancedInitializer
from .chain_of_neurons_input import ChainOfNeuronsInputInitializer
from .chebyshev import ChebyshevInitializer
from .chessboard import ChessboardInitializer
from .dendrocycle_input import DendrocycleInputInitializer
from .opposite_anchors import OppositeAnchorsInitializer
from .pseudo_diagonal import PseudoDiagonalInitializer
from .random_binary import RandomBinaryInitializer
from .random_input import RandomInputInitializer
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
