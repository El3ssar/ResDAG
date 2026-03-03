"""
Neural Network Layers
=====================

This module provides the core neural network layers for building
Echo State Networks and reservoir computing models.

Classes
-------
ESNLayer
    Stateful RNN reservoir layer for Echo State Networks (public-facing).
ESNCell
    Single-timestep ESN state update cell.
BaseReservoirLayer
    Abstract sequence-loop base with state management.
ReservoirCell
    Abstract single-timestep cell interface.
ReservoirLayer
    Backwards-compatible alias for ESNLayer.
ReadoutLayer
    Per-timestep linear layer with custom fitting interface.
CGReadoutLayer
    ReadoutLayer with Conjugate Gradient ridge regression solver.
Concatenate
    Layer for concatenating multiple inputs along feature dimension.
FeaturePartitioner
    Layer for partitioning features into groups.
OutliersFilteredMean
    Layer for computing mean with outlier filtering.
SelectiveDropout
    Dropout with per-feature selectivity.
SelectiveExponentiation
    Per-feature exponentiation layer.

Examples
--------
>>> from resdag.layers import ESNLayer, CGReadoutLayer
>>> import pytorch_symbolic as ps
>>>
>>> inp = ps.Input((100, 3))
>>> reservoir = ESNLayer(200, feedback_size=3)(inp)
>>> readout = CGReadoutLayer(200, 3)(reservoir)

See Also
--------
resdag.composition.ESNModel : Model composition using these layers.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from .base import BaseReservoirLayer, ReservoirCell
from .custom import (
    Concatenate,
    FeaturePartitioner,
    OutliersFilteredMean,
    Power,
    SelectiveDropout,
    SelectiveExponentiation,
)
from .esn import ESNCell, ESNLayer
from .readouts import CGReadoutLayer, ReadoutLayer

# Backwards-compatible alias — existing code using ReservoirLayer continues to work.
ReservoirLayer = ESNLayer

__all__ = [
    # New hierarchy
    "ReservoirCell",
    "BaseReservoirLayer",
    "ESNCell",
    "ESNLayer",
    # Legacy alias
    "ReservoirLayer",
    # Readouts
    "ReadoutLayer",
    "CGReadoutLayer",
    # Custom layers
    "Concatenate",
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
]
