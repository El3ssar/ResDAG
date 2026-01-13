"""
Neural Network Layers
=====================

This module provides the core neural network layers for building
Echo State Networks and reservoir computing models.

Classes
-------
ReservoirLayer
    Stateful RNN reservoir with graph-based weight initialization.
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
>>> from resdag.layers import ReservoirLayer, CGReadoutLayer
>>> import pytorch_symbolic as ps
>>>
>>> inp = ps.Input((100, 3))
>>> reservoir = ReservoirLayer(200, feedback_size=3)(inp)
>>> readout = CGReadoutLayer(200, 3)(reservoir)

See Also
--------
resdag.composition.ESNModel : Model composition using these layers.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from .custom import (
    Concatenate,
    FeaturePartitioner,
    OutliersFilteredMean,
    SelectiveDropout,
    SelectiveExponentiation,
)
from .readouts import CGReadoutLayer, ReadoutLayer
from .reservoir import ReservoirLayer

__all__ = [
    "ReservoirLayer",
    "ReadoutLayer",
    "CGReadoutLayer",
    "Concatenate",
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "SelectiveDropout",
    "SelectiveExponentiation",
]
