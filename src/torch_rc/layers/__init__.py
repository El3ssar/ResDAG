"""Layer implementations for torch_rc.

This module contains the core layer implementations:
- TorchRCModule: Base class for all torch_rc modules
- ReservoirLayer: Stateful RNN with graph-based weight initialization
- ReadoutLayer: Per-timestep linear layer with custom fitting
"""

from .custom import (
    Concatenate,
    FeaturePartitioner,
    OutliersFilteredMean,
    SelectiveDropout,
    SelectiveExponentiation,
)
from .readouts import CGReadoutLayer
from .reservoir import ReservoirLayer

__all__ = [
    "ReservoirLayer",
    "CGReadoutLayer",
    "Concatenate",
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "SelectiveDropout",
    "SelectiveExponentiation",
]
