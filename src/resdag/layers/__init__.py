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
NGReservoir
    Next-Generation Reservoir Computing layer (delay-embedded polynomial
    features; no recurrent weights).
NGCell
    Single-timestep NG-RC feature-construction cell.
BaseReservoirLayer
    Abstract sequence-loop base with state management.
ReservoirCell
    Abstract single-timestep cell interface.
ReadoutLayer
    Per-timestep linear layer with custom fitting interface.
CGReadoutLayer
    ReadoutLayer with Conjugate Gradient ridge regression solver.
RidgeReadoutLayer
    Direct ridge readout (Cholesky / LU solve of the normal equations).
SVDReadoutLayer
    SVD filter-factor readout; robust to rank-deficient Gram matrices.
PinvReadoutLayer
    Least-squares readout via ``lstsq`` / pseudo-inverse with ``rcond``.
Concatenate
    Layer for concatenating multiple inputs along feature dimension.
FeaturePartitioner
    Layer for partitioning features into groups.
Power
    Per-feature exponentiation by a fixed power.
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
resdag.core.ESNModel : Core model class used to compose these layers.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from .cells import ESNCell, NGCell, ReservoirCell
from .readouts import (
    CGReadoutLayer,
    PinvReadoutLayer,
    ReadoutLayer,
    RidgeReadoutLayer,
    SVDReadoutLayer,
)
from .reservoirs import BaseReservoirLayer, ESNLayer, NGReservoir
from .transforms import (
    Concatenate,
    FeaturePartitioner,
    Power,
    SelectiveDropout,
    SelectiveExponentiation,
)

__all__ = [
    # Reservoir base classes
    "BaseReservoirLayer",
    "ReservoirCell",
    # Reservoir layers / cells
    "ESNCell",
    "ESNLayer",
    # NG-RC layers / cells
    "NGCell",
    "NGReservoir",
    # Readouts
    "ReadoutLayer",
    "CGReadoutLayer",
    "RidgeReadoutLayer",
    "SVDReadoutLayer",
    "PinvReadoutLayer",
    # Custom layers
    "Concatenate",
    "FeaturePartitioner",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
