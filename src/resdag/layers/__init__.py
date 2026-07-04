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
CholeskyReadoutLayer
    Single-shot Cholesky ridge readout; the streaming-path direct solver.
SVDReadoutLayer
    SVD filter-factor readout; robust to rank-deficient Gram matrices.
PinvReadoutLayer
    Least-squares readout via ``lstsq`` / pseudo-inverse with ``rcond``.
IncrementalRidgeReadout
    Streaming ridge readout (``partial_fit`` / ``finalize``) for the
    DataLoader / long-sequence path.
FISTAReadout
    Sparse / robust readout: L1 (lasso) or elastic-net regression via FISTA
    (accelerated proximal gradient), batched over all output columns.
Concatenate
    Layer for concatenating multiple inputs along feature dimension.
ExtendedSquare
    Appends the elementwise square of every feature (``[x, x**2]``).
FeaturePartitioner
    Layer for partitioning features into groups.
NLAT2
    T₂ nonlinear algebraic state transform (``x[i-1] * x[i-2]``).
NLAT3
    T₃ nonlinear algebraic state transform (``x[i-1] * x[i+1]``).
Pad
    Appends a constant column to the feature dimension (``[x, padding]``).
PartialSquare
    Squares a leading fraction (or count) of features in place.
Power
    Per-feature exponentiation by a fixed power.
SelectiveDropout
    Dropout with per-feature selectivity.
SelectiveExponentiation
    Per-feature exponentiation layer (``index=0, exponent=2`` = T₁/NLAT1).
Standardize
    Per-feature z-score standardization with fit and inverse.

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

from .cells import (
    ContinuousReservoirCell,
    ES2NCell,
    ESNCell,
    EuSNCell,
    GatedReservoirCell,
    NGCell,
    ReservoirCell,
    ResESNCell,
)
from .readouts import (
    CGReadoutLayer,
    CholeskyReadoutLayer,
    FISTAReadout,
    IncrementalRidgeReadout,
    PinvReadoutLayer,
    ReadoutLayer,
    RidgeReadoutLayer,
    SVDReadoutLayer,
)
from .reservoirs import (
    BaseReservoirLayer,
    ES2NLayer,
    ESNLayer,
    EuSNLayer,
    GatedReservoir,
    NGReservoir,
    ResESNLayer,
)
from .transforms import (
    NLAT2,
    NLAT3,
    Concatenate,
    ExtendedSquare,
    FeaturePartitioner,
    Pad,
    PartialSquare,
    Power,
    SelectiveDropout,
    SelectiveExponentiation,
    Standardize,
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
    "CholeskyReadoutLayer",
    "SVDReadoutLayer",
    "PinvReadoutLayer",
    "IncrementalRidgeReadout",
    "FISTAReadout",
    # Custom layers
    "Concatenate",
    "ExtendedSquare",
    "FeaturePartitioner",
    "NLAT2",
    "NLAT3",
    "Pad",
    "PartialSquare",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
    "Standardize",
    "ES2NCell",
    "ResESNCell",
    "GatedReservoirCell",
    "ES2NLayer",
    "ResESNLayer",
    "GatedReservoir",
    "ContinuousReservoirCell",
    "EuSNCell",
    "EuSNLayer",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
