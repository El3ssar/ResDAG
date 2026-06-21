"""
Readout Layers
==============

This module provides readout layer implementations for ESN models.

Classes
-------
ReadoutLayer
    Base per-timestep linear layer with fitting interface.
CGReadoutLayer
    Readout with Conjugate Gradient ridge regression solver.
RidgeReadoutLayer
    Direct ridge readout via Cholesky (``solver='cholesky'``) or LU
    (``solver='solve'``) factorisation of the normal equations.
CholeskyReadoutLayer
    Single-shot Cholesky ridge readout; the streaming-path direct solver.
SVDReadoutLayer
    Readout solved via SVD with Tikhonov filter factors; robust to
    rank-deficient Gram matrices and ``alpha=0``.
PinvReadoutLayer
    Least-squares readout via ``torch.linalg.lstsq`` / ``pinv`` with an
    ``rcond`` cutoff.
IncrementalRidgeReadout
    Streaming ridge readout: ``partial_fit`` accumulates sufficient
    statistics chunk-by-chunk, ``finalize`` solves once. For the
    DataLoader / long-sequence path.

Examples
--------
>>> from resdag.layers.readouts import CGReadoutLayer
>>> readout = CGReadoutLayer(
...     in_features=200,
...     out_features=3,
...     alpha=1e-6,
...     name="output",
... )
>>> # Fit using ESNTrainer or directly
>>> readout.fit(states, targets)
>>> output = readout(states)

See Also
--------
resdag.training.ESNTrainer : Trainer that uses these readouts.
resdag.layers.ESNLayer : ESN layer for generating states.
"""

from .base import ReadoutLayer
from .cg_readout import CGReadoutLayer
from .cholesky_readout import CholeskyReadoutLayer
from .incremental_ridge import IncrementalRidgeReadout
from .pinv_readout import PinvReadoutLayer
from .ridge_readout import RidgeReadoutLayer
from .svd_readout import SVDReadoutLayer

__all__ = [
    "ReadoutLayer",
    "CGReadoutLayer",
    "RidgeReadoutLayer",
    "CholeskyReadoutLayer",
    "SVDReadoutLayer",
    "PinvReadoutLayer",
    "IncrementalRidgeReadout",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
