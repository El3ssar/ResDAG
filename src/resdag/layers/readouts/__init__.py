"""
Readout Layers
==============

This module provides readout layer implementations for ESN models.

Classes
-------
ReadoutLayer
    Base per-timestep linear layer with fitting interface.
AutoReadoutLayer
    Precision-aware automatic L2 fitting with explicit solver overrides.
CGReadoutLayer
    Readout with Conjugate Gradient ridge regression solver.
LSMRReadoutLayer
    Opt-in CPU float64 matrix-free ridge fit with SciPy LSMR.
RidgeReadoutLayer
    Direct ridge readout via Cholesky (`solver='cholesky'`) or LU
    (`solver='solve'`) factorisation of the normal equations.
CholeskyReadoutLayer
    Single-shot Cholesky ridge readout; the streaming-path direct solver.
SVDReadoutLayer
    Readout solved via SVD with Tikhonov filter factors; robust to
    rank-deficient Gram matrices and `alpha=0`.
PinvReadoutLayer
    Least-squares readout via `torch.linalg.lstsq` / `pinv` with an
    `rcond` cutoff.
IncrementalRidgeReadoutLayer
    Streaming ridge readout: `partial_fit` accumulates sufficient
    statistics chunk-by-chunk, `finalize` solves once. For the
    DataLoader / long-sequence path.
RLSReadout
    True-online Recursive Least Squares readout with a forgetting factor:
    `partial_fit` refines the weights one sample at a time from a running
    inverse-correlation matrix. Tracks non-stationary targets (`lambda < 1`)
    and converges to ordinary least squares (`lambda = 1`).
FISTAReadoutLayer
    Sparse / robust readout: L1 (lasso) or elastic-net regression solved
    with FISTA (accelerated proximal gradient), batched over all outputs.

The old names `IncrementalRidgeReadout` and `FISTAReadout` are retained as
deprecated aliases of `IncrementalRidgeReadoutLayer` / `FISTAReadoutLayer`.

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

from .auto_readout import AutoReadoutLayer, ReadoutSolveReport
from .base import ReadoutLayer
from .cg_readout import CGReadoutLayer
from .cholesky_readout import CholeskyReadoutLayer
from .fista_readout import FISTAReadout, FISTAReadoutLayer
from .incremental_ridge import IncrementalRidgeReadout, IncrementalRidgeReadoutLayer
from .lsmr_readout import LSMRColumnReport, LSMRReadoutLayer, LSMRSolveReport
from .pinv_readout import PinvReadoutLayer
from .ridge_readout import RidgeReadoutLayer
from .rls_readout import RLSReadout
from .svd_readout import SVDReadoutLayer

__all__ = [
    "ReadoutLayer",
    "AutoReadoutLayer",
    "ReadoutSolveReport",
    "CGReadoutLayer",
    "LSMRReadoutLayer",
    "LSMRColumnReport",
    "LSMRSolveReport",
    "RidgeReadoutLayer",
    "CholeskyReadoutLayer",
    "SVDReadoutLayer",
    "PinvReadoutLayer",
    "IncrementalRidgeReadoutLayer",
    "IncrementalRidgeReadout",
    "RLSReadout",
    "FISTAReadoutLayer",
    "FISTAReadout",
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
