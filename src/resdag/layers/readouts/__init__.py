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
resdag.layers.ReservoirLayer : Reservoir layer for generating states.
"""

from .base import ReadoutLayer
from .cg_readout import CGReadoutLayer

__all__ = ["ReadoutLayer", "CGReadoutLayer"]
