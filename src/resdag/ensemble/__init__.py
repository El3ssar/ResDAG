"""
Ensemble Reservoir Computing Models
=====================================

This module provides ensemble wrappers for groups of independently-trained
:class:`~resdag.composition.ESNModel` instances.

Classes
-------
CoupledEnsembleESNModel
    N independent ESN sub-models whose autoregressive forecasts are coupled
    through a shared averaged feedback signal at every timestep.

See Also
--------
resdag.models.coupled_ensemble_esn : Factory function for quick construction.
"""

from .coupled import CoupledEnsembleESNModel

__all__ = ["CoupledEnsembleESNModel"]
