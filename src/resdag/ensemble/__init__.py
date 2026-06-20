"""
Ensemble Reservoir Computing Models
=====================================

This module provides ensemble wrappers for groups of independently-trained
:class:`~resdag.core.ESNModel` instances.

Classes
-------
CoupledEnsembleESNModel
    N independent ESN sub-models whose autoregressive forecasts are coupled
    through a shared averaged feedback signal at every timestep.

See Also
--------
resdag.models.coupled_ensemble_esn : Factory function for quick construction.
"""

from . import aggregators
from .aggregators import OutliersFilteredMean
from .coupled import CoupledEnsembleESNModel

__all__ = ["CoupledEnsembleESNModel", "OutliersFilteredMean", "aggregators"]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
