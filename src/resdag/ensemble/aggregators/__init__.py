"""
Ensemble Aggregators
=====================

This module hosts ``nn.Module`` layers that combine N per-sub-model outputs
into a single aggregated tensor.  Aggregators expect a 4-D input of shape
``(samples, batch, timesteps, features)`` (or a list of length ``samples``
each of shape ``(batch, timesteps, features)``) and return a 3-D tensor of
shape ``(batch, timesteps, features)``.

They plug into :class:`~resdag.ensemble.CoupledEnsembleESNModel` via the
``aggregator`` argument.

Classes
-------
OutliersFilteredMean
    Mean across the samples dimension after dropping outliers (Z-score or
    IQR).

See Also
--------
resdag.ensemble.CoupledEnsembleESNModel : Coupled ensemble that consumes
    these aggregators.
"""

from .outliers_filtered_mean import OutliersFilteredMean

__all__ = ["OutliersFilteredMean"]
