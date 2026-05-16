"""
Custom Layers
=============

This subpackage contains specialized layers for advanced ESN architectures,
including utility layers for feature manipulation and regularization.

Classes
-------
Concatenate
    Concatenates multiple inputs along the feature dimension.
FeaturePartitioner
    Partitions input features into overlapping groups.
Power
    Per-feature exponentiation by a fixed power.
SelectiveDropout
    Per-feature dropout with selectivity control.
SelectiveExponentiation
    Per-feature exponentiation transformation.

Notes
-----
``OutliersFilteredMean`` moved to :mod:`resdag.ensemble.aggregators` in
0.4.0.  Import it from the new location:

>>> from resdag.ensemble.aggregators import OutliersFilteredMean

It is also still re-exported at the top level as
``resdag.OutliersFilteredMean``.

Examples
--------
>>> from resdag.layers.custom import Concatenate
>>> import torch
>>>
>>> concat = Concatenate()
>>> x1 = torch.randn(4, 100, 50)
>>> x2 = torch.randn(4, 100, 50)
>>> combined = concat(x1, x2)  # (4, 100, 100)
"""

from .concatenate import Concatenate
from .feature_partitioner import FeaturePartitioner
from .power import Power
from .selective_dropout import SelectiveDropout
from .selective_exponentiation import SelectiveExponentiation

__all__ = [
    "Concatenate",
    "FeaturePartitioner",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
]
