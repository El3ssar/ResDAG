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
OutliersFilteredMean
    Computes mean with outlier filtering.
SelectiveDropout
    Per-feature dropout with selectivity control.
SelectiveExponentiation
    Per-feature exponentiation transformation.

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
from .outliers_filtered_mean import OutliersFilteredMean
from .selective_dropout import SelectiveDropout
from .selective_exponentiation import SelectiveExponentiation

__all__ = [
    "Concatenate",
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "SelectiveDropout",
    "SelectiveExponentiation",
]
