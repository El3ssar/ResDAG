"""
Transform Layers
================

Specialized transformation layers for advanced ESN architectures,
including state augmentation, feature manipulation, and regularization.

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
    Per-feature exponentiation transformation (squares even-indexed units).

Examples
--------
>>> from resdag.layers.transforms import Concatenate, SelectiveExponentiation
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


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
