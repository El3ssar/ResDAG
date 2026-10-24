"""
Transform Layers
================

Specialized transformation layers for advanced ESN architectures,
including state augmentation, feature manipulation, and regularization.

Classes
-------
Concatenate
    Concatenates multiple inputs along the feature dimension.
ExtendedSquare
    Appends the elementwise square of every feature (`[x, x**2]`).
FeaturePartitioner
    Partitions input features into overlapping groups.
NLAT2
    T₂ nonlinear algebraic transform (`x[i-1] * x[i-2]` at even features).
NLAT3
    T₃ nonlinear algebraic transform (`x[i-1] * x[i+1]` at interior even
    features).
Pad
    Appends a constant column to the feature dimension (`[x, padding]`).
PartialSquare
    Squares a leading fraction (or count) of features in place.
Power
    Per-feature exponentiation by a fixed power.
SelectiveDropout
    Per-feature dropout with selectivity control.
SelectiveExponentiation
    Per-feature exponentiation transformation (squares even-indexed units;
    `index=0, exponent=2` is the T₁/NLAT1 transform).
Standardize
    Per-feature z-score standardization with buffer-stored mean/std,
    `fit`, and an exact `inverse`.

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
from .extended_square import ExtendedSquare
from .feature_partitioner import FeaturePartitioner
from .nlat import NLAT2, NLAT3
from .pad import Pad
from .partial_square import PartialSquare
from .power import Power
from .selective_dropout import SelectiveDropout
from .selective_exponentiation import SelectiveExponentiation
from .standardize import Standardize

__all__ = [
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
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
