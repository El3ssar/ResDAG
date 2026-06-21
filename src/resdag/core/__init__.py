"""
Core Model
==========

This module provides the main model class and input helpers for building
ESN architectures using the ``pytorch_symbolic`` library.

Classes
-------
ESNModel
    Extended SymbolicModel with ESN-specific methods for forecasting
    and reservoir state management.
ReservoirFeatureExtractor
    ``nn.Sequential``-friendly adapter that packages reservoir layers as a
    drop-in feature extractor for ordinary ``torch.nn`` pipelines.
Input
    Alias for ``pytorch_symbolic.Input`` for defining model inputs.

Functions
---------
reservoir_input
    Convenience constructor for a per-feature symbolic input tensor with
    a placeholder time dimension.  Preferred over hand-crafting
    ``ps.Input((T, F))`` since the time dimension is purely a tracing
    hint inside reservoir models.

Examples
--------
Building a simple ESN:

>>> from resdag.core import ESNModel, reservoir_input
>>> from resdag.layers import ESNLayer
>>> from resdag.layers.readouts import CGReadoutLayer
>>>
>>> inp = reservoir_input(3)             # equivalent to ps.Input((1, 3))
>>> reservoir = ESNLayer(200, feedback_size=3)(inp)
>>> readout = CGReadoutLayer(200, 3)(reservoir)
>>> model = ESNModel(inp, readout)

Multi-input model:

>>> feedback = reservoir_input(3)
>>> driver = reservoir_input(5)
>>> reservoir = ESNLayer(200, feedback_size=3, input_size=5)(feedback, driver)
>>> readout = CGReadoutLayer(200, 3)(reservoir)
>>> model = ESNModel([feedback, driver], readout)

See Also
--------
resdag.models : Premade ESN architectures.
resdag.training.ESNTrainer : Trainer for fitting readouts.
"""

import pytorch_symbolic as ps
import torch

from .feature_extractor import ReservoirFeatureExtractor
from .model import ESNModel, Input


def reservoir_input(feature_size: int, dtype: torch.dtype | None = None):
    """
    Build a symbolic input tensor for a reservoir model.

    Constructs an :func:`pytorch_symbolic.Input` of shape ``(1, feature_size)``
    where the first axis is a placeholder for the time dimension (any
    sequence length is accepted at call time).

    Parameters
    ----------
    feature_size : int
        Number of features per timestep.
    dtype : torch.dtype, optional
        Tensor dtype. Defaults to ``torch.get_default_dtype()``.

    Returns
    -------
    pytorch_symbolic.SymbolicTensor
        Placeholder input with shape ``(1, feature_size)``.

    Examples
    --------
    >>> inp = reservoir_input(3)
    >>> inp.shape
    torch.Size([1, 1, 3])
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    return ps.Input((1, feature_size), dtype=dtype)


# ``ps`` (the re-exported ``pytorch_symbolic`` module) and ``torch`` remain
# importable for backward compatibility but are intentionally left out of the
# public surface — see ``__dir__`` below.
__all__ = ["ESNModel", "Input", "ReservoirFeatureExtractor", "reservoir_input"]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
