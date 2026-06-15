"""
resdag - PyTorch Reservoir Computing Library
===============================================

A modular, GPU-accelerated library for Echo State Networks (ESN) and
reservoir computing in PyTorch.

Features
--------
- Pure PyTorch ``nn.Module`` components
- Graph-based topology initialization
- Stateful reservoir layers with Echo State Property
- GPU acceleration throughout
- Modular composition for arbitrary DAGs
- Hyperparameter optimization integration

Modules
-------
core
    Core model class (ESNModel) and input helpers.
layers
    Neural network layers (ESNLayer, ReadoutLayer, etc.).
init
    Weight initialization (topologies, input/feedback).
training
    Training utilities (ESNTrainer).
models
    Premade ESN architectures.
hpo
    Hyperparameter optimization with Optuna.
utils
    Data loading and utility functions.

Examples
--------
Basic reservoir usage:

>>> import torch
>>> from resdag.layers import ESNLayer
>>> from resdag.layers.readouts import CGReadoutLayer
>>>
>>> reservoir = ESNLayer(
...     reservoir_size=100,
...     feedback_size=10,
...     topology="erdos_renyi"
... )
>>> x = torch.randn(32, 50, 10)  # (batch, time, features)
>>> h = reservoir(x)
>>> print(h.shape)
torch.Size([32, 50, 100])

Building a complete ESN model:

>>> import pytorch_symbolic as ps
>>> from resdag import ESNModel, ESNLayer, CGReadoutLayer
>>>
>>> inp = ps.Input((100, 3))
>>> reservoir = ESNLayer(200, feedback_size=3)(inp)
>>> readout = CGReadoutLayer(200, 3, name="output")(reservoir)
>>> model = ESNModel(inp, readout)
>>> model.summary()

Using premade models:

>>> from resdag import ott_esn
>>> model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

See Also
--------
ESNModel : Main model class for reservoir computing.
ESNLayer : Core reservoir layer with recurrent dynamics.
ESNTrainer : Trainer for fitting readout layers.
"""

import importlib
from typing import Any

from . import core, ensemble, init, layers, models, training, utils

# Convenience imports for common use cases
from .core import ESNModel, reservoir_input
from .ensemble import CoupledEnsembleESNModel
from .ensemble.aggregators import OutliersFilteredMean

# Convenience submodule imports
from .init import graphs, input_feedback, topology
from .layers import (
    BaseReservoirLayer,
    CGReadoutLayer,
    Concatenate,
    ESNCell,
    ESNLayer,
    FeaturePartitioner,
    NGCell,
    NGReservoir,
    Power,
    ReadoutLayer,
    ReservoirCell,
    SelectiveDropout,
    SelectiveExponentiation,
)
from .models import (
    classic_esn,
    coupled_ensemble_esn,
    headless_esn,
    linear_esn,
    ott_esn,
    power_augmented,
)
from .training import ESNTrainer

__version__ = "0.6.2"

__all__ = [
    # Modules
    "core",
    "ensemble",
    "hpo",  # resolved lazily via __getattr__ (see below)
    "init",
    "layers",
    "models",
    "training",
    "utils",
    "__version__",
    # Convenience submodules
    "graphs",
    "topology",
    "input_feedback",
    # Composition helpers
    "ESNModel",
    "reservoir_input",
    # Core reservoir layers / cells
    "BaseReservoirLayer",
    "ESNCell",
    "ESNLayer",
    "NGCell",
    "NGReservoir",
    "ReservoirCell",
    # Readouts
    "CGReadoutLayer",
    "ReadoutLayer",
    # Custom / utility layers
    "Concatenate",
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
    # Ensemble
    "CoupledEnsembleESNModel",
    # Training
    "ESNTrainer",
    # Premade models
    "classic_esn",
    "coupled_ensemble_esn",
    "ott_esn",
    "headless_esn",
    "linear_esn",
    "power_augmented",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional and deprecated submodules (:pep:`562`).

    ``hpo`` and the deprecated ``composition`` shim are imported only on first
    access, so a plain ``import resdag`` stays warning-free and does not eagerly
    pull in scipy/optuna.  The HPO convenience names (``run_hpo``, ``LOSSES``,
    ``get_study_summary``) likewise resolve through :mod:`resdag.hpo` on demand.
    Accessing ``resdag.composition`` triggers its :class:`DeprecationWarning`.
    """
    if name in ("hpo", "composition"):
        return importlib.import_module(f".{name}", __name__)
    if name == "run_hpo":
        from .hpo import run_hpo

        return run_hpo
    if name == "LOSSES":
        from .hpo import LOSSES

        return LOSSES
    if name == "get_study_summary":
        from .hpo import get_study_summary

        return get_study_summary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
