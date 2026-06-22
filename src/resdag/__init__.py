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
Easiest API — train and forecast a series in a few lines:

>>> import numpy as np
>>> from resdag import ESN
>>> series = np.cumsum(np.random.randn(2000, 3), axis=0)  # (time, features)
>>> esn = ESN(reservoir_size=300, spectral_radius=0.9).fit(series)
>>> prediction = esn.forecast(horizon=200)  # (200, 3), numpy in -> numpy out

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

from . import core, data, ensemble, init, layers, models, training, utils

# Convenience imports for common use cases
from .core import ESNModel, ReservoirFeatureExtractor, reservoir_input
from .data import TimeSeriesWindowDataset, make_dataloader
from .ensemble import CoupledEnsembleESNModel
from .ensemble.aggregators import OutliersFilteredMean
from .facade import ESN

# Convenience submodule imports
from .init import graphs, input_feedback, topology
from .layers import (
    BaseReservoirLayer,
    CGReadoutLayer,
    CholeskyReadoutLayer,
    Concatenate,
    ESNCell,
    ESNLayer,
    FeaturePartitioner,
    IncrementalRidgeReadout,
    NGCell,
    NGReservoir,
    PinvReadoutLayer,
    Power,
    ReadoutLayer,
    ReservoirCell,
    RidgeReadoutLayer,
    SelectiveDropout,
    SelectiveExponentiation,
    Standardize,
    SVDReadoutLayer,
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

# Canonical dataset generators. The ``resdag.utils.data`` submodule is also
# exposed as ``resdag.datasets`` for a discoverable, library-style entry point.
from .utils import data as datasets
from .utils.data import henon, lorenz, mackey_glass, narma, rossler, sine

# Signature stability diagnostic, surfaced at the top level for discoverability.
from .utils.states import esp_index

__version__ = "0.6.2"

__all__ = [
    # Modules
    "core",
    "data",
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
    # High-level facade
    "ESN",
    # Composition helpers
    "ESNModel",
    "ReservoirFeatureExtractor",
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
    "RidgeReadoutLayer",
    "CholeskyReadoutLayer",
    "SVDReadoutLayer",
    "PinvReadoutLayer",
    "IncrementalRidgeReadout",
    # Custom / utility layers
    "Concatenate",
    "FeaturePartitioner",
    "OutliersFilteredMean",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
    "Standardize",
    # Ensemble
    "CoupledEnsembleESNModel",
    # Training
    "ESNTrainer",
    # Streaming / SGD data pipeline
    "TimeSeriesWindowDataset",
    "make_dataloader",
    # Premade models
    "classic_esn",
    "coupled_ensemble_esn",
    "ott_esn",
    "headless_esn",
    "linear_esn",
    "power_augmented",
    # Canonical dataset generators
    "datasets",
    "lorenz",
    "rossler",
    "henon",
    "mackey_glass",
    "narma",
    "sine",
    # Diagnostics
    "esp_index",
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


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
