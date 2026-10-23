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
metrics
    Forecast-accuracy and Lyapunov-time metrics.
hpo
    Hyperparameter optimization with Optuna.
utils
    Data loading and utility functions.

Examples
--------
Easiest API — train and forecast a series in a few lines:

>>> import torch
>>> from resdag import ESN
>>> series = torch.randn(2000, 3).cumsum(0)  # (time, features)
>>> esn = ESN(reservoir_size=300, spectral_radius=0.9).fit(series)
>>> prediction = esn.forecast(horizon=200)  # (200, 3), original-unit Torch tensor

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

from . import core, data, datasets, ensemble, init, layers, metrics, models, training, utils
from .artifacts import ArtifactError, ComponentCodec, load, register_component

# Convenience imports for common use cases
from .core import (
    ESNModel,
    Feedback,
    Input,
    Model,
    ReservoirFeatureExtractor,
    ReservoirModelModule,
    ReservoirStates,
    reservoir_input,
)
from .data import TimeSeriesWindowDataset, make_dataloader

# Canonical dataset generators. ``resdag.datasets`` (imported above) is the
# canonical public home for these generators, so ``import resdag.datasets`` and
# ``from resdag.datasets import lorenz`` both work as a discoverable,
# library-style entry point, alongside attribute access
# (``resdag.datasets.lorenz``).
from .datasets import (
    henon,
    kuramoto_sivashinsky,
    lorenz,
    mackey_glass,
    narma,
    rossler,
    sine,
)
from .ensemble import CoupledEnsembleESNModel
from .ensemble.aggregators import OutliersFilteredMean
from .facade import ESN

# Convenience submodule imports
from .init import graphs, input_feedback, topology
from .layers import (
    NLAT2,
    NLAT3,
    AutoReadoutLayer,
    BaseReservoirLayer,
    CGReadoutLayer,
    CholeskyReadoutLayer,
    Concatenate,
    ContinuousReservoirCell,
    CoRNNCell,
    CoRNNLayer,
    ES2NCell,
    ES2NLayer,
    ESNCell,
    ESNLayer,
    EuSNCell,
    EuSNLayer,
    ExtendedSquare,
    FeaturePartitioner,
    FISTAReadout,
    FISTAReadoutLayer,
    GatedReservoir,
    GatedReservoirCell,
    GatedReservoirLayer,
    IncrementalRidgeReadout,
    IncrementalRidgeReadoutLayer,
    LegendreDelayCell,
    LegendreDelayLayer,
    LIFCell,
    LIFReservoir,
    LIFReservoirLayer,
    NGCell,
    NGReservoir,
    NGReservoirLayer,
    Pad,
    PartialSquare,
    PinvReadoutLayer,
    Power,
    ReadoutLayer,
    ReadoutSolveReport,
    ReservoirCell,
    ResESNCell,
    ResESNLayer,
    RidgeReadoutLayer,
    RLSReadout,
    SelectiveDropout,
    SelectiveExponentiation,
    SpikingReservoirCell,
    Standardize,
    SVDReadoutLayer,
    TorchRecurrentCell,
    TorchRecurrentLayer,
)
from .models import (
    KnowledgeModel,
    ParallelESNModel,
    classic_esn,
    coupled_ensemble_esn,
    deep_esn,
    es2n,
    eusn,
    gated_esn,
    headless_esn,
    hybrid_esn,
    linear_esn,
    lsm,
    ott_esn,
    parallel_esn,
    power_augmented,
    resesn,
)
from .training import (
    CalibrationReport,
    ESNTrainer,
    calibrate_flow_control,
    calibrate_intrinsic_plasticity,
)
from .utils.seeds import SeedStreams

# Signature stability diagnostic, surfaced at the top level for discoverability.
from .utils.states import esp_index

__version__ = "4.0.0"

__all__ = [
    "ArtifactError",
    "ComponentCodec",
    "load",
    "register_component",
    # Modules
    "core",
    "data",
    "ensemble",
    "hpo",  # resolved lazily via __getattr__ (see below)
    "init",
    "layers",
    "metrics",
    "models",
    "training",
    "utils",
    "SeedStreams",
    "__version__",
    # Convenience submodules
    "graphs",
    "topology",
    "input_feedback",
    # High-level facade
    "ESN",
    # Composition helpers
    "Input",
    "Model",
    "Feedback",
    "ESNModel",
    "ReservoirFeatureExtractor",
    "ReservoirModelModule",
    "ReservoirStates",
    "reservoir_input",
    # Core reservoir layers / cells
    "BaseReservoirLayer",
    "CoRNNCell",
    "CoRNNLayer",
    "LegendreDelayCell",
    "LegendreDelayLayer",
    "TorchRecurrentCell",
    "TorchRecurrentLayer",
    "ESNCell",
    "ES2NCell",
    "ResESNCell",
    "GatedReservoirCell",
    "ContinuousReservoirCell",
    "EuSNCell",
    "ESNLayer",
    "NGCell",
    "NGReservoirLayer",
    "NGReservoir",
    "ES2NLayer",
    "ResESNLayer",
    "GatedReservoirLayer",
    "GatedReservoir",
    "EuSNLayer",
    "SpikingReservoirCell",
    "LIFCell",
    "LIFReservoirLayer",
    "LIFReservoir",
    "ReservoirCell",
    # Readouts
    "AutoReadoutLayer",
    "ReadoutSolveReport",
    "CGReadoutLayer",
    "ReadoutLayer",
    "RidgeReadoutLayer",
    "CholeskyReadoutLayer",
    "SVDReadoutLayer",
    "PinvReadoutLayer",
    "IncrementalRidgeReadoutLayer",
    "IncrementalRidgeReadout",
    "RLSReadout",
    "FISTAReadoutLayer",
    "FISTAReadout",
    # Custom / utility layers
    "Concatenate",
    "ExtendedSquare",
    "FeaturePartitioner",
    "NLAT2",
    "NLAT3",
    "OutliersFilteredMean",
    "Pad",
    "PartialSquare",
    "Power",
    "SelectiveDropout",
    "SelectiveExponentiation",
    "Standardize",
    # Ensemble
    "CoupledEnsembleESNModel",
    # Training
    "ESNTrainer",
    "CalibrationReport",
    "calibrate_flow_control",
    "calibrate_intrinsic_plasticity",
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
    "deep_esn",
    "hybrid_esn",
    "KnowledgeModel",
    "es2n",
    "resesn",
    "gated_esn",
    "eusn",
    "lsm",
    "parallel_esn",
    "ParallelESNModel",
    # Canonical dataset generators
    "datasets",
    "lorenz",
    "rossler",
    "henon",
    "mackey_glass",
    "narma",
    "sine",
    "kuramoto_sivashinsky",
    # Diagnostics
    "esp_index",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the optional :mod:`resdag.hpo` submodule (:pep:`562`).

    ``hpo`` is imported only on first access, so a plain ``import resdag`` does
    not eagerly pull in scipy/optuna.  The HPO convenience names (``run_hpo``,
    ``LOSSES``, ``get_study_summary``) likewise resolve through
    :mod:`resdag.hpo` on demand.
    """
    if name == "hpo":
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
