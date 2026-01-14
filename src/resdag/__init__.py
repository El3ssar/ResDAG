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
composition
    Model composition using pytorch_symbolic.
layers
    Neural network layers (ReservoirLayer, ReadoutLayer, etc.).
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
>>> from resdag.layers import ReservoirLayer
>>> from resdag.layers.readouts import CGReadoutLayer
>>>
>>> reservoir = ReservoirLayer(
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
>>> from resdag import ESNModel, ReservoirLayer, CGReadoutLayer
>>>
>>> inp = ps.Input((100, 3))
>>> reservoir = ReservoirLayer(200, feedback_size=3)(inp)
>>> readout = CGReadoutLayer(200, 3, name="output")(reservoir)
>>> model = ESNModel(inp, readout)
>>> model.summary()

Using premade models:

>>> from resdag import ott_esn
>>> model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

See Also
--------
ESNModel : Main model class for ESN composition.
ReservoirLayer : Core reservoir layer with recurrent dynamics.
ESNTrainer : Trainer for fitting readout layers.
"""

from . import composition, hpo, init, layers, models, training, utils

# Convenience imports for common use cases
from .composition import ESNModel

# Convenience submodule imports
from .init import graphs, input_feedback, topology
from .layers import ReservoirLayer
from .layers.readouts import CGReadoutLayer
from .models import classic_esn, headless_esn, linear_esn, ott_esn
from .training import ESNTrainer

__version__ = "0.1.3"

__all__ = [
    # Modules
    "composition",
    "hpo",
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
    # Core layers
    "ReservoirLayer",
    "CGReadoutLayer",
    # Model composition
    "ESNModel",
    # Training
    "ESNTrainer",
    # Premade models
    "classic_esn",
    "ott_esn",
    "headless_esn",
    "linear_esn",
]


def __getattr__(name: str):
    """Lazy import for optional HPO functions."""
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
