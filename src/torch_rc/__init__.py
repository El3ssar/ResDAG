"""torch_rc: PyTorch-native reservoir computing library.

A modular, GPU-accelerated library for Echo State Networks (ESN) and
reservoir computing in PyTorch.

Key Features:
- Pure PyTorch nn.Module components
- Graph-based topology initialization
- Stateful reservoir layers
- GPU acceleration throughout
- Modular composition for arbitrary DAGs

Basic Usage:
>>> from torch_rc.layers import ReservoirLayer, ReadoutLayer
>>> from torch_rc.init.topology import TopologyRegistry
>>>
>>> # Create reservoir with graph topology
>>> reservoir = ReservoirLayer(
...     reservoir_size=100,
...     feedback_size=10,
...     topology="erdos_renyi"
... )
>>> readout = ReadoutLayer(in_features=100, out_features=1, name="output")
>>>
>>> # Use as standard PyTorch modules
>>> x = torch.randn(32, 50, 10)  # (batch, time, features)
>>> h = reservoir(x)
>>> y = readout(h)
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

__version__ = "0.1.0"

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
