"""
Premade ESN Architectures
=========================

This module provides pre-configured ESN model architectures that can be
used directly or customized for specific tasks.

Functions
---------
classic_esn
    Traditional ESN with input concatenation.
ott_esn
    Ott's ESN with state augmentation (squared even units).
power_augmented
    Power Augmented ESN with state augmentation (exponentiated reservoir states).
deep_esn
    Deep (hierarchical) ESN — a stack of reservoir layers whose readout sees the
    concatenation of every layer's state.
hybrid_esn
    Hybrid physics-prior ESN (Pathak et al. 2018) — a reservoir combined with a
    knowledge-based expert wired in before the reservoir and before the readout.
headless_esn
    Reservoir only (no readout) for analysis.
linear_esn
    Linear reservoir for baseline comparison.
lsm
    Spiking Liquid State Machine — leaky integrate-and-fire reservoir with a
    filtered-spike-trace readout (input-driven; not a chaos forecaster).
parallel_esn
    Parallel local-state reservoir (Pathak et al. 2018) — one shared reservoir
    advanced batched over overlapping spatial patches, for forecasting large
    spatiotemporally chaotic systems (PDEs / weather / lattices).
coupled_ensemble_esn
    Build a `resdag.ensemble.CoupledEnsembleESNModel` of N coupled
    ESN sub-models sharing an averaged feedback signal.

Classes
-------
KnowledgeModel
    `torch.nn.Module` adapter wrapping a knowledge-based expert
    `f(u) -> u_next` for use inside `hybrid_esn`.
ParallelESNModel
    The model class `parallel_esn` builds — a shared reservoir plus a
    weight-tied readout with patch windowing / stitching and a spatial-
    autoregressive `resdag.models.ParallelESNModel.forecast`.

Each architecture accepts individual parameters for full customization
while providing sensible defaults for quick experimentation.

Examples
--------
Quick start with Ott's ESN:

>>> from resdag.models import ott_esn
>>> model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
>>> predictions = model.forecast(warmup_data, horizon=100)

Classic ESN with custom topology:

>>> from resdag.models import classic_esn
>>> from resdag.init.topology import get_topology
>>>
>>> model = classic_esn(
...     reservoir_size=500,
...     feedback_size=3,
...     output_size=3,
...     topology=get_topology("watts_strogatz", k=4, p=0.3),
...     spectral_radius=0.95,
... )

See Also
--------
resdag.core.ESNModel : Base ESN model class.
resdag.layers.ESNLayer : ESN layer used by these models.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from .classic_esn import classic_esn
from .coupled_ensemble_esn import coupled_ensemble_esn
from .deep_esn import deep_esn
from .es2n import es2n
from .eusn import eusn
from .gated_esn import gated_esn
from .headless_esn import headless_esn
from .hybrid_esn import KnowledgeModel, hybrid_esn
from .linear_esn import linear_esn
from .lsm import lsm
from .ott_esn import ott_esn
from .parallel_esn import ParallelESNModel, parallel_esn
from .power_augmented import power_augmented
from .resesn import resesn

__all__ = [
    "KnowledgeModel",
    "classic_esn",
    "coupled_ensemble_esn",
    "deep_esn",
    "headless_esn",
    "hybrid_esn",
    "linear_esn",
    "ott_esn",
    "power_augmented",
    "es2n",
    "resesn",
    "gated_esn",
    "eusn",
    "lsm",
    "parallel_esn",
    "ParallelESNModel",
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
