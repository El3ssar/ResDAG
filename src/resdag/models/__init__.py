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
coupled_ensemble_esn
    Build a :class:`~resdag.ensemble.CoupledEnsembleESNModel` of N coupled
    ESN sub-models sharing an averaged feedback signal.

Classes
-------
KnowledgeModel
    :class:`~torch.nn.Module` adapter wrapping a knowledge-based expert
    ``f(u) -> u_next`` for use inside :func:`hybrid_esn`.

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
from .headless_esn import headless_esn
from .hybrid_esn import KnowledgeModel, hybrid_esn
from .linear_esn import linear_esn
from .ott_esn import ott_esn
from .power_augmented import power_augmented

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
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
