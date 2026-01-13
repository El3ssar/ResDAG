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
headless_esn
    Reservoir only (no readout) for analysis.
linear_esn
    Linear reservoir for baseline comparison.

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
resdag.composition.ESNModel : Base ESN model class.
resdag.layers.ReservoirLayer : Reservoir layer used by these models.
resdag.training.ESNTrainer : Trainer for fitting readout layers.
"""

from .classic_esn import classic_esn
from .headless_esn import headless_esn
from .linear_esn import linear_esn
from .ott_esn import ott_esn

__all__ = [
    "classic_esn",
    "ott_esn",
    "headless_esn",
    "linear_esn",
]
