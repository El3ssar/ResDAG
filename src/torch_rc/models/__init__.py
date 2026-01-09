"""Premade model architectures for torch_rc.

This module contains pre-configured ESN architectures that can be used
directly or customized for specific tasks.

Available architectures:
- classic_esn: Traditional ESN with input concatenation
- ott_esn: Ott's ESN with state augmentation (squared even units)
- headless_esn: Reservoir only (no readout) for analysis
- linear_esn: Linear reservoir for baseline comparison

Each architecture accepts config dicts for full customization while
providing sensible defaults for quick experimentation.
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
