"""Utility functions for torch_rc."""

from .functional_forecast import functional_esn_forecast
from .general import create_rng

__all__ = ["create_rng", "functional_esn_forecast"]
