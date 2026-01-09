"""Headless ESN architecture without readout layer."""

from typing import Any, Dict, Optional

import pytorch_symbolic as ps

from ..composition import ESNModel
from ..layers import ReservoirLayer


def headless_esn(
    reservoir_size: int,
    input_size: int,
    reservoir_config: Optional[Dict[str, Any]] = None,
    name: str = "headless_esn",
) -> ESNModel:
    """Build an ESN model with no readout layer.

    This model can be used to study the dynamics of the reservoir by applying
    different transformations to the reservoir states without a readout layer.
    Useful for analyzing reservoir dynamics, state space properties, and
    feature extraction.

    Architecture:
        Input -> Reservoir (output)

    The reservoir is not connected to a readout layer, allowing direct
    access to reservoir states for analysis or custom processing.

    Args:
        reservoir_size: Number of units in the reservoir
        input_size: Number of input features
        reservoir_config: Optional dict with ReservoirLayer parameters
        name: Name for the model (currently unused with pytorch_symbolic)

    Returns:
        ESNModel with reservoir output only

    Example:
        >>> from torch_rc.models import headless_esn
        >>> model = headless_esn(100, 1)
        >>> reservoir_states = model(input_data)  # Direct reservoir output
    """
    # Prepare config with defaults
    res_config = reservoir_config or {}

    # Architecture-specific requirements
    res_config["feedback_size"] = input_size
    res_config["input_size"] = 0
    res_config["reservoir_size"] = res_config.get("reservoir_size", reservoir_size)

    # Build model - just input and reservoir
    inp = ps.Input((100, input_size))
    reservoir = ReservoirLayer(**res_config)(inp)

    return ESNModel(inp, reservoir)
