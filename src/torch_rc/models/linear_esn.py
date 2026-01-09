"""Linear ESN architecture with identity activation."""

from typing import Any, Dict, Optional

import pytorch_symbolic as ps

from ..composition import ESNModel
from ..layers import ReservoirLayer


def linear_esn(
    reservoir_size: int,
    input_size: int,
    reservoir_config: Optional[Dict[str, Any]] = None,
    name: str = "linear_esn",
) -> ESNModel:
    """Build an ESN model with no readout layer and a linear activation function.

    This model uses a linear activation function in the reservoir, which can be
    useful for studying linear dynamics or as a baseline for comparison with
    nonlinear reservoirs.

    Architecture:
        Input -> Reservoir(activation='identity') (output)

    Args:
        reservoir_size: Number of units in the reservoir
        input_size: Number of input features
        reservoir_config: Optional dict with ReservoirLayer parameters.
            Note: 'activation' will be forced to 'identity'
        name: Name for the model (currently unused with pytorch_symbolic)

    Returns:
        ESNModel with linear reservoir

    Example:
        >>> from torch_rc.models import linear_esn
        >>> model = linear_esn(100, 1)
        >>> linear_states = model(input_data)
    """
    # Prepare config with defaults
    res_config = reservoir_config or {}

    # Architecture-specific requirements
    res_config["feedback_size"] = input_size
    res_config["input_size"] = 0
    res_config["reservoir_size"] = res_config.get("reservoir_size", reservoir_size)
    
    # Force linear activation (architecture requirement)
    res_config["activation"] = "identity"

    # Build model - just input and reservoir with linear activation
    inp = ps.Input((100, input_size))
    reservoir = ReservoirLayer(**res_config)(inp)

    return ESNModel(inp, reservoir)
