"""Classic ESN architecture with input concatenation."""

from typing import Any, Dict, Optional

import pytorch_symbolic as ps

from ..composition import ESNModel
from ..layers import ReservoirLayer
from ..layers.custom import Concatenate
from ..layers.readouts import CGReadoutLayer


def classic_esn(
    reservoir_size: int,
    input_size: int,
    output_size: int,
    reservoir_config: Optional[Dict[str, Any]] = None,
    readout_config: Optional[Dict[str, Any]] = None,
    name: str = "classic_esn",
) -> ESNModel:
    """Build a classic Echo State Network (ESN) model.

    This architecture concatenates the input with the reservoir output before
    passing to the readout layer, following the traditional ESN design.

    Architecture:
        Input -> Reservoir -> Concatenate(Input, Reservoir) -> Readout

    The readout sees both the raw input and the reservoir's nonlinear
    transformation, which can improve performance on many tasks.

    Args:
        reservoir_size: Number of units in the reservoir
        input_size: Number of input features
        output_size: Number of output features
        reservoir_config: Optional dict with ReservoirLayer parameters.
            Can include: topology, spectral_radius, leak_rate, input_scaling,
            feedback_scaling, bias, trainable, etc.
            Note: feedback_size and input_size are set automatically.
        readout_config: Optional dict with CGReadoutLayer parameters.
            Can include: max_iter, tol, bias, trainable, etc.
            Note: in_features is set automatically to reservoir_size + input_size.
        name: Name for the model (currently unused with pytorch_symbolic)

    Returns:
        ESNModel ready for training and inference

    Example:
        >>> from torch_rc.models import classic_esn
        >>> import torch
        >>>
        >>> # Simple usage with defaults
        >>> model = classic_esn(100, 1, 1)
        >>>
        >>> # With custom configuration
        >>> model = classic_esn(
        ...     reservoir_size=100,
        ...     input_size=1,
        ...     output_size=1,
        ...     reservoir_config={
        ...         'topology': 'erdos_renyi',
        ...         'spectral_radius': 0.9,
        ...         'leak_rate': 0.1,
        ...         'input_scaling': 0.5,
        ...     },
        ...     readout_config={
        ...         'max_iter': 200,
        ...         'tol': 1e-6,
        ...     }
        ... )
        >>>
        >>> # Forward pass (much simpler now!)
        >>> x = torch.randn(4, 50, 1)  # (batch, time, features)
        >>> y = model(x)  # Direct call, no dict needed!
    """
    # Prepare configs with defaults
    res_config = reservoir_config or {}
    read_config = readout_config or {}

    # Architecture-specific requirements
    res_config["feedback_size"] = input_size
    res_config["input_size"] = 0  # No driving input in classic ESN
    res_config["reservoir_size"] = res_config.get("reservoir_size", reservoir_size)

    # Readout sees concatenated input + reservoir
    read_config["in_features"] = reservoir_size + input_size
    read_config["out_features"] = output_size

    # Build model with pytorch_symbolic (much simpler!)
    inp = ps.Input((100, input_size))  # Use typical seq_len for tracing
    reservoir = ReservoirLayer(**res_config)(inp)
    concat = Concatenate()(inp, reservoir)
    readout = CGReadoutLayer(**read_config)(concat)

    return ESNModel(inp, readout)
