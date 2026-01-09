"""Ott's ESN architecture with state augmentation."""

from typing import Any, Dict, Optional

import pytorch_symbolic as ps

from ..composition import ESNModel
from ..layers import ReservoirLayer
from ..layers.custom import Concatenate, SelectiveExponentiation
from ..layers.readouts import CGReadoutLayer


def ott_esn(
    reservoir_size: int,
    input_size: int,
    output_size: int,
    reservoir_config: Optional[Dict[str, Any]] = None,
    readout_config: Optional[Dict[str, Any]] = None,
    name: str = "ott_esn",
) -> ESNModel:
    """Build Ott's ESN model with state augmentation.

    This model follows the architecture proposed by Edward Ott, which augments
    reservoir states by squaring even-indexed units and concatenating with input.
    This augmentation helps capture higher-order dynamics in the reservoir states.

    Architecture:
        Input -> Reservoir -> SelectiveExponentiation -> Concatenate(Input, Augmented) -> Readout

    Args:
        reservoir_size: Number of units in the reservoir
        input_size: Number of input features
        output_size: Number of output features
        reservoir_config: Optional dict with ReservoirLayer parameters
        readout_config: Optional dict with CGReadoutLayer parameters
        name: Name for the model (currently unused with pytorch_symbolic)

    Returns:
        ESNModel ready for training and inference

    References:
        E. Ott et al., "Model-Free Prediction of Large Spatiotemporally Chaotic
        Systems from Data: A Reservoir Computing Approach," Phys. Rev. Lett., 2018.

    Example:
        >>> from torch_rc.models import ott_esn
        >>> model = ott_esn(100, 1, 1)
        >>> predictions = model.forecast(warmup_data, forecast_steps=100)
    """
    # Prepare configs with defaults
    res_config = reservoir_config or {}
    read_config = readout_config or {}

    # Architecture-specific requirements
    res_config["feedback_size"] = input_size
    res_config["input_size"] = 0
    res_config["reservoir_size"] = res_config.get("reservoir_size", reservoir_size)

    # Readout sees input + augmented reservoir
    read_config["in_features"] = input_size + reservoir_size
    read_config["out_features"] = output_size

    # Build model
    inp = ps.Input((100, input_size))
    reservoir = ReservoirLayer(**res_config)(inp)

    # Augment reservoir states (square even-indexed units)
    augmented = SelectiveExponentiation(index=0, exponent=2.0)(reservoir)

    # Concatenate input with augmented reservoir
    concat = Concatenate()(inp, augmented)

    readout = CGReadoutLayer(**read_config)(concat)

    return ESNModel(inp, readout)
