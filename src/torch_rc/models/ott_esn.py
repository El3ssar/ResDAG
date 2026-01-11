"""Ott's ESN architecture with state augmentation."""

from typing import Any

import pytorch_symbolic as ps

from ..composition import ESNModel
from ..init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from ..layers import ReservoirLayer
from ..layers.custom import Concatenate, SelectiveExponentiation
from ..layers.readouts import CGReadoutLayer


def ott_esn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    # Readout params
    readout_alpha: float = 1e-6,
    readout_bias: bool = True,
    readout_name: str = "output",
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    """Build Ott's ESN model with state augmentation.

    This model follows the architecture proposed by Edward Ott, which augments
    reservoir states by squaring even-indexed units and concatenating with input.
    This augmentation helps capture higher-order dynamics in the reservoir states.

    Architecture:
        Input -> Reservoir -> SelectiveExponentiation -> Concatenate(Input, Augmented) -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features.
    output_size : int
        Number of output features.
    topology : TopologySpec, optional
        Topology for recurrent weights. Accepts:
        - str: Registry name (e.g., "erdos_renyi")
        - tuple: (name, params) like ("watts_strogatz", {"k": 6, "p": 0.1})
        - GraphTopology: Pre-configured object
    spectral_radius : float, default=0.9
        Desired spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate (1.0 = no leak).
    feedback_initializer : InitializerSpec, optional
        Initializer for feedback weights.
    activation : str, default="tanh"
        Activation function ("tanh", "relu", "sigmoid", "identity").
    bias : bool, default=True
        Whether to use bias in the reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.
    readout_alpha : float, default=1e-6
        Ridge regression regularization for readout.
    readout_bias : bool, default=True
        Whether to use bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (used in training targets).
    **reservoir_kwargs
        Additional keyword arguments passed to ReservoirLayer.

    Returns
    -------
    ESNModel
        Configured Ott ESN model ready for training and inference.

    References
    ----------
    E. Ott et al., "Model-Free Prediction of Large Spatiotemporally Chaotic
    Systems from Data: A Reservoir Computing Approach," Phys. Rev. Lett., 2018.

    Examples
    --------
    >>> from torch_rc.models import ott_esn
    >>> model = ott_esn(100, 1, 1)
    >>> predictions = model.forecast(warmup_data, horizon=100)
    """
    # Resolve topology and initializer specs
    resolved_topology = resolve_topology(topology)
    resolved_feedback_init = resolve_initializer(feedback_initializer)

    # Build model
    inp = ps.Input((100, feedback_size))

    reservoir = ReservoirLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=0,
        topology=resolved_topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=resolved_feedback_init,
        activation=activation,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(inp)

    # Augment reservoir states (square even-indexed units)
    augmented = SelectiveExponentiation(index=0, exponent=2.0)(reservoir)

    # Concatenate input with augmented reservoir
    concat = Concatenate()(inp, augmented)

    readout = CGReadoutLayer(
        in_features=feedback_size + reservoir_size,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,
        name=readout_name,
    )(concat)

    return ESNModel(inp, readout)
