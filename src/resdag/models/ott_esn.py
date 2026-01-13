"""
Ott's ESN Architecture
======================

This module provides :func:`ott_esn`, which builds an ESN model following
the architecture proposed by Edward Ott for predicting chaotic systems.

The key innovation is state augmentation: reservoir states are transformed
by squaring even-indexed units, which helps capture higher-order dynamics.

References
----------
E. Ott et al., "Model-Free Prediction of Large Spatiotemporally Chaotic
Systems from Data: A Reservoir Computing Approach," Phys. Rev. Lett., 2018.

See Also
--------
classic_esn : Traditional ESN architecture.
headless_esn : Reservoir-only model for analysis.
"""

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
    """
    Build Ott's ESN model with state augmentation.

    This model follows the architecture proposed by Edward Ott, which augments
    reservoir states by squaring even-indexed units and concatenating with input.
    This augmentation helps capture higher-order dynamics in chaotic systems.

    Architecture::

        Input -> Reservoir -> SelectiveExponentiation -> Concatenate(Input, Augmented) -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Dimension of feedback signal (input features).
    output_size : int
        Dimension of output signal.
    topology : str, tuple, or TopologyInitializer, optional
        Topology for recurrent weights. Accepts:

        - str: Registry name (e.g., ``"erdos_renyi"``)
        - tuple: ``(name, params)`` like ``("watts_strogatz", {"k": 6, "p": 0.1})``
        - TopologyInitializer: Pre-configured object

    spectral_radius : float, default=0.9
        Target spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate. 1.0 = no leaking (standard ESN).
    feedback_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for feedback weights. Same format as ``topology``.
    activation : {'tanh', 'relu', 'sigmoid', 'identity'}, default='tanh'
        Activation function for reservoir neurons.
    bias : bool, default=True
        Whether to include bias in reservoir.
    trainable : bool, default=False
        If True, reservoir weights are trainable via backpropagation.
    readout_alpha : float, default=1e-6
        L2 regularization strength for ridge regression in readout.
    readout_bias : bool, default=True
        Whether to include bias in readout layer.
    readout_name : str, default='output'
        Name for the readout layer. Used as target key in training.
    **reservoir_kwargs
        Additional keyword arguments passed to :class:`ReservoirLayer`.

    Returns
    -------
    ESNModel
        Configured Ott ESN model ready for training and inference.

    Examples
    --------
    Basic usage:

    >>> from resdag.models import ott_esn
    >>> model = ott_esn(
    ...     reservoir_size=500,
    ...     feedback_size=3,
    ...     output_size=3,
    ... )
    >>> model.summary()

    With custom topology:

    >>> from resdag.init.topology import get_topology
    >>> model = ott_esn(
    ...     reservoir_size=500,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     topology=get_topology("watts_strogatz", k=4, p=0.3),
    ...     spectral_radius=0.95,
    ... )

    Training and forecasting:

    >>> from resdag.training import ESNTrainer
    >>> trainer = ESNTrainer(model)
    >>> trainer.fit(
    ...     warmup_inputs=(warmup,),
    ...     train_inputs=(train,),
    ...     targets={"output": target},
    ... )
    >>> predictions = model.forecast(forecast_warmup, horizon=100)

    See Also
    --------
    classic_esn : Traditional ESN without state augmentation.
    resdag.training.ESNTrainer : Trainer for fitting readout.
    resdag.init.topology.get_topology : Get topology by name.
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
