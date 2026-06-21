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

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import SelectiveExponentiation

from ._builder import _esn_builder


def ott_esn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
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
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input.  When given, the
        model takes two inputs ``(feedback, driver)`` and the driver feeds the
        reservoir alongside the autoregressive feedback.  The driver is kept out
        of the input concatenation, so the readout ``in_features`` stays
        ``feedback_size + reservoir_size``.  ``None`` (default) builds a
        feedback-only model, unchanged from previous behavior.
    input_initializer : str, tuple, or InputFeedbackInitializer, optional
        Initializer for the driving-input weights.  Same format as
        ``feedback_initializer``.  Only used when ``input_size`` is given.
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
        Additional keyword arguments passed to :class:`ESNLayer`.

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
    # Ott augmentation: square even-indexed reservoir units, concatenate input.
    return _esn_builder(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        output_size=output_size,
        augment=lambda: SelectiveExponentiation(index=0, exponent=2.0),
        concat_input=True,
        input_size=input_size,
        input_initializer=input_initializer,
        topology=topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        readout_alpha=readout_alpha,
        readout_bias=readout_bias,
        readout_name=readout_name,
        **reservoir_kwargs,
    )
