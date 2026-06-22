"""
Power Augmented ESN Architecture
================================

This module provides :func:`power_augmented`, which builds an ESN model that
augments the reservoir state by raising **every** unit to a configurable
``exponent`` before concatenating with the input and feeding the readout.

It generalises the state augmentation of :func:`ott_esn`: where ``ott_esn``
squares only the even-indexed units, this factory exponentiates all units by a
free exponent, so the exponent becomes a hyperparameter you can sweep. The
default ``exponent=3.0`` is an odd integer, which preserves the sign of the
(signed, ``[-1, 1]``) ``tanh`` reservoir states — see the ``exponent`` notes in
:func:`power_augmented` for the trade-offs of other exponents.

References
----------
J. Pathak, B. Hunt, M. Girvan, Z. Lu and E. Ott, "Model-Free Prediction of
Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing
Approach," Phys. Rev. Lett. 120, 024102 (2018).

See Also
--------
ott_esn : Fixed even-index squaring; the special case this generalises.
classic_esn : Traditional ESN architecture without state augmentation.
headless_esn : Reservoir-only model for analysis.
"""

from typing import Any

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import Power

from ._builder import _esn_builder


def power_augmented(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    exponent: float = 3.0,
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
    Build Power Augmented ESN model.

    This model augments reservoir states by exponentiating to a power and concatenating with input.
    This augmentation helps capture higher-order dynamics in chaotic systems.

    Architecture::

        Input -> Reservoir -> Power -> Concatenate(Input, Augmented) -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Dimension of feedback signal (input features).
    output_size : int
        Dimension of output signal.
    exponent : float, default=3.0
        Exponent applied to **every** reservoir state. Tanh reservoir states
        live in ``[-1, 1]`` and routinely include negative and zero values, so
        the exponent must be chosen with that signed range in mind (see Notes).
        The default ``3.0`` is an odd integer and therefore *sign-preserving*;
        an even exponent such as ``2.0`` maps every state to a non-negative
        value and so discards the sign of the reservoir activations.
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
        Configured Power Augmented ESN model ready for training and inference.

    Notes
    -----
    The augmentation applies :class:`~resdag.layers.Power` to the raw reservoir
    states, which for a ``tanh`` activation lie in ``[-1, 1]`` — negatives and
    zeros included. Choose ``exponent`` accordingly:

    - **Even integers** (``2.0``, ``4.0``, …) are always safe: every state maps
      to a non-negative value with no ``nan``/``inf``.
    - **Odd integers** (``3.0``, ``5.0``, …) are safe and *sign-preserving*:
      negative states stay negative.
    - **Non-integer exponents** (e.g. ``0.5``, ``1.5``) applied to a *negative*
      state produce ``nan`` under the default :func:`torch.pow`, silently
      corrupting the readout inputs. **Negative exponents** (e.g. ``-1.0``)
      produce ``inf`` on the zeros that ``tanh`` states pass through.

    If you need a non-integer exponent on signed states, wire the model by hand
    with ``Power(exponent, sign_preserving=True)``, which applies
    ``sign(x) * abs(x) ** exponent`` and stays finite for negative bases. This
    factory always uses the default (non-sign-preserving) ``Power`` to keep the
    well-established integer-exponent behaviour unchanged.

    Examples
    --------
    Basic usage:

    >>> from resdag.models import power_augmented
    >>> model = power_augmented(
    ...     reservoir_size=500,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     exponent=3.0,
    ... )
    >>> model.summary()

    With custom topology:

    >>> from resdag.init.topology import get_topology
    >>> model = power_augmented(
    ...     reservoir_size=500,
    ...     feedback_size=3,
    ...     output_size=3,
    ...     topology=get_topology("watts_strogatz", k=4, p=0.3),
    ...     spectral_radius=0.95,
    ...     exponent=3.0,
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
    # Power augmentation: raise reservoir states to ``exponent``, concat input.
    return _esn_builder(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        output_size=output_size,
        augment=lambda: Power(exponent=exponent),
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
