"""Classic ESN architecture with input concatenation.

This module provides the :func:`classic_esn` function for building traditional
Echo State Network architectures where the input is concatenated with the
reservoir output before the readout layer.

See Also
--------
:func:`resdag.models.ott_esn` : OTT (Open Temporal Topology) ESN variant
:func:`resdag.models.linear_esn` : Linear ESN variant
:func:`resdag.models.headless_esn` : Headless ESN (no readout)
"""

from typing import Any

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec

from ._builder import _esn_builder


def classic_esn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    noise: float = 0.0,
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
    """Build a classic Echo State Network (ESN) model.

    This architecture concatenates the input with the reservoir output before
    passing to the readout layer, following the traditional ESN design.

    Architecture::

        Input -> Reservoir -> Concatenate(Input, Reservoir) -> Readout

    The readout sees both the raw input and the reservoir's nonlinear
    transformation, which can improve performance on many tasks.

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (input dimension).
    output_size : int
        Number of output features.
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input.  When given, the
        model takes two inputs ``(feedback, driver)`` and the driver feeds the
        reservoir alongside the autoregressive feedback.  The driver is kept out
        of the input concatenation, so the readout ``in_features`` stays
        ``feedback_size + reservoir_size``.  ``None`` (default) builds a
        feedback-only model, unchanged from previous behavior.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Same accepted forms as
        ``feedback_initializer``.  Only used when ``input_size`` is given.
    topology : TopologySpec, optional
        Topology for recurrent weights. Accepts:

        - str: Registry name (e.g., ``"erdos_renyi"``)
        - tuple: (name, params) like ``("watts_strogatz", {"k": 6, "p": 0.1})``
        - :class:`~resdag.init.topology.TopologyInitializer`: Pre-configured object
    spectral_radius : float, default=0.9
        Desired spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate (1.0 = no leak).
    noise : float, default=0.0
        Standard deviation of additive Gaussian state noise injected into the
        reservoir after the activation.  Active only in training mode (a no-op
        under :meth:`~torch.nn.Module.eval`); ``0.0`` disables it.  Forwarded to
        :class:`~resdag.layers.ESNLayer`.  Must be non-negative.
    feedback_initializer : InitializerSpec, optional
        Initializer for feedback weights. Accepts:

        - str: Registry name (e.g., ``"pseudo_diagonal"``)
        - tuple: (name, params) like ``("chebyshev", {"p": 0.5})``
        - :class:`~resdag.init.input_feedback.InputFeedbackInitializer`: Pre-configured object
    activation : str, default="tanh"
        Activation function (``"tanh"``, ``"relu"``, ``"sigmoid"``, ``"identity"``).
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
    **reservoir_kwargs : Any
        Additional keyword arguments passed to :class:`~resdag.layers.ESNLayer`.

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        Configured ESN model ready for training and inference.

    Examples
    --------
    Simple usage with defaults:

    >>> from resdag.models import classic_esn
    >>> import torch
    >>> model = classic_esn(100, 1, 1)

    With custom topology and initializer:

    >>> model = classic_esn(
    ...     reservoir_size=400,
    ...     feedback_size=1,
    ...     output_size=1,
    ...     topology=("watts_strogatz", {"k": 6, "p": 0.1}),
    ...     feedback_initializer="pseudo_diagonal",
    ...     spectral_radius=0.9,
    ...     leak_rate=0.5,
    ... )

    Forward pass:

    >>> x = torch.randn(4, 50, 1)  # (batch, time, features)
    >>> y = model(x)

    See Also
    --------
    :func:`resdag.models.ott_esn` : OTT ESN variant
    :func:`resdag.models.linear_esn` : Linear ESN variant
    :class:`resdag.training.ESNTrainer` : Trainer for fitting readouts
    """
    # Classic ESN: no augmentation, concatenate the raw input back in.
    return _esn_builder(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        output_size=output_size,
        augment=None,
        concat_input=True,
        input_size=input_size,
        input_initializer=input_initializer,
        topology=topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        noise=noise,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        readout_alpha=readout_alpha,
        readout_bias=readout_bias,
        readout_name=readout_name,
        **reservoir_kwargs,
    )
