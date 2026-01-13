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

import pytorch_symbolic as ps

from ..composition import ESNModel
from ..init.utils import InitializerSpec, TopologySpec, resolve_initializer, resolve_topology
from ..layers import ReservoirLayer
from ..layers.custom import Concatenate
from ..layers.readouts import CGReadoutLayer


def classic_esn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    # Reservoir params
    topology: TopologySpec = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 1.0,
    feedback_initializer: InitializerSpec = None,
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
    topology : TopologySpec, optional
        Topology for recurrent weights. Accepts:

        - str: Registry name (e.g., ``"erdos_renyi"``)
        - tuple: (name, params) like ``("watts_strogatz", {"k": 6, "p": 0.1})``
        - :class:`~resdag.init.topology.TopologyInitializer`: Pre-configured object
    spectral_radius : float, default=0.9
        Desired spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate (1.0 = no leak).
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
        Additional keyword arguments passed to :class:`~resdag.layers.ReservoirLayer`.

    Returns
    -------
    :class:`~resdag.composition.ESNModel`
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
    # Resolve topology and initializer specs
    resolved_topology = resolve_topology(topology)
    resolved_feedback_init = resolve_initializer(feedback_initializer)

    # Build model with pytorch_symbolic
    inp = ps.Input((100, feedback_size))  # Use typical seq_len for tracing

    reservoir = ReservoirLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=0,  # No driving input in classic ESN
        topology=resolved_topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=resolved_feedback_init,
        activation=activation,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(inp)

    concat = Concatenate()(inp, reservoir)

    readout = CGReadoutLayer(
        in_features=reservoir_size + feedback_size,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,
        name=readout_name,
    )(concat)

    return ESNModel(inp, readout)
