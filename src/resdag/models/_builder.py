"""Shared ESN-model builder behind the premade factories.

This private module hosts :func:`_esn_builder`, the single construction routine
that every premade factory in :mod:`resdag.models` delegates to.  The premade
architectures (``classic_esn``, ``ott_esn``, ``power_augmented``,
``headless_esn``, ``linear_esn``) differ only in a few orthogonal choices:

- which augmentation node (if any) is applied to the reservoir output,
- whether the raw input is concatenated back in before the readout,
- whether a readout layer is attached at all.

Rather than repeat the ``reservoir_input -> ESNLayer -> Concatenate ->
CGReadoutLayer -> ESNModel`` chain in five places, those choices are expressed
as arguments to :func:`_esn_builder`.

This module is intentionally private; the public surface is the factory
functions exported from :mod:`resdag.models`.
"""

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from pytorch_symbolic.symbolic_data import SymbolicTensor

from resdag.core import ESNModel, reservoir_input
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer

AugmentFactory = Callable[[], nn.Module]


def _esn_builder(
    reservoir_size: int,
    feedback_size: int,
    output_size: int | None = None,
    *,
    augment: AugmentFactory | None = None,
    concat_input: bool = True,
    input_size: int | None = None,
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
    """Build an ESN model from composable, orthogonal building blocks.

    This is the shared construction routine behind every premade factory in
    :mod:`resdag.models`.  It always builds a reservoir, then optionally applies
    an augmentation node, optionally concatenates the raw input back in, and
    optionally attaches a ridge-regression readout.

    Architecture (each optional stage is skipped when not requested)::

        Input -> Reservoir -> [augment] -> [Concatenate(Input, .)] -> [Readout]

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (input dimension).
    output_size : int or None, optional
        Number of output features.  Required (non-``None``) when a readout is
        attached; ignored when ``output_size`` is ``None`` (no readout, e.g. the
        headless and linear architectures).
    augment : callable, optional
        Zero-argument factory returning an :class:`~torch.nn.Module` applied to
        the reservoir output (e.g. ``SelectiveExponentiation`` or ``Power``).
        ``None`` (default) leaves the reservoir output untouched.
    concat_input : bool, default=True
        Whether to concatenate the raw input with the (possibly augmented)
        reservoir output before the readout.  Only meaningful when a readout is
        attached.
    input_size : int or None, optional
        Dimension of an optional driving input passed to the reservoir.
        ``None`` (default) builds a feedback-only reservoir.
    topology : TopologySpec, optional
        Topology for recurrent weights.  Accepts a registry name, a
        ``(name, params)`` tuple, a callable, or a pre-configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Desired spectral radius for recurrent weights.
    leak_rate : float, default=1.0
        Leaky integration rate (1.0 = no leak).
    feedback_initializer : InitializerSpec, optional
        Initializer for feedback weights.  Same accepted forms as ``topology``.
    activation : str, default="tanh"
        Activation function (``"tanh"``, ``"relu"``, ``"sigmoid"``,
        ``"identity"``).
    bias : bool, default=True
        Whether to use bias in the reservoir.
    trainable : bool, default=False
        Whether reservoir weights are trainable.
    readout_alpha : float, default=1e-6
        Ridge regression regularization for the readout.
    readout_bias : bool, default=True
        Whether to use bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to
        :class:`~resdag.layers.ESNLayer`.

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        The assembled model.  Its output is the readout when ``output_size`` is
        given, otherwise the (possibly augmented) reservoir output.

    Raises
    ------
    ValueError
        If a readout is requested (``output_size`` is not ``None``) but
        ``output_size`` is not a positive integer.

    See Also
    --------
    resdag.models.classic_esn : Classic ESN built on this helper.
    resdag.models.ott_esn : Ott's state-augmented ESN built on this helper.
    """
    # Build the symbolic input.  The time dimension is a placeholder — actual
    # sequence lengths are inferred from the input at call time.
    inp = reservoir_input(feedback_size)

    output: SymbolicTensor = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=input_size,
        topology=topology,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        **reservoir_kwargs,
    )(inp)

    if augment is not None:
        output = augment()(output)

    # Headless / linear variants stop at the (possibly augmented) reservoir.
    if output_size is None:
        return ESNModel(inp, output)

    if output_size <= 0:
        raise ValueError(f"output_size must be a positive integer, got {output_size!r}")

    if concat_input:
        output = Concatenate()(inp, output)
        readout_in_features = feedback_size + reservoir_size
    else:
        readout_in_features = reservoir_size

    readout = CGReadoutLayer(
        in_features=readout_in_features,
        out_features=output_size,
        bias=readout_bias,
        alpha=readout_alpha,
        name=readout_name,
    )(output)

    return ESNModel(inp, readout)
