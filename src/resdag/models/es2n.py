"""
ES2N Architecture
=================

This module provides :func:`es2n`, which builds an Edge-of-Stability State
Network (ES2N) model with input concatenation, following the same
``Input -> Reservoir -> Concatenate(Input) -> Readout`` design as
:func:`~resdag.models.classic_esn` but with an
:class:`~resdag.layers.reservoirs.es2n.ES2NLayer` reservoir.

References
----------
A. Ceni and C. Gallicchio, "Edge of Stability Echo State Networks,"
arXiv:2308.02902 (2023/2025).

See Also
--------
classic_esn : Traditional leaky-ESN architecture.
resesn : Residual ESN — the decoupled-weight sibling.
"""

from typing import Any

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.reservoirs.es2n import ES2NLayer

from ._builder import ReadoutSpec, _esn_builder


def es2n(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    beta: float = 0.9,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    orthogonal_scaling: float = 1.0,
    feedback_initializer: InitializerSpec | None = None,
    activation: str = "tanh",
    bias: bool = True,
    trainable: bool = False,
    # Readout params
    readout: ReadoutSpec = "ridge",
    readout_alpha: float = 1e-6,
    readout_bias: bool = True,
    readout_name: str = "output",
    readout_kwargs: dict[str, Any] | None = None,
    # Extra reservoir kwargs
    **reservoir_kwargs: Any,
) -> ESNModel:
    r"""Build an Edge-of-Stability State Network (ES2N) model.

    ES2N (Ceni & Gallicchio, arXiv:2308.02902) replaces the leaky-ESN state
    update with a convex combination of a nonlinear ESN branch and a fixed
    random *orthogonal* skip:

    .. math::

        x_t = \beta\,\phi(W_r x_{t-1} + \cdots) + (1 - \beta)\,O\,x_{t-1}

    Small ``beta`` sits the reservoir near the edge of stability (orthogonal,
    norm-preserving, long memory); ``beta = 1`` recovers a standard ESN.  This
    factory wires the same input-concatenation architecture as
    :func:`~resdag.models.classic_esn`::

        Input -> ES2N Reservoir -> Concatenate(Input, Reservoir) -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (autoregressive input dimension).
    output_size : int
        Number of output features.
    beta : float, default=0.9
        Proximity to the edge of stability in :math:`(0, 1]`.  Smaller values
        weight the orthogonal skip (longer memory); ``1.0`` recovers a standard
        non-leaky ESN.  For chaotic-system forecasting, smaller ``beta`` (e.g.
        0.1--0.3) is typically preferred.
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input.  When given, the
        model takes two inputs ``(feedback, driver)``; the driver is kept out of
        the input concatenation.  ``None`` (default) builds a feedback-only
        model.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Only used when
        ``input_size`` is given.
    topology : TopologySpec, optional
        Topology for the recurrent weights ``W_r``.  Accepts a registry name, a
        ``(name, params)`` tuple, a callable, or a pre-configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Target spectral radius for the recurrent weights ``W_r``.
    orthogonal_scaling : float, default=1.0
        Gain on the orthogonal skip matrix ``O``.  ``1.0`` keeps it exactly
        norm-preserving (the edge-of-stability regime).
    feedback_initializer : InitializerSpec, optional
        Initializer for the feedback weights.
    activation : {'tanh', 'relu', 'sigmoid', 'identity'}, default='tanh'
        Activation for the nonlinear branch.
    bias : bool, default=True
        Whether to use a bias in the reservoir.
    trainable : bool, default=False
        Whether the reservoir weights are trainable.  The orthogonal skip stays
        frozen regardless.
    readout : str or type[ReadoutLayer], default="ridge"
        A built-in name (``"ridge"``, ``"cholesky"``, ``"cg"``, ``"svd"``,
        ``"pinv"``, ``"incremental"``, ``"rls"``, ``"fista"``, ``"lsmr"``) or a
        :class:`~resdag.layers.readouts.ReadoutLayer` subclass. Additional
        solver options go in ``readout_kwargs``.
    readout_alpha : float, default=1e-6
        Ridge regularization for the readout.
    readout_bias : bool, default=True
        Whether to use a bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    readout_kwargs : dict, optional
        Extra readout options, such as ``trainable=True``, ``solver``,
        ``state_noise`` or ``gram_dtype``. Architecture dimensions are fixed.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to
        :class:`~resdag.layers.reservoirs.es2n.ES2NLayer` (e.g. ``bias_scaling``,
        ``seed``).

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        Configured ES2N model ready for training and inference.

    Examples
    --------
    >>> from resdag.models import es2n
    >>> model = es2n(reservoir_size=500, feedback_size=3, output_size=3, beta=0.1)

    See Also
    --------
    resdag.models.resesn : Residual ESN (decoupled skip / nonlinear weights).
    resdag.models.classic_esn : Traditional leaky ESN.
    resdag.layers.reservoirs.es2n.ES2NLayer : The reservoir layer.
    """
    return _esn_builder(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        output_size=output_size,
        augment=None,
        concat_input=True,
        input_size=input_size,
        input_initializer=input_initializer,
        reservoir_cls=ES2NLayer,
        topology=topology,
        spectral_radius=spectral_radius,
        feedback_initializer=feedback_initializer,
        activation=activation,
        bias=bias,
        trainable=trainable,
        readout=readout,
        readout_alpha=readout_alpha,
        readout_bias=readout_bias,
        readout_name=readout_name,
        readout_kwargs=readout_kwargs,
        beta=beta,
        orthogonal_scaling=orthogonal_scaling,
        **reservoir_kwargs,
    )
