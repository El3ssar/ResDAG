r"""
EuSN Architecture
=================

This module provides :func:`eusn`, which builds an Euler State Network (EuSN)
model with input concatenation, following the same
``Input -> Reservoir -> Concatenate(Input) -> Readout`` design as
:func:`~resdag.models.classic_esn` but with an
:class:`~resdag.layers.reservoirs.eusn.EuSNLayer` reservoir.

.. important::
   **EuSN is for input-driven tasks, not closed-loop chaos forecasting.** Its
   antisymmetric dynamics are near-conservative (norm-preserving) by design, so
   they have long linear memory but lack the *contractive* echo-state property
   that stabilises an autoregressive rollout — a free-running EuSN forecast of a
   chaotic attractor collapses immediately (valid-prediction-time ~0). Use it
   the way the paper does: driven regression / sequence classification with a
   readout over the (last or pooled) states, where its stability and long memory
   are the point. For autoregressive chaotic forecasting reach for
   :func:`~resdag.models.ott_esn` / :func:`~resdag.models.classic_esn` or
   :func:`~resdag.models.es2n` instead.

References
----------
C. Gallicchio, "Euler State Networks: Non-dissipative Reservoir Computing,"
*Neurocomputing* 531 (2023): 214-228; arXiv:2203.09382.

See Also
--------
classic_esn : Traditional leaky-ESN architecture.
es2n : Edge-of-Stability State Network (the discrete-skip sibling).
"""

from typing import Any

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.reservoirs.eusn import EuSNLayer

from ._builder import ReadoutSpec, _esn_builder


def eusn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    epsilon: float = 0.01,
    gamma: float = 0.01,
    input_size: int | None = None,
    input_initializer: InitializerSpec | None = None,
    # Reservoir params
    topology: TopologySpec | None = None,
    spectral_radius: float = 0.9,
    substeps: int = 1,
    solver: str = "euler",
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
    r"""Build an Euler State Network (EuSN) model.

    EuSN (Gallicchio, arXiv:2203.09382) replaces the leaky-ESN state update with
    one forward-Euler step of an ODE whose recurrent matrix is antisymmetric
    (norm-preserving) minus a small diffusion term:

    .. math::

        h_t = h_{t-1} + \varepsilon\,
              \tanh\!\big((W_r - W_r^\top - \gamma I)\, h_{t-1} + \cdots\big).

    Small :math:`\varepsilon` and :math:`\gamma` sit the reservoir near the edge
    of stability (non-dissipative, long memory).  This factory wires the same
    input-concatenation architecture as :func:`~resdag.models.classic_esn`::

        Input -> EuSN Reservoir -> Concatenate(Input, Reservoir) -> Readout

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (autoregressive input dimension).
    output_size : int
        Number of output features.
    epsilon : float, default=0.01
        Forward-Euler step size :math:`\varepsilon`.  Smaller values sit the
        reservoir closer to the edge of stability (longer, near-lossless
        memory).
    gamma : float, default=0.01
        Diffusion coefficient :math:`\gamma \ge 0` damping the Euler
        discretization error.  ``0.0`` recovers the undamped antisymmetric flow.
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input.  When given, the
        model takes two inputs ``(feedback, driver)``; the driver is kept out of
        the input concatenation.  ``None`` (default) builds a feedback-only
        model.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Only used when
        ``input_size`` is given.
    topology : TopologySpec, optional
        Topology for the recurrent weights ``W_r`` (antisymmetrized on the fly
        by the cell).  Accepts a registry name, a ``(name, params)`` tuple, a
        callable, or a pre-configured
        :class:`~resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Target rotation rate for ``W_r`` (largest imaginary-eigenvalue magnitude
        of the antisymmetric part).
    substeps : int, default=1
        Fixed integration sub-steps per interval.  ``1`` is the single
        forward-Euler step of the paper.
    solver : {'euler', 'rk4', 'torchdiffeq'}, default='euler'
        Integration scheme.  ``'euler'`` is the paper's method; ``'rk4'`` and
        the adaptive ``'torchdiffeq'`` (requires ``resdag[ode]``) integrate the
        same field more accurately.
    feedback_initializer : InitializerSpec, optional
        Initializer for the feedback weights.
    activation : {'tanh', 'relu', 'sigmoid', 'identity'}, default='tanh'
        Activation of the vector field (``tanh`` is the EuSN choice).
    bias : bool, default=True
        Whether to use a bias in the reservoir.
    trainable : bool, default=False
        Whether the reservoir weights are trainable.
    readout : str or type[ReadoutLayer], default="ridge"
        A built-in name (``"ridge"``, ``"cholesky"``, ``"cg"``, ``"svd"``,
        ``"pinv"``, ``"incremental"``, ``"rls"``, ``"fista"``) or a
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
        :class:`~resdag.layers.reservoirs.eusn.EuSNLayer` (e.g. ``bias_scaling``,
        ``seed``, ``torchdiffeq_method``).

    Returns
    -------
    :class:`~resdag.core.ESNModel`
        Configured EuSN model ready for training and inference.

    Examples
    --------
    >>> from resdag.models import eusn
    >>> model = eusn(
    ...     reservoir_size=500, feedback_size=3, output_size=3,
    ...     epsilon=0.01, gamma=0.01,
    ... )

    See Also
    --------
    resdag.models.es2n : Edge-of-Stability State Network (discrete orthogonal skip).
    resdag.models.classic_esn : Traditional leaky ESN.
    resdag.layers.reservoirs.eusn.EuSNLayer : The reservoir layer.
    """
    return _esn_builder(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        output_size=output_size,
        augment=None,
        concat_input=True,
        input_size=input_size,
        input_initializer=input_initializer,
        reservoir_cls=EuSNLayer,
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
        epsilon=epsilon,
        gamma=gamma,
        substeps=substeps,
        solver=solver,
        **reservoir_kwargs,
    )
