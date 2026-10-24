"""
ResESN Architecture
===================

This module provides `resesn`, which builds a Residual Echo State Network
(ResESN) model with input concatenation, following the same
`Input -> Reservoir -> Concatenate(Input) -> Readout` design as
`resdag.models.classic_esn` but with a
`resdag.layers.reservoirs.resesn.ResESNLayer` reservoir.

References
----------
A. Ceni and C. Gallicchio, "Residual Echo State Networks," Neurocomputing
(2024).

See Also
--------
es2n : Edge-of-Stability State Network — the convex-combination sibling.
classic_esn : Traditional leaky-ESN architecture.
"""

from typing import Any

from resdag.core import ESNModel
from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.reservoirs.resesn import ResESNLayer

from ._builder import ReadoutSpec, _esn_builder


def resesn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    alpha: float = 0.5,
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
    r"""Build a Residual Echo State Network (ResESN) model.

    ResESN (Ceni & Gallicchio, 2024) generalises ES2N by **decoupling** the
    orthogonal-skip and nonlinear branch weights:

    $$
    x_t = \alpha\,O\,x_{t-1} + \beta\,\phi(W_r x_{t-1} + \cdots)
    $$

    with independent coefficients $\alpha$ (skip) and $\beta$
    (nonlinear) replacing the leak rate.  $\alpha = 1 - \beta$ recovers
    ES2N.  This factory wires the same input-concatenation architecture as
    `resdag.models.classic_esn`:

    ```text
    Input -> ResESN Reservoir -> Concatenate(Input, Reservoir) -> Readout
    ```

    Parameters
    ----------
    reservoir_size : int
        Number of units in the reservoir.
    feedback_size : int
        Number of feedback features (autoregressive input dimension).
    output_size : int
        Number of output features.
    alpha : float, default=0.5
        Scaling of the orthogonal skip (residual) branch.
    beta : float, default=0.9
        Scaling of the nonlinear branch.
    input_size : int or None, optional
        Dimension of an optional driving (exogenous) input.  When given, the
        model takes two inputs `(feedback, driver)`; the driver is kept out of
        the input concatenation.  `None` (default) builds a feedback-only
        model.
    input_initializer : InitializerSpec, optional
        Initializer for the driving-input weights.  Only used when
        `input_size` is given.
    topology : TopologySpec, optional
        Topology for the recurrent weights `W_r`.  Accepts a registry name, a
        `(name, params)` tuple, a callable, or a pre-configured
        `resdag.init.topology.TopologyInitializer`.
    spectral_radius : float, default=0.9
        Target spectral radius for the recurrent weights `W_r`.
    orthogonal_scaling : float, default=1.0
        Gain on the orthogonal skip matrix `O`.  `1.0` keeps it exactly
        norm-preserving.
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
        A built-in name (`"ridge"`, `"cholesky"`, `"cg"`, `"svd"`,
        `"pinv"`, `"incremental"`, `"rls"`, `"fista"`, `"lsmr"`) or a
        `resdag.layers.readouts.ReadoutLayer` subclass. Additional
        solver options go in `readout_kwargs`.
    readout_alpha : float, default=1e-6
        Ridge regularization for the readout.
    readout_bias : bool, default=True
        Whether to use a bias in the readout.
    readout_name : str, default="output"
        Name for the readout layer (used as the target key in training).
    readout_kwargs : dict, optional
        Extra readout options, such as `trainable=True`, `solver`,
        `state_noise` or `gram_dtype`. Architecture dimensions are fixed.
    **reservoir_kwargs : Any
        Additional keyword arguments forwarded to
        `resdag.layers.reservoirs.resesn.ResESNLayer` (e.g.
        `bias_scaling`, `seed`).

    Returns
    -------
    `resdag.core.ESNModel`
        Configured ResESN model ready for training and inference.

    Examples
    --------
    >>> from resdag.models import resesn
    >>> model = resesn(500, feedback_size=3, output_size=3, alpha=0.6, beta=0.4)

    See Also
    --------
    resdag.models.es2n : Edge-of-Stability State Network (convex combination).
    resdag.models.classic_esn : Traditional leaky ESN.
    resdag.layers.reservoirs.resesn.ResESNLayer : The reservoir layer.
    """
    return _esn_builder(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        output_size=output_size,
        augment=None,
        concat_input=True,
        input_size=input_size,
        input_initializer=input_initializer,
        reservoir_cls=ResESNLayer,
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
        alpha=alpha,
        beta=beta,
        orthogonal_scaling=orthogonal_scaling,
        **reservoir_kwargs,
    )
