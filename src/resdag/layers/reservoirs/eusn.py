r"""
EuSN Reservoir Layer
====================

This module provides the public-facing Euler State Network (EuSN) sequence
layer:

- :class:`EuSNLayer` — stateful RNN layer wrapping an
  :class:`~resdag.layers.cells.eusn_cell.EuSNCell`.

References
----------
C. Gallicchio, "Euler State Networks: Non-dissipative Reservoir Computing,"
*Neurocomputing* 531 (2023): 214-228; arXiv:2203.09382.

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base (BaseReservoirLayer).
resdag.layers.cells.eusn_cell : Underlying single-step cell.
resdag.layers.reservoirs.esn : Standard leaky-ESN layer.
"""

from typing import Any

import torch

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells.eusn_cell import EuSNCell
from resdag.utils.general import SeedLike

from .base_reservoir import BaseReservoirLayer


class EuSNLayer(BaseReservoirLayer):
    r"""
    Stateful Euler State Network (EuSN) reservoir layer.

    Public-facing class.  Internally creates an
    :class:`~resdag.layers.cells.eusn_cell.EuSNCell` that owns all parameters and
    performs the single-step integration; the sequence loop and state management
    live in the parent
    :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`.

    EuSN (Gallicchio, arXiv:2203.09382) integrates an ODE with an antisymmetric
    recurrent matrix by one forward-Euler step:

    .. math::

        h_t = h_{t-1} + \varepsilon\,
              \tanh\!\big((W_r - W_r^\top - \gamma I)\, h_{t-1}
              + W_{fb} x_{fb,t} + W_{in} u_t + b\big).

    The antisymmetric part :math:`W_r - W_r^\top` conserves the state norm, so a
    small step :math:`\varepsilon` gives stable, near-norm-preserving, long-memory
    dynamics; the diffusion :math:`\gamma I` damps the Euler discretization
    error.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden-state dimension).
    feedback_size : int
        Dimension of the feedback signal.  Required.
    epsilon : float, default=0.01
        Forward-Euler step size :math:`\varepsilon`.  A *live* knob is not
        offered (the step size is baked into the integration); rebuild the layer
        to change it.
    gamma : float, default=0.01
        Diffusion coefficient :math:`\gamma \ge 0`.  A *live* knob:
        ``layer.gamma = ...`` takes effect on the next forward.
    input_size : int, optional
        Dimension of a driving input.  ``None`` builds a feedback-only layer.
    spectral_radius : float, optional
        Target rotation rate for ``W_r`` (largest imaginary-eigenvalue magnitude
        of the antisymmetric part).
    bias : bool, default=True
        Whether to include a bias term.
    bias_scaling : float, default=1.0
        Scale of the random bias (``uniform(-bias_scaling, bias_scaling)``).
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation of the vector field (``tanh`` is the EuSN choice).
    substeps : int, default=1
        Fixed integration sub-steps per interval.  ``1`` is the single
        forward-Euler step of the paper.
    solver : {'euler', 'rk4', 'torchdiffeq'}, default='euler'
        Integration scheme (see :class:`EuSNCell`).
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.
    feedback_initializer, input_initializer, topology : spec, optional
        Same accepted forms as :class:`~resdag.layers.ESNLayer`.  ``topology``
        defaults (``None``) to a dense random ``W_r``; the cell antisymmetrizes
        it on the fly.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing the recurrent, feedback, input and bias
        draws.

    recurrent_initializer : spec, optional
        First-class alias for ``topology``. Supply one spelling only. Names,
        tuples, callables and registered initializer objects are accepted.
    bias_initializer : spec, optional
        Input/feedback initializer grammar applied to a ``(1, reservoir_size)``
        view of the bias. Requires ``bias=True``; overrides ``bias_scaling``.
    initial_state_initializer : spec, optional
        Initializes a persistent ``(1, cell.state_size)`` reset template once.
        Each batch example receives an independent copy at reset. Zero by default.
    device : torch.device or str, optional
        Initial parameter location. Metal construction initializes on CPU before
        transferring to support initialization operations across backends.
    dtype : torch.dtype, optional
        Floating precision used during initialization, not only after conversion.

    Attributes
    ----------
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, reservoir_size)``, or
        ``None`` before the first forward.

    Examples
    --------
    >>> import torch
    >>> reservoir = EuSNLayer(reservoir_size=200, feedback_size=3, epsilon=0.01)
    >>> feedback = torch.randn(4, 50, 3)  # (batch, time, features)
    >>> states = reservoir(feedback)
    >>> states.shape
    torch.Size([4, 50, 200])

    See Also
    --------
    resdag.layers.cells.eusn_cell.EuSNCell : Underlying single-step cell.
    resdag.layers.ESNLayer : Standard leaky-ESN layer.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        epsilon: float = 0.01,
        gamma: float = 0.01,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        activation: str = "tanh",
        substeps: int = 1,
        solver: str = "euler",
        torchdiffeq_method: str = "dopri5",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        trainable: bool = False,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        topology: TopologySpec = None,
        seed: SeedLike = None,
        *,
        recurrent_initializer: TopologySpec = None,
        bias_initializer: InitializerSpec = None,
        initial_state_initializer: InitializerSpec = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        cell = EuSNCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            epsilon=epsilon,
            gamma=gamma,
            input_size=input_size,
            spectral_radius=spectral_radius,
            bias=bias,
            bias_scaling=bias_scaling,
            activation=activation,
            substeps=substeps,
            solver=solver,
            torchdiffeq_method=torchdiffeq_method,
            rtol=rtol,
            atol=atol,
            trainable=trainable,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            topology=topology,
            seed=seed,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            initial_state_initializer=initial_state_initializer,
            device=device,
            dtype=dtype,
        )
        super().__init__(cell)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute lookups to the wrapped cell."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Reach into _modules directly to avoid recursion during __init__.
        modules = self.__dict__.get("_modules")
        if modules is not None and "cell" in modules:
            try:
                return getattr(modules["cell"], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        """Return string representation (delegates to the cell)."""
        return repr(self.cell)
