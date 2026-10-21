"""
ResESN Reservoir Layer
======================

This module provides the public-facing Residual Echo State Network (ResESN)
sequence layer:

- :class:`ResESNLayer` — stateful RNN layer wrapping a :class:`ResESNCell`.

References
----------
A. Ceni and C. Gallicchio, "Residual Echo State Networks," Neurocomputing
(2024).  Shallow-model equation reviewed as Eq. (2) in "Deep Residual Echo
State Networks," arXiv:2508.21172 (2025).

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base (BaseReservoirLayer).
resdag.layers.cells.resesn_cell : Underlying single-step cell.
resdag.layers.reservoirs.es2n : ES2N — the convex-combination sibling.
"""

from typing import Any

import torch

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells.resesn_cell import ResESNCell
from resdag.utils.general import SeedLike

from .base_reservoir import BaseReservoirLayer


class ResESNLayer(BaseReservoirLayer):
    r"""
    Stateful Residual Echo State Network (ResESN) reservoir layer.

    Public-facing class.  Internally creates a
    :class:`~resdag.layers.cells.resesn_cell.ResESNCell` that owns all
    parameters and performs the single-step update; the sequence loop and state
    management live in the parent
    :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`.

    ResESN (Ceni & Gallicchio, 2024) generalises ES2N by **decoupling** the
    skip and nonlinear branch weights:

    .. math::

        x_t = \alpha\,O\,x_{t-1}
              + \beta\,\phi(W_r x_{t-1} + W_{fb} x_{fb,t} + W_{in} u_t + b)

    with independent scaling coefficients :math:`\alpha` (orthogonal skip) and
    :math:`\beta` (nonlinear branch) replacing the leak rate.  :math:`\alpha =
    1 - \beta` recovers ES2N; :math:`O = I` with :math:`1 - \alpha = \beta`
    recovers a leaky ESN.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden-state dimension).
    feedback_size : int
        Dimension of the feedback signal.  Required.
    alpha : float, default=0.5
        Scaling of the orthogonal skip (residual) branch.  A *live* knob:
        ``layer.alpha = ...`` takes effect on the next forward.
    beta : float, default=0.9
        Scaling of the nonlinear branch.  A *live* knob.
    input_size : int, optional
        Dimension of a driving input.  ``None`` builds a feedback-only layer.
    spectral_radius : float, optional
        Target spectral radius for the recurrent matrix ``W_r``.
    bias : bool, default=True
        Whether to include a bias term.
    bias_scaling : float, default=1.0
        Scale of the random bias (``uniform(-bias_scaling, bias_scaling)``).
    activation : {'tanh', 'relu', 'identity', 'sigmoid'}, default='tanh'
        Activation for the nonlinear branch.
    orthogonal_scaling : float, default=1.0
        Gain on the orthogonal skip matrix ``O``.  ``1.0`` keeps it exactly
        norm-preserving.
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.  The
        orthogonal skip stays a frozen buffer regardless.
    feedback_initializer, input_initializer, topology : spec, optional
        Same accepted forms as :class:`~resdag.layers.ESNLayer`.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing the recurrent, feedback, input, bias, and
        orthogonal-skip draws (the skip is drawn on an independent, deterministic
        sub-stream).

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
    >>> reservoir = ResESNLayer(200, feedback_size=3, alpha=0.6, beta=0.4)
    >>> feedback = torch.randn(4, 50, 3)  # (batch, time, features)
    >>> states = reservoir(feedback)
    >>> states.shape
    torch.Size([4, 50, 200])

    See Also
    --------
    resdag.layers.cells.resesn_cell.ResESNCell : Underlying single-step cell.
    resdag.layers.reservoirs.es2n.ES2NLayer : Convex-combination (ES2N) variant.
    resdag.layers.ESNLayer : Standard leaky ESN layer.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        alpha: float = 0.5,
        beta: float = 0.9,
        input_size: int | None = None,
        spectral_radius: float | None = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        activation: str = "tanh",
        orthogonal_scaling: float = 1.0,
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
        cell = ResESNCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
            alpha=alpha,
            beta=beta,
            input_size=input_size,
            spectral_radius=spectral_radius,
            bias=bias,
            bias_scaling=bias_scaling,
            activation=activation,
            orthogonal_scaling=orthogonal_scaling,
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
