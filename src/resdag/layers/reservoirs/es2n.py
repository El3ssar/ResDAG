"""
ES2N Reservoir Layer
====================

This module provides the public-facing Edge-of-Stability State Network (ES2N)
sequence layer:

- :class:`ES2NLayer` — stateful RNN layer wrapping an :class:`ES2NCell`.

References
----------
A. Ceni and C. Gallicchio, "Edge of Stability Echo State Networks,"
arXiv:2308.02902 (2023/2025).

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base (BaseReservoirLayer).
resdag.layers.cells.es2n_cell : Underlying single-step cell.
resdag.layers.reservoirs.resesn : Residual ESN — the decoupled-weight sibling.
"""

from typing import Any

from resdag.init.utils import InitializerSpec, TopologySpec
from resdag.layers.cells.es2n_cell import ES2NCell
from resdag.utils.general import SeedLike

from .base_reservoir import BaseReservoirLayer


class ES2NLayer(BaseReservoirLayer):
    r"""
    Stateful Edge-of-Stability State Network (ES2N) reservoir layer.

    Public-facing class.  Internally creates an
    :class:`~resdag.layers.cells.es2n_cell.ES2NCell` that owns all parameters
    and performs the single-step update; the sequence loop and state management
    live in the parent :class:`~resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer`.

    ES2N (Ceni & Gallicchio, arXiv:2308.02902) writes the state as a convex
    combination of a nonlinear ESN branch and a fixed random *orthogonal* skip:

    .. math::

        x_t = \beta\,\phi(W_r x_{t-1} + W_{fb} x_{fb,t} + W_{in} u_t + b)
              + (1 - \beta)\,O\,x_{t-1}

    with :math:`\beta \in (0, 1]` the *proximity* to the edge of stability.
    Small :math:`\beta` → nearly the orthogonal norm-preserving map (long
    memory, edge of chaos); :math:`\beta = 1` → a standard non-leaky ESN.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units (hidden-state dimension).
    feedback_size : int
        Dimension of the feedback signal.  Required.
    beta : float, default=0.9
        Proximity to the edge of stability in :math:`(0, 1]` (see
        :class:`ES2NCell`).  A *live* knob: ``layer.beta = ...`` takes effect on
        the next forward.
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
        norm-preserving (the edge-of-stability regime).
    trainable : bool, default=False
        If ``True``, reservoir weights are trainable via backpropagation.  The
        orthogonal skip stays a frozen buffer regardless.
    feedback_initializer, input_initializer, topology : spec, optional
        Same accepted forms as :class:`~resdag.layers.ESNLayer`.
    seed : int or torch.Generator, optional
        Reproducibility seed fixing the recurrent, feedback, input, bias, and
        orthogonal-skip draws (the skip is drawn on an independent, deterministic
        sub-stream).

    Attributes
    ----------
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, reservoir_size)``, or
        ``None`` before the first forward.

    Examples
    --------
    >>> import torch
    >>> reservoir = ES2NLayer(reservoir_size=200, feedback_size=3, beta=0.1)
    >>> feedback = torch.randn(4, 50, 3)  # (batch, time, features)
    >>> states = reservoir(feedback)
    >>> states.shape
    torch.Size([4, 50, 200])

    See Also
    --------
    resdag.layers.cells.es2n_cell.ES2NCell : Underlying single-step cell.
    resdag.layers.reservoirs.resesn.ResESNLayer : Decoupled-weight variant.
    resdag.layers.ESNLayer : Standard leaky ESN layer.
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
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
    ) -> None:
        cell = ES2NCell(
            reservoir_size=reservoir_size,
            feedback_size=feedback_size,
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
