"""
LMU Reservoir Layer
===================

This module provides the Legendre Memory Unit (LMU) implementation of the
reservoir layer:

- :class:`LMULayer` — public-facing sequence layer wrapping :class:`LMUCell`.

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base class (BaseReservoirLayer).
resdag.layers.cells.lmu_cell : LMUCell (single-step LMU update).
"""

import torch

from resdag.layers.cells.lmu_cell import LMUCell

from .base_reservoir import BaseReservoirLayer


class LMULayer(BaseReservoirLayer):
    """
    Stateful RNN reservoir layer based on the Legendre Memory Unit.

    Wraps an :class:`LMUCell` in the :class:`BaseReservoirLayer` sequence
    loop and state-management infrastructure.  Provides the same interface
    as :class:`ESNLayer` so it composes transparently with the DAG
    infrastructure.

    The LMU maintains a linear memory that optimally represents the
    continuous-time history of its input over a sliding window of length
    ``theta``.  For each input dimension, an independent memory vector of
    size ``order`` is maintained, yielding a total state size of
    ``feedback_size * order`` (plus ``hidden_dim`` if ``nonlinear_hidden``
    is enabled).

    Parameters
    ----------
    feedback_size : int
        Dimension of the feedback signal (required).
    input_size : int, optional
        Dimension of optional driving input.  When provided, the driving
        signal is concatenated with the feedback before entering the LMU
        memory.  Total ``input_dim = feedback_size + input_size``.
    order : int, default=8
        Order of the Legendre approximation (memory state dimension per
        input channel).
    theta : float, default=1.0
        Window length (time constant) of the sliding memory window.
    dt : float, default=1.0
        Simulation time step used for discretization.
    discretization : {'zoh', 'euler'}, default='zoh'
        Discretization method. 'zoh' (Zero-Order Hold) is more faithful
        to the continuous-time dynamics; 'euler' is simpler.
    nonlinear_hidden : bool, default=False
        If ``False`` (default), outputs the flat memory state only — a
        purely linear reservoir.  If ``True``, appends a random nonlinear
        hidden state, combining LMU memory with ESN-like nonlinear
        dynamics.
    hidden_dim : int, default=64
        Dimension of the nonlinear hidden state.  Only used when
        ``nonlinear_hidden=True``.
    w_x_scale : float, default=1.0
        Scaling factor for the random input-to-hidden weight matrix.
        Only used when ``nonlinear_hidden=True``.

    Attributes
    ----------
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, cell.state_size)``.
        ``None`` if not yet initialized.
    feedback_size : int
        Dimension of the feedback signal.
    input_size : int or None
        Dimension of the driving input, or ``None`` if not provided.

    Examples
    --------
    Feedback-only LMU reservoir:

    >>> import torch
    >>> from resdag.layers.reservoirs import LMULayer
    >>> lmu = LMULayer(feedback_size=3, order=16, theta=10.0)
    >>> x = torch.randn(4, 50, 3)  # (batch, time, feedback)
    >>> out = lmu(x)
    >>> print(out.shape)
    torch.Size([4, 50, 48])

    LMU with driving input and nonlinear hidden state:

    >>> lmu = LMULayer(
    ...     feedback_size=3, input_size=2, order=8,
    ...     nonlinear_hidden=True, hidden_dim=32
    ... )
    >>> fb = torch.randn(4, 50, 3)
    >>> drv = torch.randn(4, 50, 2)
    >>> out = lmu(fb, drv)
    >>> print(out.shape)
    torch.Size([4, 50, 72])

    State management (same API as ESNLayer):

    >>> lmu.reset_state()
    >>> lmu.reset_state(batch_size=4)
    >>> saved = lmu.get_state()
    >>> lmu.set_state(saved)

    See Also
    --------
    resdag.layers.cells.lmu_cell.LMUCell : Underlying single-step cell.
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : Sequence loop base.
    resdag.layers.reservoirs.esn.ESNLayer : Analogous ESN layer.
    """

    def __init__(
        self,
        feedback_size: int,
        input_size: int | None = None,
        order: int = 8,
        theta: float = 1.0,
        dt: float = 1.0,
        discretization: str = "zoh",
        nonlinear_hidden: bool = False,
        hidden_dim: int = 64,
        w_x_scale: float = 1.0,
    ) -> None:
        input_dim = feedback_size + (input_size if input_size is not None else 0)
        cell = LMUCell(
            input_dim=input_dim,
            order=order,
            theta=theta,
            dt=dt,
            discretization=discretization,
            nonlinear_hidden=nonlinear_hidden,
            hidden_dim=hidden_dim,
            w_x_scale=w_x_scale,
        )
        super().__init__(cell)
        self.feedback_size = feedback_size
        self.input_size = input_size

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset internal state to zero.

        Overrides the base implementation to source device/dtype from the
        cell's buffers (LMUCell has no learnable parameters).

        Parameters
        ----------
        batch_size : int, optional
            If provided, initialize state with this batch size using the
            cell's device and dtype.  If ``None``, state is set to ``None``
            and will be lazily initialized on the next forward pass.
        """
        if batch_size is not None:
            buf = next(self.cell.buffers())
            self.state = torch.zeros(
                batch_size,
                self.cell.state_size,
                device=buf.device,
                dtype=buf.dtype,
            )
        else:
            self.state = None

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the wrapped cell."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
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
