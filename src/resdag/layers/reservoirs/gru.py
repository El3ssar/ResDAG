"""
GRU Reservoir Layer
===================

This module provides the GRU-based reservoir layer:

- :class:`GRULayer` — stateful sequence layer wrapping :class:`GRUCell`.

See Also
--------
resdag.layers.reservoirs.base_reservoir : Abstract base class (BaseReservoirLayer).
resdag.layers.cells.gru_cell : GRUCell — the single-step GRU cell.
"""

import torch

from resdag.layers.cells.gru_cell import GRUCell

from .base_reservoir import BaseReservoirLayer


class GRULayer(BaseReservoirLayer):
    """
    Stateful GRU reservoir layer with frozen random weights.

    Wraps :class:`GRUCell` in the :class:`BaseReservoirLayer` sequence loop.
    All six weight matrices and three bias vectors are non-trainable (buffers).
    The update gate and reset gate provide per-dimension, input-dependent
    memory dynamics that differ qualitatively from a scalar leaky ESN.

    The layer conforms to the :class:`BaseReservoirLayer` interface:
    ``forward(feedback, *driving_inputs)`` returns reservoir states of shape
    ``(batch, timesteps, hidden_dim)``.  When driving inputs are supplied,
    they are concatenated with the feedback along the feature axis before
    being passed to the cell, so ``input_dim`` must equal
    ``feedback_dim + driving_dim`` (or just ``feedback_dim`` for
    feedback-only usage).

    Parameters
    ----------
    input_dim : int
        Total input dimensionality seen by the GRU cell.  For feedback-only
        use this equals the feedback feature size.  When a driving input is
        also passed, this must equal ``feedback_size + driving_size``.
    hidden_dim : int
        Number of GRU units (hidden state dimension).
    w_in_scale : float, default=1.0
        Scale for the input weight matrices ``W_z``, ``W_r``, ``W_h``.
    spectral_radius : float, default=0.9
        Target spectral radius applied independently to ``U_z``, ``U_r``,
        and ``U_h``.
    sparsity : float, default=0.0
        Fraction of each recurrent matrix's entries set to zero before
        spectral radius scaling.
    bias : bool, default=True
        Whether to include bias vectors.
    gate_bias_init : float, default=0.0
        Constant shift added to each bias entry after uniform initialisation.

    Attributes
    ----------
    state : torch.Tensor or None
        Current reservoir state of shape ``(batch, hidden_dim)``.
        ``None`` if not yet initialized.

    Examples
    --------
    Feedback-only GRU reservoir:

    >>> import torch
    >>> from resdag.layers.reservoirs.gru import GRULayer
    >>> layer = GRULayer(input_dim=10, hidden_dim=200)
    >>> feedback = torch.randn(4, 50, 10)  # (batch, time, features)
    >>> states = layer(feedback)
    >>> states.shape
    torch.Size([4, 50, 200])

    With a driving input (feedback_dim=10, driving_dim=5 → input_dim=15):

    >>> layer2 = GRULayer(input_dim=15, hidden_dim=200)
    >>> feedback = torch.randn(4, 50, 10)
    >>> driving = torch.randn(4, 50, 5)
    >>> states = layer2(feedback, driving)
    >>> states.shape
    torch.Size([4, 50, 200])

    State management works identically to ESNLayer:

    >>> layer.reset_state()
    >>> layer.reset_state(batch_size=4)
    >>> layer.get_state()  # (4, 200)
    >>> layer.set_random_state()

    See Also
    --------
    resdag.layers.cells.gru_cell.GRUCell : Underlying single-step cell.
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : Base class.
    resdag.layers.reservoirs.esn.ESNLayer : Analogous leaky-ESN layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        w_in_scale: float = 1.0,
        spectral_radius: float = 0.9,
        sparsity: float = 0.0,
        bias: bool = True,
        gate_bias_init: float = 0.0,
    ) -> None:
        cell = GRUCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            w_in_scale=w_in_scale,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            bias=bias,
            gate_bias_init=gate_bias_init,
        )
        super().__init__(cell)

    # ------------------------------------------------------------------
    # State management override
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset internal state to zero.

        Overrides :meth:`BaseReservoirLayer.reset_state` to look up device
        and dtype from the cell's buffers rather than parameters (GRUCell
        has no trainable parameters).

        Parameters
        ----------
        batch_size : int, optional
            If provided, initialise state to zeros with this batch size.
            If ``None``, state is set to ``None`` and lazily re-initialised
            on the next forward pass.

        Examples
        --------
        >>> layer.reset_state()            # Lazy re-init
        >>> layer.reset_state(batch_size=4)  # Explicit zero state
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

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the wrapped cell."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Reach into _modules directly to avoid recursion during __init__
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
