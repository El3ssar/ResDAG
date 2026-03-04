"""
NG-RC Reservoir Layer
=====================

This module provides the Next Generation Reservoir Computer (NG-RC) sequence
layer:

- :class:`NGReservoir` — stateful sequence layer wrapping :class:`NGCell`.

See Also
--------
resdag.layers.cells.ngrc_cell : Single-step NG-RC cell (NGCell).
resdag.layers.reservoirs.base_reservoir : Abstract base (BaseReservoirLayer).
"""

import torch

from resdag.layers.cells.ngrc_cell import NGCell

from .base_reservoir import BaseReservoirLayer


class NGReservoir(BaseReservoirLayer):
    """
    Stateful sequence layer for Next Generation Reservoir Computing.

    Wraps :class:`NGCell` and processes a full input sequence by scanning
    causally over the time axis, maintaining an internal FIFO delay buffer as
    state.

    This is a feedforward feature map — there are no recurrent weight matrices
    and no learnable parameters.  Gradients flow through it to upstream
    modules if needed.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input vector.
    k : int, default=2
        Number of delay taps (including current input).
    s : int, default=1
        Spacing between delay taps, in timesteps.
    p : int, default=2
        Polynomial degree for nonlinear feature construction.
    include_constant : bool, default=True
        Whether to prepend a constant ``1.0`` to the feature vector.
    include_linear : bool, default=True
        Whether to include the linear delay-embedded features in the output.

    Attributes
    ----------
    cell : NGCell
        The inner single-step cell.
    state : torch.Tensor or None
        Current delay buffer of shape ``(batch, state_size, input_dim)``,
        or ``None`` if not yet initialized.

    Notes
    -----
    The delay buffer needs ``warmup_length = (k-1)*s`` steps to fill.  Steps
    before that produce valid features, but those features contain zeros from
    unfilled buffer slots.  The layer returns all steps; the caller can discard
    the first ``warmup_length`` outputs if desired.

    State management is inherited from :class:`BaseReservoirLayer` with the
    exception of :meth:`set_state`, which is overridden to validate the 3-D
    buffer shape ``(batch, state_size, input_dim)``.

    Examples
    --------
    Basic usage:

    >>> layer = NGReservoir(input_dim=3)
    >>> x = torch.randn(4, 100, 3)   # (batch=4, seq_len=100, d=3)
    >>> features = layer(x)
    >>> features.shape
    torch.Size([4, 100, 28])

    Double-scroll configuration (odd-symmetry, Eq. 10):

    >>> layer = NGReservoir(
    ...     input_dim=2, k=2, s=1, p=3,
    ...     include_constant=False, include_linear=True
    ... )
    >>> x = torch.randn(1, 50, 2)
    >>> features = layer(x)
    >>> features.shape
    torch.Size([1, 50, 24])

    See Also
    --------
    resdag.layers.cells.ngrc_cell.NGCell : The wrapped single-step cell.
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : Base class.
    resdag.layers.readouts.cg_readout.CGReadoutLayer : Algebraic readout.
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 2,
        s: int = 1,
        p: int = 2,
        include_constant: bool = True,
        include_linear: bool = True,
    ) -> None:
        cell = NGCell(
            input_dim=input_dim,
            k=k,
            s=s,
            p=p,
            include_constant=include_constant,
            include_linear=include_linear,
        )
        super().__init__(cell)

    # ------------------------------------------------------------------
    # Properties (delegated to inner cell for convenience)
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        """Dimensionality of each input vector."""
        return self.cell.input_dim

    @property
    def feature_dim(self) -> int:
        """Total dimension of the output feature vector."""
        return self.cell.feature_dim

    @property
    def warmup_length(self) -> int:
        """
        Number of timesteps required to fill the delay buffer.

        Steps before this index contain zeros from unfilled buffer slots.
        Outputs are still produced for all timesteps; the caller may discard
        the first ``warmup_length`` outputs if needed.
        """
        return self.cell.state_size  # == (k-1)*s

    # ------------------------------------------------------------------
    # set_state override — 3-D buffer shape validation
    # ------------------------------------------------------------------

    def set_state(self, state: torch.Tensor) -> None:
        """
        Set the internal delay buffer to an externally supplied tensor.

        Parameters
        ----------
        state : torch.Tensor
            Buffer tensor of shape ``(batch, state_size, input_dim)``.

        Raises
        ------
        ValueError
            If the shape of ``state`` does not match
            ``(*, state_size, input_dim)``.
        """
        expected = (self.cell.state_size, self.cell.input_dim)
        if state.shape[1:] != torch.Size(expected):
            raise ValueError(
                f"State shape mismatch. Expected (batch, {self.cell.state_size}, "
                f"{self.cell.input_dim}), got {tuple(state.shape)}"
            )
        self.state = state.clone()

    # ------------------------------------------------------------------
    # Convenience delegation
    # ------------------------------------------------------------------

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
        return repr(self.cell)
