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
import torch.nn as nn

from resdag.layers.cells.ngrc_cell import NGCell


class NGReservoir(nn.Module):
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

    State management mirrors :class:`BaseReservoirLayer`:
    :meth:`reset_state`, :meth:`get_state`, :meth:`set_state`,
    :meth:`set_random_state`.

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
    resdag.layers.reservoirs.base_reservoir.BaseReservoirLayer : ESN analogue.
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
        super().__init__()
        self.cell = NGCell(
            input_dim=input_dim,
            k=k,
            s=s,
            p=p,
            include_constant=include_constant,
            include_linear=include_linear,
        )
        self.state: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties (delegated to inner cell)
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
    def state_size(self) -> int:
        """Number of rows in the delay buffer: ``(k-1)*s``."""
        return self.cell.state_size

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
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        feedback: torch.Tensor,
        *driving_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process an input sequence through the NG-RC feature map.

        Parameters
        ----------
        feedback : torch.Tensor
            Input sequence of shape ``(batch, seq_len, input_dim)``.
            Driving inputs are not used by NG-RC; this parameter is named
            ``feedback`` for API consistency with :class:`ESNLayer`.
        *driving_inputs : torch.Tensor
            Ignored.  Present for interface consistency only.

        Returns
        -------
        torch.Tensor
            Feature matrix of shape ``(batch, seq_len, feature_dim)``.

        Raises
        ------
        ValueError
            If ``feedback`` is not a 3-D tensor.

        Notes
        -----
        The layer maintains internal state across forward calls.  Call
        :meth:`reset_state` between independent sequences.
        """
        if feedback.dim() != 3:
            raise ValueError(f"Input must be 3-D (batch, seq_len, input_dim), got {feedback.shape}")

        batch_size, seq_len, _ = feedback.shape

        self._maybe_init_state(batch_size, feedback.device, feedback.dtype)

        outputs = torch.empty(
            batch_size,
            seq_len,
            self.cell.feature_dim,
            device=feedback.device,
            dtype=feedback.dtype,
        )

        for t in range(seq_len):
            features_t, self.state = self.cell(feedback[:, t, :], self.state)
            outputs[:, t, :] = features_t

        return outputs

    # ------------------------------------------------------------------
    # State management (mirrors BaseReservoirLayer API)
    # ------------------------------------------------------------------

    def _maybe_init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize state to zeros if not present or if context changed."""
        if (
            self.state is None
            or self.state.shape[0] != batch_size
            or self.state.device != device
            or self.state.dtype != dtype
        ):
            self.state = self.cell.init_state(batch_size, device, dtype)

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset the internal delay buffer.

        Parameters
        ----------
        batch_size : int, optional
            If provided, initialize the buffer to zeros with this batch size,
            reusing the device and dtype of the current state (defaults to
            CPU float32 if state is not yet initialized).  If ``None``, set
            state to ``None`` for lazy re-initialization on the next forward
            pass.

        Examples
        --------
        >>> layer.reset_state()            # Lazy reset
        >>> layer.reset_state(batch_size=4)  # Explicit zero buffer
        """
        if batch_size is not None:
            if self.state is not None:
                device = self.state.device
                dtype = self.state.dtype
            else:
                device = torch.device("cpu")
                dtype = torch.float32
            self.state = self.cell.init_state(batch_size, device, dtype)
        else:
            self.state = None

    def get_state(self) -> torch.Tensor | None:
        """
        Return a copy of the current delay buffer.

        Returns
        -------
        torch.Tensor or None
            Clone of the current buffer of shape
            ``(batch, state_size, input_dim)``, or ``None`` if not yet
            initialized.
        """
        return self.state.clone() if self.state is not None else None

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

    def set_random_state(self) -> None:
        """
        Set the delay buffer to standard-normal random values.

        Raises
        ------
        RuntimeError
            If the buffer has not been initialized yet.
        """
        if self.state is None:
            raise RuntimeError(
                "NGReservoir state is not initialized. "
                "Call reset_state(batch_size=N) or run a forward pass first."
            )
        self.state = torch.randn_like(self.state)

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
